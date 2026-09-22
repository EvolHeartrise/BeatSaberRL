#!/usr/bin/env python3
"""Backfill `metrics.jsonl` + `index.json` over the laptop run archive.

Walks `~/beat-arena-volume-backup` (every campaign dir), parses every
trainer stdout log with `metrics_log.parse_log`, writes one
`metrics.jsonl` per run — next to the run dir when it exists, else
under `<campaign>/metrics/<run>/` — and one `index.json` registry at
the archive root that the dashboard (`viz/dash`) reads.

A "run" is a run dir name: every invocation whose `--run-dir` names it
(trunk, its resumed extensions, a re-run) folds into one update list
(`metrics_log.merge_segments`: a resume discards the stale tail).
Logs that carry several invocations (the pod chain logs) split per
`argv:` line. Orchestration logs with no `update` lines are ignored.

Idempotent and cheap to re-run: a cache of (size, mtime) per log and
per bench JSON skips unchanged inputs, and `metrics.jsonl` is only
rewritten when its bytes change. A trainer-written `metrics.jsonl`
(no `.metrics.backfill` marker beside it) is never overwritten.

    python3 metrics_backfill.py                 # one pass
    python3 metrics_backfill.py --watch 30      # re-scan every 30 s
    python3 metrics_backfill.py --force         # ignore the cache

stdlib only; runs under the system python3.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics_log as ml  # noqa: E402

DEFAULT_ARCHIVE = os.path.expanduser("~/beat-arena-volume-backup")
REPO_PYTHON = os.path.dirname(os.path.abspath(__file__))
CACHE_NAME = ".metrics_backfill_cache.json"
MARKER = ".metrics.backfill"
SKIP_DIRS = {"node_modules", ".git", "__pycache__", "pool_cache", "maps"}

_RE_CKPT = re.compile(r"\.pt$")


def log(msg, *, verbose=True):
    if verbose:
        print(msg, flush=True)


def _stat(path):
    st = os.stat(path)
    return [st.st_size, round(st.st_mtime, 3)]


def find_logs(archive):
    out = []
    for root, dirs, files in os.walk(archive):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                   and not d.startswith(".")]
        for f in files:
            if f.endswith(".log"):
                out.append(os.path.join(root, f))
    return sorted(out)


def run_name_from_argv(run_dir):
    """`runs/x` -> `x`; `/abs/.../runs/leafcuts/y` -> `leafcuts/y`."""
    if not run_dir:
        return None
    rd = run_dir.replace("\\", "/").rstrip("/")
    if "/runs/" in rd:
        return rd.split("/runs/", 1)[1]
    if rd.startswith("runs/"):
        return rd[5:]
    return os.path.basename(rd)


def run_name_from_log(rel_log):
    """A log inside `<campaign>/runs/<name>/...` names that run; else
    the log's stem (`logs/foo_trunk.log` -> `foo_trunk`)."""
    parts = rel_log.split("/")
    if len(parts) >= 3 and parts[1] == "runs":
        return "/".join(parts[2:-1])
    return os.path.splitext(parts[-1])[0]


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def matching_replays(run, replays):
    """Replay files are named `<tag>_bot...`/`<tag>_top...` with ad hoc
    tags (`v13prod` for run v13_prod_2h, `v26gamma12k` for
    v26gamma_leaf12k). Match loosely: the normalized tag and the
    normalized run name share a prefix of >= 4 chars and one is a
    prefix of the other. Candidates, not proof."""
    rn = _norm(run.split("/")[-1])
    out = []
    for p in replays:
        stem = os.path.basename(p)
        tag = re.split(r"_(?:bot|top)", stem, 1)[0]
        tn = _norm(tag)
        if len(tn) >= 4 and (rn.startswith(tn) or tn.startswith(rn)):
            out.append(p)
    return out


class Backfill:
    def __init__(self, archive, repo_python=REPO_PYTHON, force=False,
                 verbose=True):
        self.archive = os.path.abspath(archive)
        self.repo_python = repo_python
        self.force = force
        self.verbose = verbose
        self.cache_path = os.path.join(self.archive, CACHE_NAME)
        self.cache = {"logs": {}, "bench": {}}
        if not force and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, encoding="utf-8") as f:
                    self.cache = json.load(f)
            except (OSError, ValueError):
                pass
        self.cache.setdefault("logs", {})
        self.cache.setdefault("bench", {})

    # ------------------------------------------------------------ paths
    def rel(self, path):
        return os.path.relpath(path, self.archive).replace(os.sep, "/")

    def campaigns(self):
        return sorted(d for d in os.listdir(self.archive)
                      if os.path.isdir(os.path.join(self.archive, d))
                      and not d.startswith("."))

    def locate_run_dir(self, campaign, name):
        """Existing `<campaign>/runs/<name>` (own campaign first, then
        any campaign holding exactly one), else None."""
        own = os.path.join(self.archive, campaign, "runs", name)
        if os.path.isdir(own):
            return own
        hits = [d for d in glob.glob(os.path.join(
            self.archive, "*", "runs", name)) if os.path.isdir(d)]
        if len(hits) == 1:
            return hits[0]
        return None

    # ------------------------------------------------------------- logs
    def _parse_one(self, rel, st, runs):
        path = os.path.join(self.archive, rel)
        try:
            segs = ml.parse_log_file(path)
        except OSError as e:
            log(f"  ! {rel}: {e}", verbose=self.verbose)
            self.cache["logs"][rel] = {"stat": st, "runs": []}
            return
        # a trainer invocation has an argv line or the PPO banner;
        # orchestration logs that echo a stray `update` line do not
        segs = [s for s in segs if s.updates
                and (s.run.get("argv") or "num_envs" in s.run)]
        campaign = rel.split("/")[0]
        run_ids = []
        for s in segs:
            name = run_name_from_argv(s.run.get("run_dir")) \
                or run_name_from_log(rel)
            run_dir = self.locate_run_dir(campaign, name)
            if run_dir is None and "/runs/" in rel:
                # the log sits inside its run dir (full backup)
                cand = os.path.join(self.archive, os.path.dirname(rel))
                if os.path.isdir(cand):
                    run_dir = cand
                    name = run_name_from_log(rel)
            run_campaign = (self.rel(run_dir).split("/")[0]
                            if run_dir else campaign)
            run_id = f"{run_campaign}/{name}"
            run_ids.append(run_id)
            r = runs.setdefault(run_id, {
                "id": run_id, "campaign": run_campaign, "run": name,
                "run_dir": run_dir, "segments": [], "logs": []})
            key = (s.updates[0]["upd"], st[1], s.index)
            r["segments"].append((key, s, rel))
            if rel not in r["logs"]:
                r["logs"].append(rel)
        self.cache["logs"][rel] = {"stat": st, "runs": run_ids}

    def parse_logs(self, prev_runs):
        """{run_id: {"segments": [(sort_key, Segment, log_rel)], ...}}

        Logs whose (size, mtime) match the cache are not re-read when
        every log of every run they feed is likewise unchanged AND the
        previous index already holds those runs — those runs are
        carried over verbatim (`reused`). Any changed log re-parses
        all logs of the runs it feeds."""
        stats = {}
        changed = set()
        for path in find_logs(self.archive):
            rel = self.rel(path)
            st = _stat(path)
            stats[rel] = st
            cached = self.cache["logs"].get(rel)
            if self.force or not cached or cached.get("stat") != st:
                changed.add(rel)
        # runs touched by a changed log -> every log feeding them re-parses
        run_logs = {}
        for rel, c in self.cache["logs"].items():
            for rid in c.get("runs", []):
                run_logs.setdefault(rid, set()).add(rel)
        dirty_runs = set()
        for rel in changed:
            for rid in self.cache["logs"].get(rel, {}).get("runs", []):
                dirty_runs.add(rid)
        to_parse = set(changed)
        for rid in dirty_runs:
            to_parse |= run_logs.get(rid, set())
        # a cached run whose previous record is gone must be rebuilt
        for rid, rels in run_logs.items():
            if rid not in prev_runs:
                to_parse |= rels
        to_parse = {r for r in to_parse if r in stats}
        reused = {}
        for rid, rels in run_logs.items():
            if rid in prev_runs and not (rels & to_parse):
                reused[rid] = prev_runs[rid]
        runs = {}
        parsed = set()
        queue = sorted(to_parse)
        while queue:
            rel = queue.pop(0)
            if rel in parsed:
                continue
            parsed.add(rel)
            self._parse_one(rel, stats[rel], runs)
            # a NEW log feeding an already-cached run (an extension
            # rsynced after its trunk) drags that run's other logs in
            for rid in self.cache["logs"][rel]["runs"]:
                for other in run_logs.get(rid, ()):
                    if other in stats and other not in parsed:
                        queue.append(other)
        for rid in list(reused):
            if rid in runs:
                del reused[rid]
        # drop cache entries for logs that vanished
        for rel in list(self.cache["logs"]):
            if rel not in stats:
                del self.cache["logs"][rel]
        for rid in list(reused):
            if rid in runs:
                del reused[rid]
        return runs, reused

    # ------------------------------------------------------------ bench
    def bench_index(self):
        """{run name: [bench summary, ...]} over every bench JSON in the
        archive campaigns and the repo's python/bench."""
        paths = glob.glob(os.path.join(self.archive, "*", "bench", "*.json"))
        paths += glob.glob(os.path.join(self.repo_python, "bench", "*.json"))
        by_run = {}
        seen = set()
        for p in sorted(paths):
            key = p if not p.startswith(self.archive) else self.rel(p)
            seen.add(key)
            st = _stat(p)
            c = self.cache["bench"].get(key)
            if c is None or c.get("stat") != st or self.force:
                try:
                    c = {"stat": st, "summary": self._bench_summary(p, key)}
                except Exception as e:   # a non-bench JSON must not kill the scan
                    log(f"  ! bench {key}: {e}", verbose=self.verbose)
                    c = {"stat": st, "summary": None}
                self.cache["bench"][key] = c
            summ = c["summary"]
            if summ and summ.get("run"):
                by_run.setdefault(summ["run"], []).append(summ)
        for k in list(self.cache["bench"]):
            if k not in seen:
                del self.cache["bench"][k]
        return by_run

    def _bench_summary(self, path, key):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(d, dict) or "levels" not in d:
            return None
        ckpt_name = d.get("checkpoint_name") or ""
        ckpt = (d.get("checkpoint") or "").replace("\\", "/")
        run = None
        if "/" in ckpt_name:
            run = ckpt_name.rsplit("/", 1)[0]
        elif "/runs/" in ckpt:
            run = ckpt.split("/runs/", 1)[1].rsplit("/", 1)[0]
        stem = os.path.splitext(os.path.basename(path))[0]
        if run is None:
            run = stem
        url = key if not os.path.isabs(key) else None
        return {"path": key, "url": url, "name": stem, "run": run,
                "ckpt": os.path.basename(ckpt_name or ckpt) or None,
                "wpp": d.get("total_weighted_pp"),
                "upp": d.get("total_unweighted_pp"),
                # wrist-bend reversals / s per hand (2026-09-22)
                "wrev_l": d.get("wrist_rev_per_s_l"),
                "wrev_r": d.get("wrist_rev_per_s_r"),
                "wrev_total": d.get("wrist_rev_total"),
                "wrev_total_l": d.get("wrist_rev_total_l"),
                "wrev_total_r": d.get("wrist_rev_total_r"),
                "n_evaluated": d.get("n_evaluated"),
                "n_levels": (len(d["levels"])
                             if isinstance(d.get("levels"), list) else 0),
                "scoring_version": d.get("scoring_version"),
                "global_step": d.get("global_step"),
                "updated": d.get("updated")}

    def replay_index(self):
        """{stem prefix (before first '_'): [rel/abs paths]}"""
        paths = glob.glob(os.path.join(self.archive, "*", "replays",
                                       "*.jsonl"))
        paths += glob.glob(os.path.join(self.repo_python, "replays",
                                        "*.jsonl"))
        out = []
        for p in sorted(paths):
            key = self.rel(p) if p.startswith(self.archive) else p
            out.append(key)
        return out

    # ------------------------------------------------------------- runs
    def build_run(self, r, bench_by_run, replays):
        segs = sorted(r["segments"], key=lambda x: x[0])
        recs, seg_runs = ml.merge_segments([s for _, s, _ in segs])
        run_dir = r["run_dir"]
        rec_out = None
        source = "backfill"
        if run_dir:
            metrics_path = os.path.join(run_dir, "metrics.jsonl")
            marker = os.path.join(run_dir, MARKER)
            if os.path.exists(metrics_path) and not os.path.exists(marker):
                source = "trainer"
        else:
            mirror = os.path.join(self.archive, r["campaign"], "metrics",
                                  r["run"])
            metrics_path = os.path.join(mirror, "metrics.jsonl")
            marker = os.path.join(mirror, MARKER)
        if source == "backfill":
            body = "".join(ml.dumps_record(x) + "\n" for x in recs)
            self._write_if_changed(metrics_path, body)
            if not os.path.exists(marker):
                with open(marker, "w") as f:
                    f.write("written by metrics_backfill.py\n")
            rec_out = recs
        else:
            rec_out = ml.load_metrics(metrics_path) or recs

        last_seg = seg_runs[-1] if seg_runs else {}
        first_seg = seg_runs[0] if seg_runs else {}
        # args: args.json when present, else the banner argv
        args = None
        if run_dir and os.path.exists(os.path.join(run_dir, "args.json")):
            try:
                with open(os.path.join(run_dir, "args.json"),
                          encoding="utf-8") as f:
                    args = json.load(f)
            except (OSError, ValueError):
                args = None
        if args is None:
            args = {"argv": last_seg.get("argv") or first_seg.get("argv"),
                    "resolved": last_seg.get("config")
                    or first_seg.get("config"), "source": "log"}
        else:
            args["source"] = "args.json"
        # wall: sum of each invocation's final elapsed
        wall = sum((sr.get("elapsed") or 0.0) for sr in seg_runs)
        log_mtimes = [_stat(os.path.join(self.archive, l))[1]
                      for l in r["logs"]]
        t_last = max(log_mtimes) if log_mtimes else None
        if rec_out and rec_out[-1].get("t"):
            t_last = rec_out[-1]["t"]
        t_first = (t_last - wall) if t_last is not None else None
        final = rec_out[-1] if rec_out else {}
        # steps/s: median of the last 50 updates' prints
        tail = [x["sps"] for x in rec_out[-50:] if x.get("sps")]
        tail.sort()
        sps = tail[len(tail) // 2] if tail else last_seg.get("sps_total")
        ckpts = []
        if run_dir:
            for f in sorted(os.listdir(run_dir)):
                if _RE_CKPT.search(f):
                    try:
                        sz = os.path.getsize(os.path.join(run_dir, f))
                    except OSError:
                        sz = None
                    ckpts.append({"name": f, "bytes": sz})
        else:
            ckpts = [{"name": n, "bytes": None} for sr in seg_runs
                     for n in sr.get("checkpoints", [])]
        peak = max([sr.get("peak_vram") or 0 for sr in seg_runs] +
                   [x.get("peak_vram") or 0 for x in rec_out]) or None
        fields = set()
        for x in rec_out:
            fields |= set(x)
        rep = matching_replays(r["run"], replays)
        return {
            "id": r["id"], "campaign": r["campaign"], "run": r["run"],
            "run_dir": self.rel(run_dir) if run_dir else None,
            "logs": r["logs"],
            "metrics": self.rel(metrics_path),
            "metrics_source": source,
            "args": args,
            "invocations": len(seg_runs),
            "role": last_seg.get("role") or first_seg.get("role"),
            "leaf_from": last_seg.get("leaf_from"),
            "resume_from": last_seg.get("resume_from"),
            "num_envs": first_seg.get("num_envs"),
            "horizon": first_seg.get("horizon"),
            "batch": first_seg.get("batch"),
            "device": first_seg.get("device"),
            "n_params": first_seg.get("n_params"),
            "seed": first_seg.get("seed"),
            "status": last_seg.get("status"),
            "samples": final.get("steps"),
            "updates": len(rec_out),
            "last_update": final.get("upd"),
            "num_updates": final.get("n_upd"),
            "t_first": t_first, "t_last": t_last, "wall_s": wall,
            "final": {k: final.get(k) for k in
                      ("acc", "accL", "accR", "return", "ep_return",
                       "ent", "ev", "entfull")},
            "sps": sps,
            "sps_total": last_seg.get("sps_total"),
            "peak_vram": peak,
            "bench": bench_by_run.get(r["run"], []),
            "checkpoints": ckpts,
            "replays": rep,
            "git_commit": None,   # not recoverable from any archive log
            "fields": sorted(fields),
            "line_counts": {k: sum(sr.get("line_counts", {}).get(k, 0)
                                   for sr in seg_runs)
                            for k in ("update", "kl", "fall", "tilt",
                                      "lift", "flail", "limitrest",
                                      "diag", "ckpt", "throughput")},
        }

    def _write_if_changed(self, path, body):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    if f.read() == body:
                        return False
            except OSError:
                pass
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, path)
        return True

    # ------------------------------------------------------------- main
    def run_once(self):
        t0 = time.time()
        prev_runs = {}
        ipath = os.path.join(self.archive, "index.json")
        if not self.force and os.path.exists(ipath):
            try:
                with open(ipath, encoding="utf-8") as f:
                    prev_runs = {r["id"]: r for r in json.load(f)["runs"]}
            except (OSError, ValueError, KeyError):
                prev_runs = {}
        runs, reused = self.parse_logs(prev_runs)
        bench_by_run = self.bench_index()
        replays = self.replay_index()
        out = []
        for rid in sorted(runs):
            try:
                out.append(self.build_run(runs[rid], bench_by_run, replays))
            except Exception as e:  # noqa: BLE001 — one bad run must not
                log(f"  ! {rid}: {e!r}", verbose=self.verbose)  # stop the index
        for rid, rec in reused.items():
            rec["bench"] = bench_by_run.get(rec["run"], [])
            rec["replays"] = matching_replays(rec["run"], replays)
            out.append(rec)
        out.sort(key=lambda x: -(x["t_last"] or 0))
        index = {
            "generated": time.time(),
            "archive": self.archive,
            "campaigns": self.campaigns(),
            "replay_dirs": sorted({os.path.dirname(p) for p in replays}),
            "runs": out,
        }
        body = json.dumps(index, indent=0, default=str)
        changed = self._write_if_changed(
            os.path.join(self.archive, "index.json"), body)
        with open(self.cache_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(self.cache, f)
        os.replace(self.cache_path + ".tmp", self.cache_path)
        log(f"[backfill] {len(out)} runs ({len(runs)} parsed, "
            f"{len(reused)} cached), {sum(x['updates'] for x in out)}"
            f" updates, index {'rewritten' if changed else 'unchanged'}"
            f" ({time.time() - t0:.1f}s)", verbose=self.verbose)
        return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", default=DEFAULT_ARCHIVE)
    ap.add_argument("--repo-python", default=REPO_PYTHON,
                    help="repo python/ dir (bench + replays live there too)")
    ap.add_argument("--watch", type=float, nargs="?", const=30.0,
                    default=None, metavar="SECS",
                    help="re-scan every SECS seconds (default 30)")
    ap.add_argument("--force", action="store_true",
                    help="ignore the (size, mtime) cache")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    bf = Backfill(a.archive, a.repo_python, force=a.force,
                  verbose=not a.quiet)
    bf.run_once()
    bf.force = False
    while a.watch:
        time.sleep(a.watch)
        try:
            bf.run_once()
        except Exception as e:  # noqa: BLE001 — keep watching
            log(f"[backfill] pass failed: {e!r}", verbose=True)


if __name__ == "__main__":
    main()
