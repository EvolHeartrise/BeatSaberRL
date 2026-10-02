"""Batched replay generation: a whole set of bench plays in ONE batched sim.

    python make_replays_batched.py --ckpt <ckpt.pt> --bench <bench.json> \\
        --tag v63 [--top 10 | --levels KEY...] [--board ss|bl]
    python make_replays_batched.py --ckpt <ckpt.pt> --composite <best.json> \\
        --sources <dir of its input benches> --tag v63 \\
        [--top 20 | --levels KEY... | --all] [--board ss|bl]
      [--exact (default) | --fast] [--write-back] [--dat] [--writers N]

make_top10_replays.py records one play at a time: a RustBeatEnv, a per-
tick Python JSON record, and a width-1024 forward carrying ONE real row
per tick. This tool plays every selected level at once, the way the bench
played them (2026-10-02):

  - one VecBeatSim (combo_walk=True, the bench's max_steps), each env with
    its source run's body (set_tau_scale_env), control rate (tick_hz_env)
    and action repeat (action_repeat_env) — so one vector step is one
    policy decision for every env, and a composite's picks from runs at
    different rates share the wave;
  - one bench forward (_GraphedArgmax; eager on CPU) per step at a FIXED
    width W over all live envs. --exact (the default) uses the source
    bench's forward_width (absent = 1024): its rows are independent at a
    fixed width (GEMM kernel choice depends on the batch shape, not on
    the other rows), so each play should equal its bench row exactly.
    --fast uses W = 32 (a different GEMM shape: near-tie argmaxes may
    flip, so a play can drift from its bench row — reported, and
    --write-back can adopt it);
  - each env records in Rust (VecBeatSim.set_record_env /
    take_record_env: one frame per SIM tick, action-repeat ticks
    included), and a finished env's capture goes to a spawn process pool
    (started before CUDA init) that writes the publish-ready compact
    .jsonl.gz with its full header (map, level, weights hash) through
    rust_env.record_header / write_replay_jsonl — and the ScoreSaber .dat
    plus its decode check only with --dat;
  - the bench-style result row is built by the bench's own finish_slot
    from the same VecBeatSim getters the bench reads, and compared to the
    source bench row (play counters: score, combo, good / bad / miss /
    bombs). One failing job (map fetch, parse, write) is reported and the
    run goes on.

Jobs: a plain bench's levels (--levels / --top N by --sort, in the --board
view, as make_top10_replays.py), or a composite's PICKS (--composite +
--sources): each selected level's SS and BL picks (only the --board one
when given) -> (source bench file ss_from / bl_from, level), deduplicated
— a level whose two picks come from one run plays once. Charts come from
bench_scoresaber.load_or_parse (the chart cache; BeatSaver only on a
miss).

--write-back: a play that does not reproduce its bench row (e.g. a
--fast run, or a cross-device drift) REPLACES that row in its source
bench — re-saved through bench_scoresaber.save_results (totals
recomputed), the row stamped `replay_override` and the file's
`replay_overrides` log extended — and, for a composite, the pick it
serves is patched in place (SS pick: the top-level play fields + `ss`;
BL pick: `bl_play` + `bl`) and the composite's totals recomputed with
tools/bench_combine.recompute_totals. The pick selection is never re-run.
publish_replays.py then verifies the replays against the written-back
files as usual.

--max-ticks N (debug): stop every play after N sim ticks (truncated plays
are written but never compared or written back). Works on CPU too (eager
forward), e.g. a 2-level, 300-tick smoke run on a laptop.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PLAY_KEYS = (("score", "score"), ("combo", "highest_combo"),
             ("good", "good_cuts"), ("bad", "bad_cuts"),
             ("miss", "misses"), ("bombs", "bombs_hit"))
FAST_WIDTH = 32
# --fast only pays below this many plays: a width-32 forward handles 32
# envs per step, so a big job takes ~n/32 waves of ~1.3 ms steps, while
# the bench-width (1024) forward runs up to 1024 envs per ~6 ms step and
# is exact. Measured on an RTX 5090 (2026-10-02, 20 plays: exact 84 s,
# fast 17 s) and modelled, the crossover is ~200-300 plays, so from
# FAST_MAX_PLAYS on --fast runs as --exact (faster AND bench-exact).
FAST_MAX_PLAYS = 256


def fast_mode(requested, n_plays):
    """Whether a run uses the width-32 forward: only when asked AND the
    job is small enough for it to be the faster one."""
    return bool(requested) and n_plays < FAST_MAX_PLAYS


def slug(s, n=24):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:n].lower() or "song"


def play_of(row):
    """The play counters a replay is checked on (publish_replays.py's)."""
    return {k: row.get(f) for k, f in PLAY_KEYS}


def row_key(row):
    """Cross-catalog level identity of a results row (bench_combine's
    lift_ss_bench keys ScoreSaber-only rows the same way)."""
    from beat_arena_env import beatleader
    return row.get("level_key") or beatleader.level_key(
        row["song_hash"], row["difficulty"], row.get("characteristic",
                                                     "Standard"))


def _ss_cat(v):
    return {"leaderboard_id": v["leaderboard_id"], "stars": v["stars"],
            "max_pp": v["max_pp"], "api_max_score": v.get("api_max_score")}


def _bl_cat(v):
    return {k: v.get(k) for k in ("leaderboard_id", "stars", "pass_rating",
                                  "acc_rating", "tech_rating",
                                  "predicted_acc", "api_max_score")}


def source_entry(row, catalog):
    """The bench TASK entry behind a results row, in its catalog's shape
    (what finish_slot eats): every catalog field it reads rides in the
    row (or its ss / bl sub-result) already."""
    base = {"hash": row["song_hash"], "song": row["song"],
            "mapper": row.get("mapper"), "difficulty": row["difficulty"]}
    if catalog == "both":
        return {**base, "level_key": row["level_key"],
                "characteristic": row.get("characteristic", "Standard"),
                "ss": _ss_cat(row["ss"]) if row.get("ss") else None,
                "bl": _bl_cat(row["bl"]) if row.get("bl") else None}
    if catalog == "beatleader":
        return {**base, **_bl_cat(row),
                "characteristic": row.get("characteristic", "Standard")}
    return {**base, **_ss_cat(row)}


class Source:
    """One source bench file: its regime and rows by level identity."""

    def __init__(self, path):
        from beat_arena_env.constants import parse_tau_scale
        self.path = os.path.abspath(path)
        with open(self.path, encoding="utf-8") as f:
            self.data = json.load(f)
        d = self.data
        name = os.path.basename(path)
        if d.get("composite"):
            raise SystemExit(f"error: {name} is a composite: pass it as "
                             f"--composite (with --sources)")
        if d.get("obs_window"):
            raise SystemExit(f"error: {name}: obs_window benches are not "
                             f"reproducible on this sim")
        if d.get("mirror"):
            raise SystemExit(f"error: {name}: mirrored benches are not "
                             f"supported")
        self.catalog = d.get("catalog", "scoresaber")
        self.sim_dtype = d.get("sim_dtype", "f32")
        self.width = int(d.get("forward_width") or 1024)
        self.tick_hz = int(d.get("tick_hz") or 30)
        self.repeat = int(d.get("action_repeat") or 1)
        self.tau = [float(v) for v in (d.get("tau_scale")
                                       or parse_tau_scale("default"))]
        self.rows = {}
        for e in d["levels"]:
            if "skipped" not in e:
                self.rows[row_key(e)] = e


def plan_bench(args, mhash):
    """Jobs from a plain bench: make_top10_replays' selection."""
    import make_top10_replays as mt
    src = Source(args.bench)
    if src.data.get("model_hash") and src.data["model_hash"] != mhash:
        raise SystemExit(f"error: the checkpoint hashes to {mhash}, "
                         f"{args.bench} was run on {src.data['model_hash']}")
    board = args.board or ("bl" if src.catalog == "beatleader" else "ss")
    pool = mt.bench_pool(src.data, board, fallback=bool(args.levels))
    if args.stars is not None:
        lo, hi = args.stars
        pool = [e for e in pool if lo <= float(e.get("stars", 0)) < hi]
    if args.levels:
        top = mt.select_levels(pool, args.levels)
    elif args.all:
        top = sorted(pool, key=lambda e: -e[args.sort])
    else:
        top = sorted(pool, key=lambda e: -e[args.sort])[:args.top]
    label = args.label or ("sel" if args.levels else
                           "top" + ("acc" if args.sort == "accuracy"
                                    else "pp"))
    jobs = []
    for rank, view in enumerate(top, 1):
        row = src.rows[row_key(view)]
        jobs.append({"src": src, "row": row, "picks": [],
                     "level": mt.level_identity(row), "pp": view["pp"],
                     "name": f"{args.tag}_{label}{rank:02d}_"
                             f"{slug(row['song'])}_{view['pp']:.0f}pp"})
    return jobs, {src.path: src}


def plan_composite(args, mhash):
    """Jobs from a composite's picks: (source bench, level), deduplicated."""
    import make_top10_replays as mt
    with open(args.composite, encoding="utf-8") as f:
        comp = json.load(f)
    if not comp.get("composite"):
        raise SystemExit(f"error: {args.composite} is not a composite")
    if comp.get("weights_hash") != mhash:
        raise SystemExit(f"error: the checkpoint hashes to {mhash}, the "
                         f"composite is of {comp.get('weights_hash')}")
    recs = [r for r in comp["levels"] if "skipped" not in r]
    by_key = {r["level_key"]: r for r in recs}
    rank_board = args.board or "ss"
    if args.levels:
        missing = [k for k in args.levels if k not in by_key]
        if missing:
            raise SystemExit(f"error: --levels not in the composite: "
                             f"{missing}")
        sel = [by_key[k] for k in args.levels]
    else:
        ranked = sorted((r for r in recs if r.get(rank_board)),
                        key=lambda r: -r[rank_board]["pp"])
        sel = ranked if args.all else ranked[:args.top]
    boards = (args.board,) if args.board else ("ss", "bl")
    sources, jobs, seen = {}, [], {}
    for rank, rec in enumerate(sel, 1):
        for board in boards:
            name = rec.get(f"{board}_from")
            if not rec.get(board) or name is None:
                continue
            path = os.path.join(args.sources, name)
            if path not in sources:
                sources[path] = Source(path)
                if sources[path].data.get("model_hash") != mhash:
                    raise SystemExit(f"error: source {name} is of another "
                                     f"model")
            src = sources[path]
            k = (src.path, rec["level_key"])
            if k in seen:                       # both picks from one run
                seen[k]["picks"].append((rec["level_key"], board))
                continue
            row = src.rows.get(rec["level_key"])
            if row is None:
                raise SystemExit(f"error: {name} has no played level "
                                 f"{rec['level_key']} ({board} pick)")
            job = {"src": src, "row": row,
                   "picks": [(rec["level_key"], board)],
                   "level": mt.level_identity(rec), "rank": rank,
                   "pp": rec[board]["pp"]}
            seen[k] = job
            jobs.append(job)
    for job in jobs:
        tag = "+".join(b for _, b in job["picks"])
        job["name"] = (f"{args.tag}_{tag}{job['rank']:02d}_"
                       f"{slug(job['row']['song'])}_{job['src'].tick_hz}hz")
    return jobs, sources, comp


# ---------------------------------------------------------------------------
# writer pool (spawn; started before CUDA init)

def _pool_init():
    import torch
    torch.set_num_threads(1)


def _pool_noop():
    return os.getpid()


def write_replay(task):
    """Pool job: the publish-ready replay of one finished play (and the
    ScoreSaber .dat + decode check with --dat). Never raises: errors come
    back as {"error": ...}."""
    t0 = time.perf_counter()
    try:
        from beat_arena_env.rust_env import record_header, write_replay_jsonl
        m = dict(task["map"])
        if m["audio_offset_s"] is None:
            # a chart-cache pickle from before 2026-10-02 has no clock
            # shift: re-parse the raw map (BeatSaver on a miss) for it
            from beat_arena_env import scoresaber
            from beat_arena_env.beatsaver import load_beatmap
            md = scoresaber.ensure_map_dats(m["hash"], task["map_cache"])
            _, meta = load_beatmap(md, m["difficulty"], first_note_s=2.0)
            m["audio_offset_s"] = -meta["shift"]
        header = record_header(
            task["chart"], task["walls"], tick_hz=task["tick_hz"],
            action_repeat=task["repeat"], tau_scale=task["tau"],
            gears=task["gears"], map=m, level=task["level"],
            model_hash=task["model_hash"])
        gz = task["out"].endswith(".gz")
        n = write_replay_jsonl(task["out"], header, task["rec"],
                               compact=gz, gzip_level=6 if gz else None)
        out = {"out": task["out"], "bytes": os.path.getsize(task["out"]),
               "ticks": n}
        if task["dat"]:
            import convert_scoresaber as conv
            from beat_arena_env import scoresaber
            sys.path.insert(0, os.path.join(HERE, "tests"))
            from test_convert_scoresaber import decode_dat
            md = scoresaber.ensure_map_dats(m["hash"], task["map_cache"])
            dat = re.sub(r"\.jsonl(\.gz)?$", ".dat", task["out"])
            conv.convert(task["out"], md, m["hash"], m["difficulty"], dat)
            d = decode_dat(dat)
            if d["final_score"] != task["score"]:
                raise AssertionError(f".dat decode {d['final_score']} != "
                                     f"play {task['score']}")
            out["dat"] = dat
        out["secs"] = time.perf_counter() - t0
        return out
    except Exception as e:
        return {"out": task["out"], "error": f"{type(e).__name__}: {e}\n"
                + traceback.format_exc()}


# ---------------------------------------------------------------------------
# the batched rollout

def make_forward(policy, width, device, no_graphs=False):
    """The bench's forward at a fixed width: CUDA graph, else eager."""
    import torch
    if device.type == "cuda" and not no_graphs:
        from bench_scoresaber import _GraphedArgmax
        return _GraphedArgmax(policy, width, device), "graphed"

    def eager(obs_t):
        with torch.no_grad():
            logits, _ = policy.forward_stacked(obs_t.to(device))
        return logits.argmax(-1)
    return eager, "eager"


def run_group(jobs, policy, device, width, sim_mod, args, on_done,
              eager=False):
    """Play `jobs` (one sim dtype, one forward width) in waves of up to
    `width` envs. A finished env takes the next pending job of the same
    control rate + action repeat (an env's rate is fixed at construction);
    jobs left without a matching env start the next wave."""
    import torch
    from bench_scoresaber import finish_slot
    from beat_arena_env.constants import N_JOINTS, OBS_DIM
    fwd, fwd_path = make_forward(policy, width, device, eager)
    pin = device.type == "cuda"
    obs_t = torch.zeros((width, OBS_DIM), dtype=torch.float32,
                        pin_memory=pin)
    obs_np = obs_t.numpy()
    stats = {"steps": 0, "ticks": 0, "forward": fwd_path}
    pending = list(jobs)
    while pending:
        n = min(len(pending), width, args.max_envs or width)
        first, pending = pending[:n], pending[n:]
        rate = [(j["src"].tick_hz, j["src"].repeat) for j in first]
        sim = sim_mod.VecBeatSim(
            n, max_steps=100_000_000, combo_walk=True, return_events=False,
            gears=policy.n_gears, tick_hz_env=[r[0] for r in rate],
            action_repeat_env=[r[1] for r in rate])
        obs_buf = obs_np[:n]
        act_np = np.zeros((n, N_JOINTS), dtype=np.int64)
        slots = [None] * n

        def start(i, job):
            p = job["payload"]
            sim.set_tau_scale_env(i, job["src"].tau)
            sim.set_chart(i, p["tuples"], p.get("walls", []))
            sim.pop_torque_env(i)            # this level's torque only
            sim.obs_one_into(i, obs_buf)
            sim.set_record_env(i, True)
            slots[i] = job

        for i, job in enumerate(first):
            start(i, job)
        active = n
        while active:
            try:
                acts = fwd(obs_t)
            except Exception as e:           # graph capture failed
                print(f"[cuda-graphs] falling back to eager: {e}",
                      flush=True)
                fwd, stats["forward"] = make_forward(policy, width, device,
                                                     True)
                acts = fwd(obs_t)
            act_np[:] = acts[:n].cpu().numpy()
            _, dones, _ = sim.step_into(act_np, obs_buf)
            stats["steps"] += 1
            for i in range(n):
                job = slots[i]
                if job is None:
                    continue
                ticks = sim.wrist_rev(i)[2]
                truncated = (not dones[i] and args.max_ticks
                             and ticks >= args.max_ticks)
                if not (dones[i] or truncated):
                    continue
                combo = sim.combo(i)
                g, b, m, n_scoreable, _, bombs = sim.counters(i)
                c_pre, c_acc, c_post = sim.components(i)
                fl_rad, fl_ticks = sim.flail(i)
                wr_l, wr_r, wr_ticks = sim.wrist_rev(i)
                w_ent, t_flags = sim.term(i)
                tq = sim.pop_torque_env(i)[:4]
                row = finish_slot(
                    {"entry": job["entry"],
                     "max_score": job["payload"]["max_score"],
                     "duration_s": job["payload"]["duration_s"]},
                    {"good_cuts": g, "bad_cuts": b, "misses": m,
                     "n_notes": n_scoreable, "bombs_hit": bombs,
                     "score_pre": c_pre, "score_acc": c_acc,
                     "score_post": c_post, "flail_rad": fl_rad,
                     "flail_ticks": fl_ticks, "wrist_rev_l": wr_l,
                     "wrist_rev_r": wr_r, "ticks": wr_ticks,
                     "wall_enters": w_ent, "term_flags": t_flags,
                     "torque": tq},
                    combo, tick_hz=job["src"].tick_hz)
                rec = sim.take_record_env(i)
                sim.set_record_env(i, False)
                stats["ticks"] += wr_ticks
                slots[i] = None
                active -= 1
                on_done(job, row, rec, bool(truncated), stats["forward"])
                nxt = next((k for k, j in enumerate(pending)
                            if (j["src"].tick_hz, j["src"].repeat)
                            == rate[i]), None)
                if nxt is not None:
                    start(i, pending.pop(nxt))
                    active += 1
                else:
                    sim.set_chart(i, [])         # idle until the wave ends
        del sim
    return stats


# ---------------------------------------------------------------------------
# write-back

def composite_views(new, rec):
    """The ss / bl pick views of a new play, for a composite record: a
    `both` row carries them; a single-catalog row (a lifted ScoreSaber-
    only source) has its SS view top-level and gets its BL view priced
    from the composite's own BL entry, as tools/bench_add_bl.py does."""
    from bench_scoresaber import bl_result
    if "level_key" in new:
        return new.get("ss"), new.get("bl")
    ss = None
    if "max_pp" in new:
        ss = {"leaderboard_id": new["leaderboard_id"], "stars": new["stars"],
              "max_pp": new["max_pp"],
              "api_max_score": new.get("api_max_score"),
              "accuracy": new["accuracy"], "pp": new["pp"]}
    bl = None
    if rec.get("bl"):
        bl = bl_result(_bl_cat(rec["bl"]), new["score"], new["max_score"],
                       new.get("terminated_tick") is not None)
    return ss, bl


def write_back(results, sources, comp, args, stamp):
    """Replace mismatching rows in their source benches (save_results
    recomputes the totals) and patch the composite's picks they serve."""
    import bench_scoresaber as bs
    by_src = {}
    for r in results:
        if r["status"] == "mismatch":
            by_src.setdefault(r["job"]["src"].path, []).append(r)
    for path, items in by_src.items():
        src = sources[path]
        data = src.data
        levels = {bs.task_key(e): e for e in data["levels"]}
        log = []
        for r in items:
            old, new = r["job"]["row"], dict(r["row"])
            new["replay_override"] = {**stamp, "old": play_of(old)}
            levels[bs.task_key(old)] = new
            log.append({"level": bs.task_key(old), "old": play_of(old),
                        "new": play_of(new)})
        data.setdefault("replay_overrides", []).append({**stamp,
                                                        "levels": log})
        bs.save_results(path, data, levels)
        print(f"[write-back] {os.path.basename(path)}: {len(items)} rows "
              f"replaced, totals re-saved", flush=True)
    if comp is None:
        return
    from tools import bench_combine
    by_key = {r["level_key"]: r for r in comp["levels"]
              if "level_key" in r}
    log = []
    for r in results:
        if r["status"] not in ("match", "mismatch"):
            continue
        new = r["row"]
        for key, board in r["job"]["picks"]:
            rec = by_key[key]
            play = rec if board == "ss" else rec.get("bl_play") or {}
            if play_of(play) == play_of(new):
                continue
            ss, bl = composite_views(new, rec)
            if board == "ss":
                for f in bench_combine.PLAY:
                    rec[f] = new.get(f)
                rec["ss"] = ss
            else:
                rec["bl_play"] = {f: new.get(f) for f in bench_combine.PLAY}
                rec["bl"] = bl
            log.append({"level": key, "pick": board, "old": play_of(play),
                        "new": play_of(new)})
    if log:
        bench_combine.recompute_totals(comp)
        comp.setdefault("replay_overrides", []).append({**stamp,
                                                        "levels": log})
        tmp = args.composite + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=1)
        os.replace(tmp, args.composite)
    print(f"[write-back] composite: {len(log)} picks patched, totals "
          f"recomputed", flush=True)


# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bench", help="a plain bench results JSON")
    src.add_argument("--composite",
                     help="a best-of composite (tools/bench_combine.py)")
    p.add_argument("--sources", default=None,
                   help="directory holding the composite's input benches "
                        "(its ss_from / bl_from files)")
    p.add_argument("--tag", required=True, help="replay filename prefix")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--levels", nargs="+", default=None, metavar="KEY",
                     help="exactly these levels (level_key; leaderboard ids "
                          "on single-catalog benches)")
    sel.add_argument("--all", action="store_true",
                     help="every played level (composite: every pick)")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--sort", choices=["pp", "accuracy"], default="pp")
    p.add_argument("--stars", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--board", choices=("ss", "bl"), default=None,
                   help="leaderboard view (plain bench, as "
                        "make_top10_replays.py); composite: rank by it and "
                        "record only its picks (default: rank by ss, both "
                        "picks)")
    p.add_argument("--label", default=None)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--exact", action="store_true",
                      help="forward at each source bench's forward_width "
                           "(absent = 1024) — the default")
    mode.add_argument("--fast", action="store_true",
                      help=f"forward at width {FAST_WIDTH} (plays may "
                           f"drift from their bench rows); jobs of "
                           f"{FAST_MAX_PLAYS}+ plays run as --exact, which "
                           f"is faster there")
    p.add_argument("--forward-width", type=int, default=None,
                   help="override the forward width (tests / debugging)")
    p.add_argument("--max-envs", type=int, default=0,
                   help="cap on concurrent envs (default: the width)")
    p.add_argument("--write-back", action="store_true",
                   help="replace mismatching rows in the source benches "
                        "and patch the composite's picks")
    p.add_argument("--dat", action="store_true",
                   help="also write + decode-check ScoreSaber .dat files")
    p.add_argument("--jsonl", action="store_true",
                   help="write plain .jsonl (make_top10_replays' format) "
                        "instead of compact .jsonl.gz")
    p.add_argument("--writers", type=int,
                   default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                   help="writer processes (0 = write inline)")
    p.add_argument("--out-dir", default="replays")
    p.add_argument("--map-cache", default=os.path.join(HERE, "maps",
                                                       "scoresaber"))
    p.add_argument("--chart-cache", default=os.path.join(HERE, "maps",
                                                         "chart_cache"))
    p.add_argument("--device", default=None)
    p.add_argument("--no-graphs", action="store_true",
                   help="eager forward on CUDA too")
    p.add_argument("--max-ticks", type=int, default=0,
                   help="debug: stop each play after N sim ticks")
    a = p.parse_args(argv)
    if a.composite and not a.sources:
        p.error("--composite needs --sources")
    return a


def main(argv=None):
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    t_start = time.perf_counter()
    # writer pool FIRST: spawn workers before anything initializes CUDA
    pool = None
    if args.writers > 0:
        pool = ProcessPoolExecutor(
            max_workers=args.writers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_pool_init)
        for f in [pool.submit(_pool_noop) for _ in range(args.writers)]:
            f.result()

    import torch
    import bench_scoresaber as bs
    from policy import load_policy
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    mhash = bs.model_hash(ck["state_dict"])
    z_asinh = float(ck.get("arch", {}).get(
        "z_asinh", ck.get("args", {}).get("z_asinh", 0.0)))
    if not z_asinh > 0:
        raise SystemExit(f"{args.ckpt}: z_asinh {z_asinh:g} — predates the "
                         f"asinh z encoding")
    if os.environ.get("BEAT_Z_ASINH") not in (None, repr(z_asinh)):
        raise SystemExit(f"BEAT_Z_ASINH={os.environ['BEAT_Z_ASINH']} "
                         f"disagrees with the checkpoint's {z_asinh}")
    os.environ["BEAT_Z_ASINH"] = repr(z_asinh)   # before any sim exists

    comp = None
    if args.composite:
        jobs, sources, comp = plan_composite(args, mhash)
    else:
        jobs, sources = plan_bench(args, mhash)
    if not jobs:
        raise SystemExit("nothing to record")

    # charts (chart cache; BeatSaver only on a miss) — a failure drops
    # that job only
    ver = bs.parser_version()
    if args.chart_cache:
        os.makedirs(args.chart_cache, exist_ok=True)
    results, ok_jobs = [], []
    for job in jobs:
        job["entry"] = source_entry(job["row"], job["src"].catalog)
        try:
            p = bs.load_or_parse(job["entry"], args.map_cache,
                                 args.chart_cache, ver)
            if not p["tuples"]:
                raise ValueError("no notes parsed")
            job["payload"] = p
            ok_jobs.append(job)
        except Exception as e:
            results.append({"job": job, "status": "failed",
                            "error": f"chart: {type(e).__name__}: {e}"})
            print(f"FAILED {job['name']}: chart: {type(e).__name__}: {e}",
                  flush=True)

    policy = load_policy(ck)
    del ck
    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"))
    policy = policy.to(device).eval()
    os.makedirs(args.out_dir, exist_ok=True)
    ext = ".jsonl" if args.jsonl else ".jsonl.gz"

    futures = {}
    n_total = len(jobs)

    def collect(block=False):
        if not futures:
            return
        done, _ = wait(list(futures), timeout=None if block else 0,
                       return_when=FIRST_COMPLETED)
        for f in done:
            r = futures.pop(f)
            _report_write(r, f.result())

    def _report_write(r, w):
        if w.get("error"):
            r["status"] = "failed"
            r["error"] = "write: " + w["error"]
            print(f"FAILED {r['job']['name']}: write: {w['error']}",
                  flush=True)
        else:
            r["write"] = w

    def on_done(job, row, rec, truncated, fwd_path):
        want = play_of(job["row"])
        got = play_of(row)
        status = ("truncated" if truncated else
                  "match" if got == want else "mismatch")
        r = {"job": job, "row": row, "status": status, "forward": fwd_path}
        results.append(r)
        k = sum(1 for x in results if x["status"] != "failed")
        picks = " ".join(f"{b} pick" for _, b in job["picks"])
        print(f"[{k}/{n_total}] {job['row']['song'][:34]!r} "
              f"[{job['row']['difficulty']}] {job['src'].tick_hz} Hz"
              + (f" ({picks})" if picks else "")
              + f": score {got['score']} vs bench {want['score']}, combo "
              f"{got['combo']}/{want['combo']}, miss {got['miss']}/"
              f"{want['miss']} -> {status.upper()}", flush=True)
        src = job["src"]
        task = {"out": os.path.join(args.out_dir, job["name"] + ext),
                "chart": job["payload"]["tuples"],
                "walls": job["payload"].get("walls", []),
                "tick_hz": src.tick_hz, "repeat": src.repeat,
                "tau": src.tau, "gears": policy.n_gears,
                "map": {"hash": row["song_hash"],
                        "difficulty": row["difficulty"],
                        "song": row["song"],
                        "audio_offset_s": (-job["payload"]["shift"]
                                           if "shift" in job["payload"]
                                           else None)},
                "level": job["level"], "model_hash": mhash,
                "rec": rec, "dat": args.dat and not truncated,
                "score": row["score"], "map_cache": args.map_cache}
        if pool is None:
            _report_write(r, write_replay(task))
            return
        futures[pool.submit(write_replay, task)] = r
        # backpressure: captures wait in this process until written
        while len(futures) > 4 * args.writers:
            collect(block=True)

    fast = fast_mode(args.fast, len(ok_jobs))
    if args.fast and not fast:
        print(f"--fast: {len(ok_jobs)} plays >= {FAST_MAX_PLAYS}, running "
              f"--exact instead (the bench-width forward is faster at this "
              f"size, and exact)", flush=True)
    groups = {}
    for job in ok_jobs:
        src = job["src"]
        width = (args.forward_width or
                 (FAST_WIDTH if fast else src.width))
        # a --no-graphs bench ran the eager forward: reproduce that too
        eager = args.no_graphs or (not fast and
                                   src.data.get("forward") == "eager")
        groups.setdefault((src.sim_dtype, width, eager), []).append(job)
    stats = []
    for (dtype, width, eager), gjobs in sorted(groups.items()):
        if dtype == "f32":
            import beat_arena_rs32 as sim_mod
        else:
            import beat_arena_rs as sim_mod
        if not hasattr(sim_mod.VecBeatSim, "take_record_env"):
            raise SystemExit(f"{sim_mod.__name__} predates the batched "
                             f"recorder: rebuild it (maturin, rust/ or "
                             f"rust32/)")
        # longest first: short plays refill the envs long ones leave idle
        gjobs.sort(key=lambda j: -j["payload"]["duration_s"])
        print(f"=== {len(gjobs)} plays: sim {dtype}, forward width {width} "
              f"on {device}", flush=True)
        st = run_group(gjobs, policy, device, width, sim_mod, args, on_done,
                       eager)
        stats.append((dtype, width, st))
    while futures:
        collect(block=True)
    if pool is not None:
        pool.shutdown()

    n = {s: sum(1 for r in results if r["status"] == s)
         for s in ("match", "mismatch", "truncated", "failed")}
    wall = time.perf_counter() - t_start
    ticks = sum(st["ticks"] for _, _, st in stats)
    print(f"\n{len(results)} plays: {n['match']} EXACT, {n['mismatch']} "
          f"MISMATCH, {n['truncated']} truncated, {n['failed']} failed | "
          f"{ticks:,} sim ticks in {wall:.1f} s "
          f"({ticks / max(wall, 1e-9):,.0f} ticks/s)", flush=True)
    for r in results:
        if r["status"] == "mismatch":
            print(f"  MISMATCH {r['job']['name']}: replay "
                  f"{play_of(r['row'])} != bench {play_of(r['job']['row'])}")
    if args.write_back and not args.max_ticks:
        if any(r["status"] == "failed" for r in results):
            print("[write-back] note: failed jobs are left as they are",
                  flush=True)
        stamp = {"at": datetime.now(timezone.utc).isoformat(
                     timespec="seconds"),
                 "tool": "make_replays_batched.py", "model_hash": mhash,
                 "device": device.type,
                 "forward": sorted({st["forward"] for _, _, st in stats}),
                 "forward_width": sorted({w for _, w, _ in stats}),
                 "mode": "fast" if fast else "exact"}
        write_back(results, sources, comp, args, stamp)
    elif args.write_back:
        print("[write-back] skipped: --max-ticks plays are truncated",
              flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(1 if any(r["status"] == "failed" for r in res) else 0)
