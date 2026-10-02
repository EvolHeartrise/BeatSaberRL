"""Publish recorded replays to the replay store that replays.flee.systems is
built from (viz/replays-site; the store itself is NOT in git).

    python viz/replays-site/scripts/publish_replays.py \\
        --model v60-60hz --bench <bench.json> --ckpt <leaf.pt> \\
        [--store ~/beat-arena-volume-backup/replays-site] [--jobs N] \\
        [--allow-mismatch] replays/*.rpl.gz

Run it with the repo's python venv (it needs torch to hash the checkpoint,
and python/ for the score walk).

For each replay (a format-2 .rpl.gz from make_replays_batched.py or
make_top10_replays.py — python/beat_arena_env/replay_format.py; header.map
written by the recorder):

  1. MATCH it to the bench it must be linked from. The weights hash of
     --ckpt must equal the bench's model_hash (and the header's, if it has
     one); the header's tick_hz / action_repeat / obs_window must equal the
     bench's (absent = 30 / 1 / off on both sides); header.map {hash,
     difficulty} must name exactly one bench level; and the play itself must
     reproduce that level's bench entry: good / bad / miss / bomb counts,
     max combo and multiplied score, walked from the recorded events. A
     replay that fails is reported and skipped (exit status 1); the others
     are published. --allow-mismatch downgrades the play check to a warning
     — for use after make_replays_batched.py --write-back has adopted the
     replays' plays (by default the check runs against the bench as it is
     on disk, i.e. the written-back one).
  2. SANITIZE it. The header is rebuilt from an allow-list (what playback
     needs plus harmless metadata), then gains `model` (the PUBLIC id) and
     `model_hash` (a 12-hex prefix of the weights hash). Only the column
     blocks the decoders know are kept, verbatim (the score walk decodes
     the event blocks alone). Everything dropped is printed. Checkpoint
     paths, run names, bench file names and pod paths never reach the
     store: none of them is on the allow-list.
  3. WRITE r/<16 hex>.rpl.gz (gzip -6, no name, mtime 0: byte-identical
     re-runs; each file as soon as it passes, files checked in parallel
     with --jobs). The name is sha256("<model_hash>|<tick_hz>|<action_repeat>|
     <obs_window 0/1>|<leaderboard_id>")[:16]: opaque on purpose — ad-block
     lists match URL substrings, and a model id like "v60" in a path got
     results/v60.json blocked by EasyPrivacy's `/v60.js` rule. It has to
     stay under Cloudflare's 25 MiB per-asset limit.
  4. RECORD it in <store>/manifest.json, which viz/site's build-data.mjs
     reads to put "Watch" links on the records site (it re-checks the
     weights hash, rate and level against the published bench), and which
     viz/replays-site/scripts/build.mjs reads to assemble the deploy dir.

Best-of composites (tools/bench_combine.py; bench header `composite`,
2026-10-01). A composite picks each level's ScoreSaber and BeatLeader play
independently from several runs of ONE checkpoint at different control
rates / bodies, so its replays have no single rate. For a composite:
  - --ckpt must hash to the composite's `weights_hash` (its model_hash is
    a pseudo-hash of the inputs);
  - the level is the header's `level.level_key` (make_top10_replays.py
    stamps it), else the unique Standard level of header.map;
  - the replay must reproduce a PICK of that level: the input run named by
    ss_from (or bl_from) must have the replay's tick_hz (action_repeat 1,
    no obs window) and body (header tau_scale), and the play counters must
    equal that pick's play (the record's own fields for the SS pick,
    `bl_play` for the BL pick). A replay matching neither pick aborts;
  - its file id is sha256("<weights>|<rate>|<level_key>|<body>")[:16] (a
    level can have two replays, one per pick run), and its manifest entry
    (keyed by that id) carries the level and the run it was recorded in
    (tickHz, tauScale) plus which picks it reproduces — never a file name.
    viz/site's build-data.mjs re-checks each against the composite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python"))

from beat_arena_env import replay_format as rf  # noqa: E402

MAX_BYTES = 25 * 1024 * 1024          # Cloudflare static assets, per file
DEFAULT_STORE = os.path.expanduser("~/beat-arena-volume-backup/replays-site")

# what the viewer's parseReplay reads, plus harmless run metadata
HEADER_KEYS = ["version", "format", "record_rev", "tick_hz",
               "action_repeat", "obs_window", "tau_scale", "body",
               "track_anchor", "config", "chart", "walls", "map", "level",
               "game", "player"]
MAP_KEYS = ["hash", "difficulty", "song", "audio_offset_s"]
LEVEL_KEYS = ["level_key", "characteristic", "ss_leaderboard_id",
              "bl_leaderboard_id"]
CONFIG_KEYS = ["chart_len_s", "colors", "note_speed_min", "note_speed_max",
               "note_prob", "extra_prob", "dot_prob", "bomb_prob", "gears",
               "handle_len"]
# the column blocks the decoders read (replay_format.py)
BLOCK_KEYS = ["omega", "q", "twist", "feet", "reward", "action", "sabers",
              "head", "activation", "torque", "contact_n", "contact_pair",
              "contact_p", "contact_j", "event_n", "event_code", "event_note",
              "event_val"]


def file_id(model_hash, tick_hz, repeat, obs_window, leaderboard_id):
    key = f"{model_hash}|{tick_hz}|{repeat}|{1 if obs_window else 0}|{leaderboard_id}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def rate_of(d):
    """(tick_hz, action_repeat, obs_window) with the absent defaults."""
    return (int(d.get("tick_hz") or 30), int(d.get("action_repeat") or 1),
            bool(d.get("obs_window") or False))


def weights_hash(ckpt_path):
    import torch
    from bench_scoresaber import model_hash
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return model_hash(ck["state_dict"])


def walk_events(tick_events):
    """Replay tallies from the recorded per-tick event lists (the viewer's
    event form) through the bench's own multiplied-score walk."""
    from beat_arena_env.scoresaber import ComboTracker
    walk = ComboTracker()
    good = bad = miss = bombs = 0
    for events in tick_events:
        evs = []
        for e in events:
            if "cut" in e:
                c = e["cut"]
                if c["good"]:
                    good += 1
                    evs.append((c["note"], "ok", c.get("points", 0)))
                else:
                    bad += 1
                    evs.append((c["note"], c.get("reason") or "bad", 0))
            elif "post" in e:
                evs.append((e["post"]["note"], "post", e["post"]["points"]))
            elif "miss" in e:
                miss += 1
                evs.append((e["miss"]["note"], "miss", 0))
            elif "bomb" in e:
                bombs += 1
                evs.append((e["bomb"]["note"], "bomb", 0))
            elif "wall_enter" in e:
                evs.append((-1, "wall_enter", 0))
        walk.consume(evs)
    return {"score": round(walk.score), "combo": walk.max_combo, "good": good,
            "bad": bad, "miss": miss, "bombs": bombs}


def pick(d, keys, where, dropped):
    out = {}
    for k in d:
        if k in keys:
            out[k] = d[k]
        else:
            dropped.add(f"{where}{k}")
    return out


def sanitize_header(h, model_id, mhash, dropped):
    out = pick(h, HEADER_KEYS, "header.", dropped)
    if isinstance(out.get("map"), dict):
        out["map"] = pick(out["map"], MAP_KEYS, "header.map.", dropped)
    if isinstance(out.get("level"), dict):
        out["level"] = pick(out["level"], LEVEL_KEYS, "header.level.", dropped)
    if isinstance(out.get("config"), dict):
        out["config"] = pick(out["config"], CONFIG_KEYS, "header.config.", dropped)
    out["model"] = model_id
    out["model_hash"] = mhash.replace("sha256:", "")[:12]
    return out


def same_body(a, b, tol=1e-6):
    """Two tau_scale lists (None = all 1.0) name the same body."""
    a = a or [1.0] * 16
    b = b or [1.0] * 16
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def body_key(tau):
    return ",".join(f"{v:.4f}" for v in (tau or [1.0] * 16))


def composite_matches(bench, rec, header, got, by_run_only=False):
    """Which picks ("ss" / "bl") of composite record `rec` this replay
    reproduces: the pick's input run has the replay's rate and body, and the
    pick's play has the replay's counters (`by_run_only`: run match alone —
    --allow-mismatch)."""
    runs = {r["file"]: r for r in bench["composite"]["inputs"]}
    out = []
    for board, play in (("ss", rec), ("bl", rec.get("bl_play"))):
        src = rec.get(f"{board}_from")
        if not rec.get(board) or src is None or play is None:
            continue
        run = runs[src]
        if rate_of(header) != (int(run.get("tick_hz") or 30), 1, False):
            continue
        if not same_body(header.get("tau_scale"), run.get("tau_scale")):
            continue
        want = {"score": play["score"], "combo": play["highest_combo"],
                "good": play["good_cuts"], "bad": play["bad_cuts"],
                "miss": play["misses"], "bombs": play["bombs_hit"]}
        if got == want or by_run_only:
            out.append(board)
    return out


_CTX = {}        # per-process check context (set by _init)


def _init(ctx):
    _CTX.clear()
    _CTX.update(ctx)


def check_one(path):
    """Verify + sanitize one replay and write it to a temp file in the
    store. Returns a plan dict, or {"name", "error"}."""
    name = os.path.basename(path)
    try:
        return _check_one(path, name)
    except Exception as e:
        return {"name": name, "error": f"{type(e).__name__}: {e}"}


def _check_one(path, name):
    c = _CTX
    bench, comp, mhash = c["bench"], c["comp"], c["mhash"]
    meta, data = rf.unpack(rf.read_raw(path))
    h = meta["header"]
    err = lambda msg: {"name": name, "error": msg}
    if h.get("version") != "v10":
        return err(f"replay version {h.get('version')}")
    if h.get("format") != rf.FORMAT:
        return err(f"replay format {h.get('format')} (need {rf.FORMAT})")
    if h.get("model_hash") and h["model_hash"] not in (mhash, mhash.replace("sha256:", "")):
        return err(f"header model_hash {h['model_hash']} != {mhash}")
    if not comp and rate_of(h) != c["brate"]:
        return err(f"rate {rate_of(h)} != bench {c['brate']}")
    m = h.get("map")
    if not m:
        return err("no header.map (record with map=..., as the replay "
                   "tools do)")
    cands = c["levels"].get((m["hash"].lower(), m["difficulty"]), [])
    lk = (h.get("level") or {}).get("level_key")
    if lk:
        cands = [c["by_key"][lk]] if lk in c["by_key"] else []
    elif comp:
        cands = [e for e in cands if e.get("characteristic", "Standard") == "Standard"]
    if len(cands) != 1:
        return err(f"{len(cands)} bench levels for {m['hash']} {m['difficulty']}")
    e = cands[0]
    dropped = set()
    got = walk_events(rf.events_only(meta, data))
    picks, warn = None, None
    if comp:
        picks = composite_matches(bench, e, h, got)
        if not picks:
            msg = (f"replay {got} at rate {rate_of(h)} reproduces neither "
                   f"pick of {e['level_key']} (ss_from {e.get('ss_from')}, "
                   f"bl_from {e.get('bl_from')})")
            if not c["allow_mismatch"]:
                return err(msg)
            picks = composite_matches(bench, e, h, got, by_run_only=True)
            if not picks:
                return err(msg + "; no pick was played at this rate/body")
            warn = msg + f" — published for the {'+'.join(picks)} pick (--allow-mismatch)"
    else:
        want = {"score": e["score"], "combo": e["highest_combo"],
                "good": e["good_cuts"], "bad": e["bad_cuts"],
                "miss": e["misses"], "bombs": e["bombs_hit"]}
        if got != want:
            if not c["allow_mismatch"]:
                return err(f"replay {got} != bench entry {want}")
            warn = f"replay {got} != bench entry {want} (--allow-mismatch)"
    header = sanitize_header(h, c["model"], mhash, dropped)
    if comp:
        fid = file_id(mhash, *rate_of(h),
                      f"{e['level_key']}|{body_key(h.get('tau_scale'))}")
    else:
        fid = file_id(mhash, *c["brate"], e["leaderboard_id"])
    keep = []
    for i, b in enumerate(meta["blocks"]):
        if b["name"] in BLOCK_KEYS:
            keep.append(i)
        else:
            dropped.add(f"block.{b['name']}")
    blob = rf.gz(rf.pack({"header": header, "ticks": meta["ticks"],
                          "blocks": [meta["blocks"][i] for i in keep]},
                         [data[i] for i in keep]), 6)
    del data
    if len(blob) >= MAX_BYTES:
        return err(f"{len(blob)} bytes gzipped, over the 25 MiB asset limit")
    tmp = os.path.join(c["store"], "r", f".{fid}{rf.EXT}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(blob)
    keep = {k: h.get(k) for k in ("tick_hz", "action_repeat", "obs_window",
                                  "tau_scale")}
    return {"name": name, "level_key": e.get("level_key"),
            "leaderboard_id": e.get("leaderboard_id"),
            "fid": fid, "tmp": tmp, "bytes": len(blob),
            "dropped": sorted(dropped), "got": got, "picks": picks,
            "h": keep, "warn": warn}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True,
                   help="the model's PUBLIC id on the records site (e.g. v60-60hz)")
    p.add_argument("--bench", required=True, help="the published bench JSON")
    p.add_argument("--ckpt", required=True,
                   help="the checkpoint the replays were recorded from")
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                   help="replays checked + compressed in parallel")
    p.add_argument("--allow-mismatch", action="store_true",
                   help="publish replays whose play differs from the bench "
                        "entry (composite: from both picks; it is then "
                        "linked to the pick(s) recorded at its rate and "
                        "body) with a warning instead of refusing them — "
                        "after make_replays_batched.py --write-back")
    p.add_argument("replays", nargs="+")
    a = p.parse_args()

    bench = json.load(open(a.bench, encoding="utf-8"))
    comp = bool(bench.get("composite"))
    brate = rate_of(bench)
    mhash = weights_hash(a.ckpt)
    want_hash = bench["weights_hash"] if comp else bench["model_hash"]
    if mhash != want_hash:
        sys.exit(f"error: {a.ckpt} hashes to {mhash}, the bench is "
                 f"{want_hash}")
    levels, by_key, by_lid = {}, {}, {}
    for e in bench["levels"]:
        if "skipped" not in e:
            levels.setdefault((e["song_hash"].lower(), e["difficulty"]), []).append(e)
            if "level_key" in e:
                by_key[e["level_key"]] = e
            if not comp:
                by_lid[e["leaderboard_id"]] = e

    # ---- store-level checks first --------------------------------------
    os.makedirs(os.path.join(a.store, "r"), exist_ok=True)
    mpath = os.path.join(a.store, "manifest.json")
    manifest = (json.load(open(mpath, encoding="utf-8"))
                if os.path.exists(mpath) else {"version": 1, "models": {}})
    entry = manifest["models"].setdefault(a.model, {})
    bhash = bench["model_hash"]          # composite: its pseudo-hash
    if entry.get("modelHash") not in (None, bhash):
        sys.exit(f"error: the store already has {a.model} with weights "
                 f"{entry['modelHash']}")
    if comp:
        entry.update({"modelHash": bhash, "weightsHash": mhash,
                      "composite": True})
    else:
        entry.update({"modelHash": mhash, "tickHz": brate[0],
                      "actionRepeat": brate[1], "obsWindow": brate[2]})
    # never overwrite another model's published file (ids are per weights,
    # so this cannot happen short of a hash collision)
    others = {r["file"] for mid, mm in manifest["models"].items() if mid != a.model
              for r in mm.get("replays", {}).values()}
    reps = entry.setdefault("replays", {})
    ctx = {"bench": bench, "comp": comp, "brate": brate, "mhash": mhash,
           "model": a.model, "levels": levels, "by_key": by_key,
           "store": a.store, "allow_mismatch": a.allow_mismatch}

    # ---- check + write each replay as it passes --------------------------
    def results():
        if a.jobs > 1 and len(a.replays) > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(
                    max_workers=min(a.jobs, len(a.replays)),
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init, initargs=(ctx,)) as ex:
                yield from ex.map(check_one, a.replays)
        else:
            _init(ctx)
            for path in a.replays:
                yield check_one(path)

    failed, done_fids = [], set()
    for r in results():
        name = r["name"]
        if r.get("error"):
            failed.append(name)
            print(f"error: {name}: {r['error']}", flush=True)
            continue
        fid, rel = r["fid"], f"r/{r['fid']}{rf.EXT}"
        problem = (f"a second replay of the same level, rate and body in "
                   f"this batch" if fid in done_fids else
                   f"{rel} belongs to another model" if rel in others
                   else None)
        if problem:
            os.remove(r["tmp"])
            failed.append(name)
            print(f"error: {name}: {problem}", flush=True)
            continue
        os.replace(r["tmp"], os.path.join(a.store, rel))
        done_fids.add(fid)
        if r["warn"]:
            print(f"warning: {name}: {r['warn']}", flush=True)
        got, picks, h = r["got"], r["picks"], r["h"]
        if comp:
            e = by_key[r["level_key"]]
            pp = max((e[b]["pp"] for b in picks), default=0.0)
            hz, rep_, obs = rate_of(h)
            reps[fid] = {
                "file": rel, "song": e["song"], "difficulty": e["difficulty"],
                "levelKey": e["level_key"],
                "ssId": (e.get("ss") or {}).get("leaderboard_id"),
                "blId": (e.get("bl") or {}).get("leaderboard_id"),
                "tickHz": hz, "actionRepeat": rep_, "obsWindow": obs,
                "tauScale": [round(v, 6) for v in (h.get("tau_scale") or [1.0] * 16)],
                "picks": picks, "score": got["score"],
                "pp": {b: round(e[b]["pp"], 2) for b in picks},
                "bytes": r["bytes"]}
        else:
            e = by_lid[r["leaderboard_id"]]
            pp = e["pp"]
            reps[str(e["leaderboard_id"])] = {
                "file": rel, "song": e["song"], "difficulty": e["difficulty"],
                "pp": round(e["pp"], 2), "score": got["score"],
                "bytes": r["bytes"]}
        print(f"{rel}  {r['bytes'] / 1048576:5.2f} MiB  {pp:7.2f}pp  "
              f"{e['song'][:40]} [{e['difficulty']}]  <- {name}  "
              + (f"[{'+'.join(picks)} pick, {rate_of(h)[0]} Hz] " if comp else "")
              + f"(score {got['score']}"
              + (" = bench" if not r["warn"] else " != bench")
              + f", dropped: {', '.join(r['dropped']) or 'nothing'})",
              flush=True)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"{len(done_fids)} replays -> {a.store} ({len(reps)} for {a.model})"
          + (f"; {len(failed)} FAILED: {', '.join(failed)}" if failed else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
