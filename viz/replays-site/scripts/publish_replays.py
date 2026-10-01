"""Publish recorded replays to the replay store that replays.flee.systems is
built from (viz/replays-site; the store itself is NOT in git).

    python viz/replays-site/scripts/publish_replays.py \\
        --model v60-60hz --bench <bench.json> --ckpt <leaf.pt> \\
        [--store ~/beat-arena-volume-backup/replays-site] replays/*.jsonl

Run it with the repo's python venv (it needs torch to hash the checkpoint,
and python/ for the score walk).

For each replay (a make_top10_replays.py recording, stamped with header.map
by annotate_replay_audio.py):

  1. MATCH it to the bench it must be linked from. The weights hash of
     --ckpt must equal the bench's model_hash (and the header's, if it has
     one); the header's tick_hz / action_repeat / obs_window must equal the
     bench's (absent = 30 / 1 / off on both sides); header.map {hash,
     difficulty} must name exactly one bench level; and the play itself must
     reproduce that level's bench entry: good / bad / miss / bomb counts,
     max combo and multiplied score, walked from the recorded events. Any
     mismatch aborts before anything is written.
  2. SANITIZE it. The header is rebuilt from an allow-list (what playback
     needs plus harmless metadata), then gains `model` (the PUBLIC id) and
     `model_hash` (a 12-hex prefix of the weights hash). Tick lines keep
     only the per-tick fields the viewer reads. Everything dropped is
     printed. Checkpoint paths, run names, bench file names and pod paths
     never reach the store: none of them is on the allow-list.
  3. WRITE r/<16 hex>.jsonl.gz (gzip -9, no name, mtime 0: byte-identical
     re-runs). The name is sha256("<model_hash>|<tick_hz>|<action_repeat>|
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
import gzip
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python"))

MAX_BYTES = 25 * 1024 * 1024          # Cloudflare static assets, per file
DEFAULT_STORE = os.path.expanduser("~/beat-arena-volume-backup/replays-site")

# what the viewer's parseReplay reads, plus harmless run metadata
HEADER_KEYS = ["version", "record_rev", "tick_hz", "action_repeat",
               "obs_window", "tau_scale", "track_anchor", "config", "chart",
               "walls", "map", "level", "game", "player"]
MAP_KEYS = ["hash", "difficulty", "song", "audio_offset_s"]
LEVEL_KEYS = ["level_key", "characteristic", "ss_leaderboard_id",
              "bl_leaderboard_id"]
CONFIG_KEYS = ["chart_len_s", "colors", "note_speed_min", "note_speed_max",
               "note_prob", "extra_prob", "dot_prob", "bomb_prob", "gears",
               "handle_len"]
TICK_KEYS = ["t", "time", "q", "twist", "omega", "sabers", "head", "action",
             "reward", "contacts", "feet", "activation", "torque", "events"]


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


def walk_events(ticks):
    """Replay tallies from the recorded events (the viewer's event form)
    through the bench's own multiplied-score walk."""
    from beat_arena_env.scoresaber import ComboTracker
    walk = ComboTracker()
    good = bad = miss = bombs = 0
    for t in ticks:
        evs = []
        for e in t["events"]:
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


def composite_matches(bench, rec, header, got):
    """Which picks ("ss" / "bl") of composite record `rec` this replay
    reproduces: the pick's input run has the replay's rate and body, and the
    pick's play has the replay's counters."""
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
        if got == want:
            out.append(board)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True,
                   help="the model's PUBLIC id on the records site (e.g. v60-60hz)")
    p.add_argument("--bench", required=True, help="the published bench JSON")
    p.add_argument("--ckpt", required=True,
                   help="the checkpoint the replays were recorded from")
    p.add_argument("--store", default=DEFAULT_STORE)
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
    levels, by_key = {}, {}
    for e in bench["levels"]:
        if "skipped" not in e:
            levels.setdefault((e["song_hash"].lower(), e["difficulty"]), []).append(e)
            if "level_key" in e:
                by_key[e["level_key"]] = e

    # ---- check everything first; write nothing unless all pass ----------
    plans = []
    for path in a.replays:
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f.read().split("\n") if l.strip()]
        h = json.loads(lines[0])["header"]
        ticks = [json.loads(l) for l in lines[1:]]
        name = os.path.basename(path)
        if h.get("version") != "v10":
            sys.exit(f"error: {name}: replay version {h.get('version')}")
        if h.get("model_hash") and h["model_hash"] not in (mhash, mhash.replace("sha256:", "")):
            sys.exit(f"error: {name}: header model_hash {h['model_hash']} != {mhash}")
        if not comp and rate_of(h) != brate:
            sys.exit(f"error: {name}: rate {rate_of(h)} != bench {brate}")
        m = h.get("map")
        if not m:
            sys.exit(f"error: {name}: no header.map — run python/annotate_replay_audio.py")
        cands = levels.get((m["hash"].lower(), m["difficulty"]), [])
        lk = (h.get("level") or {}).get("level_key")
        if lk:
            cands = [by_key[lk]] if lk in by_key else []
        elif comp:
            cands = [e for e in cands if e.get("characteristic", "Standard") == "Standard"]
        if len(cands) != 1:
            sys.exit(f"error: {name}: {len(cands)} bench levels for "
                     f"{m['hash']} {m['difficulty']}")
        e = cands[0]
        got = walk_events(ticks)
        picks = None
        if comp:
            picks = composite_matches(bench, e, h, got)
            if not picks:
                sys.exit(f"error: {name}: replay {got} at rate {rate_of(h)} "
                         f"reproduces neither pick of {e['level_key']} "
                         f"(ss_from {e.get('ss_from')}, bl_from "
                         f"{e.get('bl_from')})")
        else:
            want = {"score": e["score"], "combo": e["highest_combo"],
                    "good": e["good_cuts"], "bad": e["bad_cuts"],
                    "miss": e["misses"], "bombs": e["bombs_hit"]}
            if got != want:
                sys.exit(f"error: {name}: replay {got} != bench entry {want}")
        dropped = set()
        header = sanitize_header(h, a.model, mhash, dropped)
        clean = [json.dumps({"header": header}, separators=(",", ":"))]
        for t in ticks:
            clean.append(json.dumps(pick(t, TICK_KEYS, "tick.", dropped),
                                    separators=(",", ":")))
        if comp:
            fid = file_id(mhash, *rate_of(h),
                          f"{e['level_key']}|{body_key(h.get('tau_scale'))}")
        else:
            fid = file_id(mhash, *brate, e["leaderboard_id"])
        blob = gzip.compress(("\n".join(clean) + "\n").encode(), 9, mtime=0)
        if len(blob) >= MAX_BYTES:
            sys.exit(f"error: {name}: {len(blob)} bytes gzipped, over the "
                     f"25 MiB asset limit")
        if any(p[2] == fid for p in plans):
            sys.exit(f"error: {name}: a second replay of the same level, "
                     f"rate and body in this batch")
        plans.append((name, e, fid, blob, dropped, got, picks, h))

    # ---- write ------------------------------------------------------------
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
    for _, _, fid, *_ in plans:
        if f"r/{fid}.jsonl.gz" in others:
            sys.exit(f"error: r/{fid}.jsonl.gz belongs to another model")
    reps = entry.setdefault("replays", {})
    for name, e, fid, blob, dropped, got, picks, h in plans:
        rel = f"r/{fid}.jsonl.gz"
        with open(os.path.join(a.store, rel), "wb") as f:
            f.write(blob)
        if comp:
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
                "bytes": len(blob)}
        else:
            pp = e["pp"]
            reps[str(e["leaderboard_id"])] = {
                "file": rel, "song": e["song"], "difficulty": e["difficulty"],
                "pp": round(e["pp"], 2), "score": e["score"], "bytes": len(blob)}
        print(f"{rel}  {len(blob) / 1048576:5.2f} MiB  {pp:7.2f}pp  "
              f"{e['song'][:40]} [{e['difficulty']}]  <- {name}  "
              + (f"[{'+'.join(picks)} pick, {rate_of(h)[0]} Hz] " if comp else "")
              + f"(score {got['score']} = bench, dropped: "
              f"{', '.join(sorted(dropped)) or 'nothing'})")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"{len(plans)} replays -> {a.store} ({len(reps)} for {a.model})")


if __name__ == "__main__":
    main()
