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
               "walls", "map", "game", "player"]
MAP_KEYS = ["hash", "difficulty", "song", "audio_offset_s"]
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
    if isinstance(out.get("config"), dict):
        out["config"] = pick(out["config"], CONFIG_KEYS, "header.config.", dropped)
    out["model"] = model_id
    out["model_hash"] = mhash.replace("sha256:", "")[:12]
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
    brate = rate_of(bench)
    mhash = weights_hash(a.ckpt)
    if mhash != bench["model_hash"]:
        sys.exit(f"error: {a.ckpt} hashes to {mhash}, the bench is "
                 f"{bench['model_hash']}")
    levels = {}
    for e in bench["levels"]:
        if "skipped" not in e:
            levels.setdefault((e["song_hash"].lower(), e["difficulty"]), []).append(e)

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
        if rate_of(h) != brate:
            sys.exit(f"error: {name}: rate {rate_of(h)} != bench {brate}")
        m = h.get("map")
        if not m:
            sys.exit(f"error: {name}: no header.map — run python/annotate_replay_audio.py")
        cands = levels.get((m["hash"].lower(), m["difficulty"]), [])
        if len(cands) != 1:
            sys.exit(f"error: {name}: {len(cands)} bench levels for "
                     f"{m['hash']} {m['difficulty']}")
        e = cands[0]
        got = walk_events(ticks)
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
        fid = file_id(mhash, *brate, e["leaderboard_id"])
        blob = gzip.compress(("\n".join(clean) + "\n").encode(), 9, mtime=0)
        if len(blob) >= MAX_BYTES:
            sys.exit(f"error: {name}: {len(blob)} bytes gzipped, over the "
                     f"25 MiB asset limit")
        plans.append((name, e, fid, blob, dropped, got))

    # ---- write ------------------------------------------------------------
    os.makedirs(os.path.join(a.store, "r"), exist_ok=True)
    mpath = os.path.join(a.store, "manifest.json")
    manifest = (json.load(open(mpath, encoding="utf-8"))
                if os.path.exists(mpath) else {"version": 1, "models": {}})
    entry = manifest["models"].setdefault(a.model, {})
    if entry.get("modelHash") not in (None, mhash):
        sys.exit(f"error: the store already has {a.model} with weights "
                 f"{entry['modelHash']}")
    entry.update({"modelHash": mhash, "tickHz": brate[0],
                  "actionRepeat": brate[1], "obsWindow": brate[2]})
    reps = entry.setdefault("replays", {})
    for name, e, fid, blob, dropped, got in plans:
        rel = f"r/{fid}.jsonl.gz"
        with open(os.path.join(a.store, rel), "wb") as f:
            f.write(blob)
        reps[str(e["leaderboard_id"])] = {
            "file": rel, "song": e["song"], "difficulty": e["difficulty"],
            "pp": round(e["pp"], 2), "score": e["score"], "bytes": len(blob)}
        print(f"{rel}  {len(blob) / 1048576:5.2f} MiB  {e['pp']:7.2f}pp  "
              f"{e['song'][:40]} [{e['difficulty']}]  <- {name}  "
              f"(score {got['score']} = bench, dropped: "
              f"{', '.join(sorted(dropped)) or 'nothing'})")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"{len(plans)} replays -> {a.store} ({len(reps)} for {a.model})")


if __name__ == "__main__":
    main()
