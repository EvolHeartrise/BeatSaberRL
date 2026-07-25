"""ScoreSaber bench: greedy-eval a checkpoint on every ranked map.

    python bench_scoresaber.py --ckpt runs/<run>/final.pt [--stars 0 4]

One deterministic argmax rollout per ranked leaderboard (deterministic
sim + fixed chart: one rollout IS the exact answer), scored with the
real combo multiplier and ScoreSaber's official PP curve (see
beat_arena_env/scoresaber.py). No replays are recorded.

Output: a JSON file per checkpoint (default bench/<run>_<ckpt>.json)
with the checkpoint's name, a sha256 of its weights, one result entry
per level (score, accuracy, pp, misses, bad cuts, highest combo, ...)
and the profile-style 0.965^rank weighted total PP. Re-running merges:
already-evaluated levels are skipped (resume/extend across star ranges;
--force re-evaluates), previously-skipped levels are retried, and the
weighted total is recomputed over everything on file. A different
checkpoint writing to the same file is an error, not a merge.

The ranked catalog (~6.1k leaderboards) is cached at
maps/scoresaber_catalog.json (--refresh-catalog to refetch) and map
chart data at maps/scoresaber/<hash>/ (audio stripped, ~100 KB/song).
Downloads run on a background thread while evaluation proceeds.

Parsed-chart cache (2026-07-25): beatmap parsing (JSON + dataclass
churn, ~39 ms/level catalog-wide) was measured as ~50% of a warm bench
run and, being pure Python, it holds the GIL — the producer thread
overlaps network only, not parsing. The producer therefore caches the
parsed result (note tuples + max score + duration) per
(hash, difficulty, speed_scale, parser-source-hash) as pickles under
--chart-cache; any edit to the parser/scoring sources invalidates the
cache automatically via the source hash. Also from the same session:
tasks run LONGEST-FIRST (durations sidecar, populated as maps are
parsed) so long maps can't straggle at the tail with mostly-idle
slots, and results are saved on a wall-clock timer instead of every 25
finishes (the every-25 rewrite was quadratic in catalog size).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import queue
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import torch

from beat_arena_env import scoresaber
from beat_arena_env.beatsaver import load_beatmap
from beat_arena_env.chart import note_tuple
from beat_arena_env.constants import OBS_DIM
from beat_arena_rs import VecBeatSim
from cuda_graphs import _Graphed
from policy import load_policy


class _GraphedArgmax(_Graphed):
    """Bench forward — policy logits + stacked argmax over the fixed
    (N, 286) batch as ONE CUDA graph (2026-07-25: N=512 forwards are
    launch-bound; same capture protocol as the trainer's rollout
    capture). Falls back to eager on any capture failure."""

    def __init__(self, policy, n, device):
        super().__init__()
        self.policy = policy
        self.s_obs = torch.zeros(n, OBS_DIM, device=device)

    def _compute(self):
        with torch.no_grad():
            logits, _ = self.policy(self.s_obs)
        return torch.stack(logits, dim=1).argmax(-1)

    def __call__(self, obs_np):
        self.s_obs.copy_(torch.from_numpy(obs_np))
        return self._run()


class _FusedArgmax:
    """Bench forward via the fused single-launch CUDA kernel
    (policy_cuda.FusedNotePolicy): logits + stacked argmax in ONE launch,
    no graph capture needed. Same call contract as _GraphedArgmax."""

    def __init__(self, policy, n, device):
        from policy_cuda import FusedNotePolicy
        self.fused = FusedNotePolicy(policy, device)
        self.s_obs = torch.zeros(n, OBS_DIM, device=device)

    def __call__(self, obs_np):
        self.s_obs.copy_(torch.from_numpy(obs_np))
        return self.fused.argmax(self.s_obs)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True, help="checkpoint .pt to bench")
    p.add_argument("--stars", type=float, nargs=2, metavar=("MIN", "MAX"),
                   default=None,
                   help="only evaluate leaderboards in this star range "
                        "(inclusive; default: all ranked maps)")
    p.add_argument("--out", type=str, default=None,
                   help="results JSON (default bench/<run>_<ckpt>.json); "
                        "merged with existing data for the same checkpoint")
    p.add_argument("--force", action="store_true",
                   help="re-evaluate levels already in the results file")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after N levels (0 = no limit; for smoke runs)")
    p.add_argument("--num-envs", type=int, default=256,
                   help="maps evaluated in parallel (policy is batched)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--speed-scale", type=float, default=0.3,
                   help="note speed = NJS * this (sightread convention)")
    p.add_argument("--max-song-s", type=float, default=0.0,
                   help="skip maps longer than this (0 = play everything)")
    p.add_argument("--map-cache", type=str, default="maps/scoresaber")
    p.add_argument("--chart-cache", type=str, default="maps/chart_cache",
                   help="parsed-chart pickle cache (auto-invalidated by a "
                        "parser-source hash); empty string disables")
    p.add_argument("--catalog", type=str, default="maps/scoresaber_catalog.json")
    p.add_argument("--refresh-catalog", action="store_true")
    p.add_argument("--no-graphs", action="store_true",
                   help="disable CUDA-graph capture of the argmax forward")
    p.add_argument("--fused-forward", action="store_true",
                   help="use the fused single-kernel CUDA forward "
                        "(policy_cuda, JIT-built) for the argmax rollout; "
                        "falls back to the graphed/eager path if the kernel "
                        "is unavailable")
    return p.parse_args()


def model_hash(state_dict) -> str:
    """sha256 over the weights themselves (sorted names + shapes + data):
    stable across file re-saves, device moves, and torch versions."""
    h = hashlib.sha256()
    for name in sorted(state_dict):
        t = state_dict[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return "sha256:" + h.hexdigest()


def default_out_path(ckpt_path):
    run = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    return os.path.join("bench", f"{run}_{stem}.json")


def load_results(path, mhash):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if data.get("model_hash") != mhash:
        sys.exit(f"error: {path} holds results for a DIFFERENT model "
                 f"({data.get('model_hash')} != {mhash}).\n"
                 f"Same weights merge; different weights need their own "
                 f"file — pass --out.")
    return data


def save_results(path, data, levels):
    evaluated = [e for e in levels.values() if "skipped" not in e]
    data["levels"] = sorted(levels.values(), key=lambda e: e["stars"])
    data["n_evaluated"] = len(evaluated)
    data["n_skipped"] = len(levels) - len(evaluated)
    data["total_weighted_pp"] = round(
        scoresaber.weighted_total_pp([e["pp"] for e in evaluated]), 2)
    data["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def parser_version():
    """Chart-cache version: hash of the parser + scoring sources. Any
    edit to them invalidates every cached chart automatically (the
    parser has changed repeatedly — chains, arcs, angle snap)."""
    from beat_arena_env import beatsaver, chart
    h = hashlib.sha256()
    for mod in (beatsaver, chart, scoresaber):
        with open(mod.__file__, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:10]


def dur_key(entry):
    return f"{entry['hash']}_{entry['difficulty']}"


def load_or_parse(entry, cache_dir, chart_cache, ver, speed_scale):
    """The cacheable unit: (note tuples, max multiplied score, duration,
    last hit time) for one level. duration/hit times come from the song
    clock, so they are speed_scale-independent; the tuples are not —
    speed_scale is part of the cache key."""
    cpath = None
    if chart_cache:
        cpath = os.path.join(
            chart_cache,
            f"{entry['hash']}_{entry['difficulty']}_{speed_scale:g}"
            f"_{ver}.pkl")
        if os.path.exists(cpath):
            try:
                with open(cpath, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass                    # corrupt/partial file: reparse
    map_dir = scoresaber.ensure_map_dats(entry["hash"], cache_dir)
    notes, meta = load_beatmap(map_dir, entry["difficulty"],
                               speed_scale=speed_scale, first_note_s=2.0)
    payload = {
        "tuples": [note_tuple(n) for n in notes],
        "max_score": scoresaber.max_multiplied_score(notes) if notes else 0,
        "duration_s": meta["duration_s"] if notes else 0.0,
        "last_hit_s": notes[-1].hit_time if notes else 0.0,
    }
    if cpath:
        tmp = cpath + f".tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=5)
        os.replace(tmp, cpath)
    return payload


def producer(tasks, cache_dir, chart_cache, speed_scale, max_song_s, q, stop,
             durations):
    """Download + parse maps ahead of the eval loop (network overlaps
    compute; parsing is served from the chart cache when warm). Emits
    (entry, tuples, payload, skip_reason) then a sentinel, and records
    each level's duration into the shared `durations` dict (the
    longest-first sidecar)."""
    ver = parser_version()
    if chart_cache:
        os.makedirs(chart_cache, exist_ok=True)
    try:
        for entry in tasks:
            if stop.is_set():
                break
            try:
                p = load_or_parse(entry, cache_dir, chart_cache, ver,
                                  speed_scale)
                durations[dur_key(entry)] = round(p["duration_s"], 1)
                if not p["tuples"]:
                    q.put((entry, None, None, "no notes parsed"))
                elif max_song_s > 0 and p["last_hit_s"] > max_song_s:
                    q.put((entry, None, None,
                           f"{p['last_hit_s']:.0f}s > --max-song-s"))
                else:
                    q.put((entry, p["tuples"], p, None))
            except Exception as e:      # v4 quirks, no Standard set, network
                q.put((entry, None, None, f"{type(e).__name__}: {e}"))
    finally:
        q.put(None)                     # sentinel, unconditionally


def finish_slot(slot, info, combo):
    """Level-result entry from a finished rollout; combo =
    (multiplied_score, max_combo, breaks) from whichever walk ran."""
    entry = slot["entry"]
    c_score, max_combo, breaks = combo
    score = round(c_score)
    acc = score / slot["max_score"] if slot["max_score"] else 0.0
    return {
        "leaderboard_id": entry["leaderboard_id"],
        "song_hash": entry["hash"], "song": entry["song"],
        "mapper": entry["mapper"], "difficulty": entry["difficulty"],
        "stars": entry["stars"], "max_pp": entry["max_pp"],
        "duration_s": round(slot["duration_s"], 1),
        "n_notes": info["n_notes"],
        "score": score, "max_score": slot["max_score"],
        "api_max_score": entry["api_max_score"],
        "accuracy": round(acc, 5),
        "pp": round(scoresaber.pp_value(acc, entry["max_pp"]), 2),
        "good_cuts": info["good_cuts"], "bad_cuts": info["bad_cuts"],
        "misses": info["misses"], "bombs_hit": info["bombs_hit"],
        "highest_combo": max_combo,
        "full_combo": breaks == 0 and info["good_cuts"] == info["n_notes"],
        # unmultiplied score decomposition (2026-07-25): per-band
        # component analyses come free from the results file
        "score_pre": info["score_pre"], "score_acc": info["score_acc"],
        "score_post": info["score_post"],
    }


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):   # song titles are arbitrary
        sys.stdout.reconfigure(errors="replace")  # unicode; never crash a run
    args.ckpt = os.path.abspath(args.ckpt)
    if args.out:
        args.out = os.path.abspath(args.out)
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)                       # maps/ + bench/ live next to us

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    mhash = model_hash(ckpt["state_dict"])
    policy = load_policy(ckpt)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy = policy.to(device).eval()

    out_path = args.out or default_out_path(args.ckpt)
    ckpt_name = "/".join(os.path.normpath(os.path.abspath(args.ckpt))
                         .replace("\\", "/").split("/")[-2:])
    data = load_results(out_path, mhash) or {
        "checkpoint": args.ckpt, "checkpoint_name": ckpt_name,
        "model_hash": mhash, "arch": ckpt.get("arch"),
        "global_step": ckpt.get("global_step"),
        "speed_scale": args.speed_scale,
    }
    levels = {e["leaderboard_id"]: e for e in data.get("levels", [])}

    if args.refresh_catalog or not os.path.exists(args.catalog):
        print("[catalog] fetching all ranked leaderboards from ScoreSaber "
              "(one-time, ~2 min)...", flush=True)
        catalog = scoresaber.fetch_ranked_catalog(log=print)
        os.makedirs(os.path.dirname(args.catalog) or ".", exist_ok=True)
        with open(args.catalog, "w") as f:
            json.dump(catalog, f)
        print(f"[catalog] {len(catalog)} ranked leaderboards cached at "
              f"{args.catalog}", flush=True)
    else:
        with open(args.catalog) as f:
            catalog = json.load(f)

    tasks = [e for e in catalog
             if (args.stars is None
                 or args.stars[0] <= e["stars"] <= args.stars[1])]
    in_range = len(tasks)
    if not args.force:                   # merge policy: evaluated = skip,
        tasks = [e for e in tasks        # previously-skipped = retry
                 if e["leaderboard_id"] not in levels
                 or "skipped" in levels[e["leaderboard_id"]]]
    if args.limit > 0:
        tasks = tasks[:args.limit]
    # longest-first (durations sidecar, filled in as maps get parsed):
    # long maps started last would hold near-empty slot vectors at the
    # tail. Unknown durations sort FIRST — they could be long.
    dur_path = os.path.join(args.chart_cache or "maps", "durations.json")
    durations = {}
    if os.path.exists(dur_path):
        with open(dur_path) as f:
            durations = json.load(f)
    tasks.sort(key=lambda e: durations.get(dur_key(e), float("inf")),
               reverse=True)
    print(f"=== ScoreSaber bench: {ckpt_name} ({mhash[:19]}...) on {device}\n"
          f"    {in_range} leaderboards in range, {len(tasks)} to play, "
          f"{len(levels)} already on file -> {out_path}", flush=True)
    if not tasks:
        save_results(out_path, data, levels)
        print(f"nothing to do; totals refreshed "
              f"(weighted PP {data['total_weighted_pp']})", flush=True)
        return

    q = queue.Queue(maxsize=max(64, 2 * args.num_envs))
    stop = threading.Event()
    threading.Thread(target=producer,
                     args=(tasks, args.map_cache, args.chart_cache,
                           args.speed_scale, args.max_song_s, q, stop,
                           durations),
                     daemon=True).start()

    # Vectorized rollout (2026-07-24, the SPEC "future unlock"): ONE
    # VecBeatSim steps every active map per FFI call and returns the
    # per-env ORDERED event stream (env_idx, note_idx, code, points) —
    # order feeds each slot's combo/multiplier walk exactly as the old
    # serial per-env loop did. Idle slots hold an empty chart (done on
    # every step, eventless, ~free) so the vector call is always valid.
    N = args.num_envs
    # Rust-side combo walk (2026-07-25, the ONLY walk since same day):
    # each env's ordered events feed its walk inside the step call
    sim = VecBeatSim(N, max_steps=100_000_000, combo_walk=True)
    for i in range(N):
        sim.set_chart(i, [])
    fwd = None
    if args.fused_forward and device.type == "cuda":
        try:                             # JIT build/arch may fail: fall back
            fwd = _FusedArgmax(policy, N, device)
            print("[fused-forward] single-launch CUDA forward active",
                  flush=True)
        except Exception as e:
            print(f"[fused-forward] unavailable, falling back: {e}",
                  flush=True)
    if fwd is None:
        fwd = (None if args.no_graphs or device.type != "cuda"
               else _GraphedArgmax(policy, N, device))
    obs_buf = np.zeros((N, OBS_DIM), dtype=np.float32)
    slots = [None] * N
    free = list(range(N))
    n_active, feed_done, n_done, n_new_skip = 0, False, 0, 0
    last_save = time.monotonic()
    try:
        while True:
            while not feed_done and free:
                try:
                    item = q.get(block=n_active == 0)
                except queue.Empty:
                    break
                if item is None:
                    feed_done = True
                    break
                entry, tuples, payload, skip = item
                if skip is not None:
                    n_done += 1
                    n_new_skip += 1
                    print(f"[{n_done}/{len(tasks)}] skip {entry['song']!r} "
                          f"[{entry['difficulty']}]: {skip}", flush=True)
                    levels[entry["leaderboard_id"]] = {
                        "leaderboard_id": entry["leaderboard_id"],
                        "song_hash": entry["hash"], "song": entry["song"],
                        "difficulty": entry["difficulty"],
                        "stars": entry["stars"], "skipped": skip}
                    continue
                i = free.pop()
                sim.set_chart(i, tuples)
                obs_buf[i] = sim.obs_one(i)
                slots[i] = {
                    "entry": entry,
                    "max_score": payload["max_score"],
                    "duration_s": payload["duration_s"]}
                n_active += 1
            if n_active == 0:
                if feed_done:
                    break
                continue

            if fwd is not None:
                try:
                    actions = fwd(obs_buf).cpu().numpy()
                except Exception as e:       # capture failed: go eager
                    print(f"[cuda-graphs] falling back to eager: {e}",
                          flush=True)
                    fwd = None
            if fwd is None:
                batch = torch.as_tensor(obs_buf, device=device)
                with torch.no_grad():
                    logits, _ = policy(batch)
                # one stacked argmax kernel over the 12 heads, one D2H sync
                actions = torch.stack(logits, dim=1).argmax(-1).cpu().numpy()
            obs_flat, _, dones, _events = sim.step(
                np.ascontiguousarray(actions, dtype=np.int64))
            obs_buf = np.asarray(obs_flat).reshape(N, OBS_DIM)
            n_finished = 0
            for i in np.flatnonzero(np.asarray(dones)):
                i = int(i)
                s = slots[i]
                if s is None:                    # idle empty-chart slot
                    continue
                combo = sim.combo(i)             # BEFORE set_chart resets it
                g, b, m, n_scoreable, _, bombs = sim.counters(i)
                c_pre, c_acc, c_post = sim.components(i)
                result = finish_slot(s, {
                    "good_cuts": g, "bad_cuts": b, "misses": m,
                    "n_notes": n_scoreable, "bombs_hit": bombs,
                    "score_pre": c_pre, "score_acc": c_acc,
                    "score_post": c_post}, combo)
                levels[result["leaderboard_id"]] = result
                n_done += 1
                n_finished += 1
                print(f"[{n_done}/{len(tasks)}] {result['stars']:.2f}* "
                      f"{result['song']!r} [{result['difficulty']}] "
                      f"acc {100 * result['accuracy']:.2f}% "
                      f"pp {result['pp']:.1f}"
                      f"{'  FC' if result['full_combo'] else ''}",
                      flush=True)
                slots[i] = None
                sim.set_chart(i, [])
                free.append(i)
                n_active -= 1
            if n_finished and time.monotonic() - last_save > 30.0:
                save_results(out_path, data, levels)
                last_save = time.monotonic()
    finally:
        stop.set()
        save_results(out_path, data, levels)
        if args.chart_cache:            # longest-first sidecar for next run
            os.makedirs(args.chart_cache, exist_ok=True)
            tmp = dur_path + f".tmp{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(durations, f)
            os.replace(tmp, dur_path)

    print(f"\ndone: {n_done} played ({n_new_skip} skipped), "
          f"{data['n_evaluated']} levels on file | "
          f"total weighted PP {data['total_weighted_pp']} -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
