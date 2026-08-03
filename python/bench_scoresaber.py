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
(hash, difficulty, parser-source-hash) as pickles under
--chart-cache; any edit to the parser/scoring sources invalidates the
cache automatically via the source hash. Also from the same session:
tasks run LONGEST-FIRST (durations sidecar, populated as maps are
parsed) so long maps can't straggle at the tail with mostly-idle
slots, and results are saved on a wall-clock timer instead of every 25
finishes (the every-25 rewrite was quadratic in catalog size).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
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
from beat_arena_env.constants import N_JOINTS, OBS_DIM
from beat_arena_rs import VecBeatSim
from cuda_graphs import _Graphed
from policy import load_policy

# Scoring-rule version stamped into every results file. 1 = swept-surface
# accuracy distance (v7.0-2026-07-29); 2 = the real game's infinite-
# cut-plane accuracy (user-caught fix, 2026-07-29); 3 = REAL note speeds
# (speed_scale removed 2026-07-30 — v1/v2 played maps at NJS*0.3, a
# different physics regime). Results never merge across versions.
SCORING_VERSION = 3


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
            logits, _ = self.policy.forward_stacked(self.s_obs)
        return logits.argmax(-1)

    def __call__(self, obs_t):
        # obs_t is the loop's persistent (pinned) host buffer, so this
        # H2D is an async DMA; the .cpu() on the result syncs before the
        # sim overwrites the buffer
        self._copy_static(self.s_obs, obs_t, non_blocking=True)
        return self._run()


class _FusedArgmax:
    """Bench forward via the fused single-launch CUDA kernel
    (policy_cuda.FusedNotePolicy): logits + stacked argmax in ONE launch,
    no graph capture needed. Same call contract as _GraphedArgmax."""

    def __init__(self, policy, n, device):
        from policy_cuda import FusedNotePolicy
        self.fused = FusedNotePolicy(policy, device)
        self.s_obs = torch.zeros(n, OBS_DIM, device=device)

    def __call__(self, obs_t):
        self.s_obs.copy_(obs_t, non_blocking=True)
        return self.fused.argmax(self.s_obs)


class _Group:
    """One pipeline group: its own sim, forward, pinned obs/act buffers,
    slot bookkeeping, and a CUDA event fencing its in-flight GPU chain
    (H2D + kernel + async D2H). `issued` means actions are in flight; the
    event sync is what makes act_np safe to read AND obs_buf safe for
    step_into to overwrite (the H2D precedes the D2H in stream order)."""

    __slots__ = ("n", "sim", "fwd", "obs_t", "obs_buf", "act_t", "act_np",
                 "slots", "free", "n_active", "ev", "issued")

    def __init__(self, n, fwd, device, gears=3):
        self.n = n
        self.sim = VecBeatSim(n, max_steps=100_000_000, combo_walk=True,
                              return_events=False, gears=gears)
        for i in range(n):
            self.sim.set_chart(i, [])
        self.fwd = fwd
        pin = device.type == "cuda"
        self.obs_t = torch.empty((n, OBS_DIM), dtype=torch.float32,
                                 pin_memory=pin)
        self.obs_buf = self.obs_t.numpy()
        self.act_t = torch.empty((n, N_JOINTS), dtype=torch.int64,
                                 pin_memory=pin)
        self.act_np = self.act_t.numpy()
        self.slots = [None] * n
        self.free = list(range(n))
        self.n_active = 0
        self.ev = torch.cuda.Event() if pin else None
        self.issued = False


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
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-level result lines (6.1k prints cost "
                        "real time on a Windows console; skips and the "
                        "summary still print)")
    p.add_argument("--timing", action="store_true",
                   help="print a per-phase wall-clock decomposition of the "
                        "eval loop (feed/issue/sync/step/finish) at the end")
    p.add_argument("--pipeline", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="two-group software pipeline: while one half's "
                        "forward+D2H is in flight on the GPU, the CPU runs "
                        "the other half's Rust sim (2026-07-27). Engages "
                        "only on the fused path — its per-sample kernel is "
                        "batch-size-invariant (verified bitwise), so two "
                        "groups of N/2 score identically to one batch of N")
    p.add_argument("--num-envs", type=int, default=2048,
                   help="maps evaluated in parallel (policy is batched); "
                        "on the fused path the effective width is capped "
                        "at ~tasks/3 so small banded runs don't pay for "
                        "idle rows, and the pipeline splits it into two "
                        "groups (2048 -> 2x1024, the measured kernel "
                        "sweet spot)")
    p.add_argument("--device", type=str, default=None)
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
    p.add_argument("--fused-forward", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="use the fused single-kernel CUDA forward "
                        "(policy_cuda, JIT-built) for the argmax rollout; "
                        "falls back to the graphed/eager path if the kernel "
                        "is unavailable (2026-07-27: default ON — it is both "
                        "the fast path and the batch-size-invariant one)")
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
    # unweighted sum (2026-07-26, user request): wpp saturates once the
    # top-300 dominate the 0.965^rank series; the raw sum keeps
    # measuring breadth across the whole evaluated set
    data["total_unweighted_pp"] = round(
        sum(e["pp"] for e in evaluated), 2)
    data["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def parser_version():
    """Chart-cache version: hash of the code the cached payload actually
    depends on. beatsaver.py is hashed whole (it IS the parser); from
    chart.py and scoresaber.py only the payload-relevant symbols are
    hashed — hashing those modules whole (pre-2026-07-25) meant every
    procedural-GENERATOR edit silently invalidated the entire cache,
    turning the next bench into a full cold re-parse."""
    from beat_arena_env import beatsaver, chart
    h = hashlib.sha256()
    with open(beatsaver.__file__, "rb") as f:
        h.update(f.read())
    for sym in (chart.Note, chart.note_tuple, chart.chain_links,
                scoresaber.ComboTracker, scoresaber.max_multiplied_score):
        h.update(inspect.getsource(sym).encode())
    h.update(repr(sorted(scoresaber.MAX_UNIT_POINTS.items())).encode())
    return h.hexdigest()[:10]


def dur_key(entry):
    return f"{entry['hash']}_{entry['difficulty']}"


def load_or_parse(entry, cache_dir, chart_cache, ver):
    """The cacheable unit: (note tuples, max multiplied score, duration,
    last hit time) for one level, at the map's real NJS."""
    cpath = None
    if chart_cache:
        cpath = os.path.join(
            chart_cache,
            f"{entry['hash']}_{entry['difficulty']}_{ver}.pkl")
        if os.path.exists(cpath):
            try:
                with open(cpath, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass                    # corrupt/partial file: reparse
    map_dir = scoresaber.ensure_map_dats(entry["hash"], cache_dir)
    notes, meta = load_beatmap(map_dir, entry["difficulty"],
                               first_note_s=2.0)
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


def producer(tasks, cache_dir, chart_cache, max_song_s, q, stop,
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
                p = load_or_parse(entry, cache_dir, chart_cache, ver)
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
              f"(weighted PP {data['total_weighted_pp']}, unweighted "
              f"{data['total_unweighted_pp']})", flush=True)
        return

    # small lookahead (2026-07-27): the warm producer is ~70x faster than
    # consumer demand, so 2*N slots just parked ~290 MB of live note
    # tuples (~1.1M containers) for every gen2 GC pass to walk
    q = queue.Queue(maxsize=min(2 * args.num_envs, 128))
    stop = threading.Event()
    threading.Thread(target=producer,
                     args=(tasks, args.map_cache, args.chart_cache,
                           args.max_song_s, q, stop, durations),
                     daemon=True).start()

    # Vectorized rollout (2026-07-24, the SPEC "future unlock"): ONE
    # VecBeatSim steps every active map per FFI call and returns the
    # per-env ORDERED event stream (env_idx, note_idx, code, points) —
    # order feeds each slot's combo/multiplier walk exactly as the old
    # serial per-env loop did. Idle slots hold an empty chart (done on
    # every step, eventless, ~free) so the vector call is always valid.
    N = args.num_envs
    if args.fused_forward:
        # adaptive width (2026-07-27): wide N amortizes the per-step sync
        # drain (full catalog 81.6s@512 -> 74.1s@1024) and is score-free on
        # the fused path — the kernel is per-sample with no cross-sample
        # reduction, VERIFIED bitwise N=512 vs N=1024 over all 6101 levels.
        # Banded runs cap at ~tasks/3 so slots still refill a few times.
        # (Torch paths keep the fixed width: cuBLAS retiles with N, so an
        # adaptive width could flip near-ties between resumes there.)
        N = max(1, min(N, -(-len(tasks) // 3)))
    # Two-group pipeline (2026-07-27): with a single group the loop is a
    # strict serial chain — GPU idles during the Rust sim, CPU idles
    # during the forward. Two half-size groups alternate: while group A's
    # H2D+kernel+async-D2H chain runs on the (single) stream, the CPU
    # syncs B's event and runs B's sim + bookkeeping. Fused path only:
    # the kernel's per-sample batch invariance (verified bitwise 512 vs
    # 1024) is what makes N/2 groups score-identical; cuBLAS is not.
    use_fused = False
    if args.fused_forward and device.type == "cuda":
        try:                             # JIT build/arch may fail: fall back
            _FusedArgmax(policy, 1, device)   # availability probe
            use_fused = True
            print("[fused-forward] single-launch CUDA forward active",
                  flush=True)
        except Exception as e:
            print(f"[fused-forward] unavailable, falling back: {e}",
                  flush=True)
    pipelined = args.pipeline and use_fused and N >= 2
    if pipelined:
        n_a = (N + 1) // 2
        sizes = [n_a, N - n_a]
        print(f"[pipeline] two groups of {sizes[0]}/{sizes[1]} envs",
              flush=True)
    else:
        sizes = [N]
    groups = []
    for n_g in sizes:
        if use_fused:
            fwd_g = _FusedArgmax(policy, n_g, device)
        else:
            fwd_g = (None if args.no_graphs or device.type != "cuda"
                     else _GraphedArgmax(policy, n_g, device))
        groups.append(_Group(n_g, fwd_g, device, policy.n_gears))
    fwd = groups[0].fwd                  # for the provenance stamp below
    # Forward-path provenance (2026-07-27): fused vs torch forwards differ
    # by ~4e-7 in the logits, enough to flip near-tie argmaxes — per-level
    # results are only comparable within one path. Same weights + different
    # path must therefore NOT silently merge into one file.
    fwd_path = ("fused" if isinstance(fwd, _FusedArgmax) else
                "graphed" if isinstance(fwd, _GraphedArgmax) else "eager")
    prev_path = data.get("forward")
    if prev_path is not None and prev_path != fwd_path:
        sys.exit(f"error: {out_path} holds {prev_path!r}-forward results but "
                 f"this run would use {fwd_path!r}. Matching flag "
                 f"(--fused-forward/--no-fused-forward/--no-graphs) or a "
                 f"fresh --out required; --force with the matching flag "
                 f"re-evaluates everything on one path.")
    full_force = args.force and args.stars is None and args.limit == 0
    if (prev_path is None and not full_force
            and any("skipped" not in e for e in levels.values())):
        print(f"[forward] warning: {out_path} predates forward-path "
              f"stamping; its levels' provenance is unverifiable and the "
              f"file stays unstamped (full-range --force re-bench adopts "
              f"{fwd_path!r})", flush=True)
    else:
        data["forward"] = fwd_path
    # Scoring-rule provenance (2026-07-29): the cut-plane accuracy fix
    # changed every level's score, so old-rule (swept-surface acc, v1)
    # results must NOT merge with new-rule (infinite-cut-plane, v2) ones.
    # Files that predate the stamp but hold results are v1 by definition.
    prev_rules = data.get("scoring_version")
    if prev_rules is None and any("skipped" not in e
                                  for e in levels.values()):
        prev_rules = 1
    if (prev_rules is not None and prev_rules != SCORING_VERSION
            and not full_force):
        sys.exit(f"error: {out_path} holds scoring-version {prev_rules} "
                 f"results but the sim now scores version "
                 f"{SCORING_VERSION} (v2: 2026-07-29 cut-plane accuracy "
                 f"fix; v3: 2026-07-30 real note speeds) — results "
                 f"cannot merge across scoring rules. Use a fresh --out, "
                 f"or full-range --force to re-evaluate everything under "
                 f"the current rules.")
    data["scoring_version"] = SCORING_VERSION
    # (persistent pinned obs/act buffers live in each _Group: step_into
    # writes obs rows in place, the async D2H lands in pinned act memory,
    # and the group's event is the sync that makes both safe to touch)
    feed_done, n_done, n_new_skip = False, 0, 0
    last_save = time.monotonic()
    # GC off for the loop (2026-07-27): the hot path allocates no cycles
    # that matter, and gen2 passes were walking the whole startup graph +
    # the queued note tuples. Bounded run; freeze then disable.
    gc.collect()
    gc.freeze()
    gc.disable()
    # per-phase wall clock (--timing): feed/refill, issue (forward launch
    # + async D2H enqueue), sync (event wait — under the pipeline this is
    # GPU time NOT hidden by the other group's CPU work), step_into,
    # finish/save. perf_counter is ~100 ns — always accumulated.
    ph = [0.0, 0.0, 0.0, 0.0, 0.0]
    n_steps = 0
    pc = time.perf_counter
    try:
        while True:
            progressed = False
            for gr in groups:
                t0 = pc()
                if gr.issued:
                    if gr.ev is not None:
                        gr.ev.synchronize()  # actions landed; obs H2D done
                    gr.issued = False
                    t1 = pc()
                    _, dones, _events = gr.sim.step_into(gr.act_np,
                                                         gr.obs_buf)
                    t2 = pc()
                    n_finished = 0
                    for i in np.flatnonzero(np.asarray(dones)):
                        i = int(i)
                        s = gr.slots[i]
                        if s is None:            # idle empty-chart slot
                            continue
                        combo = gr.sim.combo(i)  # BEFORE set_chart resets
                        g, b, m, n_scoreable, _, bombs = gr.sim.counters(i)
                        c_pre, c_acc, c_post = gr.sim.components(i)
                        result = finish_slot(s, {
                            "good_cuts": g, "bad_cuts": b, "misses": m,
                            "n_notes": n_scoreable, "bombs_hit": bombs,
                            "score_pre": c_pre, "score_acc": c_acc,
                            "score_post": c_post}, combo)
                        levels[result["leaderboard_id"]] = result
                        n_done += 1
                        n_finished += 1
                        if not args.quiet:
                            print(f"[{n_done}/{len(tasks)}] "
                                  f"{result['stars']:.2f}* "
                                  f"{result['song']!r} "
                                  f"[{result['difficulty']}] "
                                  f"acc {100 * result['accuracy']:.2f}% "
                                  f"pp {result['pp']:.1f}"
                                  f"{'  FC' if result['full_combo'] else ''}",
                                  flush=True)
                        gr.slots[i] = None
                        gr.sim.set_chart(i, [])
                        gr.free.append(i)
                        gr.n_active -= 1
                    if n_finished and time.monotonic() - last_save > 30.0:
                        # accumulate §22 pd-floor engagements at every
                        # periodic save (cheap FFI counter pop) so a
                        # killed run cannot lose the count across a
                        # resume
                        data["pd_floor_hits"] = (
                            data.get("pd_floor_hits", 0)
                            + sum(g.sim.pop_pd_hits() for g in groups))
                        save_results(out_path, data, levels)
                        last_save = time.monotonic()
                    t3 = pc()
                    ph[2] += t1 - t0
                    ph[3] += t2 - t1
                    ph[4] += t3 - t2
                    n_steps += 1
                t4 = pc()
                while not feed_done and gr.free:
                    # single consumer: qsize>0 guarantees a non-blocking
                    # get. Block only when nothing is active or in flight
                    # ANYWHERE — a blocking wait with the other group's
                    # forward in flight would stall the pipeline.
                    busy = any(x.n_active or x.issued for x in groups)
                    if busy and not q.qsize():
                        break           # nothing parsed yet; keep stepping
                    item = q.get()
                    if item is None:
                        feed_done = True
                        break
                    entry, tuples, payload, skip = item
                    if skip is not None:
                        n_done += 1
                        n_new_skip += 1
                        print(f"[{n_done}/{len(tasks)}] skip "
                              f"{entry['song']!r} "
                              f"[{entry['difficulty']}]: {skip}", flush=True)
                        levels[entry["leaderboard_id"]] = {
                            "leaderboard_id": entry["leaderboard_id"],
                            "song_hash": entry["hash"], "song": entry["song"],
                            "difficulty": entry["difficulty"],
                            "stars": entry["stars"], "skipped": skip}
                        continue
                    i = gr.free.pop()
                    gr.sim.set_chart(i, tuples)
                    gr.sim.obs_one_into(i, gr.obs_buf)  # in-place row write
                    gr.slots[i] = {
                        "entry": entry,
                        "max_score": payload["max_score"],
                        "duration_s": payload["duration_s"]}
                    gr.n_active += 1
                t5 = pc()
                ph[0] += t5 - t4
                if gr.n_active > 0:
                    act_dev = None
                    if gr.fwd is not None:
                        try:
                            act_dev = gr.fwd(gr.obs_t)
                        except Exception as e:   # capture failed: go eager
                            print(f"[cuda-graphs] falling back to eager: "
                                  f"{e}", flush=True)
                            gr.fwd = None
                            if data.get("forward") is not None:
                                data["forward"] = "eager"
                    if gr.fwd is None:
                        batch = gr.obs_t.to(device, non_blocking=True)
                        with torch.no_grad():
                            logits, _ = policy.forward_stacked(batch)
                        act_dev = logits.argmax(-1)
                    # async D2H into pinned memory; the event fences it
                    gr.act_t.copy_(act_dev, non_blocking=True)
                    if gr.ev is not None:
                        gr.ev.record()
                        # WDDM batches user-mode submissions: without a
                        # flush the chain sits unsubmitted until the next
                        # blocking sync and NOTHING overlaps (measured:
                        # sync == full chain time). query() flushes.
                        gr.ev.query()
                    gr.issued = True
                    progressed = True
                    ph[1] += pc() - t5
            if not progressed and not any(x.issued for x in groups):
                if feed_done:
                    break
                # nothing active or in flight anywhere: the next feed
                # pass blocks on q.get() until the producer delivers
    finally:
        stop.set()
        # f64 Cholesky pivot-floor engagements (SPEC-v10 Divergences
        # §22) across all sims — reported, not hidden (the §20e house
        # rule): a nonzero count marks contorted gimbal-adjacent
        # states the pre-§22 engine PANICKED on. Final pop of the
        # tail since the last periodic save.
        data["pd_floor_hits"] = (data.get("pd_floor_hits", 0)
                                 + sum(gr.sim.pop_pd_hits() for gr in groups))
        pd_hits = data["pd_floor_hits"]
        save_results(out_path, data, levels)
        if args.chart_cache:            # longest-first sidecar for next run
            os.makedirs(args.chart_cache, exist_ok=True)
            tmp = dur_path + f".tmp{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(durations, f)
            os.replace(tmp, dur_path)

    if args.timing:
        tot = sum(ph)
        ms = 1000.0 / max(n_steps, 1)
        mode = "x".join(str(x.n) for x in groups)
        print(f"\n[timing] {n_steps} group-steps @ {mode} | "
              f"loop {tot:.1f}s ({tot * ms:.2f} ms/group-step) | "
              f"feed {ph[0]:.1f}s ({ph[0] * ms:.2f}) | "
              f"issue {ph[1]:.1f}s ({ph[1] * ms:.2f}) | "
              f"sync {ph[2]:.1f}s ({ph[2] * ms:.2f}) | "
              f"step_into {ph[3]:.1f}s ({ph[3] * ms:.2f}) | "
              f"finish/save {ph[4]:.1f}s ({ph[4] * ms:.2f})", flush=True)
    print(f"\ndone: {n_done} played ({n_new_skip} skipped), "
          f"{data['n_evaluated']} levels on file | "
          f"total weighted PP {data['total_weighted_pp']} | "
          f"unweighted PP {data['total_unweighted_pp']} | "
          f"pd-floor hits {pd_hits} (file total) "
          f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
