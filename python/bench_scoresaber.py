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

BeatLeader (--catalog, 2026-09-30): `--catalog scoresaber` (default,
unchanged files and behavior), `beatleader` (every playable BL-ranked
level, maps/beatleader_catalog.json; per-level BL stars + ratings and
BL pp — beat_arena_env/beatleader.py) or `both` (the UNION keyed by
(song hash, difficulty, characteristic): each level plays ONCE and its
record carries `ss` and `bl` sub-results, null where unranked on that
leaderboard). The agent's score and accuracy are one measurement; each
leaderboard's pp is computed with its own rules from it. Files carry
`catalog`, default out paths get a _beatleader / _both suffix, and
results never merge across catalogs.
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

from beat_arena_env import beatleader, scoresaber
from beat_arena_env.beatsaver import load_beatmap
from beat_arena_env.chart import note_tuple, wall_tuple
from beat_arena_env.constants import (N_JOINTS, OBS_DIM, TENDON_B_PEAK_BEND,
                                     TENDON_B_PEAK_DEV, TENDON_TAU_PEAK_BEND,
                                     TENDON_TAU_PEAK_DEV, parse_tau_scale)
from beat_arena_env.mirror import mirror_tuples, mirror_walls


def _cgroup_cpu_quota():
    """Whole CPUs the container may use, or None: cgroup v2 cpu.max
    (RunPod) or cgroup v1 cpu.cfs_quota_us / cfs_period_us (VastAI)."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota == "max":
            return None
        return max(1, int(round(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota <= 0:
            return None
        return max(1, int(round(quota / period)))
    except (OSError, ValueError):
        return None


# rayon sizes its pool to the VISIBLE cores (64 on a RunPod host) even when
# the container's cgroup quota is 13.6: threads beyond the quota just get
# throttled. Measured 2026-09-16 (1,500 levels, 4,096x2): 14 threads cut
# step_into 6% vs auto. Must be set before the Rust module spins its pool.
if "RAYON_NUM_THREADS" not in os.environ:
    _q = _cgroup_cpu_quota()
    if _q is not None and _q < (os.cpu_count() or _q):
        os.environ["RAYON_NUM_THREADS"] = str(_q)

import beat_arena_rs
try:  # f32 build of the same sources (../rust32), the default bench engine
    import beat_arena_rs32
except ImportError:
    beat_arena_rs32 = None
from cuda_graphs import _Graphed, copy_static
from policy import load_policy

# Scoring-rule version stamped into every results file. 1 = swept-surface
# accuracy distance (v7.0-2026-07-29); 2 = the real game's infinite-
# cut-plane accuracy (user-caught fix, 2026-07-29); 3 = REAL note speeds
# (speed_scale removed 2026-07-30 — v1/v2 played maps at NJS*0.3, a
# different physics regime). Results never merge across versions.
SCORING_VERSION = 4

# Termination flags of a finished level (review B09, 2026-09-30) — the
# bits of lib.rs VecBeatSim::term and of the CUDA bench_rec col 18
# (cuda_env.TERM_*): 0 = the level itself ended (notes drained / tail).
# TILT is the chest-angle fall envelope, priced exactly like a FALL, so
# the JSON's `fell` covers both (`tilted` says which); ENVELOPE is the
# §23 root-escape C backstop; TICKCAP the per-episode max_steps limit.
TERM_FELL, TERM_TILT, TERM_ENVELOPE, TERM_TICKCAP = 1, 2, 4, 8


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


class _Group:
    """One pipeline group: its own sim, forward, pinned obs/act buffers,
    slot bookkeeping, and a CUDA event fencing its in-flight GPU chain
    (H2D + kernel + async D2H). `issued` means actions are in flight; the
    event sync is what makes act_np safe to read AND obs_buf safe for
    step_into to overwrite (the H2D precedes the D2H in stream order)."""

    __slots__ = ("n", "sim", "fwd", "obs_t", "obs_buf", "act_t", "act_np",
                 "slots", "free", "n_active", "ev", "issued")

    def __init__(self, n, fwd, device, gears=3,
                 tau_scale=None, sim_mod=beat_arena_rs, tick_hz=30):
        self.n = n
        # exp/tick-rate: tick_hz is the sim's control rate (240/tick_hz
        # physics substeps per tick, per-tick rules rescaled to real time)
        self.sim = sim_mod.VecBeatSim(n, max_steps=100_000_000, combo_walk=True,
                              return_events=False, gears=gears,
                              tick_hz=tick_hz)
        # env v46: fixed body for the whole bench (all 1.0 = the v45 body)
        if tau_scale is not None:
            self.sim.set_tau_scale([float(x) for x in tau_scale])
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


def parse_args(argv=None):
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
    # (--pipeline was retired 2026-08-13 with the fused per-sample
    # forward: the two-group split needed its batch invariance; cuBLAS
    # forwards are width-sensitive, so the bench runs one group now)
    p.add_argument("--tau-scale", type=str, default="default",
                   help="env v46: fixed per-axis torque-limit scales for "
                        "the whole bench — 16 comma-separated floats "
                        "(axis order: spine pitch/yaw/roll, arm pitch/"
                        "yaw/twist/elbow/wrist roll/bend/dev, hip pitch/"
                        "yaw/roll, knee, ankle pitch/roll), group "
                        "shorthand spine=1.25,arms=1,wrists=0.8,legs=1,"
                        "knees=1.1 (unspecified = 1.0), 'default' (the "
                        "saved body TAU_BODY_DEFAULT — the default since "
                        "2026-09-22) or 'nominal' (all 1.0, the TAU_MAX "
                        "table = every bench before 2026-09-20). The "
                        "scales are stamped in the bench JSON")
    p.add_argument("--foot-contact", action="store_true", default=True,
                   help="DEPRECATED NO-OP: contact mode is the only "
                        "physics (the v10 weld harness was removed from "
                        "the engines 2026-09-27). The flag is accepted so "
                        "older drivers keep running. (Weld-era "
                        "checkpoints predate env v11 and are archival.)")
    p.add_argument("--groups", type=int, default=2,
                   help="--sim rust: split --num-envs into this many equal "
                        "groups that alternate — each group's GPU forward "
                        "runs while the CPU steps the next group's sim "
                        "(2026-09-16). Forward batch width = num_envs/groups, "
                        "which is score-relevant: results never merge "
                        "across a different width")
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
    p.add_argument("--catalog", type=str, default="scoresaber",
                   help="which ranked levels to bench: scoresaber (default; "
                        "every ScoreSaber-ranked level, the historical "
                        "file format), beatleader (every playable "
                        "BeatLeader-ranked level, BL stars + BL pp), or "
                        "both (the union keyed by song hash + difficulty "
                        "+ characteristic, each level played ONCE, with "
                        "ss and bl sub-results per level and both "
                        "leaderboards' totals). Stamped in the JSON; "
                        "results never merge across catalogs. (A .json "
                        "path here is the pre-2026-09-30 meaning: "
                        "--catalog scoresaber --ss-catalog-path PATH)")
    p.add_argument("--ss-catalog-path", type=str,
                   default="maps/scoresaber_catalog.json")
    p.add_argument("--bl-catalog-path", type=str,
                   default="maps/beatleader_catalog.json")
    p.add_argument("--refresh-catalog", action="store_true",
                   help="refetch the catalog(s) --catalog uses")
    p.add_argument("--mirror", action="store_true",
                   help="bench on left-right mirrored charts (handedness-"
                        "bias diagnostic; scores are NOT ScoreSaber-"
                        "comparable — compare only against another "
                        "--mirror or non-mirror run of the same catalog)")
    p.add_argument("--no-graphs", action="store_true",
                   help="disable CUDA-graph capture of the argmax forward")
    p.add_argument("--cuda-slots", type=int, default=4096,
                   help="--sim cuda: parallel slots (each plays a queue "
                        "of levels, longest-first; 0 = one slot per "
                        "level). 4096 measured fastest 2026-09-16")
    p.add_argument("--cuda-pop-every", type=int, default=30,
                   help="--sim cuda: ticks between host pops of finished "
                        "levels (one GPU sync each); must stay < 60")
    p.add_argument("--sim", choices=["rust", "cuda"], default="rust",
                   help="rust (default, the reference): the threaded Rust "
                        "CPU sim at --num-envs with the policy batched on "
                        "the GPU. cuda (2026-09-16): every level in ONE "
                        "wave on CudaBeatVecEnv with the combo walk in the "
                        "kernel; same JSON schema, same forward kernels "
                        "(2048-row chunks, graphed); results carry "
                        "sim='cuda' and never merge into a rust file")
    p.add_argument("--tick-hz", type=int, default=30,
                   help="exp/tick-rate: the Rust sim's control rate (30, 60, "
                        "120, 240). The physics substep stays 1/240 s (240/"
                        "tick_hz substeps per tick, the action held for the "
                        "tick) and every per-tick rule/obs lane is rescaled "
                        "to its 30 Hz real-time meaning. 30 = the training "
                        "rate, bit-identical to the fixed-rate sim. Stamped "
                        "in the JSON; results never merge across rates. "
                        "--sim rust only")
    p.add_argument("--action-repeat", type=int, default=1,
                   help="exp/tick-rate: the policy decides every N-th sim "
                        "tick and its action is held for N ticks (obs from "
                        "the decision tick). 1 = decide every tick. Stamped "
                        "in the JSON; --sim rust only")
    p.add_argument("--sim-dtype", choices=["f32", "f64"], default="f32",
                   help="sim precision, both engines. f32 (default, "
                        "2026-09-16) = the trainer's regime: --sim rust "
                        "loads beat_arena_rs32 (the ../rust32 build of the "
                        "same sources), --sim cuda the f32 kernel. f64 = "
                        "the exact-parity regime the Rust tests pin "
                        "(slower). Results never merge across dtypes")
    args = p.parse_args(argv)
    if args.catalog.endswith(".json"):        # the old path meaning
        args.ss_catalog_path, args.catalog = args.catalog, "scoresaber"
    if args.catalog not in CATALOGS:
        p.error(f"--catalog must be one of {', '.join(CATALOGS)} (or a "
                f".json ScoreSaber catalog path), got {args.catalog!r}")
    return args


CATALOGS = ("scoresaber", "beatleader", "both")


# ---------------------------------------------------------------------------
# Catalogs: ScoreSaber, BeatLeader, and their union (--catalog)
# ---------------------------------------------------------------------------

def task_key(entry):
    """The results-file identity of a level: a `both` entry/record has a
    level_key (hash_difficulty_characteristic); scoresaber and
    beatleader ones are keyed by their own leaderboard id (SS ints, BL
    strings — never mixed in one file, the catalog stamp sees to it)."""
    return entry.get("level_key", entry.get("leaderboard_id"))


def merge_catalogs(ss_catalog, bl_catalog):
    """The `both` task list: the union of both catalogs keyed by
    beatleader.entry_key (song hash + difficulty + characteristic; SS
    entries are SoloStandard = "Standard"). One entry per level, with
    the leaderboard-specific catalog entries under `ss` / `bl` (None
    where that leaderboard does not rank it). Ordered by SS stars, then
    BL stars, then key (deterministic)."""
    by_key = {}
    for lb, cat in (("ss", ss_catalog), ("bl", bl_catalog)):
        for e in cat:
            k = beatleader.entry_key(e)
            m = by_key.get(k)
            if m is None:
                m = by_key[k] = {
                    "level_key": k, "hash": e["hash"].lower(),
                    "song": e["song"], "mapper": e["mapper"],
                    "difficulty": e["difficulty"],
                    "characteristic": e.get("characteristic", "Standard"),
                    "ss": None, "bl": None}
            if m[lb] is not None:
                raise ValueError(f"duplicate {lb} level {k}")
            m[lb] = e
    out = list(by_key.values())
    out.sort(key=lambda m: (m["ss"] is None,
                            m["ss"]["stars"] if m["ss"] else 0.0,
                            m["bl"]["stars"] if m["bl"] else 0.0,
                            m["level_key"]))
    return out


def catalog_counts(tasks_or_records):
    """Shared / SS-only / BL-only counts over `both` entries or records."""
    n_ss = sum(1 for e in tasks_or_records if e.get("ss") is not None)
    n_bl = sum(1 for e in tasks_or_records if e.get("bl") is not None)
    n_sh = sum(1 for e in tasks_or_records
               if e.get("ss") is not None and e.get("bl") is not None)
    return {"n_ss_ranked": n_ss, "n_bl_ranked": n_bl, "n_shared": n_sh,
            "n_ss_only": n_ss - n_sh, "n_bl_only": n_bl - n_sh}


def in_star_range(entry, stars, mode):
    """--stars filter: each leaderboard's OWN stars (BL stars are not SS
    stars). A `both` level is in range if either leaderboard's rating
    is."""
    if stars is None:
        return True
    lo, hi = stars
    if mode != "both":
        return lo <= entry["stars"] <= hi
    return any(sub is not None and lo <= sub["stars"] <= hi
               for sub in (entry["ss"], entry["bl"]))


def _record_stars(e):
    """Sort key of a level record: its stars, or (`both`) SS stars then
    BL stars."""
    if "stars" in e:
        return (e["stars"], 0.0)
    ss, bl = e.get("ss_stars"), e.get("bl_stars")
    return (ss if ss is not None else -1.0, bl if bl is not None else -1.0)


def load_catalog(path, which, refresh):
    """A cached catalog (fetched + cached on first use or --refresh)."""
    if refresh or not os.path.exists(path):
        name = {"scoresaber": "ScoreSaber", "beatleader": "BeatLeader"}[which]
        print(f"[catalog] fetching all ranked leaderboards from {name} "
              f"(one-time)...", flush=True)
        if which == "scoresaber":
            catalog = scoresaber.fetch_ranked_catalog(log=print)
        else:
            catalog, _excl = beatleader.fetch_ranked_catalog(log=print)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(catalog, f)
        print(f"[catalog] {len(catalog)} ranked leaderboards cached at "
              f"{path}", flush=True)
        return catalog
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_tasks(args):
    """(task list, header stamp) for args.catalog."""
    if args.catalog == "scoresaber":
        return load_catalog(args.ss_catalog_path, "scoresaber",
                            args.refresh_catalog), {}
    bl_cat = load_catalog(args.bl_catalog_path, "beatleader",
                          args.refresh_catalog)
    if args.catalog == "beatleader":
        return bl_cat, {}
    ss_cat = load_catalog(args.ss_catalog_path, "scoresaber",
                          args.refresh_catalog)
    merged = merge_catalogs(ss_cat, bl_cat)
    return merged, {"catalog_levels": {"n_levels": len(merged),
                                       **catalog_counts(merged)}}


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
    data["levels"] = sorted(levels.values(), key=_record_stars)
    data["n_evaluated"] = len(evaluated)
    data["n_skipped"] = len(levels) - len(evaluated)
    data.update(pp_totals(data.get("catalog", "scoresaber"), evaluated))
    # env v19: catalog-wide mean flail (rad per note-free tick) — the
    # phantom-swing judge for the post_rem obs A/B. .get() keeps old
    # results files (pre-v19 levels lack the fields) loadable.
    # exp/tick-rate: per 30 Hz tick of real time (flail_ticks are sim
    # ticks; × tick_hz/30 == × 1.0 at the default rate)
    _hz = float(data.get("tick_hz", 30))
    _ft = sum(e.get("flail_ticks", 0) for e in evaluated)
    data["flail_rad_per_tick"] = round(
        sum(e.get("flail_rad", 0.0) for e in evaluated) / _ft * (_hz / 30.0),
        5) if _ft else 0.0
    # wrist-bend reversals (2026-09-22): catalog-wide totals per hand and
    # combined, plus the rate per second of played time per hand
    _wt = sum(e.get("ticks", 0) for e in evaluated)
    _wl = sum(int(e.get("wrist_rev_l", 0)) for e in evaluated)
    _wr = sum(int(e.get("wrist_rev_r", 0)) for e in evaluated)
    data["wrist_rev_total_l"] = _wl
    data["wrist_rev_total_r"] = _wr
    data["wrist_rev_total"] = _wl + _wr
    # the detector's hysteresis (deg): 20 before 2026-10-04 (absent on
    # older files), 60 since; counts at different values do not compare
    data["wrist_rev_hyst_deg"] = 60
    data["wrist_rev_ticks"] = _wt
    data["wrist_rev_per_s_l"] = round(_hz * _wl / _wt, 4) if _wt else 0.0
    data["wrist_rev_per_s_r"] = round(_hz * _wr / _wt, 4) if _wt else 0.0
    data.update(term_totals(evaluated))
    data["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def pp_totals(mode, evaluated):
    """Header pp totals for a catalog mode. `total_weighted_pp` /
    `total_unweighted_pp` are the file's PRIMARY leaderboard: ScoreSaber
    for scoresaber (the historical fields, unchanged) and both,
    BeatLeader for beatleader. beatleader and both files also carry
    explicit bl_total_* (and both: ss_total_* + shared / SS-only /
    BL-only counts over the evaluated levels). Each leaderboard weights
    its own pp 0.965^rank best-first (same rule on both, see
    beatleader.py) over the levels IT ranks.
    Unweighted sum (2026-07-26, user request): wpp saturates once the
    top-300 dominate the 0.965^rank series; the raw sum keeps measuring
    breadth across the whole evaluated set."""
    if mode == "scoresaber":
        pps = [e["pp"] for e in evaluated]
        return {"total_weighted_pp": round(
                    scoresaber.weighted_total_pp(pps), 2),
                "total_unweighted_pp": round(sum(pps), 2)}
    if mode == "beatleader":
        pps = [e["pp"] for e in evaluated]
        w = round(beatleader.weighted_total_pp(pps), 2)
        u = round(sum(pps), 2)
        return {"total_weighted_pp": w, "total_unweighted_pp": u,
                "bl_total_weighted_pp": w, "bl_total_unweighted_pp": u}
    ss = [e["ss"]["pp"] for e in evaluated if e.get("ss") is not None]
    bl = [e["bl"]["pp"] for e in evaluated if e.get("bl") is not None]
    out = {"ss_total_weighted_pp": round(scoresaber.weighted_total_pp(ss), 2),
           "ss_total_unweighted_pp": round(sum(ss), 2),
           "bl_total_weighted_pp": round(beatleader.weighted_total_pp(bl), 2),
           "bl_total_unweighted_pp": round(sum(bl), 2),
           **catalog_counts(evaluated)}
    out["total_weighted_pp"] = out["ss_total_weighted_pp"]
    out["total_unweighted_pp"] = out["ss_total_unweighted_pp"]
    return out


def term_totals(evaluated):
    """Catalog-wide B09 totals over the levels that carry the fields
    (older results files lack them: `term_levels` says how many of the
    evaluated levels the totals cover)."""
    tl = [e for e in evaluated if e.get("fell") is not None]
    return {
        "term_levels": len(tl),
        "breaks_total": sum(int(e.get("breaks", 0)) for e in tl),
        "wall_enters_total": sum(int(e.get("wall_enters") or 0) for e in tl),
        "fell_levels": sum(bool(e["fell"]) for e in tl),
        "tilted_levels": sum(bool(e.get("tilted")) for e in tl),
        "envelope_levels": sum(bool(e.get("envelope")) for e in tl),
        "terminated_levels": sum(e.get("terminated_tick") is not None
                                 for e in tl),
        **torque_totals(evaluated),
    }


def torque_totals(evaluated):
    """Catalog-wide applied-torque totals (2026-10-01) over the levels
    that carry them (`torque_levels`); N·m summed per tick and the
    rate-invariant N·m·s (see _torque_fields)."""
    tq = [e for e in evaluated if e.get("torque_nm") is not None]
    out = {"torque_levels": len(tq)}
    for k in ("torque_nm", "torque_nm_spine", "torque_nm_arms",
              "torque_nm_legs", "torque_nms"):
        out[k + "_total"] = round(sum(float(e[k]) for e in tq), 1)
    return out


class _SaveWriter:
    """Periodic result dumps OFF the eval loop's critical path (B5,
    2026-08-03). Measured at full-catalog scale (6101 levels): one
    save_results is ~300 ms — 1.4 ms sort, 125 ms json.dumps(indent=1),
    the rest a 3.5 MB atomic write — and it was paid inline in the group
    loop every 30 s.

    A daemon thread now owns the dump. The loop hands it a SNAPSHOT (a
    shallow data copy + a dict copy of levels, ~0.3 ms; level entries are
    replaced wholesale, never mutated in place, so the shallow copies are
    a consistent point-in-time view) and goes straight back to stepping —
    and because json.dumps' GIL hold overlaps the Rust sim's
    allow_threads window, the dump costs the loop nothing measurable.

    Snapshots COALESCE: a newer one supersedes a pending one. That is
    safe because every snapshot carries the full cumulative state,
    including the running pd_floor_hits total (the FFI pop that feeds it
    stays on the main thread — it is a stateful counter — so a dropped
    snapshot never drops a popped count; the next one still has it).

    The on-disk format is save_results' own, unchanged (same keys, same
    sort, same indent, same tmp+os.replace). The FINAL save stays
    synchronous on the main thread, after close(), so `data` itself ends
    up holding the recomputed totals the summary line prints — and so no
    in-flight older snapshot can land after it. close() DRAINS rather
    than drops (a submit the worker had not picked up yet would
    otherwise vanish, taking its popped pd_floor_hits with it).
    """

    def __init__(self, path):
        self.path = path
        self._pending = None
        self._stop = False
        self._cv = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, data, levels):
        with self._cv:
            self._pending = (dict(data), dict(levels))
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._pending is None:   # stopped AND drained
                    return
                data, levels = self._pending   # a stop still drains the
                self._pending = None           # last snapshot to disk
            try:
                save_results(self.path, data, levels)
            except Exception as e:      # a periodic save is best-effort;
                print(f"[save] background save failed: {e!r}",  # the final
                      flush=True)                               # one is not

    def close(self):
        """Stop accepting snapshots, DRAIN the last one, and join — so
        nothing submitted is silently lost (a submit that the worker had
        not picked up yet would otherwise vanish, and with it a
        crash-resume's pd_floor_hits) and the caller's synchronous final
        save is strictly the last writer of the file."""
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join()


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
                chart.Wall, chart.wall_tuple,
                scoresaber.ComboTracker, scoresaber.max_multiplied_score,
                scoresaber.max_unit_points):
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
        "walls": [wall_tuple(w) for w in meta["walls"]],
        "max_score": scoresaber.max_multiplied_score(notes) if notes else 0,
        "duration_s": meta["duration_s"] if notes else 0.0,
        "last_hit_s": notes[-1].hit_time if notes else 0.0,
        # first_note_s clock shift (chart_t = song_t + shift): the replay
        # header's audio_offset_s is -shift (make_replays_batched.py).
        # Pickles written before 2026-10-02 lack it.
        "shift": meta["shift"],
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


def finish_slot(slot, info, combo, tick_hz=30):
    """Level-result entry from a finished rollout; combo =
    (multiplied_score, max_combo, breaks) from whichever walk ran.
    `ticks` / `flail_ticks` are SIM ticks (tick_hz per second)."""
    entry = slot["entry"]
    c_score, max_combo, breaks = combo
    score = round(c_score)
    acc = score / slot["max_score"] if slot["max_score"] else 0.0
    # BeatLeader pp of a cut-short level is 0 (user, 2026-09-30): in the
    # game a fall / tilt / envelope / max-steps cut is a failed play,
    # which earns nothing, but BL's pass term does not depend on acc.
    tf = info.get("term_flags")
    failed = bool(tf) if tf is not None else False
    if "level_key" in entry:                  # --catalog both
        ss, bl = entry["ss"], entry["bl"]
        head = {
            "level_key": entry["level_key"],
            "song_hash": entry["hash"], "song": entry["song"],
            "mapper": entry["mapper"], "difficulty": entry["difficulty"],
            "characteristic": entry["characteristic"],
            "ss_stars": ss["stars"] if ss else None,
            "bl_stars": bl["stars"] if bl else None,
            "duration_s": round(slot["duration_s"], 1),
            "n_notes": info["n_notes"],
            "score": score, "max_score": slot["max_score"],
            "accuracy": round(acc, 5),
            "ss": ss_result(ss, score, slot["max_score"]) if ss else None,
            "bl": (bl_result(bl, score, slot["max_score"], failed)
                   if bl else None)}
    elif "pass_rating" in entry:              # --catalog beatleader
        blr = bl_result(entry, score, slot["max_score"], failed)
        head = {
            "leaderboard_id": entry["leaderboard_id"],
            "song_hash": entry["hash"], "song": entry["song"],
            "mapper": entry["mapper"], "difficulty": entry["difficulty"],
            "characteristic": entry["characteristic"],
            "stars": entry["stars"],
            "pass_rating": entry["pass_rating"],
            "acc_rating": entry["acc_rating"],
            "tech_rating": entry["tech_rating"],
            "predicted_acc": entry.get("predicted_acc"),
            "duration_s": round(slot["duration_s"], 1),
            "n_notes": info["n_notes"],
            "score": score, "max_score": slot["max_score"],
            "api_max_score": entry["api_max_score"],
            "accuracy": round(acc, 5),
            "bl_accuracy": blr["accuracy"],
            "pp": blr["pp"], "pp_formula": blr["pp_formula"],
            "pass_pp": blr["pass_pp"],
            "acc_pp": blr["acc_pp"], "tech_pp": blr["tech_pp"]}
    else:                                     # --catalog scoresaber
        # ScoreSaber's accuracy is against its own API maxScore (see
        # ss_result); `env_accuracy` keeps the env's computed max
        ss_max = entry.get("api_max_score") or slot["max_score"]
        ss_acc = min(1.0, score / ss_max) if ss_max else 0.0
        head = {
            "leaderboard_id": entry["leaderboard_id"],
            "song_hash": entry["hash"], "song": entry["song"],
            "mapper": entry["mapper"], "difficulty": entry["difficulty"],
            "stars": entry["stars"], "max_pp": entry["max_pp"],
            "duration_s": round(slot["duration_s"], 1),
            "n_notes": info["n_notes"],
            "score": score, "max_score": slot["max_score"],
            "api_max_score": entry["api_max_score"],
            "accuracy": round(ss_acc, 5),
            "env_accuracy": round(acc, 5),
            "pp": round(scoresaber.pp_value(ss_acc, entry["max_pp"]), 2)}
    return {
        **head,
        "good_cuts": info["good_cuts"], "bad_cuts": info["bad_cuts"],
        "misses": info["misses"], "bombs_hit": info["bombs_hit"],
        "highest_combo": max_combo,
        "full_combo": breaks == 0 and info["good_cuts"] == info["n_notes"],
        # unmultiplied score decomposition (2026-07-25): per-band
        # component analyses come free from the results file
        "score_pre": info["score_pre"], "score_acc": info["score_acc"],
        "score_post": info["score_post"],
        # env v19 flail telemetry: blade arc (rad) by follow-through-idle
        # hands on note-free ticks + those ticks' count, this level
        "flail_rad": round(info.get("flail_rad", 0.0), 3),
        "flail_ticks": info.get("flail_ticks", 0),
        # wrist-bend reversals (2026-09-22, user): per hand, counted by
        # the sims' 60 deg-hysteresis peak detector (20 deg before
        # 2026-10-04) on the bend axis on
        # every tick (one per direction change; a flick-and-return = 1);
        # rate = per second of played level time
        "wrist_rev_l": int(info.get("wrist_rev_l", 0)),
        "wrist_rev_r": int(info.get("wrist_rev_r", 0)),
        "ticks": int(info.get("ticks", 0)),
        "wrist_rev_per_s_l": (round(float(tick_hz) * info.get("wrist_rev_l", 0) / info["ticks"], 3)
                              if info.get("ticks") else 0.0),
        "wrist_rev_per_s_r": (round(float(tick_hz) * info.get("wrist_rev_r", 0) / info["ticks"], 3)
                              if info.get("ticks") else 0.0),
        **_term_fields(breaks, info),
        **_torque_fields(info, tick_hz),
    }


def ss_result(e, score, max_score):
    """ScoreSaber's view of one played level (`both` records): accuracy
    against ScoreSaber's own API maxScore (the computed full-combo max
    when the catalog has none) and pp = curve(acc) x maxPP. The two maxes
    agree on every SS level except chain maps (2026-10-02: Bad Apple!!,
    where ScoreSaber prices arc-attached chain heads at 85 like BeatSaver
    while the game grants 115); capped at 1 like BeatLeader's view."""
    ss_max = e.get("api_max_score") or max_score
    acc = min(1.0, score / ss_max) if ss_max else 0.0
    return {"leaderboard_id": e["leaderboard_id"], "stars": e["stars"],
            "max_pp": e["max_pp"], "api_max_score": e["api_max_score"],
            "accuracy": round(acc, 5),
            "pp": round(scoresaber.pp_value(acc, e["max_pp"]), 2)}


def bl_result(e, score, max_score, failed=False):
    """BeatLeader's view of one played level. BL accuracy = score / the
    difficulty's maxScore (BeatLeader-Server ReplayUtils.ProcessReplay).
    That maxScore equals the computed full-combo max: BeatLeader prices
    a chain head that is also an arc head/tail at 115 (the game's
    ChainHeadArc* types), and so does the env since scoring v4
    (2026-10-02; before it, the env scored those heads 85 and every
    mismatch was a multiple of 30x8 pts). No API maxScore: the computed
    max. Capped at 1 (BeatLeader zeroes pp above 1; a real
    play cannot exceed its own max). `failed` (the episode was cut
    short: fall, tilt, envelope or the max-steps cap) zeroes `pp` — a
    failed play earns nothing in the game, yet BL's pass term is
    accuracy-free (a 7* level pays ~36 pp at 1% acc); `pp_formula` and
    the pass/acc/tech parts keep the raw curve for diagnostics."""
    bl_max = e.get("api_max_score") or max_score
    acc = min(1.0, score / bl_max) if bl_max else 0.0
    r = beatleader.pp_from_accuracy(acc, e["pass_rating"], e["acc_rating"],
                                    e["tech_rating"])
    return {"leaderboard_id": e["leaderboard_id"], "stars": e["stars"],
            "pass_rating": e["pass_rating"], "acc_rating": e["acc_rating"],
            "tech_rating": e["tech_rating"],
            "predicted_acc": e.get("predicted_acc"),
            "api_max_score": e.get("api_max_score"),
            "accuracy": round(acc, 5),
            "pp": 0.0 if failed else round(r["pp"], 2),
            "pp_formula": round(r["pp"], 2), "failed": failed,
            "pass_pp": round(r["pass_pp"], 2),
            "acc_pp": round(r["acc_pp"], 2), "tech_pp": round(r["tech_pp"], 2)}


def skip_record(entry, reason):
    """Results-file entry for a level that could not be played."""
    if "level_key" in entry:
        return {"level_key": entry["level_key"],
                "song_hash": entry["hash"], "song": entry["song"],
                "difficulty": entry["difficulty"],
                "ss_stars": entry["ss"]["stars"] if entry["ss"] else None,
                "bl_stars": entry["bl"]["stars"] if entry["bl"] else None,
                "ss_leaderboard_id": (entry["ss"]["leaderboard_id"]
                                      if entry["ss"] else None),
                "bl_leaderboard_id": (entry["bl"]["leaderboard_id"]
                                      if entry["bl"] else None),
                "skipped": reason}
    return {"leaderboard_id": entry["leaderboard_id"],
            "song_hash": entry["hash"], "song": entry["song"],
            "difficulty": entry["difficulty"],
            "stars": entry["stars"], "skipped": reason}


def level_line(result):
    """The per-level progress line's score part."""
    if "level_key" in result:
        parts = []
        for lb, name in (("ss", "SS"), ("bl", "BL")):
            sub = result[lb]
            if sub is not None:
                parts.append(f"{name} {sub['stars']:.2f}* pp {sub['pp']:.1f}")
        return (f"{result['song']!r} [{result['difficulty']}] acc "
                f"{100 * result['accuracy']:.2f}% | " + " | ".join(parts))
    return (f"{result['stars']:.2f}* {result['song']!r} "
            f"[{result['difficulty']}] acc {100 * result['accuracy']:.2f}% "
            f"pp {result['pp']:.1f}")


def totals_line(data):
    """The summary's pp part, per catalog."""
    mode = data.get("catalog", "scoresaber")
    if mode == "scoresaber":
        return (f"total weighted PP {data['total_weighted_pp']} | "
                f"unweighted PP {data['total_unweighted_pp']}")
    if mode == "beatleader":
        return (f"BeatLeader weighted PP {data['bl_total_weighted_pp']} | "
                f"unweighted PP {data['bl_total_unweighted_pp']}")
    return (f"ScoreSaber weighted PP {data['ss_total_weighted_pp']} / "
            f"unweighted {data['ss_total_unweighted_pp']} over "
            f"{data['n_ss_ranked']} | BeatLeader weighted PP "
            f"{data['bl_total_weighted_pp']} / unweighted "
            f"{data['bl_total_unweighted_pp']} over {data['n_bl_ranked']} | "
            f"shared {data['n_shared']}, SS-only {data['n_ss_only']}, "
            f"BL-only {data['n_bl_only']}")


def _torque_fields(info, tick_hz):
    """Applied joint torque spent on this level (2026-10-01, user): the
    env v43 Σ_j|τ_j| (N·m — exactly the --energy-coef base) summed over
    the level's control ticks, in total and per group (spine / arms /
    legs). `torque_nm` sums per TICK, so it scales with --tick-hz for the
    same motion; `torque_nms` = torque_nm / tick_hz (N·m·s, the torque
    impulse) is rate-invariant — compare 30 vs 60 Hz on it — and
    `torque_per_tick` is the mean Σ|τ| per tick (the trainer's
    [torque] line unit). info["torque"] absent = an engine build that
    cannot report it: the fields are None."""
    tq = info.get("torque")
    if tq is None:
        return {"torque_nm": None, "torque_nm_spine": None,
                "torque_nm_arms": None, "torque_nm_legs": None,
                "torque_nms": None, "torque_per_tick": None}
    total, spine, arms, legs = (float(x) for x in tq[:4])
    ticks = int(info.get("ticks", 0))
    return {"torque_nm": round(total, 1), "torque_nm_spine": round(spine, 1),
            "torque_nm_arms": round(arms, 1), "torque_nm_legs": round(legs, 1),
            "torque_nms": round(total / float(tick_hz), 2),
            "torque_per_tick": round(total / ticks, 3) if ticks else None}


def _term_fields(breaks, info):
    """Review B09 (2026-09-30): combo breaks, wall entries and how the
    level ended — the pp-loss buckets the results file used to discard
    (breaks only fed full_combo; wall entries and falls were never
    recorded, so wall head-entries had to be ESTIMATED). `fell` = FALL
    or TILT termination (the two share the fall pricing; `tilted` says
    which), `envelope` = the §23 root-escape backstop, and
    `terminated_tick` = the tick the episode was cut short at (any flag,
    incl. the max_steps cap), None when the level itself ended.
    info["term_flags"] / info["wall_enters"] absent = an engine build
    that cannot report them: the fields are None, never a fake 0."""
    tf = info.get("term_flags")
    we = info.get("wall_enters")
    out = {"breaks": int(breaks),
           "wall_enters": None if we is None else int(we)}
    if tf is None:
        out.update(fell=None, tilted=None, envelope=None,
                   terminated_tick=None)
        return out
    tf = int(tf)
    out.update(fell=bool(tf & (TERM_FELL | TERM_TILT)),
               tilted=bool(tf & TERM_TILT),
               envelope=bool(tf & TERM_ENVELOPE),
               terminated_tick=int(info.get("ticks", 0)) if tf else None)
    return out


def plan_slots(expected_ticks, n_slots):
    """Longest-processing-time assignment of levels to bench slots: sort
    levels by expected ticks descending and give each to the slot with
    the smallest load so far. Returns (queues: list of level-index lists
    per slot, makespan ticks, ideal ticks = sum/n_slots). Deterministic;
    every level exactly once. Pinned by tests/test_bench_cuda_refill.py."""
    n_slots = max(1, min(int(n_slots), len(expected_ticks)))
    order = sorted(range(len(expected_ticks)),
                   key=lambda i: (-expected_ticks[i], i))
    load = [0.0] * n_slots
    queues = [[] for _ in range(n_slots)]
    for i in order:
        k = min(range(n_slots), key=lambda j: (load[j], j))
        queues[k].append(i)
        load[k] += expected_ticks[i]
    total = float(sum(expected_ticks))
    return queues, max(load), total / n_slots


def run_cuda(args, policy, device, tasks, data, levels, out_path,
             durations, dur_path):
    """--sim cuda (2026-09-16): every level of `tasks` plays ONCE in a
    single CudaBeatVecEnv wave (auto_reset=False, combo_walk=True — the
    kernel runs the ScoreSaber multiplier walk at the event sites and
    freezes a finished level, VecBeatSim semantics). The policy forward
    is the same graphed fp32 argmax the rust path uses, run in 2048-row
    chunks (the rust path's exact batch shape, tail zero-padded) so the
    cuBLAS kernels see identical shapes. Results go through finish_slot,
    so the JSON entries are schema-identical to the rust path's."""
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    from beat_arena_env.ranked_pool import _tuple_note
    if device.type != "cuda":
        sys.exit("--sim cuda needs a CUDA device")
    ver = parser_version()
    t_load = time.perf_counter()
    charts, walls, slots, n_skip = [], [], [], 0
    max_last = 0.0
    for entry in tasks:
        try:
            pl = load_or_parse(entry, args.map_cache, args.chart_cache, ver)
            durations[dur_key(entry)] = round(pl["duration_s"], 1)
            skip = (None if pl["tuples"] else "no notes parsed")
            if skip is None and args.max_song_s > 0 \
                    and pl["last_hit_s"] > args.max_song_s:
                skip = f"{pl['last_hit_s']:.0f}s > --max-song-s"
        except Exception as e:
            pl, skip = None, f"{type(e).__name__}: {e}"
        if skip is not None:
            n_skip += 1
            levels[task_key(entry)] = skip_record(entry, skip)
            continue
        tup = mirror_tuples(pl["tuples"]) if args.mirror else pl["tuples"]
        wl = pl.get("walls", [])
        walls.append(mirror_walls(wl) if args.mirror else wl)
        charts.append([_tuple_note(t) for t in tup])
        slots.append({"entry": entry, "max_score": pl["max_score"],
                      "duration_s": pl["duration_s"],
                      "last_hit_s": pl["last_hit_s"]})
        max_last = max(max_last, pl["last_hit_s"])
    n_lv = len(charts)
    print(f"[cuda-bench] {n_lv} levels loaded ({n_skip} skipped) in "
          f"{time.perf_counter() - t_load:.1f}s; longest last hit "
          f"{max_last:.0f}s", flush=True)
    if n_lv == 0:
        save_results(out_path, data, levels)
        return
    dtype = torch.float64 if args.sim_dtype == "f64" else torch.float32
    # slot refill (2026-09-16): N slots each play a QUEUE of levels
    # (longest-first LPT assignment); a finished slot resets onto its next
    # level in-kernel the same tick, so the wave's tick count is ~the
    # total work / N + the last level's tail instead of the longest
    # level's length. Levels in `slots` are indexed by position.
    if not 1 <= args.cuda_pop_every < 60:
        sys.exit("--cuda-pop-every must be in [1, 60)")
    exp_ticks = [int(sl["last_hit_s"] * 30) + 60 for sl in slots]
    n_slots = args.cuda_slots if args.cuda_slots > 0 else n_lv
    queues, makespan, ideal = plan_slots(exp_ticks, n_slots)
    n_slots = len(queues)
    max_steps = int(max_last * 30) + 300     # per-EPISODE limit in-kernel
    env = CudaBeatVecEnv(n_slots, charts=charts, walls=walls,
                         max_steps=max_steps, gears=policy.n_gears,
                         device=device, dtype=dtype,
                         auto_reset=False, want_contacts=False,
                         # main() sets it from the checkpoint; unset = the
                         # rust sim's default (8 m)
                         z_asinh=float(os.environ.get("BEAT_Z_ASINH", "8.0")),
                         combo_walk=True, bench_refill=True, seed=0)
    ts = parse_tau_scale(args.tau_scale)
    if ts is not None and any(abs(x - 1.0) > 1e-12 for x in ts):
        env.set_tau_scale(ts)
    # `charts=` builds the pool in list order, so pool chart k == level k
    current = [q.pop(0) for q in queues]          # level playing per slot
    cand_host = np.array([(q[0] if q else -1) for q in queues],
                         dtype=np.int64)
    # reset() starts every slot on the chart in the candidate array, so
    # it holds the FIRST level now and the NEXT level from then on
    env.set_next_chart(torch.as_tensor(np.array(current, dtype=np.int64),
                                       device=device))
    CH = 2048                                # the rust path's batch shape
    fwd = (None if args.no_graphs else _GraphedArgmax(policy, CH, device))
    buf = torch.zeros((CH, OBS_DIM), device=device)
    data["forward"] = "eager" if fwd is None else "graphed"
    data["sim_dtype"] = args.sim_dtype
    data["forward_chunk"] = CH
    obs = env.reset()
    env.set_next_chart(torch.as_tensor(cand_host, device=device))
    acts = torch.zeros((n_slots, N_JOINTS), dtype=torch.int64, device=device)
    n_sc_pool = env.pool["n_scoreable"].cpu().numpy()
    active = np.ones(n_slots, dtype=bool)         # slot has a live level
    active_idx = torch.arange(n_slots, device=device)
    n_done_levels = 0
    t0 = time.perf_counter()
    tick = 0
    tick_guard = int(3 * makespan) + max_steps
    while n_done_levels < n_lv and tick < tick_guard:
        # forward on the active slots only, in fixed 2048-wide chunks
        # (zero-padded) so the per-sample numerics keep the pinned shape
        n_act = int(active_idx.numel())
        obs_a = obs.index_select(0, active_idx)
        out = torch.empty((n_act, N_JOINTS), dtype=torch.int64, device=device)
        for lo in range(0, n_act, CH):
            hi = min(lo + CH, n_act)
            buf[:hi - lo].copy_(obs_a[lo:hi])
            if hi - lo < CH:
                buf[hi - lo:].zero_()
            if fwd is not None:
                fwd._copy_static(fwd.s_obs, buf)
                a = fwd._run()
            else:
                with torch.no_grad():
                    logits, _ = policy.forward_stacked(buf)
                a = logits.argmax(-1)
            out[lo:hi].copy_(a[:hi - lo])
        acts.zero_()
        acts.index_copy_(0, active_idx, out)
        obs, _, _done, _ = env.step(acts)
        tick += 1
        # pop finished levels every --cuda-pop-every ticks (one host sync
        # each). Safe below 60: a level lasts >= 60 ticks, so no slot can
        # finish twice between pops, and the kernel's refill reads
        # cand[e] = that slot's next level, advanced here in time.
        if tick % 300 == 0 and not args.quiet:
            el = time.perf_counter() - t0
            print(f"[cuda-bench] tick {tick}  levels done {n_done_levels}/"
                  f"{n_lv}  active slots {int(active.sum())}/{n_slots}  "
                  f"{el:.0f}s", flush=True)
        if tick % args.cuda_pop_every:
            continue
        ids, rows = env.pop_bench_rows()
        if len(ids):
            for e, r in zip(ids.tolist(), rows):
                lv = current[e]
                slot = slots[lv]
                ci = int(r[13])
                result = finish_slot(slot, {
                    "good_cuts": int(r[0]), "bad_cuts": int(r[1]),
                    "misses": int(r[2]), "n_notes": int(n_sc_pool[ci]),
                    "bombs_hit": int(r[4]),
                    "score_pre": float(r[5]), "score_acc": float(r[6]),
                    "score_post": float(r[7]), "flail_rad": float(r[8]),
                    "flail_ticks": int(r[9]),
                    "wrist_rev_l": int(r[14]), "wrist_rev_r": int(r[15]),
                    "ticks": int(r[16]), "wall_enters": int(r[17]),
                    "term_flags": int(r[18]),
                    "torque": (float(r[19]), float(r[20]),
                               float(r[21]), float(r[22]))},
                    (float(r[10]), int(r[11]), int(r[12])))
                result["cuda_slot"] = int(e)
                levels[task_key(result)] = result
                n_done_levels += 1
                if not args.quiet:
                    print(f"[{n_done_levels}/{n_lv}] {level_line(result)}"
                          f"{'  FC' if result['full_combo'] else ''}",
                          flush=True)
                # the kernel already reset slot e onto cand_host[e]; advance
                if cand_host[e] >= 0:
                    current[e] = int(cand_host[e])
                    q = queues[e]
                    q.pop(0)
                    cand_host[e] = q[0] if q else -1
                else:
                    current[e] = -1
                    active[e] = False
            env.set_next_chart(torch.as_tensor(cand_host, device=device))
            active_idx = torch.as_tensor(np.flatnonzero(active),
                                         device=device)
    wall = time.perf_counter() - t0
    for lv, slot in enumerate(slots):
        lid = task_key(slot["entry"])
        if lid not in levels:
            levels[lid] = skip_record(
                slot["entry"],
                f"not finished by tick {tick} (guard {tick_guard})")
    data["pd_floor_hits"] = data.get("pd_floor_hits", 0) + env.pop_pd_hits()
    data["envelope_hits"] = (data.get("envelope_hits", 0)
                             + env.pop_envelope_hits())
    useful = sum(exp_ticks)
    data["cuda_bench"] = {
        "levels": n_lv, "slots": n_slots, "ticks": tick,
        "wall_s": round(wall, 1),
        "useful_env_steps_per_s": round(useful / wall),
        "ideal_ticks": round(ideal), "planned_makespan_ticks": round(makespan),
        "scheduler": "LPT longest-first, level = position in the task list",
        "max_steps": max_steps, "pop_every": args.cuda_pop_every,
        # the reproducibility record: slot k's level positions in play order
        "assignment": [[current_lv for current_lv in q]
                       for q in plan_slots(exp_ticks, n_slots)[0]]}
    save_results(out_path, data, levels)
    if args.chart_cache:
        os.makedirs(args.chart_cache, exist_ok=True)
        tmp = dur_path + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(durations, f)
        os.replace(tmp, dur_path)
    print(f"\ndone (cuda, {args.sim_dtype}): {n_lv} played ({n_skip} skipped), "
          f"{data['n_evaluated']} levels on file | {totals_line(data)} "
          f"| wrist reversals "
          f"{data['wrist_rev_total']} (L {data['wrist_rev_total_l']} / R "
          f"{data['wrist_rev_total_r']}; {data['wrist_rev_per_s_l']} / "
          f"{data['wrist_rev_per_s_r']} per s) | {_term_summary(data)} "
          f"| pd-floor hits "
          f"{data['pd_floor_hits']} | envelope hits {data['envelope_hits']} "
          f"| {n_slots} slots, {tick} ticks (ideal {ideal:.0f}) in {wall:.0f}s "
          f"({useful / wall / 1e3:.0f}k useful env-steps/s) -> {out_path}",
          flush=True)


def _term_summary(data):
    return (f"breaks {data['breaks_total']}, wall entries "
            f"{data['wall_enters_total']}, fell {data['fell_levels']} "
            f"(tilt {data['tilted_levels']}), envelope "
            f"{data['envelope_levels']} of {data['term_levels']} levels"
            + (f" | torque {data['torque_nms_total']:,.0f} N·m·s "
               f"({data['torque_nm_total']:,.0f} N·m over ticks) on "
               f"{data['torque_levels']} levels"
               if data.get("torque_levels") else ""))


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
    # env v48: the obs z encoding is the CHECKPOINT's asinh knee C (arch,
    # else the embedded args). The rust sim reads BEAT_Z_ASINH once per
    # process, so set it before any sim exists.
    z_asinh = float(ckpt.get("arch", {}).get(
        "z_asinh", ckpt.get("args", {}).get("z_asinh", 0.0)))
    if not z_asinh > 0:
        sys.exit(f"checkpoint has z_asinh {z_asinh:g}: it predates the asinh "
                 f"z encoding (legacy clamp, removed 2026-09-22) and cannot "
                 f"be benched")
    if os.environ.get("BEAT_Z_ASINH") not in (None, repr(z_asinh)):
        sys.exit(f"BEAT_Z_ASINH={os.environ['BEAT_Z_ASINH']} disagrees with "
                 f"the checkpoint's --z-asinh {z_asinh}; unset it")
    os.environ["BEAT_Z_ASINH"] = repr(z_asinh)
    print(f"obs z lane: asinh(z / {z_asinh:g} m) (from the checkpoint)",
          flush=True)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy = policy.to(device).eval()

    out_path = args.out or default_out_path(args.ckpt)
    if args.catalog != "scoresaber" and not args.out:   # own default file
        root, ext = os.path.splitext(out_path)          # per catalog
        out_path = f"{root}_{args.catalog}{ext}"
    if args.mirror and not args.out:     # never merge mirrored scores into
        root, ext = os.path.splitext(out_path)   # the normal results file
        out_path = f"{root}_mirror{ext}"
    ckpt_name = "/".join(os.path.normpath(os.path.abspath(args.ckpt))
                         .replace("\\", "/").split("/")[-2:])
    data = load_results(out_path, mhash) or {
        "checkpoint": args.ckpt, "checkpoint_name": ckpt_name,
        "model_hash": mhash, "arch": ckpt.get("arch"),
        "global_step": ckpt.get("global_step"),
        "foot_contact": True,   # physics-regime provenance (the only mode)
        "tau_scale": parse_tau_scale(args.tau_scale),  # env v46 body
        # exp/wrist-tendon body marker: the passive wrist tendon peaks
        # (N.m) and edge-damping peaks (N.m.s/rad, issue #61) this sim ran
        # with; absent = a tendon-free body
        "wrist_tendon": [TENDON_TAU_PEAK_BEND, TENDON_TAU_PEAK_DEV,
                         TENDON_B_PEAK_BEND, TENDON_B_PEAK_DEV],
        "mirror": args.mirror,               # charts left-right reflected
    }
    if bool(data.get("mirror")) != args.mirror:
        sys.exit(f"error: {out_path} holds mirror={data.get('mirror')} "
                 f"results but this run has --mirror={args.mirror} — "
                 f"use a different --out")
    # --catalog provenance (2026-09-30): a file holds ONE catalog's levels
    # (files without the stamp are scoresaber — the only kind before);
    # level identities differ per catalog (SS id / BL id / level_key), so
    # a cross-catalog merge would silently double-count — refused, even
    # for a skipped-only file
    prev_cat = data.get("catalog", "scoresaber")
    if prev_cat != args.catalog and data.get("levels"):
        sys.exit(f"error: {out_path} holds catalog={prev_cat!r} results "
                 f"but this run is --catalog {args.catalog}; use a "
                 f"different --out")
    data["catalog"] = args.catalog
    levels = {task_key(e): e for e in data.get("levels", [])}

    catalog, cat_stamp = build_tasks(args)
    data.update(cat_stamp)

    tasks = [e for e in catalog
             if in_star_range(e, args.stars, args.catalog)]
    in_range = len(tasks)
    if not args.force:                   # merge policy: evaluated = skip,
        tasks = [e for e in tasks        # previously-skipped = retry
                 if task_key(e) not in levels
                 or "skipped" in levels[task_key(e)]]
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
    title = {"scoresaber": "ScoreSaber", "beatleader": "BeatLeader",
             "both": "ScoreSaber + BeatLeader"}[args.catalog]
    unit = "levels" if args.catalog == "both" else "leaderboards"
    extra = ""
    if args.catalog == "both":
        c = catalog_counts(tasks)
        extra = (f" (in range: shared {c['n_shared']}, SS-only "
                 f"{c['n_ss_only']}, BL-only {c['n_bl_only']})")
    print(f"=== {title} bench: {ckpt_name} ({mhash[:19]}...) on {device}\n"
          f"    {in_range} {unit} in range, {len(tasks)} to play, "
          f"{len(levels)} already on file -> {out_path}{extra}", flush=True)
    if not tasks:
        save_results(out_path, data, levels)
        if args.catalog == "scoresaber":
            print(f"nothing to do; totals refreshed "
                  f"(weighted PP {data['total_weighted_pp']}, unweighted "
                  f"{data['total_unweighted_pp']})", flush=True)
        else:
            print(f"nothing to do; totals refreshed ({totals_line(data)})",
                  flush=True)
        return

    # small lookahead (2026-07-27): the warm producer is ~70x faster than
    # consumer demand, so 2*N slots just parked ~290 MB of live note
    # tuples (~1.1M containers) for every gen2 GC pass to walk
    full_force = args.force and args.stars is None and args.limit == 0
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
    # --sim provenance (2026-09-16): rust and cuda results never merge
    prev_sim = data.get("sim", "rust")
    if prev_sim != args.sim and any("skipped" not in e
                                     for e in levels.values()):
        sys.exit(f"error: {out_path} holds sim={prev_sim!r} results but "
                 f"this run is --sim {args.sim}; use a different --out")
    data["sim"] = args.sim
    # --sim-dtype provenance (2026-09-16): rust files written before the
    # f32 build existed carry no key and are f64; cuda files always had it
    prev_dt = data.get("sim_dtype", "f64" if prev_sim == "rust" else "f32")
    if prev_dt != args.sim_dtype and any("skipped" not in e
                                         for e in levels.values()):
        sys.exit(f"error: {out_path} holds sim_dtype={prev_dt!r} results "
                 f"but this run is --sim-dtype {args.sim_dtype}; use a "
                 f"different --out")
    data["sim_dtype"] = args.sim_dtype
    # exp/tick-rate provenance: files without the keys are 30 Hz / repeat 1
    # (the only regime before 2026-09-28); results never merge across them
    if args.tick_hz not in (30, 60, 120, 240):
        sys.exit(f"--tick-hz must be 30, 60, 120 or 240, got {args.tick_hz}")
    if args.action_repeat < 1:
        sys.exit("--action-repeat must be >= 1")
    if args.sim == "cuda" and (args.tick_hz != 30 or args.action_repeat != 1):
        sys.exit("--tick-hz / --action-repeat are --sim rust only (the "
                 "CUDA kernel is fixed at 30 Hz)")
    for key, val in (("tick_hz", args.tick_hz),
                     ("action_repeat", args.action_repeat)):
        prev = data.get(key, 30 if key == "tick_hz" else 1)
        if prev != val and any("skipped" not in e for e in levels.values()):
            sys.exit(f"error: {out_path} holds {key}={prev} results but this "
                     f"run is {key}={val}; use a different --out")
        data[key] = val
    if args.sim == "cuda":
        run_cuda(args, policy, device, tasks, data, levels, out_path,
                 durations, dur_path)
        return

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
    # every step, eventless) so the vector call is always valid. Those
    # slots are ACTUALLY free since 2026-08-03 (B4): the old claim was
    # wrong — Rust ran the full 8-substep dynamics tick for every idle
    # slot on every step, then recomputed `done` from state and said
    # yes. rust/src/lib.rs now latches `finished` and early-outs, so an
    # idle slot costs exactly one tick per recycle.
    N = args.num_envs
    # Two-group pipeline (2026-07-27): with a single group the loop is a
    # strict serial chain — GPU idles during the Rust sim, CPU idles
    # during the forward. Two half-size groups alternate: while group A's
    # H2D+kernel+async-D2H chain runs on the (single) stream, the CPU
    # syncs B's event and runs B's sim + bookkeeping. Fused path only:
    # the kernel's per-sample batch invariance (verified bitwise 512 vs
    # 1024) is what makes N/2 groups score-identical; cuBLAS is not.
    # (the fused per-sample forward and its two-group pipeline were
    # retired 2026-08-13 with the 138k arch; the graphed torch forward
    # is the only CUDA path now — batch width is score-relevant on it,
    # so N stays exactly --num-envs)
    if args.groups < 1 or N % args.groups:
        sys.exit(f"--num-envs {N} must split evenly into --groups "
                 f"{args.groups}")
    sizes = [N // args.groups] * args.groups
    if args.sim_dtype == "f32":
        if beat_arena_rs32 is None:
            sys.exit("--sim-dtype f32 needs the beat_arena_rs32 module: "
                     "cd rust32 && maturin develop --release "
                     "(or pass --sim-dtype f64)")
        sim_mod = beat_arena_rs32
    else:
        sim_mod = beat_arena_rs
    if not hasattr(sim_mod.VecBeatSim, "term"):     # B09 getter
        sys.exit(f"{sim_mod.__name__} predates VecBeatSim.term (wall "
                 f"entries + termination flags, 2026-09-30): rebuild it "
                 f"(maturin develop --release in rust/ or rust32/)")
    groups = []
    for n_g in sizes:
        fwd_g = (None if args.no_graphs or device.type != "cuda"
                 else _GraphedArgmax(policy, n_g, device))
        groups.append(_Group(n_g, fwd_g, device, policy.n_gears,
                             tau_scale=parse_tau_scale(args.tau_scale),
                             sim_mod=sim_mod, tick_hz=args.tick_hz))
    fwd = groups[0].fwd                  # for the provenance stamp below
    # Forward-path provenance (2026-07-27): different forward paths
    # differ by ~4e-7 in the logits, enough to flip near-tie argmaxes —
    # per-level results are only comparable within one path. Same
    # weights + different path must therefore NOT silently merge.
    fwd_path = ("graphed" if isinstance(fwd, _GraphedArgmax) else "eager")
    prev_path = data.get("forward")
    if prev_path is not None and prev_path != fwd_path:
        sys.exit(f"error: {out_path} holds {prev_path!r}-forward results but "
                 f"this run would use {fwd_path!r}. Matching flag "
                 f"(--no-graphs) or a fresh --out required; --force with "
                 f"the matching flag re-evaluates everything on one path.")
    full_force = args.force and args.stars is None and args.limit == 0
    if (prev_path is None and not full_force
            and any("skipped" not in e for e in levels.values())):
        print(f"[forward] warning: {out_path} predates forward-path "
              f"stamping; its levels' provenance is unverifiable and the "
              f"file stays unstamped (full-range --force re-bench adopts "
              f"{fwd_path!r})", flush=True)
    else:
        data["forward"] = fwd_path
    # forward batch width (2026-09-16, --groups): score-relevant on the
    # graphed path, so a file never mixes widths (unstamped files: unknown)
    width = sizes[0]
    prev_w = data.get("forward_width")
    if (prev_w is not None and prev_w != width and not full_force
            and any("skipped" not in e for e in levels.values())):
        sys.exit(f"error: {out_path} holds forward-width {prev_w} results "
                 f"but this run uses {width} (--num-envs/--groups); use a "
                 f"fresh --out")
    data["forward_width"] = width
    # (persistent pinned obs/act buffers live in each _Group: step_into
    # writes obs rows in place, the async D2H lands in pinned act memory,
    # and the group's event is the sync that makes both safe to touch)
    feed_done, n_done, n_new_skip = False, 0, 0
    last_save = time.monotonic()
    writer = _SaveWriter(out_path)        # B5: periodic dumps off-thread
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
                    # --action-repeat: the action is held for N sim ticks
                    # inside the one call; obs = the next decision tick's
                    _, dones, _events = gr.sim.step_into(
                        gr.act_np, gr.obs_buf, repeat=args.action_repeat)
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
                        fl_rad, fl_ticks = gr.sim.flail(i)
                        wr_l, wr_r, wr_ticks = gr.sim.wrist_rev(i)
                        w_ent, t_flags = gr.sim.term(i)   # B09
                        tq = (gr.sim.pop_torque_env(i)[:4]
                              if hasattr(gr.sim, "pop_torque_env") else None)
                        result = finish_slot(s, {
                            "good_cuts": g, "bad_cuts": b, "misses": m,
                            "n_notes": n_scoreable, "bombs_hit": bombs,
                            "score_pre": c_pre, "score_acc": c_acc,
                            "score_post": c_post, "flail_rad": fl_rad,
                            "flail_ticks": fl_ticks,
                            "wrist_rev_l": wr_l, "wrist_rev_r": wr_r,
                            "ticks": wr_ticks, "wall_enters": w_ent,
                            "term_flags": t_flags, "torque": tq},
                            combo, tick_hz=args.tick_hz)
                        levels[task_key(result)] = result
                        n_done += 1
                        n_finished += 1
                        if not args.quiet:
                            print(f"[{n_done}/{len(tasks)}] "
                                  f"{level_line(result)}"
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
                        # resume. The pop MUST stay here on the main
                        # thread — it is a stateful counter — and it
                        # rides the snapshot the writer dumps.
                        data["pd_floor_hits"] = (
                            data.get("pd_floor_hits", 0)
                            + sum(g.sim.pop_pd_hits() for g in groups))
                        data["envelope_hits"] = (
                            data.get("envelope_hits", 0)
                            + sum(g.sim.pop_envelope_hits()
                                  for g in groups))
                        writer.submit(data, levels)   # B5: off-thread dump
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
                        levels[task_key(entry)] = skip_record(entry, skip)
                        continue
                    i = gr.free.pop()
                    if args.mirror:
                        tuples = mirror_tuples(tuples)
                    gr.sim.set_chart(i, tuples,
                                     mirror_walls(payload.get("walls", []))
                                     if args.mirror
                                     else payload.get("walls", []))
                    if hasattr(gr.sim, "pop_torque_env"):
                        gr.sim.pop_torque_env(i)   # this level's torque only
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
        # any in-flight periodic dump finishes first, so the synchronous
        # final save below is the last writer of out_path (B5)
        writer.close()
        # f64 Cholesky pivot-floor engagements (SPEC-v10 Divergences
        # §22) across all sims — reported, not hidden (the §20e house
        # rule): a nonzero count marks contorted gimbal-adjacent
        # states the pre-§22 engine PANICKED on. Final pop of the
        # tail since the last periodic save.
        data["pd_floor_hits"] = (data.get("pd_floor_hits", 0)
                                 + sum(gr.sim.pop_pd_hits() for gr in groups))
        pd_hits = data["pd_floor_hits"]
        # pre-integration root-envelope engagements (SPEC-v10 Divergences
        # §23) across all sims — each one terminated that env's level
        # EARLY with a zeroed tick reward, so a nonzero count is a caveat
        # on the affected level's score. Expected 0 for any converged
        # checkpoint (measured: 0 in 204,800 env-ticks).
        data["envelope_hits"] = (data.get("envelope_hits", 0)
                                 + sum(gr.sim.pop_envelope_hits()
                                       for gr in groups))
        env_hits = data["envelope_hits"]
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
    print(f"\ndone (rust, {args.sim_dtype}, {args.tick_hz} Hz, action repeat "
          f"{args.action_repeat}): {n_done} played ({n_new_skip} skipped), "
          f"{data['n_evaluated']} levels on file | "
          f"{totals_line(data)} | "
          f"wrist reversals {data['wrist_rev_total']} (L "
          f"{data['wrist_rev_total_l']} / R {data['wrist_rev_total_r']}; "
          f"{data['wrist_rev_per_s_l']} / {data['wrist_rev_per_s_r']} per s) | "
          f"{_term_summary(data)} | "
          f"pd-floor hits {pd_hits} (file total) "
          f"| envelope hits {env_hits} (file total) "
          f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
