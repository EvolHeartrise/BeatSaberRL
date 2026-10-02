"""Record + verify + convert a bench's top-N pp plays to watchable replays.

    python make_top10_replays.py --ckpt <ckpt.pt> --bench <bench.json> \
        --tag gen2_429m [--top 10] [--workers 10]

For each top play: greedy re-rollout with the bench protocol (cached
map, real NJS speeds, seed 0) in the bench's own regime — its sim dtype
(f32 by default), its forward batch width via the bench's graphed
argmax, and its recorded body (tau_scale) — recording a format-2 replay
(<name>.rpl.gz, beat_arena_env/replay_format.py), verify the outcome
counters and multiplied score against the bench entry, and (--dat,
opt-in since 2026-10-02) convert to a ScoreSaber .dat and verify the
decode. Deterministic sim ⇒ exact. A failing play is reported and the
run goes on (exit status 1 at the end). Many plays at once: see
make_replays_batched.py (one batched sim + forward for a whole set).
(Was a 598M one-off; parameterized 2026-07-24.)

Control rate (2026-10-01): the bench header's tick_hz / action_repeat
(absent = 30 / 1) are reproduced too — the sim runs at that rate, the
policy decides every action_repeat-th tick (the action held in
between, as the bench's step_into(repeat=N)), and the recording carries
one line per SIM tick with the rate in its header, so the viewer and
the .dat play it back at real speed. --tick-hz / --action-repeat
override the header.

--catalog both benches (2026-10-01): records are keyed by level_key and
carry `ss` / `bl` sub-results (None where that leaderboard does not rank
the level). --board picks the view — ss (default) ranks by ss.pp over
the ScoreSaber-ranked levels, bl by bl.pp over the BeatLeader-ranked
ones (BL-only levels included); the play itself is the record's either
way. --levels KEY... records exactly those levels (level_key, or the
leaderboard id on single-catalog benches) instead of a top-N, each in
the --board view when that board ranks it, else the other's: how a
best-of composite's picks are recorded, one SOURCE bench at a time
(each pick in its own run's regime). Composites themselves are refused:
their records mix runs. Every recording's header also carries the
weights hash (`model_hash`) and the level identity (`level`: level_key,
characteristic, SS / BL leaderboard ids) next to tick_hz and the body
(tau_scale); maps missing from maps/scoresaber/ are fetched from
BeatSaver (BL-only levels live there too, as in the bench).

--workers N (2026-08-08, pod era): plays are independent, so they run
in a spawn-context process pool — each worker loads its own policy and
sim. Serial generation left a 256-core pod ~idle; outputs are byte-
identical to serial (same seed, same deterministic rollout; verified
at introduction by hashing a full set both ways). Worker prints arrive
in rank order regardless of completion order.
"""

import argparse
import numpy as np
import math
import json
import multiprocessing
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import torch

import convert_scoresaber as conv
from beat_arena_env import scoresaber
from bench_scoresaber import model_hash
from beat_arena_env.constants import OBS_DIM, parse_tau_scale
from beat_arena_env.beatsaver import load_beatmap
from beat_arena_env.rust_env import RustBeatEnv
from policy import load_policy


def slug(s, n=24):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:n].lower() or "song"


_W = {}          # per-process state: {"policy", "dev", "push", "tau"} (set by _worker_init)


BOARDS = ("ss", "bl")


def task_key(rec):
    """A bench record's identity: level_key (--catalog both), else its
    leaderboard id (single-catalog benches) — as bench_scoresaber.task_key,
    stringified (BL ids are strings, SS ids ints; --levels are strings)."""
    return str(rec.get("level_key", rec.get("leaderboard_id")))


def board_view(rec, board):
    """`rec` as seen from one leaderboard, in the single-catalog record
    shape (`pp`, `stars`, `accuracy`, `leaderboard_id` top-level), or None
    when that board does not rank the level. A --catalog both record keeps
    its play fields (score, counters, combo) and takes the board's sub-
    result; single-catalog records are their board's view already."""
    if "level_key" not in rec:
        return dict(rec)
    sub = rec.get(board)
    if not sub:
        return None
    return {**rec, "pp": sub["pp"], "stars": sub["stars"],
            "accuracy": sub.get("accuracy", rec["accuracy"]),
            "leaderboard_id": sub["leaderboard_id"]}


def bench_pool(bench, board="ss", fallback=False):
    """The bench's played levels in `board`'s view (skips dropped). Levels
    the board does not rank are dropped, or with `fallback` (explicit
    --levels on a --catalog both bench) seen from the other board."""
    if bench.get("composite"):
        raise SystemExit("error: a best-of composite mixes runs (rates, "
                         "bodies); record each pick from its SOURCE bench "
                         "(composite ss_from / bl_from) with --levels")
    cat = bench.get("catalog", "scoresaber")
    if cat != "both" and board != {"scoresaber": "ss",
                                   "beatleader": "bl"}.get(cat, board):
        raise SystemExit(f"error: --board {board} on a catalog={cat} bench")
    other = BOARDS[1 - BOARDS.index(board)]
    pool = ((board_view(e, board) or
             (board_view(e, other) if fallback else None))
            for e in bench["levels"] if "skipped" not in e)
    return [e for e in pool if e is not None]


def select_levels(pool, keys):
    """The records named by `keys`, in `keys` order; every key must name
    exactly one played level of the pool."""
    by = {}
    for e in pool:
        by.setdefault(task_key(e), []).append(e)
    out = []
    for k in keys:
        hit = by.get(str(k), [])
        if len(hit) != 1:
            raise SystemExit(f"error: --levels {k}: {len(hit)} played "
                             f"levels in this bench/board")
        out.append(hit[0])
    return out


def level_identity(entry):
    """header.level: which leaderboard level(s) the replay is a play of."""
    ss, bl = entry.get("ss"), entry.get("bl")
    if "level_key" not in entry:                  # single-catalog record
        ss = {"leaderboard_id": entry["leaderboard_id"]} \
            if "max_pp" in entry else None
        bl = {"leaderboard_id": entry["leaderboard_id"]} \
            if "pass_rating" in entry else None
    return {"level_key": entry.get("level_key"),
            "characteristic": entry.get("characteristic", "Standard"),
            "ss_leaderboard_id": ss["leaderboard_id"] if ss else None,
            "bl_leaderboard_id": bl["leaderboard_id"] if bl else None}


def _worker_init(ckpt_path, device, push=(0.0, 0.0, 0), tau_scale=None,
                 sim_dtype="f32", width=1024, rate=(30, 1), mhash=None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # env v48: the checkpoint's obs z encoding (asinh knee C); the rust
    # sim reads BEAT_Z_ASINH once per process (this runs before any sim
    # exists)
    z_asinh = float(ckpt.get("arch", {}).get(
        "z_asinh", ckpt.get("args", {}).get("z_asinh", 0.0)))
    if not z_asinh > 0:
        raise SystemExit(f"{ckpt_path}: z_asinh {z_asinh:g} — the checkpoint "
                         f"predates the asinh z encoding (legacy clamp, "
                         f"removed 2026-09-22)")
    os.environ["BEAT_Z_ASINH"] = repr(z_asinh)
    dev = torch.device(device)
    _W["policy"] = load_policy(ckpt).to(dev).eval()
    _W["dev"] = dev
    # the bench's own regime (2026-09-26): its sim dtype and its forward —
    # the graphed fp32 argmax over a fixed (width, OBS_DIM) batch. GEMM
    # kernel choice depends on the batch width, so a batch-1 eager forward
    # differs from the bench by ~1e-7 in the logits and flips near-tie
    # argmaxes ("near-tie drift"); the play's obs rides in row 0 of a
    # zero-padded batch of the bench's width instead.
    _W["sim_dtype"] = sim_dtype
    _W["buf"] = torch.zeros(width, OBS_DIM,
                            pin_memory=dev.type == "cuda")
    if dev.type == "cuda":
        from bench_scoresaber import _GraphedArgmax
        _W["fwd"] = _GraphedArgmax(_W["policy"], width, dev)
    else:
        def _eager(buf):
            with torch.no_grad():
                logits, _ = _W["policy"].forward_stacked(buf.to(dev))
            return logits.argmax(-1)
        _W["fwd"] = _eager
    _W["push"] = push
    _W["tau"] = tau_scale   # env v46 fixed body (None = all 1.0)
    _W["rate"] = rate       # (tick_hz, action_repeat)
    _W["mhash"] = mhash     # weights hash, stamped in the replay header


def _kick(env, rng, vmax):
    """env v18 perturbation replayed on the Rust sim (2026-09-14, user
    request: 'the winning checkpoint with the random pushes turned on'):
    the CUDA trainer's per-tick horizontal root-velocity kick, |dv| ~
    U(0, vmax) m/s, direction uniform in the x-z plane. The Rust sim has
    no push path, so the kick goes through the tooling hook set_joints
    (which also zeroes the finite-difference wrist/body-point velocities
    for one tick — a one-tick obs artifact, not a physics change)."""
    q = list(env.sim.qpos())
    v = list(env.sim.twist()) + list(env.sim.omega())
    mag = rng.random() * vmax
    ang = rng.random() * 2.0 * math.pi
    v[0] += mag * math.cos(ang)
    v[2] += mag * math.sin(ang)
    env.sim.set_joints(q, v)
    return mag


def _make_one(job):
    """One play, with per-job error handling (2026-10-02): a failing play
    (map fetch, parse, convert) is reported by the parent and the run
    goes on. Returns (base, line, warn, error)."""
    try:
        return (*_make_one_inner(job), None)
    except Exception as e:
        import traceback
        rank, n_top, entry = job[0], job[1], job[2]
        return (None, None, None,
                f"[{rank:2}/{n_top}] FAILED {entry.get('song', '?')!r} "
                f"[{entry.get('difficulty')}]: {type(e).__name__}: {e}\n"
                + traceback.format_exc())


def _make_one_inner(job):
    """One play: re-rollout, verify vs bench entry, and (--dat) convert +
    verify the decode. Returns (base, line, warn) for rank-ordered
    printing by the parent."""
    rank, n_top, entry, tag, label = job[:5]
    dat = job[5] if len(job) > 5 else False
    policy, dev = _W["policy"], _W["dev"]
    # the bench's map cache (BL-only levels too), fetched on a miss: a pod
    # gets chart_cache pickles, not raw maps
    map_dir = scoresaber.ensure_map_dats(entry["song_hash"], "maps/scoresaber")
    notes, meta = load_beatmap(map_dir, entry["difficulty"], first_note_s=2.0)
    base = (f"replays/{tag}_{label}{rank:02d}_"
            f"{slug(entry['song'])}_{entry['pp']:.0f}pp")
    tick_hz, repeat = _W.get("rate", (30, 1))
    env = RustBeatEnv(external_chart=notes, seed=0, gears=policy.n_gears,
                      sim_dtype=_W["sim_dtype"],
                      external_walls=meta["walls"], tick_hz=tick_hz,
                      action_repeat=repeat)
    if _W.get("tau") is not None:
        env.set_tau_scale(_W["tau"])
    # the map on the recording so the 3D viewer can pull this song's audio
    # from BeatSaver and line it up (meta["shift"] is what first_note_s
    # did to the clock; -shift is the song time at replay t=0), the level
    # identity and the weights hash — written into the header up front
    # (2026-10-02; was a stamp() rewrite of the finished file)
    env.set_record(base + ".rpl.gz",
                   map={"hash": entry["song_hash"],
                        "difficulty": entry["difficulty"],
                        "song": entry["song"],
                        "audio_offset_s": -meta["shift"]},
                   level=level_identity(entry),
                   **({"model_hash": _W["mhash"]} if _W.get("mhash")
                      else {}))
    obs, _ = env.reset()
    walk = scoresaber.ComboTracker()
    push_prob, push_vmax, push_seed = _W.get("push", (0.0, 0.0, 0))
    rng = np.random.default_rng(push_seed + rank)
    n_push, tick = 0, 0
    done = False
    while not done:
        if push_prob > 0.0 and rng.random() < push_prob:
            _kick(env, rng, push_vmax)
            n_push += 1
        tick += 1
        buf = _W["buf"]
        buf[0].copy_(torch.as_tensor(obs, dtype=torch.float32))
        acts = _W["fwd"](buf)[0].cpu().tolist()
        obs, _, done, _, info = env.step([int(x) for x in acts])
        walk.consume(info["events"])
    exact = (round(walk.score) == entry["score"]
             and walk.max_combo == entry["highest_combo"]
             and info["good_cuts"] == entry["good_cuts"]
             and info["bad_cuts"] == entry["bad_cuts"]
             and info["misses"] == entry["misses"]
             and info["bombs_hit"] == entry["bombs_hit"])
    verify = (f"    verify: score {round(walk.score)}/{entry['score']} "
              f"combo {walk.max_combo}/{entry['highest_combo']} "
              f"good {info['good_cuts']}/{entry['good_cuts']} "
              f"bad {info['bad_cuts']}/{entry['bad_cuts']} "
              f"miss {info['misses']}/{entry['misses']} "
              f"bombs {info['bombs_hit']}/{entry['bombs_hit']} "
              f"(replay/bench) {'EXACT' if exact else 'MISMATCH'}")
    warn = None
    if not exact:
        # fp32 argmax near-tie flipped somewhere vs the bench run —
        # still a faithful greedy play of THIS checkpoint; verify
        # the pipeline self-consistently against the replay itself
        warn = (f"    WARNING {entry['song']}: replay score "
                f"{round(walk.score)} != bench {entry['score']} "
                f"(near-tie drift; replay is self-consistent)")

    # ScoreSaber .dat + decode check: opt-in (--dat, 2026-10-02) — the
    # replay site plays the .rpl.gz; the .dat is for watch.scoresaber.com
    out = base + ".rpl.gz"
    if dat:
        sys.path.insert(0, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "tests"))
        from test_convert_scoresaber import decode_dat
        conv.convert(base + ".rpl.gz", map_dir, entry["song_hash"],
                     entry["difficulty"], base + ".dat")
        d = decode_dat(base + ".dat")
        assert d["final_score"] == round(walk.score), \
            f".dat decode {d['final_score']} != replay {round(walk.score)}"
        out = base + ".dat"
    line = (f"[{rank:2}/{n_top}] {entry['pp']:6.1f}pp "
            f"{entry['stars']:5.2f}* acc {entry['accuracy']:.3f} "
            f"{'FC ' if entry['full_combo'] else ''}"
            f"{entry['song'][:34]!r} -> {out} OK"
            + (f"  [pushes: {n_push} in {tick} ticks, replay score "
               f"{round(walk.score)} vs bench {entry['score']}]"
               if push_prob > 0.0 else "")) + "\n" + verify
    return base, line, warn


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--bench", required=True, help="bench results JSON")
    p.add_argument("--tag", required=True, help="replay filename prefix")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--sort", choices=["pp", "accuracy"], default="pp",
                   help="rank plays by pp (default) or by accuracy (of "
                        "the --board view)")
    p.add_argument("--board", choices=BOARDS, default=None,
                   help="leaderboard view of a --catalog both bench: ss "
                        "(ScoreSaber pp / stars / accuracy, SS-ranked "
                        "levels) or bl (BeatLeader's, BL-ranked levels "
                        "incl. BL-only). Default: ss, or bl on a "
                        "catalog=beatleader bench")
    p.add_argument("--levels", nargs="+", default=None, metavar="KEY",
                   help="record exactly these levels (level_key on a "
                        "--catalog both bench, else leaderboard ids), in "
                        "this order, instead of a top/bottom/random pick; "
                        "label 'sel' unless --label")
    p.add_argument("--bottom", action="store_true",
                   help="take the WORST N by the sort key instead of "
                        "the best (2026-07-29: bottom-acc failure-mode "
                        "review)")
    p.add_argument("--stars", type=float, nargs=2, metavar=("LO", "HI"),
                   default=None,
                   help="only levels with LO <= stars < HI (2026-09-22: "
                        "star-band review sets)")
    p.add_argument("--random", type=int, default=0, metavar="N",
                   help="pick N levels uniformly at random from the "
                        "(star-filtered) bench instead of the top/bottom "
                        "of the sort key; label 'rand'; seeded by --seed")
    p.add_argument("--seed", type=int, default=0,
                   help="--random selection seed")
    p.add_argument("--label", default=None,
                   help="override the filename label (default: derived "
                        "from --sort/--bottom, e.g. toppp/botacc). Used "
                        "by curated selections (2026-08-21 wall-heavy "
                        "set) where the sort key isn't the story")
    p.add_argument("--dat", action="store_true",
                   help="also convert each replay to a ScoreSaber .dat and "
                        "decode-check it (off by default since 2026-10-02: "
                        "the replay site plays the .rpl.gz)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel plays (spawn pool; each worker owns a "
                        "policy + sim). Outputs byte-identical to serial.")
    p.add_argument("--tau-scale", type=str, default=None,
                   help="env v46: fixed per-axis torque-limit scales (16 "
                        "floats, group shorthand, 'default' = the saved "
                        "body TAU_BODY_DEFAULT, 'nominal' = all 1.0; see "
                        "bench_scoresaber --tau-scale). Default: the body "
                        "the bench JSON records (tau_scale), else 'default'")
    p.add_argument("--sim-dtype", choices=["f32", "f64"], default=None,
                   help="Rust sim build. Default: the bench JSON's "
                        "sim_dtype (else f32, the bench default)")
    p.add_argument("--forward-width", type=int, default=None,
                   help="policy forward batch width. Default: the bench "
                        "JSON's forward_width (else 1024)")
    p.add_argument("--tick-hz", type=int, default=None,
                   help="sim control rate (30/60/120/240). Default: the "
                        "bench JSON's tick_hz (else 30)")
    p.add_argument("--action-repeat", type=int, default=None,
                   help="sim ticks per policy decision. Default: the bench "
                        "JSON's action_repeat (else 1)")
    p.add_argument("--push-prob", type=float, default=0.0,
                   help="replay the env v18 perturbation: per-tick "
                        "probability of a horizontal root-velocity kick "
                        "(training default 0.00333; 0 = off). The bench "
                        "verification is skipped-as-warning when > 0")
    p.add_argument("--push-vmax", type=float, default=0.15,
                   help="max kick magnitude m/s (training default 0.15)")
    p.add_argument("--push-seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="forward device (default: cuda if available). "
                        "Matching the bench's device maximizes exact "
                        "reproduction — fp32 argmax near-ties can flip "
                        "across devices/batch shapes (first seen on gen4, "
                        "2026-07-26); mismatches downgrade to a warning "
                        "and the replay verifies self-consistently")
    args = p.parse_args()

    bench = json.load(open(args.bench, encoding="utf-8"))
    # reproduce the bench's regime unless told otherwise (2026-09-26)
    sim_dtype = args.sim_dtype or bench.get("sim_dtype", "f32")
    width = args.forward_width or int(bench.get("forward_width", 1024))
    rate = (args.tick_hz or int(bench.get("tick_hz") or 30),
            args.action_repeat or int(bench.get("action_repeat") or 1))
    if bench.get("obs_window"):
        sys.exit("error: obs_window benches are not reproducible on this "
                 "sim (no obs-window variant)")
    if args.tau_scale is None:
        ts = bench.get("tau_scale")
        args.tau_scale = (",".join(f"{v:g}" for v in ts) if ts else "default")
    board = args.board or ("bl" if bench.get("catalog") == "beatleader"
                           else "ss")
    print(f"regime: sim {sim_dtype}, forward width {width}, tau_scale "
          f"{args.tau_scale}, {rate[0]} Hz, action repeat {rate[1]}"
          + (f", board {board}" if bench.get("catalog") == "both" else ""),
          flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    mhash = model_hash(ck["state_dict"])
    del ck
    if bench.get("model_hash") and mhash != bench["model_hash"]:
        sys.exit(f"error: {args.ckpt} hashes to {mhash}, the bench was run "
                 f"on {bench['model_hash']}")
    pool = bench_pool(bench, board, fallback=bool(args.levels))
    if args.stars is not None:
        lo, hi = args.stars
        pool = [e for e in pool if lo <= float(e.get("stars", 0)) < hi]
        if not pool:
            sys.exit(f"error: no bench levels with {lo} <= stars < {hi}")
    if args.levels:
        top = select_levels(pool, args.levels)
    elif args.random:
        import random
        top = random.Random(args.seed).sample(pool, min(args.random, len(pool)))
        top.sort(key=lambda e: -e["pp"])          # rank order = pp within the pick
    else:
        sgn = 1.0 if args.bottom else -1.0
        top = sorted(pool, key=lambda e: sgn * e[args.sort])[:args.top]
    # label carries the sort key too (2026-07-30, user request): the pp
    # and accuracy "top" sets used to collide on topNN names
    label = args.label or ("sel" if args.levels else
                           "rand" if args.random else
                           ("bot" if args.bottom else "top") + (
                               "acc" if args.sort == "accuracy" else "pp"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # replays/ is gitignored and absent on fresh checkouts (third
    # "output dir assumed to exist" bug of the 2026-08-08 migration)
    os.makedirs("replays", exist_ok=True)
    jobs = [(rank, len(top), entry, args.tag, label, args.dat)
            for rank, entry in enumerate(top, 1)]
    made, failed = [], []
    if args.workers > 1:
        ctx = multiprocessing.get_context("spawn")   # CUDA-safe
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                 mp_context=ctx,
                                 initializer=_worker_init,
                                 initargs=(args.ckpt, device,
                                           (args.push_prob, args.push_vmax,
                                            args.push_seed),
                                           parse_tau_scale(args.tau_scale)
                                           if args.tau_scale else None,
                                           sim_dtype, width, rate,
                                           mhash)) as ex:
            for base, line, warn, err in ex.map(_make_one, jobs):
                if err:
                    print(err, flush=True)
                    failed.append(err.splitlines()[0])
                    continue
                if warn:
                    print(warn, flush=True)
                print(line, flush=True)
                made.append(base)
    else:
        _worker_init(args.ckpt, device,
                     (args.push_prob, args.push_vmax, args.push_seed),
                     parse_tau_scale(args.tau_scale) if args.tau_scale
                     else None, sim_dtype, width, rate, mhash)
        for job in jobs:
            base, line, warn, err = _make_one(job)
            if err:
                print(err, flush=True)
                failed.append(err.splitlines()[0])
                continue
            if warn:
                print(warn, flush=True)
            print(line, flush=True)
            made.append(base)
    print("\nall verified:", len(made))
    if failed:
        print(f"{len(failed)} FAILED:\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
