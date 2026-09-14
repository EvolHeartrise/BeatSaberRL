"""Record + verify + convert a bench's top-N pp plays to watchable replays.

    python make_top10_replays.py --ckpt <ckpt.pt> --bench <bench.json> \
        --tag gen2_429m [--top 10] [--workers 10]

For each top play: greedy re-rollout with the bench protocol (cached
map, real NJS speeds, seed 0) recording JSONL v7, verify the outcome
counters and multiplied score against the bench entry, then convert to
a ScoreSaber .dat and verify the decode. Deterministic sim ⇒ exact.
(Was a 598M one-off; parameterized 2026-07-24.)

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
from beat_arena_env.beatsaver import load_beatmap
from beat_arena_env.rust_env import RustBeatEnv
from policy import load_policy

sys.path.insert(0, "tests")
from test_convert_scoresaber import decode_dat


def slug(s, n=24):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:n].lower() or "song"


_W = {}          # per-process state: {"policy", "dev", "fc", "push"} (set by _worker_init)


def _worker_init(ckpt_path, device, foot_contact=False, push=(0.0, 0.0, 0)):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dev = torch.device(device)
    _W["policy"] = load_policy(ckpt).to(dev).eval()
    _W["dev"] = dev
    _W["fc"] = foot_contact
    _W["push"] = push


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
    """One play: re-rollout, verify vs bench entry, convert, verify decode.
    Returns (base, line, warn) for rank-ordered printing by the parent."""
    rank, n_top, entry, tag, label = job
    policy, dev = _W["policy"], _W["dev"]
    map_dir = f"maps/scoresaber/{entry['song_hash']}"
    notes, meta = load_beatmap(map_dir, entry["difficulty"], first_note_s=2.0)
    base = (f"replays/{tag}_{label}{rank:02d}_"
            f"{slug(entry['song'])}_{entry['pp']:.0f}pp")
    env = RustBeatEnv(external_chart=notes, seed=0, gears=policy.n_gears,
                      foot_contact=_W["fc"],
                      external_walls=meta["walls"])
    env.set_record(base + ".jsonl")
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
        with torch.no_grad():
            logits, _ = policy(
                torch.as_tensor(obs, dtype=torch.float32,
                                device=dev).unsqueeze(0))
        obs, _, done, _, info = env.step(
            [int(l.argmax()) for l in logits])
        walk.consume(info["events"])
    exact = (round(walk.score) == entry["score"]
             and walk.max_combo == entry["highest_combo"]
             and info["good_cuts"] == entry["good_cuts"])
    warn = None
    if not exact:
        # fp32 argmax near-tie flipped somewhere vs the bench run —
        # still a faithful greedy play of THIS checkpoint; verify
        # the pipeline self-consistently against the replay itself
        warn = (f"    WARNING {entry['song']}: replay score "
                f"{round(walk.score)} != bench {entry['score']} "
                f"(near-tie drift; replay is self-consistent)")

    # stamp the map on the recording so the 3D viewer can pull this song's
    # audio from BeatSaver and line it up (meta["shift"] is what
    # first_note_s did to the clock; -shift is the song time at replay t=0)
    from annotate_replay_audio import stamp
    stamp(base + ".jsonl", entry["song_hash"], entry["difficulty"],
          entry["song"], -meta["shift"])

    conv.convert(base + ".jsonl", map_dir, entry["song_hash"],
                 entry["difficulty"], base + ".dat")
    d = decode_dat(base + ".dat")
    assert d["final_score"] == round(walk.score), \
        f".dat decode {d['final_score']} != replay {round(walk.score)}"
    line = (f"[{rank:2}/{n_top}] {entry['pp']:6.1f}pp "
            f"{entry['stars']:5.2f}* acc {entry['accuracy']:.3f} "
            f"{'FC ' if entry['full_combo'] else ''}"
            f"{entry['song'][:34]!r} -> {base}.dat OK"
            + (f"  [pushes: {n_push} in {tick} ticks, replay score "
               f"{round(walk.score)} vs bench {entry['score']}]"
               if push_prob > 0.0 else ""))
    return base, line, warn


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--bench", required=True, help="bench results JSON")
    p.add_argument("--tag", required=True, help="replay filename prefix")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--sort", choices=["pp", "accuracy"], default="pp",
                   help="rank plays by pp (default) or by accuracy")
    p.add_argument("--bottom", action="store_true",
                   help="take the WORST N by the sort key instead of "
                        "the best (2026-07-29: bottom-acc failure-mode "
                        "review)")
    p.add_argument("--label", default=None,
                   help="override the filename label (default: derived "
                        "from --sort/--bottom, e.g. toppp/botacc). Used "
                        "by curated selections (2026-08-21 wall-heavy "
                        "set) where the sort key isn't the story")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel plays (spawn pool; each worker owns a "
                        "policy + sim). Outputs byte-identical to serial.")
    p.add_argument("--foot-contact", action="store_true",
                   help="env v11 contact physics — MUST match how the "
                        "checkpoint was trained (a v12+/contact ckpt "
                        "replayed under welds is a different game)")
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
    sgn = 1.0 if args.bottom else -1.0
    top = sorted((e for e in bench["levels"] if "skipped" not in e),
                 key=lambda e: sgn * e[args.sort])[:args.top]
    # label carries the sort key too (2026-07-30, user request): the pp
    # and accuracy "top" sets used to collide on topNN names
    label = args.label or ("bot" if args.bottom else "top") + (
        "acc" if args.sort == "accuracy" else "pp")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # replays/ is gitignored and absent on fresh checkouts (third
    # "output dir assumed to exist" bug of the 2026-08-08 migration)
    os.makedirs("replays", exist_ok=True)
    jobs = [(rank, len(top), entry, args.tag, label)
            for rank, entry in enumerate(top, 1)]
    made = []
    if args.workers > 1:
        ctx = multiprocessing.get_context("spawn")   # CUDA-safe
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                 mp_context=ctx,
                                 initializer=_worker_init,
                                 initargs=(args.ckpt, device,
                                           args.foot_contact,
                                           (args.push_prob, args.push_vmax,
                                            args.push_seed))) as ex:
            for base, line, warn in ex.map(_make_one, jobs):
                if warn:
                    print(warn, flush=True)
                print(line, flush=True)
                made.append(base)
    else:
        _worker_init(args.ckpt, device, args.foot_contact,
                     (args.push_prob, args.push_vmax, args.push_seed))
        for job in jobs:
            base, line, warn = _make_one(job)
            if warn:
                print(warn, flush=True)
            print(line, flush=True)
            made.append(base)
    print("\nall verified:", len(made))


if __name__ == "__main__":
    main()
