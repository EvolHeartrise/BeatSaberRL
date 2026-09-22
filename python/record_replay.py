"""Record replay JSONL episodes from the scripted bot or random actions.

Watch them in viz/replay.html (open in a browser, drop the .jsonl in).
All episodes run on the Rust sim via RustBeatEnv (JSONL v4 recording is
built from the sim's event stream). Logged rewards are the raw
interpretable ones (score/115 per good cut, -0.5 bad/miss).

Procedural charts, default body. To replay a trained checkpoint use
make_top10_replays.py, which sets up the production physics (tau body,
asinh z) and plays real maps.

  python record_replay.py --scripted --note-prob 0.35
  python record_replay.py --random
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from beat_arena_env import RustBeatEnv, scripted_action


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scripted", action="store_true", help="play the scripted bot")
    src.add_argument("--random", action="store_true", help="play uniform-random actions")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=7000, help="base episode seed")
    p.add_argument("--out-dir", type=str, default="replays")
    p.add_argument("--tag", type=str, default=None,
                   help="filename prefix (default: scripted / random)")
    p.add_argument("--cols", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--chart-len", type=float, default=20.0)
    p.add_argument("--note-speed-min", type=float, default=2.0)
    p.add_argument("--note-speed-max", type=float, default=8.0)
    p.add_argument("--note-prob", type=float, default=0.25)
    p.add_argument("--extra-prob", type=float, default=0.125)
    p.add_argument("--dot-prob", type=float, default=0.1)
    p.add_argument("--bomb-prob", type=float, default=0.125)
    p.add_argument("--gears", type=int, default=5, choices=(3, 5, 7),
                   help="action-table size")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    kw = dict(cols=tuple(args.cols),
              rows=tuple(args.rows), chart_len_s=args.chart_len,
              note_speed_min=args.note_speed_min,
              note_speed_max=args.note_speed_max,
              note_prob=args.note_prob, extra_prob=args.extra_prob,
              dot_prob=args.dot_prob, bomb_prob=args.bomb_prob,
              gears=args.gears)
    tag = args.tag or ("scripted" if args.scripted else "random")

    rng = np.random.default_rng(args.seed)
    env = RustBeatEnv(seed=args.seed, foot_contact=True, **kw)
    for ep in range(args.episodes):
        seed = args.seed + ep
        path = os.path.join(args.out_dir, f"{tag}_ep{ep:02d}_seed{seed}.jsonl")
        env.set_record(path)
        env.reset(seed=seed)
        done, tot = False, 0.0
        while not done:
            if args.scripted:
                act = scripted_action(env)
            else:
                # v10: 29 gear heads bounded by the env's action table
                act = rng.integers(0, env.gears, 29)
            _, r, done, _, info = env.step(act)
            tot += r
        print(f"  {path}: return {tot:6.2f} | good {info['good_cuts']}/{info['n_notes']} "
              f"bad {info['bad_cuts']} miss {info['misses']} "
              f"bombs {info['bombs_hit']} | "
              f"score {info['score'] / max(info['n_notes'], 1):.1f}/115")


if __name__ == "__main__":
    main()
