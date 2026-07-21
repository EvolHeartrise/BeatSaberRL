"""Record replay JSONL episodes from a checkpoint, the scripted bot, or random.

Watch them in viz/replay.html (open in a browser, drop the .jsonl in).
All episodes run on the Rust sim via RustBeatEnv (JSONL v2 recording is
built from the sim's event stream). Recorded with shaping_coef=0 so the
logged rewards are the raw interpretable ones (+1 / -0.25 / -0.5).

  python record_replay.py --ckpt runs/pure_tf_batch_4m/final.pt --episodes 3
  python record_replay.py --scripted --note-rate 2.5 --colors both --dirs 0 1 2 3 4 5 6 7
  python record_replay.py --random
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from beat_arena_env import RustBeatEnv, scripted_action
from policy import NotePolicy


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=str, help="checkpoint .pt to play")
    src.add_argument("--scripted", action="store_true", help="play the scripted bot")
    src.add_argument("--random", action="store_true", help="play uniform-random actions")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=7000, help="base episode seed")
    p.add_argument("--out-dir", type=str, default="replays")
    p.add_argument("--tag", type=str, default=None,
                   help="filename prefix (default: run-dir name / scripted / random)")
    p.add_argument("--note-rate", type=float, default=1.0)
    p.add_argument("--colors", choices=["right", "both"], default="right")
    p.add_argument("--dirs", type=int, nargs="+", default=[6])
    p.add_argument("--cols", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--chart-len", type=float, default=20.0)
    p.add_argument("--min-hand-gap", type=float, default=0.5)
    p.add_argument("--note-speed", type=float, default=4.0,
                   help="approach speed; MIN of the range if --note-speed-max set")
    p.add_argument("--note-speed-max", type=float, default=None)
    p.add_argument("--continuous-dirs", action="store_true",
                   help="angles ~ U(0, 2pi) instead of the --dirs subset")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    kw = dict(note_rate=args.note_rate, colors=args.colors, dirs=tuple(args.dirs),
              cols=tuple(args.cols), rows=tuple(args.rows),
              chart_len_s=args.chart_len, min_hand_gap_s=args.min_hand_gap,
              note_speed=args.note_speed, note_speed_max=args.note_speed_max,
              continuous_dirs=args.continuous_dirs, shaping_coef=0.0)

    policy = None
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        policy = NotePolicy()
        policy.load_state_dict(ckpt["state_dict"])
        policy.eval()
        tag = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
        print(f"loaded {args.ckpt} (update {ckpt.get('update', '?')})")
    else:
        tag = "scripted" if args.scripted else "random"
    if args.tag:
        tag = args.tag

    rng = np.random.default_rng(args.seed)
    env = RustBeatEnv(seed=args.seed, **kw)
    for ep in range(args.episodes):
        seed = args.seed + ep
        path = os.path.join(args.out_dir, f"{tag}_ep{ep:02d}_seed{seed}.jsonl")
        env.set_record(path)
        obs, _ = env.reset(seed=seed)
        done, tot = False, 0.0
        while not done:
            if args.scripted:
                act = scripted_action(env)
            elif policy is not None:
                with torch.no_grad():
                    a, _, _, _ = policy.get_action_and_value(
                        torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                act = a[0].numpy()
            else:
                act = rng.integers(0, 9, size=2)
            obs, r, done, _, info = env.step(act)
            tot += r
        print(f"  {path}: return {tot:6.2f} | good {info['good_cuts']}/{info['n_notes']} "
              f"bad {info['bad_cuts']} miss {info['misses']}")


if __name__ == "__main__":
    main()
