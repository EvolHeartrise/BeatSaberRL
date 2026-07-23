"""30-episode greedy eval of a checkpoint (defaults = the dynamic-gym config).

Prints one line: ckpt path, good-cut %, bad/miss counts, mean unshaped return.
Same protocol as the session studies: charts + env seeded 9000..9000+N-1.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from beat_arena_env import RustBeatEnv
from beat_arena_env.chart import ChartConfig, generate_chart
from policy import load_policy


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed0", type=int, default=9000)
    p.add_argument("--note-rate", type=float, default=2.5)
    p.add_argument("--colors", default="both")
    p.add_argument("--discrete-dirs", action="store_true",
                   help="use --dirs instead of continuous angles")
    p.add_argument("--dirs", type=int, nargs="+", default=[6])
    p.add_argument("--note-speed", type=float, default=3.0)
    p.add_argument("--note-speed-max", type=float, default=6.0)
    p.add_argument("--bomb-rate", type=float, default=0.25)
    p.add_argument("--chain-prob", type=float, default=0.0)
    p.add_argument("--arc-prob", type=float, default=0.0)
    args = p.parse_args()

    cfg = ChartConfig(note_rate=args.note_rate, colors=args.colors,
                      continuous_dirs=not args.discrete_dirs,
                      dirs=tuple(args.dirs), note_speed=args.note_speed,
                      note_speed_max=args.note_speed_max,
                      bomb_rate=args.bomb_rate, chain_prob=args.chain_prob,
                      arc_prob=args.arc_prob)
    pol = load_policy(torch.load(args.ckpt, map_location="cpu",
                                 weights_only=True))
    pol.eval()
    G = B = M = BOMB = 0
    score = notes = 0.0
    rets = []
    for ep in range(args.episodes):
        rng = np.random.default_rng(args.seed0 + ep)
        env = RustBeatEnv(external_chart=generate_chart(cfg, rng),
                          seed=args.seed0 + ep)
        obs, _ = env.reset()
        done, ret = False, 0.0
        while not done:
            with torch.no_grad():
                logits, _ = pol(torch.as_tensor(obs, dtype=torch.float32
                                                ).unsqueeze(0))
            obs, rew, done, _, info = env.step(
                tuple(int(l.argmax()) for l in logits))
            ret += rew
            for _, reason, _pts in info["events"]:
                if reason == "ok":
                    G += 1
                elif reason == "miss":
                    M += 1
                elif reason == "bomb":
                    BOMB += 1
                elif reason != "post":
                    B += 1
        score += info["score"]
        notes += info["n_notes"]
        rets.append(ret)
    tot = G + B + M
    print(f"{args.ckpt}: good {100 * G / tot:.1f}%  bad {B}  miss {M}  "
          f"bombs_hit {BOMB}  score {score / notes:.1f}/115  "
          f"return {np.mean(rets):+.2f}", flush=True)


if __name__ == "__main__":
    main()
