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
    p.add_argument("--colors", default="both")
    p.add_argument("--note-speed-min", type=float, default=2.0)
    p.add_argument("--note-speed-max", type=float, default=8.0)
    p.add_argument("--note-prob", type=float, default=0.25)
    p.add_argument("--extra-prob", type=float, default=0.125)
    p.add_argument("--dot-prob", type=float, default=0.1)
    p.add_argument("--bomb-prob", type=float, default=0.125)
    p.add_argument("--effort-coef", type=float, default=1.0)
    args = p.parse_args()

    cfg = ChartConfig(colors=args.colors,
                      note_speed_min=args.note_speed_min,
                      note_speed_max=args.note_speed_max,
                      note_prob=args.note_prob, extra_prob=args.extra_prob,
                      dot_prob=args.dot_prob, bomb_prob=args.bomb_prob)
    pol = load_policy(torch.load(args.ckpt, map_location="cpu",
                                 weights_only=True))
    pol.eval()
    G = B = M = BOMB = 0
    score = notes = 0.0
    pre = acc = post = 0.0
    rets = []
    for ep in range(args.episodes):
        rng = np.random.default_rng(args.seed0 + ep)
        env = RustBeatEnv(external_chart=generate_chart(cfg, rng),
                          effort_coef=args.effort_coef, seed=args.seed0 + ep)
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
        pre += info["score_pre"]
        acc += info["score_acc"]
        post += info["score_post"]
        notes += info["n_notes"]
        rets.append(ret)
    tot = G + B + M
    print(f"{args.ckpt}: good {100 * G / tot:.1f}%  bad {B}  miss {M}  "
          f"bombs_hit {BOMB}  score {score / notes:.1f}/115  "
          f"return {np.mean(rets):+.2f}", flush=True)
    # score decomposition (per scoreable object; link flat-20s are the
    # remainder): where the points actually come from
    print(f"  components/note: pre {pre / notes:.1f}/70  "
          f"acc {acc / notes:.1f}/15  post {post / notes:.1f}/30", flush=True)


if __name__ == "__main__":
    main()
