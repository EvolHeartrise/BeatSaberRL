"""Sight-read a real BeatSaver map with a trained policy (or bot/random).

  python sightread.py --map maps/spectre.zip --difficulty Normal \\
      --ckpt runs/dynamic_19m/final.pt --replay replays/spectre_normal.jsonl

Prints the parsed map's stats, then the run's score breakdown. The whole
song is one episode (same chart every reset). Rewards are the sim's raw
scoring (+1 / -0.5). Replays drop into viz/replay.html as usual.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch

from beat_arena_env import RustBeatEnv, scripted_action
from beat_arena_env.beatsaver import list_difficulties, load_beatmap
from policy import load_policy


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--map", required=True, help="map .zip or directory")
    p.add_argument("--difficulty", default="Normal")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=str)
    src.add_argument("--scripted", action="store_true")
    src.add_argument("--random", action="store_true")
    p.add_argument("--replay", type=str, default=None, help="record JSONL here")
    p.add_argument("--episodes", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    notes, meta = load_beatmap(args.map, args.difficulty, first_note_s=2.0)
    print(f"map: {meta['song_name']} [{meta['difficulty']}]  "
          f"bpm {meta['bpm']}  njs {meta['njs']} -> {meta['speed']:.2f} m/s  "
          f"{meta['n_notes']} notes + {meta['n_bombs']} bombs over "
          f"{meta['duration_s']:.0f}s "
          f"({meta['nps']:.2f} nps)  dropped {meta['dropped']}")
    print(f"difficulties available: {[d[0] for d in list_difficulties(args.map)]}")

    policy = None
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        policy = load_policy(ckpt)
        policy.eval()

    env = RustBeatEnv(external_chart=notes, seed=1,
                      gears=policy.n_gears if policy is not None else 3)
    if args.replay:
        env.set_record(args.replay)
    rng = np.random.default_rng(0)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        reasons = Counter()
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
                act = rng.integers(0, [3] * 12)
            obs, r, done, _, info = env.step(act)
            tot += r
            for _, reason, _pts in info["events"]:
                if reason != "post":   # follow-throughs aren't note outcomes
                    reasons[reason] += 1
        pct = 100 * reasons["ok"] / max(info["n_notes"], 1)
        print(f"ep{ep}: return {tot:7.2f} | good {reasons['ok']}/{info['n_notes']} "
              f"({pct:.0f}%) | score {info['score'] / max(info['n_notes'], 1):.1f}/115 "
              f"| {dict(reasons)}")
        if args.replay:
            env._record_path = None  # only record the first episode


if __name__ == "__main__":
    main()
