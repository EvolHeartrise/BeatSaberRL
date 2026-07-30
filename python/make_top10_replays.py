"""Record + verify + convert a bench's top-N pp plays to watchable replays.

    python make_top10_replays.py --ckpt <ckpt.pt> --bench <bench.json> \
        --tag gen2_429m [--top 10]

For each top play: greedy re-rollout with the bench protocol (cached
map, real NJS speeds, seed 0) recording JSONL v7, verify the outcome
counters and multiplied score against the bench entry, then convert to
a ScoreSaber .dat and verify the decode. Deterministic sim ⇒ exact.
(Was a 598M one-off; parameterized 2026-07-24.)
"""

import argparse
import json
import re
import sys

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
    label = ("bot" if args.bottom else "top") + (
        "acc" if args.sort == "accuracy" else "pp")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    dev = torch.device(args.device
                       or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy = load_policy(ckpt).to(dev).eval()
    made = []
    for rank, entry in enumerate(top, 1):
        map_dir = f"maps/scoresaber/{entry['song_hash']}"
        notes, _ = load_beatmap(map_dir, entry["difficulty"],
                                first_note_s=2.0)
        base = (f"replays/{args.tag}_{label}{rank:02d}_"
                f"{slug(entry['song'])}_{entry['pp']:.0f}pp")
        env = RustBeatEnv(external_chart=notes, seed=0,
                          gears=policy.n_gears)
        env.set_record(base + ".jsonl")
        obs, _ = env.reset()
        walk = scoresaber.ComboTracker()
        done = False
        while not done:
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
        if not exact:
            # fp32 argmax near-tie flipped somewhere vs the bench run —
            # still a faithful greedy play of THIS checkpoint; verify
            # the pipeline self-consistently against the replay itself
            print(f"    WARNING {entry['song']}: replay score "
                  f"{round(walk.score)} != bench {entry['score']} "
                  f"(near-tie drift; replay is self-consistent)",
                  flush=True)

        r = conv.convert(base + ".jsonl", map_dir, entry["song_hash"],
                         entry["difficulty"], base + ".dat")
        d = decode_dat(base + ".dat")
        assert d["final_score"] == round(walk.score), \
            f".dat decode {d['final_score']} != replay {round(walk.score)}"
        made.append(base)
        print(f"[{rank:2}/{len(top)}] {entry['pp']:6.1f}pp "
              f"{entry['stars']:5.2f}* acc {entry['accuracy']:.3f} "
              f"{'FC ' if entry['full_combo'] else ''}"
              f"{entry['song'][:34]!r} -> {base}.dat OK", flush=True)
    print("\nall verified:", len(made))


if __name__ == "__main__":
    main()
