"""One-off: record + verify + convert the 598M bench's top-10 pp plays.

For each top play: greedy re-rollout with the bench protocol (cached
map, speed_scale 0.3, seed 0) recording JSONL v7, verify the outcome
counters and multiplied score against the bench entry, then convert to
a ScoreSaber .dat and verify the decode. Deterministic sim ⇒ exact.
"""

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

CKPT = "runs/v77_wide_e4cos_700m_cont/ckpt_upd0950.pt"
BENCH = "bench/v77_wide_e4cos_700m_cont_ckpt_upd0950.json"


def slug(s, n=24):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:n].lower() or "song"


def main():
    bench = json.load(open(BENCH, encoding="utf-8"))
    top = sorted((e for e in bench["levels"] if "skipped" not in e),
                 key=lambda e: -e["pp"])[:10]
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    policy = load_policy(ckpt).eval()
    made = []
    for rank, entry in enumerate(top, 1):
        map_dir = f"maps/scoresaber/{entry['song_hash']}"
        notes, _ = load_beatmap(map_dir, entry["difficulty"],
                                speed_scale=0.3, first_note_s=2.0)
        base = (f"replays/v77_598m_top{rank:02d}_"
                f"{slug(entry['song'])}_{entry['pp']:.0f}pp")
        env = RustBeatEnv(external_chart=notes, seed=0)
        env.set_record(base + ".jsonl")
        obs, _ = env.reset()
        walk = scoresaber.ComboTracker()
        done = False
        while not done:
            with torch.no_grad():
                logits, _ = policy(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            obs, _, done, _, info = env.step(
                [int(l.argmax()) for l in logits])
            walk.consume(info["events"])
        assert round(walk.score) == entry["score"], \
            f"{entry['song']}: {round(walk.score)} != bench {entry['score']}"
        assert walk.max_combo == entry["highest_combo"]
        assert info["good_cuts"] == entry["good_cuts"]

        r = conv.convert(base + ".jsonl", map_dir, entry["song_hash"],
                         entry["difficulty"], base + ".dat")
        d = decode_dat(base + ".dat")
        assert d["final_score"] == entry["score"]
        made.append(base)
        print(f"[{rank:2}/10] {entry['pp']:6.1f}pp {entry['stars']:5.2f}* "
              f"acc {entry['accuracy']:.3f} "
              f"{'FC ' if entry['full_combo'] else ''}"
              f"{entry['song'][:34]!r} -> {base}.dat OK", flush=True)
    print("\nall verified:", len(made))


if __name__ == "__main__":
    main()
