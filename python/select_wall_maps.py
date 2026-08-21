"""Pick the N bench levels richest in BOTH dodge and duck walls, and
write a filtered bench JSON for make_top10_replays.py --label walls.

Classifier on the parsed world-space AABBs (chart.Wall, geometry
finished by the beatsaver parser): a DUCK wall's bottom edge
(cy - height/2) sits at/above 0.9 m — the player must crouch under
it; everything else forces a lateral DODGE. Ranking: primarily by
min(n_dodge, n_duck) so both kinds are abundant, then total walls.

    python select_wall_maps.py --bench bench/v32_leaf04500_cd800.json \
        --out /tmp/v32_wallmaps.json --top 5
"""
import argparse
import json

from beat_arena_env.beatsaver import load_beatmap

DUCK_BOTTOM_M = 0.9

p = argparse.ArgumentParser()
p.add_argument("--bench", required=True)
p.add_argument("--out", required=True)
p.add_argument("--top", type=int, default=5)
args = p.parse_args()

bench = json.load(open(args.bench, encoding="utf-8"))
scored = []
for e in bench["levels"]:
    if "skipped" in e:
        continue
    _, meta = load_beatmap(f"maps/scoresaber/{e['song_hash']}",
                           e["difficulty"], first_note_s=2.0)
    duck = sum(1 for w in meta["walls"]
               if w.cy - w.height / 2 >= DUCK_BOTTOM_M)
    dodge = len(meta["walls"]) - duck
    scored.append((min(dodge, duck), dodge + duck, dodge, duck, e))

scored.sort(key=lambda t: (-t[0], -t[1]))
pick = scored[:args.top]
for rank, (_, total, dodge, duck, e) in enumerate(pick, 1):
    print(f"[{rank}] {e['song'][:40]!r} walls={total} "
          f"(dodge {dodge} / duck {duck}) pp={e['pp']:.0f} "
          f"acc={e['accuracy']:.3f}", flush=True)

json.dump({"levels": [e for *_ns, e in pick]}, open(args.out, "w"))
print(f"wrote {args.out} ({len(pick)} levels)")
