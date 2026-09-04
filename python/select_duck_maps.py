"""Pick the bench levels richest in FORCED-DUCK walls and write a
filtered bench JSON for make_top10_replays.py.

Stricter than select_wall_maps.py, which ranks by having both dodge and
duck walls. Here we want walls the policy CANNOT sidestep: the wall
spans the whole playable corridor (the middle two columns, x in
(-0.6, 0.6) after the v36 outer-column drop), its bottom edge is high
enough to crouch under, and its top edge is above the resting head, so
standing upright is a hit. Crouching is the only out.

    forced-duck wall:
        cx - w/2 <= -0.6 + tol      spans to the left column edge
        cx + w/2 >=  0.6 - tol      spans to the right column edge
        cy - h/2 >= DUCK_BOTTOM_M   crouch clears it (0.9 m)
        cy + h/2 >= HEAD_STAND_M    standing does NOT clear it (1.55 m)

Measured resting head height is ~1.567 m (2026-09-04 replay study), and
the head is a POINT in the env (no radius), so HEAD_STAND_M is set just
below it.

    python select_duck_maps.py --bench bench/<run>.json \
        --out /root/duckmaps.json --top 8
"""
import argparse
import json

from beat_arena_env.beatsaver import load_beatmap

DUCK_BOTTOM_M = 0.9    # crouch clears a wall whose bottom is at/above this
HEAD_STAND_M = 1.55    # resting head ~1.567 m; below this = standing clears
CORRIDOR_HALF = 0.6    # middle two columns span x in (-0.6, 0.6)
TOL = 1e-6


def classify(walls):
    forced = spanning = duckable = 0
    for w in walls:
        bottom = w.cy - w.height / 2
        top = w.cy + w.height / 2
        half = abs(w.width) / 2
        spans = (w.cx - half <= -CORRIDOR_HALF + TOL
                 and w.cx + half >= CORRIDOR_HALF - TOL)
        can_duck = bottom >= DUCK_BOTTOM_M
        hits_standing = top >= HEAD_STAND_M
        spanning += spans
        duckable += can_duck
        forced += spans and can_duck and hits_standing
    return forced, spanning, duckable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--min-forced", type=int, default=1,
                   help="skip levels with fewer forced-duck walls")
    args = p.parse_args()

    bench = json.load(open(args.bench, encoding="utf-8"))
    scored = []
    for e in bench["levels"]:
        if "skipped" in e:
            continue
        try:
            _, meta = load_beatmap(f"maps/scoresaber/{e['song_hash']}",
                                   e["difficulty"], first_note_s=2.0)
        except Exception:
            continue
        forced, spanning, duckable = classify(meta["walls"])
        if forced >= args.min_forced:
            scored.append((forced, spanning, duckable, len(meta["walls"]), e))

    scored.sort(key=lambda t: (-t[0], -t[1]))
    pick, seen = [], set()
    for t in scored:
        if t[4]["song_hash"] in seen:
            continue
        seen.add(t[4]["song_hash"])
        pick.append(t)
        if len(pick) == args.top:
            break

    for rank, (forced, spanning, duckable, total, e) in enumerate(pick, 1):
        print(f"[{rank}] {e['song'][:38]!r} [{e['difficulty']}] "
              f"forced-duck={forced} spanning={spanning} duckable={duckable} "
              f"walls={total} pp={e['pp']:.0f} acc={e['accuracy']:.3f}",
              flush=True)

    json.dump({"levels": [e for *_ns, e in pick]},
              open(args.out, "w", encoding="utf-8"))
    print(f"wrote {args.out} ({len(pick)} levels); "
          f"{len(scored)} of {len(bench['levels'])} levels had >= "
          f"{args.min_forced} forced-duck wall(s)")


if __name__ == "__main__":
    main()
