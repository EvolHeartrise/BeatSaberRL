"""Stamp a recorded replay with the map it was played on, so the viewer can
fetch that song's audio from BeatSaver and play it in sync.

    python annotate_replay_audio.py replays/*.jsonl
    python annotate_replay_audio.py r.jsonl --hash <sha1> --difficulty Expert

Adds ONE additive header field (nothing existing is touched, so an
annotated replay still loads everywhere a bare one does):

    "map": {"hash", "difficulty", "song", "audio_offset_s"}

`audio_offset_s` is the song time that replay t=0 corresponds to. Recordings
re-base the chart so the first note lands at first_note_s (2.0 s by
convention), which throws the song's own clock away — `load_beatmap` now
reports that shift, and audio_offset_s = -shift puts it back.

Why this can be run over replays recorded before the field existed: the
header carries the whole chart, so a candidate (map, difficulty) can be
PROVEN rather than guessed. The candidate is re-parsed at the recording
convention and its note times compared against the header's; only an exact
match is written. That verifies the map, the difficulty and the time base in
one step, so a wrong guess cannot silently desync the audio — the failure
mode is "not annotated", never "annotated wrongly".

Candidates come from the ScoreSaber catalog by song slug (the same slug
make_top10_replays puts in the filename), or from --hash/--difficulty.
Maps must already be in maps/scoresaber/<hash>; prefetch_maps.py fills that.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

MAPS = os.path.join(BASE, "maps")
CATALOG = os.path.join(MAPS, "scoresaber_catalog.json")
MAP_CACHE = os.path.join(MAPS, "scoresaber")

FIRST_NOTE_S = 2.0   # make_top10_replays' recording convention
T_TOL = 1e-6         # note times are written at full float precision


def slug(s, n=24):
    """make_top10_replays.slug, verbatim — filenames are matched with it."""
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:n].lower() or "song"


def slug_of_replay(path):
    """The song slug out of `{tag}_{label}{rank}_{slug}_{pp}pp.jsonl`."""
    stem = os.path.basename(path).rsplit(".", 1)[0]
    stem = re.sub(r"_\d+pp$", "", stem)          # drop the pp suffix
    return stem.rsplit("_", 1)[-1] if "_" in stem else None


def chart_times(header):
    return [float(n["t"]) for n in header["chart"]]


def verify(map_dir, difficulty, times):
    """Parse (map, difficulty) at the recording convention; return the
    audio offset if its chart IS the replay's chart, else None."""
    from beat_arena_env.beatsaver import load_beatmap
    try:
        notes, meta = load_beatmap(map_dir, difficulty,
                                   first_note_s=FIRST_NOTE_S)
    except Exception:
        return None
    if len(notes) != len(times):
        return None
    for n, t in zip(notes, times):
        if abs(n.hit_time - t) > T_TOL:
            return None
    return -float(meta["shift"]), meta["song_name"]


def candidates(path, args):
    """(hash, difficulty, song) tuples worth verifying for this replay."""
    if args.hash:
        return [(args.hash, args.difficulty, None)] if args.difficulty else [
            (args.hash, d, None)
            for d in ("ExpertPlusStandard", "ExpertStandard", "HardStandard",
                      "NormalStandard", "EasyStandard")]
    want = slug_of_replay(path)
    if not want:
        return []
    with open(CATALOG) as f:
        cat = json.load(f)
    return [(e["hash"], e["difficulty"], e["song"])
            for e in cat if slug(e["song"]) == want]


def stamp(path, map_hash, difficulty, song, audio_offset_s, extra=None):
    """Write the additive header.map field into a recorded replay, in place.

    Streams the body rather than slurping it: a long recording's tick lines
    run to tens of MB and only the first line changes. Recorders call this
    directly — they already know the map, so nothing needs proving there.
    `extra`: more additive header fields written in the same pass
    (make_top10_replays: the weights hash and the level identity)."""
    tmp = path + ".tmp"
    with open(path) as src, open(tmp, "w") as dst:
        head = json.loads(src.readline())
        head["header"]["map"] = {
            "hash": map_hash,
            "difficulty": difficulty,
            "song": song,
            "audio_offset_s": float(audio_offset_s),
        }
        head["header"].update(extra or {})
        dst.write(json.dumps(head) + "\n")
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
    os.replace(tmp, path)


def annotate(path, args):
    with open(path) as f:
        first = f.readline()
    if not first.strip():
        return "empty"
    head = json.loads(first)
    if "map" in head.get("header", {}) and not args.force:
        return f"already annotated ({head['header']['map']['hash'][:8]})"
    times = chart_times(head["header"])
    if not times:
        return "no chart"

    tried = 0
    for h, diff, song in candidates(path, args):
        map_dir = os.path.join(MAP_CACHE, h)
        if not os.path.isdir(map_dir):
            continue
        tried += 1
        # difficulty names in the catalog are bare ("Expert"); the map dirs
        # hold "<Difficulty>Standard" files and list_difficulties reports
        # those names, so try both spellings
        for d in (diff, f"{diff}Standard", diff.replace("Standard", "")):
            got = verify(map_dir, d, times)
            if got is None:
                continue
            offset, song_name = got
            stamp(path, h, d, song or song_name, offset)
            return (f"OK {song or song_name!r} [{d}] {h[:8]} "
                    f"offset {offset:+.3f}s")
    return f"no map matched this chart ({tried} candidate(s) parsed)"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("replays", nargs="+", help="replay .jsonl paths / globs")
    p.add_argument("--hash", help="skip the catalog search, use this map")
    p.add_argument("--difficulty", help="with --hash: the difficulty to try")
    p.add_argument("--force", action="store_true",
                   help="re-annotate replays that already carry a map field")
    args = p.parse_args()

    paths = [q for pat in args.replays for q in sorted(glob.glob(pat))] \
        or args.replays
    ok = 0
    for path in paths:
        msg = annotate(path, args)
        ok += msg.startswith("OK")
        print(f"{os.path.basename(path)}: {msg}")
    print(f"\n{ok}/{len(paths)} annotated")


if __name__ == "__main__":
    main()
