"""BeatSaver map parser: real Beat Saber charts -> beat-arena Notes.

Reads a map directory or .zip (as downloaded from beatsaver.com):
Info.dat / info.dat for BPM + difficulty file names + NJS, then the
difficulty .dat in either schema:

- v2 ("2.x"): `_notes: [{_time, _lineIndex, _lineLayer, _type, _cutDirection}]`
- v3 ("3.x"): `colorNotes: [{b, x, y, c, d, a}]` (a = precision angle
  offset in degrees, CCW — flows straight into our continuous angles)

Grid conventions match ours exactly (x/lineIndex 0-3 left->right,
y/lineLayer 0-2 bottom->top). Cut direction is the direction you SWING;
direction 8 is a dot note -> `any_dir`. Times convert beats->seconds via
the map BPM. Per-note speed = NJS * speed_scale: real note-jump speeds
(10-19+ m/s) would give sub-tick cut slabs at 30 Hz, so we rescale into
the sim's trained regime (default 0.3 -> NJS 10..19 becomes 3..5.7 m/s).

NOT parsed (yet): bombs, walls/obstacles, arcs/sliders, burst sliders
(chains), BPM change events, 360/90 rotation events.
"""

from __future__ import annotations

import json
import math
import os
import zipfile

from .chart import Note

# cutDirection -> swing angle (radians; 0 = +x, CCW, y up)
_DIR_ANGLE = {
    0: math.pi / 2,        # Up
    1: 3 * math.pi / 2,    # Down
    2: math.pi,            # Left
    3: 0.0,                # Right
    4: 3 * math.pi / 4,    # UpLeft
    5: math.pi / 4,        # UpRight
    6: 5 * math.pi / 4,    # DownLeft
    7: 7 * math.pi / 4,    # DownRight
}


def _read_json(root, name):
    if isinstance(root, zipfile.ZipFile):
        names = {n.lower(): n for n in root.namelist()}
        return json.loads(root.read(names[name.lower()]))
    for cand in (name, name.lower(), name.capitalize()):
        path = os.path.join(root, cand)
        if os.path.exists(path):
            return json.load(open(path))
    raise FileNotFoundError(f"{name} not found in {root}")


def list_difficulties(map_path):
    """[(difficulty_name, filename, njs), ...] for the Standard characteristic."""
    root = zipfile.ZipFile(map_path) if str(map_path).endswith(".zip") else map_path
    info = _read_json(root, "Info.dat")
    out = []
    for s in info["_difficultyBeatmapSets"]:
        if s["_beatmapCharacteristicName"] != "Standard":
            continue
        for d in s["_difficultyBeatmaps"]:
            out.append((d["_difficulty"], d["_beatmapFilename"],
                        float(d["_noteJumpMovementSpeed"])))
    return out


def load_beatmap(map_path, difficulty="Normal", speed_scale=0.3,
                 first_note_s=None):
    """Parse one difficulty of a BeatSaver map -> (notes, meta dict).

    map_path: map directory or .zip. first_note_s: if set, shift the whole
    chart so the first note lands there (keeps our warm-up convention).
    """
    root = zipfile.ZipFile(map_path) if str(map_path).endswith(".zip") else map_path
    info = _read_json(root, "Info.dat")
    bpm = float(info["_beatsPerMinute"])

    fname = njs = None
    for name, f, j in list_difficulties(map_path):
        if name.lower() == difficulty.lower():
            fname, njs = f, j
    if fname is None:
        raise ValueError(f"difficulty {difficulty!r} not in "
                         f"{[d[0] for d in list_difficulties(map_path)]}")

    data = _read_json(root, fname)
    sec_per_beat = 60.0 / bpm
    speed = njs * speed_scale
    raw = []
    if "colorNotes" in data:                      # v3 schema
        for n in data["colorNotes"]:
            angle = _DIR_ANGLE.get(n["d"], 0.0) + math.radians(n.get("a", 0))
            raw.append((n["b"] * sec_per_beat, n["x"], n["y"], n["c"],
                        angle % (2 * math.pi), n["d"] == 8))
    else:                                         # v2 schema
        for n in data["_notes"]:
            if n["_type"] not in (0, 1):          # 3 = bomb: skip
                continue
            angle = _DIR_ANGLE.get(n["_cutDirection"], 0.0)
            raw.append((n["_time"] * sec_per_beat, n["_lineIndex"],
                        n["_lineLayer"], n["_type"], angle,
                        n["_cutDirection"] == 8))

    raw.sort(key=lambda r: r[0])
    shift = 0.0
    if first_note_s is not None and raw:
        shift = first_note_s - raw[0][0]
    notes = [Note(hit_time=t + shift, col=int(x), row=int(y), color=int(c),
                  angle=float(a), speed=speed, any_dir=bool(dot))
             for t, x, y, c, a, dot in raw
             if 0 <= x <= 3 and 0 <= y <= 2]      # drop ME/extended-grid notes

    meta = {"bpm": bpm, "njs": njs, "speed": speed, "difficulty": difficulty,
            "n_notes": len(notes), "dropped": len(raw) - len(notes),
            "song_name": info.get("_songName", "?"),
            "duration_s": notes[-1].hit_time - notes[0].hit_time if notes else 0.0,
            "nps": (len(notes) / (notes[-1].hit_time - notes[0].hit_time))
                   if len(notes) > 1 else 0.0}
    return notes, meta
