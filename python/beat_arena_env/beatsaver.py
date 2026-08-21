"""BeatSaver map parser: real Beat Saber charts -> beat-arena Notes.

Reads a map directory or .zip (as downloaded from beatsaver.com):
Info.dat / info.dat for BPM + difficulty file names + NJS, then the
difficulty .dat in any of the three schemas:

- v2 ("2.x"): `_notes` (types 0/1 = color, 3 = bomb), BPM changes via
  event `_type` 100.
- v3 ("3.x"): `colorNotes` (a = precision angle offset, degrees CCW),
  `bombNotes`, `sliders` (arcs), `burstSliders` (chains), `bpmEvents`.
- v4 ("4.x"): index-indirected element/data arrays (`colorNotes` ->
  `colorNotesData` etc.), arcs/chains referencing note data; BPM from
  the audio data file when present.

Env v7 additions: bombs (KIND_BOMB), chains (head KIND_CHAIN_HEAD +
links via chart.chain_links, real quadratic-Bezier layout with squish),
arcs as scoring grants (head note free_post, tail note free_pre), the
paired-note ANGLE SNAP rule (exactly two same-color notes on a beat
snap their arrows to the line between them when within 40 deg —
ArcViewer NoteManager.GetSnapAngles), and piecewise BPM->time
conversion.

Env v17 additions: walls/obstacles (all three schemas, ScoreSaber-
watch geometry verbatim — spec §12a; vanilla only, ME wall types
skipped), returned in meta["walls"] as chart.Wall AABBs.
NOT parsed (noted for later): 360/90
rotation events, NJS events, Noodle/ME custom data (fake notes are NOT
filtered), one-saber characteristic, left-hand mirroring.

Grid conventions match ours (x/lineIndex 0-3 left->right, y/lineLayer
0-2 bottom->top; world positions via constants.col_x / ROW_Y). Cut
direction is the direction you SWING; direction 8 is a dot note ->
`any_dir`. Per-note speed = the map's real NJS since 2026-07-30 (user
decision): the historical speed_scale rescale (0.3, chosen because
real note-jump speeds give 1-2-tick contact windows at 30 Hz) is
REMOVED — everything trains, benches, and replays at real speed;
pre-removal results and checkpoints are a different physics regime.
"""

from __future__ import annotations

import json
import math
import os
import zipfile
from dataclasses import replace

from .chart import Note, Wall, chain_links
from .constants import (KIND_BOMB, KIND_CHAIN_HEAD, KIND_NOTE,
                        PLAYER_HEIGHT_OFFSET, ROW_Y, col_x)

# env v17 wall conversion constants (ScoreSaber watch WallManager.cs /
# ObjectManager.cs — spec §12a). Grid x origin −0.9 matches our
# col_x(0) = −0.9 + 0.3; the wall y scale is 0.6 (WallHScale), NOT the
# note RowHeight 0.55.
_WALL_LANE_W = 0.6      # ObjectManager.LaneWidth
_WALL_H_SCALE = 0.6     # ObjectManager.WallHScale
_WALL_GRID_X0 = -0.9    # ObjectManager.GridBottomLeft.x (y0 is 0)
_WALL_MIN_SIZE = 0.06   # WallManager.MinWallSize (width/height/length)
_WALL_MAX_TOP = 3.0     # ClampPlayerHeight ceiling

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
SNAP_MAX_DEV = math.radians(40.0)   # paired-note snap rejection threshold
_SAME_BEAT_S = 0.001                # "same beat" epsilon, seconds (ArcViewer)


def _read_json(root, name):
    if isinstance(root, zipfile.ZipFile):
        names = {n.lower(): n for n in root.namelist()}
        if name.lower() not in names:
            raise FileNotFoundError(f"{name} not found in zip")
        return json.loads(root.read(names[name.lower()]))
    for cand in (name, name.lower(), name.capitalize()):
        path = os.path.join(root, cand)
        if os.path.exists(path):
            # explicit utf-8: beatmap .dat files are UTF-8 JSON; the
            # platform default (cp1252 on Windows) silently skipped 80
            # of 6101 catalog leaderboards with UnicodeDecodeError —
            # 76 of them inside the pp-relevant 4.18-11.61* band
            # (2026-07-25 round-4 finding)
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"{name} not found in {root}")


class _BeatClock:
    """Piecewise-linear beat -> seconds under BPM changes."""

    def __init__(self, base_bpm: float, changes=()):
        # changes: sorted [(beat, bpm)]; implicit (0, base_bpm) start.
        # A change at an existing anchor's beat REPLACES that anchor's
        # tempo — maps routinely open with a beat-0 bpmEvent whose value
        # differs slightly from Info.dat's (e.g. 171.79 vs 172), and
        # dropping it desyncs every exported time by a few ms (found via
        # watch.scoresaber.com note-matching, which allows only 1 ms).
        self.anchors = [(0.0, 0.0, 60.0 / base_bpm)]
        for beat, bpm in sorted(changes):
            if bpm <= 0:
                continue
            b_last, t_last, _ = self.anchors[-1]
            if beat < b_last:
                continue
            if beat == b_last:
                self.anchors[-1] = (b_last, t_last, 60.0 / bpm)
                continue
            self.anchors.append(
                (beat, t_last + (beat - b_last) * self.anchors[-1][2],
                 60.0 / bpm))

    def time(self, beat: float) -> float:
        b0, t0, spb = self.anchors[0]
        for b, t, s in self.anchors:
            if b > beat:
                break
            b0, t0, spb = b, t, s
        return t0 + (beat - b0) * spb


def _clock_from_audio_data(root, fname, base_bpm):
    """v4: BPM anchors live in the audio data file (sample-index ranges)."""
    try:
        audio = _read_json(root, fname)
    except (FileNotFoundError, KeyError):
        return _BeatClock(base_bpm)
    freq = float(audio.get("songFrequency", audio.get("frequency", 44100)))
    clock = _BeatClock(base_bpm)
    anchors = []
    for seg in audio.get("bpmData", []):
        sb, eb = float(seg["sb"]), float(seg["eb"])
        si, ei = float(seg["si"]), float(seg["ei"])
        if eb > sb:
            bpm = (eb - sb) / ((ei - si) / freq) * 60.0
            anchors.append((sb, bpm))
    if anchors:
        clock = _BeatClock(anchors[0][1], anchors)
    return clock


def _info(root):
    info = _read_json(root, "Info.dat")
    version = str(info.get("version", info.get("_version", "2.0.0")))
    return info, version.startswith("4")


def list_difficulties(map_path):
    """[(difficulty_name, filename, njs), ...] for the Standard characteristic."""
    root = zipfile.ZipFile(map_path) if str(map_path).endswith(".zip") else map_path
    info, is_v4 = _info(root)
    out = []
    if is_v4:
        for d in info.get("difficultyBeatmaps", []):
            if d.get("characteristic", "Standard") != "Standard":
                continue
            out.append((d["difficulty"], d["beatmapDataFilename"],
                        float(d.get("noteJumpMovementSpeed", 10.0))))
    else:
        for s in info["_difficultyBeatmapSets"]:
            if s["_beatmapCharacteristicName"] != "Standard":
                continue
            for d in s["_difficultyBeatmaps"]:
                out.append((d["_difficulty"], d["_beatmapFilename"],
                            float(d["_noteJumpMovementSpeed"])))
    return out


def _snap_paired_angles(notes: list[Note]) -> list[Note]:
    """ArcViewer's paired-note angle snap: exactly two same-color notes on
    the same beat align their arrows to the line between them (per-note
    nearest of the two line senses; rejected if any non-dot arrow is more
    than 40 deg away; skipped when both non-dot notes share a row/column
    or point different directions; both-dot pairs untouched)."""
    groups: dict[tuple, list[int]] = {}
    for i, n in enumerate(notes):
        if n.kind not in (KIND_NOTE, KIND_CHAIN_HEAD):
            continue
        key = (round(n.hit_time / _SAME_BEAT_S), n.color)
        groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        if len(idxs) != 2:
            continue
        a, b = notes[idxs[0]], notes[idxs[1]]
        if a.any_dir and b.any_dir:
            continue
        if not a.any_dir and not b.any_dir:
            if a.col == b.col or a.row == b.row:
                continue
            if abs((a.angle - b.angle + math.pi) % (2 * math.pi) - math.pi) > 1e-6:
                continue
        dx, dy = col_x(b.col) - col_x(a.col), ROW_Y[b.row] - ROW_Y[a.row]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        line = math.atan2(dy, dx)
        ok = True
        snapped = []
        for n in (a, b):
            if n.any_dir:
                snapped.append(line)          # dots take the line as-is
                continue
            dev = (line - n.angle + math.pi) % (2 * math.pi) - math.pi
            cand = line if abs(dev) <= math.pi / 2 else line + math.pi
            dev = (cand - n.angle + math.pi) % (2 * math.pi) - math.pi
            if abs(dev) > SNAP_MAX_DEV:
                ok = False
                break
            snapped.append(cand % (2 * math.pi))
        if ok:
            notes[idxs[0]] = replace(a, angle=snapped[0] % (2 * math.pi))
            notes[idxs[1]] = replace(b, angle=snapped[1] % (2 * math.pi))
    return notes


def _in_grid(x, y):
    return 0 <= x <= 3 and 0 <= y <= 2


def load_beatmap(map_path, difficulty="Normal", first_note_s=None):
    """Parse one difficulty of a BeatSaver map -> (notes, meta dict).

    map_path: map directory or .zip. first_note_s: if set, shift the whole
    chart so the first note lands there (keeps our warm-up convention).
    """
    root = zipfile.ZipFile(map_path) if str(map_path).endswith(".zip") else map_path
    info, is_v4 = _info(root)

    fname = njs = None
    for name, f, j in list_difficulties(map_path):
        if name.lower() == difficulty.lower():
            fname, njs = f, j
    if fname is None:
        raise ValueError(f"difficulty {difficulty!r} not in "
                         f"{[d[0] for d in list_difficulties(map_path)]}")

    data = _read_json(root, fname)
    speed = float(njs)                 # real NJS; speed_scale removed
    dropped = 0

    if is_v4:
        bpm = float(info["audio"]["bpm"])
        clock = _clock_from_audio_data(
            root, info["audio"].get("audioDataFilename", "AudioData.dat"), bpm)
        song_name = info.get("song", {}).get("title", "?")
    else:
        bpm = float(info["_beatsPerMinute"])
        song_name = info.get("_songName", "?")
        changes = []
        if "bpmEvents" in data:                       # v3
            changes = [(float(e["b"]), float(e["m"])) for e in data["bpmEvents"]]
        elif "_events" in data:                       # v2 (type 100)
            changes = [(float(e["_time"]), float(e["_floatValue"]))
                       for e in data["_events"]
                       if e.get("_type") == 100 and e.get("_floatValue")]
        clock = _BeatClock(bpm, changes)

    notes: list[Note] = []
    walls: list[Wall] = []

    def add_wall(beat, x, y, w, h, dur_beats, v4_walls):
        """ScoreSaber-watch Wall(BeatmapObstacle) verbatim (spec §12a).
        Vanilla only — the ME/Noodle branches (customData size/
        coordinates, precision values) are out of scope like notes'."""
        if dur_beats < 0:                       # negative walls flip
            beat, dur_beats = beat + dur_beats, -dur_beats
        y = min(max(int(y), 0), 4 if v4_walls else 2)
        h = min(float(h), 5.0 - y)
        world_w = float(w) * _WALL_LANE_W
        world_h = h * _WALL_H_SCALE
        # MinWallSize floor keeps the sign (viewer Mathf.Sign); vanilla
        # maps have positive sizes but zero-width walls do exist
        world_w = max(abs(world_w), _WALL_MIN_SIZE) * math.copysign(1.0, world_w)
        world_h = max(abs(world_h), _WALL_MIN_SIZE) * math.copysign(1.0, world_h)
        cx = _WALL_GRID_X0 + float(x) * _WALL_LANE_W + (world_w - _WALL_LANE_W) / 2
        cy = y * _WALL_H_SCALE + world_h / 2
        # ClampPlayerHeight (vanilla walls always clamp): bottom >= 0,
        # then top <= 3 m by shrinking + recentering
        bottom = cy - world_h / 2
        if bottom < 0:
            cy -= bottom
        top = cy + world_h / 2
        if top > _WALL_MAX_TOP:
            diff = top - _WALL_MAX_TOP
            world_h -= diff
            cy -= diff / 2
        t0 = clock.time(beat)
        dur_s = clock.time(beat + dur_beats) - t0
        length = max(dur_s * speed, _WALL_MIN_SIZE)
        # env v33: player-height offset AFTER all clamps — the viewer
        # adds playerHeightOffset at render time, downstream of the
        # ClampPlayerHeight conversion, so the order here matches it
        walls.append(Wall(start_time=t0, cx=cx,
                          cy=cy + PLAYER_HEIGHT_OFFSET, width=abs(world_w),
                          height=abs(world_h), length=length, speed=speed))

    def add_note(beat, x, y, color, d, angle_off_deg=0.0, kind=KIND_NOTE):
        nonlocal dropped
        if not _in_grid(x, y):
            dropped += 1
            return None
        angle = (_DIR_ANGLE.get(d, 0.0) + math.radians(angle_off_deg)) \
            % (2 * math.pi)
        n = Note(hit_time=clock.time(beat), col=int(x), row=int(y),
                 color=int(color), angle=angle, speed=speed,
                 any_dir=d == 8, kind=kind, raw_dir=int(d))
        notes.append(n)
        return n

    def add_bomb(beat, x, y):
        nonlocal dropped
        if not _in_grid(x, y):
            dropped += 1
            return
        notes.append(Note(hit_time=clock.time(beat), col=int(x), row=int(y),
                          color=0, angle=0.0, speed=speed, any_dir=True,
                          kind=KIND_BOMB))

    def find_note(beat, x, y, color):
        t = clock.time(beat)
        for i, n in enumerate(notes):
            if (n.kind in (KIND_NOTE, KIND_CHAIN_HEAD) and n.color == color
                    and n.col == x and n.row == y
                    and abs(n.hit_time - t) < _SAME_BEAT_S):
                return i
        return None

    def add_chain(head_beat, x, y, color, d, tail_beat, tx, ty, sc, squish):
        i = find_note(head_beat, x, y, color)
        if i is not None:
            notes[i] = replace(notes[i], kind=KIND_CHAIN_HEAD)
            head = notes[i]
        else:                                  # headless chain: synthesize
            head = add_note(head_beat, x, y, color, d, kind=KIND_CHAIN_HEAD)
            if head is None:
                return
        tx, ty = int(min(max(tx, 0), 3)), int(min(max(ty, 0), 2))
        notes.extend(chain_links(head, col_x(tx), ROW_Y[ty],
                                 clock.time(tail_beat), max(int(sc), 2),
                                 float(squish)))

    def add_arc(head_beat, x, y, color, tail_beat, tx, ty, tcolor):
        # reversed arc (tailBeat < beat): the game — and the ScoreSaber
        # viewer's ObjectManager — treat the earlier end as the head.
        # Without the swap free_post/free_pre invert, the derived
        # ArcHead/ArcTail scoring types flip, and the viewer's note-event
        # ID match fails for both ends (notes render as never hit)
        if tail_beat < head_beat:
            head_beat, x, y, color, tail_beat, tx, ty, tcolor = \
                tail_beat, tx, ty, tcolor, head_beat, x, y, color
        i = find_note(head_beat, x, y, color)
        if i is not None:
            notes[i] = replace(notes[i], free_post=True)
        j = find_note(tail_beat, tx, ty, tcolor)
        if j is not None:
            notes[j] = replace(notes[j], free_pre=True)

    if is_v4:                                         # v4 schema
        cdata = data.get("colorNotesData", [])
        for n in data.get("colorNotes", []):
            d = cdata[n["i"]] if n.get("i") is not None else {}
            add_note(n["b"], d.get("x", 0), d.get("y", 0), d.get("c", 0),
                     d.get("d", 0), d.get("a", 0))
        bdata = data.get("bombNotesData", [])
        for n in data.get("bombNotes", []):
            d = bdata[n["i"]] if n.get("i") is not None else {}
            add_bomb(n["b"], d.get("x", 0), d.get("y", 0))
        ch_data = data.get("chainsData", [])
        for c in data.get("chains", []):
            hd = cdata[c["i"]] if c.get("i") is not None else {}
            cd = ch_data[c["ci"]] if c.get("ci") is not None else {}
            add_chain(c["hb"], hd.get("x", 0), hd.get("y", 0),
                      hd.get("c", 0), hd.get("d", 0), c["tb"],
                      cd.get("tx", 0), cd.get("ty", 0),
                      cd.get("c", 3), cd.get("s", 1.0))
        for a in data.get("arcs", []):
            hd = cdata[a["hi"]] if a.get("hi") is not None else {}
            td = cdata[a["ti"]] if a.get("ti") is not None else {}
            add_arc(a["hb"], hd.get("x", 0), hd.get("y", 0), hd.get("c", 0),
                    a["tb"], td.get("x", 0), td.get("y", 0), td.get("c", 0))
        odata = data.get("obstaclesData", [])
        for o in data.get("obstacles", []):
            d = odata[o["i"]] if o.get("i") is not None else {}
            add_wall(o["b"], d.get("x", 0), d.get("y", 0), d.get("w", 1),
                     d.get("h", 1), float(d.get("d", 0)), v4_walls=True)
    elif "colorNotes" in data:                        # v3 schema
        # .get(..., 0) throughout: some serializers omit zero-valued
        # fields (KeyError 'y'/'c' skipped 6 catalog levels; the v4
        # branch above already guards the same way)
        for n in data["colorNotes"]:
            add_note(n["b"], n.get("x", 0), n.get("y", 0), n.get("c", 0),
                     n.get("d", 0), n.get("a", 0))
        for n in data.get("bombNotes", []):
            add_bomb(n["b"], n.get("x", 0), n.get("y", 0))
        for c in data.get("burstSliders", []):
            add_chain(c["b"], c.get("x", 0), c.get("y", 0), c.get("c", 0),
                      c.get("d", 0), c["tb"],
                      c.get("tx", 0), c.get("ty", 0), c.get("sc", 3),
                      c.get("s", 1.0))
        for a in data.get("sliders", []):
            add_arc(a["b"], a.get("x", 0), a.get("y", 0), a.get("c", 0),
                    a["tb"], a.get("tx", 0), a.get("ty", 0),
                    a.get("tc", a.get("c", 0)))
        for o in data.get("obstacles", []):
            add_wall(o["b"], o.get("x", 0), o.get("y", 0), o.get("w", 1),
                     o.get("h", 1), float(o.get("d", 0)), v4_walls=False)
    else:                                             # v2 schema
        for n in data["_notes"]:
            if n["_type"] in (0, 1):
                add_note(n["_time"], n["_lineIndex"], n["_lineLayer"],
                         n["_type"], n["_cutDirection"])
            elif n["_type"] == 3:
                add_bomb(n["_time"], n["_lineIndex"], n["_lineLayer"])
        for o in data.get("_obstacles", []):
            t = int(o.get("_type", 0))
            if t >= 1000:            # Mapping Extensions wall: out of scope
                continue
            if (o.get("_customData") or {}).get("_fake"):
                continue
            # v2 wall shape is coded in _type: 0 = full wall (grid bottom,
            # 5 rows tall -> capped to 3 m), 1 = crouch wall (hangs
            # y 1.2..3.0 m)
            add_wall(o["_time"], o.get("_lineIndex", 0), 0 if t == 0 else 2,
                     o.get("_width", 1), 5 if t == 0 else 3,
                     float(o.get("_duration", 0)), v4_walls=False)

    notes = _snap_paired_angles(notes)
    notes.sort(key=lambda n: n.hit_time)
    walls.sort(key=lambda w: w.start_time)
    shift = 0.0
    if first_note_s is not None and notes:
        shift = first_note_s - notes[0].hit_time
        notes = [replace(n, hit_time=n.hit_time + shift) for n in notes]
        walls = [replace(w, start_time=w.start_time + shift) for w in walls]

    scoreable = [n for n in notes if n.kind != KIND_BOMB]
    meta = {"bpm": bpm, "njs": njs, "speed": speed, "difficulty": difficulty,
            "n_notes": len(scoreable), "n_bombs": len(notes) - len(scoreable),
            "n_walls": len(walls), "walls": walls,
            "dropped": dropped, "song_name": song_name,
            "duration_s": notes[-1].hit_time - notes[0].hit_time if notes else 0.0,
            "nps": (len(scoreable) / (notes[-1].hit_time - notes[0].hit_time))
                   if len(notes) > 1 else 0.0}
    return notes, meta
