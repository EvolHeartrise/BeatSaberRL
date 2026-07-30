"""Convert a beat-arena JSONL v7 replay to a ScoreSaber replay .dat.

    python convert_scoresaber.py --replay replays/x.jsonl \\
        --hash <song-hash> --difficulty Hard [--out x.dat]

The output loads in ScoreSaber's web viewer (watch.scoresaber.com — an
ArcViewer derivative; open/drag the .dat onto the page and it fetches
the map from BeatSaver by the hash embedded in the metadata). Format
reverse-engineered from the viewer's own decoder in the ScoreSaber/watch
repo (mirrored at projects/beat-arena/research/vendor/scoresaber-watch):

  "ScoreSaber Replay 👌🤠\\r\\n" magic (28 bytes) + LZMA-alone body.
  Body = 9-int pointer table (metadata, poses, heights, notes, scores,
  combo, multiplier, energy, extensions) + blocks. We write replay
  format v3.1.0 — the same version the game's recorder writes today
  (per-note gameplay/scoring types + angle offset, timeDeviation +
  note transform tail, ImmediateMaxPossibleScore on score keyframes,
  game/plugin/platform metadata strings) — with real cut kinematics
  (contact point, cut-plane normal, tip speed, swing-direction
  deviation) recomputed from the recorded joint state via FK.

Fidelity notes:
- Pose frames are exact: wrist positions from the recorded joint state,
  blade orientation via the kinematics FK mirror, 30 fps.
- Coordinates are the game's PLAYER SPACE: player at z = 0, notes
  arriving at z = +0.65 (the viewer's PlayerCutPlaneDistance — it
  parents all replay poses under a rig 0.65 m behind the world origin
  and puts the note-arrival plane at head.z + 0.65). The env world
  (head -0.65, cut plane 0) IS the viewer's world frame, so positions
  shift +0.65 in z at write time; x/y agree (floor-based, same rows).
  Metadata initialHeight is the player's BODY height (1.80 for the
  env's 1.8 m player, head anchor 1.70) — the viewer lowers every
  object by (height - 1.8)/2, so writing the head y there sank all
  notes 5 cm.
- The viewer matches events to map notes by ID = scoringType*10000 +
  lineIndex*1000 + lineLayer*100 + color*10 + cutDirection, grouped by
  note time with a 1 ms epsilon — so line/layer/direction come from
  RE-PARSING the original map (the replay chart stores only derived
  angles), and event times are shifted from sim time (first note at
  2.0 s) back to song time.
- The viewer recomputes per-note scores from cut ratings. We encode
  ratings that reproduce our integers exactly: before = pre/70,
  after = post/30, centerDistance = 0.3*(1 - acc/15). The sim's cut
  event merges pre+acc, so the split assumes full pre-swing first
  (pre = min(cut_pts, 70)) — per-component popups can differ from the
  sim's true split on partial-pre-swing cuts, but every note's TOTAL
  (and the final score) is exact. Chain links are flat 20 in both.
- The multiplied score timeline reuses the bench's ComboTracker, so the
  final score equals the ScoreSaber-bench score for the same rollout.
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import struct

from beat_arena_env import scoresaber
from beat_arena_env.beatsaver import load_beatmap
from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import (KIND_BOMB, KIND_CHAIN_HEAD,
                                      KIND_CHAIN_LINK, KIND_NOTE)
from beat_arena_env.kinematics import fk

MAGIC = (b"ScoreSaber Replay " + "\U0001F44C\U0001F920".encode() + b"\r\n")
assert len(MAGIC) == 28

DIFF_RAW = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}
HEAD_POS = (0.0, 1.70, -0.65)          # env v7 look-at anchor
PLAYER_HEIGHT = 1.80                   # env body height (NOT the head y)

# env world -> replay player space: the file wants the player at z = 0
# with notes arriving at +0.65; the env's z = 0 cut plane / z = -0.65
# head is the viewer's WORLD frame, one PlayerCutPlaneDistance behind
CUT_PLANE_DIST = 0.65


def player_space(p):
    return (p[0], p[1], p[2] + CUT_PLANE_DIST)

# their ScoringType enum (ObjectManager/ScoringEvent)
ST_NOSCORE, ST_NOTE, ST_ARCHEAD, ST_ARCTAIL = 0, 1, 2, 3
ST_CHAINHEAD, ST_CHAINLINK, ST_ARCHEADARCTAIL = 4, 5, 6
ST_CHAINHEADARCTAIL, ST_CHAINLINKARCHEAD = 7, 8
ST_CHAINHEADARCHEAD, ST_CHAINHEADARCHEADARCTAIL = 9, 10


def scoring_type(n: Note) -> int:
    if n.kind == KIND_BOMB:
        return ST_NOSCORE
    if n.kind == KIND_CHAIN_LINK:
        return ST_CHAINLINKARCHEAD if n.free_post else ST_CHAINLINK
    head, tail = n.free_post, n.free_pre      # arc head grants post & v.v.
    if n.kind == KIND_CHAIN_HEAD:
        if head and tail:
            return ST_CHAINHEADARCHEADARCTAIL
        if head:
            return ST_CHAINHEADARCHEAD
        if tail:
            return ST_CHAINHEADARCTAIL
        return ST_CHAINHEAD
    if head and tail:
        return ST_ARCHEADARCTAIL
    if head:
        return ST_ARCHEAD
    if tail:
        return ST_ARCTAIL
    return ST_NOTE


# -- binary writers ----------------------------------------------------------

def w_int(b, v):
    b += struct.pack("<i", int(v))
    return b


def w_float(b, v):
    return b + struct.pack("<f", float(v))


def w_bool(b, v):
    return b + (b"\x01" if v else b"\x00")


def w_str(b, s):
    raw = s.encode("utf-8")
    return b + struct.pack("<i", len(raw)) + raw


def w_vec3(b, v):
    return b + struct.pack("<3f", *v)


def quat_from_forward(d):
    """Unity quaternion (x,y,z,w) rotating +z onto unit vector d."""
    dot = d[2]                                     # (0,0,1) . d
    if dot < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)                # 180 deg about y
    ax = (-d[1], d[0], 0.0)                        # (0,0,1) x d
    q = (ax[0], ax[1], ax[2], 1.0 + dot)
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


# -- replay reading ----------------------------------------------------------

def read_replay(path):
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert "header" in lines[0], "not a beat-arena replay"
    header = lines[0]["header"]
    assert header.get("version") == 7, f"need replay v7, got {header.get('version')}"
    return header, lines[1:]


def align_charts(header, map_dir, difficulty):
    """Re-parse the original map and align its notes 1:1 (by order) with
    the replay's chart. Returns (real_notes, time_offset sim->song)."""
    sim_notes, _ = load_beatmap(map_dir, difficulty, first_note_s=2.0)
    real_notes, _ = load_beatmap(map_dir, difficulty, first_note_s=None)
    chart = header["chart"]
    if len(sim_notes) != len(chart):
        raise SystemExit(
            f"map/replay mismatch: map parses to {len(sim_notes)} objects, "
            f"replay chart has {len(chart)} — wrong map or difficulty?")
    for i, (n, c) in enumerate(zip(sim_notes, chart)):
        t, x, y = note_tuple(n)[0:3]
        if (abs(t - c["t"]) > 1e-3 or abs(x - c["x"]) > 5e-3
                or abs(y - c["y"]) > 5e-3 or n.kind != c["kind"]
                or n.color != c["color"]):
            raise SystemExit(f"map/replay mismatch at object {i}: "
                             f"map ({t:.3f},{x:.2f},{y:.2f}) kind {n.kind} vs "
                             f"replay ({c['t']:.3f},{c['x']:.2f},{c['y']:.2f}) "
                             f"kind {c['kind']}")
    # +DT: a JSONL tick line's `time` labels the tick START, but its
    # saber state and events are END-of-tick (rust_env.py stamps the
    # post-step state with sim.time - DT; viz/replay.html compensates by
    # placing outcomes at (tick+1)*DT). Without the shift every pose and
    # event leads the song by one tick — 33 ms, 0.3-0.8 m of note travel
    # at real NJS in the ScoreSaber viewer.
    offset = real_notes[0].hit_time - sim_notes[0].hit_time + 1.0 / 30.0
    return real_notes, offset


# -- block builders ----------------------------------------------------------

def metadata_block(song_hash, difficulty):
    b = bytearray()
    b = w_str(b, "3.1.0")
    b = w_str(b, f"custom_level_{song_hash.upper()}")
    b = w_int(b, DIFF_RAW[difficulty])
    b = w_str(b, "Standard")
    b = w_str(b, "DefaultEnvironment")
    b = w_int(b, 0)                                # modifiers: empty array
    b = w_float(b, 0.0)                            # NoteSpawnOffset
    b = w_bool(b, False)                           # leftHanded
    b = w_float(b, PLAYER_HEIGHT)                  # initialHeight (body height)
    b += b"\x00" * 16                              # RoomRotation + RoomCenter
    b = w_float(b, 0.0)                            # failTime
    b = w_str(b, "beat-arena env v7")              # GameVersion (>=3.1.0)
    b = w_str(b, "convert_scoresaber/1.0")         # PluginVersion
    b = w_str(b, "PC")                             # Platform
    return bytes(b)


def pose_block(ticks, offset):
    frames = []
    prev_tips = [None, None]
    for line in ticks:
        t = line["time"] + offset
        hands = []
        for hand in (0, 1):
            s = line["sabers"][hand]
            _, wrist, blade = fk(hand, s[0:6])
            wrist = s[12:15]                       # recorded wrist is exact
            tip = tuple(wrist[i] + blade[i] for i in range(3))
            hands.append((tuple(wrist), quat_from_forward(blade)))
            prev_tips[hand] = tip
        frames.append((t, hands))
    b = bytearray()
    b = w_int(b, len(frames))
    eps = 1e-4
    last_t = None
    for t, hands in frames:
        t = max(t, eps)                            # reader drops time==0
        if last_t is not None and t <= last_t:     # and duplicate times
            t = last_t + eps
        last_t = t
        b = w_vec3(b, player_space(HEAD_POS))
        b += struct.pack("<4f", 0.0, 0.0, 0.0, 1.0)
        for pos, quat in hands:
            b = w_vec3(b, player_space(pos))
            b += struct.pack("<4f", *quat)
        b = w_int(b, 30)                           # fps
        b = w_float(b, t)
    return bytes(b)


GAMEPLAY_TYPE = {KIND_NOTE: 0, KIND_BOMB: 1, KIND_CHAIN_HEAD: 2,
                 KIND_CHAIN_LINK: 3}                 # game NoteData enum

# swing angle (radians, 0 = +x CCW) of each map cutDirection int
_DIR_ANGLE_DEG = {0: 90.0, 1: 270.0, 2: 180.0, 3: 0.0,
                  4: 135.0, 5: 45.0, 6: 225.0, 7: 315.0}


def cut_geometry(ticks, tick_idx, hand, note):
    """Real cut kinematics at a tick from the recorded joint state:
    blade dir, tip speed, contact point (closest blade point to the
    note), geometric center distance, cut-plane normal (unit, spans the
    blade and its motion — |normal.z| is the time-dependency stat), and
    the swing's xy motion angle."""
    def state(i):
        s = ticks[i]["sabers"][hand]
        _, _, blade = fk(hand, s[0:6])
        wrist = s[12:15]
        return wrist, blade
    wrist, blade = state(tick_idx)
    p_wrist, p_blade = state(max(tick_idx - 1, 0))
    tip = [wrist[j] + blade[j] for j in range(3)]
    p_tip = [p_wrist[j] + p_blade[j] for j in range(3)]
    motion = [(tip[j] - p_tip[j]) * 30.0 for j in range(3)]
    speed = math.sqrt(sum(m * m for m in motion))

    center = (note_tuple(note)[1], note_tuple(note)[2], 0.0)
    s = sum((center[j] - wrist[j]) * blade[j] for j in range(3))
    s = min(max(s, 0.0), 1.0)                      # blade is 1 m long
    contact = tuple(wrist[j] + s * blade[j] for j in range(3))
    center_dist = math.dist(contact, center)

    normal = (blade[1] * motion[2] - blade[2] * motion[1],
              blade[2] * motion[0] - blade[0] * motion[2],
              blade[0] * motion[1] - blade[1] * motion[0])
    n = math.sqrt(sum(c * c for c in normal))
    if n < 1e-9:                                   # stationary blade
        normal, n = (-blade[1], blade[0], 0.0), math.hypot(blade[0], blade[1])
    normal = tuple(c / (n or 1.0) for c in normal)
    swing_angle = math.degrees(math.atan2(motion[1], motion[0]))
    return {"blade": tuple(blade), "speed": speed, "contact": contact,
            "center_dist": center_dist, "normal": normal,
            "swing_angle": swing_angle}


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def build_events(ticks, real_notes, offset):
    """Note-event dicts + the score/combo/multiplier HUD timelines.

    The keyframe blocks use the game recorder's layouts (ComboEvent =
    int combo + float time; MultiplierEvent = int multiplier + float
    bar-progress + float time), emitted change-only like real replays.
    NOTE: the live watch.scoresaber.com decoder SKIPS the combo/
    multiplier/energy pointers (ScoreSaberDecoder reads five pointers,
    then only the extensions one) and recomputes score/combo from the
    note events and energy from the damage/heal constants — so for that
    viewer these blocks only keep the 9-pointer table valid. They are
    still written faithfully for other format consumers.
    """
    sts = [scoring_type(n) for n in real_notes]
    walk = scoresaber.ComboTracker()
    max_walk = scoresaber.ComboTracker()           # perfect-so-far walk:
    events, timeline = [], []                      # ImmediateMaxPossibleScore
    combo_tl, mult_tl = [], []
    energy_tl = [(0.5, 0.0)]
    energy = 0.5
    by_note = {}

    def emit_score(t):
        # change-only, like the game's scoreDidChangeEvent
        score = round(walk.score)
        if not timeline or timeline[-1][0] != score:
            timeline.append((score, t, round(max_walk.score)))

    def hud(t, i, outcome):
        # Change-only keyframes, mirroring the game's recorder (real
        # replays emit combo/multiplier/energy events only when the value
        # changes — e.g. the multiplier stream goes silent at a held 8x,
        # the energy stream once capped at 1.0).
        if not combo_tl or combo_tl[-1][0] != walk.combo:
            combo_tl.append((walk.combo, t))
        mult_state = (walk.multiplier, walk.progress_fraction)
        if not mult_tl or mult_tl[-1][:2] != mult_state:
            mult_tl.append((*mult_state, t))
        nonlocal energy
        link = sts[i] in (ST_CHAINLINK, ST_CHAINLINKARCHEAD)
        delta = {"good": 0.002 if link else 0.01,
                 "bad": -0.025 if link else -0.1,
                 "miss": -0.03 if link else -0.15,
                 "bomb": -0.15}[outcome]
        # floor above zero: at zero the game FAILS and the run would end;
        # our agent plays on regardless (No-Fail-like, without the
        # modifier flag). The live watch viewer IGNORES this block and
        # re-simulates energy from the note events, so it may still
        # display a fail — the floor keeps the written stream honest for
        # decoders that do read it
        new_energy = min(max(energy + delta, 0.001), 1.0)
        if new_energy != energy:
            energy = new_energy
            energy_tl.append((energy, t))
    for tick_idx, line in enumerate(ticks):
        t = line["time"] + offset
        for ev in line["events"]:
            kind, body = next(iter(ev.items()))
            i = body["note"]
            n = real_notes[i]
            if kind == "post":
                walk.on_points(i, body["points"])
                emit_score(t)
                if i in by_note:
                    by_note[i]["after"] = body["points"] / 30.0
                continue
            raw_dir = (8 if n.any_dir
                       else (n.raw_dir if n.raw_dir is not None else 8))
            # precision-angle offset: how far the note's actual swing
            # angle deviates from its cutDirection's canonical angle
            angle_off = 0.0 if raw_dir == 8 else wrap_deg(
                math.degrees(n.angle) - _DIR_ANGLE_DEG[raw_dir])
            if n.kind != KIND_BOMB:                # a scoreable was presented
                max_walk.on_cut(i)
                max_walk.on_points(i, scoresaber.MAX_UNIT_POINTS[n.kind])
            e = {"note": i, "st": sts[i], "gt": GAMEPLAY_TYPE[n.kind],
                 "line": n.col, "layer": n.row,
                 "color": 3 if n.kind == KIND_BOMB else n.color,
                 "dir": 3 if n.kind == KIND_BOMB else raw_dir,
                 "angle_off": 0.0 if n.kind == KIND_BOMB else angle_off,
                 "spawn": n.hit_time, "time": t, "time_dev": 0.0,
                 "before": 0.0, "after": 0.0, "dist": 0.3,
                 "speed": 0.0, "dir_ok": False, "saber": n.color,
                 "cut_angle": 0.0, "dir_dev": 0.0,
                 "note_pos": (note_tuple(n)[1], note_tuple(n)[2], 0.0),
                 "note_angle": 0.0 if n.any_dir else
                               math.radians(math.degrees(n.angle) - 270.0),
                 "cut_pos": (note_tuple(n)[1], note_tuple(n)[2], 0.0),
                 "normal": (0.0, 0.0, 0.0), "saber_dir": (0.0, 1.0, 0.0)}
            if kind == "miss":
                e["type"] = 3
                walk.on_break()
                hud(t, i, "miss")
            elif kind == "bomb":
                e["type"] = 4
                g = cut_geometry(ticks, tick_idx, 0, n)
                g2 = cut_geometry(ticks, tick_idx, 1, n)
                if g2["center_dist"] < g["center_dist"]:   # closer saber
                    g, e["saber"] = g2, 1
                else:
                    e["saber"] = 0
                e.update(saber_dir=g["blade"], speed=g["speed"],
                         cut_pos=g["contact"], normal=g["normal"],
                         time_dev=n.hit_time - t)
                walk.on_break()
                hud(t, i, "bomb")
            else:                                  # cut: good or bad
                good = body["good"]
                reason = body["reason"]
                hand = n.color if reason != "wrong_hand" else 1 - n.color
                g = cut_geometry(ticks, tick_idx, hand, n)
                dir_dev = 0.0 if n.any_dir else wrap_deg(
                    g["swing_angle"] - math.degrees(n.angle))
                e.update(saber=hand, saber_dir=g["blade"],
                         cut_pos=g["contact"], normal=g["normal"],
                         dir_dev=dir_dev, cut_angle=90.0 + dir_dev,
                         time_dev=n.hit_time - t,
                         speed=max(g["speed"],
                                   2.01 if reason != "slow" else 0.0))
                e["dir_ok"] = reason not in ("angle",)
                if good:
                    e["type"] = 1
                    walk.on_cut(i)
                    walk.on_points(i, body["points"])
                    emit_score(t)
                    hud(t, i, "good")
                    pts = round(body["points"])
                    if sts[i] not in (ST_CHAINLINK, ST_CHAINLINKARCHEAD):
                        # split cut points into pre-swing + accuracy:
                        # prefer the split whose accuracy matches the
                        # GEOMETRIC center distance, fall back to
                        # full-pre-swing-first; either way the encoded
                        # ratings round-trip the total exactly
                        acc_geo = round(15 * (1 - min(
                            g["center_dist"] / 0.3, 1.0)))
                        if 0 <= pts - acc_geo <= 70:
                            pre, acc = pts - acc_geo, acc_geo
                        else:
                            pre = min(pts, 70)
                            acc = max(0, min(pts - pre, 15))
                        e["before"] = pre / 70.0
                        e["dist"] = 0.3 * (1.0 - acc / 15.0)
                    by_note[i] = e
                else:
                    e["type"] = 2
                    walk.on_break()
                    emit_score(t)
                    hud(t, i, "bad")
            events.append(e)
    events.sort(key=lambda e: e["time"])
    return events, timeline, combo_tl, mult_tl, energy_tl, round(walk.score)


def note_block(events):
    """v3 note events (layout: plugin WriteNoteEvent / watch ReadNoteEvent)."""
    b = bytearray()
    b = w_int(b, len(events))
    identity = (0.0, 0.0, 0.0, 1.0)
    for e in events:
        b = w_float(b, e["spawn"])                 # noteTime (their spawnTime)
        b = w_int(b, e["layer"])
        b = w_int(b, e["line"])
        b = w_int(b, e["color"])
        b = w_int(b, e["dir"])
        b = w_int(b, e["gt"])                      # gameplayType (v3)
        b = w_int(b, e["st"])                      # scoringType (v3)
        b = w_float(b, e["angle_off"])             # cutDirAngleOffset (v3)
        b = w_int(b, e["type"])                    # 1 good 2 bad 3 miss 4 bomb
        b = w_vec3(b, player_space(e["cut_pos"]))  # cutPoint (blade contact)
        b = w_vec3(b, e["normal"])                 # cutNormal (cut plane)
        b = w_vec3(b, e["saber_dir"])
        b = w_int(b, e["saber"])
        b = w_bool(b, e["dir_ok"])
        b = w_float(b, e["speed"])
        b = w_float(b, e["cut_angle"])             # 90 deg = ideal swing
        b = w_float(b, e["dist"])
        b = w_float(b, e["dir_dev"])               # cutDirectionDeviation
        b = w_float(b, e["before"])
        b = w_float(b, e["after"])
        b = w_float(b, e["time"])                  # eventTime
        b = w_float(b, 1.0)                        # unityTimescale
        b = w_float(b, 1.0)                        # timeSyncTimescale
        # v3 64-byte tail
        b = w_float(b, e["time_dev"])              # timeDeviation (early > 0)
        b += struct.pack("<4f", *identity)         # WorldRotation
        b += struct.pack("<4f", *identity)         # InverseWorldRotation
        half = e["note_angle"] / 2.0               # NoteRotation: z-roll,
        b += struct.pack("<4f", 0.0, 0.0,          # down-swing = identity
                         math.sin(half), math.cos(half))
        b = w_vec3(b, player_space(e["note_pos"]))  # NotePosition
    return bytes(b)


def score_block(timeline):
    b = bytearray()
    b = w_int(b, len(timeline))
    for score, t, immediate_max in timeline:
        b = w_int(b, score)
        b = w_float(b, t)
        b = w_int(b, immediate_max)                # v3: perfect-so-far max
    return bytes(b)


def combo_block(combo_tl):
    b = bytearray()
    b = w_int(b, len(combo_tl))
    for combo, t in combo_tl:
        b = w_int(b, combo)
        b = w_float(b, t)
    return bytes(b)


def multiplier_block(mult_tl):
    b = bytearray()
    b = w_int(b, len(mult_tl))
    for mult, progress, t in mult_tl:
        b = w_int(b, mult)
        b = w_float(b, progress)
        b = w_float(b, t)
    return bytes(b)


def energy_block(energy_tl):
    b = bytearray()
    b = w_int(b, len(energy_tl))
    for energy, t in energy_tl:
        b = w_float(b, energy)
        b = w_float(b, t)
    return bytes(b)


def assemble(blocks):
    """blocks: 9 block byte-strings in pointer-table order."""
    table_len = 9 * 4
    ptrs, off = [], table_len
    for blk in blocks:
        ptrs.append(off)
        off += len(blk)
    body = struct.pack("<9i", *ptrs) + b"".join(blocks)
    # The LZMA-alone header MUST carry the real uncompressed size:
    # python's own encoder writes "unknown" (-1), which the vendored
    # decoder snapshot tolerates (grows until the end marker) but the
    # LIVE watch.scoresaber.com build rejects as "replay is too large"
    # (a WebGL sanity guard on the declared size). liblzma refuses to
    # ENCODE a known size, so build the container by hand: props byte +
    # dict size + real size + a raw LZMA1 stream. Their decoder reads
    # exactly `size` output bytes, so any trailing marker goes unread.
    return MAGIC + _lzma_alone(body)


_LZMA1 = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 23,
           "lc": 3, "lp": 0, "pb": 2}]


def _lzma_alone(body: bytes) -> bytes:
    props = bytes([(2 * 5 + 0) * 9 + 3])           # pb=2 lp=0 lc=3 -> 0x5D
    header = props + struct.pack("<I", 1 << 23) + struct.pack("<Q", len(body))
    return header + lzma.compress(body, format=lzma.FORMAT_RAW, filters=_LZMA1)


EMPTY = struct.pack("<i", 0)


def convert(replay_path, map_dir, song_hash, difficulty, out_path):
    header, ticks = read_replay(replay_path)
    real_notes, offset = align_charts(header, map_dir, difficulty)
    events, timeline, combo_tl, mult_tl, energy_tl, final = build_events(
        ticks, real_notes, offset)
    blocks = [
        metadata_block(song_hash, difficulty),
        pose_block(ticks, offset),
        EMPTY,                                     # heights
        note_block(events),
        score_block(timeline),
        combo_block(combo_tl),
        multiplier_block(mult_tl),
        energy_block(energy_tl),
        EMPTY,                                     # fps / extensions
    ]
    data = assemble(blocks)
    with open(out_path, "wb") as f:
        f.write(data)
    return {"out": out_path, "bytes": len(data), "frames": len(ticks),
            "note_events": len(events), "final_score": final}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--replay", required=True, help="JSONL v7 replay")
    p.add_argument("--hash", required=True, help="song hash (BeatSaver)")
    p.add_argument("--difficulty", required=True,
                   choices=list(DIFF_RAW))
    p.add_argument("--map", default=None,
                   help="map dir/zip (default maps/scoresaber/<hash>)")
    p.add_argument("--out", default=None,
                   help="output .dat (default: replay path with .dat)")
    args = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    map_dir = args.map or os.path.join(here, "maps", "scoresaber",
                                       args.hash.lower())
    out = args.out or os.path.splitext(args.replay)[0] + ".dat"
    r = convert(args.replay, map_dir, args.hash.lower(), args.difficulty,
                out)
    print(f"wrote {r['out']} ({r['bytes']:,} bytes): {r['frames']} frames, "
          f"{r['note_events']} note events, final score {r['final_score']}")


if __name__ == "__main__":
    main()
