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
  combo, multiplier, energy, fps/extension) + blocks. We write replay
  format v2.0.0 (no v3 extension fields).

Fidelity notes:
- Pose frames are exact: wrist positions from the recorded joint state,
  blade orientation via the kinematics FK mirror, 30 fps.
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


def align_charts(header, map_dir, difficulty, speed_scale):
    """Re-parse the original map and align its notes 1:1 (by order) with
    the replay's chart. Returns (real_notes, time_offset sim->song)."""
    sim_notes, _ = load_beatmap(map_dir, difficulty,
                                speed_scale=speed_scale, first_note_s=2.0)
    real_notes, _ = load_beatmap(map_dir, difficulty,
                                 speed_scale=speed_scale, first_note_s=None)
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
    offset = real_notes[0].hit_time - sim_notes[0].hit_time
    return real_notes, offset


# -- block builders ----------------------------------------------------------

def metadata_block(song_hash, difficulty):
    b = bytearray()
    b = w_str(b, "2.0.0")
    b = w_str(b, f"custom_level_{song_hash.upper()}")
    b = w_int(b, DIFF_RAW[difficulty])
    b = w_str(b, "Standard")
    b = w_str(b, "DefaultEnvironment")
    b = w_int(b, 0)                                # modifiers: empty array
    b = w_float(b, 0.0)                            # NoteSpawnOffset
    b = w_bool(b, False)                           # leftHanded
    b = w_float(b, 1.8)                            # initialHeight
    b += b"\x00" * 16                              # RoomRotation + RoomCenter
    b = w_float(b, 0.0)                            # failTime
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
        b = w_vec3(b, HEAD_POS)
        b += struct.pack("<4f", 0.0, 0.0, 0.0, 1.0)
        for pos, quat in hands:
            b = w_vec3(b, pos)
            b += struct.pack("<4f", *quat)
        b = w_int(b, 30)                           # fps
        b = w_float(b, t)
    return bytes(b)


def tip_speed(ticks, tick_idx, hand):
    """Blade-tip speed (m/s) at a tick from neighboring recorded states."""
    def tip(i):
        s = ticks[i]["sabers"][hand]
        _, _, blade = fk(hand, s[0:6])
        w = s[12:15]
        return tuple(w[j] + blade[j] for j in range(3))
    i = max(tick_idx, 1)
    a, c = tip(i - 1), tip(i)
    return math.dist(a, c) * 30.0


def build_events(ticks, real_notes, offset):
    """Note-event dicts + the score/combo/multiplier HUD timelines.

    The live viewer drives its HUD from the combo/multiplier keyframe
    blocks (layouts from ScoreSaber's own plugin writer: ComboEvent =
    int combo + float time; MultiplierEvent = int multiplier + float
    bar-progress + float time), so every scoring event emits keyframes.
    """
    sts = [scoring_type(n) for n in real_notes]
    walk = scoresaber.ComboTracker()
    events, timeline = [], []
    combo_tl = [(0, 0.0)]
    mult_tl = [(1, 0.0, 0.0)]
    by_note = {}

    def hud(t):
        combo_tl.append((walk.combo, t))
        mult_tl.append((walk.multiplier, walk.progress_fraction, t))
    for tick_idx, line in enumerate(ticks):
        t = line["time"] + offset
        for ev in line["events"]:
            kind, body = next(iter(ev.items()))
            i = body["note"]
            n = real_notes[i]
            if kind == "post":
                walk.on_points(i, body["points"])
                timeline.append((round(walk.score), t))
                if i in by_note:
                    by_note[i]["after"] = body["points"] / 30.0
                continue
            e = {"note": i, "st": sts[i], "line": n.col, "layer": n.row,
                 "color": 3 if n.kind == KIND_BOMB else n.color,
                 "dir": 3 if n.kind == KIND_BOMB else
                        (8 if n.any_dir else (n.raw_dir
                                              if n.raw_dir is not None else 8)),
                 "spawn": n.hit_time, "time": t,
                 "before": 0.0, "after": 0.0, "dist": 0.3,
                 "speed": 0.0, "dir_ok": False, "saber": n.color,
                 "cut_pos": (note_tuple(n)[1], note_tuple(n)[2], 0.0),
                 "saber_dir": (0.0, 1.0, 0.0)}
            if kind == "miss":
                e["type"] = 3
                walk.on_break()
                hud(t)
            elif kind == "bomb":
                e["type"] = 4
                e["saber"] = 0
                walk.on_break()
                timeline.append((round(walk.score), t))
                hud(t)
            else:                                  # cut: good or bad
                good = body["good"]
                reason = body["reason"]
                hand = n.color if reason != "wrong_hand" else 1 - n.color
                s = ticks[tick_idx]["sabers"][hand]
                _, _, blade = fk(hand, s[0:6])
                e["saber"] = hand
                e["saber_dir"] = tuple(blade)
                e["speed"] = max(tip_speed(ticks, tick_idx, hand),
                                 2.5 if reason != "slow" else 0.0)
                e["dir_ok"] = reason not in ("angle",)
                if good:
                    e["type"] = 1
                    walk.on_cut(i)
                    walk.on_points(i, body["points"])
                    timeline.append((round(walk.score), t))
                    hud(t)
                    pts = round(body["points"])
                    if sts[i] in (ST_CHAINLINK, ST_CHAINLINKARCHEAD):
                        pass                       # flat 20, ratings unused
                    else:
                        pre = min(pts, 70)
                        acc = max(0, min(pts - pre, 15))
                        e["before"] = pre / 70.0
                        e["dist"] = 0.3 * (1.0 - acc / 15.0)
                    by_note[i] = e
                else:
                    e["type"] = 2
                    walk.on_break()
                    timeline.append((round(walk.score), t))
                    hud(t)
            events.append(e)
    events.sort(key=lambda e: e["time"])
    return events, timeline, combo_tl, mult_tl, round(walk.score)


def note_block(events):
    b = bytearray()
    b = w_int(b, len(events))
    for e in events:
        b = w_float(b, e["spawn"])                 # noteTime (their spawnTime)
        b = w_int(b, e["layer"])
        b = w_int(b, e["line"])
        b = w_int(b, e["color"])
        b = w_int(b, e["dir"])
        b = w_int(b, e["type"])                    # 1 good 2 bad 3 miss 4 bomb
        b = w_vec3(b, e["cut_pos"])                # cutPoint
        normal = (-e["saber_dir"][1], e["saber_dir"][0], 0.0)
        nn = math.sqrt(sum(c * c for c in normal)) or 1.0
        b = w_vec3(b, tuple(c / nn for c in normal))   # cutNormal (z=0)
        b = w_vec3(b, e["saber_dir"])
        b = w_int(b, e["saber"])
        b = w_bool(b, e["dir_ok"])
        b = w_float(b, e["speed"])
        b = w_float(b, 100.0)                      # cutAngle (display only)
        b = w_float(b, e["dist"])
        b = w_float(b, 0.0)                        # cutDirectionDeviation
        b = w_float(b, e["before"])
        b = w_float(b, e["after"])
        b = w_float(b, e["time"])                  # eventTime
        b = w_float(b, 1.0)                        # unityTimescale
        b = w_float(b, 1.0)                        # timeSyncTimescale
    return bytes(b)


def score_block(timeline):
    b = bytearray()
    b = w_int(b, len(timeline))
    for score, t in timeline:
        b = w_int(b, score)
        b = w_float(b, t)
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


def energy_block(end_time):
    b = bytearray()                      # no energy/fail in our sim:
    b = w_int(b, 2)                      # full bar for the whole song
    for t in (0.0, end_time):
        b = w_float(b, 1.0)
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


def convert(replay_path, map_dir, song_hash, difficulty, out_path,
            speed_scale=0.3):
    header, ticks = read_replay(replay_path)
    real_notes, offset = align_charts(header, map_dir, difficulty, speed_scale)
    events, timeline, combo_tl, mult_tl, final = build_events(
        ticks, real_notes, offset)
    blocks = [
        metadata_block(song_hash, difficulty),
        pose_block(ticks, offset),
        EMPTY,                                     # heights
        note_block(events),
        score_block(timeline),
        combo_block(combo_tl),
        multiplier_block(mult_tl),
        energy_block(ticks[-1]["time"] + offset),
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
    p.add_argument("--speed-scale", type=float, default=0.3)
    args = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    map_dir = args.map or os.path.join(here, "maps", "scoresaber",
                                       args.hash.lower())
    out = args.out or os.path.splitext(args.replay)[0] + ".dat"
    r = convert(args.replay, map_dir, args.hash.lower(), args.difficulty,
                out, args.speed_scale)
    print(f"wrote {r['out']} ({r['bytes']:,} bytes): {r['frames']} frames, "
          f"{r['note_events']} note events, final score {r['final_score']}")


if __name__ == "__main__":
    main()
