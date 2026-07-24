"""ScoreSaber .dat converter: binary round-trip against a Python port of
the watch.scoresaber.com decoder, plus their note-matching rules.

The decoder port mirrors ScoreSaberDecoder.cs (v2 path) from the
ScoreSaber/watch repo — field-for-field. If the converter and this port
disagree with each other, one of them diverged from the real format.
"""

import json
import lzma
import math
import struct

import numpy as np
import pytest

import convert_scoresaber as conv
from beat_arena_env import scoresaber
from beat_arena_env.beatsaver import load_beatmap
from beat_arena_env.rust_env import RustBeatEnv


# --- decoder port (their ScoreSaberDecoder.cs, v2) --------------------------

def _r(fmt, data, off):
    vals = struct.unpack_from("<" + fmt, data, off)
    return vals, off + struct.calcsize("<" + fmt)


def _r_str(data, off):
    (n,), off = _r("i", data, off)
    return data[off:off + n].decode("utf-8"), off + n


def decode_dat(path):
    raw = open(path, "rb").read()
    assert raw[:28] == conv.MAGIC, "bad magic"
    # their reader: 5-byte props, 8-byte uncompressed size, LZMA stream.
    # The size MUST be real (not the -1 "unknown" python writes) — the
    # live viewer rejects unknown sizes as "replay is too large".
    props, dict_size = raw[28], struct.unpack_from("<I", raw, 29)[0]
    size = struct.unpack_from("<Q", raw, 33)[0]
    assert props == 0x5D and size < (1 << 31), "bad LZMA-alone header"
    dec = lzma.LZMADecompressor(
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA1, "dict_size": dict_size,
                  "lc": props % 9, "lp": (props // 9) % 5,
                  "pb": props // 45}])
    data = dec.decompress(raw[41:])
    assert len(data) == size, "size field does not match decoded length"
    ptrs = struct.unpack_from("<9i", data, 0)
    meta_p, pose_p, height_p, note_p, score_p = ptrs[:5]

    off = meta_p
    version, off = _r_str(data, off)
    level_id, off = _r_str(data, off)
    (difficulty,), off = _r("i", data, off)
    characteristic, off = _r_str(data, off)
    environment, off = _r_str(data, off)
    (n_mods,), off = _r("i", data, off)
    for _ in range(n_mods):
        _, off = _r_str(data, off)
    (_, ), off = _r("f", data, off)              # NoteSpawnOffset
    (left,), off = _r("?", data, off)
    (height,), off = _r("f", data, off)
    off += 16                                    # RoomRotation + RoomCenter
    (fail_time,), off = _r("f", data, off)

    off = pose_p
    (n_frames,), off = _r("i", data, off)
    frames = []
    for _ in range(n_frames):
        vals, off = _r("7f7f7fif", data, off)
        frames.append({"head": vals[0:7], "left": vals[7:14],
                       "right": vals[14:21], "fps": vals[21],
                       "time": vals[22]})

    off = height_p
    (n_heights,), off = _r("i", data, off)

    off = note_p
    (n_notes,), off = _r("i", data, off)
    notes = []
    for _ in range(n_notes):
        (spawn,), off = _r("f", data, off)
        (layer, line, color, cut_dir, ev_type), off = _r("5i", data, off)
        cut_point, off = _r("3f", data, off)
        cut_normal, off = _r("3f", data, off)
        saber_dir, off = _r("3f", data, off)
        (saber,), off = _r("i", data, off)
        (dir_ok,), off = _r("?", data, off)
        (speed, angle, dist, dev, before, after, ev_time), off = _r(
            "7f", data, off)
        off += 8                                 # timescales
        notes.append({"spawn": spawn, "layer": layer, "line": line,
                      "color": color, "dir": cut_dir, "type": ev_type,
                      "saber": saber, "dir_ok": dir_ok, "speed": speed,
                      "dist": dist, "before": before, "after": after,
                      "time": ev_time, "saber_dir": saber_dir})

    off = score_p
    (n_scores,), off = _r("i", data, off)
    final = 0
    for _ in range(n_scores):
        (final,), off = _r("i", data, off)
        (_, ), off = _r("f", data, off)

    return {"version": version, "level_id": level_id,
            "difficulty": difficulty, "characteristic": characteristic,
            "left_handed": left, "height": height, "frames": frames,
            "notes": notes, "final_score": final, "n_heights": n_heights}


# --- fixture map + recorded episode -----------------------------------------

RAW_NOTES = [
    # (_time beats@120, line, layer, type(color), cutDirection)
    (3.0, 1, 0, 1, 1),      # down-swing, right/blue
    (5.0, 2, 0, 1, 1),
    (7.0, 3, 2, 1, 8),      # dot
    (9.0, 0, 0, 0, 1),      # left/red
]
BOMB = (6.0, 0, 2)


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ssmap")
    info = {"_beatsPerMinute": 120.0, "_songName": "fixture",
            "_difficultyBeatmapSets": [{
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [
                    {"_difficulty": "Expert", "_beatmapFilename": "Expert.dat",
                     "_noteJumpMovementSpeed": 10.0}]}]}
    (tmp / "Info.dat").write_text(json.dumps(info))
    v2 = {"_version": "2.0.0", "_notes":
          [{"_time": t, "_lineIndex": x, "_lineLayer": y, "_type": c,
            "_cutDirection": d} for t, x, y, c, d in RAW_NOTES]
          + [{"_time": BOMB[0], "_lineIndex": BOMB[1], "_lineLayer": BOMB[2],
              "_type": 3, "_cutDirection": 0}]}
    (tmp / "Expert.dat").write_text(json.dumps(v2))

    notes, _ = load_beatmap(str(tmp), "Expert", speed_scale=0.3,
                            first_note_s=2.0)
    env = RustBeatEnv(external_chart=notes, seed=0)
    replay_path = str(tmp / "ep.jsonl")
    env.set_record(replay_path)
    obs, _ = env.reset()
    rng = np.random.default_rng(5)
    events, done = [], False
    while not done:
        _, _, done, _, info_d = env.step(rng.integers(0, 3, size=12))
        events.extend(info_d["events"])

    out = str(tmp / "ep.dat")
    r = conv.convert(replay_path, str(tmp), "ab" * 20, "Expert", out)
    return {"map_dir": str(tmp), "events": events, "result": r,
            "decoded": decode_dat(out)}


def test_metadata_and_magic(converted):
    d = converted["decoded"]
    assert d["version"] == "2.0.0"
    assert d["level_id"] == "custom_level_" + "AB" * 20
    assert d["difficulty"] == 7 and d["characteristic"] == "Standard"
    assert not d["left_handed"] and d["n_heights"] == 0


def test_frames_wellformed(converted):
    frames = converted["decoded"]["frames"]
    assert converted["result"]["frames"] == len(frames)
    times = [f["time"] for f in frames]
    assert times[0] > 0 and all(b > a for a, b in zip(times, times[1:]))
    # fixture first note: 3 beats @120 = 1.5 s real vs 2.0 s sim -> -0.5 s
    # offset; the last frame must land near (episode_len - 0.5)
    for f in frames:
        assert f["fps"] == 30
        for key in ("head", "left", "right"):
            q = f[key][3:7]
            assert math.fsum(c * c for c in q) == pytest.approx(1.0, abs=1e-4)
    assert frames[0]["head"][:3] == pytest.approx((0.0, 1.70, -0.65))


def test_note_events_match_their_rules(converted):
    """Their ObjectManager matching: ID = st*10000 + line*1000 + layer*100
    + color*10 + dir, events grouped by spawnTime within 1 ms; bombs match
    ignoring the direction digit. Every scoreable fixture note must find
    exactly one event; the bomb only if it was hit."""
    d = converted["decoded"]
    events = list(d["notes"])
    assert converted["result"]["note_events"] == len(events)

    def take(spawn_s, want_id, bomb=False):
        for e in events:
            if abs(e["spawn"] - spawn_s) > 1e-3:
                continue
            got = (e["line"] * 1000 + e["layer"] * 100 + e["color"] * 10
                   + e["dir"])
            if bomb:
                got -= got % 10
            if got == want_id:
                events.remove(e)
                return e
        return None

    for t_beats, x, y, c, dd in RAW_NOTES:
        spawn = t_beats * 0.5                    # 120 bpm -> 0.5 s per beat
        e = take(spawn, x * 1000 + y * 100 + c * 10 + dd)
        assert e is not None, f"no event matched note at {spawn}s"
        assert e["type"] in (1, 2, 3)
    n_bomb_hits = sum(1 for ev in converted["events"] if ev[1] == "bomb")
    bomb_ev = take(BOMB[0] * 0.5, BOMB[1] * 1000 + BOMB[2] * 100 + 3 * 10,
                   bomb=True)
    assert (bomb_ev is not None) == (n_bomb_hits > 0)
    if bomb_ev:
        assert bomb_ev["type"] == 4
    assert not events, f"{len(events)} unmatched events left over"


def test_score_reproduction(converted):
    """Final score equals the bench ComboTracker on the env's events, and
    their per-note recompute round(70*before) + round(30*after) +
    round(15*(1-dist/0.3)) reproduces every good cut's points."""
    walk = scoresaber.ComboTracker()
    walk.consume(converted["events"])
    d = converted["decoded"]
    assert d["final_score"] == round(walk.score)
    assert d["final_score"] == converted["result"]["final_score"]

    cut_pts = {}
    for i, reason, pts in converted["events"]:
        if reason == "ok":
            cut_pts[i] = cut_pts.get(i, 0.0) + pts
        elif reason == "post":
            cut_pts[i] = cut_pts.get(i, 0.0) + pts
    for e in d["notes"]:
        if e["type"] != 1:
            continue
        recomputed = (round(70 * min(e["before"], 1.0))
                      + round(30 * min(e["after"], 1.0))
                      + round(15 * (1 - min(e["dist"] / 0.3, 1.0))))
        # exactly one good note per spawn in this fixture
        want = [round(v) for k, v in cut_pts.items()]
        assert recomputed in want


def test_quat_from_forward():
    def rotate(q, v):
        x, y, z, w = q
        u = (x, y, z)
        c = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
             u[0] * v[1] - u[1] * v[0])
        dot = sum(a * b for a, b in zip(u, v))
        return tuple(
            2 * dot * u[i] + (w * w - sum(a * a for a in u)) * v[i]
            + 2 * w * c[i] for i in range(3))

    for d in [(0, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, -1),
              (0.6, 0.48, 0.64)]:
        n = math.sqrt(sum(c * c for c in d))
        d = tuple(c / n for c in d)
        q = conv.quat_from_forward(d)
        got = rotate(q, (0, 0, 1))
        assert got == pytest.approx(d, abs=1e-6)


def test_scoring_type_mapping():
    from beat_arena_env.chart import Note
    from beat_arena_env.constants import (KIND_BOMB, KIND_CHAIN_HEAD,
                                          KIND_CHAIN_LINK)

    def note(kind=0, pre=False, post=False):
        return Note(hit_time=1.0, col=0, row=0, color=1, angle=0.0,
                    speed=4.0, kind=kind, free_pre=pre, free_post=post)

    assert conv.scoring_type(note()) == conv.ST_NOTE
    assert conv.scoring_type(note(post=True)) == conv.ST_ARCHEAD
    assert conv.scoring_type(note(pre=True)) == conv.ST_ARCTAIL
    assert conv.scoring_type(note(pre=True, post=True)) == conv.ST_ARCHEADARCTAIL
    assert conv.scoring_type(note(KIND_CHAIN_HEAD)) == conv.ST_CHAINHEAD
    assert conv.scoring_type(note(KIND_CHAIN_HEAD, post=True)) == conv.ST_CHAINHEADARCHEAD
    assert conv.scoring_type(note(KIND_CHAIN_LINK)) == conv.ST_CHAINLINK
    assert conv.scoring_type(note(KIND_BOMB)) == conv.ST_NOSCORE
