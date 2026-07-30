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
    is_v3 = version.split(".")[0] >= "3"
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
    game_version = platform = ""
    if version >= "3.1":
        game_version, off = _r_str(data, off)    # GameVersion
        _, off = _r_str(data, off)               # PluginVersion
        platform, off = _r_str(data, off)

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
        (layer, line, color, cut_dir), off = _r("4i", data, off)
        gameplay_type = st = -1
        angle_off = 0.0
        if is_v3:
            (gameplay_type, st), off = _r("2i", data, off)
            (angle_off,), off = _r("f", data, off)
        (ev_type,), off = _r("i", data, off)
        cut_point, off = _r("3f", data, off)
        cut_normal, off = _r("3f", data, off)
        saber_dir, off = _r("3f", data, off)
        (saber,), off = _r("i", data, off)
        (dir_ok,), off = _r("?", data, off)
        (speed, angle, dist, dev, before, after, ev_time), off = _r(
            "7f", data, off)
        off += 8                                 # timescales
        time_dev = 0.0
        note_rot = (0.0, 0.0, 0.0, 1.0)
        note_pos = (0.0, 0.0, 0.0)
        if is_v3:                                # 64-byte tail
            (time_dev,), off = _r("f", data, off)
            off += 32                            # World + InverseWorld rot
            note_rot, off = _r("4f", data, off)
            note_pos, off = _r("3f", data, off)
        notes.append({"spawn": spawn, "layer": layer, "line": line,
                      "color": color, "dir": cut_dir, "type": ev_type,
                      "gameplay_type": gameplay_type, "scoring_type": st,
                      "angle_off": angle_off, "saber": saber,
                      "dir_ok": dir_ok, "speed": speed, "angle": angle,
                      "dist": dist, "dev": dev, "before": before,
                      "after": after, "time": ev_time,
                      "time_dev": time_dev, "cut_point": cut_point,
                      "cut_normal": cut_normal, "saber_dir": saber_dir,
                      "note_rot": note_rot, "note_pos": note_pos})

    off = score_p
    (n_scores,), off = _r("i", data, off)
    final, scores = 0, []
    for _ in range(n_scores):
        (final,), off = _r("i", data, off)
        (st_t,), off = _r("f", data, off)
        imm_max = None
        if is_v3:
            (imm_max,), off = _r("i", data, off)
        scores.append((final, st_t, imm_max))

    # HUD keyframe blocks (layouts from ScoreSaber's plugin writer)
    off = ptrs[5]
    (n_combo,), off = _r("i", data, off)
    combos = []
    for _ in range(n_combo):
        (c, t), off = _r("if", data, off)
        combos.append((c, t))
    off = ptrs[6]
    (n_mult,), off = _r("i", data, off)
    mults = []
    for _ in range(n_mult):
        (m, p, t), off = _r("iff", data, off)
        mults.append((m, p, t))
    off = ptrs[7]
    (n_energy,), off = _r("i", data, off)
    energies = []
    for _ in range(n_energy):
        (e, t), off = _r("ff", data, off)
        energies.append((e, t))

    return {"version": version, "level_id": level_id,
            "difficulty": difficulty, "characteristic": characteristic,
            "left_handed": left, "height": height, "frames": frames,
            "notes": notes, "final_score": final, "n_heights": n_heights,
            "combos": combos, "multipliers": mults, "energies": energies,
            "scores": scores, "game_version": game_version,
            "platform": platform}


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
    with open(replay_path) as f:
        ticks = [json.loads(l) for l in f][1:]
    return {"map_dir": str(tmp), "events": events, "result": r,
            "ticks": ticks, "decoded": decode_dat(out)}


def test_metadata_and_magic(converted):
    d = converted["decoded"]
    assert d["version"] == "3.1.0"
    assert d["level_id"] == "custom_level_" + "AB" * 20
    assert d["difficulty"] == 7 and d["characteristic"] == "Standard"
    assert not d["left_handed"] and d["n_heights"] == 0
    # BODY height (viewer lowers all objects by (height - 1.8)/2, so the
    # env's 1.8 m player must write 1.80 here, not the 1.70 head anchor)
    assert d["height"] == pytest.approx(1.80)
    assert d["platform"] == "PC" and d["game_version"]


def test_v3_fields(converted):
    """v3 additions: gameplay/scoring types, timeDeviation, note
    transforms, real cut geometry, ImmediateMaxPossibleScore."""
    d = converted["decoded"]
    for e in d["notes"]:
        if e["type"] == 4:
            assert e["gameplay_type"] == 1 and e["scoring_type"] == 0
        else:
            assert e["gameplay_type"] == 0 and e["scoring_type"] == 1
        if e["type"] in (1, 2):                     # actual cuts
            # timeDeviation = note time - cut time (early positive)
            assert e["time_dev"] == pytest.approx(
                e["spawn"] - e["time"], abs=1e-4)
            n = math.fsum(c * c for c in e["cut_normal"])
            assert n == pytest.approx(1.0, abs=1e-4)
            assert e["angle"] == pytest.approx(90.0 + e["dev"], abs=1e-3)
            assert e["speed"] > 0.0
        else:                                       # misses: zeroed cut info
            assert e["speed"] == 0.0 and e["before"] == 0.0
        q = e["note_rot"]
        assert math.fsum(c * c for c in q) == pytest.approx(1.0, abs=1e-4)
        # player space: the note-arrival plane sits at +0.65, not 0
        assert e["note_pos"][2] == pytest.approx(0.65)

    scores = d["scores"]
    assert scores, "score block empty"
    imm = [s[2] for s in scores]
    assert all(v is not None for v in imm)
    assert all(b >= a for a, b in zip(imm, imm[1:]))        # monotone
    assert all(s[0] <= s[2] for s in scores)                # score <= max
    # change-only: consecutive scores always differ
    vals = [s[0] for s in scores]
    assert all(b != a for a, b in zip(vals, vals[1:]))


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
    # PLAYER SPACE: head at z = 0 (the viewer parents poses under a rig
    # 0.65 m back and expects notes to arrive at head.z + 0.65; writing
    # env world coords here shifted the whole avatar behind the platform)
    assert frames[0]["head"][:3] == pytest.approx((0.0, 1.70, 0.0))
    # saber frames = recorded wrists shifted env world -> player space
    ticks = converted["ticks"]
    for i in (0, len(ticks) // 2, len(ticks) - 1):
        for hand, key in ((0, "left"), (1, "right")):
            wrist = ticks[i]["sabers"][hand][12:15]
            want = (wrist[0], wrist[1], wrist[2] + 0.65)
            assert frames[i][key][:3] == pytest.approx(want, abs=1e-5)


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


def test_hud_timelines(converted):
    """Combo/multiplier/energy keyframes mirror the game recorder's
    CHANGE-ONLY emission, verified against a real ScoreSaber replay:
    no t=0 combo/multiplier seeds, the multiplier stream goes silent at
    a held 8x, energy starts (0.5, 0) and follows the real damage/heal
    constants (viewer's own recompute: +0.01 good, -0.1 bad,
    -0.15 miss/bomb; chain links 0.002/0.025/0.03), capped at 1.0."""
    d = converted["decoded"]
    walk = scoresaber.ComboTracker()
    walk.consume(converted["events"])

    combos = d["combos"]
    assert combos, "combo block empty"
    assert combos[0][1] > 0.0                    # no t=0 seed, like real files
    assert combos[-1][0] == walk.combo
    for (c0, t0), (c1, t1) in zip(combos, combos[1:]):
        assert c1 != c0 and t1 >= t0             # change-only, ordered
    if walk.breaks:
        assert any(c == 0 for c, _ in combos)

    mults = d["multipliers"]
    assert all(m in (1, 2, 4, 8) and 0.0 <= p <= 1.0 for m, p, _ in mults)
    for a, b in zip(mults, mults[1:]):
        assert a[:2] != b[:2]                    # change-only
    assert mults[-1][0] == walk.multiplier

    energies = d["energies"]
    assert energies[0] == (0.5, 0.0)             # real-replay convention
    e = 0.5
    good = bad = 0
    for _, r, _p in converted["events"]:
        if r == "ok":
            e = min(e + 0.01, 1.0)
        elif r in ("slow", "angle", "wrong_hand"):
            e = max(e - 0.1, 0.001)
        elif r in ("miss", "bomb"):
            e = max(e - 0.15, 0.001)
    assert energies[-1][0] == pytest.approx(e, abs=1e-5)
    assert all(0.0 < v <= 1.0 for v, _ in energies)
    for (a, _), (b, _) in zip(energies, energies[1:]):
        assert a != b                            # change-only


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
