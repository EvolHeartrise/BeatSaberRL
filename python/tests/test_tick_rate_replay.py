"""exp/tick-rate replays (2026-09-29): the recorder (RustBeatEnv) and the
ScoreSaber converter at a non-default control rate.

- the replay header carries tick_hz / action_repeat (record_rev
  6) and the per-tick `time` stamps are k / tick_hz;
- action_repeat records every SIM tick, with the same trajectory and
  events as stepping the held action by hand;
- convert_scoresaber writes fps = tick_hz, folds the +1/tick_hz end-of-tick
  shift into its offset, scales the tip velocity and the per-tick wall
  drain by the rate, and its decode reproduces the score.
"""
import json
import math

import numpy as np
import pytest

import convert_scoresaber as conv
from beat_arena_env import scoresaber
from beat_arena_env.beatsaver import load_beatmap
from beat_arena_env.replay_format import read_replay
from beat_arena_env.rust_env import RustBeatEnv
from test_convert_scoresaber import (
    BOMB, FIRST_NOTE_S, NJS, RAW_NOTES, decode_dat)


def _write_map(tmp):
    info = {"_beatsPerMinute": 120.0, "_songName": "fixture",
            "_difficultyBeatmapSets": [{
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [
                    {"_difficulty": "Expert", "_beatmapFilename": "Expert.dat",
                     "_noteJumpMovementSpeed": NJS}]}]}
    (tmp / "Info.dat").write_text(json.dumps(info))
    v2 = {"_version": "2.0.0", "_notes":
          [{"_time": t, "_lineIndex": x, "_lineLayer": y, "_type": c,
            "_cutDirection": d} for t, x, y, c, d in RAW_NOTES]
          + [{"_time": BOMB[0], "_lineIndex": BOMB[1], "_lineLayer": BOMB[2],
              "_type": 3, "_cutDirection": 0}]}
    (tmp / "Expert.dat").write_text(json.dumps(v2))


def _whip(t_real):
    """The converter fixture's whip, on REAL time (so it is the same
    motion schedule at any rate): raise 1/3 s, whip down until 1 s,
    then coast."""
    a = np.zeros(29, dtype=np.int64)
    if t_real < 10 / 30:
        a[10] = 1
    elif t_real < 30 / 30:
        a[10] = 2
    return a


def _record(tmp, tick_hz, repeat=1, name="ep"):
    notes, _ = load_beatmap(str(tmp), "Expert", first_note_s=FIRST_NOTE_S)
    env = RustBeatEnv(external_chart=notes, seed=0, tick_hz=tick_hz,
                      action_repeat=repeat)
    path = str(tmp / f"{name}.rpl.gz")
    env.set_record(path)
    env.reset()
    events, done, k = [], False, 0
    while not done:
        _, _, done, _, info = env.step(_whip(k * repeat / tick_hz))
        events.extend(info["events"])
        k += 1
    head, ticks = read_replay(path)
    return path, head, ticks, events


@pytest.fixture(scope="module")
def rec60(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tick60")
    _write_map(tmp)
    path, head, ticks, events = _record(tmp, 60)
    out = str(tmp / "ep.dat")
    r = conv.convert(path, str(tmp), "ab" * 20, "Expert", out,
                     first_note_s=FIRST_NOTE_S)
    return {"tmp": tmp, "path": path, "head": head, "ticks": ticks,
            "events": events, "result": r, "decoded": decode_dat(out)}


# --- recorder --------------------------------------------------------------

def test_header_carries_the_rate(rec60):
    h = rec60["head"]
    assert h["record_rev"] == 6
    assert h["tick_hz"] == 60 and h["action_repeat"] == 1
    assert "obs_window" not in h


def test_tick_times_are_sim_ticks(rec60):
    ticks = rec60["ticks"]
    assert [t["t"] for t in ticks] == list(range(len(ticks)))
    for t in ticks:
        assert t["time"] == t["t"] / 60.0


def test_default_rate_header(tmp_path):
    _write_map(tmp_path)
    _, h, ticks, _ = _record(tmp_path, 30)
    assert h["tick_hz"] == 30 and h["action_repeat"] == 1
    for t in ticks:
        assert t["time"] == t["t"] / 30.0


def test_action_repeat_records_every_sim_tick(tmp_path):
    """repeat=2 at 60 Hz: one step() = two recorded sim ticks, and the
    recording equals holding each action for two plain steps."""
    _write_map(tmp_path)
    _, h2, t2, ev2 = _record(tmp_path, 60, repeat=2, name="rep2")
    assert h2["tick_hz"] == 60 and h2["action_repeat"] == 2

    notes, _ = load_beatmap(str(tmp_path), "Expert",
                            first_note_s=FIRST_NOTE_S)
    env = RustBeatEnv(external_chart=notes, seed=0, tick_hz=60)
    path = str(tmp_path / "held.rpl.gz")
    env.set_record(path)
    env.reset()
    ev1, done, k = [], False, 0
    while not done:
        _, _, done, _, info = env.step(_whip((k // 2) * 2 / 60))
        ev1.extend(info["events"])
        k += 1
    t1 = read_replay(path)[1]
    assert len(t1) == len(t2)
    for a, b in zip(t1, t2):
        assert a["q"] == b["q"] and a["events"] == b["events"]
        assert a["action"] == b["action"]
    assert ev1 == ev2


# --- converter at 60 fps ---------------------------------------------------

def test_converter_fps_is_tick_hz(rec60):
    frames = rec60["decoded"]["frames"]
    assert rec60["result"]["fps"] == 60
    assert len(frames) == len(rec60["ticks"])
    assert all(f["fps"] == 60 for f in frames)
    times = [f["time"] for f in frames]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_converter_offset_is_one_60hz_tick(rec60):
    """+DT with DT = 1/60: every pose/event time = tick_time +
    (first_real_hit - FIRST_NOTE_S) + 1/60."""
    offset = 1.5 - FIRST_NOTE_S + 1.0 / 60.0
    ticks = rec60["ticks"]
    want = sorted(t["time"] + offset for t in ticks
                  for ev in t["events"] if "post" not in ev)
    got = sorted(e["time"] for e in rec60["decoded"]["notes"])
    assert len(got) == len(want) and got
    for a, b in zip(got, want):
        assert a == pytest.approx(b, abs=1e-4)
    for f, t in zip(rec60["decoded"]["frames"], ticks):
        expect = t["time"] + offset
        if expect > 1e-3:
            assert f["time"] == pytest.approx(expect, abs=1e-4)


def test_converter_score_decodes(rec60):
    walk = scoresaber.ComboTracker()
    walk.consume(rec60["events"])
    d = rec60["decoded"]
    assert d["final_score"] == round(walk.score) == rec60["result"]["final_score"]
    assert any(e["type"] == 1 for e in d["notes"]), "fixture lost its cuts"


def test_cut_speed_uses_the_rate(rec60):
    """Tip speed = one-tick tip displacement x 60 at 60 fps."""
    ticks = rec60["ticks"]
    real_notes, _ = load_beatmap(str(rec60["tmp"]), "Expert", first_note_s=None)
    for k, t in enumerate(ticks):
        for ev in t["events"]:
            if "cut" in ev:
                n = real_notes[ev["cut"]["note"]]
                g60 = conv.cut_geometry(ticks, k, n.color, n, 60)
                g30 = conv.cut_geometry(ticks, k, n.color, n, 30)
                assert g60["speed"] == pytest.approx(2 * g30["speed"])
                s = ticks[k]["sabers"][n.color]
                p = ticks[max(k - 1, 0)]["sabers"][n.color]
                d = math.dist([s[j] + s[3 + j] for j in range(3)],
                              [p[j] + p[3 + j] for j in range(3)])
                assert g60["speed"] == pytest.approx(d * 60)
                return
    pytest.fail("no cut in the 60 Hz fixture")


def test_wall_drain_is_per_tick_of_the_rate():
    """The obstacle drain is 1.3/s: 1.3/60 per 60 Hz tick."""
    def ticks(n):
        return [{"time": k / 60, "events": [{"wall": {}}], "sabers": []}
                for k in range(n)]
    *_, e60, _ = conv.build_events(ticks(12), [], 0.0, fps=60)
    assert e60[-1][0] == pytest.approx(0.5 - 12 * 1.3 / 60)
    *_, e30, _ = conv.build_events(ticks(6), [], 0.0, fps=30)
    assert e30[-1][0] == pytest.approx(e60[-1][0])
