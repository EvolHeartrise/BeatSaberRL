"""Replay format 2 (beat_arena_env/replay_format.py): decode(encode(capture))
reproduces the sim's per-tick record within each field's quantization,
the derived fields (sabers / head by FK, activation / torque from the
header's body tables) within their write-time tolerances, a derivation
that fails its check is stored instead, and the viewer's committed decoder
fixtures are current. CPU only."""
import os
import sys

import numpy as np
import pytest

import beat_arena_rs
from beat_arena_env import replay_format as rf
from beat_arena_env.beatsaver import load_beatmap
from beat_arena_env.chart import note_tuple, wall_tuple
from beat_arena_env.constants import OBS_DIM
from beat_arena_env.rust_env import record_header
from test_replay_batched import FIRST, LEVEL, MAP, _actions, _write_map

TAU = [0.9, 1.1, 1.0, 1.2, 0.8, 1.0, 1.05, 0.95, 1.3, 1.0,
       1.1, 0.9, 1.0, 1.0, 1.15, 0.85]


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    """A 60 Hz, gears-5, non-unit-body play with cuts: the Rust recorder's
    raw capture and its header."""
    tmp = tmp_path_factory.mktemp("rf")
    _write_map(tmp / "map")
    notes, meta = load_beatmap(str(tmp / "map"), "Expert", first_note_s=FIRST)
    tup = [note_tuple(n) for n in notes]
    walls = [wall_tuple(w) for w in meta["walls"]]
    sim = beat_arena_rs.VecBeatSim(1, max_steps=200, gears=5,
                                   return_events=False, tick_hz_env=[60])
    sim.set_tau_scale_env(0, TAU)
    sim.set_chart(0, tup, walls)
    sim.set_record_env(0, True)
    obs = np.zeros((1, OBS_DIM), dtype=np.float32)
    rng, k, done = np.random.default_rng(7), 0, False
    while not done:
        a = _actions(k, rng)
        a[a == 2] = 4 * (k % 3 == 0) + 2 * (k % 3 != 0)   # fine gears too
        _, dones, _ = sim.step_into(a[None], obs)
        done = bool(dones[0])
        k += 1
    rec = sim.take_record_env(0)
    head = record_header(tup, walls, tick_hz=60, action_repeat=1,
                         tau_scale=TAU, gears=5, map=MAP, level=LEVEL)
    return head, rec


def _frame(rec):
    return np.asarray(rec["frame"]).reshape(-1, rf.REC_LANES)


def test_round_trip_within_quantization(capture):
    head, rec = capture
    st = {}
    raw = rf.encode(head, rec, stats=st)
    # activation / torque derive from header.body — incl. the bend joints'
    # 2x fine gears (FINE_GEAR_MULT), which this play exercises
    assert st["stored"] == [], st
    d = rf.decode(rf.gz(raw))
    assert np.isin(d["action"][:, [8, 15]], (3, 4)).any()
    assert head["header"]["body"]["gear_frac"][8][3] == 0.5
    fr = _frame(rec)
    T = len(fr)
    assert d["T"] == T and (d["t"] == np.arange(T)).all()
    assert np.allclose(d["time"], np.arange(T) / 60)
    # time = t / tick_hz: the sim clock (accumulated) agrees to float noise
    assert np.abs(d["time"] - (np.asarray(rec["time"]) - 1 / 60)).max() < 1e-9
    tol = 1e-9
    for name, a, b, q in (("q", rf.L_Q, rf.L_TWIST, 0.5e-4),
                          ("omega", rf.L_OMEGA, rf.L_SABERS, 0.5e-2),
                          ("twist", rf.L_TWIST, rf.L_OMEGA, 0.5e-3),
                          ("feet", rf.L_FEET, rf.L_ACT, 0.5e-3)):
        err = np.abs(d[name] - fr[:, a:b]).max()
        assert err <= q + tol, (name, err)
    assert np.abs(d["reward"] - fr[:, rf.L_REWARD]).max() <= 0.5e-4 + tol
    # derived: FK of the quantized q, effort from action x body
    s, h = d["sabers"], d["head"]
    rs, rh = fr[:, rf.L_SABERS:rf.L_HEAD], fr[:, rf.L_HEAD:rf.L_FEET]
    assert np.abs(s[:, [0, 1, 2, 6, 7, 8]] - rs[:, [0, 1, 2, 6, 7, 8]]).max() < 1e-3
    assert rf._dir_err(s[:, 3:6], rs[:, 3:6]).max() < rf.TOL_ANG
    assert np.abs(h[:, :3] - rh[:, :3]).max() < 1e-3
    assert rf._quat_err(h[:, 3:], rh[:, 3:]).max() < rf.TOL_ANG
    assert np.abs(d["activation"] - fr[:, rf.L_ACT:rf.L_TORQUE]).max() < 1e-9
    assert np.abs(d["torque"] - fr[:, rf.L_TORQUE:rf.L_REWARD]).max() < 1e-9
    assert d["torque"][:, 0].max() > 0
    # exact / variable-length fields
    assert (d["action"].ravel() == np.asarray(rec["action"])).all()
    assert (d["contact_off"] == np.asarray(rec["contact_off"])).all()
    assert (d["contact_pair"] == np.asarray(rec["contact_pair"])).all()
    cv = np.asarray(rec["contact_val"]).reshape(-1, 4)
    assert np.abs(d["contact_p"] - cv[:, :3]).max(initial=0) <= 0.5e-4 + tol
    assert (d["event_off"] == np.asarray(rec["event_off"])).all()
    assert (d["event_code"] == np.asarray(rec["event_code"])).all()
    note = np.asarray(rec["event_note"]).astype(np.int64)
    assert (d["event_note"] == np.where(note >= 2**31, -1, note)).all()
    ev = np.asarray(rec["event_val"]).reshape(-1, 3)
    assert len(ev) and np.abs(d["event_val"] - ev).max() <= 0.5e-2 + tol
    # the header rides through unchanged
    assert d["header"] == head["header"]
    assert d["header"]["format"] == 2


def test_body_tables_drive_derived_torque(capture):
    """A header whose body is not the one played: the torque check fails
    and the recorded torque is stored (activation still derives)."""
    head, rec = capture
    h = dict(head["header"])
    h["body"] = dict(h["body"], tau_max=[2 * v for v in h["body"]["tau_max"]])
    st = {}
    d = rf.decode(rf.encode({"header": h}, rec, stats=st))
    assert st["stored"] == ["torque"]
    fr = _frame(rec)
    assert np.abs(d["torque"] - fr[:, rf.L_TORQUE:rf.L_REWARD]).max() <= 0.5e-3 + 1e-9
    assert np.abs(d["activation"] - fr[:, rf.L_ACT:rf.L_TORQUE]).max() < 1e-9


def test_failed_fk_check_stores_poses(capture):
    head, rec = capture
    rec = dict(rec)
    fr = _frame(rec).copy()
    fr[:, rf.L_SABERS] += 0.01                      # 1 cm off the FK
    fr[:, rf.L_HEAD + 3:rf.L_FEET] *= -1            # same rotation, flipped sign
    rec["frame"] = fr.ravel()
    st = {}
    d = rf.decode(rf.encode(head, rec, stats=st))
    assert st["stored"] == ["sabers"], st           # q-sign is not an error
    assert np.abs(d["sabers"] - fr[:, rf.L_SABERS:rf.L_HEAD]).max() <= 0.5e-4 + 1e-9


def test_ranges_are_checked():
    w = rf._Writer()
    with pytest.raises(OverflowError):
        w.add("contact_n", [300], dtype="u8")
    with pytest.raises(OverflowError):
        w.add("q", [2**31])


def test_omega_prediction_is_round_half_up():
    om = np.array([[0, 3, -3, 7, -8, 1]])
    p = rf._omega_pred(om, 100, 60)
    s = np.array([0, 3, 0, 4, -1, -7])
    want = np.floor(s * 100 / 120 + 0.5).astype(np.int64)
    want[0] = 0
    assert (p[0] == want).all()


def test_viewer_fixtures_current():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import make_replay_fixture as mrf
    raw = mrf.stored_payload()
    assert rf.read_raw(mrf.STORED) == raw, "run tools/make_replay_fixture.py"
    import json
    assert json.load(open(mrf.REF)) == json.loads(json.dumps(
        mrf.reference(raw))), "run tools/make_replay_fixture.py"
