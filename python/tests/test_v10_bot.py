"""P6 gate — viewer/bot support layer (SPEC-v10 §7 P6, Python side).

Covers the v10 additions under the bot: the full-body FK mirror
(kinematics.fk_body vs the sim's own FK through `sabers()`), the
frame-table masses, the P6 additive
introspection (`weld_impulses`) and the additive replay fields
("feet", "activation"). The bot itself, its solvability gate and its
gravity/stance feedforward helpers were removed 2026-09-22; these
support-layer checks stay.
"""

import json
import math

import numpy as np
import pytest

from beat_arena_env import RustBeatEnv
from beat_arena_env.constants import (
    DT, JOINT_LIMITS, N_JOINTS, NV, NQPOS, ROOT_POS0, gear_table,
)
from beat_arena_env.kinematics import (
    F_HEAD, F_L_WRIST, F_R_WRIST, FRAMES, blade_of, fk_body,
)

from beat_arena_rs import RawBeatSim


def _q36(joints=None, root=None, quat=None):
    q = np.zeros(NQPOS)
    q[:3] = ROOT_POS0 if root is None else root
    q[3:7] = (1.0, 0.0, 0.0, 0.0) if quat is None else quat
    if joints is not None:
        q[7:] = joints
    return q


def _sim():
    sim = RawBeatSim(gears=5)
    sim.set_chart([(50.0, 0.3, 1.4, 1, 3 * math.pi / 2, 4.0, False, 0,
                    False, False)])
    return sim


# --- full-body FK mirror ------------------------------------------------------

def test_fk_body_matches_sim_sabers():
    """fk_body wrists/blades == the sim's FK (sabers()) to 1e-12 on
    random in-limit poses with random root pose — the P6 FK-mirror
    gate (the JS mirror is pinned the same way in the node suite)."""
    rng = np.random.default_rng(0)
    sim = _sim()
    worst = 0.0
    for _ in range(300):
        joints = np.array([rng.uniform(lo, hi) for lo, hi in JOINT_LIMITS])
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        q = _q36(joints, root=rng.uniform(-1, 1, 3), quat=quat)
        sim.set_joints(list(q), [0.0] * NV)
        sab = sim.sabers()
        rots, pos = fk_body(q)
        for hand, fw in ((0, F_L_WRIST), (1, F_R_WRIST)):
            w, b = pos[fw], blade_of(rots, hand)
            for i in range(3):
                worst = max(worst, abs(w[i] - sab[hand][i]),
                            abs(b[i] - sab[hand][3 + i]))
    assert worst <= 1e-12, worst


def test_fk_body_reference_pose():
    """Neutral pose reproduces the SPEC-v10 §1.1 anchor heights."""
    rots, pos = fk_body(_q36())
    assert np.allclose(pos[F_HEAD], (0.0, 1.70, -0.65), atol=1e-12)
    assert np.allclose(pos[F_L_WRIST], (-0.2, 1.47, 0.0), atol=1e-12)
    assert np.allclose(pos[F_R_WRIST], (0.2, 1.47, 0.0), atol=1e-12)
    assert np.allclose(blade_of(rots, 0), (0, 0, 1), atol=1e-12)
    # feet at their weld anchors
    from beat_arena_env.kinematics import F_L_FOOT, F_R_FOOT
    assert np.allclose(pos[F_L_FOOT], (-0.10, 0.07, -0.65), atol=1e-12)
    assert np.allclose(pos[F_R_FOOT], (0.10, 0.07, -0.65), atol=1e-12)


def test_frame_table_masses():
    """Mass bookkeeping: FRAMES + pelvis == the SPEC-v10 §1.2 total."""
    total = 8.40 + sum(f[4] for f in FRAMES)
    assert total == pytest.approx(75.02, abs=1e-9)


# --- P6 additive introspection + replay fields --------------------------------

def test_weld_impulses_carry_body_weight():
    """Vertical weld reaction ≈ body weight once the coasting body's
    initial transient settles — checked on the FIRST tick (the body is
    still near rest): total jy/DT within 25% of m·g, and the getter's
    shape/order matches the solver's row layout."""
    sim = _sim()
    sim.step([0] * N_JOINTS)
    w = sim.weld_impulses()
    assert len(w) == 2 and all(len(f) == 6 for f in w)
    fy = (w[0][1] + w[1][1]) / DT
    assert fy == pytest.approx(75.02 * 9.81, rel=0.25)
    assert w[0][1] > 0 and w[1][1] > 0  # upward on both feet


def test_replay_additive_fields(tmp_path):
    """Replay v10 stays v10; "feet" and "activation" ride along
    additively and reconstruct from the sim getters / the logged
    actions (activation = Σ|gear fraction| of the tick's action,
    [total, spine, arms, legs])."""
    path = str(tmp_path / "ep.jsonl")
    env = RustBeatEnv(foot_contact=False, seed=2, chart_len_s=4.0, gears=5)
    env.set_record(path)
    env.reset(seed=2)
    rng = np.random.default_rng(4)
    done = False
    while not done:
        _, _, done, _, _ = env.step(rng.integers(0, 5, N_JOINTS))
    lines = [json.loads(l) for l in open(path) if l.strip()]
    head, ticks = lines[0], lines[1:]
    assert head["header"]["version"] == "v10"
    assert head["header"]["record_rev"] >= 3
    gear = [abs(g) for g in gear_table(5)]
    for t in ticks:
        assert len(t["feet"]) == 2 and all(len(f) == 6 for f in t["feet"])
        a = t["activation"]
        assert len(a) == 4 and all(v >= 0.0 for v in a)
        act = t["action"]
        assert a[0] == pytest.approx(sum(gear[x] for x in act), abs=1e-3)
        assert a[1] == pytest.approx(sum(gear[x] for x in act[:3]), abs=1e-3)
        assert a[2] == pytest.approx(sum(gear[x] for x in act[3:17]), abs=1e-3)
        assert a[3] == pytest.approx(sum(gear[x] for x in act[17:]), abs=1e-3)
        assert "effort" not in t
    # some tick carries weight through the welds and commands torque
    assert max(t["feet"][0][1] + t["feet"][1][1] for t in ticks) > 0.5
    assert max(t["activation"][0] for t in ticks) > 0.0

