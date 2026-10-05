"""Mirror operator pins (sim/mirror-exact, 2026-10-04).

- the operator itself: M on obs / actions / pose / velocity / chart tuples
  is an involution;
- M(obs(s)) == obs(M s): the obs writer commutes with the reflection,
  bit for bit, over random poses and velocities with a non-symmetric
  chart and walls in view;
- env(M s, M a) == M env(s, a): one physics tick commutes with the
  reflection to rounding level (the full statistics live in the Rust
  suite, rust/src/mirror_tests.rs — this is the binding-level smoke).
"""

import math

import numpy as np
import pytest

from beat_arena_env import constants as K
from beat_arena_env.mirror import mirror_tuples, mirror_walls
from beat_arena_env.mirror_ops import (
    JOINT_PERM, JOINT_SIGN, OBS_SIGN, OBS_SRC, m_action, m_angle, m_joint,
    m_obs, m_qpos, m_qvel)

rs = pytest.importorskip("beat_arena_rs")

LIMITS = np.array(K.JOINT_LIMITS, dtype=np.float64)


def _chart(rng, n=40):
    xs = (-0.9, -0.3, 0.3, 0.9)
    out = []
    for i in range(n):
        kind = int(rng.choice([K.KIND_NOTE, K.KIND_NOTE, K.KIND_BOMB,
                               K.KIND_CHAIN_HEAD, K.KIND_CHAIN_LINK]))
        out.append((0.3 + 0.05 * i, float(rng.choice(xs)) * float(rng.uniform(0.2, 1.0)),
                    float(rng.choice(K.ROW_Y)), int(rng.integers(0, 2)),
                    float(rng.uniform(0, 2 * math.pi)), 14.0,
                    bool(rng.random() < 0.2), kind,
                    bool(rng.random() < 0.2), bool(rng.random() < 0.2)))
    return out


WALLS = [(0.5, -0.6, 1.5, 0.6, 2.0, 1.0, 14.0),
         (1.0, 0.3, 2.2, 1.2, 0.5, 2.0, 14.0)]


def test_operator_is_an_involution():
    rng = np.random.default_rng(0)
    o = rng.normal(size=(5, K.OBS_DIM)).astype(np.float32)
    assert np.array_equal(m_obs(m_obs(o)), o)
    assert sorted(OBS_SRC) == list(range(K.OBS_DIM))
    assert set(np.unique(OBS_SIGN)) <= {-1.0, 1.0}
    a = rng.integers(0, 7, size=(5, 29))
    assert np.array_equal(m_action(m_action(a)), a)
    qp = rng.normal(size=36)
    assert np.array_equal(m_qpos(m_qpos(qp)), qp)
    v = rng.normal(size=35)
    assert np.array_equal(m_qvel(m_qvel(v)), v)
    assert np.array_equal(JOINT_PERM[JOINT_PERM], np.arange(29))
    assert np.array_equal(JOINT_SIGN[JOINT_PERM], JOINT_SIGN)
    ch = _chart(rng)
    back = mirror_tuples(mirror_tuples(ch))
    assert np.allclose(np.asarray(back, dtype=float), np.asarray(ch, dtype=float),
                       rtol=0, atol=1e-12)
    assert mirror_walls(mirror_walls(WALLS)) == WALLS
    for th in rng.uniform(0, 2 * math.pi, 16):
        assert m_angle(m_angle(th)) == pytest.approx(th, abs=1e-12)


def test_joint_tables_are_mirror_consistent():
    """The limit table under the operator: equal across a pair, negated
    and swapped where the coordinate flips sign."""
    for j in range(29):
        m = JOINT_PERM[j]
        lo, hi = LIMITS[j]
        if JOINT_SIGN[j] > 0:
            assert (lo, hi) == pytest.approx(tuple(LIMITS[m]))
        else:
            assert (lo, hi) == pytest.approx((-LIMITS[m][1], -LIMITS[m][0]))


def _random_state(rng):
    q = rng.uniform(LIMITS[:, 0], LIMITS[:, 1])
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    qp = np.concatenate([[0.1 * rng.normal(), 0.93, -0.65 + 0.1 * rng.normal()],
                         quat, q])
    v = rng.normal(size=35)
    return qp, v


def test_obs_commutes_with_the_mirror():
    """M(obs(s)) == obs(M s), bitwise: kinematics, the head-relative
    entity frame, the nearest-first entity window, colour / cut-dir lanes
    and every per-hand block are mirror-exact."""
    rng = np.random.default_rng(1)
    ch = _chart(rng)
    A = rs.RawBeatSim(max_steps=100000, gears=5)
    B = rs.RawBeatSim(max_steps=100000, gears=5)
    A.set_chart(ch, WALLS)
    B.set_chart(mirror_tuples(ch), mirror_walls(WALLS))
    for _ in range(100):
        qp, v = _random_state(rng)
        A.set_joints(list(qp), list(v))
        B.set_joints(list(m_qpos(qp)), list(m_qvel(v)))
        oa = np.asarray(A.obs())
        assert oa[K.OFF_NOTE_MASK:K.OFF_NOTE_MASK + K.N_NOTE_SLOTS].sum() > 0
        assert np.array_equal(np.asarray(B.obs()), m_obs(oa))
        sa, sb = np.asarray(A.sabers()), np.asarray(B.sabers())
        ms = sa[::-1].copy()
        ms[:, [0, 3]] *= -1
        assert np.array_equal(sb, ms)


def _soak(rng, n):
    """(qpos, v, action) along reset-stance episodes under random arm
    gears and a crude leg hold (rust/src/mirror_tests.rs soak_states)."""
    out = []
    far = [(200.0, 0.0, 1.4, 1, 1.5 * math.pi, 18.0, True, K.KIND_BOMB, False, False)]
    tgt = np.zeros(29)
    tgt[[19, 22, 25, 28]] = (-0.13962634015954636, 0.13962634015954636,
                             0.13962634015954636, -0.13962634015954636)
    while len(out) < n:
        P = rs.RawBeatSim(max_steps=100000, gears=5)
        P.set_chart(far, [])
        for _ in range(20):
            qp = np.asarray(P.qpos())
            v = np.asarray(list(P.twist()) + list(P.omega()))
            if qp[1] < 0.6 or len(out) >= n:
                break
            u = -(qp[7:] - tgt) * 3.0 - 0.3 * v[6:]
            a = np.zeros(29, np.uint8)
            a[17:] = np.where(u[17:] > 0.3, 3, np.where(u[17:] < -0.3, 4, 0))
            a[3:17] = np.where(rng.random(14) < 0.5, rng.integers(1, 5, 14), 0)
            out.append((qp, v, a))
            if P.step(list(a))[2]:
                break
    return out, far


def test_tick_commutes_with_the_mirror():
    rng = np.random.default_rng(2)
    states, far = _soak(rng, 120)
    gaps = []
    for qp, v, a in states:
        A = rs.RawBeatSim(max_steps=100000, gears=5)
        B = rs.RawBeatSim(max_steps=100000, gears=5)
        A.set_chart(far, [])
        B.set_chart(mirror_tuples(far), [])
        A.set_joints(list(qp), list(v))
        B.set_joints(list(m_qpos(qp)), list(m_qvel(v)))
        A.step(list(a))
        B.step(list(m_action(a)))
        va = np.asarray(list(A.twist()) + list(A.omega()))
        vb = np.asarray(list(B.twist()) + list(B.omega()))
        gaps.append(max(np.abs(vb - m_qvel(va)).max(),
                        np.abs(np.asarray(B.qpos()) - m_qpos(A.qpos())).max()))
    gaps = np.sort(gaps)
    # pre-fix (left-first solve orders) the median here was ~0.5
    assert np.median(gaps) < 1e-12, np.median(gaps)
    # the tail is the dynamics' own discontinuities (joints exactly on a
    # stop), where rounding picks the side — rare
    assert np.mean(gaps > 1e-6) < 0.15


def test_symmetric_reset_stays_symmetric():
    far = [(200.0, 0.0, 1.4, 1, 1.5 * math.pi, 18.0, True, K.KIND_BOMB, False, False)]
    P = rs.RawBeatSim(max_steps=100000, gears=5)
    P.set_chart(far, [])
    for _ in range(10):
        P.step([0] * 29)
        qp = np.asarray(P.qpos())
        assert np.abs(qp - m_qpos(qp)).max() < 1e-10
        v = np.asarray(list(P.twist()) + list(P.omega()))
        assert np.abs(v - m_qvel(v)).max() < 1e-10
