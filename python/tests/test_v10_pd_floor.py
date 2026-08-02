"""SPEC-v10 Divergences §22 — f64 Cholesky pivot-floor regression.

Found benching v10_gears5_400m ckpt_upd0762 (a mid-surge flaily
policy): a contorted gimbal-adjacent state drove an f64 mass-matrix
Cholesky pivot non-positive and the old strict factorization PANICKED
the whole VecBeatSim bench (Rust assert) / raised (torch f64
check_pd). Both paths now clamp pivots at PD_FLOOR_F64 = 1e-17 (the
CUDA f32 §20e net scaled by the machine-epsilon ratio) with
engagements counted — pop_pd_hits on the Rust sims, pd_net_hits on
the torch mirror. The floor is a simple max, so every healthy
factorization (smallest physical pivot: wrist armature 2.7e-5, 12
decades above the floor) is BITWISE untouched — the Rust <-> torch f64
parity bar (test_v10_parity_torch) is unaffected and remains the
binding gate.

The singular family (Divergences §12): the shoulder yaw->pitch->twist
stack gimbal-locks at pitch -90 deg — exactly l/r_arm_pitch's lower
limit — where the yaw and twist world axes are anti-parallel through
one point. The armature keeps the HUMANOID's pivots positive there, so
the deterministic driver below removes the armature from the diagonal
to reproduce the P1-measured exactly-singular matrix (pivot -4.3e-18).
"""

import numpy as np
import torch

import beat_arena_rs
from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import JOINT_LIMITS, N_JOINTS, NV
from beat_arena_env.torch_dynamics import ARMATURE, BASE_POS0, DynBatch

FAR_NOTE = Note(hit_time=200.0, col=2, row=1, x=0.3, y=1.4, color=1,
                angle=4.712, speed=4.0)


def _gimbal_q36():
    """Root at neutral, both shoulder stacks parked ON the gimbal lock."""
    q = [0.0] * N_JOINTS
    q[3] = JOINT_LIMITS[3][0]    # l_arm_pitch = -pi/2 exactly
    q[10] = JOINT_LIMITS[10][0]  # r_arm_pitch = -pi/2 exactly
    q[4] = 0.3                   # arbitrary yaw/twist — lock is pitch-only
    q[5] = -0.2
    return list(BASE_POS0) + [1.0, 0.0, 0.0, 0.0] + q


def test_torch_f64_floor_engages_counts_and_stays_finite():
    b = DynBatch(2, dtype=torch.float64)
    b.set_state(_gimbal_q36(), [0.0] * NV)
    b.fk_all()
    b.crba()
    # remove the armature -> the exactly-singular §12 matrix
    for j in range(N_JOINTS):
        b.M[:, 6 + j, 6 + j] -= ARMATURE[j]
    hits0 = b.pd_net_hits
    b.chol_factor()   # pre-§22: check_pd raised "not positive definite"
    b.check_pd()      # must NOT raise now
    assert b.pd_net_hits - hits0 == 2, "both envs must count an engagement"
    assert torch.isfinite(b.L).all(), "floored factor must stay finite"
    # and a healthy (armature'd) matrix never engages: bitwise-untouched path
    b.crba()
    L_before = None
    b.chol_factor()
    L_before = b.L.clone()
    b.check_pd()
    assert b.pd_net_hits - hits0 == 2, "healthy matrix must not count"
    b.chol_factor()
    assert torch.equal(b.L, L_before), "healthy factorization is deterministic"


def test_torch_floor_constant_matches_contract():
    assert DynBatch.PD_FLOOR_F64 == 1e-17
    assert DynBatch.PD_FLOOR_F32 == 1e-8


def test_rust_sim_survives_contorted_state_and_exposes_counter():
    """End-to-end through the former panic site: park the body at the
    gimbal lock with cap-riding joint rates and step hard random gears.
    Pre-§22 this state family could panic the process ('mass matrix not
    positive definite'); now the sim must keep stepping, stay finite,
    and expose the (possibly zero — the armature protects the in-limit
    family) engagement counter."""
    sim = beat_arena_rs.RawBeatSim(max_steps=690, gears=5)
    sim.set_chart([note_tuple(FAR_NOTE)])
    v = [0.0] * NV
    for j in range(N_JOINTS):
        v[6 + j] = 12.0 if j % 2 == 0 else -12.0
    sim.set_joints(_gimbal_q36(), v)
    rng = np.random.default_rng(0x5EED)
    for _ in range(30):
        acts = [int(a) for a in rng.integers(0, 7, N_JOINTS)]
        obs, reward, done, _ev = sim.step(acts)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        if done:
            break
    hits = sim.pop_pd_hits()
    assert isinstance(hits, int) and hits >= 0
    assert sim.pop_pd_hits() == 0, "pop must zero the counter"


def test_vec_sim_pop_pd_hits_zero_in_ordinary_motion():
    n = 4
    vec = beat_arena_rs.VecBeatSim(n, combo_walk=False, return_events=False)
    for i in range(n):
        vec.set_chart(i, [note_tuple(FAR_NOTE)])
    rng = np.random.default_rng(7)
    obs = np.zeros((n, 340), dtype=np.float32)
    for _ in range(15):
        acts = rng.integers(0, 3, (n, N_JOINTS)).astype(np.int64)
        vec.step_into(acts, obs)
    assert np.isfinite(obs).all()
    assert vec.pop_pd_hits() == 0, "neutral-start gentle gears never engage"
