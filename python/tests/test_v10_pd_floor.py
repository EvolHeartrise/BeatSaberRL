"""SPEC-v10 Divergences §22 — f64 Cholesky pivot-floor regression.

Found benching v10_gears5_400m ckpt_upd0762 (a mid-surge flaily
policy): a contorted gimbal-adjacent state drove an f64 mass-matrix
Cholesky pivot non-positive and the old strict factorization PANICKED
the whole VecBeatSim bench (Rust assert). Every tier now clamps pivots
at PD_FLOOR_F64 = 1e-17 (the CUDA f32 §20e net scaled by the
machine-epsilon ratio) with engagements counted — pop_pd_hits on the
Rust sims and on CudaBeatVecEnv. The floor is a simple max, so every
healthy factorization (smallest physical pivot: wrist armature 2.7e-5,
12 decades above the floor) is BITWISE untouched.

The singular family (Divergences §12): the shoulder yaw->pitch->twist
stack gimbal-locks at pitch -90 deg — exactly l/r_arm_pitch's lower
limit — where the yaw and twist world axes are anti-parallel through
one point. The armature keeps the HUMANOID's pivots positive there, so
FORCING an engagement needs an armature-less model — which only the
Rust tier can build, so that regression lives in
`dynamics::tests_dyn::{pivot_floor_clamps_singular_gimbal_matrix,
substep_survives_pd_loss_and_counts}` (cargo). This module covers the
FFI end-to-end path, the popped counters, and the cross-tier floor
CONSTANTS. (Until 2026-08-03 the torch mirror's DynBatch drove the
forced-engagement case here; it went with the mirror.)
"""

import pathlib
import re

import numpy as np

import beat_arena_rs
from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import BASE_POS0, JOINT_LIMITS, N_JOINTS, NV

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


def test_floor_constants_match_across_tiers():
    """The §22/§20e contract is a pair of NUMBERS shared by two
    implementations that cannot import each other. Rust owns
    PD_FLOOR_F64 (asserted bitwise in tests_dyn); this pins the CUDA
    kernel's two constexprs to the same contract, which is the check
    the retired torch mirror used to carry."""
    cu = (pathlib.Path(__file__).resolve().parents[1]
          / "beat_arena_env" / "csrc" / "beat_arena_cuda.cu").read_text()
    f32 = re.search(r"constexpr\s+float\s+PD_FLOOR_F32\s*=\s*([0-9.eE+-]+)f?;",
                    cu)
    f64 = re.search(r"constexpr\s+double\s+PD_FLOOR_F64\s*=\s*([0-9.eE+-]+);",
                    cu)
    assert f32 and f64, "the .cu pivot-floor constants moved or were renamed"
    assert float(f32.group(1)) == 1e-8, "§20e f32 net changed"
    assert float(f64.group(1)) == 1e-17, "§22 f64 floor changed"


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
    obs = np.zeros((n, 362), dtype=np.float32)
    for _ in range(15):
        acts = rng.integers(0, 3, (n, N_JOINTS)).astype(np.int64)
        vec.step_into(acts, obs)
    assert np.isfinite(obs).all()
    assert vec.pop_pd_hits() == 0, "neutral-start gentle gears never engage"
