"""Passive wrist tendon torque (exp/wrist-tendon, 2026-10-05): the
constants.py reference (pure axis, mirror, -grad E), the Rust sims'
[tendon] telemetry (both f64 and f32 builds) and its exclusion from the
energy base, the replay header's body marker, and the CUDA twin's
telemetry against the f32 Rust build (GPU only).

The physics gates (gradient, conservative loops, continuity, stiffness
/ omega*h, corner soaks, release energy) are Rust unit tests:
rust/src/tendon_tests.rs."""
import math

import numpy as np
import pytest

from beat_arena_env import constants as C
from beat_arena_env.chart import note_tuple
from beat_arena_env.replay_format import body_tables
from tests._helpers import FAR_NOTE, ZERO_ACT, cuda, rust_state

LIM = C.JOINT_LIMITS


def test_peaks_and_pure_axis():
    assert C.TENDON_TAU_PEAK_BEND == pytest.approx(0.62103, abs=5e-6)
    assert C.TENDON_TAU_PEAK_DEV == pytest.approx(0.32500, abs=5e-6)
    for (jb, jd) in C.TENDON_HANDS:
        for u in np.linspace(-1.4, 1.4, 57):
            lo, hi = LIM[jb]
            tb, td, r = C.wrist_tendon(u * (hi if u >= 0 else -lo), 0.0,
                                       LIM[jb], LIM[jd])
            want = (0.0 if abs(u) <= 0.5 else
                    -math.copysign(C.TENDON_TAU_PEAK_BEND
                                   * ((abs(u) - 0.5) / 0.5) ** 2, u))
            assert td == 0.0 and tb == pytest.approx(want, abs=1e-14)
            lo, hi = LIM[jd]
            tb, td, r = C.wrist_tendon(0.0, u * (hi if u >= 0 else -lo),
                                       LIM[jb], LIM[jd])
            want = (0.0 if abs(u) <= 0.5 else
                    -math.copysign(C.TENDON_TAU_PEAK_DEV
                                   * ((abs(u) - 0.5) / 0.5) ** 2, u))
            assert tb == 0.0 and td == pytest.approx(want, abs=1e-14)


def test_mirror_and_corners():
    (lb, ld), (rb, rd) = C.TENDON_HANDS
    rng = np.random.default_rng(0)
    for qb, qd in rng.uniform([-2, -0.8], [2, 0.8], (2000, 2)):
        a = C.wrist_tendon(qb, qd, LIM[lb], LIM[ld])
        b = C.wrist_tendon(qb, -qd, LIM[rb], LIM[rd])
        assert a[0] == b[0] and a[1] == -b[1] and a[2] == b[2]
    # corners: well above the pure-axis peaks (P uncapped past r = 1)
    tb, td, r = C.wrist_tendon(LIM[lb][1], LIM[ld][1], LIM[lb], LIM[ld])
    assert r == pytest.approx(math.sqrt(2))
    assert -tb > C.TENDON_TAU_PEAK_BEND and -td > C.TENDON_TAU_PEAK_DEV
    assert (tb, td) == pytest.approx((-1.0849, -3.1951), abs=1e-4)


def test_header_marks_the_tendon_body():
    b = body_tables(5, list(C.TAU_BODY_DEFAULT))
    t = b["tendon"]
    assert t["hands"] == [[8, 9], [15, 16]]
    assert t["tau_peak_bend"] == C.TENDON_TAU_PEAK_BEND
    assert t["tau_peak_dev"] == C.TENDON_TAU_PEAK_DEV
    assert t["b_peak_bend"] == C.TENDON_B_PEAK_BEND
    assert t["b_peak_dev"] == C.TENDON_B_PEAK_DEV
    assert t["r0"] == 0.5


def test_edge_damping_constants_and_shape():
    """Issue #61: b_peak = tau_peak / 30 rad/s (bend 0.020701, dev
    0.010833 N.m.s/rad); x^2 shape zero inside r <= 0.5, 1 at r = 1,
    uncapped past it."""
    assert C.TENDON_B_PEAK_BEND == pytest.approx(C.TENDON_TAU_PEAK_BEND / 30)
    assert C.TENDON_B_PEAK_DEV == pytest.approx(C.TENDON_TAU_PEAK_DEV / 30)
    assert C.TENDON_B_PEAK_BEND == pytest.approx(0.020701, abs=1e-6)
    assert C.TENDON_B_PEAK_DEV == pytest.approx(0.010833, abs=1e-6)
    assert C.tendon_edge_x2(0.3) == 0.0 and C.tendon_edge_x2(0.5) == 0.0
    assert C.tendon_edge_x2(1.0) == 1.0
    assert C.tendon_edge_x2(math.sqrt(2)) == pytest.approx(3.343, abs=1e-3)


@pytest.mark.parametrize("mod_name", ["beat_arena_rs", "beat_arena_rs32"])
def test_rust_pop_tendon_and_energy_exclusion(mod_name):
    mod = pytest.importorskip(mod_name)
    if not hasattr(mod.RawBeatSim, "pop_tendon"):
        pytest.skip(f"{mod_name} build predates the wrist tendon — rebuild")
    sim = mod.RawBeatSim(max_steps=2000, gears=5)
    sim.set_chart([note_tuple(FAR_NOTE)])
    sim.pop_tendon()
    sim.pop_torque()
    # the reset arm pose bends both wrists 70 deg: u = 0.737, in the zone,
    # inside the ellipse, mirror-symmetric
    sim.step(ZERO_ACT)
    tl, tr, zl, zr, ol, orr, dl, dr, n = sim.pop_tendon()
    assert n == 8
    assert tl > 0 and tl == pytest.approx(tr, rel=1e-5)
    assert (zl, zr, ol, orr) == (8, 8, 0, 0)
    # the wrists start at rest and the tendon sets them moving: the edge
    # damping dissipates a little, symmetrically
    assert dl > 0 and dl == pytest.approx(dr, rel=1e-4)
    assert sim.pop_torque()[0] == 0.0          # coasting: no muscle torque
    # left wrist parked in the (+bend, +15 deg dev) corner: beyond r = 1
    q, v = rust_state(sim)
    q, v = list(q), [0.0] * len(v)
    q[7 + 8], q[7 + 9] = 0.99 * LIM[8][1], 0.99 * LIM[9][1]
    sim.set_joints(q, v)
    sim.step(ZERO_ACT)
    tl, tr, zl, zr, ol, orr, dl, dr, n = sim.pop_tendon()
    assert n == 8 and zl == 8 and ol >= 1
    assert tl / n > 1.0                         # ~4 N*m at the corner
    assert dl > 0
    assert sim.pop_torque()[0] == 0.0
    assert sim.pop_tendon() == (0.0,) * 8 + (0,)
    # the test switch turns the tendon AND its edge damping off
    sim.set_passive_wrist(False)
    sim.set_joints(q, v)
    sim.step(ZERO_ACT)
    assert sim.pop_tendon()[:8] == (0.0,) * 8
    sim.set_passive_wrist(True)


@pytest.mark.parametrize("mod_name", ["beat_arena_rs", "beat_arena_rs32"])
def test_vec_pop_tendon_sums_envs(mod_name):
    mod = pytest.importorskip(mod_name)
    if not hasattr(mod.VecBeatSim, "pop_tendon"):
        pytest.skip(f"{mod_name} build predates the wrist tendon — rebuild")
    n = 3
    vec = mod.VecBeatSim(n, max_steps=2000, gears=5, return_events=False)
    for i in range(n):
        vec.set_chart(i, [note_tuple(FAR_NOTE)], [])
    vec.pop_tendon()
    obs = np.zeros((n, vec.obs().shape[0] // n), np.float32)
    acts = np.zeros((n, C.N_JOINTS), np.int64)
    for _ in range(4):
        vec.step_into(acts, obs)
    s = vec.pop_tendon()
    assert len(s) == 9 and s[8] == n * 4 * 8
    assert s[2] == s[3] == n * 4 * 8            # reset pose: all in the zone


@cuda
@pytest.mark.parametrize("drive", [False, True])
def test_cuda_pop_tendon_matches_rust32(drive):
    """The kernel's [tendon] accumulators (tendon torque, zone / beyond
    counts, edge-damping dissipated power) against the f32 Rust build
    from the same episode start (no chart randomness touches the body
    for a few ticks): coasting (the 70 deg reset bend springs back), and
    driving both wrists at full gear into the (+bend, outer dev) corner
    (exercises the implicit edge damping beyond r = 1). Parity is
    tolerance-grade (nvcc may contract FMAs)."""
    import torch
    import beat_arena_rs32
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    n, ticks = 4, 4
    act = np.zeros(C.N_JOINTS, np.int64)
    if drive:      # gear 1 = +full, 2 = -full; dev flips sign on the right
        act[8], act[9], act[15], act[16] = 1, 1, 1, 2
    env = CudaBeatVecEnv(n, chart_len_s=6.0, seed=5, device="cuda",
                         dtype=torch.float32, gears=5, auto_reset=False,
                         start_pose_jitter=0.0)
    env.reset()
    env.pop_tendon()
    for _ in range(ticks):
        env.step(torch.as_tensor(np.tile(act, (n, 1)), device="cuda"))
    c = env.pop_tendon()
    assert c[8] == n * ticks * 8
    sim = beat_arena_rs32.RawBeatSim(max_steps=2000, gears=5)
    sim.set_chart([note_tuple(FAR_NOTE)])
    sim.pop_tendon()
    for _ in range(ticks):
        sim.step([int(a) for a in act])
    r = sim.pop_tendon()
    assert r[6] > 0 and r[7] > 0                # the edge damping worked
    for k in range(8):
        assert c[k] / n == pytest.approx(r[k], rel=5e-3, abs=1e-5), (k, c, r)
    assert env.pop_tendon()[8] == 0
