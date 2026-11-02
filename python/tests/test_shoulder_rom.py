"""Passive shoulder axial-rotation limit (exp/shoulder-rom, 2026-10-07):
the constants.py reference (anchors, fit, pure axis, mirror, -grad E), the
widened twist box, the Rust sims' [shoulder] telemetry (f64 and f32 builds)
and its exclusion from the energy base, the replay header / body marker,
and the CUDA twin's telemetry against the f32 Rust build (GPU only).

The physics gates (gradient, closed loops, the hanging pole, stiffness /
omega*h with the model's M, soaks, release energy, implicit damping) are
Rust unit tests: rust/src/shoulder_tests.rs."""
import math

import numpy as np
import pytest

from beat_arena_env import constants as C
from beat_arena_env.chart import note_tuple
from beat_arena_env.replay_format import body_tables
from tests._helpers import FAR_NOTE, ZERO_ACT, cuda, rust_state

D = math.radians


def _twist_at(y, p, u):
    lo, hi = C.shoulder_limits(y, p)
    return y - (0.5 * (hi + lo) + u * 0.5 * (hi - lo))


def test_constants_box_and_fit():
    assert C.SROM_TAU_PEAK == pytest.approx(5.0933715, abs=1e-9)
    assert C.SROM_B_PEAK == pytest.approx(C.SROM_TAU_PEAK / 16.0)
    lo, hi = C.srom_fit_coeffs()
    assert lo == pytest.approx(C.SROM_COEF_LO, abs=1e-14)
    assert hi == pytest.approx(C.SROM_COEF_HI, abs=1e-14)
    # the widened twist box (internal side) and its mirror
    assert C.JOINT_LIMITS[5] == (D(-190), D(70))
    assert C.JOINT_LIMITS[12] == (D(-70), D(190))
    assert C.q_normalizer(5) == C.q_normalizer(12) == D(190)
    for th, ph, er, ir in C.SROM_ANCHORS:
        flo, fhi = C.shoulder_limits(D(ph - 90), D(th - 90))
        assert math.degrees(flo) == pytest.approx(er, abs=1e-9)
        assert math.degrees(fhi) == pytest.approx(ir, abs=1e-9)
    # every human IR limit over the yaw/pitch box fits in the twist box
    worst = min(y - C.shoulder_limits(y, p)[1]
                for y in np.radians(np.arange(-100, 46, 1.0))
                for p in np.radians(np.arange(-90, 66, 1.0)))
    assert C.JOINT_LIMITS[5][0] < worst < D(-180)


def test_pure_axis_and_knees():
    T = C.SROM_TAU_PEAK
    for yd, pd in ((0, -80), (-90, 0), (0, 0), (-50, 60)):
        y, p = D(yd), D(pd)
        for u in np.linspace(-2, 2, 81):
            ty, tp, tt, uu, b, e = C.shoulder_rom(y, p, _twist_at(y, p, u))
            g, _ = C.srom_shape(abs(u))
            assert uu == pytest.approx(u, abs=1e-12)
            assert tt == pytest.approx(math.copysign(T * g, u) if g else 0.0,
                                       abs=1e-9)
    for u, want in ((0.5, 0.4158), (1.0, 5.0934), (1.25, 14.4747), (1.5, 29.5973)):
        assert T * C.srom_shape(u)[0] == pytest.approx(want, rel=2e-3)
    # the reset arm pose is neutral (zero torque); the zero pose is not
    assert C.shoulder_rom(0.0, D(-80), 0.0)[:3] == (0.0, 0.0, 0.0)
    assert C.shoulder_rom(0.0, 0.0, 0.0)[3] == pytest.approx(-0.677, abs=1e-3)


def test_gradient_and_mirror_reference():
    rng = np.random.default_rng(0)
    h = 1e-6
    for y, p, t in rng.uniform([-1.85, -1.62, -3.4], [0.88, 1.2, 1.3], (2000, 3)):
        ty, tp, tt, u, b, e = C.shoulder_rom(y, p, t)
        fd = [-(C.shoulder_rom(y + h, p, t)[5] - C.shoulder_rom(y - h, p, t)[5]) / (2 * h),
              -(C.shoulder_rom(y, p + h, t)[5] - C.shoulder_rom(y, p - h, t)[5]) / (2 * h),
              -(C.shoulder_rom(y, p, t + h)[5] - C.shoulder_rom(y, p, t - h)[5]) / (2 * h)]
        for an, nu in zip((ty, tp, tt), fd):
            assert abs(an - nu) <= 1e-5 * (1e-3 + abs(an)) + 1e-7
        assert b >= 0.0
    # the hanging pole: only yaw - twist matters
    for tau0 in (-1.0, 0.3, 1.2):
        ref = C.shoulder_rom(0.0, -math.pi / 2, -tau0)
        for y in (-1.7, -0.5, 0.7):
            o = C.shoulder_rom(y, -math.pi / 2, y - tau0)
            assert o[5] == pytest.approx(ref[5], rel=1e-12, abs=1e-15)
            assert o[2] == pytest.approx(ref[2], rel=1e-10, abs=1e-12)
            assert o[0] == pytest.approx(-o[2], rel=1e-10, abs=1e-12)


def test_header_marks_the_shoulder_body():
    b = body_tables(5, list(C.TAU_BODY_DEFAULT))
    s = b["shoulder_rom"]
    assert s["arms"] == [[4, 3, 5], [11, 10, 12]]
    assert s["tau_peak"] == C.SROM_TAU_PEAK and s["b_peak"] == C.SROM_B_PEAK
    assert (s["zone_start"], s["beyond_k"], s["beyond_w"], s["u_lin"]) == (0.3, 1.0, 0.25, 1.25)
    assert s["coef_lo"] == list(C.SROM_COEF_LO)
    assert b["joint_limits"][5] == [D(-190), D(70)]


def _park_past_er(sim, u=-1.15):
    """Both arms forward-raised (pitch -30, yaw -20 deg, mirrored) with the
    twist past the human ER limit, at rest."""
    q, v = rust_state(sim)
    q, v = list(q), [0.0] * len(v)
    y, p = D(-20), D(-30)
    t = _twist_at(y, p, u)
    q[7 + 4], q[7 + 3], q[7 + 5] = y, p, t
    q[7 + 11], q[7 + 10], q[7 + 12] = -y, p, -t
    sim.set_joints(q, v)


@pytest.mark.parametrize("mod_name", ["beat_arena_rs", "beat_arena_rs32"])
def test_rust_pop_shoulder_and_energy_exclusion(mod_name):
    mod = pytest.importorskip(mod_name)
    if not hasattr(mod.RawBeatSim, "pop_shoulder"):
        pytest.skip(f"{mod_name} build predates the passive shoulder — rebuild")
    sim = mod.RawBeatSim(max_steps=2000, gears=5)
    sim.set_chart([note_tuple(FAR_NOTE)])
    sim.pop_shoulder()
    sim.pop_torque()
    # the reset arm pose is neutral: nothing applied, nothing counted
    sim.step(ZERO_ACT)
    sl, sr, zl, zr, ol, orr, dl, dr, n = sim.pop_shoulder()
    assert n == 8 and (sl, sr, zl, zr, ol, orr) == (0.0,) * 6
    _park_past_er(sim)
    sim.step(ZERO_ACT)
    sl, sr, zl, zr, ol, orr, dl, dr, n = sim.pop_shoulder()
    assert n == 8 and zl == zr == 8 and ol >= 1 and orr >= 1
    assert sl / n > 3.0 and sl == pytest.approx(sr, rel=1e-4)
    assert dl > 0 and dl == pytest.approx(dr, rel=1e-3)
    assert sim.pop_torque()[0] == 0.0          # coasting: no muscle torque
    assert sim.pop_shoulder() == (0.0,) * 8 + (0,)
    sim.set_passive_shoulder(False)
    _park_past_er(sim)
    sim.step(ZERO_ACT)
    assert sim.pop_shoulder()[:8] == (0.0,) * 8
    sim.set_passive_shoulder(True)


@pytest.mark.parametrize("mod_name", ["beat_arena_rs", "beat_arena_rs32"])
def test_vec_pop_shoulder_sums_envs(mod_name):
    mod = pytest.importorskip(mod_name)
    if not hasattr(mod.VecBeatSim, "pop_shoulder"):
        pytest.skip(f"{mod_name} build predates the passive shoulder — rebuild")
    n = 3
    vec = mod.VecBeatSim(n, max_steps=2000, gears=5, return_events=False)
    for i in range(n):
        vec.set_chart(i, [note_tuple(FAR_NOTE)], [])
    vec.pop_shoulder()
    obs = np.zeros((n, vec.obs().shape[0] // n), np.float32)
    acts = np.zeros((n, C.N_JOINTS), np.int64)
    acts[:, 5], acts[:, 12] = 1, 2        # both twists driven into ER
    for _ in range(6):
        vec.step_into(acts, obs)
    s = vec.pop_shoulder()
    assert len(s) == 9 and s[8] == n * 6 * 8
    assert s[0] > 0 and s[0] == pytest.approx(s[1], rel=1e-4)   # f32 mirror rounding
    assert 0 < s[2] == s[3] <= n * 6 * 8


SCENARIOS = {
    # both twists at full gear into external rotation, then coasting back
    "er_then_coast": lambda k: {5: 1, 12: 2} if k < 4 else {},
    # internal rotation with the arms raised (pitch up) — the moving limit
    "ir_raise": lambda k: {5: 2, 12: 1, 3: 1, 10: 1},
}


@cuda
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_cuda_pop_shoulder_matches_rust32(scenario):
    """The kernel's [shoulder] accumulators (torque, zone / beyond counts,
    edge-damping power) against the f32 Rust build from the same episode
    start: drive into the limits and coast. Tolerance-grade (nvcc FMAs)."""
    import torch
    import beat_arena_rs32
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    n, ticks = 4, 8
    acts = []
    for k in range(ticks):
        a = np.zeros(C.N_JOINTS, np.int64)
        for j, g in SCENARIOS[scenario](k).items():
            a[j] = g
        acts.append(a)
    env = CudaBeatVecEnv(n, chart_len_s=6.0, seed=5, device="cuda",
                         dtype=torch.float32, gears=5, auto_reset=False,
                         start_pose_jitter=0.0)
    env.reset()
    env.pop_shoulder()
    for a in acts:
        env.step(torch.as_tensor(np.tile(a, (n, 1)), device="cuda"))
    c = env.pop_shoulder()
    assert c[8] == n * ticks * 8
    sim = beat_arena_rs32.RawBeatSim(max_steps=2000, gears=5)
    sim.set_chart([note_tuple(FAR_NOTE)])
    sim.pop_shoulder()
    for a in acts:
        sim.step([int(x) for x in a])
    r = sim.pop_shoulder()
    assert r[0] > 0 and r[2] > 0
    for k in range(8):
        assert c[k] / n == pytest.approx(r[k], rel=5e-3, abs=1e-4), (k, c, r)
    assert env.pop_shoulder()[8] == 0
