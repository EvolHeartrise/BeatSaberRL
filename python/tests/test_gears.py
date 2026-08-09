"""Env v8 gear tables on the env v10 dynamic body (P3 re-author).

Covers: the gear tables themselves, torque scaling (fine gears =
fractional TAU_MAX — SPEC-v10 §4), ternary-prefix compatibility (ids
0/1/2 behave identically at every K, bitwise), gear-scaled work-based
effort ordering, the wrist/arm omega caps, the v9 deviation-joint
geometry (kinematics mirror) + its asymmetric limits on the v10 body.

P5 (2026-07-31): test_gears_cuda_parity restored on the v10 CUDA
backend (skipped without CUDA, like the dedicated CUDA suites).
"""

import numpy as np
import pytest

from beat_arena_env import RustBeatEnv
from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import (
    GEAR_MAGS, N_JOINTS, OFF_QD, TAU_MAX, gear_table, q_lane,
)
from beat_arena_rs import RawBeatSim

COAST = [0] * N_JOINTS
R_BEND = 15        # r wrist bend (snappy tier)
R_ELBOW = 13
R_DEV = 16
L_DEV = 9


def _sim(gears, effort_coef=0.0):
    """A sim whose only note is far away — pure body dynamics."""
    sim = RawBeatSim(max_steps=690, effort_coef=effort_coef, gears=gears)
    n = Note(hit_time=15.0, col=0, row=0, color=1, angle=0.0, speed=4.0)
    sim.set_chart([note_tuple(n)])
    return sim


def _qd_obs(obs, j):
    """Normalized joint velocity (qd / OMEGA_CAP) of joint j."""
    return float(np.asarray(obs)[OFF_QD + j])


def test_gear_tables():
    assert gear_table(3) == (0.0, 1.0, -1.0)
    assert gear_table(5) == (0.0, 1.0, -1.0, 0.25, -0.25)
    t7 = gear_table(7)
    assert t7[0:3] == (0.0, 1.0, -1.0)
    assert t7[3] == pytest.approx(1.0 / 3.0)
    assert t7[4] == pytest.approx(-1.0 / 3.0)
    assert t7[5:7] == (0.125, -0.125)
    # ids 1/2 are +/-full at EVERY K (ternary-era streams stay valid)
    for k in GEAR_MAGS:
        assert gear_table(k)[1] == 1.0 and gear_table(k)[2] == -1.0


def test_gear_torque_scaling():
    """One tick of each gear from rest on the right wrist bend: the
    velocity CHANGE relative to the coasting baseline orders by |gear
    fraction| and flips sign with it (v10: tau = fraction x TAU_MAX
    through the full dynamics, so exact ratios are pose-dependent —
    the ordering and sign are the contract)."""
    def om_after(gears, aid):
        sim = _sim(gears)
        act = list(COAST)
        act[R_BEND] = aid
        obs, _, _, _ = sim.step(act)
        return _qd_obs(obs, R_BEND)

    base = om_after(7, 0)
    full, third, eighth = om_after(7, 1), om_after(7, 3), om_after(7, 5)
    neg_full = om_after(7, 2)
    quarter = om_after(5, 3)
    assert full - base > third - base > eighth - base > 0.0
    assert third - base > quarter - base > eighth - base
    assert neg_full - base < 0.0
    # +full and -full are roughly symmetric about the coast baseline
    # (gravity assists the - direction on the bend, so the band is wide)
    assert abs((full - base) + (neg_full - base)) < 0.4 * (full - base)


def test_omega_caps_observable():
    """Wrist tier caps at 40 rad/s, arm tier at 16 — full-torque drives
    saturate the normalized qd obs at exactly 1.0 (the omega-cap rows
    hold the rate AT the cap). Probes need range to cruise in before
    the position stop: L wrist roll (180 deg) and the elbow (150)."""
    L_ROLL = 7
    assert TAU_MAX[L_ROLL] == 1.06 and TAU_MAX[R_ELBOW] == 19.13
    for j in (L_ROLL, R_ELBOW):
        sim = _sim(3)
        act = list(COAST)
        act[j] = 1
        peak = 0.0
        for _ in range(10):
            obs, _, _, _ = sim.step(act)
            peak = max(peak, _qd_obs(obs, j))
        assert peak == pytest.approx(1.0, abs=1e-6), f"joint {j}"


def test_ternary_prefix_identical_across_k():
    """A 0-2 action stream produces bit-identical trajectories at K=3,
    K=5, and K=7 (same full-torque semantics, same effort)."""
    rng = np.random.default_rng(7)
    acts = rng.integers(0, 3, size=(90, N_JOINTS))
    streams = {}
    for k in (3, 5, 7):
        sim = _sim(k, effort_coef=4e-4)
        obs_l, rew_l = [], []
        for a in acts:
            obs, r, _, _ = sim.step([int(x) for x in a])
            obs_l.append(np.asarray(obs).copy())
            rew_l.append(r)
        streams[k] = (obs_l, rew_l)
    for k in (5, 7):
        np.testing.assert_array_equal(streams[3][0], streams[k][0])
        assert streams[3][1] == streams[k][1]


def test_effort_scales_with_gear():
    """Env v10 effort = coef x positive mechanical work: coast is free;
    a quarter-gear tick does less work than a full-gear tick (quarter
    torque AND slower resulting motion); the reward identity
    r = -coef x last_effort holds exactly."""
    def one_tick(gears, aid, coef=1.0):
        sim = _sim(gears, effort_coef=coef)
        act = list(COAST)
        act[R_BEND] = aid
        _, r, _, _ = sim.step(act)
        return r, sim.last_effort()

    r0, j0 = one_tick(5, 0)
    assert r0 == 0.0 and j0 == 0.0
    r_full, j_full = one_tick(5, 1)
    r_quarter, j_quarter = one_tick(5, 3)
    assert 0.0 < j_quarter < j_full
    assert r_full == pytest.approx(-j_full, abs=1e-15)
    assert r_quarter == pytest.approx(-j_quarter, abs=1e-15)
    _, j_eighth = one_tick(7, 5)
    assert 0.0 < j_eighth < j_quarter


def test_env_action_space_and_header():
    env = RustBeatEnv(gears=5, seed=0)
    assert list(env.action_space.nvec) == [5] * N_JOINTS
    with pytest.raises(AssertionError):
        RustBeatEnv(gears=4)


def test_policy_head_width_roundtrip():
    """v10 policy (29 heads, obs 340, body token) drives the env at
    every K; checkpoint arch round-trips; env/policy K mismatch and v9
    checkpoints are refused."""
    import torch
    from policy import NotePolicy, load_policy

    env = RustBeatEnv(gears=5, seed=0, chart_len_s=6.0)
    obs, _ = env.reset()
    for k in (3, 5, 7):
        torch.manual_seed(0)
        policy = NotePolicy(n_gears=k)
        assert policy.head_sizes == (k,) * N_JOINTS
        with torch.no_grad():
            logits, value = policy.forward_stacked(
                torch.as_tensor(obs).unsqueeze(0))
        assert logits.shape == (1, N_JOINTS, k) and value.shape == (1,)
        ck = {"state_dict": policy.state_dict(), "arch": policy.arch}
        p2 = load_policy(ck)
        assert p2.n_gears == k and p2.arch == policy.arch
    # a K=5 policy's greedy action stream is valid input for the env
    torch.manual_seed(0)
    policy = NotePolicy(n_gears=5)
    with torch.no_grad():
        a, _, _, _ = policy.get_action_and_value(
            torch.as_tensor(obs).unsqueeze(0))
    obs2, _r, _d, _t, _info = env.step([int(x) for x in a[0]])
    assert obs2.shape == (362,)
    # v9-shaped checkpoint (no v10 arch markers) is refused
    with pytest.raises(SystemExit):
        load_policy({"state_dict": {}, "arch": {"embed": 64, "n_gears": 5}})


def test_gears_cuda_parity():
    """P5: the CUDA env honors the gear tables like Rust. (a) A 0-2
    action stream is BITWISE identical across K=3/5/7 on the CUDA
    backend (ids 1/2 are +/-full at every K); (b) short-horizon f64
    CUDA-vs-Rust agreement holds at the transcendental-seed band for
    every table (full free-running parity semantics live in
    test_v10_parity_cuda.py)."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA env needs CUDA")
    from beat_arena_env.cuda_env import CudaBeatVecEnv

    n_far = Note(hit_time=200.0, col=2, row=1, color=1, angle=0.0, speed=4.0)
    rng = np.random.default_rng(7)
    acts = rng.integers(0, 3, size=(2, 1, N_JOINTS))

    streams = {}
    for k in (3, 5, 7):
        env = CudaBeatVecEnv(1, charts=[[n_far]], max_steps=690,
                             device="cuda", dtype=torch.float64,
                             auto_reset=False, gears=k,
                             want_contacts=True)
        env.reset()
        obs_l = []
        for t in range(2):
            obs, r, _d, _ = env.step(torch.as_tensor(acts[t]))
            obs_l.append(obs[0].cpu().numpy().copy())
        streams[k] = obs_l
    for k in (5, 7):
        np.testing.assert_array_equal(streams[3], streams[k])

    # (b) vs Rust, per table, 2 gentle ticks from neutral (same chart!)
    for k in (3, 5, 7):
        env = CudaBeatVecEnv(1, charts=[[n_far]], max_steps=690,
                             device="cuda", dtype=torch.float64,
                             auto_reset=False, gears=k,
                             want_contacts=True)
        env.reset()
        sim = RawBeatSim(max_steps=690, effort_coef=4e-4, gears=k)
        sim.set_chart([note_tuple(n_far)])
        a = list(COAST)
        a[R_BEND] = 1
        for t in range(2):
            obs_c, _r, _d, _ = env.step(torch.as_tensor([a]))
            obs_r, _rr, _dr, _ = sim.step(a)
            np.testing.assert_allclose(
                np.asarray(obs_r), obs_c[0].cpu().numpy(), atol=1e-6,
                err_msg=f"gears={k} t{t}")


def test_deviation_joint():
    """Env v9 wrist deviation carried into v10: dev is a real orthogonal
    bending axis (kinematics mirror), and the sim's ASYMMETRIC limits
    clamp (right [-15, +30] deg; left mirrored; obs normalizer 30 deg =
    max(|lo|, |hi|))."""
    from beat_arena_env.kinematics import fk

    # dev = 0 reproduces the 6-joint blade
    q6 = [0.2, -0.1, 0.3, 0.9, -1.0, 0.7]
    _, w6, b6 = fk(1, q6)
    _, w7, b7 = fk(1, q6 + [0.0])
    assert np.allclose(w6, w7) and np.allclose(b6, b7)

    # straight wrist: deviation tilts the blade off the forearm axis at
    # full rate (the old roll+bend wrist was pole-singular there)
    _, _, b0 = fk(1, [0.0] * 7)
    _, _, bd = fk(1, [0.0] * 6 + [0.3])
    assert np.allclose(b0, (0.0, 0.0, 1.0), atol=1e-12)
    assert abs(np.dot(b0, bd) - np.cos(0.3)) < 1e-9  # full-rate tilt

    # sim: sustained dev torque parks the joint at its stop; the q-block
    # lane reads q / 30 deg. (8 ticks: the stop is reached in ~2 while
    # the unactuated body is still near the reference pose — by ~1 s the
    # coasting ragdoll's flail forces exceed the 1.06 N·m wrist motor.)
    def run_to_clamp(j, aid):
        sim = _sim(3)
        act = list(COAST)
        act[j] = aid
        for _ in range(8):
            obs, _, _, _ = sim.step(act)
        return float(np.asarray(obs)[q_lane(j)])

    assert run_to_clamp(R_DEV, 1) == pytest.approx(1.0, abs=1e-3)   # R +30
    assert run_to_clamp(R_DEV, 2) == pytest.approx(-0.5, abs=1e-3)  # R -15
    assert run_to_clamp(L_DEV, 1) == pytest.approx(0.5, abs=1e-3)   # L +15
    assert run_to_clamp(L_DEV, 2) == pytest.approx(-1.0, abs=1e-3)  # L -30
