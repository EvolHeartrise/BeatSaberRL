"""Env v8 (2026-07-30): gear-table action space + doubled body limits.

Covers: the gear tables themselves, integrator torque scaling (fine
gears = fractional alpha), ternary-prefix compatibility (ids 0/1/2
behave identically at every K), gear-scaled effort pricing, policy
head-width round-trip, and Rust<->CUDA parity at K=5.
"""

import numpy as np
import pytest
import torch

from beat_arena_env import RustBeatEnv
from beat_arena_env.chart import ChartConfig, Note, generate_chart, note_tuple
from beat_arena_env.constants import (DRAG, DT, EFFORT_COST, GEAR_MAGS,
                                      JOINT_ALPHA, JOINT_OMEGA, SABER_FEAT,
                                      gear_table)
from beat_arena_rs import RawBeatSim

COAST = [0] * 14  # env v9: 7 joints/hand


def _sim(gears, effort_coef=0.0):
    """A sim whose only note is far away — pure integrator dynamics."""
    sim = RawBeatSim(max_steps=690, effort_coef=effort_coef, gears=gears)
    n = Note(hit_time=15.0, col=0, row=0, color=1, angle=0.0, speed=4.0)
    sim.set_chart([note_tuple(n)])
    return sim


def _om_obs(obs, head):
    """Normalized joint velocity (om / JOINT_OMEGA) for action head 0-11."""
    hand, j = divmod(head, 7)
    return float(np.asarray(obs)[hand * SABER_FEAT + 8 + j])


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


@pytest.mark.parametrize("gears,aid,frac", [
    (3, 1, 1.0), (3, 2, -1.0),
    (5, 3, 0.25), (5, 4, -0.25),
    (7, 3, 1.0 / 3.0), (7, 5, 0.125), (7, 6, -0.125),
])
def test_gear_torque_scaling(gears, aid, frac):
    """One tick of gear `aid` from rest: om = frac * alpha * DT * (1-DRAG)
    — checked on the right wrist bend (head 12, the snappy tier) and the
    left shoulder pitch (head 0, the slow tier)."""
    for head in (12, 0):
        sim = _sim(gears)
        act = list(COAST)
        act[head] = aid
        obs, _, _, _ = sim.step(act)
        j = head % 7
        want = frac * JOINT_ALPHA[j] * DT * (1.0 - DRAG) / JOINT_OMEGA[j]
        assert _om_obs(obs, head) == pytest.approx(want, abs=1e-6)


def test_doubled_body_limits():
    """Env v8 body: wrist alpha 400 / omega 40, arm 160 / 16. The omega
    cap is observable on the elbow (the longest angular range — at the
    new speeds every joint crosses its range in a handful of ticks, so
    we track the PEAK normalized om before the angle-limit clamp)."""
    assert JOINT_ALPHA[5] == 400.0 and JOINT_OMEGA[5] == 40.0
    assert JOINT_ALPHA[0] == 160.0 and JOINT_OMEGA[0] == 16.0
    sim = _sim(3)
    act = list(COAST)
    act[10] = 1                             # right elbow, 0..150 deg range
    peak = 0.0
    for _ in range(10):
        obs, _, _, _ = sim.step(act)
        peak = max(peak, _om_obs(obs, 10))
    assert peak == pytest.approx(1.0, abs=1e-6)  # clamped AT the cap


def test_ternary_prefix_identical_across_k():
    """A 0-2 action stream produces bit-identical trajectories at K=3,
    K=5, and K=7 (same full-torque semantics, same effort)."""
    rng = np.random.default_rng(7)
    acts = rng.integers(0, 3, size=(90, 14))
    streams = {}
    for k in (3, 5, 7):
        sim = _sim(k, effort_coef=1.0)
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
    """Effort = EFFORT_COST[j] * |torque fraction|: a quarter-gear tick
    costs a quarter of a full-gear tick; coast is free."""
    def one_tick_reward(gears, aid, head=12):
        sim = _sim(gears, effort_coef=1.0)
        act = list(COAST)
        act[head] = aid
        _, r, _, _ = sim.step(act)
        return r

    j = 5                                   # wrist bend
    assert one_tick_reward(5, 0) == 0.0
    full = one_tick_reward(5, 1)
    quarter = one_tick_reward(5, 3)
    assert full == pytest.approx(-EFFORT_COST[j], abs=1e-12)
    assert quarter == pytest.approx(-EFFORT_COST[j] * 0.25, abs=1e-12)
    eighth = one_tick_reward(7, 5)
    assert eighth == pytest.approx(-EFFORT_COST[j] * 0.125, abs=1e-12)


def test_env_action_space_and_header():
    env = RustBeatEnv(gears=5, seed=0)
    assert list(env.action_space.nvec) == [5] * 14
    with pytest.raises(AssertionError):
        RustBeatEnv(gears=4)


def test_policy_head_width_roundtrip():
    from policy import NotePolicy, load_policy
    pol = NotePolicy(n_gears=7)
    obs = torch.randn(4, 290)
    logits, value = pol.forward_stacked(obs)
    assert logits.shape == (4, 14, 7) and value.shape == (4,)
    act, logp, ent, _ = pol.get_action_and_value(obs)
    assert act.shape == (4, 14) and int(act.max()) < 7
    ckpt = {"state_dict": pol.state_dict(), "arch": pol.arch}
    re = load_policy(ckpt)
    assert re.n_gears == 7 and re.head_sizes == (7,) * 14
    # legacy checkpoints (no n_gears in arch) stay ternary
    old = NotePolicy()
    legacy = {"state_dict": old.state_dict(),
              "arch": {k: v for k, v in old.arch.items() if k != "n_gears"}}
    assert load_policy(legacy).n_gears == 3


def test_gears_cuda_parity():
    """K=5 random-action episode: CUDA f64 kernel == Rust exactly (the
    same contract as test_cuda_env's ternary parity)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    chart = generate_chart(ChartConfig(chart_len_s=6.0),
                           np.random.default_rng(31))
    max_steps = int((chart[-1].hit_time + 0.5) * 30) + 90
    rng = np.random.default_rng(41)

    r_env = RustBeatEnv(external_chart=chart, seed=0, gears=5,
                        effort_coef=1.0)
    obs, _ = r_env.reset()
    acts, r_obs, r_rews = [], [obs.copy()], []
    done = False
    while not done:
        a = rng.integers(0, 5, size=14)
        acts.append(a)
        obs, r, done, _, info = r_env.step(a)
        r_obs.append(obs.copy())
        r_rews.append(r)

    c_env = CudaBeatVecEnv(1, charts=[chart], max_steps=max_steps,
                           device="cuda", dtype=torch.float64,
                           auto_reset=False, gears=5, effort_coef=1.0)
    c_obs = [c_env.reset()[0].cpu().numpy().copy()]
    c_rews = []
    done_at = None
    for i, a in enumerate(acts):
        o, r, d, _ = c_env.step(torch.as_tensor(np.asarray([a])))
        c_obs.append(o[0].cpu().numpy().copy())
        c_rews.append(float(r[0]))
        if bool(d[0]):
            done_at = i
            break
    assert done_at == len(acts) - 1
    assert (int(c_env.good[0]), int(c_env.bad[0]), int(c_env.miss[0])) == \
        (info["good_cuts"], info["bad_cuts"], info["misses"])
    np.testing.assert_allclose(c_rews, r_rews, atol=1e-9)
    for k, (co, ro) in enumerate(zip(c_obs[:-1], r_obs[:-1])):
        np.testing.assert_allclose(co, ro, atol=1e-6,
                                   err_msg=f"obs diverge at tick {k}")


def test_deviation_joint():
    """Env v9: dev is a real orthogonal bending axis — at bend = 0 one
    tick of deviation moves the blade laterally at full rate (the old
    roll+bend wrist had out-of-plane speed roll_rate*sin(bend) = 0
    there), dev = 0 reproduces the v8 blade exactly, and the +/-45 deg
    limits clamp."""
    from beat_arena_env.constants import DEV_LIM
    from beat_arena_env.kinematics import fk

    # dev = 0 reproduces the 6-joint blade
    q6 = [0.2, -0.1, 0.3, 0.9, -1.0, 0.7]
    _, w6, b6 = fk(1, q6)
    _, w7, b7 = fk(1, q6 + [0.0])
    assert np.allclose(w6, w7) and np.allclose(b6, b7)

    # straight wrist: deviation tilts the blade off the forearm axis
    _, _, b0 = fk(1, [0.0] * 7)
    _, _, bd = fk(1, [0.0] * 6 + [0.3])
    assert np.allclose(b0, (0.0, 0.0, 1.0), atol=1e-12)
    assert abs(np.dot(b0, bd) - np.cos(0.3)) < 1e-9  # full-rate tilt

    # sim: dev action moves the joint; ASYMMETRIC limits clamp (right
    # [-15, +30] user-specified; left mirrored [-30, +15]; normalizer 30)
    assert DEV_LIM == pytest.approx(np.radians(30.0))

    def run_to_clamp(head, aid):
        sim = _sim(3)
        act = list(COAST)
        act[head] = aid
        for _ in range(30):
            obs, _, _, _ = sim.step(act)
        hand = head // 7
        return float(np.asarray(obs)[hand * SABER_FEAT + 7])

    assert run_to_clamp(13, 1) == pytest.approx(1.0, abs=1e-9)   # R +30
    assert run_to_clamp(13, 2) == pytest.approx(-0.5, abs=1e-9)  # R -15
    assert run_to_clamp(6, 1) == pytest.approx(0.5, abs=1e-9)    # L +15
    assert run_to_clamp(6, 2) == pytest.approx(-1.0, abs=1e-9)   # L -30
