"""env v23: activation cost (user direction 2026-08-17).

Contract: reward -= iso_coef * sum_j |gear_frac_j| per tick, where
gear_frac is the COMMANDED signed torque fraction (gears-5 table:
0 coast, +-1 full, +-1/4 quarter). Force-priced, velocity-blind:
isometric holds and limit-jamming pay it, coast is the only free
command; the charge for a given action vector is a known constant, so
it is pinned exactly by env pairs (coef 0 vs c) stepped with
IDENTICAL actions — identical physics, reward gap == the charge.
iso_coef = 0 (the default everywhere) is the pre-v23 reward verbatim.
"""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="needs CUDA")

KW = dict(push_prob=0.0, push_vmax=0.0, start_jitter=0.0)
COEF = 0.01


def _env(**kw):
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    return CudaBeatVecEnv(64, chart_len_s=8.0, seed=0, device="cuda",
                          foot_contact=True, gears=5, **KW, **kw)


def _pair():
    env0 = _env()
    env1 = _env(iso_coef=COEF)
    env0.reset()
    env1.reset()
    return env0, env1


def _gap(env0, env1, act):
    _, r0, _, _ = env0.step(act)
    _, r1, _, _ = env1.step(act)
    return r0 - r1


@cuda
def test_coast_is_free():
    env0, env1 = _pair()
    gap = _gap(env0, env1, env0._zero_act)
    assert gap.abs().max().item() < 1e-7


@cuda
def test_full_gear_pays_29():
    env0, env1 = _pair()
    act = torch.ones_like(env0._zero_act)          # id 1 = +full, all joints
    gap = _gap(env0, env1, act)
    expect = COEF * 29.0
    assert torch.allclose(gap, torch.full_like(gap, expect), atol=1e-6), (
        gap[:3].tolist(), expect)


@cuda
def test_quarter_pays_quarter_and_sign_blind():
    env0, env1 = _pair()
    # alternate +1/4 (id 3) and -1/4 (id 4): |frac| = 0.25 either way
    act = torch.full_like(env0._zero_act, 3)
    act[:, 1::2] = 4
    gap = _gap(env0, env1, act)
    expect = COEF * 29.0 * 0.25
    assert torch.allclose(gap, torch.full_like(gap, expect), atol=1e-6), (
        gap[:3].tolist(), expect)


@cuda
def test_mixed_command_sums_fractions():
    env0, env1 = _pair()
    # joints 0..9 full (mixed sign), 10..19 quarter, 20..28 coast
    act = env0._zero_act.clone()
    act[:, 0:10:2] = 1
    act[:, 1:10:2] = 2
    act[:, 10:20] = 3
    gap = _gap(env0, env1, act)
    expect = COEF * (10 * 1.0 + 10 * 0.25)
    assert torch.allclose(gap, torch.full_like(gap, expect), atol=1e-6), (
        gap[:3].tolist(), expect)


@cuda
def test_charge_is_velocity_blind():
    """The same command pays the same whether the body is at rest
    (tick 1: isometric-ish) or already in motion (tick 5: whipping) —
    the charge reads the COMMAND, not the resulting q-dot."""
    env0, env1 = _pair()
    act = torch.ones_like(env0._zero_act)
    gaps = [_gap(env0, env1, act) for _ in range(5)]
    expect = COEF * 29.0
    for g in gaps:
        assert torch.allclose(g, torch.full_like(g, expect), atol=1e-6)
