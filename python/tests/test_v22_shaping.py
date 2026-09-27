"""env v22: hinged foot rent + potential-based delta shaping.

Contracts: (1) with foot_hinge_r = r, deviation inside r is FREE and
rent applies only to the excess (r = 0 reproduces the v21 pure-linear
penalty, pinned by the surviving test_v20_legs penalty tests);
(2) the shaping term pays coef * (prev_pot - pot) per tick where pot =
Σ hinged deviations of both feet on both axes — verified against the
pot computed from the SAME §8 body_pts obs lanes the kernel reads;
(3) resets start on-target with prev_pot = 0, so both terms are ~0 at
a plain reset.
"""

import torch

from beat_arena_env.constants import FOOT_X0, OFF_BODY_PTS, PLANE_Z0, X_NORM
from tests._helpers import cuda


def _env(**kw):
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    return CudaBeatVecEnv(64, chart_len_s=8.0, seed=0, device="cuda",
                          **kw)


def _feet(obs):
    lx = obs[:, OFF_BODY_PTS + 4 * 3] * X_NORM
    lz = obs[:, OFF_BODY_PTS + 4 * 3 + 2] * X_NORM + PLANE_Z0
    rx = obs[:, OFF_BODY_PTS + 5 * 3] * X_NORM
    rz = obs[:, OFF_BODY_PTS + 5 * 3 + 2] * X_NORM + PLANE_Z0
    return lx, lz, rx, rz


def _pot(obs, r):
    lx, lz, rx, rz = _feet(obs)
    h = lambda v: (v.abs() - r).clamp_min(0.0)
    return (h(lx + FOOT_X0) + h(rx - FOOT_X0)
            + h(lz - PLANE_Z0) + h(rz - PLANE_Z0))


def _shifted(env, dz):
    q36 = [0.0, 0.94, -0.65 + dz] + [1.0, 0.0, 0.0, 0.0] + [0.0] * 29
    env.set_state(q36, [0.0] * 35)


@cuda
def test_hinge_frees_the_target_area():
    env0 = _env(foot_z_coef=0.0)
    env1 = _env(foot_z_coef=0.02, foot_hinge_r=0.12)
    for env in (env0, env1):
        env.reset()
        _shifted(env, 0.05)          # inside the radius
    _, r0, _, _ = env0.step(env0._zero_act)
    _, r1, _, _ = env1.step(env1._zero_act)
    assert (r0 - r1).abs().max().item() < 1e-5   # rent exactly 0 inside


@cuda
def test_hinge_prices_only_the_excess():
    env0 = _env(foot_z_coef=0.0)
    env1 = _env(foot_z_coef=0.02, foot_hinge_r=0.12)
    for env in (env0, env1):
        env.reset()
        _shifted(env, 0.30)          # 0.18 outside the radius per foot
    obs0, r0, _, _ = env0.step(env0._zero_act)
    _, r1, _, _ = env1.step(env1._zero_act)
    _, lz, _, rz = _feet(obs0)
    h = lambda v: (v.abs() - 0.12).clamp_min(0.0)
    expect = 0.02 * (h(lz - PLANE_Z0) + h(rz - PLANE_Z0))
    assert torch.allclose(r0 - r1, expect, atol=1e-4)


@cuda
def test_shaping_pays_the_delta():
    env0 = _env()
    env1 = _env(foot_shape_coef=0.05, foot_hinge_r=0.12)
    for env in (env0, env1):
        env.reset()                  # prev_pot = 0, feet on target
        _shifted(env, 0.30)
    # first step: prev_pot is still 0 (set_state does not touch it),
    # pot jumps to P1 -> shaping = 0.05 * (0 - P1)
    obs_a, r0a, _, _ = env0.step(env0._zero_act)
    _, r1a, _, _ = env1.step(env1._zero_act)
    p1 = _pot(obs_a, 0.12)
    assert torch.allclose(r0a - r1a, 0.05 * p1, atol=1e-4), (
        (r0a - r1a)[:3].tolist(), (0.05 * p1)[:3].tolist())
    # second step: shaping = 0.05 * (P1 - P2), tiny under zero torque
    obs_b, r0b, _, _ = env0.step(env0._zero_act)
    _, r1b, _, _ = env1.step(env1._zero_act)
    p2 = _pot(obs_b, 0.12)
    assert torch.allclose(r0b - r1b, -0.05 * (p1 - p2), atol=1e-4)


@cuda
def test_all_terms_zero_at_reset():
    env0 = _env()
    env1 = _env(foot_z_coef=0.02, foot_x_coef=0.02,
                foot_hinge_r=0.12, foot_shape_coef=0.05)
    env0.reset()
    env1.reset()
    _, r0, _, _ = env0.step(env0._zero_act)
    _, r1, _, _ = env1.step(env1._zero_act)
    assert (r0 - r1).abs().max().item() < 2e-3
