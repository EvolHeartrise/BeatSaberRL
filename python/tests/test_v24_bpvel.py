"""env v24: §8 body-point world-velocity obs lanes (580..598).

Contract: the 6 body points ([l_elbow, r_elbow, l_knee, r_knee,
l_foot, r_foot]) report their world velocity as a wrist_vel-style
finite difference across the tick, /V_NORM — pinned against the
positions the SAME obs reports (the velocity lane must equal the
per-tick delta of the position lane, denormalized, /DT). Zero at
reset and after a set_state refresh.
"""

import pytest
import torch

from beat_arena_env.constants import (OBS_DIM, OFF_BODY_PTS,
                                      OFF_BODY_PT_VEL, V_NORM, X_NORM,
                                      Y_NORM)

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="needs CUDA")

DT = 1.0 / 30.0
KW = dict(push_prob=0.0, push_vmax=0.0, start_jitter=0.0)


def _env():
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    return CudaBeatVecEnv(64, chart_len_s=8.0, seed=0, device="cuda",
                          dtype=torch.float64, foot_contact=True,
                          gears=5, **KW)


def _vel_lanes(obs):
    return obs[:, OFF_BODY_PT_VEL:OFF_BODY_PT_VEL + 18]


def _pos_raw(obs):
    """Denormalized xyz deltas need only the SCALE per axis — the
    per-point y0/z0 offsets cancel in the finite difference."""
    p = obs[:, OFF_BODY_PTS:OFF_BODY_PTS + 18].clone()
    scale = torch.tensor([X_NORM, Y_NORM, X_NORM],
                         device=p.device, dtype=p.dtype).repeat(6)
    return p * scale


@cuda
def test_obs_dim_and_reset_zero():
    env = _env()
    obs = env.reset()
    assert obs.shape[1] == OBS_DIM == 626
    assert _vel_lanes(obs).abs().max().item() == 0.0


@cuda
def test_vel_is_position_delta_over_dt():
    env = _env()
    # step() returns the SAME obs buffer object every call (written in
    # place) — clone, or prev aliases the current tick and the delta
    # reads zero
    prev = env.reset().clone()
    # a couple of gravity/settle ticks produce real motion at every
    # point; the vel lane must equal the position lanes' delta
    for _ in range(3):
        obs, _, _, _ = env.step(env._zero_act)
        want = (_pos_raw(obs) - _pos_raw(prev)) / DT / V_NORM
        got = _vel_lanes(obs)
        # obs lanes are f32; the position-delta path loses ~1e-6 to
        # rounding after the /DT/V_NORM rescale
        assert torch.allclose(got, want, atol=1e-5), (
            (got - want).abs().max().item())
        prev = obs.clone()
    # and the body actually moved (the contract above is not 0 == 0)
    assert _vel_lanes(prev).abs().max().item() > 1e-4


@cuda
def test_set_state_refresh_zeroes_vel():
    env = _env()
    env.reset()
    env.step(env._zero_act)
    assert _vel_lanes(env._obs).abs().max().item() > 1e-4  # moving
    q36 = [0.0, 0.94, -0.65] + [1.0, 0.0, 0.0, 0.0] + [0.0] * 29
    env.set_state(q36, [0.0] * 35)  # mode-2 refresh rewrites the obs
    assert _vel_lanes(env._obs).abs().max().item() == 0.0
