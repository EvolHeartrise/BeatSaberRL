"""env v20: start-pose leg jitter (training-only, CUDA) + foot-z
penalty (both engines; CUDA side tested here, coefficient default 0
keeps every parity suite on the pre-v20 reward verbatim).

Jitter contract (kernel reset_env): with start_pose_jitter = s > 0,
the 12 leg joints (§1.4 indices 17..28) reset to U(-s, s) clamped to
their ranges — knees (20, 26) take |draw| — and the root drops so the
lowest of the 8 sole corners (contact spec §2 GROUND_CORNERS) sits
EXACTLY on the ground plane. Verified against the python fk_body
mirror. Penalty contract: reward -= coef * (|lfoot_z - PLANE_Z0| +
|rfoot_z - PLANE_Z0|) per tick, feet read from the same FK the §8
body_pts obs lanes report.
"""

import math

import numpy as np
import pytest
import torch

from beat_arena_env.constants import (FOOT_X0, JOINT_LIMITS, OFF_BODY_PTS,
                                      PLANE_Z0, X_NORM, q_start)
from beat_arena_env.kinematics import F_L_FOOT, F_R_FOOT, fk_body

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="needs CUDA")

GROUND_CORNERS = ((-0.045, -0.070, -0.075), (-0.045, -0.070, 0.175),
                  (0.045, -0.070, -0.075), (0.045, -0.070, 0.175))
LEGS = list(range(17, 29))
KNEES = (20, 26)


def _env(n=64, **kw):
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    return CudaBeatVecEnv(n, chart_len_s=8.0, seed=0, device="cuda",
                          foot_contact=True, **kw)


def _q36(env, e):
    p = env.p[e].tolist()
    quat = env.quat[e].tolist()
    q = env.q[e].tolist()
    return p + quat + q


def _min_corner_y(q36):
    rots, pos = fk_body(q36)
    m = 1e9
    for fr in (F_L_FOOT, F_R_FOOT):
        for c in GROUND_CORNERS:
            y = pos[fr][1] + sum(rots[fr][1][k] * c[k] for k in range(3))
            m = min(m, y)
    return m


Q0_LEGS = [q_start(foot_contact=True)[j] for j in LEGS]


@cuda
def test_reset_is_stance_grounded():
    """env v21: a plain contact reset lands on the wide stance with the
    root dropped so the soles sit exactly on the ground."""
    env = _env(start_pose_jitter=0.0)
    env.reset()
    exp = torch.tensor(Q0_LEGS, device=env.q.device,
                       dtype=env.q.dtype).expand(env.num_envs, -1)
    assert torch.allclose(env.q[:, LEGS], exp, atol=1e-6)
    # arms at the ready stance too
    assert torch.allclose(
        env.q[:, 3], torch.full_like(env.q[:, 3], -1.3962634015954636))
    # root dropped ~8.5 mm; the exact contract is the corner check
    assert (env.p[:, 1] < 0.94).all()
    for e in range(0, env.num_envs, 7):
        my = _min_corner_y(_q36(env, e))
        assert abs(my) < 5e-4, (e, my)


@cuda
def test_pose_jitter_distribution_limits_and_ground():
    s = 0.3
    env = _env(start_pose_jitter=s)
    env.reset()
    legs = env.q[:, LEGS]
    # varied across envs, centered on the v21 stance, bounded by the
    # jitter scale around it
    exp = torch.tensor(Q0_LEGS, device=legs.device,
                       dtype=legs.dtype).expand_as(legs)
    assert (legs - exp).abs().max().item() <= s + 1e-6
    assert (legs.std(dim=0) > 0.01).all()
    # knees one-way: |draw| keeps them nonnegative (and bent half the
    # time rather than clamped flat)
    for j in KNEES:
        kn = env.q[:, j]
        assert (kn >= -1e-9).all()
        assert (kn > 0.05).float().mean().item() > 0.5
    # joint ranges respected
    for k, j in enumerate(LEGS):
        lo, hi = JOINT_LIMITS[j]
        assert legs[:, k].min().item() >= lo - 1e-6
        assert legs[:, k].max().item() <= hi + 1e-6
    # EXACT no-clip contract: lowest sole corner ON the ground
    for e in range(0, env.num_envs, 7):
        my = _min_corner_y(_q36(env, e))
        assert abs(my) < 5e-4, (e, my)
    # smoke: stepping the jittered poses doesn't crash (zero-torque
    # bodies with bent knees DO topple — that's physics, not a bug, so
    # no fall-count assertion here)
    for _ in range(15):
        env.step(env._zero_act)


@cuda
def test_foot_z_penalty_matches_obs_feet():
    kw = dict(push_prob=0.0, push_vmax=0.0, start_jitter=0.0)
    env0 = _env(foot_z_coef=0.0, **kw)
    env1 = _env(foot_z_coef=0.02, **kw)
    env0.reset()
    env1.reset()
    # shift both roots 0.3 m toward the notes; identical physics, so
    # the reward gap is exactly the penalty
    q36 = [0.0, 0.94, -0.65 + 0.3] + [1.0, 0.0, 0.0, 0.0] + [0.0] * 29
    v35 = [0.0] * 35
    for env in (env0, env1):
        env.set_state(q36, v35)
    obs0, r0, _, _ = env0.step(env0._zero_act)
    obs1, r1, _, _ = env1.step(env1._zero_act)
    assert torch.allclose(obs0, obs1)
    # feet from the §8 body_pts lanes of the SAME tick (z lane =
    # (z - PLANE_Z0) / X_NORM), the exact source the penalty reads
    lz = obs0[:, OFF_BODY_PTS + 4 * 3 + 2] * X_NORM
    rz = obs0[:, OFF_BODY_PTS + 5 * 3 + 2] * X_NORM
    expect = 0.02 * (lz.abs() + rz.abs())
    got = r0 - r1
    assert torch.allclose(got, expect, atol=1e-4), (
        got[:3].tolist(), expect[:3].tolist())
    # and the magnitude is the commanded shift, ~0.3 m per foot
    assert (lz.abs() > 0.25).all() and (rz.abs() > 0.25).all()


@cuda
def test_foot_x_penalty_matches_obs_feet():
    kw = dict(push_prob=0.0, push_vmax=0.0, start_jitter=0.0)
    env0 = _env(foot_x_coef=0.0, **kw)
    env1 = _env(foot_x_coef=0.02, **kw)
    env0.reset()
    env1.reset()
    # shift both roots 0.3 m to the right; identical physics, so the
    # reward gap is exactly the per-side-target x penalty
    q36 = [0.3, 0.94, -0.65] + [1.0, 0.0, 0.0, 0.0] + [0.0] * 29
    v35 = [0.0] * 35
    for env in (env0, env1):
        env.set_state(q36, v35)
    obs0, r0, _, _ = env0.step(env0._zero_act)
    obs1, r1, _, _ = env1.step(env1._zero_act)
    assert torch.allclose(obs0, obs1)
    lx = obs0[:, OFF_BODY_PTS + 4 * 3] * X_NORM
    rx = obs0[:, OFF_BODY_PTS + 5 * 3] * X_NORM
    expect = 0.02 * ((lx + FOOT_X0).abs() + (rx - FOOT_X0).abs())
    got = r0 - r1
    assert torch.allclose(got, expect, atol=1e-4), (
        got[:3].tolist(), expect[:3].tolist())


@cuda
def test_foot_x_penalty_zero_at_reset():
    """At the plain v21 reset the feet sit AT their +-FOOT_X0 targets,
    so the x penalty is ~0 — pins the target constants to the actual
    reset geometry."""
    kw = dict(push_prob=0.0, push_vmax=0.0, start_jitter=0.0)
    env0 = _env(foot_x_coef=0.0, **kw)
    env1 = _env(foot_x_coef=0.02, **kw)
    env0.reset()
    env1.reset()
    _, r0, _, _ = env0.step(env0._zero_act)
    _, r1, _, _ = env1.step(env1._zero_act)
    assert (r0 - r1).abs().max().item() < 2e-3


@cuda
def test_foot_z_penalty_zero_at_reference():
    kw = dict(push_prob=0.0, push_vmax=0.0, start_jitter=0.0)
    env0 = _env(foot_z_coef=0.0, **kw)
    env1 = _env(foot_z_coef=0.02, **kw)
    env0.reset()
    env1.reset()
    _, r0, _, _ = env0.step(env0._zero_act)
    _, r1, _, _ = env1.step(env1._zero_act)
    # feet start at the reference z: penalty ~ 0 (only the tick's tiny
    # settling motion contributes)
    assert (r0 - r1).abs().max().item() < 2e-3
