"""env v19: the remaining-follow-through-credit obs lanes (spec §12,
hand blocks 10 -> 11 lanes, OBS_DIM 580).

Lane semantics (both engines): obs[hand_base + 10] =
min(Σ_live_trackers cnt · (1 − min(acc/POST_FULL, 1)), 2), so 0 means
"swinging earns nothing now" (idle / broken / capped), a fresh
single-note tracker reads 1.0, and the value decays as follow-through
angle accrues. Cross-engine parity on real trajectories is owned by
the existing parity suites; this file pins the formula itself by
injecting tracker states directly into the CUDA tensors and reading
the obs back through the mode-2 refresh (the K19 write_obs tail runs
on every mode).
"""

import math

import torch

from beat_arena_env.constants import (OBS_DIM, OFF_HAND_L, OFF_HAND_R,
                                      POST_FULL)
from tests._helpers import cuda

LANE_L = OFF_HAND_L + 10   # 85
LANE_R = OFF_HAND_R + 10   # 96


def _env(n=4):
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    return CudaBeatVecEnv(n, chart_len_s=8.0, seed=0, device="cuda")


def _refresh(env):
    env._launch(env._zero_act, env._zero_cand, 2)
    torch.cuda.synchronize()
    return env._obs


@cuda
def test_reset_lanes_zero():
    env = _env()
    obs = env.reset()
    assert obs.shape[1] == OBS_DIM == 624
    assert not obs[:, LANE_L].any()
    assert not obs[:, LANE_R].any()


@cuda
def test_injected_tracker_formula():
    env = _env()
    env.reset()
    # env 0, left hand: quarter-accrued single tracker (term 0.75) +
    # a capped one (term 0, any cnt) -> 0.75
    env.tr_active[0, 0, 0] = True
    env.tr_acc[0, 0, 0] = 0.25 * POST_FULL
    env.tr_cnt[0, 0, 0] = 1
    env.tr_active[0, 0, 3] = True
    env.tr_acc[0, 0, 3] = 2.0 * POST_FULL
    env.tr_cnt[0, 0, 3] = 2
    # env 0, right hand: fresh triple-note tracker -> 3.0 clipped to 2
    env.tr_active[0, 1, 1] = True
    env.tr_acc[0, 1, 1] = 0.0
    env.tr_cnt[0, 1, 1] = 3
    # env 1: half-accrued pair on the right -> 2 x 0.5 = 1.0
    env.tr_active[1, 1, 0] = True
    env.tr_acc[1, 1, 0] = 0.5 * POST_FULL
    env.tr_cnt[1, 1, 0] = 2
    obs = _refresh(env)
    assert math.isclose(obs[0, LANE_L].item(), 0.75, abs_tol=1e-6)
    assert math.isclose(obs[0, LANE_R].item(), 2.0, abs_tol=1e-6)
    assert math.isclose(obs[1, LANE_R].item(), 1.0, abs_tol=1e-6)
    # untouched hands stay 0
    assert obs[1, LANE_L].item() == 0.0
    assert not obs[2:, LANE_L].any() and not obs[2:, LANE_R].any()


@cuda
def test_lane_decays_within_a_swing():
    """A live tracker's lane must be monotone non-increasing while the
    swing continues: step with zero torque (coast) — the tracker either
    accrues (lane drops) or the chain breaks (tracker resolves, lane
    goes 0). Either way the lane never rises without a new cut."""
    env = _env()
    env.reset()
    env.tr_active[0, 0, 0] = True
    env.tr_acc[0, 0, 0] = 0.1 * POST_FULL
    env.tr_cnt[0, 0, 0] = 1
    prev = _refresh(env)[0, LANE_L].item()
    assert math.isclose(prev, 0.9, abs_tol=1e-6)
    for _ in range(3):
        env.step(torch.zeros_like(env._zero_act))
        cur = env._obs[0, LANE_L].item()
        assert cur <= prev + 1e-6
        prev = cur
