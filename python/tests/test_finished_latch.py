"""Finished-episode latch on the CUDA backend (2026-09-27).

MIRROR of lib.rs's B4 `finished` latch: with auto_reset=False a slot
whose episode ended reports (reward 0, done) on every later tick, runs
no physics, accrues no telemetry, logs no second episode and keeps its
terminal obs row — until reset() re-installs a chart. Before the fix
the kernel kept stepping the finished body and re-reported its
terminal reward (-fall_const) every tick.

Contact-mode zero-action episodes fall on their own (the unactuated
body slumps), which gives a deterministic terminal tick.
"""

import numpy as np
import pytest
import torch

from beat_arena_env.chart import Note
from beat_arena_env.constants import N_JOINTS

try:
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    HAVE_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - CPU-only box
    HAVE_CUDA = False

pytestmark = pytest.mark.skipif(not HAVE_CUDA, reason="needs CUDA")

# a far-future note keeps the episode alive until the body falls
SPARSE = [Note(hit_time=100.0, col=2, row=1, x=2.5, y=1.4, color=0,
               angle=4.712, speed=4.0)]


def _env(dtype, **kw):
    return CudaBeatVecEnv(2, charts=[SPARSE, SPARSE], max_steps=3000,
                          gears=5, fall_const=5.0, dtype=dtype,
                          device="cuda", auto_reset=False, tau_rand=0.1,
                          **kw)


def _run_to_done(env, act):
    """Step until every slot is done; returns each slot's terminal-tick
    reward (tau_rand gives the two slots different bodies, so they may
    fall on different ticks)."""
    term = [None] * env.num_envs
    for _t in range(600):
        obs, rew, done, _ = env.step(act)
        for e in range(env.num_envs):
            if bool(done[e]) and term[e] is None:
                term[e] = float(rew[e])
        if all(r is not None for r in term):
            return obs.clone(), term
    raise AssertionError("zero-action bodies never fell")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finished_slot_latches(dtype):
    env = _env(dtype)
    env.reset()
    act = torch.zeros((2, N_JOINTS), dtype=torch.long)  # gear 0 = zero torque
    obs_done, term = _run_to_done(env, act)
    assert term == [-5.0, -5.0], "terminal tick carries -fall_const"
    q_done = env.qpos().clone()
    v_done = env.v.clone()
    tau_done = env.tau_stat.clone()
    n_eps = env.pop_episodes().shape[0]
    assert n_eps == 2
    for _ in range(20):
        obs, rew, done, _ = env.step(act)
        assert bool(done.all()), "done must stay latched"
        assert (rew == 0).all(), "a finished slot must earn exactly 0"
        assert torch.equal(obs, obs_done), "terminal obs row must hold"
    assert torch.equal(env.qpos(), q_done), "physics ran on a finished slot"
    assert torch.equal(env.v, v_done), "physics ran on a finished slot"
    assert torch.equal(env.tau_stat, tau_done), "telemetry kept accruing"
    assert env.pop_episodes().shape[0] == 0, "episode logged twice"
    # reset() is the only thing that re-arms the slot
    env.reset()
    _obs, rew, done, _ = env.step(act)
    assert not bool(done.any()), "reset() must clear the latch"
    assert (rew != 0).all()


def test_auto_reset_still_restarts():
    """auto_reset=True: a done flag means the slot was ALREADY reset on
    its terminal tick, so the next tick must step the fresh episode."""
    env = CudaBeatVecEnv(2, charts=[SPARSE, SPARSE], max_steps=3000,
                         gears=5, fall_const=5.0, dtype=torch.float32,
                         device="cuda", auto_reset=True)
    env.reset()
    act = torch.zeros((2, N_JOINTS), dtype=torch.long)
    _run_to_done(env, act)
    _obs, rew, done, _ = env.step(act)
    assert not bool(done.any())
    assert np.all(rew.cpu().numpy() != -5.0)
