"""Parity tests: TorchBeatVecEnv vs the Rust sim (THE reference), env v7.

Tier 1 (float64, CPU): exact agreement — same chart + same actions must give
identical rewards, done timing, event counters, score, and obs (to f32
rounding). Tier 2 (float32, GPU): bounded drift — hundreds of ticks of fp32
integration may flip knife-edge cut decisions, so counters get a small
tolerance. Skipped without CUDA.
"""

import numpy as np
import pytest
import torch

from beat_arena_env import RustBeatEnv, scripted_action
from beat_arena_env.chart import ChartConfig, generate_chart
from beat_arena_env.constants import OBS_DIM, Z_MISS
from beat_arena_env.torch_env import TorchBeatVecEnv

EASY = dict(colors="right", bomb_prob=0.0, extra_prob=0.0)
FULL = dict(colors="both")     # the generator-v2 default gym
DENSE = dict(colors="both", note_prob=0.5, extra_prob=0.4, bomb_prob=0.25)
ACTION_NVEC = [3] * 12


def _chart(cfg_kw, seed):
    return generate_chart(ChartConfig(**cfg_kw), np.random.default_rng(seed))


def _max_steps(chart):
    """Mirror RustBeatEnv's external-chart tick cap (+90: episode tail
    covers the Z_MISS trip at the slowest speeds)."""
    return int((chart[-1].hit_time + 0.5) * 30) + 90


def _rust_episode(chart, policy):
    """Run one Rust episode; returns (actions, obs_list, rewards, counters)."""
    env = RustBeatEnv(external_chart=chart, seed=0)
    obs, _ = env.reset()
    acts, obs_l, rews = [], [obs.copy()], []
    done = False
    while not done:
        a = policy(env, obs)
        acts.append(a)
        obs, r, done, _, info = env.step(a)
        obs_l.append(obs.copy())
        rews.append(r)
    return acts, obs_l, rews, (info["good_cuts"], info["bad_cuts"],
                               info["misses"], info["bombs_hit"],
                               info["score"])


def _torch_episode(chart, actions, device, dtype):
    env = TorchBeatVecEnv(1, charts=[chart], max_steps=_max_steps(chart),
                          device=device, dtype=dtype, auto_reset=False)
    obs_l = [env.reset()[0].cpu().numpy().copy()]
    rews, done_at = [], None
    for i, a in enumerate(actions):
        obs, r, done, _ = env.step(torch.as_tensor(np.asarray([a])))
        obs_l.append(obs[0].cpu().numpy().copy())
        rews.append(float(r[0]))
        if bool(done[0]):
            done_at = i
            break
    counters = (int(env.good[0]), int(env.bad[0]), int(env.miss[0]),
                int(env.bombs[0]), float(env.score[0]))
    return obs_l, rews, done_at, counters


def _random_policy(seed):
    rng = np.random.default_rng(seed)
    return lambda env, obs: rng.integers(0, ACTION_NVEC)


def _scripted_policy():
    return lambda env, obs: scripted_action(env)


@pytest.mark.parametrize("cfg_kw,pol,seed", [
    (EASY, "scripted", 11), (FULL, "scripted", 12),
    (EASY, "random", 13), (FULL, "random", 14),
    (DENSE, "scripted", 15), (DENSE, "random", 16),
])
def test_parity_exact_f64(cfg_kw, pol, seed):
    chart = _chart(cfg_kw, seed)
    policy = _scripted_policy() if pol == "scripted" else _random_policy(seed)
    acts, r_obs, r_rews, r_cnt = _rust_episode(chart, policy)
    t_obs, t_rews, done_at, t_cnt = _torch_episode(
        chart, acts, "cpu", torch.float64)
    assert done_at == len(acts) - 1, "episode length differs"
    assert t_cnt[:4] == r_cnt[:4], f"counters differ: torch {t_cnt} rust {r_cnt}"
    assert t_cnt[4] == pytest.approx(r_cnt[4], abs=1e-6)
    np.testing.assert_allclose(t_rews, r_rews, atol=1e-9)
    for k, (to, ro) in enumerate(zip(t_obs[:-1], r_obs[:-1])):
        np.testing.assert_allclose(to, ro, atol=1e-6,
                                   err_msg=f"obs diverge at tick {k}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("cfg_kw,seed", [(EASY, 21), (FULL, 22), (DENSE, 23)])
def test_parity_f32_gpu(cfg_kw, seed):
    chart = _chart(cfg_kw, seed)
    acts, r_obs, r_rews, r_cnt = _rust_episode(chart, _scripted_policy())
    t_obs, t_rews, done_at, t_cnt = _torch_episode(
        chart, acts, "cuda", torch.float32)
    # fp32 may flip knife-edge decisions; bound the drift, don't demand zero
    assert done_at is not None and abs(done_at - (len(acts) - 1)) <= 3
    for a, b in zip(t_cnt[:4], r_cnt[:4]):
        assert abs(a - b) <= 2, f"counters drifted: torch {t_cnt} rust {r_cnt}"
    assert t_cnt[4] == pytest.approx(r_cnt[4], abs=60.0)  # ~half a note's pts
    n = min(len(t_obs), len(r_obs)) - 1
    np.testing.assert_allclose(t_obs[:n], r_obs[:n], atol=5e-3)


def test_auto_reset_and_stats():
    env = TorchBeatVecEnv(4, seed=3, chart_pool=8, device="cpu",
                          dtype=torch.float32, **DENSE)
    obs = env.reset()
    assert obs.shape == (4, OBS_DIM)
    rng = np.random.default_rng(0)
    # two episode lengths (chart 20 s + tail 0.5 + Z_MISS/speed_min):
    # every env cycles
    n_steps = 2 * (int((20.0 + 0.5 + Z_MISS / 2.0) * 30) + 5)
    dones = 0
    for _ in range(n_steps):
        a = torch.as_tensor(rng.integers(0, ACTION_NVEC, size=(4, 12)))
        obs, r, done, infos = env.step(a)
        assert infos == []
        dones += int(done.sum())
    assert torch.isfinite(obs).all()
    stats = env.pop_stats()
    assert len(stats) == dones and dones >= 4
    assert all(np.isfinite(s) for s in stats)
    assert env.pop_stats() == []  # second pop: nothing new


def test_same_seed_reproduces():
    def run(seed):
        env = TorchBeatVecEnv(2, seed=seed, chart_pool=4, device="cpu",
                              dtype=torch.float32, **DENSE)
        obs = env.reset()
        rng = np.random.default_rng(1)
        tot = torch.zeros(2)
        for _ in range(400):
            a = torch.as_tensor(rng.integers(0, ACTION_NVEC, size=(2, 12)))
            obs, r, done, _ = env.step(a)
            tot += r
        return obs.clone(), tot
    o1, t1 = run(5)
    o2, t2 = run(5)
    assert torch.equal(o1, o2) and torch.equal(t1, t2)
