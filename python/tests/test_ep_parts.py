"""Per-episode swing components (2026-10-04, user): the CUDA env's ep_part
accumulators, written into the episode record's cols 28..39 (per hand
[pre, pre max, post, post max, precision, precision max]) at `done` and
zeroed by reset_env. Pinned against the kernel's own per-episode score.
Needs CUDA (run this first on the next pod run: the kernel change was
written without a local nvcc)."""
import numpy as np
import pytest
import torch

from beat_arena_env.constants import N_JOINTS
from tests._helpers import cuda


@cuda
def test_ep_parts_sum_to_the_episode_note_points():
    from beat_arena_env.cuda_env import EP_PART_COL, EP_REC, CudaBeatVecEnv
    assert EP_REC == 40 and EP_PART_COL == 28
    env = CudaBeatVecEnv(256, chart_len_s=6.0, seed=7, device="cuda",
                         dtype=torch.float64, gears=5, ep_stats=True,
                         max_steps=200)
    env.reset()
    rng = np.random.default_rng(1)
    rows = np.zeros((0, EP_REC))
    for _ in range(420):
        a = torch.as_tensor(rng.integers(0, 5, (256, N_JOINTS)),
                            device="cuda")
        env.step(a)
        rows = np.concatenate([rows, env.pop_episodes()])
    assert rows.shape[0] >= 256                       # every env ended
    part = rows[:, EP_PART_COL:EP_PART_COL + 12].reshape(-1, 2, 6)
    score = rows[:, 20]
    assert score.sum() > 0                            # notes were cut
    # generated charts have no chains (a chain link's flat 20 points are
    # in score only), so the three components ARE the episode's points
    comp = part[:, :, 0] + part[:, :, 2] + part[:, :, 4]
    np.testing.assert_allclose(comp.sum(1), score, rtol=0, atol=1e-9)
    # points never exceed their maxima; maxima are whole notes
    for k in (0, 2, 4):
        assert np.all(part[:, :, k] <= part[:, :, k + 1] + 1e-9)
        assert np.all(part[:, :, k] >= 0)
    np.testing.assert_allclose(part[:, :, 1] / 70.0, part[:, :, 5] / 15.0)
    np.testing.assert_allclose(np.mod(part[:, :, 1], 70.0), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.mod(part[:, :, 3], 30.0), 0.0, atol=1e-9)
    # one pre max per good note cut (no chains: good cuts = note cuts)
    np.testing.assert_allclose(part[:, :, 1].sum(1) / 70.0, rows[:, 19])
    # both hands score somewhere over the batch (hand-major layout)
    assert part[:, 0, 1].sum() > 0 and part[:, 1, 1].sum() > 0


@cuda
def test_ep_parts_reset_with_the_episode():
    """A row's swing cols cover THAT episode only: the per-episode
    accumulator restarts at reset_env (the trainer's acc_part flow does
    not — popped per update instead)."""
    from beat_arena_env.cuda_env import EP_PART_COL, EP_REC, CudaBeatVecEnv
    env = CudaBeatVecEnv(64, chart_len_s=6.0, seed=3, device="cuda",
                         dtype=torch.float64, gears=5, ep_stats=True,
                         max_steps=60)
    env.reset()
    rng = np.random.default_rng(2)
    rows = np.zeros((0, EP_REC))
    for _ in range(200):                  # >= 3 episodes per env
        a = torch.as_tensor(rng.integers(0, 5, (64, N_JOINTS)),
                            device="cuda")
        env.step(a)
        rows = np.concatenate([rows, env.pop_episodes()])
    assert rows.shape[0] >= 3 * 64
    part = rows[:, EP_PART_COL:EP_PART_COL + 12].reshape(-1, 2, 6)
    # never more pre max than the episode's own good cuts could earn
    assert np.all(part[:, :, 1].sum(1) <= 70.0 * rows[:, 19] + 1e-9)
    comp = (part[:, :, 0] + part[:, :, 2] + part[:, :, 4]).sum(1)
    np.testing.assert_allclose(comp, rows[:, 20], rtol=0, atol=1e-9)
