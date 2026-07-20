"""Policy tests (SPEC.md §Tests 5 + shape sanity)."""

import numpy as np
import pytest
import torch

from beat_arena_env import BeatEnv, OFF_NOTES, OFF_NOTE_MASK, NOTE_FEAT
from policy import NotePolicy, OBS_DIM


def _obs_batch(n=6):
    # short chart -> fewer pending notes than slots, so masked slots exist
    env = BeatEnv(seed=1, chart_len_s=6.0)
    obs, _ = env.reset()
    rows = [obs]
    rng = np.random.default_rng(0)
    for _ in range(n - 1):
        a = rng.integers(0, 9, size=2)
        obs, _, done, _, _ = env.step(a)
        if done:
            obs, _ = env.reset()
        rows.append(obs)
    return torch.as_tensor(np.stack(rows), dtype=torch.float32)


@pytest.mark.parametrize("encoder", ["pool", "transformer"])
def test_output_shapes(encoder):
    torch.manual_seed(0)
    policy = NotePolicy(encoder=encoder)
    obs = _obs_batch()
    left_l, right_l, value = policy(obs)
    assert left_l.shape == (6, 9) and right_l.shape == (6, 9)
    assert value.shape == (6,)
    a, logp, ent, v = policy.get_action_and_value(obs)
    assert a.shape == (6, 2) and logp.shape == (6,)


@pytest.mark.parametrize("encoder", ["pool", "transformer"])
def test_junk_in_masked_slots_cannot_leak(encoder):
    """Garbage in invalid note slots must not change any output."""
    torch.manual_seed(0)
    policy = NotePolicy(encoder=encoder)
    obs = _obs_batch()
    mask = obs[:, OFF_NOTE_MASK:]
    assert (mask == 0).any(), "test needs at least one empty slot"

    junk = obs.clone()
    g = torch.Generator().manual_seed(42)
    for b in range(obs.shape[0]):
        for slot in torch.nonzero(mask[b] == 0).flatten():
            s = OFF_NOTES + int(slot) * NOTE_FEAT
            junk[b, s:s + NOTE_FEAT] = torch.randn(NOTE_FEAT, generator=g) * 5

    action = torch.randint(0, 9, (obs.shape[0], 2))
    _, lp1, ent1, v1 = policy.get_action_and_value(obs, action)
    _, lp2, ent2, v2 = policy.get_action_and_value(junk, action)
    assert torch.allclose(lp1, lp2, atol=1e-5)
    assert torch.allclose(ent1, ent2, atol=1e-5)
    assert torch.allclose(v1, v2, atol=1e-5)


def test_obs_dim_contract():
    assert OBS_DIM == 188
