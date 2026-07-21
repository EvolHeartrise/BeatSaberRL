"""Policy + vec-env tests (transformer-only since 2026-07-20)."""

import numpy as np
import pytest
import torch

from beat_arena_env import RustBeatEnv, OFF_NOTES, OFF_NOTE_MASK, NOTE_FEAT
from beat_arena_env.vec_env import BeatVecEnv
from policy import NotePolicy, OBS_DIM


def _obs_batch(n=6):
    # short chart -> fewer pending notes than slots, so masked slots exist
    env = RustBeatEnv(seed=1, chart_len_s=6.0)
    obs, _ = env.reset()
    rows = [obs]
    rng = np.random.default_rng(0)
    for _ in range(n - 1):
        obs, _, done, _, _ = env.step(rng.integers(0, 9, size=2))
        if done:
            obs, _ = env.reset()
        rows.append(obs)
    return torch.as_tensor(np.stack(rows), dtype=torch.float32)


def test_output_shapes():
    torch.manual_seed(0)
    policy = NotePolicy()
    obs = _obs_batch()
    left_l, right_l, value = policy(obs)
    assert left_l.shape == (6, 9) and right_l.shape == (6, 9)
    assert value.shape == (6,)
    a, logp, ent, v = policy.get_action_and_value(obs)
    assert a.shape == (6, 2) and logp.shape == (6,)


def test_junk_in_masked_slots_cannot_leak():
    """Garbage in invalid note slots must not change any output."""
    torch.manual_seed(0)
    policy = NotePolicy()
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
    assert OBS_DIM == 128


def test_pool_encoder_removed():
    with pytest.raises(AssertionError):
        NotePolicy(encoder="pool")


def test_vec_env_auto_reset():
    venv = BeatVecEnv(4, colors="both", dirs=tuple(range(8)), chart_len_s=5.0, seed=3)
    obs = venv.reset()
    assert obs.shape == (4, 128)
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(400):
        obs, rew, done, infos = venv.step(rng.integers(0, 9, size=(4, 2)))
        finished += len(infos)
        for info in infos:
            assert info["n_notes"] > 0
            assert obs[info["env"], 116:].sum() > 0  # fresh pending notes
    assert finished >= 4


def test_vec_matches_single():
    """VecBeatSim(1) must step identically to RawBeatSim."""
    import beat_arena_rs
    from beat_arena_env import ChartConfig, generate_chart
    chart_notes = generate_chart(ChartConfig(colors="both", dirs=tuple(range(8))),
                                 np.random.default_rng(21))
    chart = [(n.hit_time, n.col, n.row, n.color, n.angle, n.speed)
             for n in chart_notes]
    single = beat_arena_rs.RawBeatSim(max_steps=630)
    vec = beat_arena_rs.VecBeatSim(1, max_steps=630)
    o1 = single.set_chart(chart)
    vec.set_chart(0, chart)
    np.testing.assert_array_equal(vec.obs(), o1)
    rng = np.random.default_rng(4)
    done1 = False
    while not done1:
        a = rng.integers(0, 9, size=(1, 2))
        o1, r1, done1, _ = single.step(int(a[0, 0]), int(a[0, 1]))
        ov, rv, dv = vec.step(a)
        np.testing.assert_array_equal(ov, o1)
        assert rv[0] == np.float32(r1) and bool(dv[0]) == done1
    assert vec.counters(0) == (single.good_cuts, single.bad_cuts,
                               single.misses, single.n_notes)
