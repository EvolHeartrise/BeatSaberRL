"""Parity tests: FusedNotePolicy (single-launch CUDA forward, env v10:
obs 340 / 19 tokens / 29xK heads) vs the PyTorch NotePolicy, on random
weights across the K = 3/5/7 gear tables, over random obs and real
env-rollout obs.

Tolerance: the fused kernel reassociates every dot product (different
summation order than cuBLAS), so bitwise equality is impossible in
fp32. Measured max abs diff is ~3e-7 (random init). The asserted atol
of 5e-5 gives large headroom over that while staying far below any real
defect: a wrong gather index, mask, or LN detail moves logits by
O(0.1+).

Checkpoint-weight parity re-arms at P7 (no v10 training checkpoints
exist yet; every earlier checkpoint was retired with the v10 body and
load_policy refuses them).

Skipped without CUDA and without a CUDA toolkit (nvcc) — the kernel is
JIT-built via cpp_extension like beat_arena_env/cuda_env.py.
"""

import shutil

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("FusedNotePolicy needs CUDA", allow_module_level=True)

from beat_arena_env.cuda_env import _toolchain_env

_toolchain_env()
if shutil.which("nvcc") is None:
    pytest.skip("no CUDA toolkit (nvcc) on PATH", allow_module_level=True)

from beat_arena_env import RustBeatEnv
from policy import (
    N_ACTION_HEADS, N_NOTE_SLOTS, NOTE_FEAT, NotePolicy, OBS_DIM,
    OFF_NOTES, OFF_NOTE_MASK,
)
from policy_cuda import FusedNotePolicy

ATOL = 5e-5
DEV = "cuda"


def _rand_obs(n, seed):
    """Random obs with realistic scale and a random 0/1 note mask."""
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, OBS_DIM, generator=g) * 0.7
    obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS] = (
        torch.rand(n, N_NOTE_SLOTS, generator=g) < 0.6).float()
    return obs.to(DEV)


def _real_obs(n, seed=1):
    """Real observations from env rollout steps (has empty note slots)."""
    env = RustBeatEnv(seed=seed, chart_len_s=8.0)
    obs, _ = env.reset()
    rows = [obs]
    rng = np.random.default_rng(seed)
    while len(rows) < n:
        obs, _, done, _, _ = env.step(rng.integers(0, [3] * N_ACTION_HEADS))
        if done:
            obs, _ = env.reset()
        rows.append(obs)
    return torch.as_tensor(np.stack(rows), dtype=torch.float32, device=DEV)


def _ref(policy, obs):
    with torch.no_grad():
        logits, value = policy(obs)
    return torch.stack(logits, dim=1), value


def _check(policy, fused, obs):
    ref_l, ref_v = _ref(policy, obs)
    lg, v = fused.fused_forward(obs)
    torch.testing.assert_close(lg, ref_l, atol=ATOL, rtol=0)
    torch.testing.assert_close(v, ref_v, atol=ATOL, rtol=0)
    return lg


@pytest.fixture(scope="module")
def rand_policy():
    torch.manual_seed(0)
    return NotePolicy(n_gears=5).to(DEV).eval()


@pytest.mark.parametrize("n", [1, 7, 64, 512])
def test_parity_random_weights(rand_policy, n):
    fused = FusedNotePolicy(rand_policy)
    _check(rand_policy, fused, _rand_obs(n, 100 + n))


@pytest.mark.parametrize("n_gears", [3, 5, 7])
def test_parity_gears_matrix(n_gears):
    """Every gear-table head width through the same kernel."""
    torch.manual_seed(10 + n_gears)
    pol = NotePolicy(n_gears=n_gears).to(DEV).eval()
    fused = FusedNotePolicy(pol)
    assert fused.n_gears == n_gears
    lg = _check(pol, fused, _rand_obs(64, n_gears))
    assert lg.shape == (64, N_ACTION_HEADS, n_gears)


def test_parity_real_obs(rand_policy):
    """Real rollout obs: logits/value close AND greedy actions agree.
    fp32 reassociation can flip a head only at a near-exact logit tie;
    any flip must be such a tie, and flips must be rare."""
    fused = FusedNotePolicy(rand_policy)
    obs = _real_obs(200)
    lg = _check(rand_policy, fused, obs).clone()
    act = fused.argmax(obs).clone()
    ref_act = _ref(rand_policy, obs)[0].argmax(-1)
    mism = act != ref_act
    n_mism = int(mism.sum())
    assert n_mism <= 0.005 * act.numel(), f"{n_mism} argmax flips"
    if n_mism:  # every flip must be a genuine near-tie
        top2 = lg[mism].topk(2, dim=-1).values
        gaps = (top2[:, 0] - top2[:, 1]).abs()
        assert float(gaps.max()) < 1e-3, "argmax flip beyond near-tie"


def test_junk_in_masked_slots_cannot_leak(rand_policy):
    """Mirror test_policy.py: garbage in invalid note slots must not
    change fused outputs. Masked keys score -16384, whose softmax weight
    underflows to exactly 0.0, so outputs are bitwise identical for any
    finite junk — asserted at atol 1e-6."""
    fused = FusedNotePolicy(rand_policy)
    obs = _real_obs(32)
    mask = obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS]
    assert (mask == 0).any(), "test needs at least one empty slot"
    junk = obs.clone()
    g = torch.Generator().manual_seed(42)
    for b in range(obs.shape[0]):
        for slot in torch.nonzero(mask[b] == 0).flatten():
            s = OFF_NOTES + int(slot) * NOTE_FEAT
            junk[b, s:s + NOTE_FEAT] = (
                torch.randn(NOTE_FEAT, generator=g) * 5).to(DEV)
    l1, v1 = fused.fused_forward(obs)
    l1, v1 = l1.clone(), v1.clone()
    a1 = fused.argmax(obs).clone()
    l2, v2 = fused.fused_forward(junk)
    torch.testing.assert_close(l1, l2, atol=1e-6, rtol=0)
    torch.testing.assert_close(v1, v2, atol=1e-6, rtol=0)
    assert torch.equal(a1, fused.argmax(junk))


def test_repack_tracks_new_weights(rand_policy):
    fused = FusedNotePolicy(rand_policy)
    obs = _rand_obs(16, 3)
    torch.manual_seed(123)
    other = NotePolicy(n_gears=5).to(DEV).eval()
    fused.repack(other)
    _check(other, fused, obs)
    fused.repack(rand_policy)  # and back
    _check(rand_policy, fused, obs)


def test_argmax_matches_logits(rand_policy):
    fused = FusedNotePolicy(rand_policy)
    obs = _rand_obs(64, 9)
    lg, _ = fused.fused_forward(obs)
    assert torch.equal(fused.argmax(obs), lg.argmax(-1))


def test_n_blocks_1_supported():
    torch.manual_seed(7)
    pol = NotePolicy(n_blocks=1, n_gears=3).to(DEV).eval()
    fused = FusedNotePolicy(pol)
    _check(pol, fused, _rand_obs(64, 77))


def test_unsupported_arch_raises():
    with pytest.raises(NotImplementedError):
        FusedNotePolicy(NotePolicy(embed=32))
    with pytest.raises(NotImplementedError):
        FusedNotePolicy(NotePolicy(n_heads=2))  # out_proj present
    # a pre-v10 (no body_mlp) state_dict is refused outright
    sd = NotePolicy(n_gears=3).state_dict()
    sd = {k: v for k, v in sd.items() if not k.startswith("body_mlp")}
    with pytest.raises(NotImplementedError):
        FusedNotePolicy(sd)
