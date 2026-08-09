"""Parity and behavior tests for the fused CUDA backward — env v10
layout (policy_cuda.FusedPolicyTrainer / csrc/policy_backward.cu).

Tolerance rationale (v8-era measurements, re-validated on the v10
kernels in-session):

- Gradients SUM over mb rows (fp32 atomics in the fused path, cuBLAS
  reductions in torch), so error grows with row count and cancellation.
  Asserted: rel 2e-3 plus an absolute floor scaled to the largest grad
  component, because two tensors are MATHEMATICAL zeros that fp32 fills
  with cancellation dust (blocks.*.k_proj.bias — softmax row-grads sum
  to 0).
- Against a float64 torch reference, BOTH fp32 paths deviate by
  fp32-forward ReLU-boundary flips shared by any fp32 implementation —
  the f64 test asserts the fused error stays within a small factor of
  torch-fp32's own error.
- Reproducibility: striped atomic accumulation reassociates across
  blocks; with ncopies >= mb every sample owns a private partial buffer
  and the backward IS deterministic — used to assert bitwise properties
  (junk inertness, determinism) exactly.

Checkpoint-weight tests re-arm at P7 (no v10 checkpoints exist yet).
"""

import shutil

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("fused backward needs CUDA", allow_module_level=True)

from beat_arena_env.cuda_env import _toolchain_env

_toolchain_env()
if shutil.which("nvcc") is None:
    pytest.skip("no CUDA toolkit (nvcc) on PATH", allow_module_level=True)

from beat_arena_env import RustBeatEnv
from policy import (
    N_ACTION_HEADS, N_NOTE_SLOTS, NOTE_FEAT, NotePolicy, OBS_DIM,
    OFF_NOTES, OFF_NOTE_MASK,
)
from policy_cuda import FusedPolicyTrainer, _pack, unpack
from train_batch import fused_minibatch_losses, pg_loss_fn, VF_COEF

DEV = "cuda"
K = 5               # the production gear table
REL = 2e-3
ABS_OF_MAX = 1e-3   # absolute floor as a fraction of the largest |grad|


def _rand_obs(n, seed):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, OBS_DIM, generator=g) * 0.7
    obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS] = (
        torch.rand(n, N_NOTE_SLOTS, generator=g) < 0.6).float()
    return obs.to(DEV)


def _real_obs(n, seed=1):
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


def _linear_loss_grads_torch(policy, obs, R, Rv, dtype=torch.float32):
    """Torch-autograd grads of L = (logits*R).sum() + (value*Rv).sum().
    The loss is LINEAR in the outputs, so grad_logits == R exactly in
    every precision — isolating the backward implementations."""
    pol = NotePolicy(n_gears=policy.n_gears,
                     n_blocks=len(policy.blocks)).to(DEV).to(dtype)
    pol.load_state_dict(
        {k: v.to(dtype) for k, v in policy.state_dict().items()})
    logits, value = pol(obs.to(dtype))
    L = ((torch.stack(logits, 1) * R.to(dtype)).sum()
         + (value * Rv.to(dtype)).sum())
    L.backward()
    return {k: p.grad.detach() for k, p in pol.named_parameters()}


def _linear_loss_grads_fused(policy, obs, R, Rv, **kw):
    tr = FusedPolicyTrainer(policy, **kw)
    logits, value = tr(obs)
    ((logits * R).sum() + (value * Rv).sum()).backward()
    return tr.grads_by_param(), tr


def _assert_grad_close(fused, ref, rel=REL, abs_of_max=ABS_OF_MAX):
    gmax = max(g.abs().max().item() for g in ref.values())
    atol = abs_of_max * gmax
    for k in ref:
        diff = (fused[k].float() - ref[k].float()).abs().max().item()
        bound = atol + rel * ref[k].abs().max().item()
        assert diff <= bound, (
            f"{k}: |grad diff| {diff:.3e} > {bound:.3e} "
            f"(ref inf-norm {ref[k].abs().max().item():.3e})")


@pytest.fixture(scope="module")
def rand_policy():
    torch.manual_seed(0)
    return NotePolicy(n_gears=K).to(DEV)


def test_pack_unpack_roundtrip(rand_policy):
    sd = {k: v.detach().cpu() for k, v in rand_policy.state_dict().items()}
    rt = unpack(_pack(sd, 2, K), 2, K)
    assert set(rt) == set(sd)
    for k in sd:
        assert torch.equal(rt[k], sd[k]), k


@pytest.mark.parametrize("n", [3, 512])
def test_grad_parity_random_weights(rand_policy, n):
    obs = _rand_obs(n, 100 + n)
    g = torch.Generator().manual_seed(n)
    R = torch.randn(n, N_ACTION_HEADS, K, generator=g).to(DEV)
    Rv = torch.randn(n, generator=g).to(DEV)
    ref = _linear_loss_grads_torch(rand_policy, obs, R, Rv)
    fused, _ = _linear_loss_grads_fused(rand_policy, obs, R, Rv)
    _assert_grad_close(fused, ref)


def test_grad_parity_vs_f64(rand_policy):
    """Real rollout obs, mb 2048: fused-fp32 and torch-fp32 against the
    float64 torch reference. The fused error must stay within a small
    factor of torch's own fp32 error (they share the fp32-forward
    discretization; only the reduction strategy differs)."""
    n = 2048
    obs = _real_obs(n)
    g = torch.Generator().manual_seed(5)
    R = torch.randn(n, N_ACTION_HEADS, K, generator=g).to(DEV)
    Rv = torch.randn(n, generator=g).to(DEV)
    ref64 = _linear_loss_grads_torch(rand_policy, obs, R, Rv, torch.float64)
    ref32 = _linear_loss_grads_torch(rand_policy, obs, R, Rv)
    fused, _ = _linear_loss_grads_fused(rand_policy, obs, R, Rv)
    _assert_grad_close(fused, ref32)
    gmax = max(g.abs().max().item() for g in ref64.values())
    for k in ref64:
        e_t = (ref32[k].double() - ref64[k]).abs().max().item()
        e_f = (fused[k].double() - ref64[k]).abs().max().item()
        assert e_f <= 4.0 * e_t + 1e-4 * gmax, (
            f"{k}: fused err {e_f:.3e} vs torch-fp32 err {e_t:.3e}")


def test_grad_parity_ppo_loss(rand_policy):
    """Loss-level check: the ACTUAL train_batch minibatch losses (dual-
    clip PPO + value + entropy) through both update paths on the same
    synthetic minibatch."""
    n = 1024
    obs = _real_obs(n, seed=3)
    g = torch.Generator().manual_seed(11)
    act = torch.randint(0, K, (n, N_ACTION_HEADS), generator=g).to(DEV)
    logp = (torch.randn(n, generator=g) * 0.3 - 13.0).to(DEV)
    adv = torch.randn(n, generator=g).to(DEV)
    ret = torch.randn(n, generator=g).to(DEV)

    # eager torch path (the train_batch eager body)
    _, newlogp, entropy, newval = rand_policy.get_action_and_value(obs, act)
    ratio = (newlogp - logp).exp()
    a_mb = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg_loss = pg_loss_fn(ratio, a_mb)
    v_loss = 0.5 * ((newval - ret) ** 2).mean()
    loss = pg_loss - 0.01 * entropy.mean() + VF_COEF * v_loss
    rand_policy.zero_grad()
    loss.backward()
    ref = {k: p.grad.detach().clone()
           for k, p in rand_policy.named_parameters()}

    tr = FusedPolicyTrainer(rand_policy)
    f_loss, _ = fused_minibatch_losses(tr, 0.01, obs, act, logp, adv, ret)
    f_loss.backward()
    _assert_grad_close(tr.grads_by_param(), ref)
    assert torch.isclose(f_loss.detach(), loss.detach(), atol=1e-5, rtol=1e-4)
    rand_policy.zero_grad()


def test_finite_difference_probe(rand_policy):
    """Independent-of-autograd check: directional central difference of
    the fused forward's loss along a random packed-space direction vs
    the fused backward's directional derivative."""
    n = 64
    obs = _rand_obs(n, 9)
    g = torch.Generator().manual_seed(21)
    R = torch.randn(n, N_ACTION_HEADS, K, generator=g).to(DEV) * 0.1
    Rv = torch.randn(n, generator=g).to(DEV) * 0.1
    tr = FusedPolicyTrainer(rand_policy)
    logits, value = tr(obs)
    ((logits * R).sum() + (value * Rv).sum()).backward()
    grad = tr.packed.grad.detach().clone()
    u = torch.randn(grad.shape, generator=g).to(DEV)
    u /= u.norm()
    analytic = float(u.double() @ grad.double())
    eps = 1e-3

    def loss_at(delta):
        with torch.no_grad():
            tr.packed.add_(delta)
        lg, v = tr(obs)
        L = float(((lg * R).sum() + (v * Rv).sum()).detach())
        with torch.no_grad():
            tr.packed.sub_(delta)
        return L

    fd = (loss_at(eps * u) - loss_at(-eps * u)) / (2 * eps)
    assert abs(fd - analytic) <= 2e-2 * max(abs(analytic), 1e-3), (
        f"FD {fd:.6f} vs analytic {analytic:.6f}")


def test_deterministic_and_junk_inert_with_private_copies(rand_policy):
    """With ncopies >= mb each sample owns a private partial buffer, so
    the backward is fully deterministic — asserting two bitwise
    properties exactly: (1) same input -> identical grads; (2) garbage
    in masked note slots cannot leak into ANY gradient (masked keys get
    softmax weight exactly 0.0 — every grad chain from a masked token is
    an exact-zero product)."""
    n = 32
    obs = _real_obs(n)
    mask = obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS]
    assert (mask == 0).any()
    g = torch.Generator().manual_seed(13)
    R = torch.randn(n, N_ACTION_HEADS, K, generator=g).to(DEV)
    Rv = torch.randn(n, generator=g).to(DEV)
    g1, tr = _linear_loss_grads_fused(rand_policy, obs, R, Rv, ncopies=n)
    packed1 = tr.packed.grad.detach().clone()
    g2, tr2 = _linear_loss_grads_fused(rand_policy, obs, R, Rv, ncopies=n)
    assert torch.equal(packed1, tr2.packed.grad), "nondeterministic grads"
    junk = obs.clone()
    for b in range(n):
        for slot in torch.nonzero(mask[b] == 0).flatten():
            s = OFF_NOTES + int(slot) * NOTE_FEAT
            junk[b, s:s + NOTE_FEAT] = (
                torch.randn(NOTE_FEAT, generator=g) * 5).to(DEV)
    g3, tr3 = _linear_loss_grads_fused(rand_policy, junk, R, Rv, ncopies=n)
    assert torch.equal(packed1, tr3.packed.grad), "junk leaked into grads"


def test_grad_accumulation(rand_policy):
    """Two backwards accumulate into .grad like any autograd leaf."""
    n = 128
    obs = _rand_obs(n, 4)
    g = torch.Generator().manual_seed(17)
    R = torch.randn(n, N_ACTION_HEADS, K, generator=g).to(DEV)
    Rv = torch.randn(n, generator=g).to(DEV)
    tr = FusedPolicyTrainer(rand_policy, ncopies=n)
    for _ in range(2):
        lg, v = tr(obs)
        ((lg * R).sum() + (v * Rv).sum()).backward()
    once = FusedPolicyTrainer(rand_policy, ncopies=n)
    lg, v = once(obs)
    ((lg * R).sum() + (v * Rv).sum()).backward()
    torch.testing.assert_close(tr.packed.grad, 2 * once.packed.grad,
                               atol=1e-5, rtol=1e-5)


def test_unpack_to_and_adam_step_tracks_eager(rand_policy):
    """One full minibatch update (losses+backward+clip+Adam) through
    both paths from identical weights: the unpacked fused weights must
    track the eager update. Adam's first-step normalization amplifies
    grad noise to ~sign(g)*lr wherever |g| ~ 0, so the bound is a
    multiple of lr, not of the grad tolerance."""
    import torch.nn as nn
    lr = 3e-4
    n = 512
    obs = _real_obs(n, seed=7)
    g = torch.Generator().manual_seed(23)
    act = torch.randint(0, K, (n, N_ACTION_HEADS), generator=g).to(DEV)
    logp = (torch.randn(n, generator=g) * 0.3 - 13.0).to(DEV)
    adv = torch.randn(n, generator=g).to(DEV)
    ret = torch.randn(n, generator=g).to(DEV)

    eager = NotePolicy(n_gears=K).to(DEV)
    eager.load_state_dict(rand_policy.state_dict())
    opt = torch.optim.Adam(eager.parameters(), lr=lr)
    _, newlogp, entropy, newval = eager.get_action_and_value(obs, act)
    ratio = (newlogp - logp).exp()
    a_mb = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg_loss = pg_loss_fn(ratio, a_mb)
    loss = pg_loss - 0.01 * entropy.mean() \
        + VF_COEF * 0.5 * ((newval - ret) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(eager.parameters(), 0.5)
    opt.step()

    tr = FusedPolicyTrainer(rand_policy)
    fopt = torch.optim.Adam(tr.parameters(), lr=lr)
    f_loss, _ = fused_minibatch_losses(tr, 0.01, obs, act, logp, adv, ret)
    fopt.zero_grad()
    f_loss.backward()
    nn.utils.clip_grad_norm_(tr.parameters(), 0.5)
    fopt.step()
    updated = NotePolicy(n_gears=K).to(DEV)
    updated.load_state_dict(rand_policy.state_dict())
    tr.unpack_to(updated)
    for (k, pe), (k2, pf) in zip(eager.named_parameters(),
                                 updated.named_parameters()):
        assert k == k2
        diff = (pe - pf).abs().max().item()
        assert diff <= 2.5 * lr, f"{k}: post-Adam diff {diff:.2e}"


def test_n_blocks_1_supported():
    torch.manual_seed(3)
    pol = NotePolicy(n_blocks=1, n_gears=3).to(DEV)
    n = 256
    obs = _rand_obs(n, 31)
    g = torch.Generator().manual_seed(31)
    R = torch.randn(n, N_ACTION_HEADS, 3, generator=g).to(DEV)
    Rv = torch.randn(n, generator=g).to(DEV)
    pol.zero_grad()
    logits, value = pol(obs)
    ((torch.stack(logits, 1) * R).sum() + (value * Rv).sum()).backward()
    ref = {k: p.grad.detach() for k, p in pol.named_parameters()}
    fused, _ = _linear_loss_grads_fused(pol, obs, R, Rv)
    _assert_grad_close(fused, ref)


def test_unsupported_arch_raises():
    with pytest.raises(NotImplementedError):
        FusedPolicyTrainer(NotePolicy(embed=32))
    with pytest.raises(NotImplementedError):
        FusedPolicyTrainer(NotePolicy(n_heads=2))
    with pytest.raises(NotImplementedError):
        FusedPolicyTrainer(NotePolicy(n_blocks=3))  # forward-only arch
