"""Parity and behavior tests for the fused CUDA backward
(policy_cuda.FusedPolicyTrainer / FusedPolicyFn / csrc/policy_backward.cu).

Tolerance rationale (measured 2026-07-25 on the GTX 1080):

- Reference precision: gradients SUM over mb rows (via fp32 atomics in
  the fused path, cuBLAS reductions in torch), so unlike the forward's
  4.2e-7 single-sample parity, error grows with row count and with
  cancellation. Measured fused-vs-torch-fp32 (linear loss, so both see
  identical upstream grads): worst per-tensor rel error 2e-6 (random
  weights, mb 512) -> 2.6e-4 (trained checkpoint, mb 16k — trained
  weights produce correlated per-sample grads, i.e. larger partial sums
  for the same final magnitude). Asserted: rel 2e-3 (~8x headroom),
  plus an absolute floor scaled to the largest grad component because
  two tensors are MATHEMATICAL zeros that fp32 fills with cancellation
  dust: blocks.*.k_proj.bias (softmax row-grads sum to 0, so the key
  bias receives exactly zero gradient) — dust, not defect, and torch
  fp32 shows the same magnitude of dust.
- Against a float64 torch reference, BOTH fp32 paths deviate by up to
  ~3e-3 rel at mb 16k with nearly IDENTICAL per-tensor errors — that
  deviation is fp32-forward ReLU-boundary flips shared by any fp32
  implementation, not accumulation error, so the f64 test asserts the
  fused error stays within a small factor of torch-fp32's own error.
- Reproducibility: striped atomic accumulation reassociates across
  blocks, so grads are NOT bitwise reproducible run-to-run at default
  ncopies. With ncopies >= mb every sample owns a private partial
  buffer and the whole backward IS deterministic — used below to assert
  bitwise properties (junk inertness, determinism) exactly.
"""

import os
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

from beat_arena_env import RustBeatEnv, OFF_NOTES, OFF_NOTE_MASK, NOTE_FEAT
from policy import N_NOTE_SLOTS, NotePolicy, OBS_DIM, load_policy
from policy_cuda import FusedPolicyTrainer, _pack, unpack
from train_batch import fused_minibatch_losses, pg_loss_fn, VF_COEF

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "runs", "gen2_ep2_50m", "final.pt")
DEV = "cuda"
REL = 2e-3          # vs torch fp32 (measured worst 2.6e-4 at ckpt/16k)
ABS_OF_MAX = 1e-3   # absolute floor as a fraction of the largest |grad|


def _rand_obs(n, seed):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, OBS_DIM, generator=g) * 0.7
    obs[:, OFF_NOTE_MASK:] = (
        torch.rand(n, N_NOTE_SLOTS, generator=g) < 0.6).float()
    return obs.to(DEV)


def _real_obs(n, seed=1):
    env = RustBeatEnv(seed=seed, chart_len_s=8.0)
    obs, _ = env.reset()
    rows = [obs]
    rng = np.random.default_rng(seed)
    while len(rows) < n:
        obs, _, done, _, _ = env.step(rng.integers(0, [3] * 12))
        if done:
            obs, _ = env.reset()
        rows.append(obs)
    return torch.as_tensor(np.stack(rows), dtype=torch.float32, device=DEV)


def _linear_loss_grads_torch(policy, obs, R, Rv, dtype=torch.float32):
    """Torch-autograd grads of L = (logits*R).sum() + (value*Rv).sum().
    The loss is LINEAR in the outputs, so grad_logits == R exactly in
    every precision — isolating the backward implementations."""
    pol = NotePolicy().to(DEV).to(dtype)
    pol.load_state_dict(
        {k: v.to(dtype) for k, v in policy.state_dict().items()})
    logits, value = pol(obs.to(dtype))
    L = (torch.stack(logits, 1) * R.to(dtype)).sum() + (value * Rv.to(dtype)).sum()
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
    return NotePolicy().to(DEV)


@pytest.fixture(scope="module")
def ckpt_policy():
    if not os.path.exists(CKPT):
        pytest.skip(f"checkpoint missing: {CKPT}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    return load_policy(ckpt).to(DEV)


def test_pack_unpack_roundtrip(rand_policy):
    sd = {k: v.detach().cpu() for k, v in rand_policy.state_dict().items()}
    rt = unpack(_pack(sd, 2), 2)
    assert set(rt) == set(sd)
    for k in sd:
        assert torch.equal(rt[k], sd[k]), k


@pytest.mark.parametrize("n", [3, 512])
def test_grad_parity_random_weights(rand_policy, n):
    obs = _rand_obs(n, 100 + n)
    g = torch.Generator().manual_seed(n)
    R = torch.randn(n, 12, 3, generator=g).to(DEV)
    Rv = torch.randn(n, generator=g).to(DEV)
    ref = _linear_loss_grads_torch(rand_policy, obs, R, Rv)
    fused, _ = _linear_loss_grads_fused(rand_policy, obs, R, Rv)
    _assert_grad_close(fused, ref)


def test_grad_parity_checkpoint_vs_f64(ckpt_policy):
    """Trained weights, real rollout obs, mb 2048: fused-fp32 and
    torch-fp32 against the float64 torch reference. The fused error must
    stay within a small factor of torch's own fp32 error (they share the
    fp32-forward discretization; only the reduction strategy differs)."""
    n = 2048
    obs = _real_obs(n)
    g = torch.Generator().manual_seed(5)
    R = torch.randn(n, 12, 3, generator=g).to(DEV)
    Rv = torch.randn(n, generator=g).to(DEV)
    ref64 = _linear_loss_grads_torch(ckpt_policy, obs, R, Rv, torch.float64)
    ref32 = _linear_loss_grads_torch(ckpt_policy, obs, R, Rv)
    fused, _ = _linear_loss_grads_fused(ckpt_policy, obs, R, Rv)
    _assert_grad_close(fused, ref32)
    gmax = max(g.abs().max().item() for g in ref64.values())
    for k in ref64:
        e_t = (ref32[k].double() - ref64[k]).abs().max().item()
        e_f = (fused[k].double() - ref64[k]).abs().max().item()
        assert e_f <= 4.0 * e_t + 1e-4 * gmax, (
            f"{k}: fused err {e_f:.3e} vs torch-fp32 err {e_t:.3e}")


def test_grad_parity_ppo_loss(ckpt_policy):
    """Loss-level check: the ACTUAL train_batch minibatch losses (dual-
    clip PPO + value + entropy) through both update paths on the same
    synthetic minibatch — upstream grads now come from each path's own
    fp32 loss graph, exactly as in training."""
    n = 1024
    obs = _real_obs(n, seed=3)
    g = torch.Generator().manual_seed(11)
    act = torch.randint(0, 3, (n, 12), generator=g).to(DEV)
    logp = (torch.randn(n, generator=g) * 0.3 - 13.0).to(DEV)
    adv = torch.randn(n, generator=g).to(DEV)
    ret = torch.randn(n, generator=g).to(DEV)

    # eager torch path (the train_batch eager body)
    _, newlogp, entropy, newval = ckpt_policy.get_action_and_value(obs, act)
    ratio = (newlogp - logp).exp()
    a_mb = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg_loss, _ = pg_loss_fn(ratio, a_mb)
    v_loss = 0.5 * ((newval - ret) ** 2).mean()
    loss = pg_loss - 0.01 * entropy.mean() + VF_COEF * v_loss
    ckpt_policy.zero_grad()
    loss.backward()
    ref = {k: p.grad.detach().clone()
           for k, p in ckpt_policy.named_parameters()}

    tr = FusedPolicyTrainer(ckpt_policy)
    f_loss, _ = fused_minibatch_losses(tr, 0.01, obs, act, logp, adv, ret)
    f_loss.backward()
    _assert_grad_close(tr.grads_by_param(), ref)
    assert torch.isclose(f_loss.detach(), loss.detach(), atol=1e-5, rtol=1e-4)
    ckpt_policy.zero_grad()


def test_finite_difference_probe(rand_policy):
    """Independent-of-autograd check: directional central difference of
    the fused forward's loss along a random packed-space direction vs
    the fused backward's directional derivative. fp32 FD resolution
    bounds the achievable agreement (~1e-3); asserted at 2e-2."""
    n = 64
    obs = _rand_obs(n, 9)
    g = torch.Generator().manual_seed(21)
    R = torch.randn(n, 12, 3, generator=g).to(DEV) * 0.1
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


def test_deterministic_and_junk_inert_with_private_copies(ckpt_policy):
    """With ncopies >= mb each sample owns a private partial buffer, so
    the backward is fully deterministic — which lets two bitwise
    properties be asserted exactly: (1) same input -> identical grads;
    (2) garbage in masked note slots cannot leak into ANY gradient
    (masked keys get softmax weight exactly 0.0, so every grad chain
    from a masked token is an exact-zero product, mirroring
    test_policy_cuda.test_junk_in_masked_slots_cannot_leak)."""
    n = 32
    obs = _real_obs(n)
    mask = obs[:, OFF_NOTE_MASK:]
    assert (mask == 0).any()
    g = torch.Generator().manual_seed(13)
    R = torch.randn(n, 12, 3, generator=g).to(DEV)
    Rv = torch.randn(n, generator=g).to(DEV)
    g1, tr = _linear_loss_grads_fused(ckpt_policy, obs, R, Rv, ncopies=n)
    packed1 = tr.packed.grad.detach().clone()
    g2, tr2 = _linear_loss_grads_fused(ckpt_policy, obs, R, Rv, ncopies=n)
    assert torch.equal(packed1, tr2.packed.grad), "nondeterministic grads"
    junk = obs.clone()
    for b in range(n):
        for slot in torch.nonzero(mask[b] == 0).flatten():
            s = OFF_NOTES + int(slot) * NOTE_FEAT
            junk[b, s:s + NOTE_FEAT] = (
                torch.randn(NOTE_FEAT, generator=g) * 5).to(DEV)
    g3, tr3 = _linear_loss_grads_fused(ckpt_policy, junk, R, Rv, ncopies=n)
    assert torch.equal(packed1, tr3.packed.grad), "junk leaked into grads"


def test_grad_accumulation(rand_policy):
    """Two backwards accumulate into .grad like any autograd leaf."""
    n = 128
    obs = _rand_obs(n, 4)
    g = torch.Generator().manual_seed(17)
    R = torch.randn(n, 12, 3, generator=g).to(DEV)
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


def test_unpack_to_and_adam_step_tracks_eager(ckpt_policy):
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
    act = torch.randint(0, 3, (n, 12), generator=g).to(DEV)
    logp = (torch.randn(n, generator=g) * 0.3 - 13.0).to(DEV)
    adv = torch.randn(n, generator=g).to(DEV)
    ret = torch.randn(n, generator=g).to(DEV)

    eager = NotePolicy().to(DEV)
    eager.load_state_dict(ckpt_policy.state_dict())
    opt = torch.optim.Adam(eager.parameters(), lr=lr)
    _, newlogp, entropy, newval = eager.get_action_and_value(obs, act)
    ratio = (newlogp - logp).exp()
    a_mb = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg_loss, _ = pg_loss_fn(ratio, a_mb)
    loss = pg_loss - 0.01 * entropy.mean() \
        + VF_COEF * 0.5 * ((newval - ret) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(eager.parameters(), 0.5)
    opt.step()

    tr = FusedPolicyTrainer(ckpt_policy)
    fopt = torch.optim.Adam(tr.parameters(), lr=lr)
    f_loss, _ = fused_minibatch_losses(tr, 0.01, obs, act, logp, adv, ret)
    fopt.zero_grad()
    f_loss.backward()
    nn.utils.clip_grad_norm_(tr.parameters(), 0.5)
    fopt.step()
    updated = NotePolicy().to(DEV)
    updated.load_state_dict(ckpt_policy.state_dict())
    tr.unpack_to(updated)
    for (k, pe), (k2, pf) in zip(eager.named_parameters(),
                                 updated.named_parameters()):
        assert k == k2
        diff = (pe - pf).abs().max().item()
        assert diff <= 2.5 * lr, f"{k}: post-Adam diff {diff:.2e}"


def test_n_blocks_1_supported(ckpt_policy):
    torch.manual_seed(3)
    pol = NotePolicy(n_blocks=1).to(DEV)
    n = 256
    obs = _rand_obs(n, 31)
    g = torch.Generator().manual_seed(31)
    R = torch.randn(n, 12, 3, generator=g).to(DEV)
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
