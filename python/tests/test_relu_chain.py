"""fp16 Linear-ReLU-...-Linear chains with the ReLU in cuBLASLt epilogues
(fp16_linear.fp16_relu_chain, 2026-09-27): forward RELU_AUX_BIAS (output
+ 1-bit mask), backward DRELU on the next Linear's dgrad (mask applied
as it stores; the ReLU layer's db from its own wgrad GEMM) — no
standalone relu_bwd pass.

Pins: forward and every grad match an fp32 reference (computed with the
chain's own saved ReLU mask, see test_lt_epilogue's note on mask flips)
and the old per-Linear fp16 path to fp16 last-rounding class; the
grad-free path is bitwise the grad path's forward (rollout == update
forward, the r1 contract); the backward runs no relu_bwd /
threshold_backward kernel; the whole policy's grads move by the same
class only."""

import pytest
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    pytest.skip("fp16 epilogue paths need CUDA", allow_module_level=True)

import fp16_linear as fl

if fl._load() is None or not hasattr(fl._load(), "linear_relu_aux"):
    pytest.skip("fp16_gemm extension lacks the relu-aux epilogues",
                allow_module_level=True)

# (name, rows, widths) — production chains at minibatch 16384 / 4096
CHAINS = [("ffn", 4096 * 29, (192, 768, 192)),
          ("note_mlp", 4096 * 24, (32, 192, 192)),
          ("leg_mlp_like", 8192, (64, 192, 192)),
          ("trunk", 4096, (384, 384, 384)),
          ("critic_mlp", 4096, (576, 384, 384, 101))]


def _mods(widths, seed):
    torch.manual_seed(seed)
    ms = [nn.Linear(a, b).cuda() for a, b in zip(widths[:-1], widths[1:])]
    with torch.no_grad():
        for m in ms:
            m.bias.normal_(0, 0.1)
    return ms


def _x(R, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(R, K, device="cuda", generator=g) * 0.7).half()


def _rel(a, b):
    return ((a.float() - b.float()).abs().max()
            / b.float().abs().max().clamp_min(1e-6)).item()


def _fro(a, b):
    """Relative Frobenius error: robust to the rare mask flips two ReLU
    conventions disagree on (a pre-activation within fp32 noise of 0 —
    y rounds to 0 while the epilogue's bit says > 0), which carry a full
    gradient element each and dominate any max-abs metric."""
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _masks(x, ms):
    """The chain's own saved ReLU bitmasks (unpacked), recomputed by the
    same forward GEMMs: the reference must use exactly these (see _fro)."""
    out = []
    h = x.reshape(-1, x.shape[-1])
    shifts = torch.arange(8, device="cuda", dtype=torch.uint8)
    for m in ms[:-1]:
        wh, bh = fl._half_params(m.weight, m.bias)
        y, mk = fl._load().linear_relu_aux(h, wh, bh)
        R, H = y.shape
        bits = ((mk.view(R, -1, 1) >> shifts) & 1).view(R, -1)[:, :H]
        out.append(bits.float())
        h = y
    return out


@pytest.mark.parametrize("name,R,widths", CHAINS, ids=[c[0] for c in CHAINS])
@pytest.mark.parametrize("tail_bias", [True, False])
def test_chain_vs_fp32_reference(name, R, widths, tail_bias):
    fl.bump_weights_version()
    ms = _mods(widths, seed=1)
    x = _x(R, widths[0], seed=2).requires_grad_(True)
    out = fl.fp16_relu_chain(x, ms, tail_bias=tail_bias)
    assert out is not None and out.grad_fn is not None
    masks = _masks(x.detach(), ms)
    xr = x.detach().float().requires_grad_(True)
    ref_ms = [nn.Linear(m.in_features, m.out_features).cuda() for m in ms]
    with torch.no_grad():
        for r, m in zip(ref_ms, ms):
            r.weight.copy_(m.weight.half().float())
            r.bias.copy_(m.bias.half().float())
    h = xr
    for i, m in enumerate(ref_ms):
        if i < len(ref_ms) - 1:
            h = m(h) * masks[i]
        else:
            h = h @ m.weight.t() + (m.bias if tail_bias else 0)
    assert _rel(out, h) < 3e-3
    gy = (torch.randn(out.shape, device="cuda") * 0.01).half()
    out.backward(gy)
    h.backward(gy.float())
    assert _rel(x.grad, xr.grad) < 5e-3
    for m, r in zip(ms, ref_ms):
        assert _rel(m.weight.grad, r.weight.grad) < 3e-3, name
        if m is ms[-1] and not tail_bias:
            assert m.bias.grad is None
        else:
            assert _rel(m.bias.grad, r.bias.grad) < 3e-3, name


@pytest.mark.parametrize("name,R,widths", CHAINS[:2], ids=["ffn", "note"])
def test_chain_vs_per_linear_path(name, R, widths):
    """Same class as the old _Fp16LinearReLU + fp16_linear path."""
    fl.bump_weights_version()
    ms = _mods(widths, seed=3)
    x = _x(R, widths[0], seed=4)
    xa = x.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)
    a = fl.fp16_relu_chain(xa, ms)
    gy = (torch.randn(a.shape, device="cuda") * 0.01).half()
    a.backward(gy)
    ga = [(m.weight.grad.clone(), m.bias.grad.clone()) for m in ms]
    for m in ms:
        m.weight.grad = m.bias.grad = None
    h = xb
    for m in ms[:-1]:
        h = fl.fp16_linear_relu(h, m.weight, m.bias)
    b = fl.fp16_linear(h, ms[-1].weight, ms[-1].bias)
    b.backward(gy)
    assert _rel(a, b) < 2e-3
    assert _fro(xa.grad, xb.grad) < 2e-3
    for (gw, gb), m in zip(ga, ms):
        assert _fro(gw, m.weight.grad) < 2e-3
        assert _fro(gb, m.bias.grad) < 2e-3


def test_odd_k_encoders_keep_the_per_linear_path():
    """saber (46) / torso (29) / leg (34) inputs: no fused RELU_AUX_BIAS
    kernel at odd K, so the chain declines (None) and the policy keeps
    the fused RELU_BIAS forward there."""
    for k in (29, 34, 46):
        ms = _mods((k, 192, 192), seed=9)
        assert fl.fp16_relu_chain(_x(1024, k, seed=1), ms) is None


def test_grad_free_forward_is_bitwise_grad_forward():
    """Rollout (no_grad) and update (grad) forwards are the same kernels:
    bitwise equal outputs (the r1 contract)."""
    fl.bump_weights_version()
    ms = _mods((192, 768, 192), seed=5)
    x = _x(8192 * 29, 192, seed=6)
    with torch.no_grad():
        a = fl.fp16_relu_chain(x, ms)
    b = fl.fp16_relu_chain(x.clone().requires_grad_(True), ms)
    assert torch.equal(a, b.detach())


def test_backward_has_no_relu_pass():
    from torch.profiler import ProfilerActivity, profile
    fl.bump_weights_version()
    ms = _mods((192, 768, 192), seed=7)
    x = _x(4096 * 29, 192, seed=8).requires_grad_(True)
    out = fl.fp16_relu_chain(x, ms)
    gy = torch.ones_like(out)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out.backward(gy)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events()
             if e.device_type.name == "CUDA"]
    assert not any("relu_bwd" in n or "threshold" in n for n in names), \
        names
    assert any("drelu" in n.lower() or "gemm" in n for n in names)


def test_policy_grads_vs_fp32_reference():
    """Whole policy, one PPO fwd+bwd: the chain path's logits, value and
    every param grad sit as close to the fp32 record path's as the
    per-Linear fp16 path's do (a grad-dump style gate, in-process)."""
    import bench_policy_update as bpu
    import policy as P
    from policy import NotePolicy
    torch.manual_seed(0)
    src = NotePolicy(embed=192, trunk_dim=384, n_blocks=6, n_heads=4,
                     n_gears=5, ffn_mult=4, note_geo=True)
    data = bpu.make_minibatch(2048, torch.device("cuda"), 5)

    def run(mode):
        pol = bpu.fresh_policy(src, torch.device("cuda"))
        if mode != "fp32":
            pol.fp16 = True
            pol.flatten_fp16()
        orig = P.fp16_relu_chain
        if mode == "old":
            P.fp16_relu_chain = lambda *a, **k: None
        try:
            logits, value = pol.forward_stacked(data[0])
            for p in pol.parameters():
                p.grad = None
            bpu.eager_update(pol, None, *data, step=False, scale=1024.0)
        finally:
            P.fp16_relu_chain = orig
        return logits.float(), value.float(), {
            n: p.grad.float() / 1024.0 for n, p in pol.named_parameters()}

    lr, vr, gr = run("fp32")
    la, va, ga = run("chain")
    lb, vb, gb = run("old")
    assert _fro(la, lr) <= 1.3 * _fro(lb, lr) + 1e-4
    assert _fro(va, vr) <= 1.3 * _fro(vb, vr) + 1e-4
    for n in gr:
        e_new, e_old = _fro(ga[n], gr[n]), _fro(gb[n], gr[n])
        assert e_new <= 1.3 * e_old + 2e-3, (n, e_new, e_old)
