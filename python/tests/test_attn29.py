"""T = 29 fused attention kernel (csrc/small_attn.cu, fp16_linear.attn29_*,
2026-09-27): one CTA per sample, 4 heads x 48, fp32 softmax / P / dS.

Pins, for all three production sites (packed self-attention TQ = 29, the
final block's TQ = 5 lead queries, the critic's TQ = 1 pool): output and
every gradient match an fp32 reference at least as tightly as torch's
mem-efficient SDPA does (and match SDPA itself to fp16 last-rounding
class); the grad-free call is bitwise the grad call's forward; a sample's
result does not depend on the batch it rides in (the rollout at N = 32768
and the update at 16384 must agree bitwise — the PPO ratio's r1)."""

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("fused attention needs CUDA", allow_module_level=True)

import fp16_linear as fl

if not fl.attn29_ok(29, 192, 4):
    pytest.skip("fp16_gemm extension lacks attn29", allow_module_level=True)

B, T, D, H = 1024, 29, 192, 4
DH = D // H


def _bias(seed, b=B):
    g = torch.Generator(device="cuda").manual_seed(seed)
    keep = torch.rand(b, T - 5, device="cuda", generator=g) < 0.6
    m = F.pad((keep.float().unsqueeze(1) - 1.0) * 16384.0, (5, 0))
    return m.half()                                      # (B, 1, T)


def _rand(shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=g).half()


def _heads(x, rows):
    return x.reshape(x.shape[0], rows, H, DH).transpose(1, 2)


def _merge(o):
    b, _, tq, _ = o.shape
    return o.transpose(1, 2).reshape(b, tq, D)


def _ref(q, k, v, bias):
    """fp32 reference MHA; q (B,TQ,D), k/v (B,T,D) -> (B,TQ,D)."""
    tq = q.shape[1]
    qh, kh, vh = _heads(q.float(), tq), _heads(k.float(), T), _heads(v.float(), T)
    s = qh @ kh.transpose(-2, -1) / DH ** 0.5 + bias.float().unsqueeze(1)
    return _merge(torch.softmax(s, dim=-1) @ vh)


def _sdpa(q, k, v, bias):
    tq = q.shape[1]
    return _merge(F.scaled_dot_product_attention(
        _heads(q, tq), _heads(k, T), _heads(v, T),
        attn_mask=bias.unsqueeze(1)))


def _err(a, b):
    return (a.float() - b.float()).abs().max().item()


def _case(tq, seed):
    """(fn under test, its leaf inputs, q/k/v getter) for one site."""
    bias = _bias(seed)
    if tq == T:
        qkv = _rand((B, T, 3 * D), seed).requires_grad_(True)
        fn = lambda: fl.attn29_packed(qkv, bias)
        qkv_of = lambda: qkv.split(D, dim=-1)
        return fn, [qkv], qkv_of, bias
    q = _rand((B, tq, D), seed).requires_grad_(True)
    kv = _rand((B, T, 2 * D), seed + 1).requires_grad_(True)
    fn = lambda: fl.attn29_qkv(q, kv, bias)
    qkv_of = lambda: (q, *kv.split(D, dim=-1))
    return fn, [q, kv], qkv_of, bias


@pytest.mark.parametrize("tq", [29, 5, 1])
def test_matches_fp32_reference_and_sdpa(tq):
    fn, leaves, qkv_of, bias = _case(tq, seed=tq)
    out = fn()
    assert out.shape == (B, tq, D) and out.is_contiguous()
    go = _rand(out.shape, 99) * 0.1
    out.backward(go)
    mine = [x.grad.clone() for x in leaves]
    # fp32 reference
    ref_leaves = [x.detach().float().requires_grad_(True) for x in leaves]
    if tq == T:
        rq, rk, rv = ref_leaves[0].split(D, dim=-1)
    else:
        rq = ref_leaves[0]
        rk, rv = ref_leaves[1].split(D, dim=-1)
    ref = _ref(rq, rk, rv, bias)
    ref.backward(go.float())
    # torch SDPA (mem-efficient cutlass, the path this replaces)
    sd_leaves = [x.detach().clone().requires_grad_(True) for x in leaves]
    if tq == T:
        sq, sk, sv = sd_leaves[0].split(D, dim=-1)
    else:
        sq = sd_leaves[0]
        sk, sv = sd_leaves[1].split(D, dim=-1)
    sd = _sdpa(sq, sk, sv, bias)
    sd.backward(go)
    e_mine, e_sdpa = _err(out, ref), _err(sd, ref)
    assert e_mine <= max(1.05 * e_sdpa, 2e-3), (e_mine, e_sdpa)
    assert _err(out, sd) < 4e-3
    for m, r, s in zip(mine, ref_leaves, sd_leaves):
        em, es = _err(m, r.grad), _err(s.grad, r.grad)
        assert em <= max(1.05 * es, 1e-3), (tq, em, es)
        assert _err(m, s.grad) < 8e-3


@pytest.mark.parametrize("tq", [29, 5, 1])
def test_grad_free_forward_is_bitwise(tq):
    fn, leaves, _, _ = _case(tq, seed=10 + tq)
    a = fn()
    with torch.no_grad():
        b = fn()
    assert torch.equal(a.detach(), b)


@pytest.mark.parametrize("tq", [29, 5, 1])
def test_batch_invariant(tq):
    """Rows computed in a batch of B equal the same rows computed in a
    batch of B/4 (rollout N vs update minibatch, r1)."""
    fn, leaves, _, bias = _case(tq, seed=20 + tq)
    with torch.no_grad():
        full = fn()
        sub = B // 4
        if tq == T:
            part = fl.attn29_packed(leaves[0][:sub].contiguous(), bias[:sub])
        else:
            part = fl.attn29_qkv(leaves[0][:sub].contiguous(),
                                 leaves[1][:sub].contiguous(), bias[:sub])
    assert torch.equal(full[:sub], part)


def test_masked_keys_get_no_weight():
    """A masked key's value never leaks into the output (bias -2^14)."""
    fn, leaves, _, bias = _case(T, seed=31)
    with torch.no_grad():
        a = fn()
        qkv = leaves[0].detach().clone()
        masked = bias[:, 0, :] < -1                      # (B, T)
        v = qkv[..., 2 * D:]
        v[masked] = 1000.0
        b = fl.attn29_packed(qkv, bias)
    assert torch.equal(a, b)


def test_packed_grad_layout():
    """The packed call's single (B, T, 3d) gradient is exactly dq|dk|dv in
    qkv's own column layout: bitwise the grads of the same kernel called
    on separate q and packed k|v views."""
    fn, leaves, qkv_of, bias = _case(T, seed=41)
    out = fn()
    go = _rand(out.shape, 5) * 0.1
    out.backward(go)
    g_packed = leaves[0].grad
    q, k, v = (t.detach() for t in qkv_of())
    q = q.contiguous().requires_grad_(True)
    kv = torch.cat([k, v], -1).requires_grad_(True)
    o2 = fl.attn29_qkv(q, kv, bias)
    assert torch.equal(o2, out.detach())
    o2.backward(go)
    assert torch.equal(g_packed[..., :D], q.grad)
    assert torch.equal(g_packed[..., D:], kv.grad)
