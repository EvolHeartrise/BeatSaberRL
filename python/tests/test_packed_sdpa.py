"""packed_sdpa (2026-09-26): fused self-attention straight off the
packed q|k|v projection, whose backward asks the mem-efficient kernel
for ONE dq|dk|dv chunk (_efficient_attention_backward, shared_storage_
dqdkdv=True) and hands it back as qkv's gradient instead of autograd's
split-backward cat copy.

Pins: forward output and qkv grad are BITWISE the F.sdpa path's (same
aten kernels, same inputs), and the packed grad is zero-copy — it IS
the backward's chunk (no cat kernel runs)."""

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("fused attention paths need CUDA", allow_module_level=True)

import fp16_linear as fl

B, T, D, H = 512, 29, 192, 4
DH = D // H


def _mask(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    keep = torch.rand(B, T - 5, device="cuda", generator=g) < 0.6
    m = F.pad((keep.float().unsqueeze(1) - 1.0) * 16384.0, (5, 0))
    return m.half().unsqueeze(1)                        # (B, 1, 1, T)


def _qkv(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(B, T, 3 * D, device="cuda", generator=g).half()


def _ref(qkv, mask):
    q, k, v = (t.view(B, T, H, DH).transpose(1, 2)
               for t in qkv.split(D, dim=-1))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@pytest.mark.parametrize("seed", [1, 2])
def test_bitwise_vs_sdpa(seed):
    mask = _mask(seed)
    src = _qkv(seed)
    a = src.clone().requires_grad_(True)
    b = src.clone().requires_grad_(True)
    out_ref = _ref(a, mask)
    out = fl.packed_sdpa(b, mask, H)
    assert torch.equal(out, out_ref)
    go = torch.randn(out.shape, device="cuda").half()
    out_ref.backward(go)
    out.backward(go)
    assert torch.equal(a.grad, b.grad)
    with torch.no_grad():
        assert torch.equal(fl.packed_sdpa(b, mask, H), out_ref)


def test_packed_grad_is_the_chunk():
    """Zero-copy: qkv's gradient is the backward's own chunk (the
    _as_packed data_ptr/stride check passes) and no cat kernel runs."""
    qkv = _qkv(3).requires_grad_(True)
    mask = _mask()
    o = fl.packed_sdpa(qkv, mask, H)
    seen = {}
    orig = fl._as_packed

    def spy(gs, src):
        r = orig(gs, src)
        seen["packed"] = r
        seen["gq_ptr"] = gs[0].data_ptr()
        return r

    fl._as_packed = spy
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            o.backward(torch.ones_like(o))
            torch.cuda.synchronize()
    finally:
        fl._as_packed = orig
    assert seen["packed"] is not None
    assert seen["packed"].data_ptr() == seen["gq_ptr"]
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    assert not any("cat" in n.lower() for n in names), names
