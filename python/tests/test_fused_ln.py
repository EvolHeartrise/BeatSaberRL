"""Fused LayerNorm kernel tests (csrc/layer_norm.cu via fused_ln.py).

Pins the fp32 kernels (previously covered only implicitly through the
policy parity suite) and the fp16-I/O variants added for the
fp16-storage project (fp32 stats/accumulate, one rounding on store).
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("fused LN kernels need CUDA", allow_module_level=True)

from fused_ln import FusedLayerNorm, _load

if _load() is None:
    pytest.skip("fused LN extension unavailable", allow_module_level=True)

SHAPES = [(1, 64), (33, 64), (4096, 64), (33, 128), (4096, 128),
          (589824, 64)]


def _mk(rows, d, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, d, device="cuda", generator=g, dtype=torch.float32)
    return x.to(dtype)


@pytest.mark.parametrize("rows,d", SHAPES)
def test_fp32_matches_torch(rows, d):
    ln = FusedLayerNorm(d).cuda()
    torch.nn.init.normal_(ln.weight)
    torch.nn.init.normal_(ln.bias)
    x = _mk(rows, d, torch.float32)
    ref = torch.nn.functional.layer_norm(
        x, (d,), ln.weight, ln.bias, ln.eps)
    y = ln(x)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("rows,d", [(33, 64), (4096, 64), (4096, 128)])
def test_fp16_fwd_vs_fp64(rows, d):
    ln = FusedLayerNorm(d).cuda()
    torch.nn.init.normal_(ln.weight)
    torch.nn.init.normal_(ln.bias)
    xh = _mk(rows, d, torch.float16)
    y = ln(xh)
    assert y.dtype is torch.float16
    x64 = xh.double()
    mu = x64.mean(-1, keepdim=True)
    var = x64.var(-1, unbiased=False, keepdim=True)
    ref = ((x64 - mu) / (var + ln.eps).sqrt()) * ln.weight.double() \
        + ln.bias.double()
    err = (y.double() - ref).abs().max() / ref.abs().max()
    assert err < 2e-3, f"fp16 fwd rel err {err:.2e}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fwd_add_x_is_rounded_sum(dtype):
    ln = FusedLayerNorm(64).cuda()
    a = _mk(4096, 64, dtype, seed=1)
    b = _mk(4096, 64, dtype, seed=2)
    x, y = ln.forward_add(a, b)
    if dtype is torch.float32:
        assert torch.equal(x, a + b)
    else:
        assert torch.equal(x, (a.float() + b.float()).half())
    assert y.dtype is dtype


def test_fwd_add_b_bias_lands_in_the_fp32_sum():
    """The folded producer-GEMM bias (policy ffn.2 -> next block's
    ln_attn) must be added INSIDE the pass's fp32 arithmetic, i.e.
    x == round_h(a + (b + bias)) — and the fp32 record path must refuse
    it outright (it would move a rounding position there)."""
    ln = FusedLayerNorm(64).cuda()
    a = _mk(4096, 64, torch.float16, seed=1)
    b = _mk(4096, 64, torch.float16, seed=2)
    bias = torch.randn(64, device="cuda")
    x, y = ln.forward_add(a, b, bias)
    assert torch.equal(x, (a.float() + (b.float() + bias)).half())
    assert y.dtype is torch.float16
    # and without a bias the kernel is unchanged
    x0, _ = ln.forward_add(a, b)
    assert torch.equal(x0, (a.float() + b.float()).half())
    with pytest.raises(RuntimeError):
        ln.forward_add(a.float(), b.float(), bias)


def test_fwd_add_b_bias_grad_is_the_channel_sum():
    """d/d(b_bias) is the channel sum of the gradient reaching x — the
    same reduction the producer Linear's own bias grad used to run."""
    ln = FusedLayerNorm(64).cuda()
    torch.nn.init.normal_(ln.weight)
    a = _mk(1024, 64, torch.float16, seed=1).requires_grad_(True)
    b = _mk(1024, 64, torch.float16, seed=2).requires_grad_(True)
    bias = torch.randn(64, device="cuda", requires_grad=True)
    _, y = ln.forward_add(a, b, bias)
    y.backward(_mk(1024, 64, torch.float16, seed=3))
    assert bias.grad.dtype is torch.float32
    assert torch.equal(a.grad, b.grad)
    assert torch.equal(bias.grad, a.grad.sum(0, dtype=torch.float32))


@pytest.mark.parametrize("d", [64, 128])
def test_fp16_bwd_vs_fp64(d):
    rows = 4096
    ln = FusedLayerNorm(d).cuda()
    torch.nn.init.normal_(ln.weight)
    torch.nn.init.normal_(ln.bias)
    xh = _mk(rows, d, torch.float16).requires_grad_(True)
    gh = _mk(rows, d, torch.float16, seed=3)
    y = ln(xh)
    y.backward(gh)
    # fp64 reference on the SAME fp16-rounded inputs
    x64 = xh.detach().double().requires_grad_(True)
    w64 = ln.weight.detach().double().requires_grad_(True)
    b64 = ln.bias.detach().double().requires_grad_(True)
    ref = torch.nn.functional.layer_norm(x64, (d,), w64, b64, ln.eps)
    ref.backward(gh.double())
    dx_err = ((xh.grad.double() - x64.grad).abs().max()
              / x64.grad.abs().max())
    dw_err = ((ln.weight.grad.double() - w64.grad).abs().max()
              / w64.grad.abs().max())
    db_err = ((ln.bias.grad.double() - b64.grad).abs().max()
              / b64.grad.abs().max())
    assert dx_err < 5e-3, f"dx rel err {dx_err:.2e}"
    # dw/db are fp32-accumulated -> much tighter than the fp16 dx
    assert dw_err < 1e-3, f"dw rel err {dw_err:.2e}"
    assert db_err < 1e-3, f"db rel err {db_err:.2e}"
    assert ln.weight.grad.dtype is torch.float32
    assert xh.grad.dtype is torch.float16


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_bwd_deterministic(dtype):
    ln = FusedLayerNorm(64).cuda()

    def run():
        ln.zero_grad()
        x = _mk(589824, 64, dtype, seed=5).requires_grad_(True)
        y = ln(x)
        y.backward(_mk(589824, 64, dtype, seed=6))
        return x.grad.clone(), ln.weight.grad.clone(), ln.bias.grad.clone()

    a1, w1, b1 = run()
    a2, w2, b2 = run()
    assert torch.equal(a1, a2) and torch.equal(w1, w2) \
        and torch.equal(b1, b2)
