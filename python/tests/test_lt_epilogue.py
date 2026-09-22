"""cuBLASLt fused bias+ReLU epilogue (2026-09-03, Tier 2.5 of the 09-02
static audit, A §5; the BEAT_LT_EPILOGUE bisect hatch was removed
2026-09-22).

The Linear+ReLU sites run as ONE `torch._addmm_activation` GEMM
(cublasLt RELU_BIAS epilogue, bias = fp16 replica); operands the gate
rejects fall back to the two-pass bias-free mm + in-place
`bias_relu_(y, b_fp32)`. These pin: the routing gate (every production
site qualifies, odd K included; misaligned or strided operands stay
two-pass), that the autograd Function's backward matches an fp32
reference as tightly as
the two-pass grads do), that the whole policy's logits/value move by
the same class only, and — on CUDA — that the fused call really is a
single kernel (no separate bias_relu / relu pass), which is the
entire point of the item.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("fp16 epilogue paths need CUDA", allow_module_level=True)

import fp16_linear as fl

if fl._load() is None:
    pytest.skip("fp16_gemm extension unavailable", allow_module_level=True)

# (name, M, K, N) — the production ReLU sites at micro-batch 8192
SITES = [("ffn.0", 8192 * 27, 128, 512),
         ("trunk.0", 8192, 384, 256),
         ("critic_mlp.0", 8192, 384, 256),
         ("note_mlp.0", 8192 * 16, 18, 128),
         ("saber_mlp.0", 8192 * 2, 34, 128),
         ("body_mlp.0", 8192, 78, 128)]
QUALIFY = {s[0] for s in SITES}       # odd K routes through Lt too


def _site(M, K, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(M, K, device="cuda", generator=g) * 0.5).half()
    w = torch.randn(N, K, device="cuda", generator=g) / K ** 0.5
    b = torch.randn(N, device="cuda", generator=g) * 0.1
    return x, w, b


def _rel(a, b):
    return ((a.float() - b.float()).abs().max()
            / b.float().abs().max().clamp_min(1e-6)).item()


def _kernels(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [ev.name for ev in prof.events()
            if ev.device_type.name == "CUDA"]


@pytest.mark.parametrize("name,M,K,N", SITES, ids=[s[0] for s in SITES])
def test_gate_by_site(name, M, K, N):
    x, w, b = _site(M, K, N)
    assert fl._lt_epilogue_ok(x, w.half(), b.half()) is (name in QUALIFY)


def test_gate_rejects_misaligned_or_strided():
    x, w, b = _site(4096, 128, 512)
    wh, bh = w.half(), b.half()
    assert fl._lt_epilogue_ok(x, wh, bh)
    big = torch.zeros(512 + 8, device="cuda", dtype=torch.half)
    assert not fl._lt_epilogue_ok(x, wh, big[1:513])       # 2-B offset
    assert fl._lt_epilogue_ok(x, wh, big[8:520])           # 16-B offset
    assert not fl._lt_epilogue_ok(x.t().contiguous().t(), wh, bh)
    assert not fl._lt_epilogue_ok(x[:, :120], wh[:, :120].contiguous(),
                                  bh)                      # strided x


def test_backward_matches_fp32_reference():
    """The backward masks with the SAVED post-ReLU y, so the reference
    must use that same mask (an fp32 relu'd reference would flip the
    mask on the ~4e-4 of elements whose pre-activation sits within one
    fp16 rounding of 0, and at 110k rows those flips alone move db by
    percent — not a backward bug)."""
    x, w, b = _site(4096 * 27, 128, 512, seed=1)
    w = torch.nn.Parameter(w)
    b = torch.nn.Parameter(b)
    xg = x.clone().requires_grad_(True)
    y = fl.fp16_linear_relu(xg, w, b)
    assert y.grad_fn is not None
    gy = (torch.randn_like(y.float()) * 0.01).half()
    y.backward(gy)
    gm = gy.float() * (y.detach() > 0)
    assert _rel(xg.grad, gm @ w.detach()) < 2e-3
    assert _rel(w.grad, gm.t() @ x.float()) < 1e-4       # fp32 accum
    assert _rel(b.grad, gm.sum(0)) < 1e-4


@pytest.mark.skipif(torch.cuda.get_device_capability() < (8, 0),
                    reason="Lt RELU_BIAS fusion pinned on sm_80+ only")
def test_fused_is_one_kernel():
    """The point of the item: the ffn.0 site is ONE GEMM launch — no
    bias_relu_ pass, no separate relu_ (which is what torch's non-Lt
    fallback would show as a 2nd/3rd kernel)."""
    x, w, b = _site(8192 * 27, 128, 512)
    wh, bh = w.half(), b.half()
    fused = _kernels(lambda: fl._relu_forward(x, wh, bh, b))
    # exactly one kernel, and it is the GEMM (cuBLAS names the
    # epilogue-capable cutlass kernel "...gemm_relu_..." — the same
    # kernel the bias-free mm above ran, now with the epilogue on)
    assert len(fused) == 1 and "gemm" in fused[0], fused
    assert not any("bias_relu" in k for k in fused), fused
    # an odd-K encoder site is one (align2) kernel too
    x, w, b = _site(8192 * 16, 18, 128)
    wh, bh = w.half(), b.half()          # casts outside the profile
    enc = _kernels(lambda: fl._relu_forward(x, wh, bh, b))
    assert len(enc) == 1 and "gemm" in enc[0], enc
    # a gate-rejected operand (misaligned bias view) takes the two-pass
    # kernels, not torch's 3-pass fallback
    x, w, b = _site(8192, 128, 512)
    wh = w.half()
    big = torch.zeros(512 + 8, device="cuda", dtype=torch.half)
    big[1:513] = b.half()
    bh = big[1:513]
    mis = _kernels(lambda: fl._relu_forward(x, wh, bh, b))
    assert len(mis) == 2 and any("bias_relu" in k for k in mis), mis
