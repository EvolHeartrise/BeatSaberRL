"""fp16-storage Linear for the policy's update path (2026-07-25).

`fp16_linear(x_h, w_fp32, b_fp32)` runs the forward as a half addmm
(fp16 storage, cuBLAS fp32 compute — measured 1.9-2.6x over fp32 at the
policy's shapes) and the backward as:
  dx  = gy @ w_h            half matmul (2.2x)
  dW  = gy^T x              explicit cublasGemmEx, fp16 in / fp32 out +
                            COMPUTE_32F (csrc/fp16_gemm.cu — torch's own
                            half TN dispatch is 4.5x slower AND sloppily
                            accumulated on sm_61)
  db  = gy.sum(0) in fp32
The fp32 master weights receive fp32 grads directly — no separate cast
nodes, no fp16 optimizer state, checkpoints unchanged.

Falls back to stock F.linear when the extension is unavailable (the
fallback upcasts to fp32 and re-rounds, precision-compatible but slow —
it exists so CPU tests can exercise the code path).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_ext = None
_ext_failed = False


def _load():
    global _ext, _ext_failed
    if _ext is None and not _ext_failed:
        try:
            import os
            from beat_arena_env.cuda_env import _toolchain_env
            _toolchain_env()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")
            from torch.utils.cpp_extension import load
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "beat_arena_env", "csrc")
            bdir = os.path.join(src, "build_fp16")
            os.makedirs(bdir, exist_ok=True)
            _ext = load(name="fp16_gemm",
                        sources=[os.path.join(src, "fp16_gemm.cu")],
                        build_directory=bdir,
                        extra_cuda_cflags=["-arch=sm_61"],
                        extra_ldflags=["cublas.lib"])
        except Exception as e:
            _ext_failed = True
            print(f"[fp16-gemm] unavailable, torch fallback: {e}",
                  flush=True)
    return _ext


# Explicit weight-version epoch: the trainer MUST call
# bump_weights_version() after every opt.step(). Tensor _version is NOT
# usable for invalidation here — torch's FUSED Adam updates params
# without bumping it (verified 2026-07-25: fused 0->0, foreach 0->1),
# which silently froze the cached replicas on first try.
_weights_version = 1


def bump_weights_version():
    global _weights_version
    _weights_version += 1


def _half_params(w, b):
    """Epoch-checked fp16 replica cache. The replicas are detached
    constants (the custom Function computes dw itself), so caching is
    valid in grad mode too. The win is the ROLLOUT: 128 forwards per
    update on frozen weights pay ONE cast set instead of 128 (34
    launches each — measured as --fp16 all regressing below
    update-only without the cache)."""
    cache = getattr(w, "_fp16_cache", None)
    if cache is None or cache[0] != _weights_version:
        cache = (_weights_version, w.detach().half(), b.detach().half())
        w._fp16_cache = cache
    return cache[1], cache[2]


class _Fp16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        # x: fp16 (..., K); w: fp32 master (M, K); b: fp32 master (M)
        wh, bh = _half_params(w, b)
        x2 = x.reshape(-1, x.shape[-1])
        y = torch.addmm(bh, x2, wh.t())
        ctx.save_for_backward(x2, wh)
        return y.view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, gy):
        x2, wh = ctx.saved_tensors
        g2 = gy.reshape(-1, gy.shape[-1])
        if not g2.is_contiguous():
            g2 = g2.contiguous()
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = (g2 @ wh).view(*gy.shape[:-1], wh.shape[1])
        if ctx.needs_input_grad[1]:
            ext = _load()
            if ext is not None:
                dw = ext.wgrad(g2, x2)                    # fp32 out
            else:
                dw = g2.float().t() @ x2.float()
        if ctx.needs_input_grad[2]:
            db = g2.sum(0, dtype=torch.float32)
        return dx, dw, db


def fp16_linear(x, w, b):
    """Linear on an fp16 stream against fp32 master weights."""
    if not torch.is_grad_enabled():
        # rollout fast path: same ops as _Fp16Linear.forward, minus the
        # autograd.Function.apply dispatch — measured 2026-07-25 as the
        # bulk of the --fp16 all rollout regression (49 of 65 ms/update
        # at N=1024 was Function dispatch across the 13 Linear + 4 LN
        # sites, not the input casts the ledger blamed)
        wh, bh = _half_params(w, b)
        x2 = x.reshape(-1, x.shape[-1])
        return torch.addmm(bh, x2, wh.t()).view(*x.shape[:-1], w.shape[0])
    return _Fp16Linear.apply(x, w, b)
