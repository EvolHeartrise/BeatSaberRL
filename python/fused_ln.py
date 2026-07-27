"""Drop-in fused LayerNorm (csrc/layer_norm.cu) for the policy's
d=64/128 rows — the 2026-07-25 survey measured torch's LayerNorm at
37 GB/s vs 218 achievable at this shape (+1.13x on the whole update
when replaced).

`FusedLayerNorm` subclasses nn.LayerNorm, so state_dict layout, param
names, and checkpoints are byte-identical; it routes contiguous CUDA
fp32 inputs through the custom kernel and falls back to stock
nn.LayerNorm everywhere else (CPU eval tools, exotic dtypes, machines
without nvcc — the build is lazy and a build failure permanently
falls back with one warning). Backward is deterministic (fixed-size
partials + torch reduction, no cross-block float atomics).
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ext = None
_ext_failed = False


def _load():
    """Lazy compile-or-load; returns None (forever) if the toolchain or
    build is unavailable so callers fall back to stock LayerNorm."""
    global _ext, _ext_failed
    if _ext is None and not _ext_failed:
        try:
            import os
            from beat_arena_env.jit_cache import fast_load
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "beat_arena_env", "csrc")
            bdir = os.path.join(src, "build_ln")
            os.makedirs(bdir, exist_ok=True)
            srcs = [os.path.join(src, "layer_norm.cu")]

            def _full():
                from beat_arena_env.cuda_env import _toolchain_env
                _toolchain_env()
                os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")
                from torch.utils.cpp_extension import load
                return load(name="layer_norm_cuda", sources=srcs,
                            build_directory=bdir,
                            extra_cuda_cflags=["-arch=sm_61"])

            _ext = fast_load("layer_norm_cuda", srcs, bdir,
                             ["-arch=sm_61"], _full)
        except Exception as e:  # no nvcc / cl / CUDA: stock LN forever
            _ext_failed = True
            print(f"[fused-ln] unavailable, using torch LayerNorm: {e}",
                  flush=True)
    return _ext


class _LnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        y, mean, rstd = _ext.ln_fwd(x, weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx.saved_tensors
        dx, dw, db = _ext.ln_bwd(x, dy.contiguous(), weight, mean, rstd)
        return dx, dw, db, None


class _LnAddFn(torch.autograd.Function):
    """x = a + b fused into the LN pass; returns (x, ln(x)). Backward:
    d/da == d/db == (grad flowing to x) + (LN's dx from grad on y) — the
    two-term sum is commutative bitwise, so this matches the unfused
    autograd accumulation exactly."""
    @staticmethod
    def forward(ctx, a, b, weight, bias, eps):
        x, y, mean, rstd = _ext.ln_fwd_add(a, b, weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        return x, y

    @staticmethod
    def backward(ctx, gx, gy):
        x, weight, mean, rstd = ctx.saved_tensors
        # gx rides ln_bwd's dx store loop (fp32: bitwise-identical to the
        # old separate `dx = dx + gx` kernel; fp16: one rounding instead
        # of two). materialize_grads means gx is never None, but keep the
        # guard for safety — an undefined gx just skips the fused add.
        if gx is not None and not gx.is_contiguous():
            gx = gx.contiguous()
        dx, dw, db = _ext.ln_bwd(x, gy.contiguous(), weight, mean, rstd, gx)
        # NOTE (load-bearing): dx is returned for BOTH a and b as the
        # SAME tensor object — valid because autograd only reads these
        # edges; a .clone() would cost a full-size copy for nothing.
        return dx, dx, dw, db, None


class FusedLayerNorm(nn.LayerNorm):
    """Params (weight/bias) are ALWAYS fp32; the kernels accept fp32 or
    fp16 activations (fp16 I/O with fp32 stats/affine — the fp16-storage
    path, 2026-07-25). LN statistics come back fp32 in both cases."""

    def _eligible(self, x):
        return (x.is_cuda and x.dtype in (torch.float32, torch.float16)
                and x.is_contiguous()
                and len(self.normalized_shape) == 1
                and self.normalized_shape[0] <= 256
                and self.elementwise_affine and _load() is not None)

    def forward(self, x):
        if self._eligible(x):
            if not torch.is_grad_enabled():
                # rollout fast path: skip the Function.apply dispatch
                # (ctx/bookkeeping cost real CPU in the enqueue-bound
                # rollout; same kernel, bit-identical)
                return _ext.ln_fwd(x, self.weight, self.bias, self.eps)[0]
            return _LnFn.apply(x, self.weight, self.bias, self.eps)
        if x.dtype is torch.float16:
            # stock nn.LayerNorm with fp32 params: compute in fp32,
            # round once — mirrors the kernel's precision contract
            return super().forward(x.float()).to(x.dtype)
        return super().forward(x)

    def forward_add(self, a, b):
        """(a + b, LayerNorm(a + b)) — one fused pass on the CUDA path
        (the residual sum a + b is a full-size elementwise kernel the
        fusion absorbs). Fallback computes the same bits unfused; the
        contiguous sum re-enters forward()'s kernel eligibility."""
        if (self._eligible(a) and b.is_contiguous() and a.shape == b.shape
                and b.dtype is a.dtype):
            if not torch.is_grad_enabled():
                x, y, _, _ = _ext.ln_fwd_add(a, b, self.weight, self.bias,
                                             self.eps)
                return x, y
            return _LnAddFn.apply(a, b, self.weight, self.bias, self.eps)
        if a.dtype is torch.float16:
            x = (a.float() + b.float()).to(a.dtype)
        else:
            x = a + b
        return x, self.forward(x)
