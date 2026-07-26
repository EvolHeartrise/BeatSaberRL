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
            from beat_arena_env.cuda_env import _toolchain_env
            _toolchain_env()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")
            from torch.utils.cpp_extension import load
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "beat_arena_env", "csrc")
            bdir = os.path.join(src, "build_ln")
            os.makedirs(bdir, exist_ok=True)
            _ext = load(name="layer_norm_cuda",
                        sources=[os.path.join(src, "layer_norm.cu")],
                        build_directory=bdir,
                        extra_cuda_cflags=["-arch=sm_61"])
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


class FusedLayerNorm(nn.LayerNorm):
    def forward(self, x):
        if (x.is_cuda and x.dtype is torch.float32 and x.is_contiguous()
                and len(self.normalized_shape) == 1
                and self.normalized_shape[0] <= 256
                and self.elementwise_affine and _load() is not None):
            return _LnFn.apply(x, self.weight, self.bias, self.eps)
        return super().forward(x)
