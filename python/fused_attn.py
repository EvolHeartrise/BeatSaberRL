"""Fused single-head attention (csrc/fused_attn.cu) for the policy's
tiny-sequence shapes — round-4 item #15.

Replaces baddbmm -> softmax -> bmm on the fp16 path only (the fp32
update path keeps eager for bit-exactness of the record protocol; the
round-4 SDPA probe measured torch's mem-efficient backend at 0.88x fwd /
0.65x fwd+bwd vs eager on sm_61, so this is the remaining route to the
attention slice). fp32 accumulation, fp16 I/O; p saved fp16 like eager's
softmax output. Numerics are the documented near-tie tolerance class,
NOT bitwise vs eager. Backward emits contiguous dq/dk/dv, removing the
hidden .contiguous() copies in _Fp16Linear.backward.

Falls back to eager when the extension or shape eligibility is missing
(d != 64, odd/large T, CPU, fp32).
"""

from __future__ import annotations

import torch

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
            bdir = os.path.join(src, "build_attn")
            os.makedirs(bdir, exist_ok=True)
            _ext = load(name="fused_attn_cuda",
                        sources=[os.path.join(src, "fused_attn.cu")],
                        build_directory=bdir,
                        extra_cuda_cflags=["-arch=sm_61"])
        except Exception as e:
            _ext_failed = True
            print(f"[fused-attn] unavailable, eager attention: {e}",
                  flush=True)
    return _ext


def eligible(q, k, v, attn_bias):
    return (q.dtype is torch.float16 and q.is_cuda and q.size(-1) == 64
            and k.size(1) <= 32 and k.size(1) % 2 == 0
            and q.size(1) <= k.size(1)
            and q.is_contiguous() and k.is_contiguous()
            and v.is_contiguous() and attn_bias.is_contiguous()
            and attn_bias.dtype is torch.float16
            and _load() is not None)


class _AttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, alpha):
        out, p = _ext.attn_fwd(q, k, v, bias, alpha)
        ctx.save_for_backward(p, q, k, v)
        ctx.alpha = alpha
        return out

    @staticmethod
    def backward(ctx, dout):
        p, q, k, v = ctx.saved_tensors
        dq, dk, dv = _ext.attn_bwd(dout.contiguous(), p, q, k, v, ctx.alpha)
        return dq, dk, dv, None, None


def fused_attn(q, k, v, attn_bias, alpha):
    """softmax(alpha * q @ k^T + attn_bias) @ v, one kernel per
    direction. attn_bias: (B, 1, T) additive key mask."""
    if not torch.is_grad_enabled():
        out, _ = _ext.attn_fwd(q, k, v, attn_bias, alpha)
        return out
    return _AttnFn.apply(q, k, v, attn_bias, alpha)
