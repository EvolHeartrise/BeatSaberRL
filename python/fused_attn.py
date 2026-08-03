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
(d != 64, T > 32, CPU, fp32). Odd T is supported since 2026-08-02 (the
mask-bias smem load is plain __half now — the v10 19-token layout was
silently falling back to eager before that).
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
            from beat_arena_env.jit_cache import CUDA_FLAGS, fast_load
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "beat_arena_env", "csrc")
            bdir = os.path.join(src, "build_attn")
            os.makedirs(bdir, exist_ok=True)
            srcs = [os.path.join(src, "fused_attn.cu")]

            def _full():
                from beat_arena_env.cuda_env import _toolchain_env
                _toolchain_env()
                os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")
                from torch.utils.cpp_extension import load
                return load(name="fused_attn_cuda", sources=srcs,
                            build_directory=bdir,
                            extra_cuda_cflags=CUDA_FLAGS, verbose=True)

            _ext = fast_load("fused_attn_cuda", srcs, bdir,
                             CUDA_FLAGS, _full)
        except Exception as e:
            _ext_failed = True
            print(f"[fused-attn] unavailable, eager attention: {e}",
                  flush=True)
    return _ext


def eligible(h, attn_bias):
    """Shape-level eligibility, checkable BEFORE the q/k/v projections
    run (the bias-folding path must decide first whether to emit
    bias-free mms). h: the (B, T, d) LN output feeding the projections."""
    return (h.dtype is torch.float16 and h.is_cuda and h.size(-1) == 64
            and h.size(1) <= 32
            and attn_bias is not None
            and attn_bias.dtype is torch.float16
            and attn_bias.is_contiguous()
            and _load() is not None)


class _AttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, alpha, bq, bk, bv):
        out, p = _ext.attn_fwd(q, k, v, bias, alpha, bq, bk, bv)
        ctx.save_for_backward(p, q, k, v, bq, bk, bv)
        ctx.alpha = alpha
        return out

    @staticmethod
    def backward(ctx, dout):
        p, q, k, v, bq, bk, bv = ctx.saved_tensors
        want_db = (bq is not None
                   and any(ctx.needs_input_grad[5:8]))
        dq, dk, dv, dbp = _ext.attn_bwd(dout.contiguous(), p, q, k, v,
                                        ctx.alpha, bq, bk, bv, want_db)
        # bias grads: d(b*) = channel-sum of d(*). The kernel emits
        # per-sample (3, D) fp32 partials (each block sums its own
        # L2-hot rows); one deterministic torch sum(0) finishes the
        # job — replaces three full-tensor fp16 re-reads (2026-07-28)
        dbq = dbk = dbv = None
        if want_db:
            if dbp.numel():
                s = dbp.sum(0)                   # (3, D) fp32
                dbq = s[0] if ctx.needs_input_grad[5] else None
                dbk = s[1] if ctx.needs_input_grad[6] else None
                dbv = s[2] if ctx.needs_input_grad[7] else None
            else:                                # T < 4: kernel opted out
                dbq = (dq.sum((0, 1), dtype=torch.float32)
                       if ctx.needs_input_grad[5] else None)
                dbk = (dk.sum((0, 1), dtype=torch.float32)
                       if ctx.needs_input_grad[6] else None)
                dbv = (dv.sum((0, 1), dtype=torch.float32)
                       if ctx.needs_input_grad[7] else None)
        return dq, dk, dv, None, None, dbq, dbk, dbv


def fused_attn(q, k, v, attn_bias, alpha, bq=None, bk=None, bv=None):
    """softmax(alpha * (q+bq) @ (k+bk)^T + attn_bias) @ (v+bv), one
    kernel per direction. attn_bias: (B, 1, T) additive key mask;
    bq/bk/bv: optional fp32 projection biases folded into the loads
    (pass bias-free q/k/v mms when using them)."""
    if not torch.is_grad_enabled():
        # need_p=False: p exists only as the backward's saved softmax;
        # with no grad the kernel neither stores nor allocates it
        out, _ = _ext.attn_fwd(q, k, v, attn_bias, alpha, bq, bk, bv,
                               False)
        return out
    return _AttnFn.apply(q, k, v, attn_bias, alpha, bq, bk, bv)
