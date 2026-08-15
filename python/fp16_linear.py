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

import os as _os

import torch
import torch.nn.functional as F

_ext = None
_ext_failed = False

# relu_bwd widths beyond the 138k-era 64/128 (audit Tier 1.2, added
# 2026-08-15). BEAT_NO_WIDE_RELU_BWD=1 pins the old gate for A/B —
# note db then comes from a torch gm.sum(0) instead of the kernel's
# fixed-order partials, so the two settings differ in db's LAST
# ROUNDING only (same accepted class as the existing 64/128 path).
_RELU_BWD_DIMS = ((64, 128) if _os.environ.get("BEAT_NO_WIDE_RELU_BWD")
                  == "1" else (64, 128, 256, 512))


def _load():
    global _ext, _ext_failed
    if _ext is None and not _ext_failed:
        try:
            import os
            from beat_arena_env.jit_cache import (arch_list, cuda_flags,
                                                  fast_load)
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "beat_arena_env", "csrc")
            bdir = os.path.join(src, "build_fp16")
            os.makedirs(bdir, exist_ok=True)
            srcs = [os.path.join(src, "fp16_gemm.cu")]
            cublas = "cublas.lib" if os.name == "nt" else "-lcublas"

            def _full():
                from beat_arena_env.cuda_env import _toolchain_env
                _toolchain_env()
                os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list())
                from torch.utils.cpp_extension import load
                return load(name="fp16_gemm", sources=srcs,
                            build_directory=bdir,
                            extra_cuda_cflags=cuda_flags(),
                            extra_ldflags=[cublas], verbose=True)

            _ext = fast_load("fp16_gemm", srcs, bdir,
                             cuda_flags() + [cublas], _full)
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


# Flat replica state (2026-07-28): [version, flat16, flat32]. When a
# module is flattened (flatten_fp16_replicas), every Linear's fp32
# master becomes a view into flat32 and its fp16 replica a view into
# flat16 — a version bump then costs ONE flat16.copy_(flat32) cast
# kernel instead of ~26 per-tensor copies per refresh site.
_flat = None


def flat_refresh():
    """Recast the whole flat fp16 replica buffer if stale. One kernel;
    in-place, so storages captured by a CUDA graph stay live."""
    if _flat is not None and _flat[0] != _weights_version:
        _flat[1].copy_(_flat[2])
        _flat[0] = _weights_version


def flatten_fp16_replicas(module, tail_group=None, groups=None):
    """Repack every nn.Linear master (w, b) of `module` into ONE
    contiguous fp32 buffer (p.data become views — Adam keeps mutating
    them in place, state_dict/checkpoints unchanged) with a parallel
    flat fp16 replica buffer. `tail_group` (optional, ordered list of
    Linears) is laid out LAST as [w0..wn | b0..bn] so the caller can
    view the group's concatenated weights/biases as single contiguous
    tensors (the policy's fused action head). Returns (w16, b16) views
    for the tail group, or None.

    `groups` (optional, list of ordered Linear lists, 2026-08-15 audit
    Tier 1.4): each group is ALSO laid out [w0..wn | b0..bn] so its
    concatenated fp16 (weights, biases) exist as static views —
    registered on group[0].weight._group_views for packed_halves().
    Group members keep their individual _flat_views too (a member can
    still run as a standalone Linear — the final block's q_proj does).

    MUST run before any CUDA-graph capture, and never again after —
    p.data pointers move exactly once, here."""
    import torch.nn as nn
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    tail = list(tail_group or [])
    groups = [list(g) for g in (groups or [])]
    grouped = [m for g in groups for m in g]
    others = [m for m in linears if m not in tail and m not in grouped]
    # Alignment contract: cuBLAS selects vectorized GEMM paths by
    # pointer alignment (and future consumers may cast to __half2*/
    # float2*, needing 8 B) — so every independently-consumed
    # slice starts on a 16-ELEMENT boundary (64 B fp32 / 32 B fp16,
    # comfortably past every kernel's preference). The tail BIAS group
    # is the one exception — its members stay back-to-back (unaligned)
    # so the concatenated (rows,) view exists; it is only ever consumed
    # by torch ops, which take any alignment.
    ALIGN = 16
    plan = []                      # (param, offset)
    off = 0
    for m in others:
        for p_ in (m.weight, m.bias):
            off = (off + ALIGN - 1) & ~(ALIGN - 1)
            plan.append((p_, off))
            off += p_.numel()
    group_meta = []                # (members, w_off, b_off)
    for g in groups:
        assert all(m.weight.numel() % ALIGN == 0 for m in g), \
            "group weights must keep alignment back-to-back"
        off = (off + ALIGN - 1) & ~(ALIGN - 1)
        gw0 = off
        for m in g:
            plan.append((m.weight, off))
            off += m.weight.numel()
        off = (off + ALIGN - 1) & ~(ALIGN - 1)
        gb0 = off                  # biases back-to-back: the cat view
        for m in g:                # must exist (torch-only consumers)
            plan.append((m.bias, off))
            off += m.bias.numel()
        group_meta.append((g, gw0, gb0))
    off = (off + ALIGN - 1) & ~(ALIGN - 1)
    assert all(m.weight.numel() % ALIGN == 0 for m in tail), \
        "tail weights must keep alignment back-to-back"
    for m in tail:
        plan.append((m.weight, off))
        off += m.weight.numel()
    off = (off + ALIGN - 1) & ~(ALIGN - 1)
    for m in tail:                 # biases: contiguous, no padding
        plan.append((m.bias, off))
        off += m.bias.numel()
    dev = plan[0][0].device
    flat32 = torch.zeros(off, device=dev)
    offs = {}
    for p, o in plan:
        n = p.numel()
        flat32[o:o + n].copy_(p.data.reshape(-1))
        p.data = flat32[o:o + n].view_as(p)
        offs[id(p)] = o
    flat16 = flat32.half()
    global _flat
    _flat = [_weights_version, flat16, flat32]
    for m in linears:
        wo, bo = offs[id(m.weight)], offs[id(m.bias)]
        m.weight._flat_views = (
            flat16[wo:wo + m.weight.numel()].view_as(m.weight),
            flat16[bo:bo + m.bias.numel()].view_as(m.bias))
    for g, gw0, gb0 in group_meta:
        rows = sum(m.weight.shape[0] for m in g)
        n_w = sum(m.weight.numel() for m in g)
        g[0].weight._group_views = (
            flat16[gw0:gw0 + n_w].view(rows, g[0].weight.shape[1]),
            flat16[gb0:gb0 + rows])
    if not tail:
        return None
    tw0 = offs[id(tail[0].weight)]
    tb0 = offs[id(tail[0].bias)]
    n_w = sum(m.weight.numel() for m in tail)
    n_b = sum(m.bias.numel() for m in tail)
    rows = sum(m.weight.shape[0] for m in tail)
    w16 = flat16[tw0:tw0 + n_w].view(rows, tail[0].weight.shape[1])
    b16 = flat16[tb0:tb0 + n_b]
    return w16, b16


def _half_params(w, b):
    """Epoch-checked fp16 replica cache. The replicas are detached
    constants (the custom Function computes dw itself), so caching is
    valid in grad mode too. The win is the ROLLOUT: 128 forwards per
    update on frozen weights pay ONE cast set instead of 128 (34
    launches each — measured as --fp16 all regressing below
    update-only without the cache). Flat-registered weights (see
    flatten_fp16_replicas) resolve to views and share one global
    refresh."""
    fv = getattr(w, "_flat_views", None)
    if fv is not None:
        flat_refresh()
        return fv
    cache = getattr(w, "_fp16_cache", None)
    if cache is None:
        cache = (_weights_version, w.detach().half(), b.detach().half())
        w._fp16_cache = cache
    elif cache[0] != _weights_version:
        # refresh IN PLACE: the replica storages never move, so a CUDA
        # graph that captured them keeps reading fresh weights after
        # every opt.step (a reallocating rebuild would silently freeze
        # the graphed rollout's weights — the r1-diagnostic failure
        # class from the first _version-keyed cut)
        cache[1].copy_(w.detach())
        cache[2].copy_(b.detach())
        cache = (_weights_version, cache[1], cache[2])
        w._fp16_cache = cache
    return cache[1], cache[2]


# Packed Linear groups (audit Tier 1.4/1.5, 2026-08-15): several
# same-input Linears run as ONE GEMM against their concatenated fp16
# replicas. Flat-registered groups (flatten_fp16_replicas groups=) get
# static views refreshed by the single flat cast; otherwise a
# version-keyed cat cache (registered here so refresh_packed() can
# force-refresh under CUDA graphs, mirroring _half_params).
_packed_caches = []


def _refresh_packed_one(c):
    o = 0
    w16 = c[1].view(-1)
    for m in c[3]:
        n = m.weight.numel()
        w16[o:o + n].copy_(m.weight.detach().view(-1))
        o += n
    o = 0
    for m in c[3]:
        n = m.bias.numel()
        c[2][o:o + n].copy_(m.bias.detach())
        o += n
    c[0] = _weights_version


def refresh_packed():
    """Force-refresh every non-flat packed cat cache (graphed callers —
    flat groups ride flat_refresh and never appear here)."""
    for c in _packed_caches:
        if c[0] != _weights_version:
            _refresh_packed_one(c)


def packed_halves(ms):
    """(w16, b16) cat replicas for an ordered Linear group `ms`: static
    flat-group views when flattened, else a version-keyed cache."""
    gv = getattr(ms[0].weight, "_group_views", None)
    if gv is not None:
        flat_refresh()
        return gv
    c = getattr(ms[0].weight, "_packed_cache", None)
    if c is None:
        w = torch.cat([m.weight.detach() for m in ms]).half()
        b = torch.cat([m.bias.detach() for m in ms]).half()
        c = [_weights_version, w, b, list(ms)]
        ms[0].weight._packed_cache = c
        _packed_caches.append(c)
    elif c[0] != _weights_version:
        _refresh_packed_one(c)
    return c[1], c[2]


class _Fp16PackedLinear(torch.autograd.Function):
    """One addmm over a group's concatenated fp16 replicas; the backward
    computes the CONCATENATED dW/db (one wgrad + one sum instead of one
    per member) and returns row slices routed to each member's fp32
    master. The masters ride along as unused forward args purely for
    grad routing — with the trainer's flat-grad views, autograd
    accumulates the slices in place, so the flat-guard contract holds."""

    @staticmethod
    def forward(ctx, x, wh, bh, splits, *params):
        x2 = x.reshape(-1, x.shape[-1])
        y = torch.addmm(bh, x2, wh.t())
        ctx.save_for_backward(x2, wh)
        ctx.splits = splits
        return y.view(*x.shape[:-1], wh.shape[0])

    @staticmethod
    def backward(ctx, gy):
        x2, wh = ctx.saved_tensors
        splits = ctx.splits
        g2 = gy.reshape(-1, gy.shape[-1])
        if not g2.is_contiguous():
            g2 = g2.contiguous()
        dx = None
        if ctx.needs_input_grad[0]:
            dx = (g2 @ wh).view(*gy.shape[:-1], wh.shape[1])
        n = len(splits)
        dws = [None] * n
        dbs = [None] * n
        if any(ctx.needs_input_grad[4:4 + n]):
            ext = _load()
            dwcat = (ext.wgrad(g2, x2) if ext is not None
                     else g2.float().t() @ x2.float())
            o = 0
            for i, r in enumerate(splits):
                dws[i] = dwcat[o:o + r]
                o += r
        if any(ctx.needs_input_grad[4 + n:]):
            dbcat = g2.sum(0, dtype=torch.float32)
            o = 0
            for i, r in enumerate(splits):
                dbs[i] = dbcat[o:o + r]
                o += r
        return (dx, None, None, None, *dws, *dbs)


def fp16_packed_linear(x, wh, bh, ms):
    """x @ cat(w).T + cat(b) for a Linear group on the fp16 stream. wh/bh
    are the group's cat fp16 replicas (packed_halves or a slice of them
    — the final block packs k/v only via the tail rows of its qkv
    group)."""
    if not torch.is_grad_enabled():
        x2 = x.reshape(-1, x.shape[-1])
        return torch.addmm(bh, x2, wh.t()).view(*x.shape[:-1], wh.shape[0])
    splits = tuple(m.weight.shape[0] for m in ms)
    ws = [m.weight for m in ms]
    bs = [m.bias for m in ms]
    return _Fp16PackedLinear.apply(x, wh, bh, splits, *ws, *bs)


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


class _Fp16LinearNoBias(torch.autograd.Function):
    """Bias-free half mm (round-4 #2: the cuBLAS bias epilogue costs
    33-59% of each addmm on sm_61). The bias VALUE is consumed by the
    downstream fused-attention kernel, whose backward supplies the bias
    grad — this Function deliberately returns None for it."""
    @staticmethod
    def forward(ctx, x, w, b):
        wh, _ = _half_params(w, b)
        x2 = x.reshape(-1, x.shape[-1])
        y = torch.mm(x2, wh.t())
        ctx.save_for_backward(x2, wh)
        return y.view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, gy):
        x2, wh = ctx.saved_tensors
        g2 = gy.reshape(-1, gy.shape[-1])
        if not g2.is_contiguous():
            g2 = g2.contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = (g2 @ wh).view(*gy.shape[:-1], wh.shape[1])
        if ctx.needs_input_grad[1]:
            ext = _load()
            dw = (ext.wgrad(g2, x2) if ext is not None
                  else g2.float().t() @ x2.float())
        return dx, dw, None


def fp16_linear_nobias(x, w, b):
    """mm-only Linear; bias rides the downstream kernel (fused attn)."""
    if not torch.is_grad_enabled():
        wh, _ = _half_params(w, b)
        x2 = x.reshape(-1, x.shape[-1])
        return torch.mm(x2, wh.t()).view(*x.shape[:-1], w.shape[0])
    return _Fp16LinearNoBias.apply(x, w, b)


class _Fp16LinearReLU(torch.autograd.Function):
    """Linear + ReLU with the bias in a custom epilogue: bias-free mm,
    then in-place y = relu(y + b_fp32) (csrc/fp16_gemm.cu bias_relu_).
    Replaces the standalone ReLU pass at identical traffic while
    dodging the slow sm_61 addmm bias epilogue."""
    @staticmethod
    def forward(ctx, x, w, b):
        wh, _ = _half_params(w, b)
        x2 = x.reshape(-1, x.shape[-1])
        y = torch.mm(x2, wh.t())
        _load().bias_relu_(y, b.detach())
        ctx.save_for_backward(x2, wh, y)
        return y.view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, gy):
        x2, wh, y = ctx.saved_tensors
        g2 = gy.reshape(-1, gy.shape[-1])
        if not g2.is_contiguous():
            g2 = g2.contiguous()
        ext = _load()
        part = None
        if hasattr(ext, "relu_bwd") and g2.shape[-1] in _RELU_BWD_DIMS:
            # one pass: mask + fp32 column-sum partials — replaces
            # threshold_backward + a separate full-tensor gm.sum(0)
            # re-read (2026-07-28)
            gm, part = ext.relu_bwd(g2, y)
        else:
            gm = torch.ops.aten.threshold_backward(g2, y, 0)
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = (gm @ wh).view(*gy.shape[:-1], wh.shape[1])
        if ctx.needs_input_grad[1]:
            dw = ext.wgrad(gm, x2)
        if ctx.needs_input_grad[2]:
            db = (part.sum(0) if part is not None
                  else gm.sum(0, dtype=torch.float32))
        return dx, dw, db


def fp16_linear_relu(x, w, b):
    """relu(x @ w.T + b) on the fp16 stream; requires the extension
    (callers gate on ext_ok())."""
    if not torch.is_grad_enabled():
        wh, _ = _half_params(w, b)
        x2 = x.reshape(-1, x.shape[-1])
        y = torch.mm(x2, wh.t())
        _load().bias_relu_(y, b.detach())
        return y.view(*x.shape[:-1], w.shape[0])
    return _Fp16LinearReLU.apply(x, w, b)


def ext_ok():
    return _load() is not None


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
