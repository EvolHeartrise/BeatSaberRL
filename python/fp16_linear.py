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

# relu_bwd widths (the 138k-era 64/128 plus 256/512 since the audit's
# Tier 1.2, 2026-08-15): db comes from the kernel's fixed-order partials
# at these widths and from a torch gm.sum(0) elsewhere.
# 192/384/768 added 2026-09-26 (the 192/384/768 recipe's encoder,
# SimBa-trunk/critic and ffn-hidden widths, 384-thread blocks).
_RELU_BWD_DIMS = (64, 128, 256, 512, 192, 384, 768)

# cuBLASLt fused bias+ReLU epilogue for the Linear+ReLU sites (09-02
# static audit A §5 / Tier 2.5, measured 2026-09-03). The forward runs
# as ONE `torch._addmm_activation(b16, x, w16.t())`
# (cublasLt RELU_BIAS epilogue) instead of the two-pass bias-free
# `torch.mm` + in-place `bias_relu_(y, b_fp32)`; the backward is
# unchanged (relu_bwd from the saved post-ReLU y).
#
# History, so nobody re-runs the sm_61 conclusion: the two-pass split
# was adopted in round-4 #2 (2026-07, GTX 1080 / sm_61, d=64) because
# the cuBLAS bias epilogue there cost 33-59% of each addmm; the
# 08-16 audit's Tier 1.1 fixed the bias_relu_ widths (2026-08-15) but
# the planned cuBLASLt re-probe never ran. Numerics vs the two-pass
# path: ONE fp16 rounding instead of two and the bias enters as the
# fp16 replica instead of the fp32 master — fp16 last-rounding class
# (~1e-3 rel on logits), NOT bit-identical; the fused result is the
# CLOSER of the two to an fp32 reference (1.3e-3 vs 2.1e-3 max-abs on
# the ffn.0 shape).
#
# Default ON, decided 2026-09-03 on the RTX 5090 (sm_120, torch
# 2.8.0.dev+cu128): cublasLt runs the RELU_BIAS epilogue in the SAME
# cutlass kernel at the same time as the bias-free mm (ffn.0 at
# micro 8192: 224 us fused vs 218 us mm-only vs 509 us mm + bias_relu_),
# so the whole bias_relu_ pass (288 us per ffn.0 call, 1.24 ms per
# production minibatch) disappears. Measured: update phase 8.75 ->
# 8.41 s/4 upd (-4.0%), rollout 24.40 -> 24.01 s (-1.6%; the rollout
# forward runs the same sites at N=32768), --bench 126.3k -> 129.1k
# steps/s (+2.3%), bench_policy_update 16384/accum 2 32.1 -> 30.9 ms,
# bench_policy_forward @32768 18.4 -> 16.0 ms. Side effect: the
# rollout-vs-update logprob consistency diagnostic r1 went from ~1e-6
# to exactly 0 — the Lt heuristic picks the same kernel at both M, the
# plain-mm heuristic did not. (The BEAT_LT_EPILOGUE=0 bisect hatch was
# removed 2026-09-22; the two-pass form survives only as the routing
# fallback below.)


def _lt_epilogue_ok(x2, wh, bh):
    """True when torch will actually route `_addmm_activation` to the
    cublasLt epilogue: row-major contiguous operands with a contiguous
    1-D bias. Anything else makes torch fall back to bias-copy +
    GEMM(beta=1) + a separate relu_ (3 passes, worse than the 2 we
    replace). Pointer alignment is not a routing condition (Lt picks
    an align1/align2 kernel), but a 2-B-offset bias measured 1.8x
    slower on the ffn.0 shape (2026-09-03 probe), so 16-B alignment is
    required too — every flat replica view satisfies it (ALIGN = 16
    elements in flatten_fp16_replicas). Odd K is fine: the encoder
    first layers (K = 18 / 34 / 78) measured as ONE align2 cutlass
    kernel at 0.55-0.64x the two-pass time."""
    return (x2.is_contiguous() and wh.is_contiguous()
            and bh.is_contiguous() and bh.dim() == 1
            and x2.data_ptr() % 16 == 0 and wh.data_ptr() % 16 == 0
            and bh.data_ptr() % 16 == 0)


def _relu_forward(x2, wh, bh, b):
    """relu(x2 @ wh.T + bias) on the fp16 stream: the fused cublasLt
    epilogue (bias = fp16 replica `bh`) when the operands route to it,
    else the two-pass bias-free mm + bias_relu_ (bias = fp32 master
    `b`)."""
    if _lt_epilogue_ok(x2, wh, bh):
        return torch._addmm_activation(bh, x2, wh.t())
    y = torch.mm(x2, wh.t())
    _load().bias_relu_(y, b.detach())
    return y


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
            cublas = ["cublas.lib", "cublasLt.lib"] if os.name == "nt" \
                else ["-lcublas", "-lcublasLt"]

            def _full():
                from beat_arena_env.cuda_env import _toolchain_env
                _toolchain_env()
                os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list())
                from torch.utils.cpp_extension import load
                return load(name="fp16_gemm", sources=srcs,
                            build_directory=bdir,
                            extra_cuda_cflags=cuda_flags(),
                            extra_ldflags=cublas, verbose=True)

            _ext = fast_load("fp16_gemm", srcs, bdir,
                             cuda_flags() + cublas, _full)
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


# Row count from which a Linear's (dW, db) pair runs as ONE cuBLASLt
# GEMM with the BGRADB epilogue (csrc/fp16_gemm.cu wgrad_bgrad) instead
# of GemmEx wgrad + a separate torch column sum over dy (2026-09-26).
# Measured on the RTX 5090 at the token-level sites (R = B x 24..29
# rows): 436 -> 247 us (192x192), 907 -> 692 (qkv 576x192), 592 -> 427
# (critic k|v); at R = 16384 (trunk/critic/heads) it is a wash or much
# worse (the 145-row head GEMM: 99 -> 869 us), hence the gate. Numerics:
# dW near-tie (fp32 accumulation order, <= 4e-5 rel to max), db near-tie
# and closer to an fp64 reference than the torch sum.
_LT_BGRAD_MIN_ROWS = 65536


def _wgrad_db(g2, x2, need_w, need_b):
    """(dW, db) fp32 for dy = g2 (R, M), x = x2 (R, K); each None when
    not needed."""
    ext = _load()
    if (need_w and need_b and ext is not None
            and g2.shape[0] >= _LT_BGRAD_MIN_ROWS
            and hasattr(ext, "wgrad_bgrad")):
        r = ext.wgrad_bgrad(g2, x2)
        if r:
            return r[0], r[1]
    dw = db = None
    if need_w:
        dw = (ext.wgrad(g2, x2) if ext is not None             # fp32 out
              else g2.float().t() @ x2.float())
    if need_b:
        db = g2.sum(0, dtype=torch.float32)
    return dw, db


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
        dwcat, dbcat = _wgrad_db(g2, x2, any(ctx.needs_input_grad[4:4 + n]),
                                 any(ctx.needs_input_grad[4 + n:]))
        if dwcat is not None:
            o = 0
            for i, r in enumerate(splits):
                dws[i] = dwcat[o:o + r]
                o += r
        if dbcat is not None:
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
        if not x2.is_contiguous():
            # a (B, 1, K) token slice reshapes to a strided (B, K) VIEW
            # (the critic's body-row query, 2026-09-02); addmm takes the
            # leading-dim stride but wgrad's cublasGemmEx contract needs
            # a contiguous x — copy here, once (2 MB at micro 8192).
            x2 = x2.contiguous()
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
        dw, db = _wgrad_db(g2, x2, ctx.needs_input_grad[1],
                           ctx.needs_input_grad[2])
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
    """Linear + ReLU with the bias in an epilogue. Two-pass (default
    unless the operands route to the Lt epilogue): bias-free mm, then in-place
    y = relu(y + b_fp32) (csrc/fp16_gemm.cu bias_relu_) — replaces the
    standalone ReLU pass at identical traffic while dodging the slow
    sm_61 addmm bias epilogue. Fused: one cublasLt GEMM with the
    RELU_BIAS epilogue (see _relu_forward). Either way the backward
    only needs the saved post-ReLU y."""
    @staticmethod
    def forward(ctx, x, w, b):
        wh, bh = _half_params(w, b)
        x2 = x.reshape(-1, x.shape[-1])
        y = _relu_forward(x2, wh, bh, b)
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
        wh, bh = _half_params(w, b)
        x2 = x.reshape(-1, x.shape[-1])
        y = _relu_forward(x2, wh, bh, b)
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


# --- packed-gradient fused attention (2026-09-26) -------------------------
# The non-final blocks' q/k/v come out of ONE packed GEMM (qkv = x @
# [wq wk wv]^T, split into three views). Under F.scaled_dot_product_
# attention the mem-efficient backward writes dq, dk, dv into three
# separate buffers and autograd routes them through split's backward —
# a torch.cat that copies the whole (B*T, 3d) fp16 gradient per block
# (~5 ms per production minibatch across the 5 blocks). The aten
# kernel can write them straight into ONE (B, T, 3, H, dh) chunk — i.e.
# exactly qkv's [dq | dk | dv] row layout — via _efficient_attention_
# backward(shared_storage_dqdkdv=True); the SDPA-level backward just
# never asks for it. This Function runs the same forward op F.sdpa
# dispatches to and the same backward kernel with that flag, and hands
# the chunk back as qkv's gradient (a data_ptr/stride check proves it is
# the chunk; anything else falls back to the cat). Same kernels, same
# arithmetic: bitwise identical to the F.sdpa path (tests/
# test_packed_sdpa.py). Only the 3-way self-attention case qualifies
# (the flag needs Tq == T), so the final block's / the critic's packed
# k|v keep F.sdpa.


def _sdpa_bias(attn_bias, B, H, Tq, T):
    """Mirror aten's preprocess_mask for the mem-efficient backend:
    re-pad a mask whose non-last strides are not multiples of 8 (then
    slice back), and expand to (B, H, Tq, T). Values are untouched."""
    m = attn_bias
    if not (m.stride(-1) == 1
            and all(m.stride(i) % 8 == 0 for i in range(m.dim() - 1))):
        n = m.shape[-1]
        m = F.pad(m, (0, 8 - n % 8))[..., :n]
    return m.expand(B, H, Tq, T)


def _qkv_views(qkv, H):
    """(B, H, T, dh) q/k/v views of the packed (B, T, 3d) projection,
    laid out exactly as the block's split + view + transpose."""
    B, T, W = qkv.shape
    d = W // 3
    return tuple(t.view(B, T, H, d // H).transpose(1, 2)
                 for t in qkv.split(d, dim=-1))


def _as_packed(gs, src):
    """The (B, T, 3d) packed gradient whose [.., i*d:(i+1)*d] slices ARE
    the (B, T, H, dh) grads gs[i] (the backward's chunk), or None."""
    g0 = gs[0]
    B, T, H, dh = g0.shape
    d = H * dh
    W = d * len(gs)
    if tuple(src.shape) != (B, T, W):
        return None
    esz = g0.element_size()
    for i, g in enumerate(gs):
        if (g.stride() != (T * W, W, dh, 1)
                or g.data_ptr() != g0.data_ptr() + i * d * esz):
            return None
    return g0.as_strided((B, T, W), (T * W, W, 1))


class _PackedSdpa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, attn_bias, n_heads):
        q, k, v = _qkv_views(qkv, n_heads)
        B, H, T, _ = q.shape
        bias = _sdpa_bias(attn_bias, B, H, T, T)
        out, lse, seed, off = \
            torch.ops.aten._scaled_dot_product_efficient_attention(
                q, k, v, bias, True, 0.0, False)
        ctx.save_for_backward(qkv, bias, out, lse, seed, off)
        ctx.n_heads = n_heads
        return out

    @staticmethod
    def backward(ctx, gout):
        qkv, bias, out, lse, seed, off = ctx.saved_tensors
        q, k, v = _qkv_views(qkv, ctx.n_heads)
        T = q.shape[2]
        t = lambda x: x.transpose(1, 2)
        # exactly the SDPA-level backward's call (attention_backward.cu,
        # _scaled_dot_product_efficient_attention_backward_cuda), plus
        # the shared-storage flag
        gq, gk, gv, _ = torch.ops.aten._efficient_attention_backward(
            t(gout), t(q), t(k), t(v), bias, t(out), None, None, T, T,
            lse, 0.0, seed, off, 0, False, shared_storage_dqdkdv=True)
        g = _as_packed([gq, gk, gv], qkv)
        if g is None:
            B, _, H, dh = gq.shape
            g = torch.cat([x.reshape(B, T, H * dh) for x in (gq, gk, gv)],
                          dim=-1)
        return g, None, None


def packed_sdpa(qkv, attn_bias, n_heads):
    """Fused self-attention over a packed (B, T, 3d) q|k|v projection ->
    (B, H, T, dh): the same tensor F.scaled_dot_product_attention(q, k,
    v, attn_mask=attn_bias) returns for qkv's split/view/transpose
    views, with the gradient delivered as ONE packed (B, T, 3d) tensor.
    attn_bias: (B, 1, 1, T) additive mask, no grad. fp16 CUDA only;
    grad-free calls take F.sdpa itself (same forward kernel)."""
    if not torch.is_grad_enabled():
        q, k, v = _qkv_views(qkv, n_heads)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    return _PackedSdpa.apply(qkv, attn_bias, n_heads)
