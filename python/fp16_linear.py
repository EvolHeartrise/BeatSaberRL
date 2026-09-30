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


# --- fp16-ACCUMULATE GEMMs (issue #3, 2026-09-27) --------------------------
# The RTX 5090's fp16 tensor cores run ~2x the fp32-accumulate peak with
# an fp16 accumulator (measured 353 vs 222 TFLOPS on 8192^3; the policy's
# K = 192/768 GEMMs are near memory-bound and gain 1.0-1.5x). The mode is
# a POLICY property (NotePolicy.fp16_accum, stamped in the checkpoint arch):
#   "off"  every GEMM accumulates in fp32 (the historical path)
#   "fwd"  the forward GEMMs that go through torch mm/addmm accumulate
#          in fp16 (torch's allow_fp16_accumulation -> CUBLAS_COMPUTE_16F)
#   "all"  "fwd" plus the torch dgrad GEMMs (dx = g @ W)
# NEVER flipped, in any mode:
#   - the wgrad GEMMs (dW = dY^T X: GemmEx COMPUTE_32F / the Lt BGRADB
#     plan) and every bias-grad reduction — they sum over ~475k rows;
#   - the Lt RELU_AUX_BIAS forward / DRELU dgrad epilogue GEMMs (block
#     FFN up-projection forward, the chains' mid dgrads): cuBLASLt on
#     sm_120 has no fast COMPUTE_16F kernel with an aux epilogue (every
#     heuristic algo measured 26-39 TFLOPS vs 160-170 at 32F);
#   - the odd-K encoder first layers' _addmm_activation (raw obs inputs;
#     measured no speed gain from the flag there).
# Both phases run the same GEMM calls under the same mode, so rollout ==
# update logp (r1) stays bitwise. The flag is set per call (host-side,
# read by cuBLAS at launch), so CUDA-graph capture records the mode in
# force at capture time.
ACCUM_MODES = ("off", "fwd", "all")
_ACC_FWD = False
_ACC_DGRAD = False


class fp16_accum:
    """Context manager: run the enclosed fp16-stream forward (and the
    backward of every Function it creates — they latch the dgrad mode at
    forward time) under accumulate mode `mode` (ACCUM_MODES)."""

    def __init__(self, mode):
        if mode not in ACCUM_MODES:
            raise ValueError(f"fp16 accumulate mode must be one of "
                             f"{ACCUM_MODES}, got {mode!r}")
        self.mode = mode

    def __enter__(self):
        global _ACC_FWD, _ACC_DGRAD
        self.prev = (_ACC_FWD, _ACC_DGRAD)
        _ACC_FWD = self.mode in ("fwd", "all")
        _ACC_DGRAD = self.mode == "all"
        return self

    def __exit__(self, *exc):
        global _ACC_FWD, _ACC_DGRAD
        _ACC_FWD, _ACC_DGRAD = self.prev
        return False


def _acc16(on, fn, *args):
    """fn(*args) with torch's cuBLAS fp16-accumulate flag set when `on`."""
    if not on:
        return fn(*args)
    prev = torch._C._get_cublas_allow_fp16_accumulation()
    torch._C._set_cublas_allow_fp16_accumulation(True)
    try:
        return fn(*args)
    finally:
        torch._C._set_cublas_allow_fp16_accumulation(prev)


def _addmm_f(b, x, wt):
    """Forward bias GEMM on the fp16 stream (mode-dependent accumulate)."""
    return _acc16(_ACC_FWD, torch.addmm, b, x, wt)


def _mm_f(x, wt):
    """Forward bias-free GEMM on the fp16 stream."""
    return _acc16(_ACC_FWD, torch.mm, x, wt)


def _linear_f(x, w, b):
    """F.linear on the fp16 stream (the rollout head)."""
    return _acc16(_ACC_FWD, F.linear, x, w, b)


def _dgrad(g, wh, on):
    """dx = g @ wh; `on` = the dgrad mode latched at forward time."""
    return _acc16(on, torch.mm, g, wh)


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
            srcs = [os.path.join(src, "fp16_gemm.cu"),
                    os.path.join(src, "small_attn.cu"),
                    os.path.join(src, "fused_ffn.cu")]
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
        y = _addmm_f(bh, x2, wh.t())
        ctx.save_for_backward(x2, wh)
        ctx.splits = splits
        ctx.acc = _ACC_DGRAD
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
            dx = _dgrad(g2, wh, ctx.acc).view(*gy.shape[:-1], wh.shape[1])
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
        return _addmm_f(bh, x2, wh.t()).view(*x.shape[:-1], wh.shape[0])
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
        y = _addmm_f(bh, x2, wh.t())
        ctx.save_for_backward(x2, wh)
        ctx.acc = _ACC_DGRAD
        return y.view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, gy):
        x2, wh = ctx.saved_tensors
        g2 = gy.reshape(-1, gy.shape[-1])
        if not g2.is_contiguous():
            g2 = g2.contiguous()
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = _dgrad(g2, wh, ctx.acc).view(*gy.shape[:-1], wh.shape[1])
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
        y = _mm_f(x2, wh.t())
        ctx.save_for_backward(x2, wh)
        ctx.acc = _ACC_DGRAD
        return y.view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, gy):
        x2, wh = ctx.saved_tensors
        g2 = gy.reshape(-1, gy.shape[-1])
        if not g2.is_contiguous():
            g2 = g2.contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = _dgrad(g2, wh, ctx.acc).view(*gy.shape[:-1], wh.shape[1])
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
        return _mm_f(x2, wh.t()).view(*x.shape[:-1], w.shape[0])
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
        ctx.acc = _ACC_DGRAD
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
            dx = _dgrad(gm, wh, ctx.acc).view(*gy.shape[:-1], wh.shape[1])
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


# --- Linear-ReLU-...-Linear chains with the ReLU in GEMM epilogues --------
# (2026-09-27, #7a.) Every fp16 ReLU on this network sits BETWEEN two
# Linears (block FFN ffn.0 -> ffn.2, the four encoder MLPs, SimBa trunk
# fc1 -> fc2, the critic MLP). A chain runs each Linear+ReLU forward as
# ONE cuBLASLt GEMM with the RELU_AUX_BIAS epilogue (output + a 1-bit
# ReLU mask), and in the backward the NEXT Linear's dgrad GEMM (g @ W)
# takes the DRELU epilogue: it applies the mask as it stores. The
# standalone relu_bwd pass (read g + y, write gm: ~7.5 ms of a production
# minibatch at the 768-wide ffn hidden) disappears; the ReLU layer's db
# comes from its own wgrad GEMM (_wgrad_db: BGRADB epilogue at >= 64k
# rows, else a torch column sum over the small gm). The post-ReLU y is
# still saved, for the next Linear's wgrad.
# Rollout and update run the SAME forward GEMM (the grad-free path also
# calls linear_relu_aux and drops the mask), so rollout/update logp
# agreement (r1) keeps its footing. Numerics vs the _Fp16LinearReLU path:
# forward near-tie (Lt kernel choice; the fp16 bias replica rides the
# epilogue in both), gm near-tie (the dgrad GEMM's kernel), db fp32 from
# the BGRADB epilogue / torch sum instead of relu_bwd's partials.


def _chain_forward(x2, ms, tail_bias, ext):
    """Forward over the chain; returns (out, inputs, masks) or None when
    an Lt plan is unavailable (caller falls back)."""
    h = x2
    ins, masks = [], []
    n = len(ms)
    for i, m in enumerate(ms):
        wh, bh = _half_params(m.weight, m.bias)
        ins.append(h)
        if i < n - 1:
            r = ext.linear_relu_aux(h, wh, bh)
            if not r:
                return None
            h, mk = r
            masks.append(mk)
        elif tail_bias:
            h = _addmm_f(bh, h, wh.t())
        else:
            h = _mm_f(h, wh.t())
    return h, ins, masks


class _Fp16ReluChain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ms, tail_bias, *params):
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        out, ins, masks = _chain_forward(x2, ms, tail_bias, _load())
        ctx.save_for_backward(*ins, *masks)
        ctx.ms = ms
        ctx.tail_bias = tail_bias
        ctx.lead = x.shape[:-1]
        ctx.acc = _ACC_DGRAD
        return out.view(*x.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, gy):
        ms = ctx.ms
        n = len(ms)
        saved = ctx.saved_tensors
        ins, masks = saved[:n], saved[n:]
        ext = _load()
        g = gy.reshape(-1, gy.shape[-1])
        if not g.is_contiguous():
            g = g.contiguous()
        nig = ctx.needs_input_grad
        dws = [None] * n
        dbs = [None] * n
        dx = None
        for i in range(n - 1, -1, -1):
            # (dW, db) of layer i from its own wgrad GEMM (BGRADB epilogue
            # at >= 64k rows), g already ReLU-masked by the dgrad below
            dws[i], dbs[i] = _wgrad_db(
                g, ins[i], nig[3 + i],
                nig[3 + n + i] and (i < n - 1 or ctx.tail_bias))
            wh = _half_params(ms[i].weight, ms[i].bias)[0]
            if i > 0:
                g = ext.dgrad_drelu(g, wh, masks[i - 1])
            elif nig[0]:
                dx = _dgrad(g, wh, ctx.acc).view(*ctx.lead, wh.shape[1])
        return (dx, None, None, *dws, *dbs)


def fp16_relu_chain(x, ms, tail_bias=True):
    """Linear(ms[0]) -> ReLU -> ... -> Linear(ms[-1]) on the fp16 stream
    with every ReLU in a GEMM epilogue (forward RELU_AUX_BIAS, backward
    DRELU). tail_bias=False runs the last Linear bias-free (the
    fold_tail contract: its bias is consumed downstream). Returns None
    when the extension lacks the Lt epilogues for these shapes (caller
    falls back to the per-Linear path)."""
    ext = _load()
    if ext is None or not hasattr(ext, "linear_relu_aux"):
        return None
    if ms[0].in_features % 8:
        # odd-K encoder inputs (saber 49 / torso 29 / leg 34): cuBLASLt
        # has no fused RELU_AUX_BIAS kernel there (it appends a separate
        # epilogue kernel), so they keep the fused RELU_BIAS forward +
        # relu_bwd (their backward is tiny; the forward runs every tick)
        return None
    if not torch.is_grad_enabled():
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        r = _chain_forward(x2, ms, tail_bias, ext)
        if r is None:
            return None
        return r[0].view(*x.shape[:-1], r[0].shape[-1])
    # probe the plans once per shape set before committing to autograd
    x2 = x.reshape(-1, x.shape[-1])
    key = (x2.shape[0],) + tuple(tuple(m.weight.shape) for m in ms)
    ok = _chain_ok.get(key)
    if ok is None:
        ok = _probe_chain(x2.shape[0], ms, ext)
        _chain_ok[key] = ok
    if not ok:
        return None
    ws = [m.weight for m in ms]
    bs = [m.bias for m in ms]
    return _Fp16ReluChain.apply(x, tuple(ms), bool(tail_bias), *ws, *bs)


_chain_ok = {}


def _probe_chain(R, ms, ext):
    """True when every Lt plan of the chain exists at R rows (tiny
    zero-size calls would not exercise the heuristic, so run R-row
    dummies once)."""
    dev = ms[0].weight.device
    for i, m in enumerate(ms[:-1]):
        H, K = m.weight.shape
        x = torch.zeros(R, K, device=dev, dtype=torch.float16)
        w = torch.zeros(H, K, device=dev, dtype=torch.float16)
        b = torch.zeros(H, device=dev, dtype=torch.float16)
        r = ext.linear_relu_aux(x, w, b)
        if not r:
            return False
        M = ms[i + 1].weight.shape[0]
        g = torch.zeros(R, M, device=dev, dtype=torch.float16)
        w2 = torch.zeros(M, H, device=dev, dtype=torch.float16)
        if ext.dgrad_drelu(g, w2, r[1]) is None:
            return False
    return True


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
        return _addmm_f(bh, x2, wh.t()).view(*x.shape[:-1], w.shape[0])
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


# --- T = 29 fused attention (csrc/small_attn.cu, 2026-09-27, #10) ----------
# One CTA per sample (4 warps = 4 heads, lane = query), fp32 softmax/P/dS,
# no logsumexp or output saved (the backward recomputes P exactly). Used by
# BOTH the rollout (grad-free) and the update forwards at every attention
# site on the fp16 stream — the non-final blocks' packed self-attention,
# the final block's 5 lead-token queries and the critic's 1-query pool —
# so rollout and update logp stay bitwise consistent (r1) at identical
# weights. Numerics vs torch's mem-efficient SDPA (which rounds P and dS
# to fp16 for its tensor-core GEMMs): fp16 last-rounding class, closer to
# an fp32 reference (tests/test_attn29.py). Other shapes keep SDPA.
_ATTN29 = (29, 192, 4)          # (keys, d, heads) the kernel is built for


def attn29_ok(T, d, n_heads):
    ext = _load()
    return (ext is not None and hasattr(ext, "attn29_fwd")
            and (T, d, n_heads) == _ATTN29)


class _Attn29Packed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, bias2):
        d = qkv.shape[-1] // 3
        out = _load().attn29_fwd(*qkv.split(d, dim=-1), bias2)
        ctx.save_for_backward(qkv, bias2)
        return out

    @staticmethod
    def backward(ctx, gout):
        qkv, bias2 = ctx.saved_tensors
        d = qkv.shape[-1] // 3
        g = torch.empty_like(qkv)
        _load().attn29_bwd(*qkv.split(d, dim=-1), bias2,
                           gout.contiguous(), *g.split(d, dim=-1))
        return g, None


class _Attn29QKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, bias2):
        out = _load().attn29_fwd(q, *kv.split(q.shape[-1], dim=-1), bias2)
        ctx.save_for_backward(q, kv, bias2)
        return out

    @staticmethod
    def backward(ctx, gout):
        q, kv, bias2 = ctx.saved_tensors
        dq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        dkv = torch.empty_like(kv)
        _load().attn29_bwd(q, *kv.split(q.shape[-1], dim=-1), bias2,
                           gout.contiguous(), dq,
                           *dkv.split(q.shape[-1], dim=-1))
        return dq, dkv, None


def attn29_packed(qkv, attn_bias):
    """Self-attention over the packed (B, 29, 3d) q|k|v projection ->
    contiguous (B, 29, d), heads concatenated (the out-projection's input
    layout). attn_bias: the (B, 1, 29) additive fp16 key mask. Gradient:
    ONE packed (B, 29, 3d) tensor."""
    B, T, W = qkv.shape
    bias2 = attn_bias.reshape(B, T)
    if not torch.is_grad_enabled():
        return _load().attn29_fwd(*qkv.split(W // 3, dim=-1), bias2)
    return _Attn29Packed.apply(qkv, bias2)


def attn29_qkv(q, kv, attn_bias):
    """Attention of TQ in {29, 5, 1} queries q (B, TQ, d) over the packed
    (B, 29, 2d) k|v projection -> contiguous (B, TQ, d). Gradients: dq
    and ONE packed (B, 29, 2d) dkv."""
    B, T, _ = kv.shape
    bias2 = attn_bias.reshape(B, T)
    if q.stride(-1) != 1 or q.stride(1) % 8 or q.stride(0) % 8:
        q = q.contiguous()
    if not torch.is_grad_enabled():
        return _load().attn29_fwd(q, *kv.split(q.shape[-1], dim=-1), bias2)
    return _Attn29QKV.apply(q, kv, bias2)


# --- fused block FFN (csrc/fused_ffn.cu, 2026-09-30, review item I13) -----
# A non-final block's FFN plus the NEXT block's residual add + ln_attn runs
# as ONE kernel: out = x + relu(h2 W0^T + b0) W2^T + b2, hn = LN(out). The
# 768-wide hidden stays on-chip; the grad-free (rollout) call writes only
# out and hn — no hidden, no ReLU mask, no ffn.2 output, no separate LN
# pass. The final block uses the residual-only variant (no LN).
# Grad mode runs the SAME kernel (bitwise-identical out / hn: the r1
# contract) and additionally writes what the backward needs: the fp16
# hidden (ffn.2's wgrad input), the ReLU bitmask in cuBLASLt's RELU_AUX
# layout (so ffn.2's dgrad is the same DRELU-epilogue GEMM the unfused
# chain runs) and the LN mean / rstd. The backward is the unfused path's
# kernels: ln_bwd (+ its fused bias-grad column sum), GemmEx / BGRADB
# wgrads, dgrad_drelu, torch dgrad under the latched accumulate mode.
# Numerics vs the unfused path: fp16 last-rounding class, NOT bitwise
# (b0 enters as the fp32 master instead of the fp16 replica, ffn.2's
# output is not rounded to fp16 before the residual add, and the MMA
# order is not cuBLAS's). Accumulate: the up-projection stays fp32 (as
# the unfused RELU_AUX GEMM), the down-projection follows the policy's
# fp16_accum forward flag (as the unfused ffn.2 torch mm); mode "up16"
# (experimental) also fp16-accumulates the up-projection. Opt-in per
# policy (NotePolicy.fused_ffn, arch "fused_ffn"; "off" by default until
# pod-verified).
FFN_MODES = ("off", "on", "up16")
_FFN_SHAPE = (192, 768)          # (d, hidden) the kernel is built for
_FFN_MODE = "off"                # set by ffn_fusion() around a forward
_ffn_dev_ok = {}                 # device index -> ext.ffn_fused_supported()


class ffn_fusion:
    """Context manager: run the enclosed fp16-stream forward with the
    fused block FFN in mode `mode` (FFN_MODES). A Function created inside
    latches what its backward needs — backward() may run outside."""

    def __init__(self, mode):
        if mode not in FFN_MODES:
            raise ValueError(f"fused FFN mode must be one of {FFN_MODES}, "
                             f"got {mode!r}")
        self.mode = mode

    def __enter__(self):
        global _FFN_MODE
        self.prev = _FFN_MODE
        _FFN_MODE = self.mode
        return self

    def __exit__(self, *exc):
        global _FFN_MODE
        _FFN_MODE = self.prev
        return False


def fused_ffn_ok(x, ffn, ln=None):
    """True when the fused kernel applies to a block FFN `ffn` (the
    [Linear, ReLU, Linear] Sequential) on the activations `x` under the
    current ffn_fusion mode, with `ln` (a LayerNorm, or None) as the
    epilogue LN. False = the caller keeps the unfused path (mode off, not
    fp16 CUDA, other widths, no extension, device without the smem)."""
    if _FFN_MODE == "off" or x.dtype is not torch.float16 or not x.is_cuda:
        return False
    ms = list(ffn)
    if (len(ms) != 3 or not isinstance(ms[1], torch.nn.ReLU)
            or not all(isinstance(m, torch.nn.Linear) and m.bias is not None
                       for m in (ms[0], ms[2]))
            or (ms[0].in_features, ms[0].out_features) != _FFN_SHAPE
            or (ms[2].in_features, ms[2].out_features) != _FFN_SHAPE[::-1]
            or x.shape[-1] != _FFN_SHAPE[0]):
        return False
    if ln is not None and not (
            tuple(ln.normalized_shape) == (_FFN_SHAPE[0],)
            and ln.elementwise_affine and ln.weight is not None
            and ln.bias is not None):
        return False
    ext = _load()
    if ext is None or not hasattr(ext, "ffn_fused"):
        return False
    dev = (x.device.index if x.device.index is not None
           else torch.cuda.current_device())
    ok = _ffn_dev_ok.get(dev)
    if ok is None:
        with torch.cuda.device(dev):
            ok = bool(ext.ffn_fused_supported())
        _ffn_dev_ok[dev] = ok
    return ok


def _ffn_acc():
    """Kernel accumulate code: 0 = fp32 both GEMMs, 1 = fp16 down-
    projection (fp16_accum "fwd" / "all"), 2 = fp16 both ("up16")."""
    if _FFN_MODE == "up16":
        return 2
    return 1 if _ACC_FWD else 0


def _ffn_call(x2, h22, ffn0, ffn2, ln, acc, save):
    """ext.ffn_fused on (R, 192) views; returns (result list, w0h, w2h)."""
    w0h, _ = _half_params(ffn0.weight, ffn0.bias)
    w2h, _ = _half_params(ffn2.weight, ffn2.bias)
    r = _load().ffn_fused(
        x2, h22, w0h, ffn0.bias.detach(), w2h, ffn2.bias.detach(),
        None if ln is None else ln.weight.detach(),
        None if ln is None else ln.bias.detach(),
        float(ln.eps) if ln is not None else 0.0, acc, save)
    return r, w0h, w2h


class _FusedFfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h2, w0, b0, w2, b2, lnw, lnb, mods, acc):
        # w0/b0/w2/b2/lnw/lnb: the fp32 masters, forward args purely for
        # grad routing (the kernel reads the replicas / detached masters)
        ffn0, ffn2, ln = mods
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        h22 = h2.reshape(-1, h2.shape[-1])
        r, w0h, w2h = _ffn_call(x2, h22, ffn0, ffn2, ln, acc, True)
        xo, ho, mean, rstd, hid, mask = r
        ctx.save_for_backward(h22, hid, mask, w0h, w2h, xo, mean, rstd,
                              lnw)
        ctx.ln = ln is not None
        ctx.lead = lead
        ctx.acc = _ACC_DGRAD
        if ctx.ln:
            return xo.view(*lead, -1), ho.view(*lead, -1)
        return xo.view(*lead, -1)

    @staticmethod
    def backward(ctx, *grads):
        h22, hid, mask, w0h, w2h, xo, mean, rstd, lnw = ctx.saved_tensors
        nig = ctx.needs_input_grad
        ext = _load()
        if ctx.ln:
            import fused_ln
            gxo, gho = (gr.reshape(-1, gr.shape[-1]).contiguous()
                        for gr in grads)
            # g = dL/d(out): the residual grad plus the LN's dx, summed in
            # ln_bwd's store loop (exactly _LnAddFn.backward's call); db2 =
            # its column sum (fused in the kernel at d = 192)
            r = fused_ln._load().ln_bwd(xo, gho, lnw, mean, rstd, gxo,
                                        nig[5])
            g, dlnw, dlnb = r[0], r[1], r[2]
            db2 = None
            if nig[5]:
                db2 = r[3] if len(r) > 3 else g.sum(0, dtype=torch.float32)
            dw2, _ = _wgrad_db(g, hid, nig[4], False)
        else:
            g = grads[0].reshape(-1, grads[0].shape[-1]).contiguous()
            dlnw = dlnb = None
            dw2, db2 = _wgrad_db(g, hid, nig[4], nig[5])
        # ffn.2's dgrad with the ReLU mask applied in the GEMM epilogue (the
        # unfused chain's DRELU GEMM); relu_bwd over the saved hidden if no
        # Lt plan exists at this row count
        gm = ext.dgrad_drelu(g, w2h, mask)
        if gm is None:
            gm, _ = ext.relu_bwd(_dgrad(g, w2h, ctx.acc), hid)
        dw0, db0 = _wgrad_db(gm, h22, nig[2], nig[3])
        dh2 = None
        if nig[1]:
            dh2 = _dgrad(gm, w0h, ctx.acc).view(*ctx.lead, -1)
        dx = g.view(*ctx.lead, -1) if nig[0] else None
        return dx, dh2, dw0, db0, dw2, db2, dlnw, dlnb, None, None


def fused_ffn(x, h2, ffn, ln=None):
    """x + ffn(h2) for a block FFN on the fp16 stream as ONE kernel
    (callers gate on fused_ffn_ok); with `ln` also LN(x + ffn(h2)).
    Returns out, or (out, ln_out)."""
    ffn0, ffn2 = ffn[0], ffn[2]
    x = x if x.is_contiguous() else x.contiguous()
    h2 = h2 if h2.is_contiguous() else h2.contiguous()
    acc = _ffn_acc()
    if not torch.is_grad_enabled():
        # rollout fast path: no Function dispatch, no saved tensors — the
        # kernel writes neither the hidden nor the mask
        lead = x.shape[:-1]
        r, _, _ = _ffn_call(x.reshape(-1, x.shape[-1]),
                            h2.reshape(-1, h2.shape[-1]), ffn0, ffn2, ln,
                            acc, False)
        if ln is None:
            return r[0].view(*lead, -1)
        return r[0].view(*lead, -1), r[1].view(*lead, -1)
    lnw = None if ln is None else ln.weight
    lnb = None if ln is None else ln.bias
    return _FusedFfn.apply(x, h2, ffn0.weight, ffn0.bias, ffn2.weight,
                           ffn2.bias, lnw, lnb, (ffn0, ffn2, ln), acc)
