"""CUDA-kernel vectorized Beat Arena sim, env v10 — one kernel launch per
step (P5).

rust/src/lib.rs remains THE reference implementation; this backend runs
the whole v10 step — 8 constrained dynamics substeps (SPEC-v10 §3),
gear->torque actuation, activation telemetry, swept-surface judgment,
auto-reset, the obs write — inside one custom CUDA kernel
(csrc/beat_arena_cuda.cu), one THREAD per env (P5 parallelization
decision: the Rust expression trees transcribe serially per thread, so
every accumulation keeps Rust's exact order; occupancy comes from batch
width — production runs N = 1024..8192).

Parity is pinned by tests/test_v10_parity_cuda.py: the f64 tier is
HORIZON-BOUNDED tight-tolerance (CUDA f64 sin/cos/acos differ from the
CRT libm by 1 ulp on a few %% of inputs and nvcc contracts FMAs —
SPEC-v10 Divergences §21), f32 is the bounded
well-formedness tier. Both dtypes carry a Cholesky pivot-floor safety
net with engagements counted (`pop_pd_hits`): f32 floors at 1e-8
(§20e), f64 at 1e-17 (§22 — the same floor Rust/torch f64 now carry).

State is a structure-of-arrays of torch tensors mutated in place by the
kernel; charts live in a RAGGED (CSR) device pool — every segment's
objects back to back in one flat array per field, `note_off[P+1]` /
`wall_off[P+1]` giving each segment's start (2026-09-03, audit 2.7;
the padded (P, C) layout before it spent 7.6 GB at ~7% occupancy on
the ranked pool) — and each env holds only a pool index, so reset
never copies chart data. The per-env `status` / `wall_passed` arrays
stay (N, C) / (N, W) with C = max objects per segment. The step loop
performs ZERO host<->device syncs; episode returns go to a device ring
buffer drained by pop_stats() once per PPO update.

The extension is built on first use via torch.utils.cpp_extension.load
(ninja-cached under csrc/build; -arch pinned to the current device via
jit_cache.cuda_flags()). Windows toolchain quirks
(CUDA_HOME, cl.exe, ninja discovery) are handled in _toolchain_env().

Env v11 (unilateral-contact spec, 2026-08-08): the feet stand on
unilateral frictional ground contact at the 8 sole-box corners, with the
fall termination (root y < FALL_ROOT_Y, counted by `pop_fall_hits()`).
The v10 welded-feet mode (`foot_contact=False`) was a test harness from
2026-08-16 and was REMOVED on 2026-09-27 (issue #2): the argument is gone
and passing it is a TypeError.

Limits vs BeatVecEnv: no per-step info dicts, no event stream, CUDA only.
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import math
import operator
import os
import shutil
import sys

import numpy as np
import torch

from .chart import ChartConfig, generate_chart
from .constants import (TAU_FEAT, 
    BAD_HALF, BAD_HALF_Y_LINK, FOLLOW_TIMEOUT, GOOD_HALF_Y,
    GOOD_HALF_Y_DOT, GOOD_HALF_Y_LINK, KIND_BAR, KIND_BOMB,
    KIND_CHAIN_HEAD, KIND_CHAIN_LINK, KIND_CIRCLE, N_JOINTS, NV, OBS_DIM,
    PALM_OFF, note_center,
)
from .flux_chart import FluxConfig, flux_tuple, generate_flux_chart

NUM_PAIRS = 55  # SPEC-v10 §2 live pair table (rust dynamics/collide.rs)

# Follow-through tracker ring (v7 design, see the kernel header):
# same-tick same-saber trackers share (arc, age) forever, so they
# aggregate into ONE slot with a note COUNT — exact, not approximate.
N_TRACKERS = FOLLOW_TIMEOUT + 1
# per-episode record width (kernel EP_REC): cols 0..21 tau_draws, 22..27
# env v47 ep_stats, 28..39 per-hand swing components (2026-10-04), 40..43
# wrist resets [resets L, resets R, swings L, swings R] (2026-10-04)
EP_REC = 44
EP_WRS_COL = 40
# cols 28..39: hand-major [L, R] x EP_PART_FIELDS (raw note points)
EP_PART_COL = 28
EP_PART_FIELDS = ("pre", "pre_max", "post", "post_max", "prec", "prec_max")
# bench slot-refill record (kernel BENCH_REC): the host-facing columns
# 0..18 = [good, bad, miss, score, bombs, pre, acc, post, flail_rad,
# flail_ticks, cw_score, cw_maxc, cw_breaks, chart_idx, wrev_L, wrev_R,
# ticks, wall_enters, term_flags] (cols 17/18: review B09, 2026-09-30),
# 19..22 = this level's applied torque Σ|tau| over ticks [total, spine,
# arms, legs] in N.m (2026-10-01), then scratch: col 23 = the live
# wall-entry accumulator, cols 24..27 = the live torque accumulators
BENCH_REC = 28
BENCH_COLS = 23
# term_flags bits (= lib.rs VecBeatSim::term); 0 = the level itself ended
TERM_FELL, TERM_TILT, TERM_ENVELOPE, TERM_TICKCAP = 1, 2, 4, 8

# lib.rs Z_MISS_TAIL: the episode tail covers the slowest object's trip
# from the hit plane to the WORST-CASE live head plane (env v10).
Z_MISS_TAIL = 1.5

_ext = None

# CUDA toolkits with a full nvcc known on this machine / typical installs
_CUDA_CANDIDATES = (
    r"D:\Cuda-12-6",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
)


_toolchain_done = False


def _toolchain_env():
    """Make torch.utils.cpp_extension work from a bare venv: put the
    venv's bin/Scripts dir (where pip installs ninja) on PATH on any OS
    (2026-08-08: torch probes PATH, not the venv, so a pod session that
    runs /root/venv/bin/python without activating never sees ninja), and
    on Windows additionally locate nvcc (CUDA_HOME) and cl.exe (newest
    VS2022 Hostx64/x64). Memoized."""
    global _toolchain_done
    if _toolchain_done:
        return
    _toolchain_done = True
    scripts = os.path.dirname(sys.executable)
    if shutil.which("ninja") is None:
        os.environ["PATH"] = scripts + os.pathsep + os.environ["PATH"]
    if os.name != "nt":
        return
    if shutil.which("cl") is None:
        pats = (r"C:\Program Files\Microsoft Visual Studio\2022\*"
                r"\VC\Tools\MSVC\*\bin\Hostx64\x64",
                r"C:\Program Files (x86)\Microsoft Visual Studio\2019\*"
                r"\VC\Tools\MSVC\*\bin\Hostx64\x64")
        hits = sorted(h for p in pats for h in glob.glob(p))
        if hits:
            os.environ["PATH"] = hits[-1] + os.pathsep + os.environ["PATH"]
    ch = os.environ.get("CUDA_HOME")
    if not (ch and os.path.exists(os.path.join(ch, "bin", "nvcc.exe"))):
        for cand in _CUDA_CANDIDATES:
            if os.path.exists(os.path.join(cand, "bin", "nvcc.exe")):
                os.environ["CUDA_HOME"] = ch = cand
                break
    if ch:
        os.environ["PATH"] = (os.path.join(ch, "bin") + os.pathsep
                              + os.environ["PATH"])


def _load_ext():
    """Compile-or-load the cached extension (ninja rebuilds only on source
    change; a warm cache loads in ~1 s)."""
    global _ext
    if _ext is None:
        from beat_arena_env.jit_cache import arch_list, cuda_flags, fast_load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
        bdir = os.path.join(src, "build")
        os.makedirs(bdir, exist_ok=True)
        srcs = [os.path.join(src, "beat_arena_cuda.cu")]

        def _full():
            _toolchain_env()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list())
            from torch.utils.cpp_extension import load
            return load(name="beat_arena_cuda", sources=srcs,
                        build_directory=bdir,
                        extra_cuda_cflags=cuda_flags(), verbose=True)

        _ext = fast_load("beat_arena_cuda", srcs, bdir, cuda_flags(), _full)
    return _ext


# one-pass field extraction for the pool fill (see _charts_to_tensors):
# ONE Python-level touch per note instead of eleven numpy scalar stores.
_NOTE_FIELDS = np.dtype([
    ("hit", np.float64), ("color", np.float64), ("angle", np.float64),
    ("speed", np.float64), ("any_dir", bool), ("kind", np.int64),
    ("fpre", bool), ("fpost", bool),
])
_note_get = operator.attrgetter("hit_time", "color", "angle", "speed",
                                "any_dir", "kind", "free_pre", "free_post")


def _charts_to_padded(charts, device, dtype, pad_to=None):
    """Pad a list[list[Note]] into per-field (P, C) tensors — env v10:
    episode tail = 0.5 + 1.5/min_speed (live head-plane miss boundary).
    Chart sin/cos are computed by numpy (the CRT libm — bitwise what
    Rust's set_chart_notes computes).

    pad_to forces the C dimension (callers packing in chunks pass the
    global max so chunks concatenate; every pad value is a per-field
    constant and every derived row stat is row-local, so chunked output
    is byte-identical to one whole-pool call — pinned by
    test_ranked_pool_chunked_pack).

    The fill is column-wise (2026-08-03): the notes are flattened once,
    every field is pulled in a single np.fromiter pass, and one scatter
    per field places them at the ragged (p, j) slots. The old per-note
    loop cost ~11 numpy scalar stores per note (~96 s at chart_pool
    65536, ~45M stores for the ranked segment pool); this is a pure
    re-ordering of the same copies, so the output is byte-identical."""
    P = len(charts)
    C = max(1, pad_to if pad_to is not None
            else max((len(ch) for ch in charts), default=1))
    lens = np.fromiter((len(ch) for ch in charts), np.int64, P)
    n_tot = int(lens.sum())
    flat = list(itertools.chain.from_iterable(charts))
    # ragged (p, j) -> flat (P*C) destination slots: chart p starts at
    # p*C in the destination and at cumsum(lens)[p]-lens[p] in `flat`
    idx = (np.repeat(np.arange(P, dtype=np.int64) * C
                     - (np.cumsum(lens) - lens), lens)
           + np.arange(n_tot, dtype=np.int64))
    rec = np.fromiter(map(_note_get, flat), _NOTE_FIELDS, n_tot)
    # note_center resolves explicit x/y over col/row, per note
    xy = np.fromiter((note_center(n) for n in flat), (np.float64, 2), n_tot)

    def fill(vals, dt=np.float64, pad=0.0):
        a = np.full(P * C, pad, dtype=dt)
        a[idx] = vals
        return a.reshape(P, C)

    hit = fill(rec["hit"]); x = fill(xy[:, 0]); y = fill(xy[:, 1])
    color = fill(rec["color"]); angle = fill(rec["angle"])
    speed = fill(rec["speed"], pad=1.0)
    any_dir = fill(rec["any_dir"], bool, False)
    kind = fill(rec["kind"], np.int64, 0)
    fpre = fill(rec["fpre"], bool, False)
    fpost = fill(rec["fpost"], bool, False)
    valid = fill(np.ones(n_tot, bool), bool, False)
    is_bomb = valid & (kind == KIND_BOMB)
    is_head = valid & (kind == KIND_CHAIN_HEAD)
    is_link = valid & (kind == KIND_CHAIN_LINK)
    show_dir = ~is_bomb & (~any_dir | is_link)
    good_hy = np.where(is_link, GOOD_HALF_Y_LINK,
                       np.where(any_dir, GOOD_HALF_Y_DOT, GOOD_HALF_Y))
    bad_hy = np.where(is_link, BAD_HALF_Y_LINK, BAD_HALF)
    rotates = is_link | ~any_dir
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    rot_c = np.where(rotates, -sin_a, 1.0)
    rot_s = np.where(rotates, cos_a, 0.0)
    min_speed = np.where(valid, speed, np.inf).min(1)
    finite = np.isfinite(min_speed)
    # the kernel's high-water bound needs a FINITE min_speed; 1.0 for
    # empty charts is inert (no valid notes to scan)
    min_speed = np.where(finite, min_speed, 1.0)
    tail = np.where(finite, 0.5 + Z_MISS_TAIL / min_speed, 0.5)
    t = lambda a, dt=dtype: torch.as_tensor(a, device=device, dtype=dt)
    return {
        "hit": t(hit), "speed": t(speed),
        "center": torch.stack([t(x), t(y)], -1),
        "dvec": torch.stack([t(cos_a), t(sin_a)], -1),
        "rot": torch.stack([t(rot_c), t(rot_s)], -1),
        "is_right": t(color, torch.bool),
        "any_dir": t(any_dir, torch.bool), "valid": t(valid, torch.bool),
        "is_bomb": t(is_bomb, torch.bool), "is_head": t(is_head, torch.bool),
        "is_link": t(is_link, torch.bool),
        "show_dir": t(show_dir, torch.bool),
        "free_pre": t(fpre, torch.bool), "free_post": t(fpost, torch.bool),
        "good_hy": t(good_hy), "bad_hy": t(bad_hy),
        "n_notes": t(valid.sum(1), torch.long),
        "n_scoreable": t((valid & ~is_bomb).sum(1), torch.long),
        "last_hit": t(np.where(valid.any(1), (hit * valid).max(1), 0.0)),
        "tail": t(tail),
        "min_speed": t(min_speed),
    }


def _flux_to_padded(charts, device, dtype):
    """Pack a list[list[FluxObj | flux_tuple]] (Flux game mode, SPEC.md
    §Flux) into the standard per-field (P, C) pool dict plus `len` /
    `sweet`. Validation and per-object values mirror lib.rs flux.rs
    set_flux_chart: circles carry colour, len/sweet 0; bars carry
    colour 0 and their rotation in `rot` as (cos θ, sin θ) — the pair
    the kernel feeds note_frame (lam = 0) and derives the axis
    u = (-sin θ, cos θ) from. Every object is any_dir, never shows a
    direction, and is neither bomb, chain head nor link; a Flux object
    is a BAR iff len > 0. sin/cos come from numpy (the CRT libm — what
    Rust's set_flux_chart computes)."""
    P = len(charts)
    tups = [[o if isinstance(o, tuple) else flux_tuple(o) for o in ch]
            for ch in charts]
    for ch in tups:
        assert all(a[0] <= b[0] for a, b in zip(ch, ch[1:])), \
            "flux chart must be sorted by hit_time"
        for (_t, _x, _y, kind, color, _rot, speed, length, sweet) in ch:
            assert kind in (KIND_CIRCLE, KIND_BAR), \
                "flux chart kinds are 4 (circle) / 5 (bar)"
            assert color in (0, 1), "colour must be 0 (left) or 1 (right)"
            assert speed > 0.0, "speed must be positive"
            if kind == KIND_BAR:
                assert length > 0.0, "bar length must be positive"
                assert -1.0 <= sweet <= 1.0, \
                    "sweet-spot offset must be in [-1, 1]"
    C = max(1, max((len(ch) for ch in tups), default=1))
    lens = np.fromiter((len(ch) for ch in tups), np.int64, P)
    n_tot = int(lens.sum())
    idx = (np.repeat(np.arange(P, dtype=np.int64) * C
                     - (np.cumsum(lens) - lens), lens)
           + np.arange(n_tot, dtype=np.int64))
    rec = (np.array(list(itertools.chain.from_iterable(tups)), np.float64)
           .reshape(n_tot, 9) if n_tot else np.zeros((0, 9), np.float64))

    def fill(vals, dt=np.float64, pad=0.0):
        a = np.full(P * C, pad, dtype=dt)
        a[idx] = vals
        return a.reshape(P, C)

    kind = fill(rec[:, 3].astype(np.int64), np.int64, 0)
    valid = fill(np.ones(n_tot, bool), bool, False)
    is_bar = valid & (kind == KIND_BAR)
    hit = fill(rec[:, 0]); x = fill(rec[:, 1]); y = fill(rec[:, 2])
    color = np.where(is_bar, 0, fill(rec[:, 4].astype(np.int64),
                                     np.int64, 0))
    rot = fill(rec[:, 5])
    speed = fill(rec[:, 6], pad=1.0)
    length = np.where(is_bar, fill(rec[:, 7]), 0.0)
    sweet = np.where(is_bar, fill(rec[:, 8]), 0.0)
    sin_a = np.where(is_bar, np.sin(rot), 0.0)
    cos_a = np.where(is_bar, np.cos(rot), 1.0)
    no = np.zeros((P, C), bool)
    min_speed = np.where(valid, speed, np.inf).min(1)
    finite = np.isfinite(min_speed)
    min_speed = np.where(finite, min_speed, 1.0)
    tail = np.where(finite, 0.5 + Z_MISS_TAIL / min_speed, 0.5)
    t = lambda a, dt=dtype: torch.as_tensor(a, device=device, dtype=dt)
    return {
        "hit": t(hit), "speed": t(speed),
        "center": torch.stack([t(x), t(y)], -1),
        "dvec": torch.stack([t(cos_a), t(sin_a)], -1),
        "rot": torch.stack([t(cos_a), t(sin_a)], -1),
        "is_right": t(color, torch.bool),
        "any_dir": t(valid, torch.bool), "valid": t(valid, torch.bool),
        "is_bomb": t(no, torch.bool), "is_head": t(no, torch.bool),
        "is_link": t(no, torch.bool), "show_dir": t(no, torch.bool),
        "free_pre": t(no, torch.bool), "free_post": t(no, torch.bool),
        "good_hy": t(np.full((P, C), GOOD_HALF_Y_DOT)),
        "bad_hy": t(np.full((P, C), BAD_HALF)),
        "len": t(length), "sweet": t(sweet),
        "n_notes": t(valid.sum(1), torch.long),
        "n_scoreable": t(valid.sum(1), torch.long),
        "last_hit": t(np.where(valid.any(1), (hit * valid).max(1), 0.0)),
        "tail": t(tail),
        "min_speed": t(min_speed),
    }


def _walls_to_padded(walls, device, dtype, pad_to=None):
    """Pad a list[list[wall_tuple]] into per-field (P, W) tensors (env
    v17, spec §12). wall_tuple order: (t0, cx, cy, width, height,
    length, speed). Same chunk-stable contract as _charts_to_tensors:
    pad values are per-field constants (speed pads 1.0 to stay
    division-safe), so chunked packing concatenates byte-identically."""
    P = len(walls)
    W = max(1, pad_to if pad_to is not None
            else max((len(ws) for ws in walls), default=1))
    lens = np.fromiter((len(ws) for ws in walls), np.int64, P)
    n_tot = int(lens.sum())
    flat = list(itertools.chain.from_iterable(walls))
    idx = (np.repeat(np.arange(P, dtype=np.int64) * W
                     - (np.cumsum(lens) - lens), lens)
           + np.arange(n_tot, dtype=np.int64))
    rec = np.array(flat, np.float64).reshape(n_tot, 7) if n_tot else \
        np.zeros((0, 7), np.float64)

    def fill(col, pad=0.0):
        a = np.full(P * W, pad, dtype=np.float64)
        a[idx] = rec[:, col]
        return a.reshape(P, W)

    valid = np.zeros(P * W, bool)
    valid[idx] = True
    valid = valid.reshape(P, W)
    t = lambda a, dt=dtype: torch.as_tensor(a, device=device, dtype=dt)
    return {
        "wall_t": t(fill(0)), "wall_cx": t(fill(1)), "wall_cy": t(fill(2)),
        "wall_w": t(fill(3)), "wall_h": t(fill(4)), "wall_len": t(fill(5)),
        "wall_speed": t(fill(6, pad=1.0)),
        "wall_valid": t(valid, torch.bool),
        "n_walls": t(lens, torch.long),
    }


# Ragged (CSR) pool layout (2026-09-03, audit 2.7). Per-OBJECT fields
# are flat: notes (T,) / (T, 2) with T = total objects over the pool,
# walls (Tw,) / (Tw, 2); `note_off` / `wall_off` are (P + 1,) int64
# exclusive prefix sums (segment p's objects are [off[p], off[p + 1])).
# Per-SEGMENT fields (n_notes, n_scoreable, last_hit, tail, min_speed,
# n_walls) are (P,) as before. The kernel reads object i of chart cc at
# note_off[cc] + i where it read cc * C + i; scan bounds (n_notes),
# values and arithmetic are untouched, so the layouts are bit-identical
# in every kernel output by construction.
NOTE_OBJ_FIELDS = ("hit", "speed", "center", "dvec", "rot", "is_right",
                   "any_dir", "valid", "is_bomb", "is_head", "is_link",
                   "show_dir", "free_pre", "free_post", "good_hy", "bad_hy")
# Flux-only per-object fields (bar length / sweet offset): present in
# Flux pools only — the layout helpers below skip absent keys, so
# standard pools and their caches are unchanged (the env fills zeros)
FLUX_OBJ_FIELDS = ("len", "sweet")
WALL_OBJ_FIELDS = ("wall_t", "wall_cx", "wall_cy", "wall_w", "wall_h",
                   "wall_len", "wall_speed", "wall_valid")


def _offsets(counts):
    off = torch.zeros(counts.shape[0] + 1, dtype=torch.int64,
                      device=counts.device)
    off[1:] = torch.cumsum(counts, 0)
    return off


def _flatten_block(pool, fields, valid_key, off_key, count_key):
    if off_key in pool or valid_key not in pool:
        return               # already ragged (or block absent)
    mask = pool[valid_key]
    for k in fields:
        if k in pool:
            pool[k] = pool[k][mask].contiguous()
    pool[off_key] = _offsets(pool[count_key])


def pad_to_ragged(pool):
    """Padded (P, C) / (P, W) pool dict -> ragged, IN PLACE (returns the
    dict). The valid mask selects each segment's objects in (p, j) order,
    which is exactly CSR order; pads are dropped. Idempotent."""
    _flatten_block(pool, NOTE_OBJ_FIELDS + FLUX_OBJ_FIELDS, "valid",
                   "note_off", "n_notes")
    _flatten_block(pool, WALL_OBJ_FIELDS, "wall_valid", "wall_off",
                   "n_walls")
    return pool


def _unflatten_block(pool, fields, off_key, count_key, empty):
    if off_key not in pool:
        return
    off, n = pool[off_key], pool[count_key]
    P = n.shape[0]
    C = max(1, int(n.max()) if P else 1)
    j = torch.arange(C, device=n.device)
    src = off[:-1, None] + j[None, :]                # (P, C) flat index
    hit = j[None, :] < n[:, None]
    src = torch.where(hit, src, torch.zeros_like(src))
    for k in fields:
        if k not in pool:
            continue
        a = pool[k]
        pad = empty[k][0, 0]                          # per-field pad value
        out = pad.expand(P, C, *a.shape[1:]).clone()
        m = hit.reshape(P, C, *([1] * (a.dim() - 1)))
        gathered = a[src.reshape(-1)].reshape(P, C, *a.shape[1:]) \
            if a.shape[0] else out
        pool[k] = torch.where(m, gathered, out)
    del pool[off_key]


def ragged_to_padded(pool):
    """Inverse of pad_to_ragged (test / cache-migration tool, not a
    runtime path): (P, C) with C = max objects, pads taken from what the
    padded builders emit for an empty chart, so the round trip through
    pad_to_ragged reproduces the padded builders' output exactly."""
    dt = pool["hit"].dtype
    dev = pool["hit"].device
    empty = _charts_to_padded([[]], dev, dt)
    for k in ("len", "sweet"):          # Flux-only fields pad with 0
        empty[k] = torch.zeros((1, 1), device=dev, dtype=dt)
    _unflatten_block(pool, NOTE_OBJ_FIELDS + FLUX_OBJ_FIELDS, "note_off",
                     "n_notes", empty)
    _unflatten_block(pool, WALL_OBJ_FIELDS, "wall_off", "n_walls",
                     _walls_to_padded([[]], dev, dt))
    return pool


def cat_pools(parts):
    """Concatenate ragged pool dicts along the segment axis (the chunked
    ranked-pool build): per-object fields and per-segment fields cat,
    offsets rebase by the running object total. Byte-identical to one
    whole-list build (test_ranked_pool_chunked_pack)."""
    out = {}
    for k in parts[0]:
        if k in ("note_off", "wall_off"):
            base, chunks = 0, [parts[0][k][:1]]
            for p in parts:
                chunks.append(p[k][1:] + base)
                base += int(p[k][-1])
            out[k] = torch.cat(chunks)
        else:
            out[k] = torch.cat([p[k] for p in parts])
    return out


def _charts_to_tensors(charts, device, dtype):
    """list[list[Note]] -> ragged pool dict (see pad_to_ragged). Built
    through the padded path so every value is the byte the (P, C)
    builder produced (its numpy sin/cos over the padded block included);
    only the layout changes."""
    return pad_to_ragged(_charts_to_padded(charts, device, dtype))


def _walls_to_tensors(walls, device, dtype):
    """list[list[wall_tuple]] -> ragged wall block (see pad_to_ragged)."""
    return pad_to_ragged(_walls_to_padded(walls, device, dtype))


def _flux_to_tensors(charts, device, dtype):
    """list[list[FluxObj | flux_tuple]] -> ragged Flux pool dict."""
    return pad_to_ragged(_flux_to_padded(charts, device, dtype))


def _flux_generator_version():
    """_generator_version's twin for the Flux generator (flux_chart.py):
    the config, the generator (closures ride along), the object record,
    the shared lattice tunables it imports and the grid geometry."""
    import inspect

    from . import chart
    from . import constants as K
    from . import flux_chart as F
    h = hashlib.sha256()
    for sym in (F.FluxConfig, F.generate_flux_chart, F._rect, F.bar_axis,
                F.FluxObj, F.flux_tuple, K.col_x):
        h.update(inspect.getsource(sym).encode())
    h.update(repr((chart.TICK_DT, chart.TICK_JITTER, chart.SPEED_DEV,
                   F.LEFT, F.RIGHT, K.ROW_Y, K.CELL_X, K.KIND_CIRCLE,
                   K.KIND_BAR)).encode())
    return h.hexdigest()[:16]


def _generator_version():
    """Hash of the code the CACHED POOL CONTENT actually depends on —
    the bench's parser_version() pattern (bench_scoresaber.py). Hashing
    chart.py whole (pre-2026-08-03) meant every unrelated edit there —
    a docstring, `chain_links`, the parser-only helpers — retired the
    whole 35 GB pool cache and forced a cold ~96 s rebuild per pool.

    Covered: the config class, the generator itself (its `spawn` /
    `walk_speed` closures ride along in getsource), the position
    helpers it calls, the Note record whose fields the pool reads, the
    module-level generator tunables (getsource of a function does NOT
    see them), and the constants.py geometry that resolves a note to a
    world position."""
    import inspect

    from . import chart
    from . import constants as K
    h = hashlib.sha256()
    for sym in (chart.ChartConfig, chart.generate_chart, chart._pos_ranges,
                chart._nearest_cell, chart.Note, K.note_center, K.col_x):
        h.update(inspect.getsource(sym).encode())
    h.update(repr((
        chart.TICK_DT, chart.TICK_JITTER, chart.ANGLE_SIGMA, chart.SPEED_DEV,
        chart.EXTRA_PATH, chart.EXTRA_ROT, chart.EXTRA_PERP, chart.EXTRA_DZ,
        chart.POS_PAD, chart.N_COLS, chart.N_ROWS, chart.LEFT, chart.RIGHT,
        chart._BOMB_PATH, K.ROW_Y, K.CELL_X, K.KIND_NOTE, K.KIND_BOMB,
    )).encode())
    return h.hexdigest()[:16]


def _pool_cache_path(cfg, seed, pool_size, dtype, game="standard"):
    """Cache file for a procedural chart pool. Keyed on the FULL chart
    config (it determines content) + seed + pool size + dtype +
    _generator_version() + a pool LAYOUT version.

    Layout v11 (2026-08-03): the vectorized _charts_to_tensors landed
    (byte-identical, but a clean break beats any old-key/new-key
    aliasing) and the key narrowed from all-of-chart.py to the
    generator symbols. v10 was the 0.5 + 1.5/min_speed tail rule.

    Layout v12 (2026-09-03, audit 2.7): ragged (CSR) pool — the npz
    holds flat per-object arrays + note_off / wall_off; v11 caches were
    (P, C)-padded and are not reused."""
    if game == "flux":
        # Flux pools key on the Flux generator + a game tag and live
        # under their own file prefix — never aliased with standard
        blob = repr(("v12-flux", _flux_generator_version(), repr(cfg),
                     seed, pool_size, str(dtype))).encode()
        prefix = "pool_flux"
    else:
        blob = repr(("v12", _generator_version(), repr(cfg), seed,
                     pool_size, str(dtype))).encode()
        prefix = "pool"
    key = hashlib.sha1(blob).hexdigest()[:16]
    cdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "pool_cache")
    os.makedirs(cdir, exist_ok=True)
    return os.path.join(cdir, f"{prefix}_{pool_size}_{key}.npz")


def _load_pool_cache(path, device):
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return pad_to_ragged(
        {k: torch.as_tensor(data[k], device=device) for k in data.files})


def _fix_pool_tail(pool):
    """Recompute the v10 episode tail from min_speed IN PLACE — covers
    pools cached before the v10 tail rule (ranked pools are keyed on the
    parser version only). Empty-chart pools keep tail 0.5 via last_hit=0
    + the no-notes done branch, so the blanket rule is safe."""
    ms = pool["min_speed"]
    pool["tail"] = torch.where(
        torch.isfinite(ms) & (ms > 0),
        0.5 + Z_MISS_TAIL / ms, torch.full_like(ms, 0.5))
    return pool


def _max_count(counts):
    return max(1, int(counts.max()) if counts.numel() else 1)


class CudaBeatVecEnv:
    """BeatVecEnv-alike backed by the v10 CUDA kernel. step() takes an
    (N, 29) long gear-id action tensor (SPEC-v10 §1.4 joint order) and
    returns (obs, reward, done, []) — all tensors on the CUDA device,
    infos always empty (use pop_stats() once per update)."""

    def __init__(self, num_envs, colors="both", chart_len_s=20.0,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed_min=None, note_speed_max=None,
                 note_prob=0.25, extra_prob=0.125, dot_prob=0.1,
                 bomb_prob=0.125, seed=0,
                 charts=None, chart_pool=1024, max_steps=None,
                 pool=None, gears=3,
                 device="cuda", dtype=torch.float32, auto_reset=True,
                 want_contacts=False,
                 fall_const=5.0, walls=None,
                 push_prob=0.0, push_vmax=0.0, start_jitter=0.0,
                 foot_z_coef=0.0, foot_x_coef=0.0, foot_hinge_r=0.0,
                 foot_shape_coef=0.0, energy_coef=0.0, limit_coef=0.0,
                 limit_arms=False,
                 wall_despawn=False,
                 start_pose_jitter=0.0, crouch_start_prob=0.0,
                 self_contact_coef=0.0, tau_rand=0.0,
                 combo_walk=False, bench_refill=False, ep_stats=False,
                 tau_center=None, z_asinh=8.0, env_order=False,
                 game="standard", circle_prob=0.3, bar_prob=0.15,
                 flux_charts=None):
        # game (SPEC.md §Flux): "standard" (the default; every existing
        # caller) or "flux" — circles + bars, palms instead of sabers.
        # A compile-time kernel switch, baked here.
        # Flux draws procedural charts from flux_chart.generate_flux_chart
        # (circle_prob / bar_prob + the shared chart kwargs), or plays
        # flux_charts= (list[list[FluxObj | flux_tuple]], the charts=
        # test-mode twin); ranked pools, charts= and walls are standard-
        # only. note_speed_min/max None = the mode's default (standard
        # 2 / 8 m/s, Flux 10 / 18 — FluxConfig's real-NJS range).
        assert game in ("standard", "flux"), game
        self.game = game
        self.flux = game == "flux"
        if self.flux:
            assert pool is None and charts is None and walls is None, \
                "flux: use flux_charts= (no ranked pool / charts= / walls)"
            _fd = FluxConfig()
            note_speed_min = (_fd.note_speed_min if note_speed_min is None
                              else note_speed_min)
            note_speed_max = (_fd.note_speed_max if note_speed_max is None
                              else note_speed_max)
        else:
            assert flux_charts is None, "flux_charts= needs game='flux'"
            note_speed_min = 2.0 if note_speed_min is None else note_speed_min
            note_speed_max = 8.0 if note_speed_max is None else note_speed_max
        # walls (env v17, charts= test mode only): list[list[wall_tuple]]
        # aligned with `charts` — chart i plays with walls[i]. Ranked
        # pools carry their wall tensors themselves.
        self.num_envs = num_envs
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CudaBeatVecEnv requires a CUDA device")
        self.dtype = dtype
        self.auto_reset = auto_reset  # False: terminal envs freeze (tests)
        # contacts() introspection (per-pair impulses + points) costs a
        # 55-wide zeroing plus uncoalesced RMWs per active contact row every
        # tick and feeds NOTHING in the kernel (the obs contact bits come
        # from contact_arm, which is always maintained). Off for training;
        # the parity harnesses pass True. Baked here, never mutated =>
        # constant per instance => safe under CUDA-graph capture.
        self.want_contacts = bool(want_contacts)
        # env v18 (2026-08-15, sole mode): flat -fall_const terminal
        # reward on a fall (kills the sparse-state fall-indifference
        # exploit; the legacy remaining-notes charge is REMOVED).
        self.fall_const = float(fall_const)
        # env v20 (2026-08-16): per-tick |foot_z - PLANE_Z0| penalty
        # (both engines) + training-only start-pose leg jitter (radians;
        # kernel reset_env samples the 12 leg joints from per-env U(-1,1)
        # draws and drops the root so no sole corner clips the ground).
        self.foot_z_coef = float(foot_z_coef)
        # env v21: the X twin — each foot priced against its own
        # wide-stance x (constants.FOOT_X0)
        self.foot_x_coef = float(foot_x_coef)
        # env v22: hinged rent radius (0 = v21 pure-linear) + potential-
        # based delta-shaping coefficient (see lib.rs field comments)
        self.foot_hinge_r = float(foot_hinge_r)
        self.foot_shape_coef = float(foot_shape_coef)
        self.energy_coef = float(energy_coef)
        self.limit_coef = float(limit_coef)
        self.limit_arms = bool(limit_arms)
        # env v37: bomb-like wall despawn on charge (training-only aid;
        # bench/replays leave it off for faithful persistent walls)
        self.wall_despawn = bool(wall_despawn)
        # env v48: asinh z lane knee C (m), the only z encoding; default
        # 8 = the recipe (the rust twin reads BEAT_Z_ASINH, unset = 8)
        self.z_asinh = float(z_asinh)
        if not self.z_asinh > 0:
            raise ValueError(f"z_asinh must be > 0 (metres), got {z_asinh}")
        # env v41 (2026-09-12): price on the tick-summed self-collision
        # impulse (N.s); 0 = off. Telemetry accumulates regardless.
        self.self_contact_coef = float(self_contact_coef)
        self.start_pose_jitter = float(start_pose_jitter)
        # env v38 crouch-rich starts (training-only; see kernel table)
        self.crouch_start_prob = float(crouch_start_prob)

        # env v18 perturbation training (push-recovery lit): with
        # probability push_prob per (env, tick) add a horizontal root
        # velocity kick, |dv| ~ U(0, push_vmax) m/s, direction uniform.
        # start_jitter: same kick for every env that resets (RSI-flavored
        # start-state variety), |dv| ~ U(0, start_jitter) — drawn into
        # _start_kick before each launch and installed by the kernel's
        # reset_env BEFORE the reset obs is written (review B10; it used
        # to land after the launch, so every first obs read root
        # velocity 0). Applies to auto-resets, reset() and bench
        # refills alike. The push kick is a Python-side v[:, 0/2]
        # mutation before the launch — an impulse by construction.
        # Both draw from the default CUDA RNG (not self._gen). Training
        # only: bench/replay use the Rust sim, which has neither path.
        self.push_prob = float(push_prob)
        self.push_vmax = float(push_vmax)
        self.start_jitter = float(start_jitter)
        assert gears in (3, 5, 7), f"gears must be 3, 5, or 7, got {gears}"
        self.gears = int(gears)
        # per-joint action-id counts (constants.joint_gear_counts: the
        # build's JOINT_EXTRA_GEARS on top of the base table; raises if
        # they do not fit) — the kernel's JOINT_EXTRA_* mirror decodes them
        from .constants import joint_gear_counts
        self.joint_gears = joint_gear_counts(self.gears)
        # pool=: prebuilt tensor dict (ranked_pool.py; ragged, or padded
        # (P, C) from an older builder — converted here) — unlike
        # charts=, resets keep the normal RANDOM chart draw (charts= is
        # the deterministic-sequential test mode)
        self._test_charts = ((charts is not None or flux_charts is not None)
                             and pool is None)
        if self.flux and flux_charts is not None:
            self.pool = _flux_to_tensors(flux_charts, self.device, dtype)
        elif self.flux:
            fcfg = FluxConfig(
                chart_len_s=chart_len_s, cols=tuple(cols), rows=tuple(rows),
                note_speed_min=note_speed_min,
                note_speed_max=note_speed_max,
                circle_prob=circle_prob, bar_prob=bar_prob)
            cpath = _pool_cache_path(fcfg, seed, chart_pool, dtype,
                                     game="flux")
            self.pool = _load_pool_cache(cpath, self.device)
            if self.pool is None:
                rng = np.random.default_rng(seed)
                fch = [generate_flux_chart(fcfg, rng)
                       for _ in range(chart_pool)]
                self.pool = _flux_to_tensors(fch, self.device, dtype)
                np.savez(cpath, **{k: v.cpu().numpy()
                                   for k, v in self.pool.items()})
                print(f"[pool-cache] generated {chart_pool} flux charts -> "
                      f"{os.path.basename(cpath)}", flush=True)
            else:
                print(f"[pool-cache] hit: {os.path.basename(cpath)}",
                      flush=True)
        elif pool is not None:
            self.pool = pad_to_ragged(_fix_pool_tail(
                {k: v.to(self.device, dtype if v.is_floating_point() else
                         v.dtype) for k, v in pool.items()}))
        elif charts is None:
            cfg = ChartConfig(
                chart_len_s=chart_len_s, colors=colors,
                cols=tuple(cols), rows=tuple(rows),
                note_speed_min=note_speed_min,
                note_speed_max=note_speed_max,
                note_prob=note_prob, extra_prob=extra_prob,
                dot_prob=dot_prob, bomb_prob=bomb_prob)
            cpath = _pool_cache_path(cfg, seed, chart_pool, dtype)
            self.pool = _load_pool_cache(cpath, self.device)
            if self.pool is None:
                rng = np.random.default_rng(seed)
                charts = [generate_chart(cfg, rng)
                          for _ in range(chart_pool)]
                self.pool = _charts_to_tensors(charts, self.device, dtype)
                np.savez(cpath, **{k: v.cpu().numpy()
                                   for k, v in self.pool.items()})
                print(f"[pool-cache] generated {chart_pool} charts -> "
                      f"{os.path.basename(cpath)}", flush=True)
            else:
                print(f"[pool-cache] hit: {os.path.basename(cpath)}",
                      flush=True)
        else:
            self.pool = _charts_to_tensors(charts, self.device, dtype)
            if walls is not None:
                assert len(walls) == len(charts)
                self.pool.update(
                    _walls_to_tensors(walls, self.device, dtype))
        # C / W: the max object count over the pool — the per-env
        # status / wall_passed stride and the scan bound, exactly the
        # padded width the (P, C) layout had (one host read, init only)
        self.pool_size = int(self.pool["n_notes"].shape[0])
        self.C = _max_count(self.pool["n_notes"])
        # env v17: pools without wall tensors (procedural generator,
        # charts= test mode, pre-v17 caches) get an all-empty wall
        # block — every wall lane inert, kernel behavior = v16
        if "wall_t" not in self.pool:
            self.pool.update(_walls_to_tensors(
                [[] for _ in range(self.pool_size)], self.device, dtype))
        self.W = _max_count(self.pool["n_walls"])
        # Flux bar fields: standard pools (procedural, ranked, caches)
        # never carry them — zeros, which the standard kernel never reads
        for _k in ("len", "sweet"):
            if _k not in self.pool:
                self.pool[_k] = torch.zeros_like(self.pool["hit"])
        if max_steps is None:
            max_steps = int(chart_len_s * 30) + 90
        self.max_steps = int(max_steps)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        self._episode = 0  # test-mode sequential chart assignment counter
        self._ext = _load_ext()

        N, C = num_envs, self.C
        dv = self.device
        z = lambda *s, dt=dtype: torch.zeros(*s, device=dv, dtype=dt)
        # v10 dynamics state
        self.p = z(N, 3); self.quat = z(N, 4)
        self.q = z(N, N_JOINTS); self.v = z(N, NV)
        # FK caches
        self.wrist = z(N, 2, 3); self.blade = z(N, 2, 3)
        self.wrist_vel = z(N, 2, 3)
        self.head = z(N, 3); self.shoulder_z = z(N, 2)
        # swing chain
        self.swing_acc = z(N, 2)
        self.axis = z(N, 2, 3)
        self.axis_valid = z(N, 2, dt=torch.bool)
        # follow-through tracker ring (aggregated)
        self.tr_acc = z(N, 2, N_TRACKERS)
        self.tr_age = z(N, 2, N_TRACKERS, dt=torch.int32)
        self.tr_cnt = z(N, 2, N_TRACKERS, dt=torch.int32)
        self.tr_active = z(N, 2, N_TRACKERS, dt=torch.bool)
        self.tr_ptr = z(N, 2, dt=torch.int32)
        self.tick = z(N, dt=torch.int32)
        # 0 pending, 1 good, 2 bad/bomb-hit, 3 missed, 4 bomb passed clean
        self.status = z(N, C, dt=torch.uint8)
        # env v45 per-hand previous-note record [L, R] by note colour:
        # chart index of the last resolved note of that colour + valid
        self.prev_idx = z(N, 2, dt=torch.int32)
        self.prev_valid = z(N, 2, dt=torch.uint8)
        # env v46: per-env axis torque-limit scales (N, 16; 1.0 = the
        # v45 body), the U(-1,1) draw buffer the kernel consumes at
        # reset when tau_rand > 0 (refilled per launch from self._gen),
        # and the per-episode outcome record (N, 22) + done flags the
        # trainer pops for tau_draws (see pop_episodes)
        self.tau_rand = float(tau_rand)
        assert 0.0 <= self.tau_rand < 1.0, "tau_rand in [0, 1)"
        self.tau_scale = torch.ones((N, TAU_FEAT), device=dv, dtype=dtype)
        self._tau_jit = z(N, TAU_FEAT)
        # env v47b: the randomization's CENTER per axis (16 floats; 1.0 =
        # the TAU_MAX table). Draws are center * (1 + tau_rand * U(-1,1));
        # obs lanes stay (s - 1) / 0.25 relative to TAU_MAX, so the policy
        # always reads the absolute body. The default-body search
        # (ep_stats + tau_fit.py) moves this between trunks.
        tc = [1.0] * TAU_FEAT if tau_center is None else [float(x) for x in tau_center]
        assert len(tc) == TAU_FEAT and all(x > 0 for x in tc), "tau_center: 16 positive floats"
        self.tau_center = torch.tensor(tc, device=dv, dtype=dtype).view(1, TAU_FEAT)
        if self.tau_rand == 0.0 and tau_center is not None:
            self.tau_scale.copy_(self.tau_center.expand(N, TAU_FEAT))
        # env v47 (2026-09-19, user): --ep-stats — the ScoreSaber combo
        # walk runs in TRAINING envs too and every episode's outcome row
        # carries the multiplied score, max combo, breaks, completed
        # (level end reached, no fall), bad cuts, misses (cols 22..28);
        # the trainer turns them into per-map accuracy / PP. Auto-reset
        # is unchanged (the bench's done-freeze is gated off).
        self.ep_stats = bool(ep_stats)
        if self.ep_stats:
            combo_walk = True
        self.ep_rec = z(N, EP_REC)
        self.ep_flag = z(N, dt=torch.uint8)
        # bench combo walk (2026-09-16): the ScoreSaber multiplied-score
        # walk (lib.rs ComboWalk) run IN the kernel at the event sites,
        # plus the "finished level stays finished" freeze — the CUDA
        # bench's counters. Off (training): the tensors are untouched
        # and the note-multiplier plane is a 1x1 dummy.
        self.combo_walk = bool(combo_walk)
        # bench slot refill (2026-09-16): the kernel snapshots a finished
        # level into bench_rec and resets the slot onto _bench_cand[e]
        # (>= 0) in the same tick, or freezes it (< 0). Host bookkeeping:
        # set_next_chart / pop_bench_rows. Requires combo_walk.
        self.bench_refill = bool(bench_refill)
        assert not self.bench_refill or self.combo_walk, \
            "bench_refill needs combo_walk=True"
        self._bench_cand = None
        self.cw_mult = z(N, dt=torch.int32)
        self.cw_prog = z(N, dt=torch.int32)
        self.cw_combo = z(N, dt=torch.int32)
        self.cw_maxc = z(N, dt=torch.int32)
        self.cw_breaks = z(N, dt=torch.int32)
        self.cw_score = z(N)
        self.cw_note_mult = (z(N, C, dt=torch.uint8) if self.combo_walk
                             else z(1, 1, dt=torch.uint8))
        self.tr_msum = z(N, 2, N_TRACKERS, dt=torch.int32)
        self.bench_rec = z(N, BENCH_REC)   # see BENCH_REC / BENCH_COLS
        self.bench_flag = z(N, dt=torch.uint8)
        # Flux: palm-centre finite-difference velocity (kernel-written in
        # Flux mode only; the start kick at reset, zeroed by set_state)
        self.palm_vel = z(N, 2, 3)
        # env v49: blade-tip finite-difference velocity (kernel-written
        # every tick; zeroed at reset / set_state; obs lanes 624..630)
        self.tip_vel = z(N, 2, 3)
        self.first_pending = z(N, dt=torch.int32)
        self.ep_return = z(N)
        self.good = z(N, dt=torch.int32); self.bad = z(N, dt=torch.int32)
        self.miss = z(N, dt=torch.int32); self.bombs = z(N, dt=torch.int32)
        self.score = z(N)
        self.score_pre = z(N)
        self.score_acc = z(N)
        self.score_post = z(N)
        # contact aggregation (obs bits / parity introspection)
        self.contact_imp = z(N, NUM_PAIRS)
        self.contact_pt = z(N, NUM_PAIRS, 3)
        self.contact_arm = z(N, 2, dt=torch.bool)
        # Cholesky pivot-floor engagements (§20e f32 / §22 f64)
        self._pd_hits = z(1, dt=torch.int32)
        self._pd_popped = 0
        # PRE-integration root-envelope engagements (§23), counted per
        # (env, substep); an env that engages any of them terminates that
        # tick with a zeroed reward (the C backstop)
        self._env_hits = z(1, dt=torch.int32)
        self._env_popped = 0
        # env v11 FALL terminations (unilateral-contact spec §4), counted
        # per (env, tick)
        self._fall_hits = z(1, dt=torch.int32)
        self._fall_popped = 0
        # TILT terminations (2026-08-20): chest up-axis > 40 deg
        # sagittal / > 40 deg lateral, fall-priced
        self._tilt_hits = z(1, dt=torch.int32)
        self._tilt_popped = 0
        # FOOT-LIFT penalty (2026-08-20): lift-event pop counter +
        # per-foot grounded-last-tick edge state. Starts False
        # (UNARMED): a foot that begins an episode in the air is never
        # charged — the edge arms on the first grounded tick (kernel
        # re-falses it on every reset).
        self._lift_hits = z(2, dt=torch.int32)   # per-foot [L, R]
        self._lift_popped = [0, 0]
        self.foot_down_prev = z(N, 2, dt=torch.bool)
        # env v19 flail telemetry (per env, kernel-accrued): blade arc
        # on note-free ticks by hands with no live follow-through, and
        # the note-free tick count. Popped + zeroed by pop_flail;
        # deliberately NOT zeroed by reset_env so episode boundaries
        # don't drop data between pops. CUDA-backend-only telemetry —
        # the Rust reference carries the obs change but not this.
        self.flail_rad = z(N, 2)   # per-hand [L, R] since 2026-08-20
        self.flail_ticks = z(N, dt=torch.int32)
        # wrist-bend reversal telemetry (2026-09-22, user): per env x
        # hand — count since the last pop (kernel peak detection with a
        # 60 deg hysteresis on the bend joints; flail lifecycle), the
        # running extremes and the detector state (reset per
        # episode by the kernel's reset_env). Popped by pop_wrist_rev.
        self.wrev = z(N, 2, dt=torch.int32)
        self.wrev_ext = z(N, 2, 2)          # running [lo, hi] per hand
        self.wrev_dir = z(N, 2, dt=torch.int32)
        # wrist resets (2026-10-04; lib.rs WristReset): per env x hand
        # [dir, reversal since hit, last swing sign, resets, swings] this
        # episode (zeroed by reset_env), the 30 deg detector's extremes,
        # and the per-update [resets, swings] accumulator
        # (pop_wrist_resets)
        self.wrs = z(N, 2, 5, dt=torch.int32)
        self.wrs_ext = z(N, 2, 2)
        self.wrs_acc = z(N, 2, 2, dt=torch.int32)
        # env v34 limit-rest telemetry (per env, kernel-accrued):
        # tick-summed joint-limit-row impulses [legs, spine, arms]
        # (arms priced only under limit_arms — env v35)
        # (N*m*s) and the tick count. Popped + zeroed by
        # pop_limit_rest; reset_env leaves them alone (flail pattern).
        self.limit_rest = z(N, 3)
        self.limit_rest_ticks = z(N, dt=torch.int32)
        # env v41 self-contact telemetry per env: [impulse sum (N.s),
        # ticks with contact, ticks]; popped + zeroed by
        # pop_self_contact, untouched by reset_env
        self.sc_stat = z(N, 3)
        # activation telemetry per env (2026-09-13, replaces the retired
        # positive-work reading): [sum total, sum spine, sum arms, sum
        # legs, ticks] of the commanded |gear fraction| (total = the
        # retired v23 iso-price base); popped + zeroed by pop_activation, untouched
        # by reset_env
        self.act_stat = z(N, 5)
        # env v43 torque telemetry per env: [sum total, sum spine, sum
        # arms, sum legs, ticks] of the applied |tau| in N.m (total = the
        # --energy-coef base); popped + zeroed by pop_torque, untouched
        # by reset_env
        self.tau_stat = z(N, 5)
        # 2026-08-20 training-accuracy telemetry (user metric decision):
        # note-score points as credited + good-cut count, per env.
        # Popped + zeroed by pop_acc; NOT zeroed by reset_env — a
        # continuous flow, phase-free by construction.
        self.acc_pts = z(N, 2)                  # per-hand [L, R]
        self.acc_cuts = z(N, 2, dt=torch.int32)
        # 2026-10-04 (user): the acc broken into its components, per env
        # x hand x [pre pts, pre max, post pts, post max, precision pts,
        # precision max] (note cuts only); popped by pop_acc_parts, same
        # lifecycle as acc_pts
        self.acc_part = z(N, 2, 6)
        # the same components per EPISODE (2026-10-04, user): zeroed by
        # reset_env, copied into ep_rec cols 28..39 at done
        self.ep_part = z(N, 2, 6)
        # env v20 start-pose jitter draws: (N, 12) in [-1, 1), refilled
        # in-place before every launch when enabled (in-place uniform_
        # keeps the fixed address CUDA graphs capture; torch's RNG
        # offset handling makes replays draw fresh values, the _kick
        # precedent). Consumed only by envs that reset this launch.
        self._pose_jit = z(N, 14)  # 12 jitter + crouch gate/depth
        # review B10: start-jitter kick [dvx, dvz] per env (N, 2), refilled
        # before every launch while start_jitter > 0 (else stays zero);
        # the kernel's reset_env installs it as the root's horizontal
        # velocity before the reset obs is written. Fixed address =>
        # CUDA-graph safe (the _pose_jit precedent).
        self._start_kick = z(N, 2)
        # env v24: §8 body-point finite-diff base + world velocities
        self._bp_pos = z(N, 18)
        self._bp_vel = z(N, 18)
        # env v22: previous tick's total hinged foot deviation
        self._foot_pot = z(N)
        # env v12 (contact spec §7): tick-summed ground normal impulses
        # per (env, [L_heel, L_toe, R_heel, R_toe]); feeds the four
        # OFF_GROUND_IMP obs lanes.
        self.ground_imp = z(N, 4)
        self.chart_idx = z(N, dt=torch.int64)
        # env v17 wall state (spec §12b): passed latches, passed-prefix
        # cursor, head-in-wall tick counter
        self.wall_passed = z(N, self.W, dt=torch.bool)
        self.first_wall = z(N, dt=torch.int32)
        self.walls_in = z(N, dt=torch.int32)
        # head-in-wall state last tick — entry-edge for the
        # once-per-entry WALL_PEN (2026-08-20; kernel-reset per episode)
        self.in_wall_prev = z(N, dt=torch.bool)
        # R12 warp-coherence permutation: thread gid serves env
        # env_order[gid]. Identity unless env_order=True (the trainer
        # passes it), which sorts by (chart, first_pending) each tick so
        # warp-mates share note windows. A bijection on thread->env
        # assignment only — per-env results are bitwise-identical under
        # ANY permutation (the sorted mode is pinned against identity in
        # tests/test_env_order_bitwise.py).
        self.env_order = torch.arange(N, dtype=torch.int32,
                                      device=self.device)
        self._env_order_on = bool(env_order)
        self._gbits = torch.tensor([1, 2, 4, 8], device=self.device)
        # audit-2 #6b sort-key tables: joint limits / speed caps (the
        # kernel's LIMIT_LO/HI and OMEGA_CAP; a proxy, exactness moot)
        from .constants import H_SUB, JOINT_LIMITS, OMEGA_CAP
        _lim = torch.tensor(JOINT_LIMITS, dtype=self.dtype,
                            device=self.device)
        _cap = torch.tensor(OMEGA_CAP, dtype=self.dtype, device=self.device)
        self._ord_lo, self._ord_hi = _lim[:, 0], _lim[:, 1]
        self._ord_reach = _cap * H_SUB
        self._ord_cap = _cap * (1.0 - 1e-6)
        # episode-return ring buffer (device-side)
        self._cap = 4096
        self._ret_buf = z(self._cap)
        self._ret_total = z(1, dt=torch.int32)
        self._popped = 0
        self._obs = z(N, OBS_DIM, dt=torch.float32)
        self._reward = z(N)
        self._done = z(N, dt=torch.bool)
        self._zero_act = z(N, N_JOINTS, dt=torch.int64)
        self._zero_cand = z(N, dt=torch.int64)
        p = self.pool
        # order is a contract with the kernel's tensor-list enum
        self._tensors = [
            self.p, self.quat, self.q, self.v,
            self.wrist, self.blade, self.wrist_vel, self.head,
            self.shoulder_z,
            self.swing_acc, self.axis, self.axis_valid,
            self.tr_acc, self.tr_age, self.tr_cnt, self.tr_active,
            self.tr_ptr,
            self.tick, self.status, self.first_pending, self.ep_return,
            self.good, self.bad, self.miss, self.bombs, self.score,
            self.score_pre, self.score_acc, self.score_post,
            self.contact_imp, self.contact_pt, self.contact_arm,
            self._pd_hits, self._env_hits, self._fall_hits,
            self._tilt_hits, self._lift_hits, self.foot_down_prev,
            self.flail_rad, self.flail_ticks,
            self.wrev, self.wrev_ext, self.wrev_dir,
            self.wrs, self.wrs_ext, self.wrs_acc,
            self.acc_pts, self.acc_cuts, self.acc_part, self.ep_part,
            self.ground_imp,
            self.chart_idx,
            self.limit_rest, self.limit_rest_ticks,
            self._pose_jit,
            self._foot_pot,
            self._bp_pos, self._bp_vel,
            self.sc_stat,
            self.act_stat,
            self.tau_stat,
            self.prev_idx, self.prev_valid,
            self.tau_scale, self._tau_jit, self.ep_rec, self.ep_flag,
            self.cw_mult, self.cw_prog, self.cw_combo, self.cw_maxc,
            self.cw_breaks, self.cw_score, self.cw_note_mult, self.tr_msum,
            self.bench_rec, self.bench_flag,
            self.palm_vel,
            self._start_kick,
            self.tip_vel,
            p["hit"], p["speed"], p["center"], p["dvec"], p["rot"],
            p["good_hy"], p["bad_hy"],
            p["valid"], p["is_right"], p["any_dir"], p["is_bomb"],
            p["is_head"], p["is_link"], p["show_dir"], p["free_pre"],
            p["free_post"],
            p["len"], p["sweet"],
            p["n_notes"], p["last_hit"], p["tail"], p["min_speed"],
            p["note_off"],
            p["wall_t"], p["wall_cx"], p["wall_cy"], p["wall_w"],
            p["wall_h"], p["wall_len"], p["wall_speed"],
            p["wall_valid"], p["n_walls"], p["wall_off"],
            self.wall_passed, self.first_wall, self.walls_in,
            self.in_wall_prev,
            self.env_order,
            self._ret_buf, self._ret_total, self._obs, self._reward,
            self._done,
        ]

    def _launch(self, actions, cand, mode):
        self._draw_reset_jitter(mode)
        self._ext_step(actions, cand, mode)

    def _draw_reset_jitter(self, mode):
        """Refill the per-episode start draws the kernel's reset_env
        consumes (pose jitter / crouch gate, tau body, start kick) —
        every one of them lands BEFORE the reset obs is written."""
        if self.start_pose_jitter > 0.0 or self.crouch_start_prob > 0.0:
            self._pose_jit.uniform_(-1.0, 1.0)
        if self.tau_rand > 0.0:
            # seeded generator: the body draws are reproducible per seed;
            # the kernel installs this buffer as-is at reset (env v47b)
            self._tau_jit.uniform_(-1.0, 1.0, generator=self._gen)
            self._tau_jit.mul_(self.tau_rand).add_(1.0).mul_(self.tau_center)
        if self.start_jitter > 0.0 and (
                mode == 1 or (mode == 0 and (self.auto_reset
                                             or self.bench_refill))):
            # review B10: the start kick, precomputed for every env and
            # consumed in-kernel only by envs that reset this launch
            # (launches that can reset nobody — set_state, latched
            # auto_reset=False steps — draw nothing, as before B10).
            # Same default-RNG draws, in the same order, and the same
            # arithmetic as the pre-B10 post-launch _kick(self._done, ..)
            # (x * gate 1.0 is exact, 0 + dv is exact), so mode-0
            # auto-reset trajectories keep their random stream and kick
            # values; only the obs of the reset tick changes.
            mag = torch.rand(self.num_envs, device=self.device,
                             dtype=self.dtype) * self.start_jitter
            ang = torch.rand(self.num_envs, device=self.device,
                             dtype=self.dtype) * (2.0 * math.pi)
            self._start_kick[:, 0].copy_(mag * torch.cos(ang))
            self._start_kick[:, 1].copy_(mag * torch.sin(ang))

    def _ext_step(self, actions, cand, mode):
        """The raw kernel launch on the buffers as they stand (tests
        replay a recorded set of start draws through it)."""
        self._ext.step(self._tensors, actions, cand, mode,
                       self.max_steps,
                       self.auto_reset, self.gears, self.want_contacts,
                       self.fall_const,
                       self.foot_z_coef, self.foot_x_coef,
                       self.foot_hinge_r, self.foot_shape_coef,
                       self.energy_coef, self.limit_coef,
                       int(self.limit_arms),
                       int(self.wall_despawn),
                       self.start_pose_jitter, self.crouch_start_prob,
                       self.z_asinh, self.self_contact_coef,
                       self.tau_rand, int(self.combo_walk),
                       int(self.bench_refill), int(self.ep_stats),
                       self.flux)

    # -- env v46: torque-limit scales ---------------------------------------
    def set_tau_scale(self, s16):
        """Fix the 16 per-axis torque-limit scales on EVERY env (bench /
        eval at a chosen body; RawBeatSim.set_tau_scale mirror). Needs
        tau_rand == 0, otherwise the next reset redraws them. Takes effect
        on the next step (torque + obs lanes)."""
        assert self.tau_rand == 0.0, "set_tau_scale needs tau_rand == 0"
        s = torch.as_tensor(list(s16), dtype=self.tau_scale.dtype,
                            device=self.device)
        assert s.numel() == TAU_FEAT, "tau scale must have 16 axis values"
        self.tau_scale.copy_(s.reshape(1, TAU_FEAT).expand_as(self.tau_scale))

    def pop_episodes(self):
        """Episodes that ended since the last pop, as an (M, EP_REC)
        float32 numpy array: [s0..s15, ep_ticks, ep_return, fell,
        good_cuts, score_pts, chart_idx, cw_score, max_combo, breaks,
        completed, bad_cuts, misses, then 12 swing cols (2026-10-04):
        per hand [L, R] x (pre pts, pre max, post pts, post max,
        precision pts, precision max) — the episode's raw note points
        before the combo multiplier, note cuts only (EP_PART_COL)] (the
        kernel writes a row at `done` while tau_rand > 0 or ep_stats;
        cols 22..27 are zero unless ep_stats; an env ending twice inside
        one pop keeps the later row — logging-grade: an update is
        horizon (32) ticks and a level lasts far longer, so only an
        instant fall can lose a row). Two syncs; call once per update."""
        flags = self.ep_flag.bool()
        n = int(flags.sum().item())
        if n == 0:
            return np.zeros((0, EP_REC), dtype=np.float32)
        rows = self.ep_rec[flags].float().cpu().numpy()
        self.ep_flag.zero_()
        return rows

    # -- bench slot refill ---------------------------------------------------
    def set_next_chart(self, cand):
        """(N,) int64 on the device: the pool chart each slot starts when
        its current level ends (-1 = none: the slot freezes). Read by the
        kernel at that slot's `done`, so update it right after popping the
        slot's row and before the next step."""
        assert self.bench_refill, "bench_refill=False at construction"
        cand = torch.as_tensor(cand, device=self.device, dtype=torch.int64)
        assert cand.shape == (self.num_envs,)
        self._bench_cand = cand.contiguous()

    def pop_bench_rows(self):
        """Levels that finished since the last pop: (env ids (M,) int64
        numpy, rows (M, BENCH_COLS) float64 numpy) with the row layout of
        the kernel's bench_rec [good, bad, miss, score, bombs, pre, acc,
        post, flail_rad, flail_ticks, cw_score, cw_maxc, cw_breaks,
        chart_idx, wrev_L, wrev_R, ticks, wall_enters, term_flags]
        (term_flags: TERM_* bits, 0 = the level itself ended). One row
        per env — pop every tick (a level lasts >= 60 ticks, so no slot
        can finish twice between pops)."""
        flags = self.bench_flag.bool()
        n = int(flags.sum().item())
        if n == 0:
            return (np.zeros((0,), dtype=np.int64),
                    np.zeros((0, BENCH_COLS), dtype=np.float64))
        ids = flags.nonzero(as_tuple=False).squeeze(1)
        rows = self.bench_rec[ids, :BENCH_COLS].double().cpu().numpy()
        self.bench_flag.zero_()
        return ids.cpu().numpy(), rows

    def _cand(self):
        """Fresh chart index per env (used only for envs that reset)."""
        N = self.num_envs
        if self._test_charts:  # deterministic: env i -> pool[(i + ep*N) % P]
            return (torch.arange(N, device=self.device)
                    + self._episode * N) % self.pool_size
        return torch.randint(self.pool_size, (N,), device=self.device,
                             generator=self._gen)

    def reset(self):
        if self.combo_walk or not self.auto_reset:
            # the finished-episode latch keys on done: clear stale flags
            self._done.zero_()
        if self.bench_refill:
            assert self._bench_cand is not None, "set_next_chart() first"
            assert bool((self._bench_cand >= 0).all()), \
                "every slot needs a first level for reset()"
            self._launch(self._zero_act, self._bench_cand, 1)
        else:
            self._launch(self._zero_act, self._cand(), 1)
        self._episode += 1
        return self._obs

    def _kick(self, gate, vmax):
        """Horizontal root-velocity impulse (env v18 perturbation
        training): per env, |dv| ~ U(0, vmax) x gate, direction uniform
        in the x-z plane. CUDA-GRAPH-SAFE by construction — fixed
        shapes (full-width masked mul, never boolean indexing), no host
        syncs, default CUDA RNG (torch handles its offset across graph
        replays; the action-sampling stream already relies on this)."""
        g = gate.to(self.dtype)
        mag = torch.rand(self.num_envs, device=self.device,
                         dtype=self.dtype) * vmax * g
        ang = torch.rand(self.num_envs, device=self.device,
                         dtype=self.dtype) * (2.0 * math.pi)
        self.v[:, 0] += mag * torch.cos(ang)
        self.v[:, 2] += mag * torch.sin(ang)

    def step(self, actions):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(np.asarray(actions))
        actions = actions.to(self.device).long().reshape(
            self.num_envs, N_JOINTS).contiguous()
        if self.push_prob > 0.0:
            kick = torch.rand(self.num_envs, device=self.device,
                              dtype=self.dtype) < self.push_prob
            if not self.auto_reset:   # a finished (latched) slot is frozen
                kick &= ~self._done
            self._kick(kick, self.push_vmax)
        if self.bench_refill:
            cand = self._bench_cand
        else:
            cand = self._cand() if self.auto_reset else self._zero_cand
        if self._env_order_on:
            # R12: group envs by DYNAMICS state — ground-contact config
            # (heel/toe bitmask, the substep's ng branch driver) then
            # quantized root height. Chart-phase keying measured NULL
            # (-0.06%); this key measured +8.7% ms/tick at N=32768
            # (2026-08-21 A/B, r12_dyn_ab). stable=True is part of the
            # run-to-run determinism gate.
            # audit-2 #6b (perf/kernel-limits): the height is binned 4x
            # coarser (4 cm) and the LAST key is the limit / cap
            # activity — joints inside their limit-row reach plus joints
            # at their speed cap, the drivers of the kernel's per-lane
            # limit-row count, cap pass and step-11 passes (a warp pays
            # its worst lane). Measured at N=32768 on the #6c/#6e kernel:
            # standard -2.2%, Flux -2.0%, trained standard even; the
            # other orders tried (activity above the contact mask, above
            # the height, or bucketed) were null or worse.
            gmask = ((self.ground_imp > 0).long() * self._gbits).sum(1)
            hq = (self.p[:, 1] * 100).long().clamp(0, 4095)
            gap = torch.minimum(self.q - self._ord_lo,
                                self._ord_hi - self.q)
            act = ((gap < self._ord_reach).sum(1)
                   + (self.v[:, 6:].abs() >= self._ord_cap).sum(1))
            key = (gmask * 1024 + hq // 4) * 64 + act
            self.env_order.copy_(
                torch.argsort(key, stable=True).to(torch.int32))
        self._launch(actions, cand, 0)
        if self.auto_reset:
            self._episode += 1
            # envs that terminated this tick were auto-reset in-kernel,
            # start kick included (reset_env, before the obs write)
        return self._obs, self._reward, self._done, []

    # -- parity / tooling hooks (tests/test_v10_parity_cuda.py) ---------------
    def set_state(self, q36, v35):
        """RawBeatSim.set_joints mirror: q36 = root pos + quat wxyz + 29
        joints (quat normalized with the same expression), v35 full
        velocity. Broadcasts a single state or takes (N, ·) batches;
        FK caches refresh in-kernel; finite-diff wrist velocity resets."""
        q36 = torch.as_tensor(q36, device=self.device, dtype=self.dtype)
        v35 = torch.as_tensor(v35, device=self.device, dtype=self.dtype)
        if q36.dim() == 1:
            q36 = q36.unsqueeze(0).expand(self.num_envs, -1)
        if v35.dim() == 1:
            v35 = v35.unsqueeze(0).expand(self.num_envs, -1)
        self.p.copy_(q36[:, :3])
        qn = torch.sqrt((q36[:, 3:7] * q36[:, 3:7]).sum(-1, keepdim=True))
        self.quat.copy_(q36[:, 3:7] / qn)
        self.q.copy_(q36[:, 7:])
        self.v.copy_(v35)
        self._launch(self._zero_act, self._zero_cand, 2)

    def qpos(self):
        return torch.cat([self.p, self.quat, self.q], -1)

    def twist(self):
        return self.v[:, :6]

    def omega(self):
        return self.v[:, 6:]

    def sabers(self):
        return torch.cat([self.wrist, self.blade], -1)  # (N, 2, 6)

    def palms(self):
        """Flux palm-centre points (N, 2, 3): wrist + PALM_OFF along the
        hand axis (RawBeatSim.palms mirror)."""
        return self.wrist + PALM_OFF * self.blade

    def counters(self):
        return (self.good, self.bad, self.miss,
                self.pool["n_scoreable"][self.chart_idx], self.score,
                self.bombs)

    def components(self):
        return (self.score_pre, self.score_acc, self.score_post)

    def combo(self):
        """Bench combo walk per env: (multiplied score (N,), highest
        combo (N,), breaks (N,)) — requires combo_walk=True at
        construction (VecBeatSim.combo(i) semantics, vectorized)."""
        assert self.combo_walk, "combo_walk=False at construction"
        return self.cw_score, self.cw_maxc, self.cw_breaks

    def contacts(self):
        """(imp (N,55) tick-summed lambda, point (N,55,3) last substep).
        Requires want_contacts=True at construction — the kernel skips
        these writes otherwise (they feed nothing; obs uses contact_arm)."""
        if not self.want_contacts:
            raise RuntimeError(
                "contacts() needs CudaBeatVecEnv(..., want_contacts=True); "
                "this env was built with it off, so contact_imp/contact_pt "
                "are stale (never written by the kernel)")
        return self.contact_imp, self.contact_pt

    def pop_pd_hits(self):
        """Cholesky pivot-floor engagements since the last pop
        (Divergences §20e f32 / §22 f64 — reported, not hidden).
        One sync."""
        total = int(self._pd_hits.item())
        new = total - self._pd_popped
        self._pd_popped = total
        return new

    def pop_envelope_hits(self):
        """PRE-integration root-velocity-envelope engagements since the
        last pop (SPEC-v10 Divergences §23), counted per (env, substep) —
        the Rust backend's pop_envelope_hits mirror. Nonzero means at
        least one env's root left the kinematically reachable envelope
        and its episode was terminated with a zeroed reward (the C
        backstop). 0 on any healthy population. One sync."""
        total = int(self._env_hits.item())
        new = total - self._env_popped
        self._env_popped = total
        return new

    def pop_fall_hits(self):
        """FALL terminations since the last pop (unilateral-contact spec
        §4) — the Rust backend's pop_fall_hits mirror, same shape as
        pop_envelope_hits. Counted per (env, tick): each one terminated an
        env's episode with a zeroed reward because its root dropped below
        FALL_ROOT_Y. One sync."""
        total = int(self._fall_hits.item())
        new = total - self._fall_popped
        self._fall_popped = total
        return new

    def pop_tilt_hits(self):
        """TILT terminations since the last pop (2026-08-20): CHEST
        up-axis (root x spine stack) tilted more than 40 deg
        forward, 20 deg backward, or 40 deg to the
        side at end of tick (per-TICK check, not per substep) —
        episode ends priced at the flat fall_const, exactly like a
        fall. One sync."""
        total = int(self._tilt_hits.item())
        new = total - self._tilt_popped
        self._tilt_popped = total
        return new

    def pop_foot_lifts(self):
        """Foot-lift events since the last pop (2026-08-20): a foot's
        tick-summed ground normal impulse hit zero after a grounded
        tick — each charged the flat FOOT_LIFT_PEN (-0.1). Returns
        (left, right) counts since the last pop (per-foot split, user
        2026-08-20). One sync."""
        tl, tr = (int(x) for x in self._lift_hits.tolist())
        new = (tl - self._lift_popped[0], tr - self._lift_popped[1])
        self._lift_popped = [tl, tr]
        return new

    def pop_flail(self):
        """env v19 flail telemetry since the last pop: (L blade arc in
        rad, R blade arc, count of note-free env-ticks) accrued by
        follow-through-idle hands on note-free ticks. Per-hand since
        2026-08-20 (user direction). Sums the per-env accumulators and
        zeroes them — logging-grade, two syncs per call."""
        per_hand = self.flail_rad.sum(dim=0)
        rad_l, rad_r = float(per_hand[0].item()), float(per_hand[1].item())
        ticks = int(self.flail_ticks.sum().item())
        self.flail_rad.zero_()
        self.flail_ticks.zero_()
        return rad_l, rad_r, ticks

    def pop_wrist_rev(self):
        """Wrist-bend reversals since the last pop, (L, R) counts summed
        over envs (2026-09-22; 60 deg hysteresis since 2026-10-04, peak detection in the
        kernel, every tick). Zeroes the counters. The caller divides by
        the env-ticks it stepped (num_envs x horizon per update)."""
        per_hand = self.wrev.sum(dim=0)
        l, r = int(per_hand[0].item()), int(per_hand[1].item())
        self.wrev.zero_()
        return l, r

    def pop_limit_rest(self):
        """Limit-rest telemetry since the last pop: (legs lambda-flow
        in N*m*s, spine lambda-flow, arms lambda-flow, env-ticks) —
        tick-summed joint-limit-row impulses (spine 0..3, arms 3..17,
        legs 17..29). All buckets accumulate regardless of limit_coef /
        limit_arms; arms are PRICED only under limit_arms (env v35).
        Sums the per-env accumulators and zeroes them — logging-grade,
        two syncs per call."""
        buckets = self.limit_rest.sum(dim=0)
        legs, spine = float(buckets[0].item()), float(buckets[1].item())
        arms = float(buckets[2].item())
        ticks = int(self.limit_rest_ticks.sum().item())
        self.limit_rest.zero_()
        self.limit_rest_ticks.zero_()
        return legs, spine, arms, ticks

    def pop_self_contact(self):
        """env v41 self-contact telemetry since the last pop: (summed
        self-collision impulse in N.s, env-ticks with any contact,
        env-ticks). Accumulates regardless of self_contact_coef. Sums
        the per-env accumulators and zeroes them — one sync."""
        s = self.sc_stat.sum(dim=0).tolist()
        self.sc_stat.zero_()
        return float(s[0]), int(s[1]), int(s[2])

    def pop_activation(self):
        """Activation telemetry since the last pop: (sum total, sum
        spine, sum arms, sum legs, env-ticks) of the commanded |gear
        fraction| per tick — the retired v23 --iso-coef base (telemetry);
        accumulates regardless of the coefficient. Sums the per-env
        accumulators and zeroes them — one sync."""
        s = self.act_stat.sum(dim=0).tolist()
        self.act_stat.zero_()
        return (float(s[0]), float(s[1]), float(s[2]), float(s[3]),
                int(s[4]))

    def pop_torque(self):
        """env v43 torque telemetry since the last pop: (sum total, sum
        spine, sum arms, sum legs, env-ticks) of the applied |tau| in
        N.m per tick — total is exactly the --energy-coef base;
        accumulates regardless of the coefficient. One sync."""
        s = self.tau_stat.sum(dim=0).tolist()
        self.tau_stat.zero_()
        return (float(s[0]), float(s[1]), float(s[2]), float(s[3]),
                int(s[4]))

    def pop_acc(self):
        """Training-accuracy telemetry since the last pop, PER HAND
        since 2026-08-20 (user): (pts_l, pts_r, cuts_l, cuts_r) —
        note-score points credited and good cuts, [L, R]. Combined
        acc = (pts_l+pts_r) / (115 * (cuts_l+cuts_r)). Sums the
        per-env accumulators and zeroes them — logging-grade, two
        syncs per call."""
        p = self.acc_pts.sum(dim=0)
        c = self.acc_cuts.sum(dim=0)
        pts_l, pts_r = float(p[0].item()), float(p[1].item())
        cuts_l, cuts_r = int(c[0].item()), int(c[1].item())
        self.acc_pts.zero_()
        self.acc_cuts.zero_()
        return pts_l, pts_r, cuts_l, cuts_r

    def pop_acc_parts(self):
        """The training acc's components since the last pop (2026-10-04,
        user): (pre, post, precision), each points credited / points
        possible over note cuts — pre-swing out of 70, post-swing
        (follow-through) out of 30, precision (cut centre distance) out of
        15. NaN for a component with nothing credited. Sums the per-env x
        hand accumulators and zeroes them — logging-grade, one sync."""
        s = self.acc_part.sum(dim=(0, 1)).tolist()
        self.acc_part.zero_()
        f = lambda p, m: p / m if m > 0 else float("nan")  # noqa: E731
        return f(s[0], s[1]), f(s[2], s[3]), f(s[4], s[5])

    def pop_wrist_resets(self):
        """Wrist resets since the last pop (2026-10-04): (resets L,
        resets R, swings L, swings R) summed over envs, then zeroed. A
        reset = a good note cut whose bend-velocity sign equals the
        previous swing's with a >= 30 deg reversal of the bend in between
        (lib.rs WristReset). One sync, logging-grade."""
        s = self.wrs_acc.sum(dim=0).tolist()     # [[rL, sL], [rR, sR]]
        self.wrs_acc.zero_()
        return int(s[0][0]), int(s[1][0]), int(s[0][1]), int(s[1][1])

    def pop_stats(self):
        """Episode returns finished since the last pop (order approximate
        under ring overflow; logging-grade, two syncs per call — the
        counter .item() and the ring-read .tolist())."""
        total = int(self._ret_total.item())
        new = total - self._popped
        if new <= 0:
            return []
        take = min(new, self._cap)
        idx = (torch.arange(total - take, total, device=self.device)
               % self._cap)
        vals = self._ret_buf[idx].tolist()
        self._popped = total
        return vals
