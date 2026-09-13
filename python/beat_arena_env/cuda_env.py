"""CUDA-kernel vectorized Beat Arena sim, env v10 — one kernel launch per
step (P5).

rust/src/lib.rs remains THE reference implementation; this backend runs
the whole v10 step — 8 constrained dynamics substeps (SPEC-v10 §3),
gear->torque actuation, work-based effort, swept-surface judgment,
auto-reset, the obs write — inside one custom CUDA kernel
(csrc/beat_arena_cuda.cu), one THREAD per env (P5 parallelization
decision: the Rust expression trees transcribe serially per thread, so
every accumulation keeps Rust's exact order; occupancy comes from batch
width — production runs N = 1024..8192).

Parity is pinned by tests/test_v10_parity_cuda.py: the f64 tier is
HORIZON-BOUNDED tight-tolerance (CUDA f64 sin/cos/acos differ from the
CRT libm by 1 ulp on a few %% of inputs and nvcc contracts FMAs — both
measured there; SPEC-v10 Divergences §21), f32 is the bounded
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

Env v11 (unilateral-contact spec, 2026-08-08) adds `foot_contact=`:
foot_mode 1 swaps the two 6-DOF foot welds for unilateral frictional
ground contact on the 8 sole-box corners and arms the fall termination
(root y < 0.45, counted by `pop_fall_hits()`). It selects a COMPILE-TIME
kernel instantiation, so the default weld mode is the v10 kernel
unchanged; test_v11_contact_parity.py is the contact-mode gate.

WELD MODE IS TEST-HARNESS ONLY since 2026-08-16 (user decision): the
production surfaces (train_batch, bench_scoresaber) force contact mode
unconditionally. foot_contact=False survives solely so the
scripted-mechanics suite has a stable platform (an unactuated contact
body tips in ~0.7 s, which would end every multi-second swing script).

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
from .constants import (
    BAD_HALF, BAD_HALF_Y_LINK, FOLLOW_TIMEOUT, GOOD_HALF_Y,
    GOOD_HALF_Y_DOT, GOOD_HALF_Y_LINK, KIND_BOMB, KIND_CHAIN_HEAD,
    KIND_CHAIN_LINK, N_JOINTS, NV, OBS_DIM, note_center,
)

NUM_PAIRS = 55  # SPEC-v10 §2 live pair table (rust dynamics/collide.rs)

# Follow-through tracker ring (v7 design, see the kernel header):
# same-tick same-saber trackers share (arc, age) forever, so they
# aggregate into ONE slot with a note COUNT — exact, not approximate.
N_TRACKERS = FOLLOW_TIMEOUT + 1

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
    _flatten_block(pool, NOTE_OBJ_FIELDS, "valid", "note_off", "n_notes")
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
    _unflatten_block(pool, NOTE_OBJ_FIELDS, "note_off", "n_notes",
                     _charts_to_padded([[]], dev, dt))
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


def _pool_cache_path(cfg, seed, pool_size, dtype):
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
    blob = repr(("v12", _generator_version(), repr(cfg), seed, pool_size,
                 str(dtype))).encode()
    key = hashlib.sha1(blob).hexdigest()[:16]
    cdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "pool_cache")
    os.makedirs(cdir, exist_ok=True)
    return os.path.join(cdir, f"pool_{pool_size}_{key}.npz")


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
                 note_speed_min=2.0, note_speed_max=8.0,
                 note_prob=0.25, extra_prob=0.125, dot_prob=0.1,
                 bomb_prob=0.125, seed=0,
                 charts=None, chart_pool=1024, max_steps=None,
                 pool=None, wide_acc=False, gears=3,
                 device="cuda", dtype=torch.float32, auto_reset=True,
                 want_contacts=False, foot_contact=False,
                 fall_const=5.0, walls=None,
                 push_prob=0.0, push_vmax=0.0, start_jitter=0.0,
                 foot_z_coef=0.0, foot_x_coef=0.0, foot_hinge_r=0.0,
                 foot_shape_coef=0.0, iso_coef=0.0, limit_coef=0.0,
                 limit_arms=False,
                 wall_despawn=False,
                 start_pose_jitter=0.0, crouch_start_prob=0.0,
                 z_clip_hi=12.0, self_contact_coef=0.0):
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
        # env v11 foot_mode (unilateral-contact spec §1): False = weld
        # (v10, the DEFAULT — every existing trajectory unchanged), True =
        # unilateral frictional ground contact on the 8 sole corners plus
        # the fall termination. Baked here, never mutated: the kernel
        # selects a COMPILE-TIME instantiation on it, so it is constant
        # per instance and safe under CUDA-graph capture.
        self.foot_contact = bool(foot_contact)
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
        self.iso_coef = float(iso_coef)
        self.limit_coef = float(limit_coef)
        self.limit_arms = bool(limit_arms)
        # env v37: bomb-like wall despawn on charge (training-only aid;
        # bench/replays leave it off for faithful persistent walls)
        self.wall_despawn = bool(wall_despawn)
        # v28: obs z-lane upper clamp (m), runtime per-run (arg wins;
        # the rust twin reads BEAT_Z_CLIP_HI instead)
        self.z_clip_hi = float(z_clip_hi)
        # env v41 (2026-09-12): price on the tick-summed self-collision
        # impulse (N.s); 0 = off. Telemetry accumulates regardless.
        self.self_contact_coef = float(self_contact_coef)
        self.start_pose_jitter = float(start_pose_jitter)
        assert start_pose_jitter == 0.0 or foot_contact, \
            "start_pose_jitter fights the v10 foot welds; contact mode only"
        # env v38 crouch-rich starts (training-only; see kernel table)
        self.crouch_start_prob = float(crouch_start_prob)
        assert crouch_start_prob == 0.0 or foot_contact, \
            "crouch starts need contact mode (welded feet fight the pose)"

        # env v18 perturbation training (push-recovery lit): with
        # probability push_prob per (env, tick) add a horizontal root
        # velocity kick, |dv| ~ U(0, push_vmax) m/s, direction uniform.
        # start_jitter: same kick applied to every env that reset this
        # tick (RSI-flavored start-state variety), |dv| ~ U(0, start_jitter).
        # Python-side v[:, 0/2] mutation between kernel launches — an
        # impulse by construction; seeded via self._gen. Training only:
        # bench/replay use the Rust sim, which has no push path.
        self.push_prob = float(push_prob)
        self.push_vmax = float(push_vmax)
        self.start_jitter = float(start_jitter)
        self.wide_acc = bool(wide_acc)
        assert gears in (3, 5, 7), f"gears must be 3, 5, or 7, got {gears}"
        self.gears = int(gears)
        # pool=: prebuilt tensor dict (ranked_pool.py; ragged, or padded
        # (P, C) from an older builder — converted here) — unlike
        # charts=, resets keep the normal RANDOM chart draw (charts= is
        # the deterministic-sequential test mode)
        self._test_charts = charts is not None and pool is None
        if pool is not None:
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
        self.first_pending = z(N, dt=torch.int32)
        self.ep_return = z(N)
        self.good = z(N, dt=torch.int32); self.bad = z(N, dt=torch.int32)
        self.miss = z(N, dt=torch.int32); self.bombs = z(N, dt=torch.int32)
        self.score = z(N)
        self.score_pre = z(N)
        self.score_acc = z(N)
        self.score_post = z(N)
        # contact aggregation + effort (obs bits / parity introspection)
        self.contact_imp = z(N, NUM_PAIRS)
        self.contact_pt = z(N, NUM_PAIRS, 3)
        self.contact_arm = z(N, 2, dt=torch.bool)
        self.effort_j = z(N)
        # Cholesky pivot-floor engagements (§20e f32 / §22 f64)
        self._pd_hits = z(1, dt=torch.int32)
        self._pd_popped = 0
        # PRE-integration root-envelope engagements (§23), counted per
        # (env, substep); an env that engages any of them terminates that
        # tick with a zeroed reward (the C backstop)
        self._env_hits = z(1, dt=torch.int32)
        self._env_popped = 0
        # env v11 FALL terminations (unilateral-contact spec §4), counted
        # per (env, tick); always 0 in weld mode
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
        # 2026-08-20 training-accuracy telemetry (user metric decision):
        # note-score points as credited + good-cut count, per env.
        # Popped + zeroed by pop_acc; NOT zeroed by reset_env — a
        # continuous flow, phase-free by construction.
        self.acc_pts = z(N, 2)                  # per-hand [L, R]
        self.acc_cuts = z(N, 2, dt=torch.int32)
        # env v20 start-pose jitter draws: (N, 12) in [-1, 1), refilled
        # in-place before every launch when enabled (in-place uniform_
        # keeps the fixed address CUDA graphs capture; torch's RNG
        # offset handling makes replays draw fresh values, the _kick
        # precedent). Consumed only by envs that reset this launch.
        self._pose_jit = z(N, 14)  # 12 jitter + crouch gate/depth
        # env v24: §8 body-point finite-diff base + world velocities
        self._bp_pos = z(N, 18)
        self._bp_vel = z(N, 18)
        # env v22: previous tick's total hinged foot deviation
        self._foot_pot = z(N)
        # env v12 (contact spec §7): tick-summed ground normal impulses
        # per (env, [L_heel, L_toe, R_heel, R_toe]); feeds the four
        # OFF_GROUND_IMP obs lanes. Stays 0 in weld mode.
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
        # env_order[gid]. Identity unless BEAT_ENV_ORDER=1, which sorts
        # by (chart, first_pending) each tick so warp-mates share note
        # windows. A bijection on thread->env assignment only — per-env
        # results are bitwise-identical under ANY permutation
        # (tests/test_env_order_bitwise.py).
        self.env_order = torch.arange(N, dtype=torch.int32,
                                      device=self.device)
        self._env_order_on = os.environ.get("BEAT_ENV_ORDER", "0") == "1"
        self._gbits = torch.tensor([1, 2, 4, 8], device=self.device)
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
            self.effort_j, self._pd_hits, self._env_hits, self._fall_hits,
            self._tilt_hits, self._lift_hits, self.foot_down_prev,
            self.flail_rad, self.flail_ticks,
            self.acc_pts, self.acc_cuts,
            self.ground_imp,
            self.chart_idx,
            self.limit_rest, self.limit_rest_ticks,
            self._pose_jit,
            self._foot_pot,
            self._bp_pos, self._bp_vel,
            self.sc_stat,
            p["hit"], p["speed"], p["center"], p["dvec"], p["rot"],
            p["good_hy"], p["bad_hy"],
            p["valid"], p["is_right"], p["any_dir"], p["is_bomb"],
            p["is_head"], p["is_link"], p["show_dir"], p["free_pre"],
            p["free_post"],
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
        if self.start_pose_jitter > 0.0 or self.crouch_start_prob > 0.0:
            self._pose_jit.uniform_(-1.0, 1.0)
        self._ext.step(self._tensors, actions, cand, mode,
                       self.max_steps, self.wide_acc,
                       self.auto_reset, self.gears, self.want_contacts,
                       self.foot_contact, self.fall_const,
                       self.foot_z_coef, self.foot_x_coef,
                       self.foot_hinge_r, self.foot_shape_coef,
                       self.iso_coef, self.limit_coef,
                       int(self.limit_arms),
                       int(self.wall_despawn),
                       self.start_pose_jitter, self.crouch_start_prob,
                       self.z_clip_hi, self.self_contact_coef)

    def _cand(self):
        """Fresh chart index per env (used only for envs that reset)."""
        N = self.num_envs
        if self._test_charts:  # deterministic: env i -> pool[(i + ep*N) % P]
            return (torch.arange(N, device=self.device)
                    + self._episode * N) % self.pool_size
        return torch.randint(self.pool_size, (N,), device=self.device,
                             generator=self._gen)

    def reset(self):
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
            self._kick(torch.rand(self.num_envs, device=self.device,
                                  dtype=self.dtype) < self.push_prob,
                       self.push_vmax)
        cand = self._cand() if self.auto_reset else self._zero_cand
        if self._env_order_on:
            # R12: group envs by DYNAMICS state — ground-contact config
            # (heel/toe bitmask, the substep's ng branch driver) then
            # quantized root height. Chart-phase keying measured NULL
            # (-0.06%); this key measured +8.7% ms/tick at N=32768
            # (2026-08-21 A/B, r12_dyn_ab). stable=True is part of the
            # run-to-run determinism gate.
            gmask = ((self.ground_imp > 0).long() * self._gbits).sum(1)
            key = (gmask * 4096
                   + (self.p[:, 1] * 100).long().clamp(0, 4095))
            self.env_order.copy_(
                torch.argsort(key, stable=True).to(torch.int32))
        self._launch(actions, cand, 0)
        if self.auto_reset:
            self._episode += 1
            # envs that terminated this tick were auto-reset in-kernel;
            # jitter their fresh start state (obs catches up next tick)
            if self.start_jitter > 0.0:
                self._kick(self._done, self.start_jitter)
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

    def counters(self):
        return (self.good, self.bad, self.miss,
                self.pool["n_scoreable"][self.chart_idx], self.score,
                self.bombs)

    def components(self):
        return (self.score_pre, self.score_acc, self.score_post)

    def last_effort(self):
        return self.effort_j

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
        FALL_ROOT_Y = 0.3 m. Always 0 with foot_contact=False, where the
        feet cannot let go. One sync."""
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
        fall. Always 0 with foot_contact=False. One sync."""
        total = int(self._tilt_hits.item())
        new = total - self._tilt_popped
        self._tilt_popped = total
        return new

    def pop_foot_lifts(self):
        """Foot-lift events since the last pop (2026-08-20): a foot's
        tick-summed ground normal impulse hit zero after a grounded
        tick — each charged the flat FOOT_LIFT_PEN (-0.1). Returns
        (left, right) counts since the last pop (per-foot split, user
        2026-08-20). Always (0, 0) with foot_contact=False. One
        sync."""
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
