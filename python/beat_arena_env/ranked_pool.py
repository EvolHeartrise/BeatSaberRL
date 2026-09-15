"""Ranked-map segment pool for CUDA-env training (sweep s2, 2026-07-29).

Slices every ranked ScoreSaber level (maps/scoresaber_catalog.json via
the bench's warm chart_cache pickles; real-NJS speeds since 2026-07-30,
speed_scale removed) into fixed-length windows and packs them into the
ragged (CSR) tensor dict CudaBeatVecEnv eats (cuda_env.pad_to_ragged) — training on the real catalog
directly, no procedural generator, to probe the procedural->real gap
behind the ~87% accuracy ceiling.

SEGMENTS, not whole maps, by design (user decision 2026-07-29): median
ranked map is 884 notes / 173 s vs the env's ~90-object / 690-step
regime. 25 s windows keep C in the hundreds, episode length uniform,
and every kernel/graph shape static. (A pre-2026-08 comment here
claimed a "C~1536 smem-scratch occupancy budget"; the current kernel
has NO shared-memory structures at all — thread-per-env, local-memory
scratch, register-bound — and its chart scans are first_pending- and
sorted-time-windowed, so per-tick cost tracks the in-flight window,
not C. C's real costs are global memory (N*C status bytes) and one
O(C) zeroing per reset: both trivial.)

Windows are seg_len long and start every STRIDE = seg_len - lead_s
seconds of song time (window k covers [k*stride, k*stride + seg_len)),
notes rebased to window start; the first lead_s seconds of EVERY
window are dropped (k > 0: those notes were mid-flight at the cut;
k = 0 since 2026-08-23: negative-shift intro walls otherwise reach the
head on the first tick — an unavoidable charge). Because consecutive
windows OVERLAP by exactly the lead gap (2026-09-06, user), the kept
ranges [k*stride + lead_s, k*stride + seg_len) tile song time from
lead_s on with no holes: every note past lead_s lands in exactly one
window. Before this the stride equalled seg_len and the lead gaps were
holes — at the default lead 1.5 s that is 5.77% of the catalog's
scoreable notes (373,887 of 6,476,219); the 2026-08/09 campaigns ran
--lead-s 0.5, where the holes held 1.94% (125,768 notes). Measured
2026-09-06. The last
window is whatever is left: its content runs from lead_s to
(last note - base), median ~15 s, and the episode ends at
last_hit + tail, not at seg_len. Windows with < MIN_SCOREABLE
scoreable objects or > MAX_OBJECTS total are dropped (counts are
reported). Built once and cached as npz keyed on (parser version,
seg_len, catalog bytes) — every sweep leg then pays only the npz load
+ a seeded subsample.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

from .chart import Note, Wall
from .constants import KIND_BOMB
from .mirror import mirror_tuples, mirror_walls

LEAD_S = 1.5          # k>0 windows: drop notes cut mid-flight
MIN_SCOREABLE = 4     # drop near-empty tail windows
# 2026-08-23 (user): 512 -> 1536. The old "smem guard" rationale was
# stale (see module docstring — the kernel is not smem-bound and never
# scans O(C) per tick). Measured over the current catalog: densest
# real window is 1,489 entities; 1,536 admits every segment (the old
# cap dropped 128, 0.27%). Cost: status bytes N*C (50 MB at N=32768)
# and pool tensors padded to the observed max — both trivial.
MAX_OBJECTS = 1536

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cache_path(seg_len, ver, catalog_path, lead_s, mirror=False):
    with open(catalog_path, "rb") as f:
        cat = f.read()
    # v3: env v17 — wall tensors joined the pool payload
    # v4: 2026-08-23 — MAX_OBJECTS 512 -> 1536 (cap change is invisible
    # to the parser hash, so it must bump the pool key itself)
    # v5: 2026-08-23 — lead gap applies to k=0 (intro walls clipped to
    # >= lead_s; kills unavoidable first-tick wall charges)
    # v6: 2026-09-03 — ragged (CSR) pool layout (audit 2.7): flat
    # per-object arrays + note_off / wall_off instead of (P, C) padding
    # (7,655 MB -> ~0.5 GB on the 93k-segment mirror pool; same bytes
    # per object, so the v5 caches convert but are simply not reused)
    # v7: 2026-09-06 — window stride = seg_len - lead_s (overlap by the
    # lead gap): every note past lead_s is in exactly one window; the
    # v6 pools had 5.77% (lead 1.5) / 1.94% (lead 0.5) of scoreable
    # notes in no window
    # v8: 2026-09-15 — whole-level mode (seg_len <= 0): one window per
    # level, no MAX_OBJECTS drop; keyed separately from any seg_len
    key = hashlib.sha1(
        (("whole_level_v8" if seg_len <= 0 else f"{seg_len:g}")
         + f"_{ver}_v7_lead{lead_s:g}"
         + ("_mirror" if mirror else "")).encode() + cat
    ).hexdigest()[:16]
    cdir = os.path.join(_BASE, "pool_cache")
    os.makedirs(cdir, exist_ok=True)
    return os.path.join(cdir, f"ranked_{key}.npz")


def _tuple_note(t):
    # note_tuple order: (hit, x, y, color, angle, speed, any_dir, kind,
    # free_pre, free_post); col/row are dead when x/y are explicit
    return Note(hit_time=t[0], col=0, row=0, color=int(t[3]),
                angle=float(t[4]), speed=float(t[5]), any_dir=bool(t[6]),
                kind=int(t[7]), x=float(t[1]), y=float(t[2]),
                free_pre=bool(t[8]), free_post=bool(t[9]))


def _window_walls(wtups, lo, hi, base):
    """Walls (as wall_tuples) clipped to the window [lo, hi) and rebased
    to window start (env v17). A wall's time-span is [start, start +
    length/speed]; spans intersecting the window are kept with the
    outside portion CUT OFF — the same philosophy as dropping the k>0
    lead notes (objects mid-flight at the cut don't materialize
    mid-episode around the player)."""
    out = []
    for (t0, cx, cy, w, h, ln, sp) in wtups:
        t1 = t0 + ln / sp
        c0, c1 = max(t0, lo), min(t1, hi)
        if c1 <= c0:
            continue
        out.append((c0 - base, cx, cy, w, h, (c1 - c0) * sp, sp))
    return out


def window_starts(last, seg_len, lead_s=LEAD_S):
    """Song-time starts of the windows slicing a chart whose last
    object is at `last`. Stride = seg_len - lead_s so the kept ranges
    [base + lead_s, base + seg_len) tile [lead_s, inf) without holes or
    double coverage; the object at `last` falls in the final window
    (last >= final base + lead_s). Charts ending before lead_s get no
    window. seg_len <= lead_s would give a non-positive stride and is
    rejected. Pure function, pinned by tests/test_ranked_pool_windows.py."""
    stride = seg_len - lead_s
    if stride <= 0:
        raise ValueError(f"seg_len {seg_len} must exceed lead_s {lead_s}")
    if last < lead_s:
        return []
    n = int(np.floor((last - lead_s + 1e-6) / stride)) + 1
    return [k * stride for k in range(n)]


def _build_segments(seg_len, catalog_path, log, lead_s=LEAD_S,
                    mirror=False):
    import json
    import sys
    sys.path.insert(0, _BASE)
    import bench_scoresaber as bs      # load_or_parse + parser_version:
    # the EXACT bench parsing path, so training segments and bench plays
    # see byte-identical note streams for the same level
    ver = bs.parser_version()
    catalog = json.load(open(catalog_path, encoding="utf-8"))
    segments, seg_walls, dropped_sparse, dropped_fat = [], [], 0, 0
    for i, entry in enumerate(catalog):
        payload = bs.load_or_parse(entry, os.path.join(_BASE, "maps",
                                                       "scoresaber"),
                                   os.path.join(_BASE, "maps",
                                                "chart_cache"), ver)
        tup = sorted(payload["tuples"], key=lambda t: t[0])
        wtups = sorted(payload.get("walls", []), key=lambda t: t[0])
        if not tup:
            continue
        last = tup[-1][0]
        whole = seg_len <= 0
        # 2026-09-15 (user): whole-level mode — the episode IS the level:
        # one window from song start (the parser already anchors the
        # first note at 2.0 s >= lead_s) to the last object; the
        # per-window object cap does not apply (the CUDA env's only
        # per-env per-object tensor is the 1-byte status array).
        bases = ([0.0] if last >= lead_s else []) if whole \
            else window_starts(last, seg_len, lead_s)
        for base in bases:
            # 2026-08-23 (user): lead gap applies to k=0 too. The
            # note-anchored parser shift (first note -> 2.0 s) can be
            # NEGATIVE, putting intro walls at t <= 0; without the gap
            # they clip to c0 = 0 and reach the head in 0.4/speed s —
            # 32 catalog segments charged an unavoidable first-tick
            # WALL_PEN (measured 19-21 ms entry). Notes are unaffected
            # (none before 2.0 s); intro walls now start >= lead_s out,
            # making "nothing materializes mid-flight" uniform.
            # 2026-09-06: bases step by seg_len - lead_s (window_starts),
            # so this window's lead gap is the previous window's tail.
            lo = base + lead_s
            hi = float("inf") if whole else base + seg_len
            win = [t for t in tup if lo <= t[0] < hi]
            if not whole and len(win) > MAX_OBJECTS:
                dropped_fat += 1
                continue
            if sum(1 for t in win if t[7] != KIND_BOMB) < MIN_SCOREABLE:
                dropped_sparse += 1
                continue
            reb = [(t[0] - base,) + tuple(t[1:]) for t in win]
            wreb = _window_walls(wtups, lo, hi, base)
            segments.append([_tuple_note(t) for t in reb])
            seg_walls.append(wreb)
            if mirror:   # augmentation: every window also plays reflected
                segments.append([_tuple_note(t)
                                 for t in mirror_tuples(reb)])
                seg_walls.append(mirror_walls(wreb))
        if (i + 1) % 1000 == 0:
            log(f"[ranked-pool] {i + 1}/{len(catalog)} levels -> "
                f"{len(segments)} segments", flush=True)
    n_wall_segs = sum(1 for w in seg_walls if w)
    n_max = max((len(sg) for sg in segments), default=0)
    log(f"[ranked-pool] {len(segments)} segments of "
        f"{'WHOLE LEVELS' if seg_len <= 0 else f'{seg_len:g}s'} "
        f"({n_wall_segs} with walls; dropped {dropped_sparse} sparse, "
        f"{dropped_fat} >{MAX_OBJECTS}-object; max {n_max} objects)",
        flush=True)
    return segments, seg_walls


def pool_max_steps(pool, tick_hz=30, margin_ticks=90):
    """Episode tick limit for a pool: the longest level's last object
    plus its tail, plus the same margin the segment formula uses
    (int(seg_len * 30) + 90). Whole-level mode needs this because there
    is no seg_len to derive it from; for segment pools it is <= the
    formula's value."""
    if int(pool["n_notes"].shape[0]) == 0:
        return margin_ticks
    end = (pool["last_hit"].double() + pool["tail"].double()).max().item()
    return int(end * tick_hz) + margin_ticks


def load_ranked_pool(seg_len=25.0, sample=None, seed=0,   # seg_len <= 0: whole levels
                     device="cuda", dtype=torch.float32,
                     catalog=None, log=print, lead_s=LEAD_S,
                     mirror=False):
    # lead_s (env v16, contact spec §11): the k>0 windows' head gap —
    # the earliest possible first note of a segment episode. Default
    # keeps the historical 1.5 s; the survival-drift postmortem
    # hypothesis #4 is that a long pre-note phase feeds the
    # survival-consolidation feedback loop.
    """Ragged pool dict on `device` (cuda_env.pad_to_ragged layout).
    First call builds + caches the full segment set (minutes); later
    calls load the npz (~seconds). `sample` takes a seeded
    without-replacement subset (VRAM control; every leg with the same
    seed/sample sees the same subset)."""
    catalog = catalog or os.path.join(_BASE, "maps",
                                      "scoresaber_catalog.json")
    import sys
    sys.path.insert(0, _BASE)
    import bench_scoresaber as bs
    cpath = _cache_path(seg_len, bs.parser_version(), catalog, lead_s,
                        mirror=mirror)
    if not os.path.exists(cpath):
        from .cuda_env import (_charts_to_tensors, _walls_to_tensors,
                               cat_pools)
        segs, seg_walls = _build_segments(seg_len, catalog, log,
                                          lead_s=lead_s, mirror=mirror)
        # Chunked packing (2026-08-08): one whole-pool _charts_to_tensors
        # call holds ~4.6M Note objects + a dozen (P, C) f64 scratch
        # arrays + the f32 outputs simultaneously (~5.5 GB at 46k
        # segments — OOM-killed a 14 GB laptop). Each chunk is built
        # padded and flattened to ragged on the spot (the padded
        # scratch is per chunk); cat_pools rebases the offsets, byte-
        # identical to the whole-pool call; each chunk's Notes are
        # released as it is consumed.
        parts = []
        chunk = 2048
        for lo in range(0, len(segs), chunk):
            ch = segs[lo:lo + chunk]
            cw = seg_walls[lo:lo + chunk]
            segs[lo:lo + chunk] = [None] * len(ch)
            part = _charts_to_tensors(ch, "cpu", dtype)
            part.update(_walls_to_tensors(cw, "cpu", dtype))
            parts.append(part)
            del ch
        del segs
        if not parts:                    # empty catalog: P = 0 pool
            parts = [_charts_to_tensors([], "cpu", dtype)]
            parts[0].update(_walls_to_tensors([], "cpu", dtype))
        pool = cat_pools(parts)
        del parts
        # tmp + atomic rename: the cache lives on a SHARED network
        # volume and other pods poll for it — a direct savez exposes a
        # partial npz for the whole multi-GB write (the v17 lr-ladder
        # launch raced exactly this, 2026-08-13)
        # (tmp must END in .npz — np.savez appends the suffix otherwise
        # and the rename source would not exist)
        tmp = cpath[:-len(".npz")] + f".tmp{os.getpid()}.npz"
        np.savez(tmp, **{k: v.numpy() for k, v in pool.items()})
        os.replace(tmp, cpath)
        log(f"[ranked-pool] cached -> {os.path.basename(cpath)}",
            flush=True)
    data = np.load(cpath)
    P = data["n_notes"].shape[0]
    idx = None
    if sample is not None and sample < P:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(P, size=sample, replace=False))
    arrs = _subset_ragged({k: data[k] for k in data.files}, idx)
    out = {k: torch.as_tensor(a, device=device) for k, a in arrs.items()}
    nn, nw = arrs["n_notes"], arrs["n_walls"]
    log(f"[ranked-pool] {P if idx is None else len(idx)} segments on "
        f"{device} (C={max(1, int(nn.max())) if nn.size else 1}, "
        f"{int(nn.sum())} notes + {int(nw.sum())} walls ragged, "
        f"{sum(v.numel() * v.element_size() for v in out.values()) / 2**20:.0f}"
        f" MB)", flush=True)
    return out


def _gather_index(off, idx):
    """Flat object indices of segments `idx` in a ragged block with
    exclusive offsets `off` (P + 1,), in segment order — the CSR order
    of the subset pool. Returns (index, new_offsets)."""
    lens = off[idx + 1] - off[idx]
    n_tot = int(lens.sum())
    new_off = np.zeros(len(idx) + 1, np.int64)
    np.cumsum(lens, out=new_off[1:])
    gidx = (np.repeat(off[idx] - new_off[:-1], lens)
            + np.arange(n_tot, dtype=np.int64))
    return gidx, new_off


def _subset_ragged(arrs, idx):
    """Seeded-subsample a ragged pool (numpy dict): per-segment fields
    take rows `idx`, per-object fields take each chosen segment's
    object range, offsets are rebuilt. idx=None returns arrs as is."""
    if idx is None:
        return arrs
    from .cuda_env import NOTE_OBJ_FIELDS, WALL_OBJ_FIELDS
    gn, non = _gather_index(arrs["note_off"], idx)
    gw, wof = _gather_index(arrs["wall_off"], idx)
    out = {}
    for k, a in arrs.items():
        if k == "note_off":
            out[k] = non
        elif k == "wall_off":
            out[k] = wof
        elif k in NOTE_OBJ_FIELDS:
            out[k] = a[gn]
        elif k in WALL_OBJ_FIELDS:
            out[k] = a[gw]
        else:
            out[k] = a[idx]
    return out
