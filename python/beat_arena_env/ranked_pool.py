"""Ranked-map segment pool for CUDA-env training (sweep s2, 2026-07-29).

Slices every ranked ScoreSaber level (maps/scoresaber_catalog.json via
the bench's warm chart_cache pickles; real-NJS speeds since 2026-07-30,
speed_scale removed) into fixed-length windows and packs them into the
(P, C) tensor dict CudaBeatVecEnv eats — training on the real catalog
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

Windows cover [k*seg_len, (k+1)*seg_len) in song time, notes rebased
to window start; the first LEAD_S seconds of EVERY window are dropped
(k > 0: those notes were mid-flight at the cut; k = 0 since
2026-08-23: negative-shift intro walls otherwise reach the head on
the first tick — an unavoidable charge). Windows with < MIN_SCOREABLE
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
    key = hashlib.sha1(
        (f"{seg_len:g}_{ver}_v5_lead{lead_s:g}"
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
        for k in range(int(np.ceil((last + 1e-6) / seg_len))):
            # 2026-08-23 (user): lead gap applies to k=0 too. The
            # note-anchored parser shift (first note -> 2.0 s) can be
            # NEGATIVE, putting intro walls at t <= 0; without the gap
            # they clip to c0 = 0 and reach the head in 0.4/speed s —
            # 32 catalog segments charged an unavoidable first-tick
            # WALL_PEN (measured 19-21 ms entry). Notes are unaffected
            # (none before 2.0 s); intro walls now start >= lead_s out,
            # making "nothing materializes mid-flight" uniform.
            lo = k * seg_len + lead_s
            hi = (k + 1) * seg_len
            win = [t for t in tup if lo <= t[0] < hi]
            if len(win) > MAX_OBJECTS:
                dropped_fat += 1
                continue
            if sum(1 for t in win if t[7] != KIND_BOMB) < MIN_SCOREABLE:
                dropped_sparse += 1
                continue
            base = k * seg_len
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
    log(f"[ranked-pool] {len(segments)} segments of {seg_len:g}s "
        f"({n_wall_segs} with walls; dropped {dropped_sparse} sparse, "
        f"{dropped_fat} >{MAX_OBJECTS}-object)", flush=True)
    return segments, seg_walls


def load_ranked_pool(seg_len=25.0, sample=None, seed=0,
                     device="cuda", dtype=torch.float32,
                     catalog=None, log=print, lead_s=LEAD_S,
                     mirror=False):
    # lead_s (env v16, contact spec §11): the k>0 windows' head gap —
    # the earliest possible first note of a segment episode. Default
    # keeps the historical 1.5 s; the survival-drift postmortem
    # hypothesis #4 is that a long pre-note phase feeds the
    # survival-consolidation feedback loop.
    """(P, C) pool dict on `device`. First call builds + caches the full
    segment set (minutes); later calls load the npz (~seconds). `sample`
    takes a seeded without-replacement subset (VRAM control; every leg
    with the same seed/sample sees the same subset)."""
    catalog = catalog or os.path.join(_BASE, "maps",
                                      "scoresaber_catalog.json")
    import sys
    sys.path.insert(0, _BASE)
    import bench_scoresaber as bs
    cpath = _cache_path(seg_len, bs.parser_version(), catalog, lead_s,
                        mirror=mirror)
    if not os.path.exists(cpath):
        from .cuda_env import _charts_to_tensors, _walls_to_tensors
        segs, seg_walls = _build_segments(seg_len, catalog, log,
                                          lead_s=lead_s, mirror=mirror)
        # Chunked packing (2026-08-08): one whole-pool _charts_to_tensors
        # call holds ~4.6M Note objects + a dozen (P, C) f64 scratch
        # arrays + the f32 outputs simultaneously (~5.5 GB at 46k
        # segments — OOM-killed a 14 GB laptop). Chunks pad to the
        # global C (pad_to) so per-field concatenation is byte-identical
        # to the whole-pool call; each chunk's Notes are released as it
        # is consumed.
        cmax = max((len(s) for s in segs), default=1)
        wmax = max((len(w) for w in seg_walls), default=1)
        parts = []
        chunk = 2048
        for lo in range(0, len(segs), chunk):
            ch = segs[lo:lo + chunk]
            cw = seg_walls[lo:lo + chunk]
            segs[lo:lo + chunk] = [None] * len(ch)
            part = _charts_to_tensors(ch, "cpu", dtype, pad_to=cmax)
            part.update(_walls_to_tensors(cw, "cpu", dtype, pad_to=wmax))
            parts.append(part)
            del ch
        del segs
        if not parts:                    # empty catalog: (0, 1) pool
            parts = [_charts_to_tensors([], "cpu", dtype)]
            parts[0].update(_walls_to_tensors([], "cpu", dtype))
        pool = {}
        for k in list(parts[0]):
            pool[k] = torch.cat([p[k] for p in parts])
            for p in parts:
                del p[k]
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
    P = data["hit"].shape[0]
    idx = None
    if sample is not None and sample < P:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(P, size=sample, replace=False))
    out = {}
    n_notes = data["n_notes"][idx] if idx is not None else data["n_notes"]
    cmax = max(1, int(n_notes.max()))
    # wall tensors trim on THEIR own ragged dim (W), not the note C
    n_walls = data["n_walls"][idx] if idx is not None else data["n_walls"]
    wmax = max(1, int(n_walls.max()))
    for k in data.files:
        a = data[k]
        if idx is not None:
            a = a[idx]
        lim = wmax if k.startswith("wall_") else cmax
        if a.ndim >= 2 and a.shape[1] > lim:
            a = a[:, :lim]
        out[k] = torch.as_tensor(a, device=device)
    log(f"[ranked-pool] {out['hit'].shape[0]} segments on {device} "
        f"(C={out['hit'].shape[1]}, "
        f"{sum(v.numel() * v.element_size() for v in out.values()) / 2**20:.0f}"
        f" MB)", flush=True)
    return out
