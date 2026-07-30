"""Ranked-map segment pool for CUDA-env training (sweep s2, 2026-07-29).

Slices every ranked ScoreSaber level (maps/scoresaber_catalog.json via
the bench's warm chart_cache pickles; real-NJS speeds since 2026-07-30,
speed_scale removed) into fixed-length windows and packs them into the
(P, C) tensor dict CudaBeatVecEnv eats — training on the real catalog
directly, no procedural generator, to probe the procedural->real gap
behind the ~87% accuracy ceiling.

SEGMENTS, not whole maps, by design (user decision 2026-07-29): median
ranked map is 884 notes / 173 s vs the env's ~90-object / 690-step
regime, and 1,235 maps exceed the kernel's C~1536 smem-scratch
occupancy budget. 25 s windows keep C in the hundreds, episode length
uniform, and every kernel/graph shape static.

Windows cover [k*seg_len, (k+1)*seg_len) in song time, notes rebased
to window start; for k > 0 the first LEAD_S seconds are dropped (those
notes were mid-flight at the cut). Windows with < MIN_SCOREABLE
scoreable objects or > MAX_OBJECTS total are dropped (the latter is
the smem guard; counts are reported). Built once and cached as npz
keyed on (parser version, seg_len, catalog bytes) — every sweep leg
then pays only the npz load + a seeded subsample.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

from .chart import Note
from .constants import KIND_BOMB

LEAD_S = 1.5          # k>0 windows: drop notes cut mid-flight
MIN_SCOREABLE = 4     # drop near-empty tail windows
MAX_OBJECTS = 512     # smem guard: C*4B must stay well under 6 KB/block

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cache_path(seg_len, ver, catalog_path):
    with open(catalog_path, "rb") as f:
        cat = f.read()
    key = hashlib.sha1(
        f"{seg_len:g}_{ver}_v2".encode() + cat
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


def _build_segments(seg_len, catalog_path, log):
    import json
    import sys
    sys.path.insert(0, _BASE)
    import bench_scoresaber as bs      # load_or_parse + parser_version:
    # the EXACT bench parsing path, so training segments and bench plays
    # see byte-identical note streams for the same level
    ver = bs.parser_version()
    catalog = json.load(open(catalog_path, encoding="utf-8"))
    segments, dropped_sparse, dropped_fat = [], 0, 0
    for i, entry in enumerate(catalog):
        payload = bs.load_or_parse(entry, os.path.join(_BASE, "maps",
                                                       "scoresaber"),
                                   os.path.join(_BASE, "maps",
                                                "chart_cache"), ver)
        tup = sorted(payload["tuples"], key=lambda t: t[0])
        if not tup:
            continue
        last = tup[-1][0]
        for k in range(int(np.ceil((last + 1e-6) / seg_len))):
            lo = k * seg_len + (LEAD_S if k else 0.0)
            hi = (k + 1) * seg_len
            win = [t for t in tup if lo <= t[0] < hi]
            if len(win) > MAX_OBJECTS:
                dropped_fat += 1
                continue
            if sum(1 for t in win if t[7] != KIND_BOMB) < MIN_SCOREABLE:
                dropped_sparse += 1
                continue
            base = k * seg_len
            segments.append([_tuple_note((t[0] - base,) + tuple(t[1:]))
                             for t in win])
        if (i + 1) % 1000 == 0:
            log(f"[ranked-pool] {i + 1}/{len(catalog)} levels -> "
                f"{len(segments)} segments", flush=True)
    log(f"[ranked-pool] {len(segments)} segments of {seg_len:g}s "
        f"(dropped {dropped_sparse} sparse, {dropped_fat} >"
        f"{MAX_OBJECTS}-object)", flush=True)
    return segments


def load_ranked_pool(seg_len=25.0, sample=None, seed=0,
                     device="cuda", dtype=torch.float32,
                     catalog=None, log=print):
    """(P, C) pool dict on `device`. First call builds + caches the full
    segment set (minutes); later calls load the npz (~seconds). `sample`
    takes a seeded without-replacement subset (VRAM control; every leg
    with the same seed/sample sees the same subset)."""
    catalog = catalog or os.path.join(_BASE, "maps",
                                      "scoresaber_catalog.json")
    import sys
    sys.path.insert(0, _BASE)
    import bench_scoresaber as bs
    cpath = _cache_path(seg_len, bs.parser_version(), catalog)
    if not os.path.exists(cpath):
        from .cuda_env import _charts_to_tensors
        segs = _build_segments(seg_len, catalog, log)
        pool = _charts_to_tensors(segs, "cpu", dtype)
        np.savez(cpath, **{k: v.numpy() for k, v in pool.items()})
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
    for k in data.files:
        a = data[k]
        if idx is not None:
            a = a[idx]
        if a.ndim >= 2 and a.shape[1] > cmax:
            a = a[:, :cmax]
        out[k] = torch.as_tensor(a, device=device)
    log(f"[ranked-pool] {out['hit'].shape[0]} segments on {device} "
        f"(C={out['hit'].shape[1]}, "
        f"{sum(v.numel() * v.element_size() for v in out.values()) / 2**20:.0f}"
        f" MB)", flush=True)
    return out
