"""Ranked-map WHOLE-LEVEL pool for CUDA-env training.

Packs every ranked ScoreSaber level (maps/scoresaber_catalog.json via
the bench's warm chart_cache pickles; real-NJS speeds since 2026-07-30)
into the ragged (CSR) tensor dict CudaBeatVecEnv eats
(cuda_env.pad_to_ragged): one pool entry = one level, played from song
time 0 to its last object + tail. Training on the real catalog directly,
no procedural generator.

One episode = one level (the only mode since 2026-09-16, user decision).
History: from 2026-07-29 the pool sliced levels into fixed 25-s windows
(stride seg_len - lead_s, <= 1,536 objects per window, episodes ~690
ticks) on the argument that whole maps (median 884 notes / 173 s) were
too long for the env's regime. The v46_whole probe (2026-09-15/16, same
recipe otherwise) retired that: trunk accuracy identical, but the
bench-faithful leaf trained on whole levels benched 2,583,201 upp vs
2,495,323 on the windows (+3.5%), catastrophic levels 20 -> 4, full
combos +12%, misses -15%, at zero throughput cost. The window machinery
(window_starts, MAX_OBJECTS, --seg-len) was removed with it.

What a level entry is: the parser anchors the first note at 2.0 s
(bench protocol); notes with hit >= lead_s are kept (so every note, at
the recipe's lead_s 0.5) and walls are clipped to [lead_s, inf) — the
same lead-gap rule the k = 0 window used, so no extra padding is added
at the front of a level. Levels with < MIN_SCOREABLE scoreable objects
are dropped; there is no object cap (the CUDA env's only per-env
per-object tensor is the 1-byte status array: N * C = 181 MB at
N = 32,768, C = 5,586). Episode limit: pool_max_steps() = the longest
level's last object + tail + 90 ticks (~18k ticks, 602 s). Built once
and cached as npz keyed on (parser version, lead_s, mirror, catalog
bytes) — every leg then pays only the npz load + a seeded subsample.
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
# The training recipe's pool (train_batch --lead-s / --mirror-augment
# defaults). Both are in the npz cache key, so anything that pre-builds
# the cache (prefetch_maps) must use these or it writes a pool no run loads.
RECIPE_LEAD_S = 0.5
RECIPE_MIRROR = True
MIN_SCOREABLE = 4     # drop near-empty tail windows

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cache_path(ver, catalog_path, lead_s, mirror=False):
    with open(catalog_path, "rb") as f:
        cat = f.read()
    # Key string history (each bump invalidated the caches on file):
    # v3 walls in the payload; v4 MAX_OBJECTS 512 -> 1536; v5 lead gap on
    # k = 0; v6 ragged (CSR) layout; v7 window stride = seg_len - lead_s;
    # v8 (2026-09-15) whole-level mode. The string below is FROZEN at the
    # v8 whole-level form ("whole_level_v8_{ver}_v7_lead{lead_s}") so the
    # caches built by the v46_whole probe (e.g. ranked_d415ec0c16a4408e.npz
    # on the pod) are reused byte-for-byte now that whole-level is the
    # only mode — do not "tidy" it.
    key = hashlib.sha1(
        (f"whole_level_v8_{ver}_v7_lead{lead_s:g}"
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


def _build_segments(catalog_path, log, lead_s=LEAD_S, mirror=False):
    """One entry per level: notes with hit >= lead_s (rebased to song
    time 0, i.e. unchanged — the parser's 2.0-s first-note anchor is the
    only front padding), walls clipped to [lead_s, inf), levels with
    < MIN_SCOREABLE scoreable objects dropped, no object cap. Byte-
    identical to the 2026-09-15 `--seg-len 0` mode."""
    import json
    import sys
    sys.path.insert(0, _BASE)
    import bench_scoresaber as bs      # load_or_parse + parser_version:
    # the EXACT bench parsing path, so training levels and bench plays
    # see byte-identical note streams for the same level
    ver = bs.parser_version()
    catalog = json.load(open(catalog_path, encoding="utf-8"))
    segments, seg_walls, dropped_sparse = [], [], 0
    meta = []   # env v47: (leaderboard_id, max_pp, max_score) per segment
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
        if last < lead_s:
            continue
        # 2026-08-23 (user): the lead gap applies from song time 0. The
        # note-anchored parser shift (first note -> 2.0 s) can be
        # NEGATIVE, putting intro walls at t <= 0; without the gap they
        # clip to c0 = 0 and reach the head in 0.4/speed s — 32 catalog
        # levels charged an unavoidable first-tick WALL_PEN. Notes are
        # unaffected (none before 2.0 s).
        lo, hi, base = lead_s, float("inf"), 0.0
        win = [t for t in tup if lo <= t[0] < hi]
        if sum(1 for t in win if t[7] != KIND_BOMB) < MIN_SCOREABLE:
            dropped_sparse += 1
            continue
        reb = [(t[0] - base,) + tuple(t[1:]) for t in win]
        wreb = _window_walls(wtups, lo, hi, base)
        segments.append([_tuple_note(t) for t in reb])
        seg_walls.append(wreb)
        meta.append(_entry_meta(entry, payload))
        if mirror:   # augmentation: every level also plays reflected
            segments.append([_tuple_note(t)
                             for t in mirror_tuples(reb)])
            seg_walls.append(mirror_walls(wreb))
            meta.append(meta[-1])
        if (i + 1) % 1000 == 0:
            log(f"[ranked-pool] {i + 1}/{len(catalog)} levels -> "
                f"{len(segments)} segments", flush=True)
    n_wall_segs = sum(1 for w in seg_walls if w)
    n_max = max((len(sg) for sg in segments), default=0)
    log(f"[ranked-pool] {len(segments)} whole-level segments "
        f"({n_wall_segs} with walls; dropped {dropped_sparse} sparse; "
        f"max {n_max} objects)", flush=True)
    return segments, seg_walls, meta


def _entry_meta(entry, payload):
    from .scoresaber import PP_PER_STAR
    # test catalogs carry only hash/difficulty: -1 / 0 / 0 then
    return (int(entry.get("leaderboard_id", -1)),
            float(entry.get("max_pp")
                  or float(entry.get("stars", 0.0)) * PP_PER_STAR),
            float(payload.get("max_score", 0)))


def _level_meta(catalog_path, lead_s=LEAD_S, mirror=False):
    """The (P, 3) level metadata for a pool built by _build_segments —
    env v47 per-map score/PP telemetry (--ep-stats). Replays the SAME
    drop rules (empty, last < lead_s, sparse) over the chart cache
    without packing any tensors, so a pool cached before the meta
    existed gets a sidecar instead of a multi-GB rebuild."""
    import json
    import sys
    sys.path.insert(0, _BASE)
    import bench_scoresaber as bs
    ver = bs.parser_version()
    catalog = json.load(open(catalog_path, encoding="utf-8"))
    meta = []
    for entry in catalog:
        payload = bs.load_or_parse(entry, os.path.join(_BASE, "maps",
                                                       "scoresaber"),
                                   os.path.join(_BASE, "maps",
                                                "chart_cache"), ver)
        tup = sorted(payload["tuples"], key=lambda t: t[0])
        if not tup or tup[-1][0] < lead_s:
            continue
        win = [t for t in tup if lead_s <= t[0]]
        if sum(1 for t in win if t[7] != KIND_BOMB) < MIN_SCOREABLE:
            continue
        meta.append(_entry_meta(entry, payload))
        if mirror:
            meta.append(meta[-1])
    return meta


def _meta_arrays(meta):
    m = np.asarray(meta, dtype=np.float64).reshape(-1, 3)
    return {"level_lb": m[:, 0].astype(np.int64),
            "level_max_pp": m[:, 1].astype(np.float32),
            "level_max_score": m[:, 2].astype(np.float32)}


def pool_max_steps(pool, tick_hz=30, margin_ticks=90):
    """Episode tick limit for a pool: the longest level's last object
    plus its tail, plus a 90-tick margin (the same margin the retired
    segment formula int(seg_len * 30) + 90 used)."""
    if int(pool["n_notes"].shape[0]) == 0:
        return margin_ticks
    end = (pool["last_hit"].double() + pool["tail"].double()).max().item()
    return int(end * tick_hz) + margin_ticks


def load_ranked_pool(sample=None, seed=0, device="cuda",
                     dtype=torch.float32, catalog=None, log=print,
                     lead_s=LEAD_S, mirror=False):
    # lead_s (env v16, contact spec §11): the front gap — notes before
    # lead_s are dropped and walls clipped to it. With the parser's
    # 2.0-s first-note anchor and the recipe's lead_s 0.5 no note is
    # ever dropped; the survival-drift postmortem hypothesis #4 (a long
    # pre-note phase feeds the survival-consolidation loop) is why the
    # recipe runs 0.5 rather than the historical 1.5.
    """Ragged pool dict on `device` (cuda_env.pad_to_ragged layout).
    First call builds + caches the whole-level set (minutes); later
    calls load the npz (~seconds). `sample` takes a seeded
    without-replacement subset (VRAM control; every leg with the same
    seed/sample sees the same subset)."""
    catalog = catalog or os.path.join(_BASE, "maps",
                                      "scoresaber_catalog.json")
    import sys
    sys.path.insert(0, _BASE)
    import bench_scoresaber as bs
    cpath = _cache_path(bs.parser_version(), catalog, lead_s,
                        mirror=mirror)
    if not os.path.exists(cpath):
        from .cuda_env import (_charts_to_tensors, _walls_to_tensors,
                               cat_pools)
        segs, seg_walls, meta = _build_segments(catalog, log, lead_s=lead_s,
                                                mirror=mirror)
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
        pool.update({k: torch.as_tensor(v)
                     for k, v in _meta_arrays(meta).items()})
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
    arrs = {k: data[k] for k in data.files}
    if "level_lb" not in arrs:
        # pool cached before env v47: level meta from a sidecar (built
        # once from the chart cache, seconds; the frozen pool key stays)
        mpath = cpath[:-len(".npz")] + ".meta.npz"
        if not os.path.exists(mpath):
            log("[ranked-pool] building level meta sidecar (env v47)",
                flush=True)
            ma = _meta_arrays(_level_meta(catalog, lead_s, mirror))
            assert ma["level_lb"].shape[0] == P, \
                f"meta {ma['level_lb'].shape[0]} != pool {P} levels"
            tmp = mpath[:-len(".npz")] + f".tmp{os.getpid()}.npz"
            np.savez(tmp, **ma)
            os.replace(tmp, mpath)
        md = np.load(mpath)
        arrs.update({k: md[k] for k in md.files})
    idx = None
    if sample is not None and sample < P:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(P, size=sample, replace=False))
    arrs = _subset_ragged(arrs, idx)
    out = {k: torch.as_tensor(a, device=device) for k, a in arrs.items()}
    nn, nw = arrs["n_notes"], arrs["n_walls"]
    log(f"[ranked-pool] {P if idx is None else len(idx)} levels on "
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
