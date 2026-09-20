"""Context-window sufficiency over the ranked catalogue (user, 2026-09-20).

The entity window shows the N_NOTE_SLOTS (24) pending objects nearest the
head (notes, bombs, chain links, walls; nearest-first by 3-D distance,
objects behind the miss boundary excluded — the lib.rs / .cu write_obs
rule). For each hand this asks: how often is the hand's UPCOMING note
(the earliest pending note of its colour) NOT in the window? Two views:

  tick-level  P(next note of hand h not visible | one exists), over the
              map's 30 Hz ticks
  note-level  for each coloured note n with a previous same-colour note p:
              was n visible at the tick the hand became free (hit_p)?
              (= could the hand have started moving toward n as soon as
              it finished p) plus the anticipation coverage
              (hit_n − t_first_visible) / (hit_n − hit_p), clipped to 1

Perfect play is assumed (a note leaves the pool at its hit time; bombs
and walls leave when they pass the head), the head stands at
(0, HEAD_Y, PLANE_Z0). One-colour maps are skipped. Runs on the laptop's
chart cache (maps/chart_cache) with multiprocessing.

  python tools/context_window_analysis.py --slots 24 32 48 --workers 8
"""
import argparse
import glob
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
sys.path.insert(0, PY)
# constants only (importing the package pulls in gymnasium/torch, which
# the laptop does not have)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "_ba_constants", os.path.join(PY, "beat_arena_env", "constants.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)
KIND_BOMB, KIND_CHAIN_HEAD, KIND_NOTE = _c.KIND_BOMB, _c.KIND_CHAIN_HEAD, _c.KIND_NOTE
PLANE_Z0, WALL_FRONT_OFF = _c.PLANE_Z0, _c.WALL_FRONT_OFF

HZ = 30.0
HEAD_X, HEAD_Y = 0.0, 1.60
M_EXTRA = 64            # candidate window beyond the largest K (exactness checked)


def analyze(args):
    path, stars, slot_list = args
    with open(path, "rb") as f:
        d = pickle.load(f)
    tup = d["tuples"]
    if not tup:
        return None
    hit = np.array([t[0] for t in tup])
    x = np.array([t[1] for t in tup])
    y = np.array([t[2] for t in tup])
    col = np.array([t[3] for t in tup])
    spd = np.array([t[5] for t in tup])
    kind = np.array([t[7] for t in tup])
    is_note = (kind == KIND_NOTE) | (kind == KIND_CHAIN_HEAD)
    if not ((is_note & (col == 0)).any() and (is_note & (col == 1)).any()):
        return None                                   # one-handed map
    order = np.argsort(hit, kind="stable")
    hit, x, y, col, spd, kind, is_note = (a[order] for a in
                                          (hit, x, y, col, spd, kind, is_note))
    n = len(hit)
    # pool exit time: notes/links at their hit (perfect play); bombs when
    # behind the miss boundary (dz < 0)
    exit_t = np.where(kind == KIND_BOMB, hit + (-PLANE_Z0) / spd, hit)
    walls = d["walls"]
    W = len(walls)
    if W:
        wt = np.array([w[0] for w in walls]); wcx = np.array([w[1] for w in walls])
        wcy = np.array([w[2] for w in walls]); ww = np.array([w[3] for w in walls])
        wh = np.array([w[4] for w in walls]); wl = np.array([w[5] for w in walls])
        ws = np.array([w[6] for w in walls])
    Kmax = max(slot_list)
    M = Kmax + M_EXTRA
    ticks = np.arange(0.0, hit[-1] + 1.0 / HZ, 1.0 / HZ)
    T = len(ticks)
    # first pending note index per tick (pool = exit_t >= t); exit_t is
    # near-sorted (bombs linger a hair) — use the sorted-hit lower bound
    # and mask exits inside the candidate window
    p0 = np.searchsorted(hit, ticks - (-PLANE_Z0) / spd.min(), side="left")
    cand = p0[:, None] + np.arange(M)[None, :]           # (T, M)
    valid = cand < n
    ci = np.minimum(cand, n - 1)
    dz = (hit[ci] - ticks[:, None]) * spd[ci] - PLANE_Z0
    pend = valid & (exit_t[ci] >= ticks[:, None]) & (dz >= 0)
    dx = x[ci] - HEAD_X
    dy = y[ci] - HEAD_Y
    d2 = np.where(pend, dx * dx + dy * dy + dz * dz, np.inf)
    if W:
        dzf = (wt[None, :] - ticks[:, None]) * ws[None, :] - PLANE_Z0 - WALL_FRONT_OFF
        alive = (dzf + wl[None, :]) >= 0
        sx = np.maximum(np.abs(HEAD_X - wcx)[None, :] - ww[None, :] * 0.5, 0)
        sy = np.maximum(np.abs(HEAD_Y - wcy)[None, :] - wh[None, :] * 0.5, 0)
        sz = np.maximum(np.maximum(dzf, -(dzf + wl[None, :])), 0)
        wd2 = np.where(alive, sx * sx + sy * sy + sz * sz, np.inf)
        d2 = np.concatenate([d2, wd2], axis=1)
    # rank of every candidate among the pending objects at its tick
    rank = np.argsort(np.argsort(d2, axis=1, kind="stable"), axis=1)[:, :M]
    # exactness: the farthest-in-time note candidate's z alone must beat
    # the Kmax-th kept distance (the kernel's early-exit invariant)
    kth = np.sort(d2, axis=1)[:, Kmax - 1]
    last_ok = ~valid[:, -1] | (dz[:, -1] * dz[:, -1] >= kth) | ~np.isfinite(kth)
    inexact = int((~last_ok).sum())
    res = {"stars": stars, "n_notes": int(is_note.sum()), "ticks": T,
           "inexact_ticks": inexact, "per_K": {}}
    # next note of each hand per tick = earliest pending coloured note
    # within the window (it is always inside: it is the first pending of
    # its colour by time)
    for K in slot_list:
        vis = (rank < K) & pend                                # (T, M)
        out = {}
        for h in (0, 1):
            ish = is_note[ci] & (col[ci] == h) & pend
            has = ish.any(axis=1)
            first = np.argmax(ish, axis=1)                    # first True
            nvis = vis[np.arange(T), first]
            out[f"h{h}_ticks"] = int(has.sum())
            out[f"h{h}_ticks_invisible"] = int((has & ~nvis).sum())
            # note-level: t_first_visible for every note of colour h
            idx_h = np.nonzero(is_note & (col == h))[0]
            # first tick at which note j is visible: scan (T, M) where ci==j
            tvis = np.full(n, np.inf)
            tt, mm = np.nonzero(vis & (col[ci] == h) & is_note[ci])
            js = ci[tt, mm]
            # first occurrence per note (ticks ascending)
            firsts = {}
            for t_i, j in zip(tt, js):
                if j not in firsts:
                    firsts[j] = ticks[t_i]
            for j, tv in firsts.items():
                tvis[j] = tv
            prev = np.full(len(idx_h), np.nan)
            prev[1:] = hit[idx_h[:-1]]
            cur = hit[idx_h]
            gap = cur - prev
            ok = np.isfinite(gap) & (gap > 0)
            lead = cur - tvis[idx_h]
            cov = np.clip(lead / np.where(ok, gap, 1.0), 0, 1)
            visible_at_free = tvis[idx_h] <= prev + 1e-9
            out[f"h{h}_notes"] = int(ok.sum())
            out[f"h{h}_notes_not_visible_at_free"] = int((ok & ~visible_at_free).sum())
            out[f"h{h}_cov_sum"] = float(cov[ok].sum())
            out[f"h{h}_lead_hist"] = np.histogram(
                np.clip(lead[ok], 0, 10), bins=[0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10.01])[0].tolist()
            out[f"h{h}_gap_hist"] = np.histogram(
                np.clip(gap[ok], 0, 10), bins=[0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10.01])[0].tolist()
        res["per_K"][K] = out
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=os.path.join(PY, "maps/scoresaber_catalog.json"))
    ap.add_argument("--chart-cache", default=os.path.join(PY, "maps/chart_cache"))
    ap.add_argument("--slots", type=int, nargs="+", default=[24, 32, 48])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(PY, "analysis/context_window_analysis.json"))
    a = ap.parse_args()
    cat = json.load(open(a.catalog))
    jobs = []
    for e in cat:
        g = glob.glob(os.path.join(a.chart_cache, f"{e['hash']}_{e['difficulty']}_*.pkl"))
        if g:
            jobs.append((g[0], float(e["stars"]), a.slots))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} levels with a cached chart", flush=True)
    with Pool(a.workers) as pool:
        results = [r for r in pool.imap_unordered(analyze, jobs, chunksize=8) if r]
    print(f"{len(results)} two-handed levels analysed "
          f"({len(jobs) - len(results)} one-handed/empty skipped)", flush=True)
    bands = [(0, 4), (4, 6), (6, 8), (8, 10), (10, 99)]
    summary = {}
    for K in a.slots:
        agg = {}
        for lo, hi in [(0, 99)] + bands:
            rs = [r for r in results if lo <= r["stars"] < hi]
            row = {"levels": len(rs), "inexact_ticks": sum(r["inexact_ticks"] for r in rs),
                   "ticks": sum(r["ticks"] for r in rs)}
            for h in (0, 1):
                tk = sum(r["per_K"][K][f"h{h}_ticks"] for r in rs)
                ti = sum(r["per_K"][K][f"h{h}_ticks_invisible"] for r in rs)
                nn = sum(r["per_K"][K][f"h{h}_notes"] for r in rs)
                nf = sum(r["per_K"][K][f"h{h}_notes_not_visible_at_free"] for r in rs)
                cs = sum(r["per_K"][K][f"h{h}_cov_sum"] for r in rs)
                lh = np.sum([r["per_K"][K][f"h{h}_lead_hist"] for r in rs], axis=0).tolist()
                gh = np.sum([r["per_K"][K][f"h{h}_gap_hist"] for r in rs], axis=0).tolist()
                row[f"h{h}"] = {"ticks": tk, "tick_invisible_frac": ti / max(tk, 1),
                                "notes": nn, "not_visible_at_free_frac": nf / max(nn, 1),
                                "mean_coverage": cs / max(nn, 1),
                                "lead_hist": lh, "gap_hist": gh}
            agg[f"{lo}-{hi}"] = row
        summary[K] = agg
    with open(a.out, "w") as f:
        json.dump({"slots": a.slots, "levels": len(results), "summary": summary}, f)
    print(f"-> {a.out}")
    for K in a.slots:
        print(f"\n== K = {K} slots")
        print(f"{'stars':>7} {'levels':>6} | {'L tick-invis':>12} {'R tick-invis':>12} | "
              f"{'L not@free':>10} {'R not@free':>10} | {'L cov':>6} {'R cov':>6}")
        for band, row in summary[K].items():
            L, R = row["h0"], row["h1"]
            print(f"{band:>7} {row['levels']:>6} | {L['tick_invisible_frac']:>11.2%} "
                  f"{R['tick_invisible_frac']:>11.2%} | {L['not_visible_at_free_frac']:>9.2%} "
                  f"{R['not_visible_at_free_frac']:>9.2%} | {L['mean_coverage']:>6.3f} "
                  f"{R['mean_coverage']:>6.3f}")
        ine = summary[K]["0-99"]["inexact_ticks"]
        print(f"(candidate-window exactness violations: {ine} ticks)")


if __name__ == "__main__":
    main()
