"""Default-body search (user plan 2026-09-19): fit the training envs'
per-episode outcomes on the 16 τmax draws and recommend the next
--tau-center.

    python tau_fit.py runs/<name> [--last N] [--cap 0.10] [--tmin 3]

Inputs (row-aligned, same pop order): runs/<name>/ep_stats.f32 (ep_stats.py
layout) and runs/<name>/tau_draws.f32 (22 cols; s0..s15 = the body that
played, ABSOLUTE scales relative to TAU_MAX).

Objective y = PP with unfinished maps as 0 (what upp rewards), DEMEANED
PER MAP (leaderboard id) so which map was drawn does not swamp the body
effect; also fitted: completion and accuracy (completed maps). Model: OLS
y ~ 1 + Σ_a b_a · x_a with x_a = (s_a / c_a − 1) / r, i.e. one unit = the
full +r deviation from the run's center c (given with --center, default
1.0 everywhere). HC0 standard errors. Recommendation: move each axis by
cap · b_a / max|b| along the sign of b_a for axes with |t| ≥ tmin, bounded
±cap (a step of the gradual scheme; retrain, refit, repeat until the
slopes sit inside their error bars)."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beat_arena_env.constants import TAU_AXIS_NAMES, TAU_FEAT, parse_tau_scale  # noqa: E402

EP_COLS, TAU_COLS = 16, 22


def load(run_dir, last=0):
    import ep_stats
    E = ep_stats.load(run_dir)[:, :EP_COLS]   # v1 or v2 (cols 0..15 shared)
    T = np.fromfile(os.path.join(run_dir, "tau_draws.f32"), dtype=np.float32).reshape(-1, TAU_COLS)
    n = min(len(E), len(T))
    E, T = E[-n:], T[-n:]          # both files grow from the same pops; align on the tail
    assert np.allclose(E[:, 15], T[:, 17], atol=1e-4), "ep_stats/tau_draws rows do not align"
    if last:
        E, T = E[-last:], T[-last:]
    m = E[:, 4] >= 60              # skip lead-only episodes
    return E[m], T[m]


def demean_by_map(y, lb):
    ids, inv = np.unique(lb, return_inverse=True)
    means = np.bincount(inv, y) / np.bincount(inv)
    return y - means[inv]


def ols_hc0(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ beta
    XtX_inv = np.linalg.inv(X1.T @ X1)
    meat = (X1 * (r * r)[:, None]).T @ X1
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta[1:], se[1:], beta[0]


def fit(E, T, center, r):
    X = (T[:, :TAU_FEAT] / center[None, :] - 1.0) / r
    lb = E[:, 1].astype(np.int64)
    completed = E[:, 6] > 0.5
    out = {}
    y = demean_by_map(E[:, 9], lb)                       # PP, 0 if unfinished
    out["pp"] = ols_hc0(X, y) + (float(E[:, 9].mean()),)
    out["completion"] = ols_hc0(X, E[:, 6].astype(np.float64)) + (float(completed.mean()),)
    if completed.sum() > 100:
        yc = demean_by_map(E[completed, 8], lb[completed])
        out["acc"] = ols_hc0(X[completed], yc) + (float(E[completed, 8].mean()),)
    return out, int(len(E)), int(completed.sum())


def recommend(b, se, cap, tmin):
    t = np.where(se > 0, b / np.maximum(se, 1e-12), 0.0)
    sig = np.abs(t) >= tmin
    if not sig.any():
        return np.zeros_like(b), t
    step = np.zeros_like(b)
    scale = cap / np.abs(b[sig]).max()
    step[sig] = np.clip(b[sig] * scale, -cap, cap)
    return step, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--last", type=int, default=0, help="only the last N episodes")
    p.add_argument("--center", type=str, default="default",
                   help="the run's --tau-center (16 floats, group syntax, 'default' = "
                        "the saved body TAU_BODY_DEFAULT, 'nominal' = all 1.0); default "
                        "= the saved body (a fresh trunk trains around it)")
    p.add_argument("--r", type=float, default=0.25, help="the run's --tau-rand")
    p.add_argument("--cap", type=float, default=0.10, help="max center move per axis per round")
    p.add_argument("--tmin", type=float, default=3.0, help="|t| needed to move an axis")
    args = p.parse_args()
    center = np.asarray(parse_tau_scale(args.center), dtype=np.float64)
    E, T = load(args.run_dir, args.last)
    res, n, nc = fit(E, T, center, args.r)
    print(f"{n:,} episodes ({nc:,} completed, {100 * nc / max(n, 1):.1f}%); "
          f"center = {np.round(center, 3).tolist()}")
    for name, (b, se, b0, mean) in res.items():
        print(f"\n{name}: mean {mean:.4f}; slope per +{args.r:.0%} of center (HC0 se, t):")
        for a in range(TAU_FEAT):
            t = b[a] / se[a] if se[a] > 0 else 0.0
            flag = " *" if abs(t) >= args.tmin else ""
            print(f"  {a:2d} {TAU_AXIS_NAMES[a]:<14s} {b[a]:+9.4f} ({se[a]:.4f}, t {t:+6.1f}){flag}")
    b, se, _, _ = res["pp"]
    step, t = recommend(b, se, args.cap, args.tmin)
    new = center * (1.0 + step)
    print(f"\nrecommended next center (PP objective, cap {args.cap:.0%}, |t| >= {args.tmin}):")
    moved = [f"{TAU_AXIS_NAMES[a]} {center[a]:.3f}->{new[a]:.3f}" for a in range(TAU_FEAT) if step[a] != 0]
    print("  " + (", ".join(moved) if moved else "no axis clears the t threshold — center is at a fixed point within noise"))
    print("  --tau-center " + ",".join(f"{x:.4f}" for x in new))
    print(f"  predicted PP change (linear): {float((b * step / args.r).sum()):+.2f} per episode")


if __name__ == "__main__":
    main()
