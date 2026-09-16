#!/usr/bin/env python3
"""env v46 body-draw regression (user design 2026-09-16): which joint
torque-limit scales help?

Reads runs/<name>/tau_draws.f32 (raw float32 rows, 22 columns:
s0..s15 (per-axis scales), ep_ticks, ep_return, fell, good_cuts,
score_pts, chart_idx — see cuda_env.pop_episodes) and prints, per axis
and per group, ridge-regression coefficients of three outcomes on the
centred scales ((s - 1)/0.25, so a coefficient is "per +25% torque"):
fall rate (fell), accuracy (score_pts / (115 * good_cuts), cut episodes
only) and return, with bootstrap 90% CIs.

usage: python tau_regress.py runs/<name>/tau_draws.f32 [--min-ticks 60]
       [--boot 200] [--ridge 1e-3] [--last N]
"""
import argparse
import sys

import numpy as np

from beat_arena_env.constants import TAU_AXIS_NAMES, TAU_GROUPS, TAU_FEAT

NCOL = 22


def load(path, last=None):
    a = np.fromfile(path, dtype=np.float32)
    if a.size % NCOL:
        a = a[: a.size - a.size % NCOL]      # a torn tail write
    a = a.reshape(-1, NCOL)
    if last:
        a = a[-last:]
    return a


def ridge(X, y, lam):
    Xc = np.hstack([X, np.ones((X.shape[0], 1))])
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    A[-1, -1] -= lam
    return np.linalg.solve(A, Xc.T @ y)[:-1]


def fit(X, y, lam, boot, rng):
    b = ridge(X, y, lam)
    if boot <= 0 or X.shape[0] < 50:
        return b, None
    bs = []
    n = X.shape[0]
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        bs.append(ridge(X[idx], y[idx], lam))
    bs = np.array(bs)
    return b, (np.percentile(bs, 5, axis=0), np.percentile(bs, 95, axis=0))


def report(name, b, ci, fmt="{:+.4f}"):
    print(f"\n{name}: coefficient per +25% torque on the axis")
    for a in range(TAU_FEAT):
        c = f"  [{fmt.format(ci[0][a])}, {fmt.format(ci[1][a])}]" if ci else ""
        star = " *" if ci and (ci[0][a] > 0 or ci[1][a] < 0) else ""
        print(f"  {TAU_AXIS_NAMES[a]:12s} {fmt.format(b[a])}{c}{star}")
    print("  groups (mean of member axes):")
    for g, axes in TAU_GROUPS.items():
        print(f"    {g:7s} {fmt.format(float(np.mean(b[list(axes)])))}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--min-ticks", type=int, default=60,
                   help="drop episodes shorter than this (the 2-s lead)")
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--last", type=int, default=None,
                   help="use only the last N episodes (late-training body)")
    args = p.parse_args()
    a = load(args.path, args.last)
    a = a[a[:, 16] >= args.min_ticks]
    if a.shape[0] < 10:
        sys.exit(f"only {a.shape[0]} episodes after filtering")
    X = (a[:, :TAU_FEAT] - 1.0) / 0.25
    fell = a[:, 18]
    ret = a[:, 17]
    cuts = a[:, 19]
    acc = np.where(cuts > 0, a[:, 20] / (115.0 * np.maximum(cuts, 1)), np.nan)
    rng = np.random.default_rng(0)
    print(f"{a.shape[0]:,} episodes | fall rate {fell.mean():.4f} | mean acc "
          f"{np.nanmean(acc):.4f} | mean return {ret.mean():.3f} | mean length "
          f"{a[:, 16].mean():.0f} ticks")
    b, ci = fit(X, fell, args.ridge, args.boot, rng)
    report("FALL RATE (lower is better)", b, ci)
    m = ~np.isnan(acc)
    b, ci = fit(X[m], acc[m], args.ridge, args.boot, rng)
    report("ACCURACY (higher is better)", b, ci)
    b, ci = fit(X, ret, args.ridge, args.boot, rng)
    report("RETURN per tick-sum (higher is better)", b, ci, "{:+.3f}")
    print("\n* = bootstrap 90% CI excludes 0")


if __name__ == "__main__":
    main()
