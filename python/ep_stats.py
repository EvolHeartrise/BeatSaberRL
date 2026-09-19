"""env v47 --ep-stats reader: per-map score / accuracy / PP from the
training envs' own plays.

    python ep_stats.py runs/<name>            # summary + top/bottom maps
    python ep_stats.py runs/<name> --tau      # + acc/PP vs body (joins tau_draws.f32)

Reads runs/<name>/ep_stats.f32 (raw float32 rows, EP_STATS_COLS = 16):
  0 chart_idx  1 leaderboard_id  2 max_pp  3 max_score  4 ticks  5 fell
  6 completed  7 cw_score  8 acc  9 pp  10 max_combo  11 breaks  12 good
  13 bad  14 miss  15 ep_return
Rows are written in the SAME order as tau_draws.f32 (the same pop), so
row i of both files is the same episode when both logs are on — that is
what --tau joins on. Plays are the sampled training policy at the drawn
body: tracking-grade, lower than the greedy bench; compare across
updates / bodies, not against bench upp."""
import argparse
import os
import sys

import numpy as np

NCOL = 16


def load(run_dir):
    a = np.fromfile(os.path.join(run_dir, "ep_stats.f32"), dtype=np.float32)
    return a.reshape(-1, NCOL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--last", type=int, default=0,
                   help="only the last N rows (0 = all)")
    p.add_argument("--tau", action="store_true",
                   help="join tau_draws.f32 by row and regress completed-"
                        "map acc / PP on each tau group's mean scale")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()
    R = load(args.run_dir)
    if args.last:
        R = R[-args.last:]
    R = R[R[:, 4] >= 60]
    done = R[:, 6] > 0.5
    C = R[done]
    print(f"{R.shape[0]:,} episodes (ticks >= 60), completed {100 * done.mean():.1f}%")
    if not C.shape[0]:
        return
    print(f"completed: acc {C[:, 8].mean():.4f}  pp mean {C[:, 9].mean():.1f} "
          f"median {np.median(C[:, 9]):.1f} max {C[:, 9].max():.1f}  "
          f"FC {100 * (C[:, 11] == 0).mean():.1f}%")
    # per-map (leaderboard) aggregation over completed plays
    lbs = C[:, 1].astype(np.int64)
    ids, inv, cnt = np.unique(lbs, return_inverse=True, return_counts=True)
    acc_m = np.bincount(inv, C[:, 8]) / cnt
    pp_m = np.bincount(inv, C[:, 9]) / cnt
    maxpp = np.bincount(inv, C[:, 2]) / cnt
    print(f"{len(ids):,} distinct maps completed; mean per-map PP "
          f"{pp_m.mean():.1f}; sum of per-map mean PP {pp_m.sum():,.0f}")
    order = np.argsort(pp_m)
    print(f"\ntop {args.top} maps by mean PP (lb_id  plays  acc  pp / maxpp):")
    for i in order[::-1][:args.top]:
        print(f"  {ids[i]:8d}  {cnt[i]:4d}  {acc_m[i]:.4f}  {pp_m[i]:7.1f} / {maxpp[i]:.0f}")
    print(f"\nbottom {args.top}:")
    for i in order[:args.top]:
        print(f"  {ids[i]:8d}  {cnt[i]:4d}  {acc_m[i]:.4f}  {pp_m[i]:7.1f} / {maxpp[i]:.0f}")
    if args.tau:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from beat_arena_env.constants import TAU_GROUPS
        T = np.fromfile(os.path.join(args.run_dir, "tau_draws.f32"),
                        dtype=np.float32).reshape(-1, 22)
        n = min(len(T), len(load(args.run_dir)))
        T = T[-len(load(args.run_dir)):][:n] if args.last == 0 else T[-args.last:]
        Rf = load(args.run_dir)[-n:] if args.last == 0 else load(args.run_dir)[-args.last:]
        m = Rf[:, 4] >= 60
        Rf, T = Rf[m], T[m]
        assert np.allclose(Rf[:, 15], T[:, 17]), "row join mismatch (ep_return)"
        dm = Rf[:, 6] > 0.5
        print("\nd(metric)/d(+25% group scale), completed maps "
              f"(n={int(dm.sum()):,}):")
        for name, col in (("acc", 8), ("pp", 9)):
            y = Rf[dm, col]
            parts = []
            for g, axes in TAU_GROUPS.items():
                x = (T[dm][:, list(axes)].mean(1) - 1.0) / 0.25
                xc = x - x.mean()
                den = float((xc * xc).sum())
                slope = float((xc * (y - y.mean())).sum() / den) if den > 0 else 0.0
                parts.append(f"{g} {slope:+.4f}")
            print(f"  {name}: " + "  ".join(parts))
        y = Rf[:, 6]
        parts = []
        for g, axes in TAU_GROUPS.items():
            x = (T[:, list(axes)].mean(1) - 1.0) / 0.25
            xc = x - x.mean()
            den = float((xc * xc).sum())
            slope = float((xc * (y - y.mean())).sum() / den) if den > 0 else 0.0
            parts.append(f"{g} {slope:+.4f}")
        print("  completed: " + "  ".join(parts))


if __name__ == "__main__":
    main()
