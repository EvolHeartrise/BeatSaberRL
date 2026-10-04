"""env v47 --ep-stats reader: per-map score / accuracy / PP from the
training envs' own plays.

    python ep_stats.py runs/<name>            # summary + top/bottom maps
    python ep_stats.py runs/<name> --tau      # + acc/PP vs body (joins tau_draws.f32)

Reads runs/<name>/ep_stats.f32 (raw float32 rows). Two layouts:

  v1 (2026-09-19 .. 2026-10-04): no header, EP_STATS_COLS_V1 = 16 cols
    0 chart_idx  1 leaderboard_id  2 max_pp  3 max_score  4 ticks  5 fell
    6 completed  7 cw_score  8 acc  9 pp  10 max_combo  11 breaks  12 good
    13 bad  14 miss  15 ep_return
  v2 (2026-10-04): a HEADER ROW first — [HEADER_MAGIC, 2, ncol, 0, ...]
    (ncol floats; a v1 file's first float is a chart index >= 0, so the
    two never collide) — then EP_STATS_COLS = 29 cols: the 16 above
    unchanged, then
    16 update (the trainer's update index the row was popped at)
    17..22 left hand  [pre, pre_max, post, post_max, prec, prec_max]
    23..28 right hand [the same]
    — the episode's raw note points before the combo multiplier
    (pre-swing out of 70, post-swing / follow-through out of 30,
    precision out of 15 per note cut; chain links score none of them).
    Levels resolve through chart_idx -> the run's levels.json sidecar
    (ranked_pool.level_table: SS / BL ids, stars, max scores, ...).
  v3 (2026-10-04): the same header row, EP_STATS_COLS = 33 cols: the 29
    of v2 unchanged, then
    29 wreset_l  30 wreset_r  31 wswing_l  32 wswing_r
    — the episode's WRIST RESETS and swings per hand (lib.rs WristReset:
    a good note cut whose bend-velocity sign equals the previous swing's
    with a >= 30 deg reversal of the bend in between is a reset; hits of
    one continuous bend motion are one swing). The trainer upgrades a v2
    file IN PLACE when it resumes onto it (the v2 rows get NaN there).

load() returns the data rows of any layout (header stripped); cols
0..15 mean the same in all, 16..28 in v2 and v3, so older readers keep
working on their prefix.
Rows are written in the SAME order as tau_draws.f32 (the same pop), so
row i of both files is the same episode when both logs are on — that is
what --tau joins on. Plays are the sampled training policy at the drawn
body: tracking-grade, lower than the greedy bench; compare across
updates / bodies, not against bench upp."""
import argparse
import os
import sys

import numpy as np

NCOL = 16                 # v1 width (kept for old callers)
EP_STATS_COLS_V1 = 16
EP_STATS_COLS_V2 = 29
EP_STATS_COLS = 33        # v3
EP_STATS_VERSION = 3
HEADER_MAGIC = -1234567.0  # exact in float32
COL_UPDATE = 16
COL_PART = 17             # 12 swing cols: hand-major [L, R] x PART_FIELDS
COL_WRS = 29              # v3: [resets L, resets R, swings L, swings R]
WRS_FIELDS = ("wreset_l", "wreset_r", "wswing_l", "wswing_r")
PART_FIELDS = ("pre", "pre_max", "post", "post_max", "prec", "prec_max")
COLUMNS_V1 = ("chart_idx", "leaderboard_id", "max_pp", "max_score", "ticks",
              "fell", "completed", "cw_score", "acc", "pp", "max_combo",
              "breaks", "good", "bad", "miss", "ep_return")
COLUMNS = COLUMNS_V1 + ("update",) + tuple(
    f"{f}_{h}" for h in ("l", "r") for f in PART_FIELDS) + WRS_FIELDS
assert len(COLUMNS) == EP_STATS_COLS
# the env record's swing / wrist-reset columns (cuda_env.EP_PART_COL,
# EP_WRS_COL)
EP_REC_PART_COL = 28
EP_REC_WRS_COL = 40


def header_row(ncol=EP_STATS_COLS, version=EP_STATS_VERSION):
    h = np.zeros(ncol, dtype=np.float32)
    h[0], h[1], h[2] = HEADER_MAGIC, version, ncol
    return h


def file_layout(path):
    """(version, ncol, header_bytes) of an ep_stats file; (None, None, 0)
    for a missing / empty one. No header = v1 (16 cols)."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return None, None, 0
    if len(head) < 4:
        return None, None, 0
    first = np.frombuffer(head[:4], dtype=np.float32)[0]
    if first == np.float32(HEADER_MAGIC):
        if len(head) < 12:
            return None, None, 0       # header still being written
        _, ver, ncol = np.frombuffer(head, dtype=np.float32)
        return int(ver), int(ncol), int(ncol) * 4
    return 1, EP_STATS_COLS_V1, 0


def read_rows(path, offset=0):
    """Incremental read: (rows, new_offset, version, ncol). `offset` is a
    byte offset previously returned (0 = from the start); only whole rows
    are consumed, so a file caught mid-write (rsync, a live trainer) is
    simply read further next time."""
    ver, ncol, hb = file_layout(path)
    if ver is None:
        return np.zeros((0, 0), np.float32), offset, None, None
    start = max(offset, hb)
    size = os.path.getsize(path)
    rb = ncol * 4
    n = max(0, (size - start) // rb)
    if n == 0:
        return np.zeros((0, ncol), np.float32), start, ver, ncol
    with open(path, "rb") as f:
        f.seek(start)
        buf = f.read(n * rb)
    n = len(buf) // rb
    rows = np.frombuffer(buf[:n * rb], dtype=np.float32).reshape(n, ncol)
    return rows, start + n * rb, ver, ncol


def load(run_dir):
    """Every data row of runs/<name>/ep_stats.f32 (either layout)."""
    rows, _, _, _ = read_rows(os.path.join(run_dir, "ep_stats.f32"))
    return rows


def build_rows(rows, ep_meta, update):
    """Trainer side: one popped batch of episode records (cuda_env
    pop_episodes, (M, EP_REC)) -> (M, EP_STATS_COLS) float32 ep_stats rows.
    ep_meta: the pool's per-chart arrays level_lb / level_max_pp /
    level_max_score. acc = cw_score / max_score (the env's computed
    full-combo max) and pp = SS curve(acc) x max_pp, 0 for an unfinished
    level — the [pp] line's tracking numbers; the dashboard's best-
    possible totals recompute both per leaderboard from cw_score."""
    from beat_arena_env.scoresaber import pp_multiplier
    ci = rows[:, 21].astype(np.int64)
    max_score = ep_meta["level_max_score"][ci]
    acc = np.where(max_score > 0, rows[:, 22] / np.maximum(max_score, 1.0), 0.0)
    acc = np.clip(acc, 0.0, 1.0)
    completed = rows[:, 25] > 0.5
    pp = np.array([pp_multiplier(float(x)) for x in acc], dtype=np.float32) \
        * ep_meta["level_max_pp"][ci]
    pp = np.where(completed, pp, 0.0)   # an unfinished map scores no PP
    if rows.shape[1] >= EP_REC_PART_COL + 12:
        part = rows[:, EP_REC_PART_COL:EP_REC_PART_COL + 12]
    else:                               # an env without the swing cols
        part = np.full((rows.shape[0], 12), np.nan, np.float32)
    if rows.shape[1] >= EP_REC_WRS_COL + 4:
        wrs = rows[:, EP_REC_WRS_COL:EP_REC_WRS_COL + 4]
    else:                               # an env without wrist resets
        wrs = np.full((rows.shape[0], 4), np.nan, np.float32)
    return np.column_stack([
        rows[:, 21], ep_meta["level_lb"][ci].astype(np.float32),
        ep_meta["level_max_pp"][ci], max_score,
        rows[:, 16], rows[:, 18], rows[:, 25], rows[:, 22], acc, pp,
        rows[:, 23], rows[:, 24], rows[:, 19], rows[:, 26], rows[:, 27],
        rows[:, 17], np.full(rows.shape[0], float(update)), part, wrs,
    ]).astype(np.float32)


def upgrade_v2_file(path):
    """Rewrite a v2 ep_stats file as v3 in place (temp file + rename):
    the v3 header, every v2 row + 4 NaN wrist-reset cols. Returns the
    row count."""
    rows, _, ver, ncol = read_rows(path, 0)
    assert (ver, ncol) == (2, EP_STATS_COLS_V2), (ver, ncol)
    out = np.full((len(rows), EP_STATS_COLS), np.nan, np.float32)
    out[:, :EP_STATS_COLS_V2] = rows
    tmp = path + ".v3tmp"
    with open(tmp, "wb") as f:
        f.write(header_row().tobytes())
        f.write(out.tobytes())
    os.replace(tmp, path)
    return len(rows)


class EpStatsWriter:
    """Appends v3 rows to <run_dir>/ep_stats.f32, writing the header row
    on a fresh file. A run dir that already holds a v2 file — a resume
    across the v3 change — has it upgraded in place (upgrade_v2_file: the
    old rows keep their place, NaN wrist-reset cols); a v1 (headerless)
    file is moved aside to ep_stats.v1.f32 first (tau_fit / ep_stats
    --tau align on the tail, so tau_draws.f32 keeps working)."""

    def __init__(self, run_dir, log=print):
        self.path = os.path.join(run_dir, "ep_stats.f32")
        ver, ncol, _ = file_layout(self.path)
        if (ver, ncol) == (2, EP_STATS_COLS_V2):
            n = upgrade_v2_file(self.path)
            log(f"--ep-stats: upgraded the v2 ep_stats.f32 ({n:,} rows) "
                f"to v{EP_STATS_VERSION} in place", flush=True)
            ver, ncol = EP_STATS_VERSION, EP_STATS_COLS
        if ver is not None and (ver, ncol) != (EP_STATS_VERSION,
                                               EP_STATS_COLS):
            old = os.path.join(run_dir, f"ep_stats.v{ver}.f32")
            os.replace(self.path, old)
            log(f"--ep-stats: existing v{ver} ep_stats.f32 moved to "
                f"{os.path.basename(old)} (new layout v{EP_STATS_VERSION})",
                flush=True)
            ver = None
        if ver is None:
            with open(self.path, "wb") as f:
                f.write(header_row().tobytes())

    def append(self, er):
        with open(self.path, "ab") as f:
            f.write(np.ascontiguousarray(er, dtype=np.float32).tobytes())


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
    if C.shape[1] >= COL_PART + 12:
        # v2+: swing components over the completed plays' note cuts
        P = np.nansum(C[:, COL_PART:COL_PART + 12], axis=0).reshape(2, 6)
        parts = []
        for name, k in (("pre", 0), ("post", 2), ("prec", 4)):
            tot = P[:, k].sum() / max(P[:, k + 1].sum(), 1e-9)
            lr = [P[h, k] / max(P[h, k + 1], 1e-9) for h in (0, 1)]
            parts.append(f"{name} {100 * tot:.1f}% (L {100 * lr[0]:.1f} / "
                         f"R {100 * lr[1]:.1f})")
        print("completed swing: " + "  ".join(parts))
    if C.shape[1] >= COL_WRS + 4:
        # v3: wrist resets per 100 swings over the completed plays
        W = np.nansum(C[:, COL_WRS:COL_WRS + 4], axis=0)
        f = lambda r, s: 100.0 * r / s if s > 0 else float("nan")  # noqa: E731
        print(f"completed wrist resets / 100 swings: L {f(W[0], W[2]):.2f}"
              f"  R {f(W[1], W[3]):.2f}  ({int(W[2] + W[3]):,} swings)")
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
