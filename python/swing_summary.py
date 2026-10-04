#!/usr/bin/env python3
"""Per-level swing quality + best-possible PP totals from a run's
ep_stats.f32 (2026-10-04, user) — the dashboard's "swing" tab.

    python3 swing_summary.py <run_dir> [...]         # (re)build swing.json
    python3 swing_summary.py --archive ~/beat-arena-volume-backup
    python/.venv/bin/python swing_summary.py --reconstruct <run_dir>
        # old runs: rebuild levels.json from args.json + the laptop's maps

Inputs, both in the run dir:
  ep_stats.f32  the trainer's per-episode rows (ep_stats.py; v1 = 16 cols,
                v2 = + update index + 12 per-hand swing cols)
  levels.json   chart_idx -> level (ranked_pool.level_table; written by
                the trainer since 2026-10-04, or by --reconstruct for an
                older run, validated against the file's ids / max scores)

Output: <run_dir>/swing.json (columnar, one entry per level the run
attempted; a level's mirrored training copy is the same level):
  - per window — the last N = 1 / 5 / 20 completed plays of each level,
    the completed plays of the last K = 100 / 500 updates, the whole run —
    plays and the average pre-swing / post-swing / precision points as %
    of their max (70 / 30 / 15 per note), overall and per hand, plus the
    mean accuracy; ONLY completed plays (the level's end reached: no fall,
    tilt, envelope escape or tick cap) count;
  - per level over the whole run: attempts, completions, the best
    completed play's score -> SS and BL accuracy / PP;
  - totals: the best-possible SS / BL WPP (descending, 0.965^i) and UPP
    (plain sum) — each level's best completed play, per leaderboard, with
    the bench's accuracy definitions (bench_scoresaber ss_result /
    bl_result, scoring v4: SS acc = score / the SS API maxScore, BL acc =
    score / BL's maxScore, each falling back to the env's computed max;
    capped at 1). Shared levels count on both boards, SS-only levels on
    SS, BL-only on BL.

Incremental: a small state (.swing_state.json + .swing_state.npz next to
the run) keeps the byte offset read so far, the whole-run sums, the last
20 completed plays per level and the plays of the last 500 updates; a
pass reads only the new bytes, and a pass over an unchanged file only
stats it. A changed levels.json, a shrunk file or a new layout rebuilds
from scratch. swing.json is rewritten only when rows arrived.

numpy only (runs under the system python3, from metrics_backfill.py);
the PP formulas are the repo's own (beat_arena_env/scoresaber.py,
beat_arena_env/beatleader.py), loaded without the package's torch-
importing __init__. --reconstruct needs the venv (bench parser)."""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ep_stats as es  # noqa: E402

SWING_FORMAT = "beat-arena-swing"
SWING_VERSION = 1
STATE_VERSION = 5   # 2: best-possible history; 3/4: v1 update index; 5: best-play stats (2026-10-04)
N_WINDOWS = (1, 5, 20)
K_WINDOWS = (100, 500)
WINDOWS = ([("n%d" % n, "last %d play%s" % (n, "" if n == 1 else "s"))
            for n in N_WINDOWS]
           + [("k%d" % k, "last %d updates" % k) for k in K_WINDOWS]
           + [("all", "whole run")])
N_RING = max(N_WINDOWS)
K_KEEP = max(K_WINDOWS)
MIN_TICKS = 60            # ep_stats.py: shorter = lead-only episodes
PP_DECAY = 0.965   # both boards: play i of the descending list weighs 0.965^i
HIST_JSON = "swing_hist.json"
STATE_JSON = ".swing_state.json"
STATE_NPZ = ".swing_state.npz"
# tail-table columns: level, seq, update, acc, then the 12 swing cols
T_LEVEL, T_SEQ, T_UPD, T_ACC, T_PART = 0, 1, 2, 3, 4
T_COLS = T_PART + 12


# ------------------------------------------------------------------ PP code
def _pp_modules():
    """(scoresaber, beatleader) from beat_arena_env without running the
    package __init__ (it imports the Rust env, gymnasium and torch):
    a private package alias over the same directory."""
    if "beat_arena_env" in sys.modules:       # already imported (venv)
        return (importlib.import_module("beat_arena_env.scoresaber"),
                importlib.import_module("beat_arena_env.beatleader"))
    name = "_swing_bae"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [os.path.join(HERE, "beat_arena_env")]
        sys.modules[name] = pkg
    return (importlib.import_module(name + ".scoresaber"),
            importlib.import_module(name + ".beatleader"))


def ss_pp(score, level, comp_max):
    """bench_scoresaber.ss_result: (acc, pp) on ScoreSaber."""
    ss_mod, _ = _pp_modules()
    s = level["ss"]
    m = s.get("max_score") or comp_max
    acc = min(1.0, score / m) if m else 0.0
    return acc, round(ss_mod.pp_value(acc, s["max_pp"]), 2)


def bl_pp(score, level, comp_max):
    """bench_scoresaber.bl_result for a passed play: (acc, pp)."""
    _, bl_mod = _pp_modules()
    b = level["bl"]
    m = b.get("max_score") or comp_max
    acc = min(1.0, score / m) if m else 0.0
    r = bl_mod.pp_from_accuracy(acc, b["pass"], b["acc"], b["tech"])
    return acc, round(r["pp"], 2)


def totals(pps_ss, pps_bl):
    ss_mod, bl_mod = _pp_modules()
    return {"ss_wpp": round(ss_mod.weighted_total_pp(pps_ss), 2),
            "ss_upp": round(sum(pps_ss), 2),
            "bl_wpp": round(bl_mod.weighted_total_pp(pps_bl), 2),
            "bl_upp": round(sum(pps_bl), 2),
            "ss_levels": len(pps_ss), "bl_levels": len(pps_bl)}


# --------------------------------------------------------------- levels
def _stat(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def load_levels(run_dir):
    p = os.path.join(run_dir, "levels.json")
    try:
        with open(p, encoding="utf-8") as f:
            t = json.load(f)
    except (OSError, ValueError):
        return None
    if t.get("format") != "beat-arena-levels":
        return None
    return t


class LevelMap:
    """chart_idx -> dense level index. With a levels.json: its levels
    (original + mirror merged by construction). Without: levels keyed by
    the row's SS leaderboard id (mirrored copies share it), BL-only
    charts (id -1) by chart — no names, no BL view."""

    def __init__(self, table):
        self.table = table
        if table is not None:
            self.chart_level = np.asarray(table["chart_level"], np.int64)
            self.levels = table["levels"]
        else:
            self.chart_level = None
            self.levels = []
            self._by = {}

    def resolve(self, chart, lb, max_score):
        """Dense level ids for rows (chart, lb) — grows self.levels in
        the fallback mode. Rows whose chart is outside the table get -1."""
        if self.table is not None:
            ok = (chart >= 0) & (chart < len(self.chart_level))
            out = np.full(len(chart), -1, np.int64)
            out[ok] = self.chart_level[chart[ok]]
            return out
        code = np.where(lb > 0, lb, -(chart + 1))
        uniq, first, inv = np.unique(code, return_index=True,
                                     return_inverse=True)
        ids = np.empty(len(uniq), np.int64)
        for j, (u, i0) in enumerate(zip(uniq.tolist(), first.tolist())):
            key = ("ss", u) if u > 0 else ("chart", -u - 1)
            li = self._by.get(key)
            if li is None:
                li = self._by[key] = len(self.levels)
                self.levels.append({
                    "key": f"{key[0]}:{key[1]}", "song": None,
                    "difficulty": None, "max_score": float(max_score[i0]),
                    "nps": None, "ss": None, "bl": None,
                    "_ss_id": u if u > 0 else -1})
            ids[j] = li
        return ids[inv.reshape(-1)]

    def state(self):
        if self.table is not None:
            return None
        return {"levels": self.levels,
                "by": [[k[0], k[1], v] for k, v in self._by.items()]}

    def restore(self, st):
        if self.table is None and st:
            self.levels = st["levels"]
            self._by = {(a, b): v for a, b, v in st["by"]}


# ------------------------------------------------------ v1 update index
def v1_updates(run_dir, seq0, n, ticks=None, tick0=0.0):
    """Update index of v1 rows seq0..seq0+n-1 (the rows carry none), and
    whether it is exact. NaN where not recoverable.

    Exact: from metrics.jsonl `ep_stats_episodes` (rows written so far by
    this trainer invocation, after each update). The counter restarts at
    every invocation (a resume); the rows are appended in order, so the
    counts chain across restarts: each restart continues from the previous
    invocation's last count. (Checked on wb2_trunk and ab_on: the chained
    total equals the file's rows exactly.)

    Approximate (metrics that predate the counter): from the rows' own
    episode lengths. Every update advances num_envs x horizon env-ticks,
    and an episode's row is written when it ends, so the running sum of
    the episode ticks written so far (tick0 + cumsum(ticks)) over
    num_envs x horizon tracks the update count, behind it by the episodes
    still in progress: R = E[L^2] / (2 E[L]) / horizon updates in steady
    state (the mean residual of an episode in progress), ramping in from 0
    — modelled as t = x + R (1 - exp(-t / R)), solved by iteration.
    Checked against the exact counters: completed plays land within ~25
    updates on single-invocation runs (about -100 per resume that retrains
    updates, which the rows alone cannot reveal). Spreading rows evenly
    over updates does not work: early training writes ~32k short fallen
    episodes per update, late training a few hundred."""
    out = np.full(n, np.nan)
    p = os.path.join(run_dir, "metrics.jsonl")
    if n == 0 or not os.path.exists(p):
        return out, True
    upds, cnts, all_upd = [], [], []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "upd" in r:
                    all_upd.append(r["upd"])
                if "ep_stats_episodes" in r:
                    upds.append(r["upd"])
                    cnts.append(r["ep_stats_episodes"])
    except OSError:
        return out, True
    seq = np.arange(seq0, seq0 + n)
    if cnts:
        chain, off, prev = [], 0, None
        for c in cnts:
            if prev is not None and c < prev:
                off += prev                 # a restarted counter
            chain.append(off + c)
            prev = c
        chain = np.asarray(chain)
        j = np.searchsorted(chain, seq, side="right")
        ok = j < len(chain)
        out[ok] = np.asarray(upds, np.float64)[j[ok]]
        return out, True
    if ticks is not None and all_upd:
        try:
            with open(os.path.join(run_dir, "args.json"), encoding="utf-8") as f:
                a = json.load(f)
            a = a.get("resolved", a)
            per_upd = float(a["num_envs"]) * float(a["horizon"])
        except (OSError, ValueError, KeyError, TypeError):
            return out, True
        lo, hi = min(all_upd), max(all_upd)
        L = np.asarray(ticks, np.float64)
        x = (tick0 + np.cumsum(L)) / per_upd
        res = (float((L ** 2).mean()) / (2.0 * float(L.mean()))
               / float(a["horizon"])) if L.size and L.mean() > 0 else 0.0
        t = x.copy()
        if res > 0:
            for _ in range(30):
                t = x + res * (1.0 - np.exp(-t / res))
        out[:] = np.minimum(lo + np.floor(t), hi)
        return out, False
    return out, True


# ---------------------------------------------------------------- state
class State:
    def __init__(self, L=0):
        self.offset = 0
        self.seq = 0
        self.layout = None
        self.levels_stat = None
        self.last_update = None
        self.n_rows = 0
        self.n_done = 0
        self.resize(L)
        self.tail_n = np.zeros((0, T_COLS))
        self.tail_k = np.zeros((0, T_COLS))
        # [update, ss_wpp, ss_upp, bl_wpp, bl_upp], one row per update at
        # which the totals changed
        self.hist = np.zeros((0, 5))
        self.hist_approx = False     # v1 update indices estimated (no counter)
        self.v1_ticks = 0.0          # episode ticks of the v1 rows so far
        self.summary = None
        self.fallback = None
        self.v1_stat = None

    def resize(self, L):
        def grow(a, shape, fill=0.0):
            if a is None:
                return np.full(shape, fill)
            if a.shape[0] >= L:
                return a
            b = np.full(shape, fill)
            b[:a.shape[0]] = a
            return b
        self.run_sum = grow(getattr(self, "run_sum", None), (L, 12))
        self.run_done = grow(getattr(self, "run_done", None), (L,))
        self.run_att = grow(getattr(self, "run_att", None), (L,))
        self.acc_sum = grow(getattr(self, "acc_sum", None), (L,))
        self.best = grow(getattr(self, "best", None), (L,), -1.0)
        self.best_upd = grow(getattr(self, "best_upd", None), (L,), np.nan)
        self.max_score = grow(getattr(self, "max_score", None), (L,), np.nan)
        # the best-possible history: per level the best completed score
        # and its SS / BL pp so far, replayed in update order
        self.h_best = grow(getattr(self, "h_best", None), (L,), -1.0)
        self.h_ss = grow(getattr(self, "h_ss", None), (L,), np.nan)
        self.h_bl = grow(getattr(self, "h_bl", None), (L,), np.nan)
        # the whole run's best play per level among plays WITH swing data
        # (the Swing tab's "best" view): its accuracy and 12 swing cols
        self.sw_best_acc = grow(getattr(self, "sw_best_acc", None), (L,), -1.0)
        self.sw_best_part = grow(getattr(self, "sw_best_part", None), (L, 12), np.nan)

    ARRAYS = ("run_sum", "run_done", "run_att", "acc_sum", "best",
              "best_upd", "max_score", "tail_n", "tail_k",
              "h_best", "h_ss", "h_bl", "hist", "sw_best_acc", "sw_best_part")

    def save(self, run_dir):
        meta = {"version": STATE_VERSION, "offset": self.offset,
                "seq": self.seq, "layout": self.layout,
                "levels_stat": self.levels_stat,
                "ep_stat": _stat(os.path.join(run_dir, "ep_stats.f32")),
                "last_update": self.last_update, "n_rows": self.n_rows,
                "n_done": self.n_done, "summary": self.summary,
                "fallback": self.fallback, "v1_stat": self.v1_stat,
                "hist_approx": self.hist_approx, "v1_ticks": self.v1_ticks}
        tmp = os.path.join(run_dir, STATE_NPZ + ".tmp.npz")
        np.savez_compressed(tmp, **{k: getattr(self, k) for k in self.ARRAYS})
        os.replace(tmp, os.path.join(run_dir, STATE_NPZ))
        tmp = os.path.join(run_dir, STATE_JSON + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(tmp, os.path.join(run_dir, STATE_JSON))

    @staticmethod
    def load_meta(run_dir):
        try:
            with open(os.path.join(run_dir, STATE_JSON), encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError):
            return None
        return m if m.get("version") == STATE_VERSION else None

    @classmethod
    def load(cls, run_dir, meta):
        st = cls(0)
        try:
            z = np.load(os.path.join(run_dir, STATE_NPZ))
            for k in cls.ARRAYS:
                setattr(st, k, z[k].astype(np.float64))
        except (OSError, ValueError, KeyError):
            return None
        for k in ("offset", "seq", "layout", "levels_stat", "last_update",
                  "n_rows", "n_done", "summary", "fallback", "v1_stat",
                  "hist_approx", "v1_ticks"):
            setattr(st, k, meta.get(k))
        return st


# ----------------------------------------------------------- the update
def _keep_last_n(tail, n):
    """Rows of `tail` that are among the last n (by seq) of their level."""
    if not len(tail):
        return tail
    order = np.lexsort((tail[:, T_SEQ], tail[:, T_LEVEL]))
    t = tail[order]
    lv = t[:, T_LEVEL]
    # rank from the end within each level group
    return t[_rank_from_end(lv) < n]


def _rank_from_end(lv):
    """For a level column sorted by (level, seq): 0 for each level's
    newest row, 1 for the one before, ..."""
    last = np.r_[lv[1:] != lv[:-1], True]
    ends = np.flatnonzero(last)
    i = np.arange(len(lv))
    return ends[np.searchsorted(ends, i)] - i


def _wpp(pps):
    """Profile total over a set of pp (both boards: descending, x 0.965^i)."""
    if not len(pps):
        return 0.0
    v = np.sort(pps)[::-1]
    return float(np.dot(v, PP_DECAY ** np.arange(len(v))))


def _history(st, levmap, d_lev, score, d_upd):
    """Replay completed plays in update order into the best-possible
    history: per update, each level's best score so far -> its SS / BL pp
    (ss_pp / bl_pp, the totals' own scoring) -> the four totals, appended
    to st.hist when they changed. Rows whose update is unknown (an old run
    without the metrics counter) count from the start (update NaN, not
    plotted)."""
    if not len(d_lev):
        return
    u = np.where(np.isfinite(d_upd), d_upd, -np.inf)
    lv = levmap.levels
    for g in np.unique(u):
        m = u == g
        gl, gs = d_lev[m], score[m]
        top = np.full(st.h_best.shape[0], -1.0)
        np.maximum.at(top, gl, gs)
        changed = np.flatnonzero(top > st.h_best)
        if not len(changed):
            continue
        for li in changed.tolist():
            st.h_best[li] = top[li]
            L_ = lv[li]
            comp = st.max_score[li]
            if not np.isfinite(comp) or comp <= 0:
                comp = float(L_.get("max_score") or 0.0)
            if L_.get("ss") is not None:
                st.h_ss[li] = ss_pp(top[li], L_, comp)[1]
            if L_.get("bl") is not None:
                st.h_bl[li] = bl_pp(top[li], L_, comp)[1]
        if not np.isfinite(g):
            continue
        ss = st.h_ss[np.isfinite(st.h_ss)]
        bl = st.h_bl[np.isfinite(st.h_bl)]
        row = [g, round(_wpp(ss), 2), round(float(ss.sum()), 2),
               round(_wpp(bl), 2), round(float(bl.sum()), 2)]
        if len(st.hist) and st.hist[-1, 0] == g:
            st.hist[-1] = row
        elif not len(st.hist) or (st.hist[-1, 1:] != row[1:]).any():
            st.hist = np.vstack([st.hist, row])


def ingest(st, rows, ver, levmap, run_dir, best_only=False, total=None):
    """Fold new ep_stats rows into the state. best_only: rows without
    swing columns folded into a swing run (a resumed run's pre-resume
    ep_stats.v1.f32) — they count toward the run's best plays (the
    best-possible totals) but not toward plays / accuracy / swing
    averages, which would otherwise mix in plays with no swing data."""
    n = len(rows)
    if n == 0:
        return
    seq = np.arange(st.seq, st.seq + n, dtype=np.float64)
    if ver >= 2 and rows.shape[1] >= es.EP_STATS_COLS:
        upd = rows[:, es.COL_UPDATE].astype(np.float64)
        part = rows[:, es.COL_PART:es.COL_PART + 12].astype(np.float64)
    else:
        upd, exact = v1_updates(run_dir, st.seq, n, ticks=rows[:, 4],
                                tick0=st.v1_ticks)
        st.v1_ticks += float(rows[:, 4].astype(np.float64).sum())
        if not exact:
            st.hist_approx = True
        part = np.full((n, 12), np.nan)
    st.seq += n
    chart = rows[:, 0].astype(np.int64)
    lb = rows[:, 1].astype(np.int64)
    ms = rows[:, 3].astype(np.float64)
    lev = levmap.resolve(chart, lb, ms)
    L = max(len(levmap.levels), st.run_sum.shape[0])
    st.resize(L)
    good = lev >= 0
    ticks_ok = rows[:, 4] >= MIN_TICKS
    done = (rows[:, 6] > 0.5) & good
    st.n_rows += int(n)
    st.n_done += int(done.sum())
    if not best_only:
        fu = upd[np.isfinite(upd)]
        if len(fu):
            st.last_update = max(st.last_update or 0, int(fu.max()))
        np.add.at(st.run_att, lev[good & ticks_ok], 1.0)
    if not done.any():
        return
    d_lev = lev[done]
    score = np.round(rows[done, 7].astype(np.float64))
    d_ms = ms[done]
    acc = np.where(d_ms > 0, np.minimum(1.0, score / np.maximum(d_ms, 1.0)), 0.0)
    if not best_only:
        np.add.at(st.run_done, d_lev, 1.0)
        np.add.at(st.acc_sum, d_lev, acc)
        np.add.at(st.run_sum, d_lev, np.nan_to_num(part[done]))
    # the run's computed max per level (the env's scoring at run time)
    st.max_score[d_lev] = np.where(d_ms > 0, d_ms, st.max_score[d_lev])
    _history(st, levmap, d_lev, score, upd[done])
    # best completed play per level (later plays win ties: max is max)
    order = np.lexsort((score, d_lev))
    sl, ss, su = d_lev[order], score[order], upd[done][order]
    lastm = np.r_[sl[1:] != sl[:-1], True]
    bl, bs, bu = sl[lastm], ss[lastm], su[lastm]
    better = bs > st.best[bl]
    st.best[bl[better]] = bs[better]
    st.best_upd[bl[better]] = bu[better]
    if best_only:
        return
    # the best play per level of the plays folded here (by accuracy)
    o2 = np.lexsort((acc, d_lev))
    l2, a2, p2 = d_lev[o2], acc[o2], part[done][o2]
    last2 = np.r_[l2[1:] != l2[:-1], True]
    bl2, ba2, bp2 = l2[last2], a2[last2], p2[last2]
    up = ba2 > st.sw_best_acc[bl2]
    st.sw_best_acc[bl2[up]] = ba2[up]
    st.sw_best_part[bl2[up]] = bp2[up]
    tail = np.column_stack([d_lev.astype(np.float64), seq[done], upd[done],
                            acc, part[done]])
    st.tail_n = _keep_last_n(np.concatenate([st.tail_n, tail]), N_RING)
    tk = np.concatenate([st.tail_k, tail])
    if st.last_update is not None:
        tk = tk[np.isfinite(tk[:, T_UPD])
                & (tk[:, T_UPD] > st.last_update - K_KEEP)]
    else:
        tk = tk[:0]
    st.tail_k = tk


def _pct(p, m):
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(m > 0, 100.0 * p / np.where(m > 0, m, 1.0), np.nan)
    return v


def _r1(a):
    return [None if not np.isfinite(x) else round(float(x), 1) for x in a]


def window_stats(sums, plays, acc_sum, swing_ok):
    """{plays, pre, post, prec, preL, ..., acc} lists from per-level sums
    (12 swing cols) and play counts."""
    out = {"plays": [int(x) for x in plays]}
    P = sums.reshape(-1, 2, 6)
    for name, k in (("pre", 0), ("post", 2), ("prec", 4)):
        if swing_ok:
            out[name] = _r1(_pct(P[:, :, k].sum(1), P[:, :, k + 1].sum(1)))
            out[name + "L"] = _r1(_pct(P[:, 0, k], P[:, 0, k + 1]))
            out[name + "R"] = _r1(_pct(P[:, 1, k], P[:, 1, k + 1]))
        else:
            out[name] = out[name + "L"] = out[name + "R"] = None
    with np.errstate(invalid="ignore", divide="ignore"):
        out["acc"] = _r1(np.where(plays > 0, 100.0 * acc_sum
                                  / np.maximum(plays, 1), np.nan))
    return out


def build_summary(st, levmap, run_dir, swing_ok, levels_source):
    L = st.run_sum.shape[0]
    att = st.run_att[:L]
    # levels with a best play only (a resumed run's pre-resume rows,
    # best_only) are listed too: they carry the best-possible totals
    seen = np.flatnonzero((att > 0) | (st.run_done[:L] > 0)
                          | (st.best[:L] >= 0))
    idx = {int(li): j for j, li in enumerate(seen)}
    lv = levmap.levels
    # whole-run per level + best-possible totals
    ss_pp_l, bl_pp_l, ss_acc_l, bl_acc_l, best_acc_l = [], [], [], [], []
    pps_ss, pps_bl = [], []
    for li in seen.tolist():
        L_ = lv[li]
        b = st.best[li]
        comp = st.max_score[li]
        if not np.isfinite(comp) or comp <= 0:
            comp = float(L_.get("max_score") or 0.0)
        if b < 0:
            ss_pp_l.append(None); bl_pp_l.append(None)
            ss_acc_l.append(None); bl_acc_l.append(None)
            best_acc_l.append(None)
            continue
        best_acc_l.append(round(100.0 * min(1.0, b / comp), 2) if comp else None)
        if L_.get("ss") is not None:
            a, p = ss_pp(b, L_, comp)
            ss_pp_l.append(p); ss_acc_l.append(round(100 * a, 2))
            pps_ss.append(p)
        else:
            ss_pp_l.append(None); ss_acc_l.append(None)
        if L_.get("bl") is not None:
            a, p = bl_pp(b, L_, comp)
            bl_pp_l.append(p); bl_acc_l.append(round(100 * a, 2))
            pps_bl.append(p)
        else:
            bl_pp_l.append(None); bl_acc_l.append(None)

    def col(f):
        return [f(lv[li]) for li in seen.tolist()]

    levels = {
        "key": col(lambda v: v.get("key")),
        "song": col(lambda v: v.get("song")),
        "mapper": col(lambda v: v.get("mapper")),
        "diff": col(lambda v: v.get("difficulty")),
        "ss_stars": col(lambda v: v["ss"]["stars"] if v.get("ss") else None),
        "bl_stars": col(lambda v: v["bl"]["stars"] if v.get("bl") else None),
        "ss_id": col(lambda v: v["ss"]["id"] if v.get("ss") else
                     (v.get("_ss_id") if (v.get("_ss_id") or 0) > 0 else None)),
        "bl_id": col(lambda v: v["bl"]["id"] if v.get("bl") else None),
        "nps": col(lambda v: v.get("nps")),
        "att": [int(x) for x in att[seen]],
        "done": [int(x) for x in st.run_done[seen]],
        "best_acc": best_acc_l,
        "ss_acc": ss_acc_l, "ss_pp": ss_pp_l,
        "bl_acc": bl_acc_l, "bl_pp": bl_pp_l,
        "best_upd": [None if not np.isfinite(x) else int(x)
                     for x in st.best_upd[seen]],
    }
    win = {}
    avail = []
    S = len(seen)

    def best_of(t):
        """Per level the window's best play (highest accuracy): its swing
        cols and accuracy, as window_stats input (one play each)."""
        sums = np.full((S, 12), np.nan)
        one = np.zeros(S)
        accs = np.zeros(S)
        if len(t):
            j = np.array([idx.get(int(x), -1) for x in t[:, T_LEVEL]])
            t, j = t[j >= 0], j[j >= 0]
            o = np.lexsort((t[:, T_ACC], j))
            t, j = t[o], j[o]
            last = np.r_[j[1:] != j[:-1], True] if len(j) else np.zeros(0, bool)
            sums[j[last]] = t[last, T_PART:]
            one[j[last]] = 1.0
            accs[j[last]] = t[last, T_ACC]
        return sums, one, accs

    def from_tail(t):
        sums = np.zeros((S, 12))
        plays = np.zeros(S)
        accs = np.zeros(S)
        if len(t):
            j = np.array([idx.get(int(x), -1) for x in t[:, T_LEVEL]])
            ok = j >= 0
            np.add.at(sums, j[ok], np.nan_to_num(t[ok, T_PART:]))
            np.add.at(plays, j[ok], 1.0)
            np.add.at(accs, j[ok], t[ok, T_ACC])
        return sums, plays, accs

    if len(st.tail_n):
        order = np.lexsort((st.tail_n[:, T_SEQ], st.tail_n[:, T_LEVEL]))
        tn = st.tail_n[order]
        rank = _rank_from_end(tn[:, T_LEVEL])
    else:
        tn, rank = st.tail_n, np.zeros(0)
    win_best = {}
    for key, label in WINDOWS:
        ok = True
        bsrc = None
        if key.startswith("n"):
            bsrc = tn[rank < int(key[1:])]
            sums, plays, accs = from_tail(bsrc)
        elif key.startswith("k"):
            K = int(key[1:])
            if st.last_update is None:
                ok = False
                sums, plays, accs = np.zeros((S, 12)), np.zeros(S), np.zeros(S)
            else:
                t = st.tail_k
                bsrc = t[t[:, T_UPD] > st.last_update - K]
                sums, plays, accs = from_tail(bsrc)
        else:
            sums, plays, accs = (st.run_sum[seen], st.run_done[seen],
                                 st.acc_sum[seen])
        win[key] = window_stats(sums, plays, accs, swing_ok) if ok else None
        if ok:
            if bsrc is not None:
                bs_, b1, ba = best_of(bsrc)
            else:                                    # the whole run
                bs_ = st.sw_best_part[seen]
                ba = np.where(st.sw_best_acc[seen] >= 0, st.sw_best_acc[seen], 0.0)
                b1 = (st.sw_best_acc[seen] >= 0).astype(float)
            wb = window_stats(np.nan_to_num(bs_), b1, ba, swing_ok)
            wb["plays"] = win[key]["plays"]          # the window's play count
            win_best[key] = wb
        else:
            win_best[key] = None
        avail.append({"key": key, "label": label, "available": ok})
    tot = totals(pps_ss, pps_bl)
    tab = levmap.table or {}
    return {
        "format": SWING_FORMAT, "version": SWING_VERSION,
        "generated": time.time(),
        "ep_layout": st.layout[0] if st.layout else None,
        "swing": bool(swing_ok),
        "levels_source": levels_source,
        "catalog": tab.get("catalog"),
        "rows": st.n_rows, "completed": st.n_done,
        "last_update": st.last_update,
        "n_levels": S,
        "totals": tot,
        "windows": avail,
        "levels": levels,
        "win": win,
        "win_best": win_best,
    }


def index_entry(summary, run_dir, rel=None):
    """The small per-run record metrics_backfill puts in index.json."""
    if not summary:
        return None
    path = os.path.join(run_dir, "swing.json")
    hist = os.path.join(run_dir, HIST_JSON)
    return {"path": (rel + "/swing.json") if rel else path,
            "hist": (((rel + "/" + HIST_JSON) if rel else hist)
                     if os.path.exists(hist) else None),
            "ep_layout": summary["ep_layout"], "swing": summary["swing"],
            "levels_source": summary["levels_source"],
            "rows": summary["rows"], "completed": summary["completed"],
            "last_update": summary["last_update"],
            "n_levels": summary["n_levels"], "totals": summary["totals"]}


def update_run(run_dir, force=False, log=None):
    """Bring <run_dir>/swing.json up to date; returns its summary dict
    without the per-level arrays (None: no ep_stats.f32). Cheap when
    nothing changed (two stats + the small state json)."""
    ep = os.path.join(run_dir, "ep_stats.f32")
    ep_st = _stat(ep)
    if ep_st is None:
        return None
    lv_st = _stat(os.path.join(run_dir, "levels.json"))
    # a resumed run's pre-resume rows: the trainer moves an old-layout
    # file to ep_stats.v1.f32 when it starts writing v2 (best plays only)
    v1p = os.path.join(run_dir, "ep_stats.v1.f32")
    v1_st = _stat(v1p)
    meta = None if force else State.load_meta(run_dir)
    if meta and meta.get("v1_stat") != v1_st:
        meta = None                      # the v1 file appeared / changed
    out_path = os.path.join(run_dir, "swing.json")
    if (meta and meta.get("ep_stat") == ep_st
            and meta.get("levels_stat") == lv_st and meta.get("summary")
            and os.path.exists(out_path)):
        return meta["summary"]
    ver, ncol, _ = es.file_layout(ep)
    if ver is None:
        return None
    table = load_levels(run_dir)
    levmap = LevelMap(table)
    st = None
    if meta and meta.get("levels_stat") == lv_st \
            and meta.get("layout") == [ver, ncol] \
            and meta.get("offset", 0) <= ep_st[0]:
        st = State.load(run_dir, meta)
        if st is not None and table is None:
            levmap.restore(st.fallback)
    if st is None:
        st = State(len(levmap.levels))
        st.layout = [ver, ncol]
        st.levels_stat = lv_st
        if v1_st is not None and ver >= 2:
            rows1, _, ver1, _ = es.read_rows(v1p, 0)
            if ver1 is not None and len(rows1):
                ingest(st, rows1, ver1, levmap, run_dir, best_only=True)
    st.v1_stat = v1_st
    rows, off, ver2, _ = es.read_rows(ep, st.offset)
    if ver2 != ver:
        return None
    t0 = time.time()
    ingest(st, rows, ver, levmap, run_dir)
    st.offset = off
    st.fallback = levmap.state()
    swing_ok = ver >= 2
    src = (table.get("source", "trainer") if table else None)
    full = build_summary(st, levmap, run_dir, swing_ok, src)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(full, f, separators=(",", ":"))
    os.replace(tmp, out_path)
    # the best-possible curve for the Live / Compare charts: its own small
    # file, so a chart refresh never pulls the per-level arrays
    h = st.hist
    hist = {"format": "beat-arena-swing-hist", "version": 1,
            "approx": bool(st.hist_approx),
            "upd": [int(x) for x in h[:, 0]],
            "ss_wpp": h[:, 1].tolist(), "ss_upp": h[:, 2].tolist(),
            "bl_wpp": h[:, 3].tolist(), "bl_upp": h[:, 4].tolist()}
    tmp = os.path.join(run_dir, HIST_JSON + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(hist, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(run_dir, HIST_JSON))
    st.summary = {k: v for k, v in full.items()
                  if k not in ("levels", "win", "win_best")}
    st.save(run_dir)
    if log:
        log(f"[swing] {run_dir}: +{len(rows):,} rows ({st.n_rows:,} total, "
            f"{st.n_done:,} completed), {full['n_levels']} levels, "
            f"{time.time() - t0:.1f}s")
    return st.summary


# ---------------------------------------------------------- reconstruct
def reconstruct(run_dir, log=print, max_ms_bad=0.03):
    """Old runs (no trainer-written levels.json): rebuild the pool's
    level table from args.json (catalog, lead_s, mirror_augment,
    chart_pool, seed) and the laptop's maps, then validate it against
    the run's own rows — every chart's SS leaderboard id (col 1) must
    match and at most `max_ms_bad` of the charts may disagree on the
    computed max score (col 3; a scoring change since the run: the
    pre-scoring-v4 runs of 2026-10-01 differ on ~1% of the charts, the
    arc-attached chain heads; a wrong mapping would miss on nearly every
    BL-only chart, whose id is -1). The totals use the run's own max
    (col 3) for its accuracy, so those charts stay exact. Writes
    levels.json (source "reconstructed") or raises. Needs the venv."""
    from beat_arena_env import ranked_pool as rp
    with open(os.path.join(run_dir, "args.json"), encoding="utf-8") as f:
        a = json.load(f)
    r = a.get("resolved", a)
    if not r.get("ranked_maps", True):
        raise ValueError("not a ranked-maps run")
    table = rp.level_table(r.get("catalog") or "scoresaber",
                           lead_s=float(r.get("lead_s", rp.RECIPE_LEAD_S)),
                           mirror=bool(r.get("mirror_augment",
                                             rp.RECIPE_MIRROR)),
                           sample=r.get("chart_pool"), seed=r.get("seed", 0),
                           log=log)
    R = es.load(run_dir)
    P = table["n_charts"]
    ci = R[:, 0].astype(np.int64)
    if len(ci) and ci.max() >= P:
        raise ValueError(f"rows reference chart {ci.max()} but the "
                         f"rebuilt pool has {P}")
    lb = np.full(P, np.nan)
    ms = np.full(P, np.nan)
    lb[ci] = R[:, 1]
    ms[ci] = R[:, 3]
    n, bad_lb, bad_ms = rp.check_levels(table, lb, ms)
    log(f"[reconstruct] {run_dir}: {n} charts checked, {bad_lb} id / "
        f"{bad_ms} max-score mismatches")
    if bad_lb or bad_ms > max_ms_bad * max(n, 1):
        raise ValueError(f"rebuilt level table does not match the run "
                         f"({bad_lb} ids, {bad_ms} max scores)")
    table["source"] = "reconstructed"
    table["written"] = time.time()
    table["check"] = {"charts": n, "lb_bad": bad_lb, "max_score_bad": bad_ms}
    path = os.path.join(run_dir, "levels.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(table, f, separators=(",", ":"))
    os.replace(path + ".tmp", path)
    return table


def find_runs(archive):
    return sorted(os.path.dirname(p) for p in glob.glob(
        os.path.join(archive, "*", "runs", "**", "ep_stats.f32"),
        recursive=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--archive", help="every run dir with an ep_stats.f32")
    ap.add_argument("--force", action="store_true",
                    help="ignore the incremental state")
    ap.add_argument("--reconstruct", action="store_true",
                    help="write levels.json for runs that lack one "
                         "(venv; see reconstruct())")
    a = ap.parse_args(argv)
    runs = list(a.run_dirs)
    if a.archive:
        runs += find_runs(a.archive)
    for rd in runs:
        if a.reconstruct:
            if load_levels(rd) is not None and not a.force:
                print(f"[reconstruct] {rd}: has levels.json")
            else:
                try:
                    reconstruct(rd)
                except Exception as e:  # noqa: BLE001
                    print(f"[reconstruct] {rd}: FAILED {e!r}")
                    continue
        s = update_run(rd, force=a.force or a.reconstruct, log=print)
        if s:
            t = s["totals"]
            print(f"  {rd}: SS wpp {t['ss_wpp']:,.0f} upp {t['ss_upp']:,.0f}"
                  f" | BL wpp {t['bl_wpp']:,.0f} upp {t['bl_upp']:,.0f} "
                  f"({t['ss_levels']} SS / {t['bl_levels']} BL levels; "
                  f"levels {s['levels_source']}, ep layout v{s['ep_layout']})")


if __name__ == "__main__":
    main()
