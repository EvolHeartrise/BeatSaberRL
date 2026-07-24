"""Procedural chart generation (SPEC.md §Chart generation).

Generator v2 (2026-07-24, user-designed): notes come from a jittered
tick lattice (~0.15 s) driven by three independent paths — left hand,
right hand, bombs — replacing the old rate-scheduled uniform stream.
Each path keeps PARITY FLOW: a spawned note's direction is the path's
previous ORIGINAL direction flipped 180° plus a normal offset
(σ = 45°), so consecutive swings mostly oppose like real maps but can
occasionally reverse outright. Note speed is one shared random walk
(±0.4 m/s per note, clamped to [note_speed_min, note_speed_max]) across
ALL paths, so the two hands' speeds never drift far apart. A spawned
note may bring an EXTRA note: a replica placed along the original's
swing line (±1.5 m; discarded if that point leaves the spawn
rectangle), then given a ±45° rotation offset, a ±0.5 m perpendicular
offset, and a ±0.1 m z offset (implemented as an arrival-time shift) —
stacks/windows in spirit: one swing can take both. Extras never feed
the parity chain and never roll their own extras. At the end of each
tick every generated note independently becomes a dot (P = dot_prob).
The bomb path follows the exact note logic (its parity angle steers
extra placement) at a lower spawn chance, then emits directionless
bombs. Chains and arcs are no longer generated procedurally — real
BeatSaver maps are their only source (`chain_links` below serves the
parser).

Positions stay continuous (env v7.5): x, y ~ U over the rectangle
spanned by the allowed cols/rows padded by POS_PAD (the cell dials
survive as range clamps); real BeatSaver maps keep their grid cells;
note col/row are nearest-cell metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .constants import (KIND_BOMB, KIND_CHAIN_LINK, KIND_NOTE, ROW_Y,
                        col_x, note_center)

N_COLS = 4
N_ROWS = 3
LEFT, RIGHT = 0, 1
_BOMB_PATH = 2                # parity-chain key for the bomb path

# generator v2 lattice + path constants (user-specified 2026-07-24)
TICK_DT = 0.15                # mean generation-tick interval (s)
TICK_JITTER = 0.03            # per-tick uniform jitter (± s)
ANGLE_SIGMA = math.radians(45.0)  # parity-offset spread (user-picked σ)
SPEED_DEV = 0.4               # speed random-walk step (± m/s per note)
EXTRA_PATH = 1.5              # extra note: ± range along the swing line (m)
EXTRA_ROT = math.radians(45.0)    # extra note rotation offset (±)
EXTRA_PERP = 0.5              # extra note perpendicular offset (± m)
EXTRA_DZ = 0.1                # extra note z offset (± m, as a time shift)


@dataclass(frozen=True)
class Note:
    hit_time: float
    col: int
    row: int
    color: int    # 0 = left/red, 1 = right/blue (bombs: 0, unused)
    angle: float  # required swing direction in RADIANS (0 = +x, CCW)
    speed: float  # approach speed toward the plane (m/s)
    any_dir: bool = False  # dot note: any swing direction scores
    kind: int = KIND_NOTE  # KIND_NOTE / _BOMB / _CHAIN_HEAD / _CHAIN_LINK
    x: float = None        # explicit world pos (chain links); None -> col/row
    y: float = None
    free_pre: bool = False   # arc tail: full 70 pre-swing granted
    free_post: bool = False  # arc head: full 30 follow-through granted
    raw_dir: int = None    # original map cutDirection int (0-8), real maps
    # only — `angle` may deviate from it (precision offsets, paired-note
    # snap), but external tools (ScoreSaber replay export) need the int


@dataclass
class ChartConfig:
    chart_len_s: float = 20.0
    colors: str = "right"         # "right" (all blue) or "both"
    first_note_s: float = 1.0     # first generation tick (s)
    cols: tuple = (0, 1, 2, 3)    # allowed columns (curriculum: restrict)
    rows: tuple = (0, 1, 2)       # allowed rows
    note_speed_min: float = 2.0   # speed random-walk clamp (m/s)
    note_speed_max: float = 8.0
    note_prob: float = 0.25       # P(note) per hand path per tick
    extra_prob: float = 0.125     # P(a spawned note brings an extra)
    dot_prob: float = 0.1         # P(a generated note turns into a dot)
    bomb_prob: float = 0.125      # P(bomb) per tick

    def __post_init__(self):
        assert self.colors in ("right", "both"), self.colors
        assert self.note_speed_max >= self.note_speed_min > 0
        for p in (self.note_prob, self.extra_prob, self.dot_prob,
                  self.bomb_prob):
            assert 0.0 <= p <= 1.0


def note_tuple(n: Note):
    """The (t, x, y, color, angle, speed, any_dir, kind, free_pre,
    free_post) tuple the Rust sim eats (positions resolved to meters)."""
    x, y = note_center(n)
    return (n.hit_time, x, y, n.color, n.angle, n.speed, n.any_dir,
            n.kind, n.free_pre, n.free_post)


def chain_links(head: Note, tail_x: float, tail_y: float, tail_time: float,
                sc: int, squish: float = 1.0) -> list[Note]:
    """The sc-1 link Notes of a chain (head excluded), real-BS layout:
    quadratic Bezier whose control point sits half the head-tail
    distance along the head's cut direction; link i at curve parameter
    (i/(sc-1))*squish, time lerped unsquished, angle = curve tangent
    (BeatLeader beat.js:424-463). Parser-only since generator v2."""
    if squish < 1e-3:
        squish = 1.0                       # BS sanitizes degenerate squish
    x0, y0 = note_center(head)
    dist = math.hypot(tail_x - x0, tail_y - y0)
    cx = x0 + math.cos(head.angle) * dist * 0.5
    cy = y0 + math.sin(head.angle) * dist * 0.5
    links = []
    for i in range(1, sc):
        tp = i / (sc - 1)
        t = tp * squish
        u = 1.0 - t
        px = u * u * x0 + 2 * u * t * cx + t * t * tail_x
        py = u * u * y0 + 2 * u * t * cy + t * t * tail_y
        dx = 2 * (u * (cx - x0) + t * (tail_x - cx))
        dy = 2 * (u * (cy - y0) + t * (tail_y - cy))
        links.append(Note(
            hit_time=head.hit_time + (tail_time - head.hit_time) * tp,
            col=head.col, row=head.row, color=head.color,
            angle=math.atan2(dy, dx) % (2 * math.pi), speed=head.speed,
            any_dir=True, kind=KIND_CHAIN_LINK, x=px, y=py))
    return links


# env v7.5: the continuous sampling area extends 0.2 m past the
# outermost allowed cell centers in every direction (user-specified) —
# under uniform sampling the old edge CELLS (half the note mass) became
# measure-zero extremes; the margin keeps outer-edge capability trained
# instead of regressing. Defaults: x [-1.1, 1.1], y [0.65, 2.1].
POS_PAD = 0.2


def _pos_ranges(cfg):
    """Continuous (x_lo, x_hi, y_lo, y_hi): the span of the allowed
    cols/rows padded by POS_PAD on every side (a single allowed col/row
    becomes a +-POS_PAD band — the spatial-curriculum dial survives as
    a narrow corridor)."""
    xs = [col_x(c) for c in cfg.cols]
    ys = [ROW_Y[r] for r in cfg.rows]
    return (min(xs) - POS_PAD, max(xs) + POS_PAD,
            min(ys) - POS_PAD, max(ys) + POS_PAD)


def _nearest_cell(x, y):
    """Nearest (col, row) metadata for a continuous position."""
    col = int(np.clip(round(x / 0.6 + 1.5), 0, N_COLS - 1))
    row = int(np.argmin([abs(y - ry) for ry in ROW_Y]))
    return col, row


def generate_chart(cfg: ChartConfig, rng: np.random.Generator) -> list[Note]:
    x_lo, x_hi, y_lo, y_hi = _pos_ranges(cfg)
    hands = (RIGHT,) if cfg.colors == "right" else (LEFT, RIGHT)
    prev_angle = {}       # per path: last ORIGINAL angle (extras excluded)
    prev_speed = None     # ONE walk shared by all paths (hands stay similar)
    notes: list[Note] = []

    def walk_speed():
        nonlocal prev_speed
        if prev_speed is None:
            s = float(rng.uniform(cfg.note_speed_min, cfg.note_speed_max))
        else:
            s = float(np.clip(
                prev_speed + rng.uniform(-SPEED_DEV, SPEED_DEV),
                cfg.note_speed_min, cfg.note_speed_max))
        prev_speed = s
        return s

    def spawn(t, path, color, kind):
        """One original (+ maybe its extra) on a path; the caller applies
        kind-specific finishing (dot roll / bomb conversion)."""
        pa = prev_angle.get(path)
        if pa is None:
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
        else:
            angle = float((pa + np.pi + rng.normal(0.0, ANGLE_SIGMA))
                          % (2.0 * np.pi))
        prev_angle[path] = angle
        speed = walk_speed()
        x = float(rng.uniform(x_lo, x_hi))
        y = float(rng.uniform(y_lo, y_hi))
        col, row = _nearest_cell(x, y)
        out = [Note(hit_time=float(t), col=col, row=row, x=x, y=y,
                    color=color, angle=angle, speed=speed, kind=kind)]
        if rng.random() < cfg.extra_prob:
            # replica along the swing line; the discard check runs on the
            # along-line point BEFORE the perpendicular/z offsets, so the
            # final position may overhang the rectangle by <= EXTRA_PERP
            d = float(rng.uniform(-EXTRA_PATH, EXTRA_PATH))
            ex = x + math.cos(angle) * d
            ey = y + math.sin(angle) * d
            if x_lo <= ex <= x_hi and y_lo <= ey <= y_hi:
                ea = float((angle + rng.uniform(-EXTRA_ROT, EXTRA_ROT))
                           % (2.0 * np.pi))
                p = float(rng.uniform(-EXTRA_PERP, EXTRA_PERP))
                ex += -math.sin(angle) * p
                ey += math.cos(angle) * p
                dz = float(rng.uniform(-EXTRA_DZ, EXTRA_DZ))
                ecol, erow = _nearest_cell(ex, ey)
                # z(t) = (hit_time - t)*speed: +dz m of z at any moment
                # == arriving dz/speed s later
                out.append(Note(hit_time=float(t + dz / speed), col=ecol,
                                row=erow, x=ex, y=ey, color=color,
                                angle=ea, speed=speed, kind=kind))
        return out

    t = cfg.first_note_s
    while t <= cfg.chart_len_s:
        for color in hands:
            if rng.random() < cfg.note_prob:
                for n in spawn(t, color, color, KIND_NOTE):
                    # end-of-tick dot roll, originals and extras alike
                    if rng.random() < cfg.dot_prob:
                        n = replace(n, any_dir=True, angle=0.0)
                    notes.append(n)
        if rng.random() < cfg.bomb_prob:
            for b in spawn(t, _BOMB_PATH, 0, KIND_BOMB):
                # the path angle steered placement; the emitted bomb is
                # directionless (parser convention: any_dir, angle 0)
                notes.append(replace(b, any_dir=True, angle=0.0))
        t += TICK_DT + float(rng.uniform(-TICK_JITTER, TICK_JITTER))

    notes.sort(key=lambda n: n.hit_time)
    return notes
