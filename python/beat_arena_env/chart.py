"""Procedural chart generation (SPEC.md §Chart generation).

A chart is a time-sorted list of Note objects — since env v7 "note" means
any gameplay object: color notes, bombs, chain heads, and chain links
(`kind`). Difficulty is controlled by the config knobs: note_rate is the
main dial, colors/dirs widen the skill demanded, min_hand_gap_s
guarantees physical feasibility, and bomb_rate / chain_prob / arc_prob
mix in the v7 object types.

Chain links follow the real Beat Saber layout: a quadratic Bezier from
head to tail whose control point sits half the head-tail distance along
the head's cut direction; link i of sc sits at curve parameter
t = (i/(sc-1)) * squish with its time lerped (unsquished) head->tail
and its angle from the curve tangent (BeatLeader beat.js:424-463).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .constants import (KIND_BOMB, KIND_CHAIN_HEAD, KIND_CHAIN_LINK,
                        KIND_NOTE, ROW_Y, col_x, note_center)

N_COLS = 4
N_ROWS = 3
LEFT, RIGHT = 0, 1

CHAIN_SC = (3, 6)             # link count range (inclusive), incl. head
CHAIN_TAIL_DT = (0.10, 0.25)  # head->tail seconds
ARC_MAX_GAP_S = 2.0           # max head->tail gap for a generated arc


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


@dataclass
class ChartConfig:
    note_rate: float = 1.0        # notes per second (overall)
    chart_len_s: float = 20.0
    colors: str = "right"         # "right" (all blue) or "both"
    dirs: tuple = (6,)            # allowed cut directions (6 = down-swing)
    continuous_dirs: bool = False # True: angle ~ U(0, 2pi), ignores `dirs`
    note_speed_max: float = None  # if set: per-note speed ~ U(note_speed, max)
    min_hand_gap_s: float = 0.5   # min spacing between same-hand notes
    first_note_s: float = 1.0
    cols: tuple = (0, 1, 2, 3)    # allowed columns (curriculum: restrict)
    rows: tuple = (0, 1, 2)       # allowed rows
    note_speed: float = 4.0       # m/s toward the plane; z(t) = (hit_time - t) * speed
    bomb_rate: float = 0.0        # bombs per second (0 = none)
    chain_prob: float = 0.0       # P(a non-dot note becomes a chain)
    arc_prob: float = 0.0         # P(consecutive same-color pair gets an arc)

    def __post_init__(self):
        assert self.colors in ("right", "both"), self.colors
        assert len(self.dirs) > 0 and all(0 <= d <= 7 for d in self.dirs)
        if self.note_speed_max is not None:
            assert self.note_speed_max >= self.note_speed
        assert self.note_rate > 0
        assert self.note_speed > 0
        assert 0.0 <= self.chain_prob <= 1.0 and 0.0 <= self.arc_prob <= 1.0


def note_tuple(n: Note):
    """The (t, x, y, color, angle, speed, any_dir, kind, free_pre,
    free_post) tuple the Rust sim eats (positions resolved to meters)."""
    x, y = note_center(n)
    return (n.hit_time, x, y, n.color, n.angle, n.speed, n.any_dir,
            n.kind, n.free_pre, n.free_post)


def chain_links(head: Note, tail_x: float, tail_y: float, tail_time: float,
                sc: int, squish: float = 1.0) -> list[Note]:
    """The sc-1 link Notes of a chain (head excluded), real-BS layout."""
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


def _chain_tail_cell(cfg, head: Note, rng) -> tuple[int, int]:
    """A grid cell 1-2 steps from the head, biased along its direction."""
    steps = int(rng.integers(1, 3))
    dc = round(math.cos(head.angle)) * steps
    dr = round(math.sin(head.angle)) * steps
    col = int(np.clip(head.col + dc, 0, N_COLS - 1))
    row = int(np.clip(head.row + dr, 0, N_ROWS - 1))
    if (col, row) == (head.col, head.row):   # degenerate (clamped away)
        col = int(np.clip(head.col + (1 if head.col < N_COLS - 1 else -1),
                          0, N_COLS - 1))
    return col, row


def generate_chart(cfg: ChartConfig, rng: np.random.Generator) -> list[Note]:
    notes: list[Note] = []
    last_hand_time = {LEFT: -1e9, RIGHT: -1e9}
    t = cfg.first_note_s
    while t <= cfg.chart_len_s:
        color = RIGHT if cfg.colors == "right" else int(rng.integers(0, 2))
        # feasibility: never schedule a hand faster than it can recover
        earliest = last_hand_time[color] + cfg.min_hand_gap_s
        hit = max(t, earliest)
        if hit <= cfg.chart_len_s:
            if cfg.continuous_dirs:
                angle = float(rng.uniform(0.0, 2.0 * np.pi))
                any_dir = False
            else:
                d = int(cfg.dirs[rng.integers(0, len(cfg.dirs))])
                angle = d * np.pi / 4
                any_dir = False
            if cfg.note_speed_max is not None:
                speed = float(rng.uniform(cfg.note_speed, cfg.note_speed_max))
            else:
                speed = cfg.note_speed
            head = Note(
                hit_time=float(hit),
                col=int(cfg.cols[rng.integers(0, len(cfg.cols))]),
                row=int(cfg.rows[rng.integers(0, len(cfg.rows))]),
                color=color,
                angle=angle,
                speed=speed,
                any_dir=any_dir,
            )
            if not any_dir and rng.random() < cfg.chain_prob:
                sc = int(rng.integers(CHAIN_SC[0], CHAIN_SC[1] + 1))
                tail_dt = float(rng.uniform(*CHAIN_TAIL_DT))
                tc, tr = _chain_tail_cell(cfg, head, rng)
                head = replace(head, kind=KIND_CHAIN_HEAD)
                notes.append(head)
                notes.extend(chain_links(
                    head, col_x(tc), ROW_Y[tr], hit + tail_dt, sc))
                last_hand_time[color] = hit + tail_dt
            else:
                notes.append(head)
                last_hand_time[color] = hit
        t += (1.0 / cfg.note_rate) * rng.uniform(0.8, 1.2)

    # bombs: independent stream; re-place (not reschedule) on cell clashes
    if cfg.bomb_rate > 0:
        tb = cfg.first_note_s + float(rng.uniform(0.0, 1.0 / cfg.bomb_rate))
        while tb <= cfg.chart_len_s:
            col = int(cfg.cols[rng.integers(0, len(cfg.cols))])
            row = int(cfg.rows[rng.integers(0, len(cfg.rows))])
            for _ in range(4):
                clash = any(abs(n.hit_time - tb) < 0.25
                            and abs(note_center(n)[0] - col_x(col)) < 0.3
                            and abs(note_center(n)[1] - ROW_Y[row]) < 0.3
                            for n in notes)
                if not clash:
                    break
                col = int(cfg.cols[rng.integers(0, len(cfg.cols))])
                row = int(cfg.rows[rng.integers(0, len(cfg.rows))])
            else:
                tb += (1.0 / cfg.bomb_rate) * rng.uniform(0.8, 1.2)
                continue
            notes.append(Note(hit_time=float(tb), col=col, row=row, color=0,
                              angle=0.0, speed=cfg.note_speed, any_dir=True,
                              kind=KIND_BOMB))
            tb += (1.0 / cfg.bomb_rate) * rng.uniform(0.8, 1.2)

    notes.sort(key=lambda n: n.hit_time)

    # arcs: flag consecutive same-color PLAIN-note pairs (head gets the
    # follow-through grant, tail the pre-swing grant — real BS semantics)
    if cfg.arc_prob > 0:
        last_idx = {LEFT: None, RIGHT: None}
        for i, n in enumerate(notes):
            if n.kind != KIND_NOTE:
                continue
            j = last_idx[n.color]
            if (j is not None
                    and n.hit_time - notes[j].hit_time <= ARC_MAX_GAP_S
                    and rng.random() < cfg.arc_prob):
                notes[j] = replace(notes[j], free_post=True)
                notes[i] = replace(notes[i], free_pre=True)
            last_idx[n.color] = i
    return notes
