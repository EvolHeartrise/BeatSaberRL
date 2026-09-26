"""Procedural Flux chart generation (SPEC.md §Flux — generator f1).

There is no official Flux map data yet, so charts come from this
generator. It reuses generator v2's clock (chart.py): a jittered tick
lattice whose interval stretches by note_speed_max / walk speed, with
one shared random speed walk across all objects.

Where generator v2 keeps swing PARITY flow, Flux keeps HAND flow: the
generator walks two imaginary palms through the play rectangle (one
step per lattice tick, at most `hand_speed · dt` in a uniform disc,
left kept at least `min_sep` left of right) and places objects ON that
trajectory, so every chart is playable by a hand path it already
contains:

- **Circles**: each hand path spawns a circle at its palm's current
  position with P = circle_prob per tick (colour = that hand).
- **Bars**: with P = bar_prob per tick, a bar whose SWEET SPOT lies on
  the segment between the two palms (at fraction U(0.2, 0.8) from left
  to right), rotation θ ~ U(0, 2π), length U(bar_len_min, bar_len_max),
  sweet offset U(−1, 1); its centre follows as
  sweet_point − sweet·(len/2)·u, u = (−sin θ, cos θ).

Positions are continuous world metres in the standard grid's frame;
the default rectangle is the grid span itself (x ∈ [−0.9, 0.9] lane
centres, y ∈ the three row heights) — `cols`/`rows` narrow it like the
standard curriculum dials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .chart import SPEED_DEV, TICK_DT, TICK_JITTER
from .constants import KIND_BAR, KIND_CIRCLE, ROW_Y, col_x

LEFT, RIGHT = 0, 1


@dataclass(frozen=True)
class FluxObj:
    hit_time: float
    x: float          # world centre at the hit plane (m)
    y: float
    kind: int         # KIND_CIRCLE / KIND_BAR
    speed: float      # approach speed (m/s)
    color: int = 0    # circles: 0 left / 1 right; bars: 0 (colourless)
    rot: float = 0.0  # bars: rotation about z (rad; 0 = vertical, CCW)
    length: float = 0.0  # bars: length (m)
    sweet: float = 0.0   # bars: sweet-spot offset in [-1, 1] along the axis


@dataclass
class FluxConfig:
    chart_len_s: float = 20.0
    first_note_s: float = 1.0     # first lattice tick (s)
    cols: tuple = (0, 1, 2, 3)    # rectangle x span = these lane centres
    rows: tuple = (0, 1, 2)       # rectangle y span = these row heights
    note_speed_min: float = 10.0  # speed walk clamp (m/s) — real-NJS range
    note_speed_max: float = 18.0
    circle_prob: float = 0.3      # P(circle) per hand path per tick
    bar_prob: float = 0.15        # P(bar) per tick
    hand_speed: float = 2.5       # imaginary palm speed cap (m/s)
    min_sep: float = 0.3          # left palm stays >= this left of right (m)
    bar_len_min: float = 0.3      # bar length range (m)
    bar_len_max: float = 1.2

    def __post_init__(self):
        assert self.note_speed_max >= self.note_speed_min > 0
        for p in (self.circle_prob, self.bar_prob):
            assert 0.0 <= p <= 1.0
        assert self.bar_len_max >= self.bar_len_min > 0
        assert self.hand_speed > 0 and self.min_sep >= 0


def flux_tuple(o: FluxObj):
    """The (t, x, y, kind, color, rot, speed, length, sweet) tuple
    `set_flux_chart` eats."""
    return (o.hit_time, o.x, o.y, o.kind, o.color, o.rot, o.speed,
            o.length, o.sweet)


def bar_axis(rot: float):
    """Unit bar axis u = (−sin θ, cos θ): θ = 0 points up, CCW from the
    player's view (the note-angle convention)."""
    return (-math.sin(rot), math.cos(rot))


def sweet_point(o: FluxObj):
    """World (x, y) of a bar's sweet spot."""
    ux, uy = bar_axis(o.rot)
    s = o.sweet * 0.5 * o.length
    return (o.x + s * ux, o.y + s * uy)


def _rect(cfg):
    xs = [col_x(c) for c in cfg.cols]
    ys = [ROW_Y[r] for r in cfg.rows]
    return min(xs), max(xs), min(ys), max(ys)


def generate_flux_chart(cfg: FluxConfig, rng: np.random.Generator) -> list[FluxObj]:
    x_lo, x_hi, y_lo, y_hi = _rect(cfg)
    # the separation rule needs room; a one-lane rectangle drops it
    sep = min(cfg.min_sep, x_hi - x_lo)
    mid = 0.5 * (x_lo + x_hi)
    palms = [
        [float(rng.uniform(x_lo, max(x_lo, mid - 0.5 * sep))),
         float(rng.uniform(y_lo, y_hi))],
        [float(rng.uniform(min(x_hi, mid + 0.5 * sep), x_hi)),
         float(rng.uniform(y_lo, y_hi))],
    ]
    prev_speed = None
    objs: list[FluxObj] = []

    def walk_speed():
        nonlocal prev_speed
        if prev_speed is None:
            s = float(rng.uniform(cfg.note_speed_min, cfg.note_speed_max))
        else:
            s = float(np.clip(prev_speed + rng.uniform(-SPEED_DEV, SPEED_DEV),
                              cfg.note_speed_min, cfg.note_speed_max))
        prev_speed = s
        return s

    def step_palms(dt):
        r_max = cfg.hand_speed * dt
        for p in palms:
            # uniform in the disc of radius r_max
            r = r_max * math.sqrt(float(rng.random()))
            a = float(rng.uniform(0.0, 2.0 * math.pi))
            p[0] = float(np.clip(p[0] + r * math.cos(a), x_lo, x_hi))
            p[1] = float(np.clip(p[1] + r * math.sin(a), y_lo, y_hi))
        gap = palms[RIGHT][0] - palms[LEFT][0]
        if gap < sep:
            # push both apart symmetrically, then re-clamp and fix up
            # against the wall that clipped
            c = 0.5 * (palms[LEFT][0] + palms[RIGHT][0])
            lx = min(max(c - 0.5 * sep, x_lo), x_hi - sep)
            palms[LEFT][0] = lx
            palms[RIGHT][0] = lx + sep

    t = cfg.first_note_s
    dt = 0.0
    while t <= cfg.chart_len_s:
        step_palms(dt)
        for hand in (LEFT, RIGHT):
            if rng.random() < cfg.circle_prob:
                objs.append(FluxObj(hit_time=float(t), x=palms[hand][0],
                                    y=palms[hand][1], kind=KIND_CIRCLE,
                                    speed=walk_speed(), color=hand))
        if rng.random() < cfg.bar_prob:
            f = float(rng.uniform(0.2, 0.8))
            px = palms[LEFT][0] + f * (palms[RIGHT][0] - palms[LEFT][0])
            py = palms[LEFT][1] + f * (palms[RIGHT][1] - palms[LEFT][1])
            rot = float(rng.uniform(0.0, 2.0 * math.pi))
            length = float(rng.uniform(cfg.bar_len_min, cfg.bar_len_max))
            sweet = float(rng.uniform(-1.0, 1.0))
            ux, uy = bar_axis(rot)
            s = sweet * 0.5 * length
            objs.append(FluxObj(hit_time=float(t), x=px - s * ux,
                                y=py - s * uy, kind=KIND_BAR,
                                speed=walk_speed(), rot=rot, length=length,
                                sweet=sweet))
        pace = cfg.note_speed_max / (prev_speed if prev_speed is not None
                                     else cfg.note_speed_max)
        dt = (TICK_DT + float(rng.uniform(-TICK_JITTER, TICK_JITTER))) * pace
        t += dt

    objs.sort(key=lambda o: o.hit_time)
    return objs
