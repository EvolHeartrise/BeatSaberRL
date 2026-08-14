"""Left-right mirroring of chart data (2026-08-14).

Reflects a chart across the x=0 plane — the grid's own symmetry axis
(col_x(c) = (c-1.5)*CELL_X, so column centers are already symmetric)
— producing the map a left-dominant player would face. Ranked maps
are overwhelmingly right-lead, so mirroring serves two jobs:

- diagnostic: bench a checkpoint on the mirrored catalog and diff
  against its normal bench to MEASURE handedness bias;
- augmentation: double the training pool so the optimal data-fitting
  policy can't camp a stance that only works right-handed.

The transform on note_tuple (t, x, y, color, angle, speed, any_dir,
kind, free_pre, free_post):
- x -> -x (grid is symmetric about 0)
- color 0 <-> 1 (red/left <-> blue/right; bombs keep color 0 — the
  field is unused for them)
- angle theta -> pi - theta mod 2pi (0 = +x CCW convention: mirrors
  the horizontal component, keeps vertical; Up->Up, Left->Right,
  UpLeft->UpRight, Down->Down)
- everything else (time, y, speed, any_dir, kind, arc flags) unchanged

wall_tuple (t, cx, cy, w, h, len, speed): cx -> -cx only — an AABB is
symmetric about its center, so reflecting the center reflects the box.

Mirroring is an involution: mirror(mirror(chart)) == chart exactly
(all ops are sign flips / swaps; pi - (pi - theta) = theta in floats
only when done mod-free, so the test pins a round-trip tolerance).
"""

from __future__ import annotations

import math

from .constants import KIND_BOMB

_TWO_PI = 2.0 * math.pi


def mirror_angle(angle):
    """theta -> pi - theta, wrapped to [0, 2pi)."""
    return (math.pi - angle) % _TWO_PI


def mirror_note_tuple(t):
    kind = t[7]
    color = t[3] if kind == KIND_BOMB else 1 - t[3]
    return (t[0], -t[1], t[2], color, mirror_angle(t[4]),
            t[5], t[6], kind, t[8], t[9])


def mirror_wall_tuple(w):
    return (w[0], -w[1], w[2], w[3], w[4], w[5], w[6])


def mirror_tuples(tuples):
    return [mirror_note_tuple(t) for t in tuples]


def mirror_walls(walls):
    return [mirror_wall_tuple(w) for w in walls]
