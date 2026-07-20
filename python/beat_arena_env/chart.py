"""Procedural chart generation (SPEC.md §Chart generation).

A chart is a list of Note(hit_time, col, row, color, dir) sorted by
hit_time. Difficulty is controlled by the config knobs, the analog of
kiting-arena's K range: note_rate is the main dial, colors/dirs widen the
skill demanded, min_hand_gap_s guarantees physical feasibility (one hand is
never asked to be in two places faster than it can travel).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_COLS = 4
N_ROWS = 3
LEFT, RIGHT = 0, 1


@dataclass(frozen=True)
class Note:
    hit_time: float
    col: int
    row: int
    color: int  # 0 = left/red, 1 = right/blue
    dir: int    # 0..7, required swing direction = dir * 45 deg (0 = +x, CCW)


@dataclass
class ChartConfig:
    note_rate: float = 1.0        # notes per second (overall)
    chart_len_s: float = 20.0
    colors: str = "right"         # "right" (all blue) or "both"
    dirs: tuple = (6,)            # allowed cut directions (6 = down-swing)
    min_hand_gap_s: float = 0.5   # min spacing between same-hand notes
    first_note_s: float = 1.0
    cols: tuple = (0, 1, 2, 3)    # allowed columns (curriculum: restrict)
    rows: tuple = (0, 1, 2)       # allowed rows
    note_speed: float = 4.0       # m/s toward the plane; z(t) = (hit_time - t) * speed
    # sim-level knob (this dataclass doubles as the env config): coefficient
    # of the potential-based shaping reward, 0 disables (SPEC.md §Sim rules).
    # DEFAULT 0 since 2026-07-20 (user direction): rewards come purely from
    # the sim's scoring, like real Beat Saber; the mechanism remains for
    # ablation studies only.
    shaping_coef: float = 0.0

    def __post_init__(self):
        assert self.colors in ("right", "both"), self.colors
        assert len(self.dirs) > 0 and all(0 <= d <= 7 for d in self.dirs)
        assert self.note_rate > 0
        assert self.shaping_coef >= 0
        assert self.note_speed > 0


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
            notes.append(
                Note(
                    hit_time=float(hit),
                    col=int(cfg.cols[rng.integers(0, len(cfg.cols))]),
                    row=int(cfg.rows[rng.integers(0, len(cfg.rows))]),
                    color=color,
                    dir=int(cfg.dirs[rng.integers(0, len(cfg.dirs))]),
                )
            )
            last_hand_time[color] = hit
        t += (1.0 / cfg.note_rate) * rng.uniform(0.8, 1.2)
    notes.sort(key=lambda n: n.hit_time)
    return notes
