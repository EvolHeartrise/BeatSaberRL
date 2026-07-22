"""Gymnasium env over the Rust sim core — THE single-env implementation.

Chart generation stays in Python (once per episode); physics, cut logic,
scoring, and obs building live in rust/src/lib.rs. Per-step cut/miss
events come back from Rust and are exposed in `info["events"]` as
(note_idx, reason, points) tuples; replay recording (JSONL v5, SPEC.md)
is built from them via `set_record`.

Build the extension once with:
  VIRTUAL_ENV=<venv> <venv>/bin/maturin develop --release   (from rust/)
"""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np

from .chart import ChartConfig, generate_chart, note_tuple
from .constants import EVENT_REASONS, OBS_DIM

from beat_arena_rs import RawBeatSim


class RustBeatEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, note_rate=1.0, colors="right", dirs=(6,),
                 continuous_dirs=False, chart_len_s=20.0, min_hand_gap_s=0.5,
                 shaping_coef=0.0, cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed=4.0, note_speed_max=None, external_chart=None,
                 seed=0):
        """external_chart: a fixed list[Note] (e.g. a parsed BeatSaver map)
        played on EVERY reset instead of procedural generation."""
        self.cfg = ChartConfig(
            note_rate=note_rate, colors=colors, dirs=tuple(dirs),
            continuous_dirs=continuous_dirs, chart_len_s=chart_len_s,
            min_hand_gap_s=min_hand_gap_s, shaping_coef=shaping_coef,
            cols=tuple(cols), rows=tuple(rows), note_speed=note_speed,
            note_speed_max=note_speed_max,
        )
        self._base_seed = seed
        self._episode = 0
        self._record_path = None
        self._record_lines = None
        self.chart = []
        self._external = list(external_chart) if external_chart else None
        if self._external:
            max_steps = int((self._external[-1].hit_time + 0.5) * 30) + 30
        else:
            max_steps = int(chart_len_s * 30) + 30
        self.sim = RawBeatSim(shaping_coef=shaping_coef, max_steps=max_steps)
        self.observation_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )
        # per hand: move(9), rot(3), tilt(3); left hand first
        self.action_space = gym.spaces.MultiDiscrete([9, 3, 3, 9, 3, 3])

    # -- recording (JSONL v4, same format the 3D viewer reads) ---------------
    def set_record(self, path: str):
        """Record the NEXT episode (and later ones) to JSONL at path."""
        self._record_path = path

    def _start_record(self):
        header = {"header": {
            "version": 5,
            "config": {
                "note_rate": self.cfg.note_rate,
                "chart_len_s": self.cfg.chart_len_s,
                "colors": self.cfg.colors,
                "dirs": list(self.cfg.dirs),
                "min_hand_gap_s": self.cfg.min_hand_gap_s,
                "note_speed": self.cfg.note_speed,
                "note_speed_max": self.cfg.note_speed_max,
                "continuous_dirs": self.cfg.continuous_dirs,
            },
            "chart": [{"t": n.hit_time, "col": n.col, "row": n.row,
                       "color": n.color, "angle": round(n.angle, 5),
                       "speed": round(n.speed, 4), "any": n.any_dir}
                      for n in self.chart],
        }}
        self._record_lines = [json.dumps(header)]

    def _record_tick(self, action, reward, events):
        sabers = self.sim.sabers()
        self._record_lines.append(json.dumps({
            "t": self.sim.tick - 1,
            "time": round(self.sim.time - 1.0 / 30.0, 5),
            "sabers": [[round(v, 4) for v in s] for s in sabers],
            "action": [int(a) for a in action],
            "reward": round(float(reward), 4),
            "events": [
                {"miss": {"note": int(i)}} if code == 4 else
                {"post": {"note": int(i), "points": round(float(pts), 2)}}
                if code == 5 else
                {"cut": {"note": int(i), "good": code == 0,
                         "reason": EVENT_REASONS[code],
                         "points": round(float(pts), 2)}}
                for i, code, pts in events
            ],
        }))

    # -- gym API --------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = self._base_seed + self._episode * 9973
        self._episode += 1
        self.chart = (self._external if self._external is not None
                      else generate_chart(self.cfg, np.random.default_rng(seed)))
        obs = self.sim.set_chart([note_tuple(n) for n in self.chart])
        if self._record_path is not None:
            self._start_record()
        return obs, {}

    def step(self, action):
        a = np.asarray(action).reshape(-1)
        obs, reward, done, events = self.sim.step(
            int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), int(a[5]))
        if self._record_lines is not None:
            self._record_tick(a, reward, events)
            if done:
                with open(self._record_path, "w") as f:
                    f.write("\n".join(self._record_lines) + "\n")
                self._record_lines = None
        info = {"events": [(int(i), EVENT_REASONS[c], float(p))
                           for i, c, p in events]}
        if done:
            info.update(good_cuts=self.sim.good_cuts, bad_cuts=self.sim.bad_cuts,
                        misses=self.sim.misses, n_notes=self.sim.n_notes,
                        score=self.sim.score)
        return obs, float(reward), bool(done), False, info
