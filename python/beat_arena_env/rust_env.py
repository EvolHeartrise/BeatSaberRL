"""Gymnasium wrapper over the Rust sim core (rust/src/lib.rs).

Drop-in replacement for BeatEnv on the training/eval path: same obs/action
spaces, same chart generation (Python stays the single source of truth for
charts — one generate_chart call per episode), same episode-end info dict.
No replay recording (use the Python BeatEnv when recording).

Build the extension once with:
  VIRTUAL_ENV=<venv> <venv>/bin/maturin develop --release   (from rust/)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .chart import ChartConfig, generate_chart
from .env import OBS_DIM

from beat_arena_rs import RawBeatSim


class RustBeatEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, note_rate=1.0, colors="right", dirs=(6,),
                 chart_len_s=20.0, min_hand_gap_s=0.5, shaping_coef=0.0,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2), note_speed=4.0, seed=0):
        self.cfg = ChartConfig(
            note_rate=note_rate, colors=colors, dirs=tuple(dirs),
            chart_len_s=chart_len_s, min_hand_gap_s=min_hand_gap_s,
            shaping_coef=shaping_coef, cols=tuple(cols), rows=tuple(rows),
            note_speed=note_speed,
        )
        self._base_seed = seed
        self._episode = 0
        max_steps = int(chart_len_s * 30) + 30
        self.sim = RawBeatSim(note_speed=note_speed, shaping_coef=shaping_coef,
                              max_steps=max_steps)
        self.observation_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([9, 9])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = self._base_seed + self._episode * 9973
        self._episode += 1
        chart = generate_chart(self.cfg, np.random.default_rng(seed))
        obs = self.sim.set_chart([(n.hit_time, n.col, n.row, n.color, n.dir)
                                  for n in chart])
        return obs, {}

    def step(self, action):
        a = np.asarray(action).reshape(-1)
        obs, reward, done = self.sim.step(int(a[0]), int(a[1]))
        info = {}
        if done:
            info = {
                "good_cuts": self.sim.good_cuts,
                "bad_cuts": self.sim.bad_cuts,
                "misses": self.sim.misses,
                "n_notes": self.sim.n_notes,
            }
        return obs, float(reward), bool(done), False, info
