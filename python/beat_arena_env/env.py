"""Gymnasium wrapper + observation builder (SPEC.md §Observation layout).

Flat f32 obs: sabers(8) ++ notes(12 x 14, row-major)(168) ++ note_mask(12)
-> length 188. Offsets are a cross-file contract mirrored in policy.py.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .chart import ChartConfig
from .sim import BeatSim, V_MAX

N_NOTE_SLOTS = 12
NOTE_FEAT = 14
OFF_NOTES = 8
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT  # 176
OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS                # 188

TIME_SCALE = 2.0
T_UNTIL_CLIP = (-0.2, 3.0)


def build_obs(sim: BeatSim) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    for s in range(2):
        obs[s * 4 + 0] = sim.pos[s, 0] / 4.0
        obs[s * 4 + 1] = sim.pos[s, 1] / 3.0
        obs[s * 4 + 2] = sim.vel[s, 0] / V_MAX
        obs[s * 4 + 3] = sim.vel[s, 1] / V_MAX
    pending = sim.pending_notes()[:N_NOTE_SLOTS]
    for slot, i in enumerate(pending):
        n = sim.chart[i]
        base = OFF_NOTES + slot * NOTE_FEAT
        t_until = np.clip(n.hit_time - sim.time, *T_UNTIL_CLIP)
        obs[base + 0] = t_until / TIME_SCALE
        obs[base + 1] = (n.col + 0.5) / 4.0
        obs[base + 2] = (n.row + 0.5) / 3.0
        obs[base + 3 + n.color] = 1.0          # is_left, is_right
        obs[base + 5 + n.dir] = 1.0            # dir one-hot
        obs[base + 13] = 1.0                   # valid
        obs[OFF_NOTE_MASK + slot] = 1.0
    return obs


class BeatEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, note_rate=1.0, colors="right", dirs=(6,),
                 chart_len_s=20.0, min_hand_gap_s=0.5, shaping_coef=0.2,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2), seed=0):
        self.cfg = ChartConfig(
            note_rate=note_rate, colors=colors, dirs=tuple(dirs),
            chart_len_s=chart_len_s, min_hand_gap_s=min_hand_gap_s,
            shaping_coef=shaping_coef, cols=tuple(cols), rows=tuple(rows),
        )
        self._base_seed = seed
        self._episode = 0
        self._record_path = None
        self.sim: BeatSim | None = None
        self.observation_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([9, 9])

    def set_record(self, path: str):
        """Record the NEXT episode (and later ones) to JSONL at path."""
        self._record_path = path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = self._base_seed + self._episode * 9973
        self._episode += 1
        self.sim = BeatSim(self.cfg, seed=seed)
        if self._record_path is not None:
            self.sim.set_record(self._record_path)
        return build_obs(self.sim), {}

    def step(self, action):
        a = np.asarray(action).reshape(-1)
        reward, done, events = self.sim.step(int(a[0]), int(a[1]))
        obs = build_obs(self.sim)
        info = {}
        if done:
            info = {
                "good_cuts": self.sim.good_cuts,
                "bad_cuts": self.sim.bad_cuts,
                "misses": self.sim.misses,
                "n_notes": len(self.sim.chart),
            }
        return obs, float(reward), bool(done), False, info
