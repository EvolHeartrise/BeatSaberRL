"""Gymnasium wrapper + observation builder (SPEC.md §Observation layout).

Obs v2 (spatial notes): flat f32 vector
sabers(8) ++ notes(12 x 15, row-major)(180) ++ note_mask(12) -> length 200.
Offsets are a cross-file contract mirrored in policy.py.

Notes carry (x, y, z, vz) — z is distance from the hit plane (+z incoming),
vz the (negative) approach velocity. There is NO explicit time feature:
with the approach axis in the obs, time-until-hit = z / |vz| is a
reparametrization the network reads directly, exactly as a player reads
depth. vz is included so timing stays recoverable if per-note speeds ever
vary (z alone would then be an insufficient statistic for arrival).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .chart import ChartConfig
from .sim import BeatSim, V_MAX, note_center

N_NOTE_SLOTS = 12
NOTE_FEAT = 15
OFF_NOTES = 8
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT  # 188
OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS                # 200

X_NORM, Y_NORM = 1.0, 0.75     # grid half-extents (m)
Z_SCALE = 8.0                  # normalizer for z and vz
Z_CLIP = (-1.0, 12.0)          # keep "just past the plane" visible; cap lookahead


def build_obs(sim: BeatSim) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    for s in range(2):
        obs[s * 4 + 0] = sim.pos[s, 0] / X_NORM
        obs[s * 4 + 1] = sim.pos[s, 1] / Y_NORM
        obs[s * 4 + 2] = sim.vel[s, 0] / V_MAX
        obs[s * 4 + 3] = sim.vel[s, 1] / V_MAX
    pending = sim.pending_notes()[:N_NOTE_SLOTS]
    for slot, i in enumerate(pending):
        n = sim.chart[i]
        cx, cy = note_center(n)
        base = OFF_NOTES + slot * NOTE_FEAT
        z = np.clip(sim.note_z(i), *Z_CLIP)
        obs[base + 0] = cx / X_NORM
        obs[base + 1] = cy / Y_NORM
        obs[base + 2] = z / Z_SCALE
        obs[base + 3] = -sim.cfg.note_speed / Z_SCALE   # vz (toward the plane)
        obs[base + 4 + n.color] = 1.0                   # is_left, is_right
        obs[base + 6 + n.dir] = 1.0                     # dir one-hot
        obs[base + 14] = 1.0                            # valid
        obs[OFF_NOTE_MASK + slot] = 1.0
    return obs


class BeatEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, note_rate=1.0, colors="right", dirs=(6,),
                 chart_len_s=20.0, min_hand_gap_s=0.5, shaping_coef=0.2,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2), note_speed=4.0, seed=0):
        self.cfg = ChartConfig(
            note_rate=note_rate, colors=colors, dirs=tuple(dirs),
            chart_len_s=chart_len_s, min_hand_gap_s=min_hand_gap_s,
            shaping_coef=shaping_coef, cols=tuple(cols), rows=tuple(rows),
            note_speed=note_speed,
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
