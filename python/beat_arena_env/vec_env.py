"""Batched env over the Rust VecBeatSim: N envs, one FFI call per vector
step, auto-reset with Python-side chart generation.

Semantics match the PufferLib Serial path used so far: when an env
finishes, its returned obs row is replaced by the fresh reset obs, and
episode stats (return + cut counters) are emitted in `infos`.
"""

from __future__ import annotations

import numpy as np

from .chart import ChartConfig, generate_chart, note_tuple
from .constants import OBS_DIM

from beat_arena_rs import VecBeatSim


class BeatVecEnv:
    def __init__(self, num_envs, note_rate=1.0, colors="right", dirs=(6,),
                 continuous_dirs=False, chart_len_s=20.0, min_hand_gap_s=0.5,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed=4.0, note_speed_max=None, seed=0,
                 bomb_rate=0.25, dot_prob=0.0, chain_prob=0.0, arc_prob=0.0):
        self.num_envs = num_envs
        self.cfg = ChartConfig(
            note_rate=note_rate, colors=colors, dirs=tuple(dirs),
            continuous_dirs=continuous_dirs, chart_len_s=chart_len_s,
            min_hand_gap_s=min_hand_gap_s,
            cols=tuple(cols), rows=tuple(rows), note_speed=note_speed,
            note_speed_max=note_speed_max,
            bomb_rate=bomb_rate, dot_prob=dot_prob, chain_prob=chain_prob,
            arc_prob=arc_prob,
        )
        # +90: the episode tail must cover the miss boundary trip
        # (Z_MISS 2.65 m at the slowest speeds) past the last note
        self.sim = VecBeatSim(num_envs, max_steps=int(chart_len_s * 30) + 90)
        self._rng = np.random.default_rng(seed)
        self._returns = np.zeros(num_envs, dtype=np.float64)

    def _new_chart(self, i):
        rng = np.random.default_rng(int(self._rng.integers(0, 2**63 - 1)))
        self.sim.set_chart(i, [note_tuple(n) for n in generate_chart(self.cfg, rng)])

    def reset(self):
        for i in range(self.num_envs):
            self._new_chart(i)
        self._returns[:] = 0.0
        return self.sim.obs().reshape(self.num_envs, OBS_DIM)

    def step(self, actions):
        """actions: (N, 12) int array — per hand (pitch, yaw, twist,
        elbow, roll, bend), left first. Returns (obs, rewards, dones,
        infos); infos = list of dicts for envs that finished THIS step."""
        obs, rew, done = self.sim.step(np.ascontiguousarray(actions, dtype=np.int64))
        obs = obs.reshape(self.num_envs, OBS_DIM)
        self._returns += rew
        infos = []
        if done.any():
            for i in np.flatnonzero(done):
                g, b, m, n, score, bombs = self.sim.counters(i)
                infos.append({"env": int(i), "episode_return": float(self._returns[i]),
                              "good_cuts": g, "bad_cuts": b, "misses": m,
                              "n_notes": n, "score": score, "bombs_hit": bombs})
                self._returns[i] = 0.0
                self._new_chart(i)
                obs[i] = self.sim.obs_one(int(i))
        return obs, rew, done, infos
