"""Batched env over the Rust VecBeatSim: N envs, one FFI call per vector
step, auto-reset with Python-side chart generation.

Semantics match the PufferLib Serial path used so far: when an env
finishes, its returned obs row is replaced by the fresh reset obs, and
episode stats (return + cut counters) are emitted in `infos`.

Since 2026-08-02 reset()/step() return views of ONE persistent
(N, OBS_DIM) buffer filled by the zero-copy `step_into` FFI (the
bench_scoresaber pattern) instead of a fresh 696 KB-at-N=512 allocation
per step — callers must consume/copy the obs before the next step()
(the trainer and tests already do).
"""

from __future__ import annotations

import numpy as np

from .chart import ChartConfig, generate_chart, note_tuple
from .constants import OBS_DIM

from beat_arena_rs import VecBeatSim


class BeatVecEnv:
    def __init__(self, num_envs, colors="both", chart_len_s=20.0,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed_min=2.0, note_speed_max=8.0,
                 note_prob=0.25, extra_prob=0.125, dot_prob=0.1,
                 bomb_prob=0.125, wide_acc=False,
                 foot_z_coef=0.0, foot_x_coef=0.0, foot_hinge_r=0.0,
                 foot_shape_coef=0.0, iso_coef=0.0, self_contact_coef=0.0,
                 gears=3, seed=0, foot_contact=False, fall_const=5.0):
        """foot_contact: env v11 foot_mode (unilateral-contact spec §1);
        False = the v10 foot welds (TEST-HARNESS ONLY since 2026-08-16
        — production is contact-only and train/bench force it), True =
        unilateral ground contact + the fall termination."""
        self.num_envs = num_envs
        self.cfg = ChartConfig(
            chart_len_s=chart_len_s, colors=colors,
            cols=tuple(cols), rows=tuple(rows),
            note_speed_min=note_speed_min, note_speed_max=note_speed_max,
            note_prob=note_prob, extra_prob=extra_prob, dot_prob=dot_prob,
            bomb_prob=bomb_prob,
        )
        # +90: the episode tail must cover the miss boundary trip
        # (live head plane, env v10) past the last note
        # return_events=False: the event list is discarded below; skip
        # materializing it into Python tuples every step
        self.sim = VecBeatSim(num_envs, max_steps=int(chart_len_s * 30) + 90, return_events=False,
                              wide_acc=wide_acc, gears=gears,
                              foot_contact=foot_contact,
                              fall_const=fall_const,
                              foot_z_coef=foot_z_coef,
                              foot_x_coef=foot_x_coef,
                              foot_hinge_r=foot_hinge_r,
                              foot_shape_coef=foot_shape_coef,
                              iso_coef=iso_coef,
                              self_contact_coef=self_contact_coef)
        self.foot_contact = bool(foot_contact)
        self._rng = np.random.default_rng(seed)
        self._returns = np.zeros(num_envs, dtype=np.float64)
        self._obs = np.zeros((num_envs, OBS_DIM), dtype=np.float32)

    def pop_fall_hits(self):
        """FALL terminations summed over all envs since the last pop
        (unilateral-contact spec §4) — the VecBeatSim getter, exposed so
        the trainer's [fall] line works on this backend too. Always 0
        without foot_contact=True."""
        return self.sim.pop_fall_hits()

    def pop_flail(self):
        """env v19 flail telemetry — CUDA-backend-only; this Rust-backed
        wrapper reports zeros so the trainer's [flail] line degrades
        gracefully instead of branching on backend. Per-hand (L, R,
        ticks) since the CUDA contract went per-hand on 2026-08-20 —
        the 2-tuple stub had broken the rust-backend trainer path
        (found 2026-09-06 by the --head-clip CPU smoke)."""
        return 0.0, 0.0, 0

    def pop_self_contact(self):
        """env v41 self-contact telemetry since the last pop (impulse
        N.s, ticks with contact, env-ticks) — forwarded to the Rust sim."""
        return self.sim.pop_self_contact()

    def _new_chart(self, i):
        rng = np.random.default_rng(int(self._rng.integers(0, 2**63 - 1)))
        self.sim.set_chart(i, [note_tuple(n) for n in generate_chart(self.cfg, rng)])

    def reset(self):
        for i in range(self.num_envs):
            self._new_chart(i)
        self._returns[:] = 0.0
        self._obs[:] = self.sim.obs().reshape(self.num_envs, OBS_DIM)
        return self._obs

    def step(self, actions):
        """actions: (N, 29) int array — gear ids in SPEC-v10 §1.4 joint
        order. Returns (obs, rewards, dones, infos); infos = list of
        dicts for envs that finished THIS step. obs is a view of the
        persistent buffer — consume before the next step()."""
        rew, done, _events = self.sim.step_into(  # events empty (ctor flag)
            np.ascontiguousarray(actions, dtype=np.int64), self._obs)
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
                self.sim.obs_one_into(int(i), self._obs)  # in-place row
        return self._obs, rew, done, infos
