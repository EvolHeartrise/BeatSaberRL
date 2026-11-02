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
from .constants import OBS_DIM, joint_gear_counts

from beat_arena_rs import VecBeatSim


class BeatVecEnv:
    def __init__(self, num_envs, colors="both", chart_len_s=20.0,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed_min=2.0, note_speed_max=8.0,
                 note_prob=0.25, extra_prob=0.125, dot_prob=0.1,
                 bomb_prob=0.125,
                 foot_z_coef=0.0, foot_x_coef=0.0, foot_hinge_r=0.0,
                 foot_shape_coef=0.0, energy_coef=0.0, self_contact_coef=0.0,
                 gears=3, seed=0, fall_const=5.0,
                 limit_coef=0.0, limit_arms=False, wall_despawn=False,
                 tau_rand=0.0, tau_center=None):
        """Unilateral ground contact + the fall termination is the only
        foot mode (the v10 weld harness and its `foot_contact=` argument
        were removed 2026-09-27).

        limit_coef / limit_arms / wall_despawn / tau_rand / tau_center
        (2026-09-30): the CudaBeatVecEnv reward/body kwargs, which the
        trainer's --env rust path used to drop silently (the Rust sim
        always had the first three). tau_rand > 0 draws each env's 16
        axis scales center * (1 + r U(-1,1)) at every episode reset, the
        CUDA semantics, from a numpy stream of its own (the chart stream
        is unchanged); tau_rand 0 + tau_center fixes every env at the
        center. No per-episode tau_draws log here (CUDA pop_episodes)."""
        self.num_envs = num_envs
        self.gears = gears
        # per-joint action-id counts (constants.joint_gear_counts)
        self.joint_gears = joint_gear_counts(gears)
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
                              gears=gears,
                              fall_const=fall_const,
                              foot_z_coef=foot_z_coef,
                              foot_x_coef=foot_x_coef,
                              foot_hinge_r=foot_hinge_r,
                              foot_shape_coef=foot_shape_coef,
                              energy_coef=energy_coef,
                              limit_coef=limit_coef,
                              limit_arms=bool(limit_arms),
                              wall_despawn=bool(wall_despawn),
                              self_contact_coef=self_contact_coef)
        self._rng = np.random.default_rng(seed)
        self.tau_rand = float(tau_rand)
        assert 0.0 <= self.tau_rand < 1.0, "tau_rand in [0, 1)"
        tc = [1.0] * 16 if tau_center is None else [float(x) for x in tau_center]
        assert len(tc) == 16 and all(x > 0 for x in tc), \
            "tau_center: 16 positive floats"
        self.tau_center = np.asarray(tc, dtype=np.float64)
        self._tau_rng = np.random.default_rng([int(seed), 46])
        if self.tau_rand == 0.0 and tau_center is not None:
            self.sim.set_tau_scale(self.tau_center.tolist())
        self._returns = np.zeros(num_envs, dtype=np.float64)
        self._obs = np.zeros((num_envs, OBS_DIM), dtype=np.float32)

    def pop_fall_hits(self):
        """FALL terminations summed over all envs since the last pop
        (unilateral-contact spec §4) — the VecBeatSim getter, exposed so
        the trainer's [fall] line works on this backend too."""
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

    def pop_activation(self):
        """Activation telemetry since the last pop (sum total, sum
        spine, sum arms, sum legs, env-ticks) — forwarded to the Rust
        sim."""
        return self.sim.pop_activation()

    def pop_torque(self):
        """env v43 torque telemetry: (sum total, sum spine, sum arms,
        sum legs, env-ticks) of applied |tau| in N.m — CUDA mirror."""
        return self.sim.pop_torque()

    def pop_shoulder(self):
        """exp/shoulder-rom telemetry since the last pop: (sum |tau_p| L,
        sum |tau_p| R, zone L, zone R, beyond L, beyond R, dissipated
        power L, dissipated power R, substeps) — forwarded to the Rust sim
        (CUDA pop_shoulder mirror)."""
        return self.sim.pop_shoulder()

    def pop_tendon(self):
        """exp/wrist-tendon telemetry since the last pop: (sum |tau_p| L,
        sum |tau_p| R, zone L, zone R, beyond L, beyond R, dissipated
        power L, dissipated power R, substeps) —
        forwarded to the Rust sim (CUDA pop_tendon mirror)."""
        return self.sim.pop_tendon()

    def set_tau_scale(self, s16):
        """env v46: fix the 16 per-axis torque-limit scales on every env
        (CUDA set_tau_scale mirror). Needs tau_rand == 0, otherwise the
        next reset redraws them (as on CUDA)."""
        assert self.tau_rand == 0.0, "set_tau_scale needs tau_rand == 0"
        self.sim.set_tau_scale([float(x) for x in s16])

    def _new_chart(self, i):
        rng = np.random.default_rng(int(self._rng.integers(0, 2**63 - 1)))
        self.sim.set_chart(i, [note_tuple(n) for n in generate_chart(self.cfg, rng)])
        if self.tau_rand > 0.0:
            # env v46 per-episode body draw (the CUDA reset's semantics)
            u = self._tau_rng.uniform(-1.0, 1.0, 16)
            self.sim.set_tau_scale_env(
                int(i), (self.tau_center * (1.0 + self.tau_rand * u)).tolist())

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
