"""Flux-mode envs over the Rust sim (SPEC.md §Flux): the single gym env
(with JSONL recording for the viewer) and the batched Rust vector env.

Same body, physics, action space (29 gear heads), obs SHAPE (OBS_DIM)
and reward terms as the standard game; the sim is built with
`flux=True`, charts come from `flux_chart.generate_flux_chart`, and the
obs hand/entity blocks carry the Flux lane meanings (constants.py
`FLUX_LANE_*`).
"""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np

from .constants import EVENT_REASONS, N_JOINTS, OBS_DIM, TAU_FEAT
from .flux_chart import FluxConfig, flux_tuple, generate_flux_chart

from beat_arena_rs import RawBeatSim, VecBeatSim


def _flux_cfg(chart_len_s, cols, rows, note_speed_min, note_speed_max,
              circle_prob, bar_prob):
    return FluxConfig(chart_len_s=chart_len_s, cols=tuple(cols),
                      rows=tuple(rows), note_speed_min=note_speed_min,
                      note_speed_max=note_speed_max,
                      circle_prob=circle_prob, bar_prob=bar_prob)


class RustFluxEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, chart_len_s=20.0, cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed_min=10.0, note_speed_max=18.0,
                 circle_prob=0.3, bar_prob=0.15,
                 external_chart=None, gears=3, seed=0, *,
                 foot_contact, fall_const=5.0, energy_coef=0.0):
        """external_chart: a fixed list[FluxObj] played on every reset.
        foot_contact: REQUIRED keyword, as RustBeatEnv (True = the
        production unilateral ground contact)."""
        self.cfg = _flux_cfg(chart_len_s, cols, rows, note_speed_min,
                             note_speed_max, circle_prob, bar_prob)
        self._base_seed = seed
        self._episode = 0
        self._record_path = None
        self._record_lines = None
        self.chart = []
        self._external = list(external_chart) if external_chart else None
        if self._external:
            max_steps = int((self._external[-1].hit_time + 0.5) * 30) + 90
        else:
            max_steps = int(chart_len_s * 30) + 90
        assert gears in (3, 5, 7), f"gears must be 3, 5, or 7, got {gears}"
        self.sim = RawBeatSim(max_steps=max_steps, gears=gears,
                              foot_contact=foot_contact,
                              fall_const=fall_const,
                              energy_coef=energy_coef, flux=True)
        self.gears = gears
        self.foot_contact = bool(foot_contact)
        self.tau_scale = [1.0] * TAU_FEAT
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([gears] * N_JOINTS)

    def set_tau_scale(self, s16):
        s = [float(x) for x in s16]
        assert len(s) == TAU_FEAT, "tau scale must have 16 axis values"
        self.sim.set_tau_scale(s)
        self.tau_scale = s

    # -- recording: the v10 JSONL body record + a Flux header/palms ----------
    def set_record(self, path: str):
        """Record the NEXT episode (and later ones) to JSONL at path."""
        self._record_path = path

    def _start_record(self):
        header = {"header": {
            "version": "v10",
            "record_rev": 5,
            "game": "flux",       # absent = the standard game
            "tau_scale": [round(v, 6) for v in self.tau_scale],
            "track_anchor": "head",
            "config": {
                "chart_len_s": self.cfg.chart_len_s,
                "note_speed_min": self.cfg.note_speed_min,
                "note_speed_max": self.cfg.note_speed_max,
                "circle_prob": self.cfg.circle_prob,
                "bar_prob": self.cfg.bar_prob,
                "gears": self.gears,
            },
            "chart": [{"t": o.hit_time, "x": round(o.x, 4),
                       "y": round(o.y, 4), "kind": o.kind,
                       "color": o.color, "rot": round(o.rot, 5),
                       "speed": round(o.speed, 4),
                       "len": round(o.length, 4),
                       "sweet": round(o.sweet, 4)}
                      for o in self.chart],
            "walls": [],
        }}
        self._record_lines = [json.dumps(header)]

    def _record_tick(self, action, reward, events):
        self._record_lines.append(json.dumps({
            "t": self.sim.tick - 1,
            "time": round(self.sim.time - 1.0 / 30.0, 5),
            "q": [round(v, 5) for v in self.sim.qpos()],
            "twist": [round(v, 4) for v in self.sim.twist()],
            "omega": [round(v, 4) for v in self.sim.omega()],
            # wrist + hand axis per hand (the standard "sabers" lanes;
            # Flux draws no blade) and the palm points the game uses
            "sabers": [[round(v, 4) for v in s] for s in self.sim.sabers()],
            "palms": [[round(v, 4) for v in p] for p in self.sim.palms()],
            "head": [round(v, 6) for v in self.sim.head_pose()],
            "action": [int(a) for a in action],
            "reward": round(float(reward), 4),
            "contacts": [{"pair": int(p), "p": [round(x, 4), round(y, 4),
                                                round(z, 4)],
                          "j": round(imp, 4)}
                         for p, x, y, z, imp in self.sim.contacts()],
            "feet": [[round(w + (g[k] if k < 3 else 0.0), 4)
                      for k, w in enumerate(wf)]
                     for wf, g in zip(self.sim.weld_impulses(),
                                      self.sim.ground_reaction())],
            "activation": [round(v, 4) for v in self.sim.last_activation()],
            "torque": [round(v, 3) for v in self.sim.last_torque()],
            "events": [
                {"miss": {"note": int(i)}} if code == 4 else
                # good touches: pre = base contact points (50 / 100),
                # acc = accuracy points
                {"cut": {"note": int(i), "good": code == 0,
                         "reason": EVENT_REASONS[code],
                         "points": round(float(pts), 2),
                         "pre": round(float(pre), 2),
                         "acc": round(float(acc), 2)}}
                for i, code, pts, pre, acc in events
            ],
        }))

    # -- gym API --------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = self._base_seed + self._episode * 9973
        self._episode += 1
        self.chart = (self._external if self._external is not None
                      else generate_flux_chart(self.cfg,
                                               np.random.default_rng(seed)))
        obs = self.sim.set_flux_chart([flux_tuple(o) for o in self.chart])
        if self._record_path is not None:
            self._start_record()
        return obs, {}

    def step(self, action):
        a = np.asarray(action).reshape(-1)
        obs, reward, done, events = self.sim.step([int(x) for x in a])
        if self._record_lines is not None:
            self._record_tick(a, reward, events)
            if done:
                with open(self._record_path, "w") as f:
                    f.write("\n".join(self._record_lines) + "\n")
                self._record_lines = None
        info = {"events": [(int(i), EVENT_REASONS[c], float(p))
                           for i, c, p, _pre, _acc in events]}
        if done:
            base, acc, _post = self.sim.score_components()
            info.update(score_base=base, score_acc=acc,
                        good=self.sim.good_cuts, bad=self.sim.bad_cuts,
                        misses=self.sim.misses, n_objects=self.sim.n_notes,
                        score=self.sim.score, falls=self.sim.falls)
        return obs, float(reward), bool(done), False, info


class FluxVecEnv:
    """Batched Flux env over VecBeatSim(flux=True) — the BeatVecEnv
    contract (persistent obs buffer, auto-reset, per-episode infos)."""

    def __init__(self, num_envs, chart_len_s=20.0, cols=(0, 1, 2, 3),
                 rows=(0, 1, 2), note_speed_min=10.0, note_speed_max=18.0,
                 circle_prob=0.3, bar_prob=0.15,
                 foot_z_coef=0.0, foot_x_coef=0.0, foot_hinge_r=0.0,
                 foot_shape_coef=0.0, energy_coef=0.0, self_contact_coef=0.0,
                 gears=3, seed=0, foot_contact=False, fall_const=5.0):
        self.num_envs = num_envs
        self.cfg = _flux_cfg(chart_len_s, cols, rows, note_speed_min,
                             note_speed_max, circle_prob, bar_prob)
        self.sim = VecBeatSim(num_envs, max_steps=int(chart_len_s * 30) + 90,
                              return_events=False, gears=gears,
                              foot_contact=foot_contact,
                              fall_const=fall_const,
                              foot_z_coef=foot_z_coef,
                              foot_x_coef=foot_x_coef,
                              foot_hinge_r=foot_hinge_r,
                              foot_shape_coef=foot_shape_coef,
                              energy_coef=energy_coef,
                              self_contact_coef=self_contact_coef,
                              flux=True)
        self.foot_contact = bool(foot_contact)
        self._rng = np.random.default_rng(seed)
        self._returns = np.zeros(num_envs, dtype=np.float64)
        self._obs = np.zeros((num_envs, OBS_DIM), dtype=np.float32)

    def pop_fall_hits(self):
        return self.sim.pop_fall_hits()

    def pop_flail(self):
        return 0.0, 0.0, 0

    def pop_self_contact(self):
        return self.sim.pop_self_contact()

    def pop_activation(self):
        return self.sim.pop_activation()

    def pop_torque(self):
        return self.sim.pop_torque()

    def set_tau_scale(self, s16):
        self.sim.set_tau_scale([float(x) for x in s16])

    def _new_chart(self, i):
        rng = np.random.default_rng(int(self._rng.integers(0, 2**63 - 1)))
        self.sim.set_flux_chart(
            i, [flux_tuple(o) for o in generate_flux_chart(self.cfg, rng)])

    def reset(self):
        for i in range(self.num_envs):
            self._new_chart(i)
        self._returns[:] = 0.0
        self._obs[:] = self.sim.obs().reshape(self.num_envs, OBS_DIM)
        return self._obs

    def step(self, actions):
        rew, done, _events = self.sim.step_into(
            np.ascontiguousarray(actions, dtype=np.int64), self._obs)
        self._returns += rew
        infos = []
        if done.any():
            for i in np.flatnonzero(done):
                g, b, m, n, score, _bombs = self.sim.counters(i)
                infos.append({"env": int(i),
                              "episode_return": float(self._returns[i]),
                              "good_cuts": g, "bad_cuts": b, "misses": m,
                              "n_notes": n, "score": score, "bombs_hit": 0})
                self._returns[i] = 0.0
                self._new_chart(i)
                self.sim.obs_one_into(int(i), self._obs)
        return self._obs, rew, done, infos
