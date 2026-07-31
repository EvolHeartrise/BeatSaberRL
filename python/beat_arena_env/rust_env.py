"""Gymnasium env over the Rust sim core — THE single-env implementation.

Env v10 (P3): the body is the SPEC-v10 dynamic humanoid — 29 gear-id
action heads (SPEC-v10 §1.4 joint order), obs 340 (§5), replay JSONL
v10 (§6: q36 + root twist + omega29 + per-hand FK sabers + contact
events with pair ids). Chart generation stays in Python (once per
episode); physics, cut logic, scoring, and obs building live in
rust/src/lib.rs. Per-step cut/miss events come back from Rust and are
exposed in `info["events"]` as (note_idx, reason, points) tuples.

Build the extension once with:
  VIRTUAL_ENV=<venv> <venv>/bin/maturin develop --release   (from rust/)
"""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np

from .chart import ChartConfig, generate_chart, note_tuple
from .constants import (
    EFFORT_COEF, EVENT_REASONS, N_JOINTS, OBS_DIM, note_center,
)

from beat_arena_rs import RawBeatSim


class RustBeatEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, colors="both", chart_len_s=20.0,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed_min=2.0, note_speed_max=8.0,
                 note_prob=0.25, extra_prob=0.125, dot_prob=0.1,
                 bomb_prob=0.125, effort_coef=EFFORT_COEF,
                 external_chart=None, wide_acc=False, gears=3, seed=0):
        """external_chart: a fixed list[Note] (e.g. a parsed BeatSaver map)
        played on EVERY reset instead of procedural generation."""
        self.cfg = ChartConfig(
            chart_len_s=chart_len_s, colors=colors,
            cols=tuple(cols), rows=tuple(rows),
            note_speed_min=note_speed_min, note_speed_max=note_speed_max,
            note_prob=note_prob, extra_prob=extra_prob, dot_prob=dot_prob,
            bomb_prob=bomb_prob,
        )
        self._base_seed = seed
        self._episode = 0
        self._record_path = None
        self._record_lines = None
        self.chart = []
        self._external = list(external_chart) if external_chart else None
        # +90 ticks: the episode tail must cover the miss-boundary trip
        # (live head plane, env v10) past the last note
        if self._external:
            max_steps = int((self._external[-1].hit_time + 0.5) * 30) + 90
        else:
            max_steps = int(chart_len_s * 30) + 90
        # wide_acc: TRAINING-ONLY wide accuracy reward (30*clamp(1-d/0.6),
        # unrounded — double the real range and reward); default False so
        # bench/eval/replay paths keep real scoring
        assert gears in (3, 5, 7), f"gears must be 3, 5, or 7, got {gears}"
        self.sim = RawBeatSim(max_steps=max_steps, effort_coef=effort_coef,
                              wide_acc=wide_acc, gears=gears)
        self.effort_coef = effort_coef
        self.wide_acc = wide_acc
        self.gears = gears
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        # env v10: 29 gear-id heads in SPEC-v10 §1.4 joint order —
        # spine(3), arm_l(7), arm_r(7), leg_l(6), leg_r(6); id 0 coast,
        # 1/2 = +/-full, finer +/- gear pairs append (constants.gear_table)
        self.action_space = gym.spaces.MultiDiscrete([gears] * N_JOINTS)

    # -- recording (JSONL v10 — SPEC-v10 §6, same format the viewer reads) ----
    def set_record(self, path: str):
        """Record the NEXT episode (and later ones) to JSONL at path."""
        self._record_path = path

    def _start_record(self):
        header = {"header": {
            "version": "v10",     # env v10 body: viewers reject others
            "config": {
                "chart_len_s": self.cfg.chart_len_s,
                "colors": self.cfg.colors,
                "note_speed_min": self.cfg.note_speed_min,
                "note_speed_max": self.cfg.note_speed_max,
                "note_prob": self.cfg.note_prob,
                "extra_prob": self.cfg.extra_prob,
                "dot_prob": self.cfg.dot_prob,
                "bomb_prob": self.cfg.bomb_prob,
                "effort_coef": self.effort_coef,
                "gears": self.gears,       # env v8 action-table size
            },
            # explicit world positions (meters), object kinds and arc
            # grants; the viewer derives everything from these
            "chart": [{"t": n.hit_time,
                       "x": round(note_center(n)[0], 4),
                       "y": round(note_center(n)[1], 4),
                       "color": n.color, "angle": round(n.angle, 5),
                       "speed": round(n.speed, 4), "any": n.any_dir,
                       "kind": n.kind,
                       "fpre": n.free_pre, "fpost": n.free_post}
                      for n in self.chart],
        }}
        self._record_lines = [json.dumps(header)]

    def _record_tick(self, action, reward, events):
        # SPEC-v10 §6 per-tick record: q36 (root pos + quat wxyz + 29
        # joints), root twist, omega29, per-hand FK sabers [w3 + b3],
        # contact events (pair id, world point, tick-summed impulse),
        # score events unchanged from v7+.
        self._record_lines.append(json.dumps({
            "t": self.sim.tick - 1,
            "time": round(self.sim.time - 1.0 / 30.0, 5),
            "q": [round(v, 5) for v in self.sim.qpos()],
            "twist": [round(v, 4) for v in self.sim.twist()],
            "omega": [round(v, 4) for v in self.sim.omega()],
            "sabers": [[round(v, 4) for v in s] for s in self.sim.sabers()],
            "action": [int(a) for a in action],
            "reward": round(float(reward), 4),
            "contacts": [{"pair": int(p), "p": [round(x, 4), round(y, 4),
                                                round(z, 4)],
                          "j": round(imp, 4)}
                         for p, x, y, z, imp in self.sim.contacts()],
            "events": [
                {"miss": {"note": int(i)}} if code == 4 else
                {"post": {"note": int(i), "points": round(float(pts), 2)}}
                if code == 5 else
                {"bomb": {"note": int(i)}} if code == 6 else
                # good cuts carry the component split (2026-07-25):
                # points = pre + acc (+30 on arc-headed free_post grants)
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
                      else generate_chart(self.cfg, np.random.default_rng(seed)))
        obs = self.sim.set_chart([note_tuple(n) for n in self.chart])
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
            pre, acc, post = self.sim.score_components()
            info.update(score_pre=pre, score_acc=acc, score_post=post)
            info.update(good_cuts=self.sim.good_cuts, bad_cuts=self.sim.bad_cuts,
                        misses=self.sim.misses, n_notes=self.sim.n_notes,
                        score=self.sim.score, bombs_hit=self.sim.bombs_hit)
        return obs, float(reward), bool(done), False, info
