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

from .chart import ChartConfig, generate_chart, note_tuple, wall_tuple
from .constants import (TAU_FEAT, HANDLE_LEN, 
    EVENT_REASONS, N_JOINTS, OBS_DIM, note_center,
)

from beat_arena_rs import RawBeatSim


class RustBeatEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, colors="both", chart_len_s=20.0,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed_min=2.0, note_speed_max=8.0,
                 note_prob=0.25, extra_prob=0.125, dot_prob=0.1,
                 bomb_prob=0.125,
                 external_chart=None, wide_acc=False, gears=3, seed=0,
                 foot_contact=False, fall_const=5.0,
                 external_walls=None):
        """external_chart: a fixed list[Note] (e.g. a parsed BeatSaver map)
        played on EVERY reset instead of procedural generation.
        external_walls: list[Wall] riding along (env v17, spec §12) —
        real maps only; the procedural generator emits no walls.

        foot_contact: env v11 foot_mode (unilateral-contact spec §1) —
        False = the v10 foot welds (DEFAULT, physics unchanged), True =
        unilateral frictional ground contact on the 8 sole corners, with
        the fall termination at root y < 0.55 m."""
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
        self.walls = list(external_walls) if external_walls else []
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
        self.sim = RawBeatSim(max_steps=max_steps,
                              wide_acc=wide_acc, gears=gears,
                              foot_contact=foot_contact,
                              fall_const=fall_const)
        self.wide_acc = wide_acc
        self.gears = gears
        self.foot_contact = bool(foot_contact)
        # env v46: explicit per-axis torque-limit scales (all 1.0 = the
        # v45 body); recorded in the replay header (record_rev 5)
        self.tau_scale = [1.0] * TAU_FEAT
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        # env v10: 29 gear-id heads in SPEC-v10 §1.4 joint order —
        # spine(3), arm_l(7), arm_r(7), leg_l(6), leg_r(6); id 0 coast,
        # 1/2 = +/-full, finer +/- gear pairs append (constants.gear_table)
        self.action_space = gym.spaces.MultiDiscrete([gears] * N_JOINTS)

    def set_tau_scale(self, s16):
        """env v46: fix the 16 per-axis torque-limit scales (spine 3, arm 7
        shared L/R, leg 6 shared L/R) for every later episode."""
        s = [float(x) for x in s16]
        assert len(s) == TAU_FEAT, "tau scale must have 16 axis values"
        self.sim.set_tau_scale(s)
        self.tau_scale = s

    # -- recording (JSONL v10 — SPEC-v10 §6, same format the viewer reads) ----
    def set_record(self, path: str):
        """Record the NEXT episode (and later ones) to JSONL at path."""
        self._record_path = path

    def _start_record(self):
        header = {"header": {
            "version": "v10",     # env v10 body: viewers reject others
            # RECORD-SCHEMA revision, additive-only (SPEC-v10 §6).
            # "version" names the BODY generation and is hard-gated
            # ("v10" exactly) by viz/app/src/lib/replay.js and
            # convert_scoresaber.py, so bumping it to advertise a new
            # per-tick FIELD would retire every reader for no reason —
            # schema growth is versioned here instead. Absent/1 = the
            # pre-2026-08-13 record (no "head"); 2 = per-tick "head"
            # (FK head pose) present on every tick; 3 = per-tick
            # "activation" [total, spine, arms, legs] replaces the P6
            # "effort" work reading (2026-09-13); 4 = per-tick "torque"
            # [total, spine, arms, legs] applied |tau| in N.m (env v43
            # energy-price base, 2026-09-14). Readers must treat a
            # HIGHER rev as "fields I do not know about are present",
            # never as an error. 5 = header "tau_scale" (env v46, the 16
            # per-axis torque-limit scales this episode played with;
            # 2026-09-16) — a HEADER field, per-tick record unchanged.
            "record_rev": 5,
            "tau_scale": [round(v, 6) for v in self.tau_scale],
            # env v15 (contact spec §10), ADDITIVE: the note track is
            # head-anchored — viewers must add head_z - PLANE_Z0 to
            # every note's world z. Absent on pre-v15 recordings
            # (fixed track); readers treat absence as fixed.
            "track_anchor": "head",
            "config": {
                "chart_len_s": self.cfg.chart_len_s,
                "colors": self.cfg.colors,
                "note_speed_min": self.cfg.note_speed_min,
                "note_speed_max": self.cfg.note_speed_max,
                "note_prob": self.cfg.note_prob,
                "extra_prob": self.cfg.extra_prob,
                "dot_prob": self.cfg.dot_prob,
                "bomb_prob": self.cfg.bomb_prob,
                "gears": self.gears,       # env v8 action-table size
                # env v42 hilt grip: blade base sits HANDLE_LEN in front
                # of the wrist along the blade dir; absent = 0 (pre-v42)
                "handle_len": HANDLE_LEN,
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
            # env v17 (spec §12): obstacles — world-space AABBs on the
            # same head-anchored track; front face runs 0.25 m behind
            # the note plane (the sim applies the offset, the header
            # carries the parser's finished geometry)
            "walls": [{"t": w.start_time, "cx": round(w.cx, 4),
                       "cy": round(w.cy, 4), "w": round(w.width, 4),
                       "h": round(w.height, 4), "l": round(w.length, 4),
                       "speed": round(w.speed, 4)}
                      for w in self.walls],
        }}
        self._record_lines = [json.dumps(header)]

    def _record_tick(self, action, reward, events):
        # SPEC-v10 §6 per-tick record: q36 (root pos + quat wxyz + 29
        # joints), root twist, omega29, per-hand FK sabers [w3 + b3],
        # contact events (pair id, world point, tick-summed impulse),
        # score events unchanged from v7+.
        # P6 ADDITIVE fields (version stays "v10"; readers must treat
        # them as optional): "feet" = per-foot [L, R] tick-summed weld
        # impulses [jx, jy, jz, jax, jay, jaz] (N·s; ground reaction —
        # vertical force ≈ jy/DT). τ per joint is NOT logged: it is
        # exactly gear_table(gears)[action[j]] × TAU_MAX[j] (derivable).
        # record_rev 3 (2026-09-13): the P6 "effort" (positive work, J)
        # field is GONE; "activation" = this tick's commanded |gear
        # fraction| summed [total, spine, arms, legs] (the retired v23
        # --iso-coef base — derivable from "action" too; logged for the viewer
        # meter and energy-conservation stats).
        # record_rev 2 ADDITIVE field: "head" = the FK head frame's
        # world pose [px, py, pz, qw, qx, qy, qz] (quaternion scalar-
        # first, like q's root block). Derivable from q by a tree walk
        # — logged for the same reason "sabers" is (§6 "viewer
        # convenience"), and because convert_scoresaber.py needs the
        # head ROTATION: watch.scoresaber.com's first-person camera is
        # driven straight off the .dat's head frame.
        self._record_lines.append(json.dumps({
            "t": self.sim.tick - 1,
            "time": round(self.sim.time - 1.0 / 30.0, 5),
            "q": [round(v, 5) for v in self.sim.qpos()],
            "twist": [round(v, 4) for v in self.sim.twist()],
            "omega": [round(v, 4) for v in self.sim.omega()],
            "sabers": [[round(v, 4) for v in s] for s in self.sim.sabers()],
            # 6 decimals on the quaternion too: it is REnormalized
            # downstream, and 1e-6 keeps the residual well under the
            # 1e-4 unit-norm tolerance the .dat round-trip asserts
            "head": [round(v, 6) for v in self.sim.head_pose()],
            "action": [int(a) for a in action],
            "reward": round(float(reward), 4),
            "contacts": [{"pair": int(p), "p": [round(x, 4), round(y, 4),
                                                round(z, 4)],
                          "j": round(imp, 4)}
                         for p, x, y, z, imp in self.sim.contacts()],
            # replay fix (2026-08-17): "feet" was wired to the retired
            # welds' impulses and read all-zero in contact mode. Sum
            # both sources — exactly one is nonzero per mode (weld rows
            # don't exist in contact mode and vice versa); the ground
            # reaction has no angular part, so lanes 3..5 keep the weld
            # value (0 in contact mode) and the schema is unchanged.
            "feet": [[round(w + (g[k] if k < 3 else 0.0), 4)
                      for k, w in enumerate(wf)]
                     for wf, g in zip(self.sim.weld_impulses(),
                                      self.sim.ground_reaction())],
            "activation": [round(v, 4) for v in self.sim.last_activation()],
            # record_rev 4 (env v43): applied |tau| summed [total, spine,
            # arms, legs] in N.m — the energy-priced quantity (derivable
            # from "action" x TAU_MAX; logged for energy stats)
            "torque": [round(v, 3) for v in self.sim.last_torque()],
            "events": [
                {"miss": {"note": int(i)}} if code == 4 else
                {"post": {"note": int(i), "points": round(float(pts), 2)}}
                if code == 5 else
                {"bomb": {"note": int(i)}} if code == 6 else
                # env v17 (spec §12b): wall drain tick / wall-entry
                # combo break — no note id (the sim sends u32::MAX)
                {"wall": {}} if code == 7 else
                {"wall_enter": {}} if code == 8 else
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
        obs = self.sim.set_chart([note_tuple(n) for n in self.chart],
                                 [wall_tuple(w) for w in self.walls])
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
                        score=self.sim.score, bombs_hit=self.sim.bombs_hit,
                        # contact spec §4: 0 or 1 — a fall ends the
                        # episode. Reported, never hidden; always 0 in
                        # weld mode.
                        falls=self.sim.falls,
                        # env v17: head-in-wall ticks + wall entries
                        # (= combo breaks, spec §12b)
                        walls_in=self.sim.walls_in,
                        wall_enters=self.sim.wall_enters)
        return obs, float(reward), bool(done), False, info
