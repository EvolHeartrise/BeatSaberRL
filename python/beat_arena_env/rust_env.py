"""Gymnasium env over the Rust sim core — THE single-env implementation.

Env v10 (P3): the body is the SPEC-v10 dynamic humanoid — 29 gear-id
action heads (SPEC-v10 §1.4 joint order), obs 340 (§5), replays in
format 2 (replay_format.py: q36 + root twist + omega29 + contacts with
pair ids + score events). Chart generation stays in Python (once per
episode); physics, cut logic, scoring, and obs building live in
rust/src/lib.rs. Per-step cut/miss events come back from Rust and are
exposed in `info["events"]` as (note_idx, reason, points) tuples.

Build the extension once with:
  VIRTUAL_ENV=<venv> <venv>/bin/maturin develop --release   (from rust/)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .chart import ChartConfig, generate_chart, note_tuple, wall_tuple
from .constants import (TAU_FEAT, HANDLE_LEN,
    EVENT_REASONS, N_JOINTS, OBS_DIM, joint_gear_counts,
)

from . import replay_format
from beat_arena_rs import RawBeatSim


def record_header(chart, walls, *, tick_hz, action_repeat, tau_scale, gears,
                  cfg=None, map=None, level=None, model_hash=None):
    """The replay header object (SPEC-v10 §6; replay_format.py), shared
    by RustBeatEnv's recorder and the batched writer
    (make_replays_batched.py). chart / walls are the sim's note_tuple /
    wall_tuple tuples; cfg the ChartConfig (default: RustBeatEnv's);
    tau_scale the 16 per-axis torque-limit scales the sim plays with.
    map ({hash, difficulty, song, audio_offset_s}: the viewer fetches
    the song's audio with it), level (level identity) and model_hash
    (weights hash) are optional."""
    cfg = cfg if cfg is not None else ChartConfig()
    head = {
        "version": "v10",     # env v10 body: viewers reject others
        # replay file format (replay_format.py): 2 = the compact binary
        # replay, the only one (the gzip-JSONL format 1 was retired
        # 2026-10-02)
        "format": replay_format.FORMAT,
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
        # 6 = header "tick_hz" / "action_repeat" (exp/tick-rate,
        # 2026-09-29): one record per SIM tick at tick_hz (tick k
        # spans [k, k+1] / tick_hz s; `time` = k / tick_hz); absent =
        # 30 / 1, the only regime before. (Readers also accept an
        # "obs_window" flag, absent = false; this sim has no
        # obs-window variant, so it is never written.)
        "record_rev": 6,
        "tick_hz": int(tick_hz),
        "action_repeat": int(action_repeat),
        "tau_scale": [round(v, 6) for v in tau_scale],
        # format 2: the per-joint effective torque limits and signed gear
        # fractions this sim applied — the decoders derive per-tick
        # activation / torque from the actions with them
        "body": replay_format.body_tables(gears, tau_scale),
        # env v15 (contact spec §10), ADDITIVE: the note track is
        # head-anchored — viewers must add head_z - PLANE_Z0 to
        # every note's world z. Absent on pre-v15 recordings
        # (fixed track); readers treat absence as fixed.
        "track_anchor": "head",
        "config": {
            "chart_len_s": cfg.chart_len_s,
            "colors": cfg.colors,
            "note_speed_min": cfg.note_speed_min,
            "note_speed_max": cfg.note_speed_max,
            "note_prob": cfg.note_prob,
            "extra_prob": cfg.extra_prob,
            "dot_prob": cfg.dot_prob,
            "bomb_prob": cfg.bomb_prob,
            "gears": int(gears),       # env v8 action-table size
            # env v42 hilt grip: blade base sits HANDLE_LEN in front
            # of the wrist along the blade dir; absent = 0 (pre-v42)
            "handle_len": HANDLE_LEN,
        },
        # explicit world positions (meters), object kinds and arc
        # grants; the viewer derives everything from these
        "chart": [{"t": t, "x": round(x, 4), "y": round(y, 4),
                   "color": color, "angle": round(angle, 5),
                   "speed": round(speed, 4), "any": any_dir,
                   "kind": kind, "fpre": fpre, "fpost": fpost}
                  for t, x, y, color, angle, speed, any_dir, kind, fpre,
                  fpost in chart],
        # env v17 (spec §12): obstacles — world-space AABBs on the
        # same head-anchored track; front face runs 0.25 m behind
        # the note plane (the sim applies the offset, the header
        # carries the parser's finished geometry)
        "walls": [{"t": t0, "cx": round(cx, 4), "cy": round(cy, 4),
                   "w": round(w, 4), "h": round(h, 4), "l": round(length, 4),
                   "speed": round(speed, 4)}
                  for t0, cx, cy, w, h, length, speed in walls],
    }
    if map is not None:
        head["map"] = {"hash": map["hash"], "difficulty": map["difficulty"],
                       "song": map["song"],
                       "audio_offset_s": float(map["audio_offset_s"])}
    if level is not None:
        head["level"] = level
    if model_hash is not None:
        head["model_hash"] = model_hash
    return {"header": head}


class RustBeatEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, colors="both", chart_len_s=20.0,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed_min=2.0, note_speed_max=8.0,
                 note_prob=0.25, extra_prob=0.125, dot_prob=0.1,
                 bomb_prob=0.125,
                 external_chart=None, gears=3, seed=0, *,
                 fall_const=5.0,
                 external_walls=None, sim_dtype="f64",
                 tick_hz=30, action_repeat=1):
        """external_chart: a fixed list[Note] (e.g. a parsed BeatSaver map)
        played on EVERY reset instead of procedural generation.
        external_walls: list[Wall] riding along (env v17, spec §12) —
        real maps only; the procedural generator emits no walls.

        The body stands on unilateral frictional ground contact at the 8
        sole corners (env v11), with the fall (root y below FALL_ROOT_Y,
        rust/src/lib.rs) and chest-tilt terminations. The retired v10
        foot-weld harness and its `foot_contact=` keyword were removed
        2026-09-27 (issue #2); passing it is a TypeError.

        exp/tick-rate (2026-09-29): tick_hz = the sim's control rate
        (30/60/120/240; physics substep fixed at 1/240 s, per-tick rules
        rescaled to real time), action_repeat = sim ticks each step()
        holds its action for (obs/done from the last of them, events
        concatenated in tick order — the bench's step_into(repeat=N)).
        Defaults are the 30 Hz regime, bit-identical to before. Every SIM
        tick is recorded (the replay keeps the sim's own frame rate) and
        the rate goes into the replay header."""
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
        self._record_extra = {}
        self._rec = None
        self.chart = []
        self._external = list(external_chart) if external_chart else None
        self.walls = list(external_walls) if external_walls else []
        # +90 ticks: the episode tail must cover the miss-boundary trip
        # (live head plane, env v10) past the last note
        if self._external:
            max_steps = int((self._external[-1].hit_time + 0.5) * 30) + 90
        else:
            max_steps = int(chart_len_s * 30) + 90
        assert gears in (3, 5, 7), f"gears must be 3, 5, or 7, got {gears}"
        # sim_dtype (2026-09-26): "f64" = the reference build every test
        # pins; "f32" = beat_arena_rs32, the bench's (and training's) regime
        assert sim_dtype in ("f64", "f32"), sim_dtype
        if sim_dtype == "f32":
            from beat_arena_rs32 import RawBeatSim as Sim
        else:
            Sim = RawBeatSim
        # max_steps is in 30 Hz ticks; the sim scales it to its rate
        assert int(action_repeat) >= 1, action_repeat
        self.sim = Sim(max_steps=max_steps, gears=gears,
                       fall_const=fall_const, tick_hz=int(tick_hz))
        self.tick_hz = int(self.sim.tick_hz)
        # python double 1/tick_hz (== the pre-tick-rate 1.0 / 30.0 at the
        # default, so 30 Hz `time` stamps stay byte-identical)
        self.dt = 1.0 / self.tick_hz
        self.action_repeat = int(action_repeat)
        self.gears = gears
        # per-joint action-id counts (constants.joint_gear_counts; the
        # Rust sim's joint_gear_rows mirror decodes them)
        self.joint_gears = joint_gear_counts(gears)
        # env v46: explicit per-axis torque-limit scales (all 1.0 = the
        # v45 body); recorded in the replay header (record_rev 5)
        self.tau_scale = [1.0] * TAU_FEAT
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        # env v10: 29 gear-id heads in SPEC-v10 §1.4 joint order —
        # spine(3), arm_l(7), arm_r(7), leg_l(6), leg_r(6); id 0 coast,
        # 1/2 = +/-full, finer +/- gear pairs append (constants.gear_table)
        self.action_space = gym.spaces.MultiDiscrete(list(self.joint_gears))

    def set_tau_scale(self, s16):
        """env v46: fix the 16 per-axis torque-limit scales (spine 3, arm 7
        shared L/R, leg 6 shared L/R) for every later episode."""
        s = [float(x) for x in s16]
        assert len(s) == TAU_FEAT, "tau scale must have 16 axis values"
        self.sim.set_tau_scale(s)
        self.tau_scale = s

    # -- recording (replay format 2, replay_format.py) -----------------------
    def set_record(self, path: str, **extra):
        """Record the NEXT episode (and later ones) to a format-2 replay at
        path (conventionally <name>.rpl.gz). extra: record_header's
        optional fields (map, level, model_hash)."""
        self._record_path = path
        self._record_extra = extra

    def _start_record(self):
        self._rec_header = record_header(
            [note_tuple(n) for n in self.chart],
            [wall_tuple(w) for w in self.walls],
            tick_hz=self.tick_hz, action_repeat=self.action_repeat,
            tau_scale=self.tau_scale, gears=self.gears, cfg=self.cfg,
            **self._record_extra)
        self._rec = {"frame": [], "action": [], "contact_off": [0],
                     "contact_pair": [], "contact_val": [],
                     "event_off": [0], "event_note": [], "event_code": [],
                     "event_val": []}

    def _record_tick(self, action, reward, events):
        # one recorder frame per SIM tick: the lanes of VecBeatSim's Rust
        # recorder (replay_format.REC_LANES layout), from the same getters
        r = self._rec
        s = self.sim
        sab = s.sabers()
        gr = s.ground_reaction()
        r["frame"].append([*s.qpos(), *s.twist(), *s.omega(), *sab[0],
                           *sab[1], *s.head_pose(), *gr[0], *gr[1],
                           *s.last_activation(), *s.last_torque(),
                           float(reward)])
        r["action"].append([int(a) for a in action])
        for p, x, y, z, imp in s.contacts():
            r["contact_pair"].append(p)
            r["contact_val"].append((x, y, z, imp))
        r["contact_off"].append(len(r["contact_pair"]))
        for i, code, pts, pre, acc in events:
            r["event_note"].append(i)
            r["event_code"].append(code)
            r["event_val"].append((pts, pre, acc))
        r["event_off"].append(len(r["event_note"]))

    def _finish_record(self):
        r, self._rec = self._rec, None
        T = len(r["frame"])
        rec = {"lanes": replay_format.REC_LANES, "tick": np.arange(T),
               "frame": np.asarray(r["frame"], dtype=np.float64).ravel(),
               "action": np.asarray(r["action"], dtype=np.uint8).ravel(),
               "contact_val": np.asarray(r["contact_val"],
                                         dtype=np.float64).ravel(),
               "event_val": np.asarray(r["event_val"],
                                       dtype=np.float64).ravel()}
        for k in ("contact_off", "contact_pair", "event_off", "event_note",
                  "event_code"):
            rec[k] = np.asarray(r[k], dtype=np.int64)
        replay_format.write_replay(self._record_path, self._rec_header, rec)

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
        acts = [int(x) for x in a]
        # action_repeat: hold the action for N sim ticks (stops early on
        # done); reward summed, events concatenated in tick order
        all_events, reward = [], 0.0
        for _ in range(self.action_repeat):
            obs, r, done, events = self.sim.step(acts)
            reward += r
            all_events.extend(events)
            if self._rec is not None:
                self._record_tick(a, r, events)
                if done:
                    self._finish_record()
            if done:
                break
        info = {"events": [(int(i), EVENT_REASONS[c], float(p))
                           for i, c, p, _pre, _acc in all_events]}
        if done:
            pre, acc, post = self.sim.score_components()
            info.update(score_pre=pre, score_acc=acc, score_post=post)
            info.update(good_cuts=self.sim.good_cuts, bad_cuts=self.sim.bad_cuts,
                        misses=self.sim.misses, n_notes=self.sim.n_notes,
                        score=self.sim.score, bombs_hit=self.sim.bombs_hit,
                        # contact spec §4: 0 or 1 — a fall ends the
                        # episode. Reported, never hidden.
                        falls=self.sim.falls,
                        # env v17: head-in-wall ticks + wall entries
                        # (= combo breaks, spec §12b)
                        walls_in=self.sim.walls_in,
                        wall_enters=self.sim.wall_enters)
        return obs, float(reward), bool(done), False, info
