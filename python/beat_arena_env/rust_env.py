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
    EVENT_REASONS, N_JOINTS, OBS_DIM,
)

from beat_arena_rs import RawBeatSim


def record_header(chart, walls, *, tick_hz, action_repeat, tau_scale, gears,
                  cfg=None, map=None, level=None, model_hash=None):
    """The replay JSONL header line's object (SPEC-v10 §6), shared by
    RustBeatEnv's per-tick recorder and the batched columnar writer
    (make_replays_batched.py). chart / walls are the sim's note_tuple /
    wall_tuple tuples; cfg the ChartConfig (default: RustBeatEnv's).
    map / level / model_hash are the additive fields that used to be
    stamped in afterwards (annotate_replay_audio.stamp: header.map =
    {hash, difficulty, song, audio_offset_s}; make_top10_replays: level
    identity + weights hash) — passed here they are written up front, in
    the same key order, so no rewrite pass is needed."""
    cfg = cfg if cfg is not None else ChartConfig()
    head = {
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
        # 6 = header "tick_hz" / "action_repeat" (exp/tick-rate,
        # 2026-09-29): one JSONL line per SIM tick at tick_hz (tick k
        # spans [k, k+1] / tick_hz s; `time` = k / tick_hz); absent =
        # 30 / 1, the only regime before. (Readers also accept an
        # "obs_window" flag, absent = false; this sim has no
        # obs-window variant, so it is never written.)
        "record_rev": 6,
        "tick_hz": int(tick_hz),
        "action_repeat": int(action_repeat),
        "tau_scale": [round(v, 6) for v in tau_scale],
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


def event_json(i, code, pts, pre, acc):
    """One sim event (note_idx, code, points, pre, acc) in the replay's
    per-tick "events" form."""
    return ({"miss": {"note": int(i)}} if code == 4 else
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
                     "acc": round(float(acc), 2)}})


# VecBeatSim recorder frame (take_record_env "frame", lanes per tick; the
# REC_* layout in rust/src/lib.rs) and the per-field rounding of the
# per-tick record: (first lane, end lane, decimals)
REC_LANES = 105
_REC_ROUND = ((0, 36, 5), (36, 42, 4), (42, 71, 4), (71, 83, 4),
              (83, 90, 6), (90, 96, 4), (96, 100, 4), (100, 104, 3),
              (104, 105, 4))


def write_replay_jsonl(path, header, rec, *, compact=False, gzip_level=None,
                       max_ticks=None):
    """Write a replay from a VecBeatSim recorder capture
    (take_record_env) — the per-tick record of RustBeatEnv._record_tick
    (same fields, order and rounding; values agree up to numpy-vs-Python
    decimal rounding of the last digit), built column-wise: one vectorized
    round per field, then a dict per tick. `header` is record_header()'s
    object (its tick_hz sets the `time` stamps). compact: no spaces after
    separators — the form publish_replays.py stores. gzip_level: write
    gzip (no name, mtime 0: byte-identical re-runs) instead of plain text.
    max_ticks: keep only the first N ticks (debug runs)."""
    import gzip
    lanes = int(rec["lanes"])
    assert lanes == REC_LANES, f"recorder frame has {lanes} lanes"
    T = len(rec["tick"])
    if max_ticks is not None:
        T = min(T, int(max_ticks))
    frame = np.asarray(rec["frame"], dtype=np.float64)[:T * lanes]
    frame = frame.reshape(T, lanes)
    for a, b, d in _REC_ROUND:
        frame[:, a:b] = np.round(frame[:, a:b], d)
    rows = frame.tolist()
    dt = 1.0 / int(header["header"]["tick_hz"])
    times = np.round(np.asarray(rec["time"][:T], dtype=np.float64) - dt,
                     5).tolist()
    ticks = np.asarray(rec["tick"][:T]).tolist()
    acts = np.asarray(rec["action"][:T * N_JOINTS]).reshape(
        T, N_JOINTS).tolist()
    coff = np.asarray(rec["contact_off"]).tolist()
    cpair = np.asarray(rec["contact_pair"]).tolist()
    cval = np.round(np.asarray(rec["contact_val"], dtype=np.float64)
                    .reshape(-1, 4), 4).tolist()
    eoff = np.asarray(rec["event_off"]).tolist()
    enote = np.asarray(rec["event_note"]).tolist()
    ecode = np.asarray(rec["event_code"]).tolist()
    evals = np.asarray(rec["event_val"], dtype=np.float64).reshape(
        -1, 3).tolist()
    enc = json.JSONEncoder(
        separators=(",", ":") if compact else (", ", ": ")).encode
    raw = open(path, "wb")
    f = (gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                       compresslevel=gzip_level)
         if gzip_level is not None else raw)
    try:
        out = [enc(header)]
        for k in range(T):
            r = rows[k]
            out.append(enc({
                "t": ticks[k], "time": times[k], "q": r[0:36],
                "twist": r[36:42], "omega": r[42:71],
                "sabers": [r[71:77], r[77:83]], "head": r[83:90],
                "action": acts[k], "reward": r[104],
                "contacts": [{"pair": cpair[m], "p": cval[m][:3],
                              "j": cval[m][3]}
                             for m in range(coff[k], coff[k + 1])],
                "feet": [r[90:93] + [0.0, 0.0, 0.0],
                         r[93:96] + [0.0, 0.0, 0.0]],
                "activation": r[96:100], "torque": r[100:104],
                "events": [event_json(enote[m], ecode[m], *evals[m])
                           for m in range(eoff[k], eoff[k + 1])]}))
            if len(out) >= 2048:
                f.write(("\n".join(out) + "\n").encode())
                out = []
        if out:
            f.write(("\n".join(out) + "\n").encode())
    finally:
        if f is not raw:
            f.close()
        raw.close()
    return T


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
        the rate goes into the replay header (record_rev 6)."""
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
    def set_record(self, path: str, **extra):
        """Record the NEXT episode (and later ones) to JSONL at path.
        extra: record_header's additive fields (map, level, model_hash),
        written into the header up front."""
        self._record_path = path
        self._record_extra = extra

    def _start_record(self):
        header = record_header(
            [note_tuple(n) for n in self.chart],
            [wall_tuple(w) for w in self.walls],
            tick_hz=self.tick_hz, action_repeat=self.action_repeat,
            tau_scale=self.tau_scale, gears=self.gears, cfg=self.cfg,
            **self._record_extra)
        self._record_lines = [json.dumps(header)]

    def _record_tick(self, action, reward, events):
        # SPEC-v10 §6 per-tick record: q36 (root pos + quat wxyz + 29
        # joints), root twist, omega29, per-hand FK sabers [w3 + b3],
        # contact events (pair id, world point, tick-summed impulse),
        # score events unchanged from v7+.
        # P6 ADDITIVE fields (version stays "v10"; readers must treat
        # them as optional): "feet" = per-foot [L, R] tick-summed ground
        # reaction impulses [jx, jy, jz, 0, 0, 0] (N·s — vertical force
        # ≈ jy/DT; lanes 3..5 were the retired welds' angular impulses
        # and stay 0 so the schema is unchanged). τ per joint is NOT logged: it is
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
            "time": round(self.sim.time - self.dt, 5),
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
            # replay fix (2026-08-17): the ground reaction per foot; it
            # has no angular part, so lanes 3..5 are 0 (schema unchanged)
            "feet": [[round(g[0], 4), round(g[1], 4), round(g[2], 4),
                      0.0, 0.0, 0.0]
                     for g in self.sim.ground_reaction()],
            "activation": [round(v, 4) for v in self.sim.last_activation()],
            # record_rev 4 (env v43): applied |tau| summed [total, spine,
            # arms, legs] in N.m — the energy-priced quantity (derivable
            # from "action" x TAU_MAX; logged for energy stats)
            "torque": [round(v, 3) for v in self.sim.last_torque()],
            "events": [event_json(i, code, pts, pre, acc)
                       for i, code, pts, pre, acc in events],
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
        acts = [int(x) for x in a]
        # action_repeat: hold the action for N sim ticks (stops early on
        # done); reward summed, events concatenated in tick order
        all_events, reward = [], 0.0
        for _ in range(self.action_repeat):
            obs, r, done, events = self.sim.step(acts)
            reward += r
            all_events.extend(events)
            if self._record_lines is not None:
                self._record_tick(a, r, events)
                if done:
                    with open(self._record_path, "w") as f:
                        f.write("\n".join(self._record_lines) + "\n")
                    self._record_lines = None
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
