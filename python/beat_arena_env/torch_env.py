"""GPU-resident vectorized Beat Arena sim (torch tensors, one row per env).

A semantics-exact port of rust/src/lib.rs step_inner + build_obs (env v6,
kinematic arm chains) — the Rust crate remains THE reference implementation; this
port exists so rollouts can run wide (N in the thousands) on the GPU with
observations born on-device. Parity with Rust is pinned by
tests/test_torch_env.py: exact event/reward/obs agreement in float64,
bounded drift in float32.

Design notes (why it looks the way it does):
- All charts are padded to a common length C; a `valid` mask marks real notes.
  Consumed/padded notes never match any condition, so every rule is a pure
  masked tensor op — no per-env branching anywhere.
- The swept-blade cut test loops the 8 blade samples STATICALLY (bounded
  memory, no data-dependent shapes, no host syncs).
- Follow-through trackers are fixed ring slots per saber. Trackers spawned
  by the same saber in the same tick share identical (arc, age) forever, so
  they aggregate into ONE slot with a note COUNT — at most one spawn per
  saber per tick. A surviving tracker lives <= FOLLOW_TIMEOUT ticks and the
  ring has K = FOLLOW_TIMEOUT + 1 slots, so a slot being rewritten is
  provably inactive: the ring is EXACT, not an approximation.
- Auto-reset draws charts from a pregenerated pool; episode returns go to an
  on-device ring buffer fetched with pop_stats() once per PPO update: the
  step loop performs ZERO host<->device synchronizations.

Limits vs BeatVecEnv: no curriculum sampler, no per-step info dicts, no
event stream (rewards/counters/score only).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .chart import ChartConfig, generate_chart
from .constants import OBS_DIM

# Physics constants — mirror rust/src/lib.rs (SPEC.md is the contract).
DT = 1.0 / 30.0
DRAG = 0.02
V_NORM = 6.0
V_MIN_CUT = 1.0
CUT_RADIUS = 0.28
COS_TOL = 0.5
CELL = 0.5
REWARD_BAD = -0.5
REWARD_MISS = -0.5
SABER_LEN = 1.0
# env v6 upper body (see lib.rs / constants.py)
SHOULDER = ((-0.2, 0.45, -0.35), (0.2, 0.45, -0.35))
L_UPPER = 0.35
L_FORE = 0.30
N_JOINTS = 6
ALPHA_ARM, OMEGA_ARM = 80.0, 8.0
ALPHA_WRIST, OMEGA_WRIST = 200.0, 20.0
JOINT_ALPHA = (ALPHA_ARM,) * 4 + (ALPHA_WRIST,) * 2
JOINT_OMEGA = (OMEGA_ARM,) * 4 + (OMEGA_WRIST,) * 2
PITCH_LIM = 1.5707963267948966
YAW_LO = (-1.7453292519943295, -0.7853981633974483)
YAW_HI = (0.7853981633974483, 1.7453292519943295)
TWIST_LIM = 1.5707963267948966
ELBOW_HI = 2.6179938779914944
# env v5.2 forearm-anchored joints (mirror lib.rs): neutral blade points
# at the notes; roll [-90, 190] deg left / [-190, 90] right; bend +-95
ROLL_LO = (-1.5707963267948966, -3.3161255787892263)
ROLL_HI = (3.3161255787892263, 1.5707963267948966)
BEND_LIM = 1.6580627893946132
N_BLADE_SAMPLES = 8
REACH_Z = 1.5       # max wrist z (0.3) + blade (1) + HALF_BLOCK
HALF_BLOCK = 0.2    # env v5.3: contact = swept blade vs the ROTATED cube
Z_MISS = 0.6
PRE_FULL = 1.7453292519943295   # 100 deg
POST_FULL = 1.0471975511965976  # 60 deg
PTS_PRE, PTS_POST, PTS_ACC = 70.0, 30.0, 15.0
SCORE_MAX = 115.0
SWING_BREAK = 0.005
FOLLOW_TIMEOUT = 15
N_TRACKERS = FOLLOW_TIMEOUT + 1  # ring slots; see module docstring
X_NORM = 1.0
Y_NORM = 0.75
Z_SCALE = 8.0
Z_CLIP = (-1.0, 12.0)
N_NOTE_SLOTS = 12
NOTE_FEAT = 9
SABER_FEAT = 23
OFF_NOTES = 2 * SABER_FEAT                             # 46
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT   # 154


def _charts_to_tensors(charts, device, dtype):
    """Pad a list[list[Note]] into per-field (P, C) tensors."""
    P = len(charts)
    C = max(1, max(len(ch) for ch in charts))
    hit = np.zeros((P, C)); col = np.zeros((P, C)); row = np.zeros((P, C))
    color = np.zeros((P, C)); angle = np.zeros((P, C)); speed = np.ones((P, C))
    any_dir = np.zeros((P, C), dtype=bool); valid = np.zeros((P, C), dtype=bool)
    for p, ch in enumerate(charts):
        for j, n in enumerate(ch):
            hit[p, j] = n.hit_time; col[p, j] = n.col; row[p, j] = n.row
            color[p, j] = n.color; angle[p, j] = n.angle; speed[p, j] = n.speed
            any_dir[p, j] = n.any_dir; valid[p, j] = True
    t = lambda a, dt=dtype: torch.as_tensor(a, device=device, dtype=dt)
    return {
        "hit": t(hit), "speed": t(speed),
        "center": torch.stack([(t(col) - 1.5) * CELL, (t(row) - 1.0) * CELL], -1),
        "is_right": t(color, torch.bool),
        "dvec": torch.where(
            t(any_dir, torch.bool).unsqueeze(-1),
            torch.zeros((P, C, 2), device=device, dtype=dtype),
            torch.stack([t(np.cos(angle)), t(np.sin(angle))], -1)),
        # box rotation about z (env v5.3): rot = angle - 270 deg, derived
        # exactly from the angle's own sin/cos (cos_rot = -sin a,
        # sin_rot = cos a); dot notes stay axis-aligned
        "rot": torch.where(
            t(any_dir, torch.bool).unsqueeze(-1),
            torch.stack([torch.ones((P, C), device=device, dtype=dtype),
                         torch.zeros((P, C), device=device, dtype=dtype)], -1),
            torch.stack([t(-np.sin(angle)), t(np.cos(angle))], -1)),
        "any_dir": t(any_dir, torch.bool), "valid": t(valid, torch.bool),
        "n_notes": t(valid.sum(1), torch.long),
        "last_hit": t(np.where(valid.any(1), (hit * valid).max(1), 0.0)),
    }


class TorchBeatVecEnv:
    """BeatVecEnv-alike over pure tensor state. step() takes an (N, 12)
    long action tensor (per hand: pitch, yaw, twist, elbow, roll, bend;
    left first) and returns (obs, reward, done, []) — all tensors on
    `device`, infos always empty (use pop_stats() once per update)."""

    def __init__(self, num_envs, note_rate=1.0, colors="right", dirs=(6,),
                 continuous_dirs=False, chart_len_s=20.0, min_hand_gap_s=0.5,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed=4.0, note_speed_max=None, seed=0,
                 charts=None, chart_pool=1024, max_steps=None,
                 device="cuda", dtype=torch.float32, auto_reset=True):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.auto_reset = auto_reset  # False: terminal envs freeze (tests)
        self._test_charts = charts is not None
        if charts is None:
            cfg = ChartConfig(
                note_rate=note_rate, colors=colors, dirs=tuple(dirs),
                continuous_dirs=continuous_dirs, chart_len_s=chart_len_s,
                min_hand_gap_s=min_hand_gap_s, cols=tuple(cols),
                rows=tuple(rows), note_speed=note_speed,
                note_speed_max=note_speed_max)
            rng = np.random.default_rng(seed)
            charts = [generate_chart(cfg, rng) for _ in range(chart_pool)]
        self.pool = _charts_to_tensors(charts, self.device, dtype)
        self.pool_size, self.C = self.pool["hit"].shape
        if max_steps is None:
            max_steps = int(chart_len_s * 30) + 30
        self.max_steps = max_steps
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        self._episode = 0  # test-mode sequential chart assignment counter

        N, C, K = num_envs, self.C, N_TRACKERS
        zeros = lambda *s, dt=dtype: torch.zeros(*s, device=self.device, dtype=dt)
        # arm chains (env v6): 6 joints per hand + FK caches
        self.q = zeros(N, 2, N_JOINTS); self.om = zeros(N, 2, N_JOINTS)
        self.wrist = zeros(N, 2, 3); self.blade = zeros(N, 2, 3)
        self.wrist_vel = zeros(N, 2, 3)
        # swing chain
        self.swing_acc = zeros(N, 2)
        self.axis = zeros(N, 2, 3)
        self.axis_valid = zeros(N, 2, dt=torch.bool)
        # follow-through tracker ring (aggregated: `cnt` notes per slot)
        self.tr_acc = zeros(N, 2, K); self.tr_age = zeros(N, 2, K, dt=torch.long)
        self.tr_cnt = zeros(N, 2, K, dt=torch.long)
        self.tr_active = zeros(N, 2, K, dt=torch.bool)
        self.tr_ptr = zeros(N, 2, dt=torch.long)
        self.tick = zeros(N, dt=torch.long)
        self.status = zeros(N, C, dt=torch.uint8)  # 0 pend 1 good 2 bad 3 miss
        self.ep_return = zeros(N)
        self.good = zeros(N, dt=torch.long); self.bad = zeros(N, dt=torch.long)
        self.miss = zeros(N, dt=torch.long)
        self.score = zeros(N)
        # per-env chart fields (rows copied from the pool at reset)
        for k in ("hit", "speed", "valid", "is_right", "any_dir"):
            setattr(self, "c_" + k, zeros(N, C, dt=self.pool[k].dtype))
        self.c_center = zeros(N, C, 2); self.c_dvec = zeros(N, C, 2)
        self.c_rot = zeros(N, C, 2)
        self.c_last_hit = zeros(N); self.c_n_notes = zeros(N, dt=torch.long)
        # per-joint action table: action a in 0..2 -> sign {0, +1, -1},
        # scaled by the joint's alpha
        self._sgn = torch.tensor([0.0, 1.0, -1.0],
                                 device=self.device, dtype=dtype)
        self._alpha = torch.tensor(JOINT_ALPHA, device=self.device, dtype=dtype)
        self._omega = torch.tensor(JOINT_OMEGA, device=self.device, dtype=dtype)
        # per-hand per-joint limits, broadcast over (N, 2, 6)
        self._q_lo = torch.tensor(
            [[-PITCH_LIM, YAW_LO[h], -TWIST_LIM, 0.0, ROLL_LO[h], -BEND_LIM]
             for h in (0, 1)], device=self.device, dtype=dtype)
        self._q_hi = torch.tensor(
            [[PITCH_LIM, YAW_HI[h], TWIST_LIM, ELBOW_HI, ROLL_HI[h], BEND_LIM]
             for h in (0, 1)], device=self.device, dtype=dtype)
        self._shoulder = torch.tensor(SHOULDER, device=self.device, dtype=dtype)
        # episode-return ring buffer (device-side; slot `cap` is the trash row)
        self._cap = 4096
        self._ret_buf = zeros(self._cap + 1)
        self._ret_total = zeros((), dt=torch.long)
        self._popped = 0
        self._obs = zeros(N, OBS_DIM, dt=torch.float32)

    # -- forward kinematics (mirrors lib.rs fk(), same expression tree) -------
    def _fk(self):
        """Refresh the (N, 2, 3) wrist + blade caches from self.q."""
        q = self.q
        sp, cp = q[..., 0].sin(), q[..., 0].cos()
        sy, cy = q[..., 1].sin(), q[..., 1].cos()
        st, ct = q[..., 2].sin(), q[..., 2].cos()
        se, ce = q[..., 3].sin(), q[..., 3].cos()
        m00 = cy * ct - sy * sp * st
        m01 = -cy * st - sy * sp * ct
        m02 = sy * cp
        m10 = cp * st
        m11 = cp * ct
        m12 = sp
        m20 = -sy * ct - cy * sp * st
        m21 = sy * st - cy * sp * ct
        m22 = cy * cp
        f0 = m01 * se + m02 * ce
        f1 = m11 * se + m12 * ce
        f2 = m21 * se + m22 * ce
        sh = self._shoulder  # (2, 3), broadcasts over N
        self.wrist = torch.stack([
            sh[:, 0] + L_UPPER * m02 + L_FORE * f0,
            sh[:, 1] + L_UPPER * m12 + L_FORE * f1,
            sh[:, 2] + L_UPPER * m22 + L_FORE * f2], -1)
        sr, cr = q[..., 4].sin(), q[..., 4].cos()
        sb, cb = q[..., 5].sin(), q[..., 5].cos()
        d0, d1, d2 = -sr * sb, cr * sb, cb
        self.blade = torch.stack([
            m00 * d0 + (m01 * ce - m02 * se) * d1 + f0 * d2,
            m10 * d0 + (m11 * ce - m12 * se) * d1 + f1 * d2,
            m20 * d0 + (m21 * ce - m22 * se) * d1 + f2 * d2], -1)

    # -- resets ---------------------------------------------------------------
    # All resets are full-size torch.where merges (never boolean fancy
    # indexing): data-dependent shapes would force a host<->device sync on
    # every step, which is exactly what this env exists to avoid.
    def _reset_envs(self, mask):
        N = self.num_envs
        if self._test_charts:  # deterministic: env i -> pool[(i + ep*N) % P]
            idx = (torch.arange(N, device=self.device)
                   + self._episode * N) % self.pool_size
        else:
            idx = torch.randint(self.pool_size, (N,), device=self.device,
                                generator=self._gen)
        m1 = mask.unsqueeze(-1)
        m2 = mask.unsqueeze(-1).unsqueeze(-1)
        for k in ("hit", "speed", "valid", "is_right", "any_dir"):
            dst = getattr(self, "c_" + k)
            setattr(self, "c_" + k, torch.where(m1, self.pool[k][idx], dst))
        self.c_center = torch.where(m2, self.pool["center"][idx], self.c_center)
        self.c_dvec = torch.where(m2, self.pool["dvec"][idx], self.c_dvec)
        self.c_rot = torch.where(m2, self.pool["rot"][idx], self.c_rot)
        self.c_last_hit = torch.where(mask, self.pool["last_hit"][idx],
                                      self.c_last_hit)
        self.c_n_notes = torch.where(mask, self.pool["n_notes"][idx],
                                     self.c_n_notes)
        # start pose: all joints zero (arm straight forward, blade its
        # extension), at rest — refresh the FK caches for the reset rows
        self.q = torch.where(m2, torch.zeros_like(self.q), self.q)
        self.om = torch.where(m2, torch.zeros_like(self.om), self.om)
        self.wrist_vel = torch.where(
            m2, torch.zeros_like(self.wrist_vel), self.wrist_vel)
        self._fk()
        self.swing_acc = torch.where(
            m1, torch.zeros_like(self.swing_acc), self.swing_acc)
        self.axis_valid = self.axis_valid & ~m1
        self.tr_active = self.tr_active & ~m2
        self.tr_ptr = torch.where(m1, torch.zeros_like(self.tr_ptr), self.tr_ptr)
        self.tick = torch.where(mask, torch.zeros_like(self.tick), self.tick)
        self.status = torch.where(m1, torch.zeros_like(self.status),
                                  self.status)
        self.ep_return = torch.where(mask, torch.zeros_like(self.ep_return),
                                     self.ep_return)
        zl = torch.zeros_like(self.good)
        self.good = torch.where(mask, zl, self.good)
        self.bad = torch.where(mask, zl, self.bad)
        self.miss = torch.where(mask, zl, self.miss)
        self.score = torch.where(mask, torch.zeros_like(self.score), self.score)

    def reset(self):
        ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_envs(ones)
        self._episode += 1
        self._build_obs()
        return self._obs

    # -- core step (mirrors lib.rs step_inner) --------------------------------
    def step(self, actions):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(np.asarray(actions), device=self.device)
        actions = actions.to(self.device).long().view(
            self.num_envs, 2, N_JOINTS)
        prev_pos = self.wrist.clone()
        prev_dir = self.blade.clone()

        # one integrator for every joint (env v6): accel, drag, omega cap,
        # angle step, per-hand limit clamp (hitting a limit kills omega)
        self.om = ((self.om + self._sgn[actions] * (self._alpha * DT))
                   * (1.0 - DRAG)).clamp(-self._omega, self._omega)
        self.q = self.q + self.om * DT
        q_out = (self.q < self._q_lo) | (self.q > self._q_hi)
        self.q = torch.max(torch.min(self.q, self._q_hi), self._q_lo)
        self.om = torch.where(q_out, torch.zeros_like(self.om), self.om)
        self._fk()
        self.wrist_vel = (self.wrist - prev_pos) / DT

        # swing chain update (feeds pre-swing + follow-through scores)
        d_new = self.blade
        dot = (prev_dir * d_new).sum(-1).clamp(-1.0, 1.0)
        delta = dot.acos()                                       # (N, 2)
        cross = torch.cross(prev_dir, d_new, dim=-1)
        cn = cross.norm(dim=-1, keepdim=True)
        ax = cross / cn.clamp_min(1e-30)
        sw_ok = delta >= SWING_BREAK
        rev = sw_ok & self.axis_valid & ((self.axis * ax).sum(-1) < 0.0)
        cont = sw_ok & ~rev                                      # (N, 2)
        self.swing_acc = torch.where(
            cont, self.swing_acc + delta,
            torch.where(rev, delta, torch.zeros_like(delta)))
        self.axis = torch.where(sw_ok.unsqueeze(-1), ax, self.axis)
        self.axis_valid = sw_ok

        # follow-through trackers (existing ones, BEFORE this tick's cuts):
        # chain broke/reversed -> resolve; else accumulate, resolve at 60 deg
        # or timeout. Trackers with cnt notes pay cnt * points.
        cont_k = cont.unsqueeze(-1)                              # (N, 2, 1)
        acc_new = torch.where(cont_k, self.tr_acc + delta.unsqueeze(-1),
                              self.tr_acc)
        age_new = torch.where(cont_k, self.tr_age + 1, self.tr_age)
        resolve = self.tr_active & (
            ~cont_k | (acc_new >= POST_FULL) | (age_new >= FOLLOW_TIMEOUT))
        post_pts = (PTS_POST * (acc_new / POST_FULL).clamp(max=1.0)
                    * self.tr_cnt.to(self.dtype))
        post_paid = (post_pts * resolve.to(self.dtype)).sum((1, 2))  # (N,)
        reward = post_paid / SCORE_MAX
        self.score = self.score + post_paid
        self.tr_acc, self.tr_age = acc_new, age_new
        self.tr_active = self.tr_active & ~resolve

        time_new = (self.tick + 1).to(self.dtype) * DT              # (N,)
        z = (self.c_hit - time_new[:, None]) * self.c_speed         # (N,C)
        pending = (self.status == 0) & self.c_valid

        missed = pending & (z < -Z_MISS)
        cuttable = pending & ~missed & (z <= REACH_Z)

        # swept-blade test (env v5.3): 8 sample points along each blade,
        # each tracing its own 3D displacement segment against every note's
        # ROTATED cube (slab clip in the box frame); distance-to-center of
        # the hitting samples still grades accuracy / picks s*.
        # Static loop over samples bounds memory; first minimum wins ties
        # (matches Rust's strict <, smallest sample fraction kept).
        center3 = torch.cat([self.c_center, z.unsqueeze(-1)], -1)   # (N,C,3)
        c3 = center3.unsqueeze(1)                                   # (N,1,C,3)
        cr = self.c_rot[..., 0].unsqueeze(1)                        # (N,1,C)
        sr = self.c_rot[..., 1].unsqueeze(1)
        pos3_prev = prev_pos                                        # (N,2,3)
        pos3 = self.wrist
        d_min = torch.full((self.num_envs, 2, self.C), float("inf"),
                           device=self.device, dtype=self.dtype)
        s_star = torch.zeros_like(d_min)

        def _box_frame(p):                                # (N,2,C,3) local
            q = p - c3
            return torch.stack(
                [cr * q[..., 0] + sr * q[..., 1],
                 -sr * q[..., 0] + cr * q[..., 1],
                 q[..., 2]], -1)

        def _box_hit(la, lb):
            # segment vs origin-centered cube of half-extent HALF_BLOCK,
            # slab clipping of the segment parameter to [0, 1] — the same
            # branch structure as Rust's seg_box_hit (small-axis test is
            # point-in-slab; divisions, not reciprocal multiplies)
            d = lb - la
            small = d.abs() < 1e-12
            d_safe = torch.where(small, torch.ones_like(d), d)
            t1 = (-HALF_BLOCK - la) / d_safe
            t2 = (HALF_BLOCK - la) / d_safe
            lo = torch.minimum(t1, t2)
            hi = torch.maximum(t1, t2)
            inside = la.abs() <= HALF_BLOCK
            inf = torch.full_like(lo, float("inf"))
            lo = torch.where(small, torch.where(inside, torch.zeros_like(lo),
                                                inf), lo)
            hi = torch.where(small, torch.where(inside, torch.ones_like(hi),
                                                -inf), hi)
            return lo.amax(-1).clamp_min(0.0) <= hi.amin(-1).clamp_max(1.0)

        for j in range(1, N_BLADE_SAMPLES + 1):
            r = (j / N_BLADE_SAMPLES) * SABER_LEN
            a = (pos3_prev + r * prev_dir).unsqueeze(2)             # (N,2,1,3)
            b = (pos3 + r * d_new).unsqueeze(2)
            hit = _box_hit(_box_frame(a), _box_frame(b))            # (N,2,C)
            ab = b - a
            denom = (ab * ab).sum(-1)                               # (N,2,1)
            t = ((c3 - a) * ab).sum(-1) / denom.clamp_min(1e-12)
            t = torch.where(denom < 1e-12, torch.zeros_like(t),
                            t.clamp(0.0, 1.0)).unsqueeze(-1)
            d = (a + t * ab - c3).norm(dim=-1)                      # (N,2,C)
            better = hit & (d < d_min)
            d_min = torch.where(better, d, d_min)
            s_star = torch.where(better, torch.full_like(s_star, r), s_star)

        crossed = torch.isfinite(d_min)                             # (N,2,C)
        is_r = self.c_is_right                                      # (N,C)
        match_cross = torch.where(is_r, crossed[:, 1], crossed[:, 0])
        other_cross = torch.where(is_r, crossed[:, 0], crossed[:, 1])
        dm = torch.where(is_r, d_min[:, 1], d_min[:, 0])            # (N,C)
        ss = torch.where(is_r, s_star[:, 1], s_star[:, 0])

        # contact velocity (env v6): the matching hand's contact-point
        # displacement over the tick / DT — same segments the sweep tests
        def _pick(t):                                               # (N,C,3)
            return torch.where(is_r.unsqueeze(-1),
                               t[:, 1].unsqueeze(1).expand(-1, self.C, -1),
                               t[:, 0].unsqueeze(1).expand(-1, self.C, -1))

        ss3 = ss.unsqueeze(-1)
        vc = (_pick(self.wrist) + ss3 * _pick(self.blade)
              - _pick(prev_pos) - ss3 * _pick(prev_dir)) / DT       # (N,C,3)
        vc_mag = vc.norm(dim=-1)
        slow = vc_mag < V_MIN_CUT
        v2_mag = vc[..., :2].norm(dim=-1)
        cosv = (vc[..., :2] * self.c_dvec).sum(-1) / v2_mag.clamp_min(1e-9)
        angle_ok = self.c_any_dir | (cosv >= COS_TOL)

        matched = cuttable & match_cross
        wrong = cuttable & ~match_cross & other_cross
        good = matched & ~slow & angle_ok
        bad = (matched & (slow | ~angle_ok)) | wrong

        # graded score for good cuts: pre-swing + accuracy now, the
        # follow-through via a tracker
        pre_s = PTS_PRE * (self.swing_acc / PRE_FULL).clamp(max=1.0)  # (N,2)
        pre = torch.where(is_r, pre_s[:, 1:2], pre_s[:, 0:1])         # (N,C)
        # corner clips can sit outside the circumscribed circle -> floor 0
        # (also sanitizes dm = inf on never-hit notes: -inf clamps to 0)
        acc_pts = PTS_ACC * (1.0 - dm / CUT_RADIUS).clamp_min(0.0)
        cut_pts = ((pre + acc_pts) * good.to(self.dtype)).sum(1)      # (N,)
        reward = reward + cut_pts / SCORE_MAX
        self.score = self.score + cut_pts

        self.status = torch.where(
            good, torch.ones_like(self.status),
            torch.where(bad, torch.full_like(self.status, 2),
                        torch.where(missed, torch.full_like(self.status, 3),
                                    self.status)))
        n_good = good.sum(1); n_bad = bad.sum(1); n_miss = missed.sum(1)
        self.good += n_good; self.bad += n_bad; self.miss += n_miss
        reward = reward + (n_bad.to(self.dtype) * REWARD_BAD
                           + n_miss.to(self.dtype) * REWARD_MISS)

        # spawn aggregated follow-through trackers (one slot per saber per
        # tick; same-tick trackers share (arc, age) forever, so a count is
        # exact). The ring slot being written is provably inactive.
        n_new = torch.stack([(good & ~is_r).sum(1), (good & is_r).sum(1)], 1)
        spawn = n_new > 0                                           # (N,2)
        slot_hot = torch.nn.functional.one_hot(
            self.tr_ptr, N_TRACKERS).bool() & spawn.unsqueeze(-1)   # (N,2,K)
        self.tr_acc = torch.where(slot_hot, torch.zeros_like(self.tr_acc),
                                  self.tr_acc)
        self.tr_age = torch.where(slot_hot, torch.zeros_like(self.tr_age),
                                  self.tr_age)
        self.tr_cnt = torch.where(slot_hot, n_new.unsqueeze(-1), self.tr_cnt)
        self.tr_active = self.tr_active | slot_hot
        self.tr_ptr = (self.tr_ptr + spawn.long()) % N_TRACKERS

        self.tick = self.tick + 1
        time_now = self.tick.to(self.dtype) * DT
        pending_after = ((self.status == 0) & self.c_valid).any(1)
        no_trackers = ~self.tr_active.any((1, 2))
        done = ((~pending_after & no_trackers
                 & ((self.c_n_notes == 0) | (time_now > self.c_last_hit)))
                | (time_now > self.c_last_hit + 0.5)
                | (self.tick >= self.max_steps))

        # hard episode ends force-resolve pending follow-throughs (partial)
        end_resolve = self.tr_active & done.view(-1, 1, 1)
        end_pts = (PTS_POST * (self.tr_acc / POST_FULL).clamp(max=1.0)
                   * self.tr_cnt.to(self.dtype))
        end_paid = (end_pts * end_resolve.to(self.dtype)).sum((1, 2))
        reward = reward + end_paid / SCORE_MAX
        self.score = self.score + end_paid
        self.tr_active = self.tr_active & ~end_resolve

        self.ep_return = self.ep_return + reward
        self._record_returns(done)
        if self.auto_reset:
            self._reset_envs(done)
            self._episode += 1
        self._build_obs()
        return self._obs, reward, done, []

    # -- episode stats (no per-step host sync) --------------------------------
    def _record_returns(self, done):
        d = done.long()
        pos = self._ret_total + torch.cumsum(d, 0) - 1
        slot = torch.where(done, pos % self._cap,
                           torch.full_like(pos, self._cap))
        self._ret_buf.scatter_(0, slot, self.ep_return.to(self._ret_buf.dtype))
        self._ret_total = self._ret_total + d.sum()

    def pop_stats(self):
        """Returns episode returns finished since the last pop (order
        approximate under ring overflow; logging-grade, one sync per call)."""
        total = int(self._ret_total.item())
        new = total - self._popped
        if new <= 0:
            return []
        take = min(new, self._cap)
        idx = (torch.arange(total - take, total, device=self.device)
               % self._cap)
        vals = self._ret_buf[idx].tolist()
        self._popped = total
        return vals

    # -- observation (mirrors lib.rs build_obs) -------------------------------
    def _build_obs(self):
        N, C = self.num_envs, self.C
        obs = self._obs
        obs.zero_()
        sab = torch.cat([
            torch.stack([
                self.q[..., 0] / PITCH_LIM,
                self.q[..., 1] / YAW_HI[1],
                self.q[..., 2] / TWIST_LIM,
                (2.0 * self.q[..., 3] - ELBOW_HI) / ELBOW_HI,
                self.q[..., 4].cos(), self.q[..., 4].sin(),
                self.q[..., 5] / BEND_LIM], -1),
            self.om / self._omega,
            torch.stack([
                self.wrist[..., 0] / X_NORM,
                self.wrist[..., 1] / Y_NORM,
                self.wrist[..., 2]], -1),
            self.wrist_vel / V_NORM,
            self.blade,
            (self.swing_acc / PRE_FULL).clamp(max=2.0).unsqueeze(-1),
        ], -1)                                                   # (N,2,23)
        obs[:, :OFF_NOTES] = sab.reshape(N, OFF_NOTES).to(torch.float32)

        time_now = self.tick.to(self.dtype) * DT
        pending = (self.status == 0) & self.c_valid
        slot = torch.cumsum(pending.long(), 1) - 1
        sel = pending & (slot < N_NOTE_SLOTS)
        z = ((self.c_hit - time_now[:, None]) * self.c_speed).clamp(*Z_CLIP)
        feat = torch.stack([
            self.c_center[..., 0] / X_NORM, self.c_center[..., 1] / Y_NORM,
            z / Z_SCALE, -self.c_speed / Z_SCALE,
            (~self.c_is_right).to(self.dtype), self.c_is_right.to(self.dtype),
            self.c_dvec[..., 0], self.c_dvec[..., 1],
            torch.ones_like(z)], -1)                                # (N,C,9)

        env_i = torch.arange(N, device=self.device).unsqueeze(1).expand(N, C)
        flat_slot = torch.where(
            sel, env_i * N_NOTE_SLOTS + slot,
            torch.full_like(slot, N * N_NOTE_SLOTS))  # trash row at the end
        notes = torch.zeros(N * N_NOTE_SLOTS + 1, NOTE_FEAT,
                            device=self.device, dtype=self.dtype)
        notes.scatter_(0, flat_slot.reshape(-1, 1).expand(-1, NOTE_FEAT),
                       feat.reshape(-1, NOTE_FEAT))
        obs[:, OFF_NOTES:OFF_NOTE_MASK] = (
            notes[:-1].reshape(N, N_NOTE_SLOTS * NOTE_FEAT).to(torch.float32))
        mask = torch.zeros(N * N_NOTE_SLOTS + 1,
                           device=self.device, dtype=self.dtype)
        mask.scatter_(0, flat_slot.reshape(-1), torch.ones_like(z).reshape(-1))
        obs[:, OFF_NOTE_MASK:] = (
            mask[:-1].reshape(N, N_NOTE_SLOTS).to(torch.float32))
