"""GPU-resident vectorized Beat Arena sim (torch tensors, one row per env).

A semantics-exact port of rust/src/lib.rs step_inner + build_obs (env v5,
blade sabers) — the Rust crate remains THE reference implementation; this
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
- Shaping is not implemented (pure-reward era; the Rust sim keeps it for
  reference runs) — shaping_coef must be 0.

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
ACCEL = 30.0
DRAG = 0.02
V_MAX = 6.0
V_MIN_CUT = 1.0
CUT_RADIUS = 0.28
COS_TOL = 0.5
CELL = 0.5
POS_LO = (-1.5, -1.25)
POS_HI = (1.5, 1.25)
REWARD_BAD = -0.5
REWARD_MISS = -0.5
SABER_LEN = 1.0
ALPHA_ROT = 200.0
OMEGA_MAX = 20.0
PHI_LIM = 1.309
N_BLADE_SAMPLES = 8
REACH_Z = 1.25
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
SABER_FEAT = 10
OFF_NOTES = 2 * SABER_FEAT                             # 20
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT   # 128


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
        "any_dir": t(any_dir, torch.bool), "valid": t(valid, torch.bool),
        "n_notes": t(valid.sum(1), torch.long),
        "last_hit": t(np.where(valid.any(1), (hit * valid).max(1), 0.0)),
    }


class TorchBeatVecEnv:
    """BeatVecEnv-alike over pure tensor state. step() takes an (N, 6) long
    action tensor (per hand: move, rot, tilt; left first) and returns
    (obs, reward, done, []) — all tensors on `device`, infos always empty
    (use pop_stats() once per update)."""

    def __init__(self, num_envs, note_rate=1.0, colors="right", dirs=(6,),
                 continuous_dirs=False, chart_len_s=20.0, min_hand_gap_s=0.5,
                 shaping_coef=0.0, cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed=4.0, note_speed_max=None, seed=0,
                 charts=None, chart_pool=1024, max_steps=None,
                 device="cuda", dtype=torch.float32, auto_reset=True):
        assert shaping_coef == 0.0, (
            "torch env is pure-reward only (shaping lives in the Rust sim)")
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
        # wrist
        self.pos = zeros(N, 2, 2); self.vel = zeros(N, 2, 2)
        # blade pose + swing chain
        self.theta = zeros(N, 2); self.phi = zeros(N, 2)
        self.om_th = zeros(N, 2); self.om_ph = zeros(N, 2)
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
        self.c_last_hit = zeros(N); self.c_n_notes = zeros(N, dt=torch.long)
        # move-action direction table: 0 = coast, k = accel at (k-1)*45 deg
        ang = (torch.arange(8, device=self.device, dtype=dtype)) * (math.pi / 4)
        self._act = torch.cat(
            [zeros(1, 2), torch.stack([ang.cos(), ang.sin()], -1)])
        # rot/tilt action table: 0 = coast, 1 = +alpha, 2 = -alpha
        self._rot = torch.tensor([0.0, ALPHA_ROT * DT, -ALPHA_ROT * DT],
                                 device=self.device, dtype=dtype)
        self._lo = torch.tensor(POS_LO, device=self.device, dtype=dtype)
        self._hi = torch.tensor(POS_HI, device=self.device, dtype=dtype)
        # episode-return ring buffer (device-side; slot `cap` is the trash row)
        self._cap = 4096
        self._ret_buf = zeros(self._cap + 1)
        self._ret_total = zeros((), dt=torch.long)
        self._popped = 0
        self._obs = zeros(N, OBS_DIM, dt=torch.float32)

    # -- blade geometry -------------------------------------------------------
    def _blade_dir(self):
        """(N, 2, 3) unit blade direction from the current pose."""
        ct, st = self.theta.cos(), self.theta.sin()
        cp, sp = self.phi.cos(), self.phi.sin()
        return torch.stack([cp * ct, cp * st, sp], -1)

    def _blade_dir_dot(self):
        """(N, 2, 3) d/dt of the blade direction."""
        ct, st = self.theta.cos(), self.theta.sin()
        cp, sp = self.phi.cos(), self.phi.sin()
        return torch.stack([
            -self.om_th * cp * st - self.om_ph * sp * ct,
            self.om_th * cp * ct - self.om_ph * sp * st,
            self.om_ph * cp], -1)

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
        self.c_last_hit = torch.where(mask, self.pool["last_hit"][idx],
                                      self.c_last_hit)
        self.c_n_notes = torch.where(mask, self.pool["n_notes"][idx],
                                     self.c_n_notes)
        start = torch.tensor([[-0.5, 0.0], [0.5, 0.0]],
                             device=self.device, dtype=self.dtype)
        self.pos = torch.where(m2, start, self.pos)
        self.vel = torch.where(m2, torch.zeros_like(self.vel), self.vel)
        half_pi = torch.full_like(self.theta, math.pi / 2)
        self.theta = torch.where(m1, half_pi, self.theta)
        for t in ("phi", "om_th", "om_ph", "swing_acc"):
            setattr(self, t, torch.where(m1, torch.zeros_like(getattr(self, t)),
                                         getattr(self, t)))
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
        actions = actions.to(self.device).long().view(self.num_envs, 2, 3)
        prev_pos = self.pos.clone()
        prev_dir = self._blade_dir()

        # translation: accel -> drag -> speed cap -> move -> clamp (zeroing v)
        self.vel = self.vel + self._act[actions[..., 0]] * (ACCEL * DT)
        self.vel = self.vel * (1.0 - DRAG)
        spd = self.vel.norm(dim=-1, keepdim=True)
        self.vel = self.vel * torch.clamp(V_MAX / spd.clamp_min(1e-12), max=1.0)
        self.pos = self.pos + self.vel * DT
        out = (self.pos < self._lo) | (self.pos > self._hi)
        self.pos = torch.max(torch.min(self.pos, self._hi), self._lo)
        self.vel = torch.where(out, torch.zeros_like(self.vel), self.vel)

        # rotation: azimuth (CW/CCW) and tilt (supination/pronation)
        self.om_th = ((self.om_th + self._rot[actions[..., 1]])
                      * (1.0 - DRAG)).clamp(-OMEGA_MAX, OMEGA_MAX)
        self.om_ph = ((self.om_ph + self._rot[actions[..., 2]])
                      * (1.0 - DRAG)).clamp(-OMEGA_MAX, OMEGA_MAX)
        self.theta = self.theta + self.om_th * DT
        self.theta = torch.where(self.theta >= math.pi,
                                 self.theta - 2.0 * math.pi, self.theta)
        self.theta = torch.where(self.theta < -math.pi,
                                 self.theta + 2.0 * math.pi, self.theta)
        self.phi = self.phi + self.om_ph * DT
        phi_out = (self.phi < -PHI_LIM) | (self.phi > PHI_LIM)
        self.phi = self.phi.clamp(-PHI_LIM, PHI_LIM)
        self.om_ph = torch.where(phi_out, torch.zeros_like(self.om_ph),
                                 self.om_ph)

        # swing chain update (feeds pre-swing + follow-through scores)
        d_new = self._blade_dir()
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

        # swept-blade test: 8 sample points along each blade, each tracing
        # its own 3D displacement segment against every note center.
        # Static loop over samples bounds memory; first minimum wins ties
        # (matches Rust's strict <, smallest sample fraction kept).
        center3 = torch.cat([self.c_center, z.unsqueeze(-1)], -1)   # (N,C,3)
        c3 = center3.unsqueeze(1)                                   # (N,1,C,3)
        pos3_prev = torch.cat(
            [prev_pos, torch.zeros_like(prev_pos[..., :1])], -1)    # (N,2,3)
        pos3 = torch.cat([self.pos, torch.zeros_like(self.pos[..., :1])], -1)
        d_min = torch.full((self.num_envs, 2, self.C), float("inf"),
                           device=self.device, dtype=self.dtype)
        s_star = torch.zeros_like(d_min)
        for j in range(1, N_BLADE_SAMPLES + 1):
            r = (j / N_BLADE_SAMPLES) * SABER_LEN
            a = (pos3_prev + r * prev_dir).unsqueeze(2)             # (N,2,1,3)
            b = (pos3 + r * d_new).unsqueeze(2)
            ab = b - a
            denom = (ab * ab).sum(-1)                               # (N,2,1)
            t = ((c3 - a) * ab).sum(-1) / denom.clamp_min(1e-12)
            t = torch.where(denom < 1e-12, torch.zeros_like(t),
                            t.clamp(0.0, 1.0)).unsqueeze(-1)
            d = (a + t * ab - c3).norm(dim=-1)                      # (N,2,C)
            better = d < d_min
            d_min = torch.where(better, d, d_min)
            s_star = torch.where(better, torch.full_like(s_star, r), s_star)

        crossed = d_min <= CUT_RADIUS                               # (N,2,C)
        is_r = self.c_is_right                                      # (N,C)
        match_cross = torch.where(is_r, crossed[:, 1], crossed[:, 0])
        other_cross = torch.where(is_r, crossed[:, 0], crossed[:, 1])
        dm = torch.where(is_r, d_min[:, 1], d_min[:, 0])            # (N,C)
        ss = torch.where(is_r, s_star[:, 1], s_star[:, 0])

        # contact velocity of the MATCHING saber at its contact point
        vel3 = torch.cat([self.vel, torch.zeros_like(self.vel[..., :1])], -1)
        dd = self._blade_dir_dot()                                  # (N,2,3)
        vel_m = torch.where(is_r.unsqueeze(-1),
                            vel3[:, 1].unsqueeze(1).expand(-1, self.C, -1),
                            vel3[:, 0].unsqueeze(1).expand(-1, self.C, -1))
        dd_m = torch.where(is_r.unsqueeze(-1),
                           dd[:, 1].unsqueeze(1).expand(-1, self.C, -1),
                           dd[:, 0].unsqueeze(1).expand(-1, self.C, -1))
        vc = vel_m + ss.unsqueeze(-1) * dd_m                        # (N,C,3)
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
        acc_pts = PTS_ACC * (1.0 - dm / CUT_RADIUS)
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
        sab = torch.stack([
            self.pos[..., 0] / X_NORM, self.pos[..., 1] / Y_NORM,
            self.vel[..., 0] / V_MAX, self.vel[..., 1] / V_MAX,
            self.theta.cos(), self.theta.sin(),
            self.phi / PHI_LIM,
            self.om_th / OMEGA_MAX, self.om_ph / OMEGA_MAX,
            (self.swing_acc / PRE_FULL).clamp(max=2.0)], -1)     # (N,2,10)
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
