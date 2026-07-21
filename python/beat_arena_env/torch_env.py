"""GPU-resident vectorized Beat Arena sim (torch tensors, one row per env).

A semantics-exact port of rust/src/lib.rs step_inner + build_obs — the Rust
crate remains THE reference implementation; this port exists so rollouts can
run wide (N in the thousands) on the GPU with observations born on-device.
Parity with Rust is pinned by tests/test_torch_env.py: exact event/reward/obs
agreement in float64, bounded drift in float32.

Design notes (why it looks the way it does):
- All charts are padded to a common length C; a `valid` mask marks real notes.
  Consumed/padded notes never match any condition, so every rule is a pure
  masked tensor op — no per-env branching anywhere.
- Auto-reset draws charts from a pregenerated pool (chart generation stays in
  Python/NumPy — single source of truth — but runs ONCE at construction, not
  per reset).
- Episode returns are recorded into an on-device ring buffer at reset time and
  fetched with pop_stats() once per PPO update: the step loop performs ZERO
  host↔device synchronizations.
- Shaping is not implemented (pure-reward era; the Rust sim keeps it for
  reference runs) — shaping_coef must be 0.

Limits vs BeatVecEnv: no curriculum sampler, no per-step info dicts.
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
Z_WINDOW = 0.6
CELL = 0.5
POS_LO = (-1.5, -1.25)
POS_HI = (1.5, 1.25)
REWARD_GOOD = 1.0
REWARD_BAD = -0.5
REWARD_MISS = -0.5
X_NORM = 1.0
Y_NORM = 0.75
Z_SCALE = 8.0
Z_CLIP = (-1.0, 12.0)
N_NOTE_SLOTS = 12
NOTE_FEAT = 9
OFF_NOTES = 8
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT  # 116


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
    """BeatVecEnv-alike over pure tensor state. step() takes an (N, 2) long
    action tensor and returns (obs, reward, done, []) — all tensors on
    `device`, infos always empty (use pop_stats() once per update)."""

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

        N, C = num_envs, self.C
        zeros = lambda *s, dt=dtype: torch.zeros(*s, device=self.device, dtype=dt)
        self.pos = zeros(N, 2, 2); self.vel = zeros(N, 2, 2)
        self.tick = zeros(N, dt=torch.long)
        self.status = zeros(N, C, dt=torch.uint8)  # 0 pend 1 good 2 bad 3 miss
        self.ep_return = zeros(N)
        self.good = zeros(N, dt=torch.long); self.bad = zeros(N, dt=torch.long)
        self.miss = zeros(N, dt=torch.long)
        # per-env chart fields (rows copied from the pool at reset)
        for k in ("hit", "speed", "valid", "is_right", "any_dir"):
            setattr(self, "c_" + k, zeros(N, C, dt=self.pool[k].dtype))
        self.c_center = zeros(N, C, 2); self.c_dvec = zeros(N, C, 2)
        self.c_last_hit = zeros(N); self.c_n_notes = zeros(N, dt=torch.long)
        # action direction table: 0 = coast, k = accel at (k-1)*45 deg
        ang = (torch.arange(8, device=self.device, dtype=dtype)) * (math.pi / 4)
        self._act = torch.cat(
            [zeros(1, 2), torch.stack([ang.cos(), ang.sin()], -1)])
        self._lo = torch.tensor(POS_LO, device=self.device, dtype=dtype)
        self._hi = torch.tensor(POS_HI, device=self.device, dtype=dtype)
        # episode-return ring buffer (device-side; slot `cap` is the trash row)
        self._cap = 4096
        self._ret_buf = zeros(self._cap + 1)
        self._ret_total = zeros((), dt=torch.long)
        self._popped = 0
        self._obs = zeros(N, OBS_DIM, dt=torch.float32)

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
        self.tick = torch.where(mask, torch.zeros_like(self.tick), self.tick)
        self.status = torch.where(m1, torch.zeros_like(self.status),
                                  self.status)
        self.ep_return = torch.where(mask, torch.zeros_like(self.ep_return),
                                     self.ep_return)
        zl = torch.zeros_like(self.good)
        self.good = torch.where(mask, zl, self.good)
        self.bad = torch.where(mask, zl, self.bad)
        self.miss = torch.where(mask, zl, self.miss)

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
        actions = actions.to(self.device).long()
        prev_pos = self.pos.clone()

        # integrate: accel -> drag -> speed cap -> move -> clamp (zeroing v)
        self.vel = self.vel + self._act[actions] * (ACCEL * DT)
        self.vel = self.vel * (1.0 - DRAG)
        spd = self.vel.norm(dim=-1, keepdim=True)
        self.vel = self.vel * torch.clamp(V_MAX / spd.clamp_min(1e-12), max=1.0)
        self.pos = self.pos + self.vel * DT
        out = (self.pos < self._lo) | (self.pos > self._hi)
        self.pos = torch.max(torch.min(self.pos, self._hi), self._lo)
        self.vel = torch.where(out, torch.zeros_like(self.vel), self.vel)

        time_new = (self.tick + 1).to(self.dtype) * DT              # (N,)
        z = (self.c_hit - time_new[:, None]) * self.c_speed         # (N,C)
        pending = (self.status == 0) & self.c_valid

        missed = pending & (z < -Z_WINDOW)
        in_slab = pending & (z.abs() <= Z_WINDOW)

        # segment->point distance per saber, against every note center
        crossed = []
        for s in (0, 1):
            a = prev_pos[:, s, :].unsqueeze(1)                      # (N,1,2)
            ab = (self.pos[:, s, :] - prev_pos[:, s, :]).unsqueeze(1)
            denom = (ab * ab).sum(-1)                               # (N,1)
            t = ((self.c_center - a) * ab).sum(-1) / denom.clamp_min(1e-12)
            t = torch.where(denom < 1e-12, torch.zeros_like(t),
                            t.clamp(0.0, 1.0)).unsqueeze(-1)
            dist = (a + t * ab - self.c_center).norm(dim=-1)        # (N,C)
            crossed.append(dist <= CUT_RADIUS)
        match_cross = torch.where(self.c_is_right, crossed[1], crossed[0])
        other_cross = torch.where(self.c_is_right, crossed[0], crossed[1])

        # matching saber first (Rust breaks before checking the other hand)
        matched = in_slab & match_cross
        wrong = in_slab & ~match_cross & other_cross
        vel_note = torch.where(self.c_is_right.unsqueeze(-1),
                               self.vel[:, 1, :].unsqueeze(1),
                               self.vel[:, 0, :].unsqueeze(1))      # (N,C,2)
        vspeed = vel_note.norm(dim=-1)
        slow = vspeed < V_MIN_CUT
        cosv = (vel_note * self.c_dvec).sum(-1) / vspeed.clamp_min(1e-9)
        angle_ok = self.c_any_dir | (cosv >= COS_TOL)
        good = matched & ~slow & angle_ok
        bad = (matched & (slow | ~angle_ok)) | wrong

        self.status = torch.where(
            good, torch.ones_like(self.status),
            torch.where(bad, torch.full_like(self.status, 2),
                        torch.where(missed, torch.full_like(self.status, 3),
                                    self.status)))
        n_good = good.sum(1); n_bad = bad.sum(1); n_miss = missed.sum(1)
        self.good += n_good; self.bad += n_bad; self.miss += n_miss
        reward = (n_good.to(self.dtype) * REWARD_GOOD
                  + n_bad.to(self.dtype) * REWARD_BAD
                  + n_miss.to(self.dtype) * REWARD_MISS)
        self.ep_return = self.ep_return + reward

        self.tick = self.tick + 1
        time_now = self.tick.to(self.dtype) * DT
        pending_after = ((self.status == 0) & self.c_valid).any(1)
        done = ((~pending_after
                 & ((self.c_n_notes == 0) | (time_now > self.c_last_hit)))
                | (time_now > self.c_last_hit + 0.5)
                | (self.tick >= self.max_steps))

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
        sab = torch.stack([self.pos[..., 0] / X_NORM, self.pos[..., 1] / Y_NORM,
                           self.vel[..., 0] / V_MAX, self.vel[..., 1] / V_MAX],
                          -1)                                       # (N,2,4)
        obs[:, :OFF_NOTES] = sab.reshape(N, 8).to(torch.float32)

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
