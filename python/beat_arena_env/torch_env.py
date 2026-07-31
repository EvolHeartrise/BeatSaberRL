"""Env v10 batched torch sim — the training home (P4).

A semantics-exact port of rust/src/lib.rs `step_inner` + `write_obs` on
top of the batched dynamics engine (torch_dynamics.DynBatch — the
literal §1–3 mirror). The Rust crate remains THE reference
implementation; parity is pinned by tests/test_v10_parity_torch.py:
EXACT (bitwise) f64 obs/reward/done/status/state agreement, bounded
drift in f32.

Bitwise discipline carried from the engine (see torch_dynamics
docstring): transcendentals via the CRT shims, no scalar/tensor
divisions, and every f64 accumulation that Rust performs sequentially is
performed in the same order here — notably the per-tick reward/score
stream: effort first, then follow-through resolutions per saber in
tracker SPAWN order, then the note scan in index order (a host-synced
active-note union keeps the Python loop short; skipped notes contribute
exact zeros, which is bitwise-neutral).

Follow-through trackers are ring slots per saber (v7 design): same-tick
spawns share (arc, age) — and therefore identical resolution points —
so they aggregate into one slot with a note COUNT and a spawn-sequence
stamp. Reward pays each tracker individually (cnt sequential adds of the
same value, matching Rust's per-tracker loop bitwise); the integer-
valued score components use exact bulk sums. The only Rust observable
this loses is WHICH note a `post` event belongs to — the parity suite
pins post totals through reward/score_post instead.
"""

from __future__ import annotations

import numpy as np
import torch

from .chart import ChartConfig, generate_chart
from .constants import (
    ACC_NORM, BAD_HALF, BAD_HALF_Y_LINK, BOMB_RADIUS, CHAIN_LINK_PTS,
    COS_TOL, DT, EFFORT_COEF, FOLLOW_TIMEOUT, GOOD_HALF_X, GOOD_HALF_Y,
    GOOD_HALF_Y_DOT, GOOD_HALF_Y_LINK, GOOD_HALF_Z, GOOD_Z_SHIFT, KIND_BOMB,
    KIND_CHAIN_HEAD, KIND_CHAIN_LINK, KIND_NOTE, LOOK_DIST, LOOK_VERT,
    N_BLADE_SAMPLES, N_JOINTS, N_NOTE_SLOTS, NOTE_FEAT, OBS_DIM,
    OFF_CONTACT, OFF_HAND_L, OFF_HAND_R, OFF_HEAD, OFF_NOTE_MASK, OFF_NOTES,
    OFF_Q, OFF_QD, OFF_ROOT_POS, OFF_ROOT_ROT, OFF_ROOT_VANG, OFF_ROOT_VLIN,
    ONE_WAY_JOINTS, POST_FULL, PRE_FULL, PTS_ACC, PTS_POST, PTS_PRE,
    REACH_PAST_SHOULDER, REWARD_BAD, REWARD_BOMB, REWARD_MISS, SABER_LEN,
    SCORE_MAX, SWING_BREAK, V_MIN_CUT, V_NORM, WRAP_JOINTS, X_NORM, Y_CENTER,
    Y_NORM, Z_CLIP, Z_SCALE, gear_table, note_center,
)
from .constants import JOINT_LIMITS, OMEGA_CAP as OMEGA_CAP_T, TAU_MAX
from .torch_dynamics import (
    DynBatch, H_SUB, N_SUB, NUM_PAIRS, PAIRS, quat_to_rot, tacos, tsqrt,
    tsin, tcos,
)

N_TRACKERS = FOLLOW_TIMEOUT + 1  # ring slots (see module docstring)
Z_MISS_TAIL = 1.5                # lib.rs Z_MISS_TAIL (episode tail, v10)

# arm-side collider ids for the obs contact bits (lib.rs Body::tick)
_ARM_L_IDS = (2, 4, 6)
_ARM_R_IDS = (3, 5, 7)


# -- geometry kernels (mirror lib.rs; identical f64 expression trees) ---------
def _dot(a, b):
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _cross(a, b):
    return torch.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


def _tri_box_overlap(v0, v1, v2, hx, hy, hz):
    """Akenine-Möller triangle-AABB SAT (lib.rs tri_box_overlap); Rust's
    early returns fold into one AND-chain over identical predicates."""
    h = (hx, hy, hz)
    ok = None
    for k in range(3):
        lo = torch.minimum(torch.minimum(v0[..., k], v1[..., k]), v2[..., k])
        hi = torch.maximum(torch.maximum(v0[..., k], v1[..., k]), v2[..., k])
        sep = (lo > h[k]) | (hi < -h[k])
        ok = ~sep if ok is None else ok & ~sep
    edges = (v1 - v0, v2 - v1, v0 - v2)
    zero = torch.zeros((), device=v0.device, dtype=v0.dtype)
    for e in edges:
        ex, ey, ez = e[..., 0], e[..., 1], e[..., 2]
        for ax, ay, az in ((zero, -ez, ey), (ez, zero, -ex), (-ey, ex, zero)):
            p0 = v0[..., 0] * ax + v0[..., 1] * ay + v0[..., 2] * az
            p1 = v1[..., 0] * ax + v1[..., 1] * ay + v1[..., 2] * az
            p2 = v2[..., 0] * ax + v2[..., 1] * ay + v2[..., 2] * az
            r = hx * ax.abs() + hy * ay.abs() + hz * az.abs()
            lo = torch.minimum(torch.minimum(p0, p1), p2)
            hi = torch.maximum(torch.maximum(p0, p1), p2)
            ok = ok & ~((lo > r) | (hi < -r))
    e0, e1 = edges[0], edges[1]
    nx = e0[..., 1] * e1[..., 2] - e0[..., 2] * e1[..., 1]
    ny = e0[..., 2] * e1[..., 0] - e0[..., 0] * e1[..., 2]
    nz = e0[..., 0] * e1[..., 1] - e0[..., 1] * e1[..., 0]
    r = hx * nx.abs() + hy * ny.abs() + hz * nz.abs()
    plane = (nx * v0[..., 0] + ny * v0[..., 1] + nz * v0[..., 2]).abs() <= r
    return ok & plane


def _tri_point_dist(a, b, c, p):
    """Ericson point-triangle distance (lib.rs tri_point_dist); region
    cascade as a priority where-ladder, |denom| < 1e-15 guards mirrored."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    bp = p - b
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    cp = p - c
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    def gdiv(num, den):
        return torch.where(den.abs() < 1e-15, torch.zeros_like(num), num / den)

    r1 = (d1 <= 0.0) & (d2 <= 0.0)
    r2 = (d3 >= 0.0) & (d4 <= d3)
    r3 = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    r4 = (d6 >= 0.0) & (d5 <= d6)
    r5 = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    r6 = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    v3 = gdiv(d1, d1 - d3).unsqueeze(-1)
    q3 = a + v3 * ab
    w5 = gdiv(d2, d2 - d6).unsqueeze(-1)
    q5 = a + w5 * ac
    w6 = gdiv(d4 - d3, (d4 - d3) + (d5 - d6)).unsqueeze(-1)
    q6 = b + w6 * (c - b)
    denom = torch.full_like(va, 1.0) / (va + vb + vc)
    v = (vb * denom).unsqueeze(-1)
    w = (vc * denom).unsqueeze(-1)
    q = a + ab * v + ac * w
    for cond, qi in ((r6, q6), (r5, q5), (r4, c), (r3, q3), (r2, b), (r1, a)):
        q = torch.where(cond.unsqueeze(-1), qi, q)
    d = p - q
    return tsqrt(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]
                 + d[..., 2] * d[..., 2])


def _seg_point_dist(a, b, c):
    """lib.rs seg_point_dist3."""
    ab = b - a
    denom = _dot(ab, ab)
    ac = c - a
    t = torch.where(denom < 1e-12, torch.zeros_like(denom),
                    (_dot(ac, ab) / denom).clamp(0.0, 1.0))
    px = a[..., 0] + t * ab[..., 0] - c[..., 0]
    py = a[..., 1] + t * ab[..., 1] - c[..., 1]
    pz = a[..., 2] + t * ab[..., 2] - c[..., 2]
    return tsqrt(px * px + py * py + pz * pz)


def _note_frame(p, cos_r, sin_r, lam, head):
    """lib.rs note_frame — env v10: `head` is the LIVE FK head (N,1,3)."""
    zero = torch.zeros_like(cos_r)
    up = torch.stack([-sin_r, cos_r, zero], -1)
    hx, hy, hz = head[..., 0], head[..., 1], head[..., 2]
    hp_y = hy + LOOK_VERT * (p[..., 1] - hy)
    d = torch.stack([p[..., 0] - hx, p[..., 1] - hp_y, p[..., 2] - hz], -1)
    dn = tsqrt(_dot(d, d)).clamp_min(1e-9)
    g = torch.stack([lam * d[..., 0] / dn, lam * d[..., 1] / dn,
                     (1.0 - lam) + lam * d[..., 2] / dn], -1)
    gn = tsqrt(_dot(g, g)).clamp_min(1e-9)
    fz = g / gn.unsqueeze(-1)
    cx = _cross(up, fz)
    cn = tsqrt(_dot(cx, cx)).clamp_min(1e-9)
    fx = cx / cn.unsqueeze(-1)
    fy = _cross(fz, fx)
    return fx, fy, fz


def _to_frame(p, center, fx, fy, fz):
    q = p - center
    return torch.stack([_dot(q, fx), _dot(q, fy), _dot(q, fz)], -1)


def _charts_to_tensors(charts, device, dtype):
    """Pad list[list[Note]] into per-field (P, C) tensors — v10: episode
    tail 0.5 + 1.5/min_speed (live head-plane miss boundary)."""
    P = len(charts)
    C = max(1, max((len(ch) for ch in charts), default=1))
    hit = np.zeros((P, C)); x = np.zeros((P, C)); y = np.zeros((P, C))
    color = np.zeros((P, C)); angle = np.zeros((P, C)); speed = np.ones((P, C))
    any_dir = np.zeros((P, C), dtype=bool); kind = np.zeros((P, C), np.int64)
    fpre = np.zeros((P, C), dtype=bool); fpost = np.zeros((P, C), dtype=bool)
    valid = np.zeros((P, C), dtype=bool)
    for p, ch in enumerate(charts):
        for j, n in enumerate(ch):
            hit[p, j] = n.hit_time
            x[p, j], y[p, j] = note_center(n)
            color[p, j] = n.color; angle[p, j] = n.angle; speed[p, j] = n.speed
            any_dir[p, j] = n.any_dir; kind[p, j] = n.kind
            fpre[p, j] = n.free_pre; fpost[p, j] = n.free_post
            valid[p, j] = True
    is_bomb = valid & (kind == KIND_BOMB)
    is_head = valid & (kind == KIND_CHAIN_HEAD)
    is_link = valid & (kind == KIND_CHAIN_LINK)
    is_plain = valid & (kind == KIND_NOTE)
    show_dir = ~is_bomb & (~any_dir | is_link)
    good_hy = np.where(is_link, GOOD_HALF_Y_LINK,
                       np.where(any_dir, GOOD_HALF_Y_DOT, GOOD_HALF_Y))
    bad_hy = np.where(is_link, BAD_HALF_Y_LINK, BAD_HALF)
    rotates = is_link | ~any_dir
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    rot_c = np.where(rotates, -sin_a, 1.0)
    rot_s = np.where(rotates, cos_a, 0.0)
    min_speed = np.where(valid, speed, np.inf).min(1)
    finite = np.isfinite(min_speed)
    tail = np.where(finite,
                    0.5 + Z_MISS_TAIL / np.where(finite, min_speed, 1.0), 0.5)
    t = lambda a, dt=dtype: torch.as_tensor(a, device=device, dtype=dt)
    return {
        "hit": t(hit), "speed": t(speed),
        "center": torch.stack([t(x), t(y)], -1),
        "dvec": torch.stack([t(cos_a), t(sin_a)], -1),
        "rot": torch.stack([t(rot_c), t(rot_s)], -1),
        "is_right": t(color, torch.bool),
        "any_dir": t(any_dir, torch.bool), "valid": t(valid, torch.bool),
        "is_bomb": t(is_bomb, torch.bool), "is_head": t(is_head, torch.bool),
        "is_link": t(is_link, torch.bool), "is_plain": t(is_plain, torch.bool),
        "show_dir": t(show_dir, torch.bool),
        "free_pre": t(fpre, torch.bool), "free_post": t(fpost, torch.bool),
        "good_hy": t(good_hy), "bad_hy": t(bad_hy),
        "n_notes": t(valid.sum(1), torch.long),
        "n_scoreable": t((valid & ~is_bomb).sum(1), torch.long),
        "last_hit": t(np.where(valid.any(1), (hit * valid).max(1), 0.0)),
        "tail": t(tail),
        "min_speed": t(np.where(finite, min_speed, np.inf)),
    }


class TorchBeatVecEnv:
    """BeatVecEnv-alike over the batched v10 dynamics. step() takes an
    (N, 29) long gear-id tensor (SPEC-v10 §1.4 joint order) and returns
    (obs, reward, done, []) — obs f32 on `device`; use pop_stats() once
    per update for episode returns."""

    _F_NC = ("hit", "speed", "valid", "is_right", "any_dir", "is_bomb",
             "is_head", "is_link", "is_plain", "show_dir", "free_pre",
             "free_post", "good_hy", "bad_hy")
    _F_NC2 = ("center", "dvec", "rot")
    _F_N = ("last_hit", "tail", "n_notes", "n_scoreable", "min_speed")

    def __init__(self, num_envs, colors="both", chart_len_s=20.0,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed_min=2.0, note_speed_max=8.0,
                 note_prob=0.25, extra_prob=0.125, dot_prob=0.1,
                 bomb_prob=0.125, seed=0, effort_coef=EFFORT_COEF,
                 wide_acc=False, gears=3,
                 charts=None, chart_pool=1024, max_steps=None,
                 device="cpu", dtype=torch.float32, auto_reset=True):
        assert gears in (3, 5, 7), gears
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.auto_reset = auto_reset
        self.effort_coef = effort_coef
        self.wide_acc = wide_acc
        self.gears = gears
        self._test_charts = charts is not None
        if charts is None:
            cfg = ChartConfig(
                chart_len_s=chart_len_s, colors=colors,
                cols=tuple(cols), rows=tuple(rows),
                note_speed_min=note_speed_min, note_speed_max=note_speed_max,
                note_prob=note_prob, extra_prob=extra_prob,
                dot_prob=dot_prob, bomb_prob=bomb_prob)
            rng = np.random.default_rng(seed)
            charts = [generate_chart(cfg, rng) for _ in range(chart_pool)]
        self.pool = _charts_to_tensors(charts, self.device, dtype)
        self.pool_size, self.C = self.pool["hit"].shape
        if max_steps is None:
            max_steps = int(chart_len_s * 30) + 90
        self.max_steps = max_steps
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        self._episode = 0

        N, C, K = num_envs, self.C, N_TRACKERS
        dev = self.device
        zeros = lambda *s, dt=dtype: torch.zeros(*s, device=dev, dtype=dt)
        self.dyn = DynBatch(N, device=dev, dtype=dtype)
        # gear table -> signed torque fraction; tau = frac * TAU_MAX
        self._gear = torch.tensor(gear_table(gears), device=dev, dtype=dtype)
        self._tau_max = torch.tensor(TAU_MAX, device=dev, dtype=dtype)
        # derived caches (lib.rs Body)
        self.wrist = zeros(N, 2, 3)
        self.blade = zeros(N, 2, 3)
        self.wrist_vel = zeros(N, 2, 3)
        self.head = zeros(N, 3)
        self.shoulder_z = zeros(N, 2)
        self.swing_acc = zeros(N, 2)
        self.axis = zeros(N, 2, 3)
        self.axis_valid = zeros(N, 2, dt=torch.bool)
        self.effort_j = zeros(N)
        self.contact_arm = zeros(N, 2, dt=torch.bool)
        self.contact_imp = zeros(N, NUM_PAIRS)
        self.contact_pt = zeros(N, NUM_PAIRS, 3)
        # arm-side pair masks for the contact bits
        sl = [any(i in _ARM_L_IDS for i in pr) for pr in PAIRS]
        sr = [any(i in _ARM_R_IDS for i in pr) for pr in PAIRS]
        self._pair_arm_l = torch.tensor(sl, device=dev)
        self._pair_arm_r = torch.tensor(sr, device=dev)
        # follow-through tracker ring (+ spawn-seq stamps for pay order)
        self.tr_acc = zeros(N, 2, K)
        self.tr_age = zeros(N, 2, K, dt=torch.long)
        self.tr_cnt = zeros(N, 2, K, dt=torch.long)
        self.tr_seq = zeros(N, 2, K, dt=torch.long)
        self.tr_active = zeros(N, 2, K, dt=torch.bool)
        self.tr_ptr = zeros(N, 2, dt=torch.long)
        self.tick = zeros(N, dt=torch.long)
        self.status = zeros(N, C, dt=torch.uint8)
        self.ep_return = zeros(N)
        self.good = zeros(N, dt=torch.long)
        self.bad = zeros(N, dt=torch.long)
        self.miss = zeros(N, dt=torch.long)
        self.bombs = zeros(N, dt=torch.long)
        self.score = zeros(N)
        self.score_pre = zeros(N)
        self.score_acc = zeros(N)
        self.score_post = zeros(N)
        for k in self._F_NC:
            setattr(self, "c_" + k, zeros(N, C, dt=self.pool[k].dtype))
        for k in self._F_NC2:
            setattr(self, "c_" + k, zeros(N, C, 2))
        for k in self._F_N:
            setattr(self, "c_" + k, zeros(N, dt=self.pool[k].dtype))
        # obs q-block static lane structure
        self._q_lane = []
        lane = OFF_Q
        for j in range(N_JOINTS):
            self._q_lane.append(lane)
            lane += 2 if j in WRAP_JOINTS else 1
        assert lane == OFF_QD
        self._cap = 4096
        self._ret_buf = zeros(self._cap + 1)
        self._ret_total = torch.zeros((), device=dev, dtype=torch.long)
        self._popped = 0
        self._obs = zeros(N, OBS_DIM, dt=torch.float32)

    # -- resets ---------------------------------------------------------------
    def _reset_envs(self, mask):
        N = self.num_envs
        if self._test_charts:
            idx = (torch.arange(N, device=self.device)
                   + self._episode * N) % self.pool_size
        else:
            idx = torch.randint(self.pool_size, (N,), device=self.device,
                                generator=self._gen)
        m1 = mask.unsqueeze(-1)
        m2 = mask.unsqueeze(-1).unsqueeze(-1)
        for k in self._F_NC:
            dst = getattr(self, "c_" + k)
            setattr(self, "c_" + k, torch.where(m1, self.pool[k][idx], dst))
        for k in self._F_NC2:
            dst = getattr(self, "c_" + k)
            setattr(self, "c_" + k, torch.where(m2, self.pool[k][idx], dst))
        for k in self._F_N:
            dst = getattr(self, "c_" + k)
            setattr(self, "c_" + k, torch.where(mask, self.pool[k][idx], dst))
        self.dyn.reset_mask(mask)
        self._refresh_fk()
        self.wrist_vel = torch.where(m2, torch.zeros_like(self.wrist_vel),
                                     self.wrist_vel)
        self.swing_acc = torch.where(m1, torch.zeros_like(self.swing_acc),
                                     self.swing_acc)
        self.axis_valid = self.axis_valid & ~m1
        self.tr_active = self.tr_active & ~m2
        self.tr_ptr = torch.where(m1, torch.zeros_like(self.tr_ptr),
                                  self.tr_ptr)
        self.tick = torch.where(mask, torch.zeros_like(self.tick), self.tick)
        self.status = torch.where(m1, torch.zeros_like(self.status),
                                  self.status)
        self.ep_return = torch.where(mask, torch.zeros_like(self.ep_return),
                                     self.ep_return)
        zl = torch.zeros_like(self.good)
        self.good = torch.where(mask, zl, self.good)
        self.bad = torch.where(mask, zl, self.bad)
        self.miss = torch.where(mask, zl, self.miss)
        self.bombs = torch.where(mask, zl, self.bombs)
        for name in ("score", "score_pre", "score_acc", "score_post",
                     "effort_j"):
            t = getattr(self, name)
            setattr(self, name, torch.where(mask, torch.zeros_like(t), t))
        self.contact_arm = self.contact_arm & ~m1
        self.contact_imp = torch.where(m1, torch.zeros_like(self.contact_imp),
                                       self.contact_imp)

    def reset(self):
        ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_envs(ones)
        self._episode += 1
        self._build_obs()
        return self._obs

    def _refresh_fk(self):
        self.dyn.refresh_fk()
        self.wrist = self.dyn.wrist
        self.blade = self.dyn.blade
        self.head = self.dyn.head
        self.shoulder_z = self.dyn.shoulder_z

    # -- parity / tooling hooks ----------------------------------------------
    def set_state(self, q36, v35):
        """RawBeatSim.set_joints mirror (broadcasts a single state or
        takes (N, ·) batches); resets the finite-diff wrist velocity."""
        self.dyn.set_state(q36, v35)
        self._refresh_fk()
        self.wrist_vel = torch.zeros_like(self.wrist_vel)

    def qpos(self):
        return torch.cat([self.dyn.p, self.dyn.quat, self.dyn.q], -1)

    def twist(self):
        return self.dyn.v[:, :6]

    def omega(self):
        return self.dyn.v[:, 6:]

    def sabers(self):
        return torch.cat([self.wrist, self.blade], -1)  # (N, 2, 6)

    def counters(self):
        return (self.good, self.bad, self.miss, self.c_n_scoreable,
                self.score, self.bombs)

    def components(self):
        return (self.score_pre, self.score_acc, self.score_post)

    def last_effort(self):
        return self.effort_j

    def contacts(self):
        """(imp (N,55) tick-summed lambda, point (N,55,3) last substep)."""
        return self.contact_imp, self.contact_pt

    # -- the physics tick (lib.rs Body::tick) ---------------------------------
    def _tick_dynamics(self, actions):
        frac = self._gear[actions]                     # (N, 29)
        tau = frac * self._tau_max
        prev_wrist = self.wrist.clone()
        self.contact_arm = torch.zeros_like(self.contact_arm)
        self.contact_imp = torch.zeros_like(self.contact_imp)
        effort = torch.zeros(self.num_envs, device=self.device,
                             dtype=self.dtype)
        for _s in range(N_SUB):
            self.dyn.substep(tau)
            # SPEC-v10 §4 effort: positive work at the end-of-substep
            # velocity, accumulated in Rust's joint order
            pj = tau * self.dyn.v[:, 6:]
            term = torch.where(pj > 0.0, pj * H_SUB, torch.zeros_like(pj))
            for j in range(N_JOINTS):
                effort = effort + term[:, j]
            active, lam, point = self.dyn._sub_contacts
            self.contact_imp = self.contact_imp + lam
            pt = torch.stack(point, -1)                # (N, 55, 3)
            self.contact_pt = torch.where(active.unsqueeze(-1), pt,
                                          self.contact_pt)
            arm_l = (active & self._pair_arm_l).any(1)
            arm_r = (active & self._pair_arm_r).any(1)
            self.contact_arm = self.contact_arm | torch.stack(
                [arm_l, arm_r], -1)
        self.effort_j = effort
        self.dyn.check_pd()
        self._refresh_fk()
        self.wrist_vel = (self.wrist - prev_wrist) / DT
        return prev_wrist

    # -- ordered payment helper (tracker resolutions, Rust spawn order) -------
    def _pay_trackers(self, resolve, reward):
        """Add each resolving tracker's post points to reward
        (pts/SCORE_MAX per TRACKER, saber 0 then 1, spawn order — the
        Rust Vec walk) and the exact integer totals to score/score_post.
        Returns (reward, paid_pts_total)."""
        pts = torch.floor(
            PTS_POST * (self.tr_acc / POST_FULL).clamp(max=1.0) + 0.5)
        cnt = self.tr_cnt
        # exact integer bulk sums for the score components
        paid = (pts * (resolve & self.tr_active).to(self.dtype)
                * cnt.to(self.dtype)).sum((1, 2))
        live = resolve & self.tr_active
        if bool(live.any()):
            seq = torch.where(live, self.tr_seq,
                              torch.full_like(self.tr_seq, 2**62))
            order = seq.argsort(stable=True, dim=-1)   # (N, 2, K)
            pts_o = pts.gather(-1, order)
            cnt_o = cnt.gather(-1, order)
            live_o = live.gather(-1, order)
            cnt_max = int(cnt_o[live_o].max().item()) if bool(live.any()) else 0
            unit = pts_o / SCORE_MAX
            for s in range(2):
                for r in range(N_TRACKERS):
                    lv = live_o[:, s, r]
                    if not bool(lv.any()):
                        continue
                    for c in range(cnt_max):
                        m = lv & (c < cnt_o[:, s, r])
                        reward = reward + torch.where(
                            m, unit[:, s, r], torch.zeros_like(reward))
        self.score = self.score + paid
        self.score_post = self.score_post + paid
        self.tr_active = self.tr_active & ~resolve
        return reward

    # -- core step (lib.rs step_inner) ----------------------------------------
    def step(self, actions):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(np.asarray(actions), device=self.device)
        actions = actions.to(self.device).long().view(self.num_envs, N_JOINTS)
        N, C = self.num_envs, self.C
        prev_dir = self.blade.clone()
        prev_pos_saved = self._tick_dynamics(actions)
        prev_pos = prev_pos_saved

        # reward = 0 − coef*effort (Rust: reward -= effort_coef * effort_j)
        reward = 0.0 - self.effort_coef * self.effort_j

        # swing chain update
        d_new = self.blade
        dot = (prev_dir[..., 0] * d_new[..., 0] + prev_dir[..., 1] * d_new[..., 1]
               + prev_dir[..., 2] * d_new[..., 2]).clamp(-1.0, 1.0)
        delta = tacos(dot)                                       # (N, 2)
        cx = _cross(prev_dir, d_new)
        cn = tsqrt(cx[..., 0] * cx[..., 0] + cx[..., 1] * cx[..., 1]
                   + cx[..., 2] * cx[..., 2])
        ax = cx / cn.unsqueeze(-1)     # NaN where cn == 0 — masked below
        sw_ok = delta >= SWING_BREAK
        adot = (self.axis[..., 0] * ax[..., 0] + self.axis[..., 1] * ax[..., 1]
                + self.axis[..., 2] * ax[..., 2])
        rev = sw_ok & self.axis_valid & (adot < 0.0)
        cont = sw_ok & ~rev
        self.swing_acc = torch.where(
            cont, self.swing_acc + delta,
            torch.where(rev, delta, torch.zeros_like(delta)))
        self.axis = torch.where(sw_ok.unsqueeze(-1), ax, self.axis)
        self.axis_valid = sw_ok

        # follow-through trackers (existing, BEFORE this tick's cuts)
        cont_k = cont.unsqueeze(-1)
        acc_new = torch.where(cont_k & self.tr_active,
                              self.tr_acc + delta.unsqueeze(-1), self.tr_acc)
        age_new = torch.where(cont_k & self.tr_active, self.tr_age + 1,
                              self.tr_age)
        self.tr_acc, self.tr_age = acc_new, age_new
        resolve = self.tr_active & (~cont_k | (acc_new >= POST_FULL)
                                    | (age_new >= FOLLOW_TIMEOUT))
        reward = self._pay_trackers(resolve, reward)

        time_new = (self.tick + 1).to(self.dtype) * DT           # (N,)
        z = (self.c_hit - time_new[:, None]) * self.c_speed      # (N, C)
        pending = (self.status == 0) & self.c_valid

        # live gates (SPEC-v10 §4)
        reach_z = (torch.maximum(self.shoulder_z[:, 0], self.shoulder_z[:, 1])
                   + REACH_PAST_SHOULDER).unsqueeze(-1)          # (N, 1)
        head_z = self.head[:, 2].unsqueeze(-1)

        missed = pending & (z < head_z)
        miss_note = missed & ~self.c_is_bomb
        bomb_pass = missed & self.c_is_bomb
        cuttable = pending & ~missed & (z <= reach_z)

        center3 = torch.cat([self.c_center, z.unsqueeze(-1)], -1)  # (N,C,3)
        tip_prev = prev_pos + SABER_LEN * prev_dir
        tip_new = self.wrist + SABER_LEN * self.blade
        w0 = prev_pos.unsqueeze(2); t0 = tip_prev.unsqueeze(2)
        t1 = tip_new.unsqueeze(2); w1 = self.wrist.unsqueeze(2)
        bc = center3.unsqueeze(1)
        ghy = self.c_good_hy.unsqueeze(1)
        bhy = self.c_bad_hy.unsqueeze(1)
        lam = torch.where(self.c_is_plain,
                          (1.0 - z / LOOK_DIST).clamp(0.0, 1.0),
                          torch.zeros_like(z))
        fx, fy, fz = _note_frame(center3, self.c_rot[..., 0],
                                 self.c_rot[..., 1], lam,
                                 self.head.unsqueeze(1))
        gc = (center3 + GOOD_Z_SHIFT * fz).unsqueeze(1)
        fx1, fy1, fz1 = fx.unsqueeze(1), fy.unsqueeze(1), fz.unsqueeze(1)
        w0g = _to_frame(w0, gc, fx1, fy1, fz1)
        t0g = _to_frame(t0, gc, fx1, fy1, fz1)
        t1g = _to_frame(t1, gc, fx1, fy1, fz1)
        w1g = _to_frame(w1, gc, fx1, fy1, fz1)
        good_hit = (
            _tri_box_overlap(w0g, t0g, t1g, GOOD_HALF_X, ghy, GOOD_HALF_Z)
            | _tri_box_overlap(w0g, t1g, w1g, GOOD_HALF_X, ghy, GOOD_HALF_Z))
        w0b = _to_frame(w0, bc, fx1, fy1, fz1)
        t0b = _to_frame(t0, bc, fx1, fy1, fz1)
        t1b = _to_frame(t1, bc, fx1, fy1, fz1)
        w1b = _to_frame(w1, bc, fx1, fy1, fz1)
        bad_hit = (
            _tri_box_overlap(w0b, t0b, t1b, BAD_HALF, bhy, BAD_HALF)
            | _tri_box_overlap(w0b, t1b, w1b, BAD_HALF, bhy, BAD_HALF))
        d_sweep = torch.minimum(_tri_point_dist(w0, t0, t1, bc),
                                _tri_point_dist(w0, t1, w1, bc))  # (N,2,C)

        is_r = self.c_is_right
        match_hit = torch.where(is_r, good_hit[:, 1], good_hit[:, 0])
        wrong_hit = torch.where(is_r, bad_hit[:, 0], bad_hit[:, 1])

        bomb_hit = cuttable & self.c_is_bomb & (
            (d_sweep[:, 0] <= BOMB_RADIUS) | (d_sweep[:, 1] <= BOMB_RADIUS))

        # matching-saber kinematics per note
        def _pick(t):
            return torch.where(is_r.unsqueeze(-1),
                               t[:, 1].unsqueeze(1).expand(-1, C, -1),
                               t[:, 0].unsqueeze(1).expand(-1, C, -1))

        w0m, d0m = _pick(prev_pos), _pick(prev_dir)
        w1m, d1m = _pick(self.wrist), _pick(self.blade)
        d_seg = torch.full((N, C), float("inf"), device=self.device,
                           dtype=self.dtype)
        s_star = torch.zeros_like(d_seg)
        for jj in range(1, N_BLADE_SAMPLES + 1):
            r = (jj / N_BLADE_SAMPLES) * SABER_LEN
            d = _seg_point_dist(w0m + r * d0m, w1m + r * d1m, center3)
            better = d < d_seg
            d_seg = torch.where(better, d, d_seg)
            s_star = torch.where(better, torch.full_like(s_star, r), s_star)
        ss3 = s_star.unsqueeze(-1)
        vc = (w1m + ss3 * d1m - w0m - ss3 * d0m) / DT
        vc_mag = tsqrt(vc[..., 0] * vc[..., 0] + vc[..., 1] * vc[..., 1]
                       + vc[..., 2] * vc[..., 2])
        # cut-plane accuracy (2026-07-29 rule; lib.rs step_inner)
        tip_move = _pick(tip_new) - _pick(tip_prev)
        cw = center3 - w1m
        pn = _cross(d1m, tip_move)
        pn_mag = tsqrt(_dot(pn, pn))
        along = _dot(cw, d1m)
        d_min = torch.where(
            pn_mag > 1e-9, (_dot(cw, pn) / pn_mag).abs(),
            tsqrt((_dot(cw, cw) - along * along).clamp_min(0.0)))
        slow = vc_mag < V_MIN_CUT
        vlx = _dot(vc, fx)
        vly = _dot(vc, fy)
        v2_mag = tsqrt(vlx * vlx + vly * vly)
        cosv = -vly / v2_mag.clamp_min(1e-9)
        angle_ok = self.c_any_dir | (cosv >= COS_TOL)

        note_cut = cuttable & ~self.c_is_bomb
        matched = note_cut & match_hit
        good = matched & ~slow & angle_ok
        bad = (matched & (slow | ~angle_ok)) | (note_cut & ~match_hit
                                                & wrong_hit)

        # scoring (integer components via floor(x + 0.5); wide_acc train)
        swing_m = torch.where(is_r, self.swing_acc[:, 1:2],
                              self.swing_acc[:, 0:1])
        pre = torch.where(
            self.c_free_pre, torch.full_like(d_min, float(PTS_PRE)),
            torch.floor(PTS_PRE * (swing_m / PRE_FULL).clamp(max=1.0) + 0.5))
        if self.wide_acc:
            acc_pts = (2.0 * PTS_ACC
                       * (1.0 - (d_min / (2.0 * ACC_NORM)).clamp(0.0, 1.0)))
        else:
            acc_pts = torch.floor(
                PTS_ACC * (1.0 - (d_min / ACC_NORM).clamp(0.0, 1.0)) + 0.5)
        post_now = torch.where(
            self.c_free_post & ~self.c_is_head,
            torch.full_like(d_min, float(PTS_POST)), torch.zeros_like(d_min))
        pts = torch.where(
            self.c_is_link, torch.full_like(d_min, float(CHAIN_LINK_PTS)),
            pre + acc_pts + post_now)
        graded = good & ~self.c_is_link

        # ── ordered accumulation over the active-note union (index order;
        #    skipped notes contribute exact zeros — bitwise-neutral) ────────
        any_event = miss_note | bomb_hit | good | bad
        zero = torch.zeros_like(reward)
        gd = good.to(self.dtype)
        union = torch.nonzero(any_event.any(0)).flatten().tolist()
        for i in union:
            contrib = torch.where(
                miss_note[:, i], torch.full_like(reward, REWARD_MISS),
                torch.where(
                    bomb_hit[:, i], torch.full_like(reward, REWARD_BOMB),
                    torch.where(
                        good[:, i], pts[:, i] / SCORE_MAX,
                        torch.where(bad[:, i],
                                    torch.full_like(reward, REWARD_BAD),
                                    zero))))
            reward = reward + contrib
            self.score = self.score + pts[:, i] * gd[:, i]
            gr = graded[:, i].to(self.dtype)
            self.score_pre = self.score_pre + pre[:, i] * gr
            self.score_acc = self.score_acc + acc_pts[:, i] * gr
            self.score_post = self.score_post + post_now[:, i] * gr

        st = self.status
        st = torch.where(good, torch.ones_like(st), st)
        st = torch.where(bad | bomb_hit, torch.full_like(st, 2), st)
        st = torch.where(miss_note, torch.full_like(st, 3), st)
        st = torch.where(bomb_pass, torch.full_like(st, 4), st)
        self.status = st
        self.good = self.good + good.sum(1)
        self.bad = self.bad + bad.sum(1)
        self.miss = self.miss + miss_note.sum(1)
        self.bombs = self.bombs + bomb_hit.sum(1)

        # spawn aggregated trackers (same-tick spawns share (arc, age))
        spawn_note = (good & ~self.c_is_link & ~self.c_is_head
                      & ~self.c_free_post)
        n_new = torch.stack([(spawn_note & ~is_r).sum(1),
                             (spawn_note & is_r).sum(1)], 1)
        spawn = n_new > 0
        slot_hot = torch.nn.functional.one_hot(
            self.tr_ptr, N_TRACKERS).bool() & spawn.unsqueeze(-1)
        self.tr_acc = torch.where(slot_hot, torch.zeros_like(self.tr_acc),
                                  self.tr_acc)
        self.tr_age = torch.where(slot_hot, torch.zeros_like(self.tr_age),
                                  self.tr_age)
        self.tr_cnt = torch.where(slot_hot, n_new.unsqueeze(-1), self.tr_cnt)
        self.tr_seq = torch.where(slot_hot,
                                  self.tick.view(-1, 1, 1).expand_as(self.tr_seq),
                                  self.tr_seq)
        self.tr_active = self.tr_active | slot_hot
        self.tr_ptr = (self.tr_ptr + spawn.long()) % N_TRACKERS

        self.tick = self.tick + 1
        pending_after = ((self.status == 0) & self.c_valid).any(1)
        no_trackers = ~self.tr_active.any((1, 2))
        done = ((~pending_after & no_trackers
                 & ((self.c_n_notes == 0) | (time_new > self.c_last_hit)))
                | (time_new > self.c_last_hit + self.c_tail)
                | (self.tick >= self.max_steps))

        # done: resolve pending follow-throughs with their partial arcs
        end_resolve = self.tr_active & done.view(-1, 1, 1)
        reward = self._pay_trackers(end_resolve, reward)

        self.ep_return = self.ep_return + reward
        self._record_returns(done)
        if self.auto_reset:
            self._reset_envs(done)
            self._episode += 1
        self._build_obs()
        return self._obs, reward, done, []

    # -- episode stats (device-side ring; one sync per pop) -------------------
    def _record_returns(self, done):
        d = done.long()
        pos = self._ret_total + torch.cumsum(d, 0) - 1
        slot = torch.where(done, pos % self._cap,
                           torch.full_like(pos, self._cap))
        self._ret_buf.scatter_(0, slot, self.ep_return.to(self._ret_buf.dtype))
        self._ret_total = self._ret_total + d.sum()

    def pop_stats(self):
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

    # -- observation (lib.rs write_obs, SPEC-v10 §5) --------------------------
    def _build_obs(self):
        N, C = self.num_envs, self.C
        obs = self._obs
        obs.zero_()
        f32 = lambda t: t.to(torch.float32)
        dyn = self.dyn
        # root position
        obs[:, OFF_ROOT_POS] = f32(dyn.p[:, 0] / X_NORM)
        obs[:, OFF_ROOT_POS + 1] = f32((dyn.p[:, 1] - 0.94) / Y_NORM)
        obs[:, OFF_ROOT_POS + 2] = f32((dyn.p[:, 2] - (-0.65)) / X_NORM)
        # 6D two-column rotation
        r = quat_to_rot(dyn.quat[:, 0], dyn.quat[:, 1], dyn.quat[:, 2],
                        dyn.quat[:, 3])
        for i in range(3):
            obs[:, OFF_ROOT_ROT + i] = f32(r[i][0])
            obs[:, OFF_ROOT_ROT + 3 + i] = f32(r[i][1])
        for k in range(3):
            obs[:, OFF_ROOT_VLIN + k] = f32(dyn.v[:, k] / 6.0)
            obs[:, OFF_ROOT_VANG + k] = f32(dyn.v[:, 3 + k] / 16.0)
        # q block (31 lanes)
        for j in range(N_JOINTS):
            lane = self._q_lane[j]
            qj = dyn.q[:, j]
            if j in WRAP_JOINTS:
                obs[:, lane] = f32(tcos(qj))
                obs[:, lane + 1] = f32(tsin(qj))
            elif j in ONE_WAY_JOINTS:
                hi = JOINT_LIMITS[j][1]
                obs[:, lane] = f32((2.0 * qj - hi) / hi)
            else:
                lo, hi = JOINT_LIMITS[j]
                obs[:, lane] = f32(qj / max(abs(lo), abs(hi)))
        for j in range(N_JOINTS):
            obs[:, OFF_QD + j] = f32(dyn.v[:, 6 + j] / OMEGA_CAP_T[j])
        # hand derived blocks
        for s, base in ((0, OFF_HAND_L), (1, OFF_HAND_R)):
            obs[:, base] = f32(self.wrist[:, s, 0] / X_NORM)
            obs[:, base + 1] = f32((self.wrist[:, s, 1] - Y_CENTER) / Y_NORM)
            obs[:, base + 2] = f32(self.wrist[:, s, 2])
            for k in range(3):
                obs[:, base + 3 + k] = f32(self.wrist_vel[:, s, k] / V_NORM)
                obs[:, base + 6 + k] = f32(self.blade[:, s, k])
            obs[:, base + 9] = f32(
                (self.swing_acc[:, s] / PRE_FULL).clamp(max=2.0))
        # head + contact bits
        obs[:, OFF_HEAD] = f32(self.head[:, 0] / X_NORM)
        obs[:, OFF_HEAD + 1] = f32((self.head[:, 1] - 1.70) / Y_NORM)
        obs[:, OFF_HEAD + 2] = f32((self.head[:, 2] - (-0.65)) / X_NORM)
        obs[:, OFF_CONTACT] = self.contact_arm[:, 0].to(torch.float32)
        obs[:, OFF_CONTACT + 1] = self.contact_arm[:, 1].to(torch.float32)
        # note block (v9 NOTE_FEAT verbatim) via slot scatter
        time_now = self.tick.to(self.dtype) * DT
        pending = (self.status == 0) & self.c_valid
        slot = torch.cumsum(pending.long(), 1) - 1
        sel = pending & (slot < N_NOTE_SLOTS)
        z = ((self.c_hit - time_now[:, None]) * self.c_speed).clamp(*Z_CLIP)
        zero = torch.zeros_like(z)
        colored = ~self.c_is_bomb
        feat = torch.stack([
            self.c_center[..., 0] / X_NORM,
            (self.c_center[..., 1] - Y_CENTER) / Y_NORM,
            z / Z_SCALE, -self.c_speed / Z_SCALE,
            (colored & ~self.c_is_right).to(self.dtype),
            (colored & self.c_is_right).to(self.dtype),
            torch.where(self.c_show_dir, self.c_dvec[..., 0], zero),
            torch.where(self.c_show_dir, self.c_dvec[..., 1], zero),
            self.c_is_bomb.to(self.dtype),
            self.c_is_head.to(self.dtype),
            self.c_is_link.to(self.dtype),
            self.c_free_pre.to(self.dtype),
            self.c_free_post.to(self.dtype),
            torch.ones_like(z)], -1)
        env_i = torch.arange(N, device=self.device).unsqueeze(1).expand(N, C)
        flat_slot = torch.where(
            sel, env_i * N_NOTE_SLOTS + slot,
            torch.full_like(slot, N * N_NOTE_SLOTS))
        notes = torch.zeros(N * N_NOTE_SLOTS + 1, NOTE_FEAT,
                            device=self.device, dtype=self.dtype)
        notes.scatter_(0, flat_slot.reshape(-1, 1).expand(-1, NOTE_FEAT),
                       feat.reshape(-1, NOTE_FEAT))
        obs[:, OFF_NOTES:OFF_NOTE_MASK] = (
            notes[:-1].reshape(N, N_NOTE_SLOTS * NOTE_FEAT).to(torch.float32))
        mask = torch.zeros(N * N_NOTE_SLOTS + 1, device=self.device,
                           dtype=self.dtype)
        mask.scatter_(0, flat_slot.reshape(-1), torch.ones_like(z).reshape(-1))
        obs[:, OFF_NOTE_MASK:] = (
            mask[:-1].reshape(N, N_NOTE_SLOTS).to(torch.float32))
