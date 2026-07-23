"""GPU-resident vectorized Beat Arena sim (torch tensors, one row per env).

A semantics-exact port of rust/src/lib.rs step_inner + build_obs (env v7,
real BS geometry/judgment on the v6 kinematic arm chains) — the Rust crate
remains THE reference implementation; this port exists so rollouts can run
wide (N in the thousands) on the GPU with observations born on-device.
Parity with Rust is pinned by tests/test_torch_env.py: exact event/reward/
obs agreement in float64, bounded drift in float32.

Design notes (why it looks the way it does):
- All charts are padded to a common length C; a `valid` mask marks real
  objects. Consumed/padded objects never match any condition, so every rule
  is a pure masked tensor op — no per-env branching anywhere.
- Collision is the swept blade SURFACE per tick (quad prev->cur blade as
  two triangles) vs the axis-aligned good/bad boxes via the Akenine-Möller
  triangle-AABB SAT, and Ericson point-triangle distances for accuracy and
  bombs — the exact expression trees of lib.rs, with Rust's early-return
  branches folded into boolean AND-chains / priority torch.where ladders
  (identical f64 predicates -> identical booleans).
- The contact-point argmin loops the 8 blade samples STATICALLY (bounded
  memory, no data-dependent shapes, no host syncs); first strict minimum
  wins, matching Rust's `<` scan in sample order.
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

import numpy as np
import torch

from .chart import ChartConfig, generate_chart
from .constants import (
    ACC_NORM, BAD_HALF, BAD_HALF_Y_LINK, BEND_LIM, BOMB_RADIUS,
    CHAIN_LINK_PTS, COS_TOL, DRAG, DT, ELBOW_HI, FOLLOW_TIMEOUT,
    GOOD_HALF_X, GOOD_HALF_Y, GOOD_HALF_Y_DOT, GOOD_HALF_Y_LINK,
    GOOD_HALF_Z, GOOD_Z_SHIFT, HEAD, JOINT_ALPHA, JOINT_OMEGA, KIND_BOMB,
    KIND_CHAIN_HEAD, KIND_CHAIN_LINK, L_FORE, L_UPPER, LOOK_DIST,
    LOOK_VERT, N_BLADE_SAMPLES,
    N_JOINTS, N_NOTE_SLOTS, NOTE_FEAT, OBS_DIM, OFF_NOTE_MASK, OFF_NOTES,
    PITCH_LIM, POST_FULL, PRE_FULL, PTS_ACC, PTS_POST, PTS_PRE, REACH_Z,
    PITCH_HI, PITCH_LO,
    REWARD_BAD, REWARD_BOMB, REWARD_MISS, ROLL_HI, ROLL_LO, SABER_LEN,
    SCORE_MAX, SHOULDER, SWING_BREAK, TWIST_HI, TWIST_LIM, TWIST_LO,
    V_MIN_CUT, V_NORM,
    X_NORM, Y_CENTER, Y_NORM, YAW_HI, YAW_LO, Z_CLIP, Z_MISS, Z_SCALE,
    note_center,
)

N_TRACKERS = FOLLOW_TIMEOUT + 1  # ring slots; see module docstring


# -- geometry kernels (mirror lib.rs; identical f64 expression trees) ---------
def _dot(a, b):
    """dot3: explicit left-to-right x*x + y*y + z*z (matches lib.rs)."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _tri_box_overlap(v0, v1, v2, hx, hy, hz):
    """Akenine-Möller triangle-AABB SAT (lib.rs tri_box_overlap). Verts
    RELATIVE to the box center; (hx, hy, hz) half extents (hy may be a
    per-note tensor). Rust's early returns fold into one AND-chain —
    identical predicates on identical f64 values give identical booleans."""
    h = (hx, hy, hz)
    ok = None
    for k in range(3):  # box-axis tests
        lo = torch.minimum(torch.minimum(v0[..., k], v1[..., k]), v2[..., k])
        hi = torch.maximum(torch.maximum(v0[..., k], v1[..., k]), v2[..., k])
        sep = (lo > h[k]) | (hi < -h[k])
        ok = ~sep if ok is None else ok & ~sep
    edges = (v1 - v0, v2 - v1, v0 - v2)
    zero = torch.zeros((), device=v0.device, dtype=v0.dtype)
    for e in edges:  # 9 cross-axis tests: a = u_k x e_j
        ex, ey, ez = e[..., 0], e[..., 1], e[..., 2]
        for ax, ay, az in ((zero, -ez, ey), (ez, zero, -ex), (-ey, ex, zero)):
            p0 = v0[..., 0] * ax + v0[..., 1] * ay + v0[..., 2] * az
            p1 = v1[..., 0] * ax + v1[..., 1] * ay + v1[..., 2] * az
            p2 = v2[..., 0] * ax + v2[..., 1] * ay + v2[..., 2] * az
            r = hx * ax.abs() + hy * ay.abs() + hz * az.abs()
            lo = torch.minimum(torch.minimum(p0, p1), p2)
            hi = torch.maximum(torch.maximum(p0, p1), p2)
            ok = ok & ~((lo > r) | (hi < -r))
    e0, e1 = edges[0], edges[1]  # triangle-plane test
    nx = e0[..., 1] * e1[..., 2] - e0[..., 2] * e1[..., 1]
    ny = e0[..., 2] * e1[..., 0] - e0[..., 0] * e1[..., 2]
    nz = e0[..., 0] * e1[..., 1] - e0[..., 1] * e1[..., 0]
    r = hx * nx.abs() + hy * ny.abs() + hz * nz.abs()
    plane = (nx * v0[..., 0] + ny * v0[..., 1] + nz * v0[..., 2]).abs() <= r
    return ok & plane


def _tri_point_dist(a, b, c, p):
    """Closest distance point p -> triangle (a, b, c) — Ericson's 7-region
    algorithm (lib.rs tri_point_dist). The region cascade becomes a
    priority torch.where ladder applied LAST-region-first so earlier
    regions win, exactly like Rust's early returns; guarded divisions
    mirror the |denom| < 1e-15 -> 0 rule."""
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

    r1 = (d1 <= 0.0) & (d2 <= 0.0)                       # vertex a
    r2 = (d3 >= 0.0) & (d4 <= d3)                        # vertex b
    r3 = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)         # edge ab
    r4 = (d6 >= 0.0) & (d5 <= d6)                        # vertex c
    r5 = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)         # edge ac
    r6 = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)  # edge bc
    v3 = gdiv(d1, d1 - d3).unsqueeze(-1)
    q3 = a + v3 * ab
    w5 = gdiv(d2, d2 - d6).unsqueeze(-1)
    q5 = a + w5 * ac
    w6 = gdiv(d4 - d3, (d4 - d3) + (d5 - d6)).unsqueeze(-1)
    q6 = b + w6 * (c - b)
    denom = 1.0 / (va + vb + vc)                         # interior
    v = (vb * denom).unsqueeze(-1)
    w = (vc * denom).unsqueeze(-1)
    q = a + ab * v + ac * w
    for cond, qi in ((r6, q6), (r5, q5), (r4, c), (r3, q3), (r2, b), (r1, a)):
        q = torch.where(cond.unsqueeze(-1), qi, q)
    d = p - q
    return (d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]
            + d[..., 2] * d[..., 2]).sqrt()


def _cross(a, b):
    """cross3: exact component formulas (matches lib.rs)."""
    return torch.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


def _note_frame(p, cos_r, sin_r, lam):
    """Orthonormal note frame columns (fx, fy, fz) — lib.rs note_frame
    (env v7.6 look-at-player): z' blends +z toward the (damped-y) head->
    note ray with weight lam; the cut-direction z-roll rides as up;
    x' = up x z', y' = z' x x'. lam = 0 reproduces the v7.4 pure z-roll
    frame EXACTLY."""
    zero = torch.zeros_like(cos_r)
    up = torch.stack([-sin_r, cos_r, zero], -1)
    hp_y = HEAD[1] + LOOK_VERT * (p[..., 1] - HEAD[1])
    d = torch.stack([p[..., 0] - HEAD[0], p[..., 1] - hp_y,
                     p[..., 2] - HEAD[2]], -1)
    dn = _dot(d, d).sqrt().clamp_min(1e-9)
    g = torch.stack([lam * d[..., 0] / dn, lam * d[..., 1] / dn,
                     (1.0 - lam) + lam * d[..., 2] / dn], -1)
    gn = _dot(g, g).sqrt().clamp_min(1e-9)
    fz = g / gn.unsqueeze(-1)
    cx = _cross(up, fz)
    cn = _dot(cx, cx).sqrt().clamp_min(1e-9)
    fx = cx / cn.unsqueeze(-1)
    fy = _cross(fz, fx)
    return fx, fy, fz


def _to_frame(p, center, fx, fy, fz):
    """p relative to `center`, expressed in the note frame
    (lib.rs to_frame)."""
    q = p - center
    return torch.stack([_dot(q, fx), _dot(q, fy), _dot(q, fz)], -1)


def _seg_point_dist(a, b, c):
    """Distance from point c to segment ab (lib.rs seg_point_dist3)."""
    ab = b - a
    denom = _dot(ab, ab)
    ac = c - a
    t = torch.where(denom < 1e-12, torch.zeros_like(denom),
                    (_dot(ac, ab) / denom).clamp(0.0, 1.0))
    px = a[..., 0] + t * ab[..., 0] - c[..., 0]
    py = a[..., 1] + t * ab[..., 1] - c[..., 1]
    pz = a[..., 2] + t * ab[..., 2] - c[..., 2]
    return (px * px + py * py + pz * pz).sqrt()


def _charts_to_tensors(charts, device, dtype):
    """Pad a list[list[Note]] into per-field (P, C) tensors (env v7:
    explicit world x/y via note_center, object kinds, arc grants, per-note
    hitbox half-heights, per-chart episode tail)."""
    P = len(charts)
    C = max(1, max(len(ch) for ch in charts))
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
    # dot notes encode obs direction as the ZERO vector; chain links carry
    # their tangent despite judging any-direction (lib.rs build_obs)
    show_dir = ~is_bomb & (~any_dir | is_link)
    good_hy = np.where(is_link, GOOD_HALF_Y_LINK,
                       np.where(any_dir, GOOD_HALF_Y_DOT, GOOD_HALF_Y))
    bad_hy = np.where(is_link, BAD_HALF_Y_LINK, BAD_HALF)
    # env v7.4: hitboxes rotate about z by rot = angle - 270 deg —
    # (cos_r, sin_r) = (-sin a, cos a), from the angle's own sin/cos
    # (lib.rs note_rot). Arrow notes/heads and chain links (tangent)
    # rotate; plain dots stay axis-aligned; bombs are spheres (unused).
    rotates = is_link | ~any_dir
    rot_c = np.where(rotates, -np.sin(angle), 1.0)
    rot_s = np.where(rotates, np.cos(angle), 0.0)
    # episode tail: 0.5 + Z_MISS / min speed (lib.rs set_chart_notes)
    min_speed = np.where(valid, speed, np.inf).min(1)
    finite = np.isfinite(min_speed)
    tail = np.where(finite, 0.5 + Z_MISS / np.where(finite, min_speed, 1.0),
                    0.5)
    t = lambda a, dt=dtype: torch.as_tensor(a, device=device, dtype=dt)
    return {
        "hit": t(hit), "speed": t(speed),
        "center": torch.stack([t(x), t(y)], -1),
        "dvec": torch.stack([t(np.cos(angle)), t(np.sin(angle))], -1),
        "rot": torch.stack([t(rot_c), t(rot_s)], -1),
        "is_right": t(color, torch.bool),
        "any_dir": t(any_dir, torch.bool), "valid": t(valid, torch.bool),
        "is_bomb": t(is_bomb, torch.bool), "is_head": t(is_head, torch.bool),
        "is_link": t(is_link, torch.bool),
        "show_dir": t(show_dir, torch.bool),
        "free_pre": t(fpre, torch.bool), "free_post": t(fpost, torch.bool),
        "good_hy": t(good_hy), "bad_hy": t(bad_hy),
        "n_notes": t(valid.sum(1), torch.long),
        "n_scoreable": t((valid & ~is_bomb).sum(1), torch.long),
        "last_hit": t(np.where(valid.any(1), (hit * valid).max(1), 0.0)),
        "tail": t(tail),
    }


class TorchBeatVecEnv:
    """BeatVecEnv-alike over pure tensor state. step() takes an (N, 12)
    long action tensor (per hand: pitch, yaw, twist, elbow, roll, bend;
    left first) and returns (obs, reward, done, []) — all tensors on
    `device`, infos always empty (use pop_stats() once per update)."""

    _F_NC = ("hit", "speed", "valid", "is_right", "any_dir", "is_bomb",
             "is_head", "is_link", "show_dir", "free_pre", "free_post",
             "good_hy", "bad_hy")
    _F_NC2 = ("center", "dvec", "rot")
    _F_N = ("last_hit", "tail", "n_notes", "n_scoreable")

    def __init__(self, num_envs, note_rate=1.0, colors="right", dirs=(6,),
                 continuous_dirs=False, chart_len_s=20.0, min_hand_gap_s=0.5,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed=4.0, note_speed_max=None, seed=0,
                 bomb_rate=0.25, chain_prob=0.0, arc_prob=0.0,
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
                note_speed_max=note_speed_max, bomb_rate=bomb_rate,
                chain_prob=chain_prob, arc_prob=arc_prob)
            rng = np.random.default_rng(seed)
            charts = [generate_chart(cfg, rng) for _ in range(chart_pool)]
        self.pool = _charts_to_tensors(charts, self.device, dtype)
        self.pool_size, self.C = self.pool["hit"].shape
        if max_steps is None:
            # +90: the tail must cover the miss-boundary trip (Z_MISS
            # 2.65 m at the slowest speeds) past the last note
            max_steps = int(chart_len_s * 30) + 90
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
        # 0 pending, 1 good, 2 bad/bomb-hit, 3 missed, 4 bomb passed clean
        self.status = zeros(N, C, dt=torch.uint8)
        self.ep_return = zeros(N)
        self.good = zeros(N, dt=torch.long); self.bad = zeros(N, dt=torch.long)
        self.miss = zeros(N, dt=torch.long)
        self.bombs = zeros(N, dt=torch.long)
        self.score = zeros(N)
        # per-env chart fields (rows copied from the pool at reset)
        for k in self._F_NC:
            setattr(self, "c_" + k, zeros(N, C, dt=self.pool[k].dtype))
        for k in self._F_NC2:
            setattr(self, "c_" + k, zeros(N, C, 2))
        for k in self._F_N:
            setattr(self, "c_" + k, zeros(N, dt=self.pool[k].dtype))
        # per-joint action table: action a in 0..2 -> sign {0, +1, -1},
        # scaled by the joint's alpha
        self._sgn = torch.tensor([0.0, 1.0, -1.0],
                                 device=self.device, dtype=dtype)
        self._alpha = torch.tensor(JOINT_ALPHA, device=self.device, dtype=dtype)
        self._omega = torch.tensor(JOINT_OMEGA, device=self.device, dtype=dtype)
        # per-hand per-joint limits, broadcast over (N, 2, 6)
        self._q_lo = torch.tensor(
            [[PITCH_LO, YAW_LO[h], TWIST_LO[h], 0.0, ROLL_LO[h], -BEND_LIM]
             for h in (0, 1)], device=self.device, dtype=dtype)
        self._q_hi = torch.tensor(
            [[PITCH_HI, YAW_HI[h], TWIST_HI[h], ELBOW_HI, ROLL_HI[h], BEND_LIM]
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
        for k in self._F_NC:
            dst = getattr(self, "c_" + k)
            setattr(self, "c_" + k, torch.where(m1, self.pool[k][idx], dst))
        for k in self._F_NC2:
            dst = getattr(self, "c_" + k)
            setattr(self, "c_" + k, torch.where(m2, self.pool[k][idx], dst))
        for k in self._F_N:
            dst = getattr(self, "c_" + k)
            setattr(self, "c_" + k, torch.where(mask, self.pool[k][idx], dst))
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
        self.bombs = torch.where(mask, zl, self.bombs)
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
        # or timeout. Points are integer-rounded PER TRACKER (round_pts),
        # then a slot with cnt notes pays cnt * points.
        cont_k = cont.unsqueeze(-1)                              # (N, 2, 1)
        acc_new = torch.where(cont_k, self.tr_acc + delta.unsqueeze(-1),
                              self.tr_acc)
        age_new = torch.where(cont_k, self.tr_age + 1, self.tr_age)
        resolve = self.tr_active & (
            ~cont_k | (acc_new >= POST_FULL) | (age_new >= FOLLOW_TIMEOUT))
        post_pts = (torch.floor(
            PTS_POST * (acc_new / POST_FULL).clamp(max=1.0) + 0.5)
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
        miss_note = missed & ~self.c_is_bomb
        bomb_pass = missed & self.c_is_bomb   # consumed SILENTLY (status 4)
        cuttable = pending & ~missed & (z <= REACH_Z)

        # swept SURFACE per tick (env v7): quad (prev_wrist->prev_tip,
        # cur_wrist->cur_tip) split into triangles (w0,t0,t1) and (w0,t1,w1),
        # tested vs the axis-aligned boxes (SAT) and graded by the closest
        # approach of the surface to the note center (Ericson)
        center3 = torch.cat([self.c_center, z.unsqueeze(-1)], -1)   # (N,C,3)
        tip_prev = prev_pos + SABER_LEN * prev_dir                  # (N,2,3)
        tip_new = self.wrist + SABER_LEN * self.blade
        w0 = prev_pos.unsqueeze(2); t0 = tip_prev.unsqueeze(2)      # (N,2,1,3)
        t1 = tip_new.unsqueeze(2); w1 = self.wrist.unsqueeze(2)
        bc = center3.unsqueeze(1)                                   # (N,1,C,3)
        ghy = self.c_good_hy.unsqueeze(1)                           # (N,1,C)
        bhy = self.c_bad_hy.unsqueeze(1)
        # env v7.6: per-tick note frame — the z-roll (cos_r, sin_r) as up,
        # plus plain notes billboarding toward the head with weight
        # lam = clamp(1 - z/LOOK_DIST, 0, 1); heads/links/bombs lam = 0.
        # Both boxes use the SAME frame; the good box's -0.25 shift rides
        # the note's LOCAL z. d_min / s* / contact velocity stay world-space.
        is_plain = ~(self.c_is_bomb | self.c_is_head | self.c_is_link)
        lam = torch.where(is_plain, (1.0 - z / LOOK_DIST).clamp(0.0, 1.0),
                          torch.zeros_like(z))                      # (N,C)
        fx, fy, fz = _note_frame(center3, self.c_rot[..., 0],
                                 self.c_rot[..., 1], lam)           # (N,C,3)
        gc = (center3 + GOOD_Z_SHIFT * fz).unsqueeze(1)             # (N,1,C,3)
        fx1, fy1, fz1 = fx.unsqueeze(1), fy.unsqueeze(1), fz.unsqueeze(1)
        w0g = _to_frame(w0, gc, fx1, fy1, fz1)
        t0g = _to_frame(t0, gc, fx1, fy1, fz1)
        t1g = _to_frame(t1, gc, fx1, fy1, fz1)
        w1g = _to_frame(w1, gc, fx1, fy1, fz1)
        good_hit = (
            _tri_box_overlap(w0g, t0g, t1g, GOOD_HALF_X, ghy, GOOD_HALF_Z)
            | _tri_box_overlap(w0g, t1g, w1g,
                               GOOD_HALF_X, ghy, GOOD_HALF_Z))      # (N,2,C)
        w0b = _to_frame(w0, bc, fx1, fy1, fz1)
        t0b = _to_frame(t0, bc, fx1, fy1, fz1)
        t1b = _to_frame(t1, bc, fx1, fy1, fz1)
        w1b = _to_frame(w1, bc, fx1, fy1, fz1)
        bad_hit = (
            _tri_box_overlap(w0b, t0b, t1b, BAD_HALF, bhy, BAD_HALF)
            | _tri_box_overlap(w0b, t1b, w1b, BAD_HALF, bhy, BAD_HALF))
        d_sweep = torch.minimum(_tri_point_dist(w0, t0, t1, bc),
                                _tri_point_dist(w0, t1, w1, bc))    # (N,2,C)

        is_r = self.c_is_right                                      # (N,C)
        match_hit = torch.where(is_r, good_hit[:, 1], good_hit[:, 0])
        wrong_hit = torch.where(is_r, bad_hit[:, 0], bad_hit[:, 1])
        dm = torch.where(is_r, d_sweep[:, 1], d_sweep[:, 0])        # d_min

        # bombs: either saber's sweep within BOMB_RADIUS of the center
        bomb_hit = cuttable & self.c_is_bomb & (
            (d_sweep[:, 0] <= BOMB_RADIUS) | (d_sweep[:, 1] <= BOMB_RADIUS))

        # contact point s*: the blade sample (r = j/8 * SABER_LEN) whose
        # displacement segment passes CLOSEST to the note center; first
        # strict minimum wins (Rust scans samples in order with `<`)
        def _pick(t):                                               # (N,C,3)
            return torch.where(is_r.unsqueeze(-1),
                               t[:, 1].unsqueeze(1).expand(-1, self.C, -1),
                               t[:, 0].unsqueeze(1).expand(-1, self.C, -1))

        w0m, d0m = _pick(prev_pos), _pick(prev_dir)
        w1m, d1m = _pick(self.wrist), _pick(self.blade)
        d_seg = torch.full((self.num_envs, self.C), float("inf"),
                           device=self.device, dtype=self.dtype)
        s_star = torch.zeros_like(d_seg)
        for j in range(1, N_BLADE_SAMPLES + 1):
            r = (j / N_BLADE_SAMPLES) * SABER_LEN
            d = _seg_point_dist(w0m + r * d0m, w1m + r * d1m, center3)
            better = d < d_seg
            d_seg = torch.where(better, d, d_seg)
            s_star = torch.where(better, torch.full_like(s_star, r), s_star)
        ss3 = s_star.unsqueeze(-1)
        vc = (w1m + ss3 * d1m - w0m - ss3 * d0m) / DT               # (N,C,3)
        vc_mag = (vc[..., 0] * vc[..., 0] + vc[..., 1] * vc[..., 1]
                  + vc[..., 2] * vc[..., 2]).sqrt()
        slow = vc_mag < V_MIN_CUT
        # env v7.6: direction check in the NOTE'S LOCAL frame — required
        # direction is local -y'; at lam = 0 this equals the old world
        # in-plane check exactly
        vlx = _dot(vc, fx)
        vly = _dot(vc, fy)
        v2_mag = (vlx * vlx + vly * vly).sqrt()
        cosv = -vly / v2_mag.clamp_min(1e-9)
        angle_ok = self.c_any_dir | (cosv >= COS_TOL)

        note_cut = cuttable & ~self.c_is_bomb
        matched = note_cut & match_hit
        good = matched & ~slow & angle_ok
        # matching saber missed the good box -> wrong-color saber vs the
        # SMALL box only
        bad = (matched & (slow | ~angle_ok)) | (note_cut & ~match_hit
                                                & wrong_hit)

        # scoring (real BS 115, integer components via floor(x + 0.5)):
        # chain link = flat 20; chain head = pre + acc (no follow-through);
        # note = pre + acc, then free_post ? +30 now : open a tracker
        swing_m = torch.where(is_r, self.swing_acc[:, 1:2],
                              self.swing_acc[:, 0:1])               # (N,C)
        pre = torch.where(
            self.c_free_pre, torch.full_like(dm, float(PTS_PRE)),
            torch.floor(PTS_PRE * (swing_m / PRE_FULL).clamp(max=1.0) + 0.5))
        acc_pts = torch.floor(
            PTS_ACC * (1.0 - (dm / ACC_NORM).clamp(0.0, 1.0)) + 0.5)
        post_now = torch.where(self.c_free_post & ~self.c_is_head,
                               torch.full_like(dm, float(PTS_POST)),
                               torch.zeros_like(dm))
        pts = torch.where(self.c_is_link,
                          torch.full_like(dm, float(CHAIN_LINK_PTS)),
                          pre + acc_pts + post_now)
        cut_pts = (pts * good.to(self.dtype)).sum(1)                # (N,)
        reward = reward + cut_pts / SCORE_MAX
        self.score = self.score + cut_pts

        st = self.status
        st = torch.where(good, torch.ones_like(st), st)
        st = torch.where(bad | bomb_hit, torch.full_like(st, 2), st)
        st = torch.where(miss_note, torch.full_like(st, 3), st)
        st = torch.where(bomb_pass, torch.full_like(st, 4), st)
        self.status = st
        n_good = good.sum(1); n_bad = bad.sum(1)
        n_miss = miss_note.sum(1); n_bomb = bomb_hit.sum(1)
        self.good += n_good; self.bad += n_bad
        self.miss += n_miss; self.bombs += n_bomb
        reward = reward + (n_bad.to(self.dtype) * REWARD_BAD
                           + n_miss.to(self.dtype) * REWARD_MISS
                           + n_bomb.to(self.dtype) * REWARD_BOMB)

        # spawn aggregated follow-through trackers (one slot per saber per
        # tick; same-tick trackers share (arc, age) forever, so a count is
        # exact). Chain heads/links and arc-headed (free_post) notes spawn
        # none. The ring slot being written is provably inactive.
        spawn_note = (good & ~self.c_is_link & ~self.c_is_head
                      & ~self.c_free_post)
        n_new = torch.stack([(spawn_note & ~is_r).sum(1),
                             (spawn_note & is_r).sum(1)], 1)
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
                | (time_now > self.c_last_hit + self.c_tail)
                | (self.tick >= self.max_steps))

        # hard episode ends force-resolve pending follow-throughs (partial)
        end_resolve = self.tr_active & done.view(-1, 1, 1)
        end_pts = (torch.floor(
            PTS_POST * (self.tr_acc / POST_FULL).clamp(max=1.0) + 0.5)
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
                (self.wrist[..., 1] - Y_CENTER) / Y_NORM,
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
        zero = torch.zeros_like(z)
        colored = ~self.c_is_bomb  # bombs: no color one-hot, no direction
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
            torch.ones_like(z)], -1)                             # (N,C,14)

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
