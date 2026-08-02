"""Env v10 batched dynamics in torch — LITERAL mirror of
rust/src/dynamics/{model,fk,algo,collide,solver,step}.rs (P4).

One tensor row per env; every arithmetic expression is transcribed from
the Rust source with the SAME evaluation order (the house discipline:
identical expression trees are what produce exact f64 parity). Where
torch reductions have unspecified order (sum/matmul), this file instead
uses explicitly ordered accumulation loops shaped to be cheap across the
batch:

- 3-vectors/3x3 matrices are Python lists of (N,) tensors; dot3/mat_mul
  accumulate left-to-right exactly like fk.rs. Multiplications by the
  model tables' exact 0.0 / +-1.0 constants are folded symbolically —
  the only tolerated deviation is the SIGN of exact zeros (0.0 vs -0.0
  compare equal and never propagate into a nonzero difference).
- Cholesky is right-looking (trailing update): at step k every entry
  (i, j) receives the SAME "- l[i][k]*l[j][k]" subtraction, in the same
  k order, as algo.rs's left-looking loops — bitwise identical.
- Forward substitution is column-oriented (same per-element operation
  sequence as algo.rs's row loops); backward substitution keeps the row
  loop (its inner k must ascend per element, which a column sweep would
  reverse).
- Transcendentals (sin/cos/acos/sqrt) on CPU f64 route through numpy —
  numpy defers to the platform CRT libm, which is bitwise what Rust's
  f64 methods call; torch's CPU f64 kernels (SLEEF) differ by 1 ulp on
  ~1-7%% of inputs (measured 2026-07-31). Other dtypes/devices use the
  native torch ops (the f32 training path is tolerance-class anyway).
- The PGS unilateral stage iterates contact rows only for the pair ids
  active anywhere in the batch (one host sync per substep); per-env
  inactivity is masked to an exact arithmetic no-op (d = 0 impulses).

The Rust crate remains THE reference; nothing here re-derives a number.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# --- SPEC-v10 §3 constants (step.rs / solver.rs) -----------------------------
H_SUB = 1.0 / 240.0
N_SUB = 8
CLAMP_PASSES = 8
ROOT_W_MAX = 96.0
ROOT_V_MAX = 48.0
PGS_ITERS = 12
BETA_WELD = 0.2
BETA_CONTACT = 0.2
COMPOSITE_EPS = 1e-4

# --- §2 collision constants (collide.rs) -------------------------------------
MARGIN = 0.01
SLOP = 0.005
SEG_EPS = 1e-12
NUM_PAIRS = 55

NQ = 29
NV = 35
VOFF = 6

G = 9.81
GRAVITY = (0.0, -G, 0.0)

# --- model tables (model.rs, verbatim) ---------------------------------------
BASE_MASS = 8.40
BASE_COM = (0.0, 0.0, 0.0)
BASE_INERTIA = (0.038640, 0.101640, 0.101640)
BASE_POS0 = (0.0, 0.94, -0.65)

_Z3 = (0.0, 0.0, 0.0)
_AX = (1.0, 0.0, 0.0)
_AXN = (-1.0, 0.0, 0.0)
_AY = (0.0, 1.0, 0.0)
_AZ = (0.0, 0.0, 1.0)

_M_CHEST, _COM_CHEST, _I_CHEST = 24.20, (0.0, 0.215, 0.0), (0.957507, 0.254964, 0.957507)
_M_HEAD, _I_HEAD = 5.20, (0.016848, 0.016848, 0.016848)
_M_UARM, _COM_UARM, _I_UARM = 2.00, (0.0, 0.0, 0.175), (0.029222, 0.029222, 0.001966)
_M_FARM, _COM_FARM, _I_FARM = 1.20, (0.0, 0.0, 0.15), (0.013056, 0.013056, 0.000931)
_M_HAND, _COM_HAND, _I_HAND = 0.61, (0.0, 0.0, 0.06), (0.000460, 0.000460, 0.000460)
_M_THIGH, _COM_THIGH, _I_THIGH = 10.60, (0.0, -0.22, 0.0), (0.250198, 0.018493, 0.250198)
_M_SHANK, _COM_SHANK, _I_SHANK = 3.20, (0.0, -0.215, 0.0), (0.065964, 0.003161, 0.065964)
_M_FOOT, _COM_FOOT, _I_FOOT = 1.00, (0.0, -0.035, 0.05), (0.005617, 0.005883, 0.001083)


def _rev(parent, anchor, axis, coord):
    return (parent, anchor, axis, coord, 0.0, _Z3, _Z3)


def _revb(parent, anchor, axis, coord, mass, com, inertia):
    return (parent, anchor, axis, coord, mass, com, inertia)


# (parent, anchor, axis, coord, mass, com, inertia) — HUMANOID_FRAMES verbatim.
FRAMES = [
    _rev(-1, (0.0, 0.10, 0.0), _AX, 0),                                  # 0 spine_pitch
    _rev(0, _Z3, _AY, 1),                                                # 1 spine_yaw
    _revb(1, _Z3, _AZ, 2, _M_CHEST, _COM_CHEST, _I_CHEST),               # 2 spine_roll+chest
    (2, (0.0, 0.66, 0.0), _AZ, -1, _M_HEAD, _Z3, _I_HEAD),               # 3 head (WELD)
    _rev(2, (-0.20, 0.43, 0.0), _AY, 4),                                 # 4 l_sh_yaw
    _rev(4, _Z3, _AXN, 3),                                               # 5 l_sh_pitch
    _revb(5, _Z3, _AZ, 5, _M_UARM, _COM_UARM, _I_UARM),                  # 6 l_sh_twist+uarm
    _revb(6, (0.0, 0.0, 0.35), _AXN, 6, _M_FARM, _COM_FARM, _I_FARM),    # 7 l_elbow+farm
    _rev(7, (0.0, 0.0, 0.30), _AZ, 7),                                   # 8 l_wr_roll
    _rev(8, _Z3, _AY, 9),                                                # 9 l_wr_dev
    _revb(9, _Z3, _AXN, 8, _M_HAND, _COM_HAND, _I_HAND),                 # 10 l_wr_bend+hand
    _rev(2, (0.20, 0.43, 0.0), _AY, 11),                                 # 11 r_sh_yaw
    _rev(11, _Z3, _AXN, 10),                                             # 12 r_sh_pitch
    _revb(12, _Z3, _AZ, 12, _M_UARM, _COM_UARM, _I_UARM),                # 13 r_sh_twist+uarm
    _revb(13, (0.0, 0.0, 0.35), _AXN, 13, _M_FARM, _COM_FARM, _I_FARM),  # 14 r_elbow+farm
    _rev(14, (0.0, 0.0, 0.30), _AZ, 14),                                 # 15 r_wr_roll
    _rev(15, _Z3, _AY, 16),                                              # 16 r_wr_dev
    _revb(16, _Z3, _AXN, 15, _M_HAND, _COM_HAND, _I_HAND),               # 17 r_wr_bend+hand
    _rev(-1, (-0.10, 0.0, 0.0), _AXN, 17),                               # 18 l_hip_pitch
    _rev(18, _Z3, _AY, 18),                                              # 19 l_hip_yaw
    _revb(19, _Z3, _AZ, 19, _M_THIGH, _COM_THIGH, _I_THIGH),             # 20 l_hip_roll+thigh
    _revb(20, (0.0, -0.44, 0.0), _AX, 20, _M_SHANK, _COM_SHANK, _I_SHANK),  # 21 l_knee
    _rev(21, (0.0, -0.43, 0.0), _AXN, 21),                               # 22 l_ankle_pitch
    _revb(22, _Z3, _AZ, 22, _M_FOOT, _COM_FOOT, _I_FOOT),                # 23 l_ankle_roll+foot
    _rev(-1, (0.10, 0.0, 0.0), _AXN, 23),                                # 24 r_hip_pitch
    _rev(24, _Z3, _AY, 24),                                              # 25 r_hip_yaw
    _revb(25, _Z3, _AZ, 25, _M_THIGH, _COM_THIGH, _I_THIGH),             # 26 r_hip_roll+thigh
    _revb(26, (0.0, -0.44, 0.0), _AX, 26, _M_SHANK, _COM_SHANK, _I_SHANK),  # 27 r_knee
    _rev(27, (0.0, -0.43, 0.0), _AXN, 27),                               # 28 r_ankle_pitch
    _revb(28, _Z3, _AZ, 28, _M_FOOT, _COM_FOOT, _I_FOOT),                # 29 r_ankle_roll+foot
]
NF = len(FRAMES)

# frame indices (model.rs)
F_CHEST, F_HEAD = 2, 3
F_L_SHOULDER, F_L_UARM, F_L_ELBOW, F_L_WRIST, F_L_HAND = 4, 6, 7, 8, 10
F_R_SHOULDER, F_R_UARM, F_R_ELBOW, F_R_WRIST, F_R_HAND = 11, 13, 14, 15, 17
F_L_THIGH, F_L_SHANK, F_L_FOOT = 20, 21, 23
F_R_THIGH, F_R_SHANK, F_R_FOOT = 26, 27, 29

_PI = math.pi
LIMITS = [
    (-0.3490658503988659, 0.7853981633974483),
    (-0.6981317007977318, 0.6981317007977318),
    (-0.5235987755982988, 0.5235987755982988),
    (-1.5707963267948966, 1.1344640137963142),
    (-1.7453292519943295, 0.7853981633974483),
    (-1.5707963267948966, 1.2217304763960306),
    (0.0, 2.6179938779914944),
    (0.0, 3.141592653589793),
    (-1.6580627893946132, 1.6580627893946132),
    (-0.5235987755982988, 0.2617993877991494),
    (-1.5707963267948966, 1.1344640137963142),
    (-0.7853981633974483, 1.7453292519943295),
    (-1.2217304763960306, 1.5707963267948966),
    (0.0, 2.6179938779914944),
    (-3.141592653589793, 0.0),
    (-1.6580627893946132, 1.6580627893946132),
    (-0.2617993877991494, 0.5235987755982988),
    (-0.3490658503988659, 2.0943951023931953),
    (-0.7853981633974483, 0.6981317007977318),
    (-0.7853981633974483, 0.5235987755982988),
    (0.0, 2.6179938779914944),
    (-0.8726646259971648, 0.3490658503988659),
    (-0.3490658503988659, 0.3490658503988659),
    (-0.3490658503988659, 2.0943951023931953),
    (-0.6981317007977318, 0.7853981633974483),
    (-0.5235987755982988, 0.7853981633974483),
    (0.0, 2.6179938779914944),
    (-0.8726646259971648, 0.3490658503988659),
    (-0.3490658503988659, 0.3490658503988659),
]
TAU_MAX = (
    800.00, 319.93, 500.00,
    113.84, 113.84, 19.45, 19.13, 1.06, 1.06, 1.06,
    113.84, 113.84, 19.45, 19.13, 1.06, 1.06, 1.06,
    484.68, 100.00, 300.00, 250.00, 180.00, 50.00,
    484.68, 100.00, 300.00, 250.00, 180.00, 50.00,
)
OMEGA_CAP = (
    16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 40.0, 40.0, 40.0,
    16.0, 16.0, 16.0, 16.0, 40.0, 40.0, 40.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
)
DAMPING = (
    4.3140, 1.1998, 3.6471,
    0.42689, 0.42689, 0.07292, 0.07174, 0.00159, 0.00159, 0.00159,
    0.42689, 0.42689, 0.07292, 0.07174, 0.00159, 0.00159, 0.00159,
    1.8175, 0.28120, 1.8133, 0.26294, 0.00561, 0.00138,
    1.8175, 0.28120, 1.8133, 0.26294, 0.00561, 0.00138,
)
ARMATURE = (
    0.071900, 0.019996, 0.060786,
    0.0071149, 0.0071149, 0.0012154, 0.0011957, 0.000026556, 0.000026556, 0.000026556,
    0.0071149, 0.0071149, 0.0012154, 0.0011957, 0.000026556, 0.000026556, 0.000026556,
    0.030292, 0.0046866, 0.030222, 0.0043823, 0.000093419, 0.000023083,
    0.030292, 0.0046866, 0.030222, 0.0043823, 0.000093419, 0.000023083,
)

# coord -> frame index (unique)
COORD_FRAME = [None] * NQ
for _k, _fr in enumerate(FRAMES):
    if _fr[3] >= 0:
        COORD_FRAME[_fr[3]] = _k

# ancestor chains (self -> root), per frame — the point/angular jacobian walks
CHAIN = []
for _k in range(NF):
    ch, cur = [], _k
    while cur >= 0:
        ch.append(cur)
        cur = FRAMES[cur][0]
    CHAIN.append(ch)

# --- §2 colliders + live pair table (collide.rs, verbatim) -------------------
COLLIDERS = [
    (F_CHEST, (0.0, 0.0, 0.0), (0.0, 0.43, 0.0), 0.15),
    (F_HEAD, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.09),
    (F_L_UARM, (0.0, 0.0, 0.0), (0.0, 0.0, 0.35), 0.045),
    (F_R_UARM, (0.0, 0.0, 0.0), (0.0, 0.0, 0.35), 0.045),
    (F_L_ELBOW, (0.0, 0.0, 0.0), (0.0, 0.0, 0.30), 0.04),
    (F_R_ELBOW, (0.0, 0.0, 0.0), (0.0, 0.0, 0.30), 0.04),
    (F_L_HAND, (0.0, 0.0, 0.06), (0.0, 0.0, 0.06), 0.05),
    (F_R_HAND, (0.0, 0.0, 0.06), (0.0, 0.0, 0.06), 0.05),
    (F_L_THIGH, (0.0, 0.0, 0.0), (0.0, -0.44, 0.0), 0.06),
    (F_R_THIGH, (0.0, 0.0, 0.0), (0.0, -0.44, 0.0), 0.06),
    (F_L_SHANK, (0.0, 0.0, 0.0), (0.0, -0.43, 0.0), 0.045),
    (F_R_SHANK, (0.0, 0.0, 0.0), (0.0, -0.43, 0.0), 0.045),
]
PAIRS = [
    (0, 4), (0, 5), (0, 6), (0, 7), (0, 10), (0, 11),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
    (1, 10), (1, 11),
    (2, 3), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11),
    (3, 4), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11),
    (4, 5), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11),
    (5, 6), (5, 8), (5, 9), (5, 10), (5, 11),
    (6, 7), (6, 8), (6, 9), (6, 10), (6, 11),
    (7, 8), (7, 9), (7, 10), (7, 11),
    (8, 9), (8, 11),
    (9, 10),
    (10, 11),
]
assert len(PAIRS) == NUM_PAIRS

# feet weld definition (solver.rs)
WELD_FRAMES = (F_L_FOOT, F_R_FOOT)
WELD_TARGET_P = ((-0.10, 0.07, -0.65), (0.10, 0.07, -0.65))
N_WELD = 12


# =============================================================================
# CRT libm shim: bitwise-Rust transcendentals on the CPU f64 parity path.
# =============================================================================
def _use_np(t):
    return t.device.type == "cpu" and t.dtype == torch.float64


def tsin(t):
    if _use_np(t):
        return torch.from_numpy(np.sin(t.contiguous().numpy()))
    return torch.sin(t)


def tcos(t):
    if _use_np(t):
        return torch.from_numpy(np.cos(t.contiguous().numpy()))
    return torch.cos(t)


def tacos(t):
    if _use_np(t):
        return torch.from_numpy(np.arccos(t.contiguous().numpy()))
    return torch.acos(t)


def tsqrt(t):
    if _use_np(t):
        return torch.from_numpy(np.sqrt(t.contiguous().numpy()))
    return torch.sqrt(t)


# =============================================================================
# symbolic scalar helpers: operands are (N,) tensors, Python floats, or None
# (an EXACT zero — sign of zero relaxed, see module docstring).
# =============================================================================
def smul(a, b):
    if a is None or b is None:
        return None
    if isinstance(a, float):
        if a == 0.0:
            return None
        if isinstance(b, float):
            return a * b
        if a == 1.0:
            return b
        if a == -1.0:
            return -b
        return b * a
    if isinstance(b, float):
        if b == 0.0:
            return None
        if b == 1.0:
            return a
        if b == -1.0:
            return -a
        return a * b
    return a * b


def sneg(a):
    if a is None:
        return None
    if isinstance(a, float):
        return -a
    return -a


def sadd(*terms):
    """Left-to-right accumulation (Rust's t1 + t2 + t3 association),
    exact zeros skipped (bitwise-neutral)."""
    acc = None
    for t in terms:
        if t is None:
            continue
        if acc is None:
            acc = t
        elif isinstance(acc, float) and isinstance(t, float):
            acc = acc + t
        else:
            acc = acc + t
    return acc


def ssub(a, b):
    if b is None:
        return a
    if a is None:
        return sneg(b)
    return a - b


def add3(a, b):
    return [sadd(a[0], b[0]), sadd(a[1], b[1]), sadd(a[2], b[2])]


def sub3(a, b):
    return [ssub(a[0], b[0]), ssub(a[1], b[1]), ssub(a[2], b[2])]


def scale3(a, s):
    return [smul(a[0], s), smul(a[1], s), smul(a[2], s)]


def dot3(a, b):
    return sadd(smul(a[0], b[0]), smul(a[1], b[1]), smul(a[2], b[2]))


def cross3(a, b):
    return [
        ssub(smul(a[1], b[2]), smul(a[2], b[1])),
        ssub(smul(a[2], b[0]), smul(a[0], b[2])),
        ssub(smul(a[0], b[1]), smul(a[1], b[0])),
    ]


def mat_vec(m, v):
    return [
        sadd(smul(m[0][0], v[0]), smul(m[0][1], v[1]), smul(m[0][2], v[2])),
        sadd(smul(m[1][0], v[0]), smul(m[1][1], v[1]), smul(m[1][2], v[2])),
        sadd(smul(m[2][0], v[0]), smul(m[2][1], v[1]), smul(m[2][2], v[2])),
    ]


def mat_mul(a, b):
    out = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = sadd(
                smul(a[i][0], b[0][j]), smul(a[i][1], b[1][j]),
                smul(a[i][2], b[2][j]))
    return out


def _dense3(v, ref):
    """Materialize a symbolic 3-vector's Nones/floats as tensors like ref."""
    out = []
    for c in v:
        if c is None:
            out.append(torch.zeros_like(ref))
        elif isinstance(c, float):
            out.append(torch.full_like(ref, c))
        else:
            out.append(c)
    return out


def rot_axis(axis, q):
    """fk.rs rot_axis (Rodrigues) with the axis a compile-time constant;
    q is an (N,) tensor. Expression order: c + k*x*x etc."""
    s = tsin(q)
    c = tcos(q)
    k = 1.0 - c
    x, y, z = axis
    return [
        [sadd(c, smul(smul(k, x), x)), ssub(smul(smul(k, x), y), smul(s, z)),
         sadd(smul(smul(k, x), z), smul(s, y))],
        [sadd(smul(smul(k, x), y), smul(s, z)), sadd(c, smul(smul(k, y), y)),
         ssub(smul(smul(k, y), z), smul(s, x))],
        [ssub(smul(smul(k, x), z), smul(s, y)), sadd(smul(smul(k, y), z), smul(s, x)),
         sadd(c, smul(smul(k, z), z))],
    ]


def quat_to_rot(qw, qx, qy, qz):
    """fk.rs quat_to_rot (wxyz scalar first)."""
    w, x, y, z = qw, qx, qy, qz
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]


def rot_log(r):
    """fk.rs rot_log — SO(3) log map (weld orientation error)."""
    w0 = ssub(r[2][1], r[1][2])
    w1 = ssub(r[0][2], r[2][0])
    w2 = ssub(r[1][0], r[0][1])
    tr = sadd(sadd(r[0][0], r[1][1]), r[2][2])
    c = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = tacos(c)
    sin_t = tsin(theta)
    # untaken-branch 0/0 NaNs are discarded by where
    factor = torch.where(theta < 1e-9, torch.full_like(theta, 0.5),
                         theta / (2.0 * sin_t))
    return [smul(w0, factor), smul(w1, factor), smul(w2, factor)]


def quat_mul(a, b):
    """fk.rs quat_mul on component tuples."""
    return (
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


def _inertia_world(r, idiag):
    """algo.rs inertia_world: R diag(i) R^T, upper computed, lower mirrored
    (bitwise symmetric). idiag are float constants."""
    out = [[None] * 3 for _ in range(3)]
    for a in range(3):
        for b in range(a, 3):
            v = sadd(
                smul(smul(r[a][0], idiag[0]), r[b][0]),
                smul(smul(r[a][1], idiag[1]), r[b][1]),
                smul(smul(r[a][2], idiag[2]), r[b][2]))
            out[a][b] = v
            out[b][a] = v
    return out


def _add_shift(i, m, d):
    """algo.rs add_shift: i += m*((d.d)I - d d^T). m is a float const."""
    if m == 0.0:
        return
    dd = dot3(d, d)
    for a in range(3):
        for b in range(3):
            sel = dd if a == b else None
            i[a][b] = sadd(i[a][b], smul(m, ssub(sel, smul(d[a], d[b]))))


def _mat_add(a, b):
    for i in range(3):
        for j in range(3):
            a[i][j] = sadd(a[i][j], b[i][j])


# =============================================================================
# Batched engine
# =============================================================================
class DynBatch:
    """N independent v10 humanoids stepped in lockstep. State:
    p (N,3), quat (N,4) wxyz, q (N,29), v (N,35)."""

    def __init__(self, n, device="cpu", dtype=torch.float64):
        self.n = n
        self.device = torch.device(device)
        self.dtype = dtype
        z = lambda *s: torch.zeros(*s, device=self.device, dtype=dtype)
        self.p = z(n, 3)
        self.quat = z(n, 4)
        self.q = z(n, NQ)
        self.v = z(n, NV)
        self.reset_all()
        cs = lambda vals: torch.tensor(vals, device=self.device, dtype=dtype)
        self._lo = cs([l[0] for l in LIMITS])
        self._hi = cs([l[1] for l in LIMITS])
        self._cap = cs(OMEGA_CAP)
        self._damp = cs(DAMPING)
        self._sign_rows = None
        # collider pair static tensors
        self._pair_ra = cs([COLLIDERS[a][3] for a, _ in PAIRS])
        self._pair_rb = cs([COLLIDERS[b][3] for _, b in PAIRS])
        self._rsum = cs([COLLIDERS[a][3] + COLLIDERS[b][3] for a, b in PAIRS])
        # membership sign per (pair, coord): +1 if coord's frame in chain of
        # collider b only, -1 if in a only, 0 if both/neither (exact cancel)
        smat = np.zeros((NUM_PAIRS, NQ))
        for pid, (ia, ib) in enumerate(PAIRS):
            ca, cb = set(CHAIN[COLLIDERS[ia][0]]), set(CHAIN[COLLIDERS[ib][0]])
            for j in range(NQ):
                k = COORD_FRAME[j]
                smat[pid, j] = float((k in cb)) - float((k in ca))
        self._pair_sign = cs(smat)
        # per-substep last-solve caches (read by the env layer)
        self.contact_active = torch.zeros(n, NUM_PAIRS, dtype=torch.bool,
                                          device=self.device)
        self.contact_lambda = z(n, NUM_PAIRS)
        self.contact_point = z(n, NUM_PAIRS, 3)
        self._pd_bad = torch.zeros(n, dtype=torch.bool, device=self.device)

    # -- state management -----------------------------------------------------
    def reset_all(self):
        self.reset_mask(torch.ones(self.n, dtype=torch.bool, device=self.device))

    def reset_mask(self, mask):
        m1 = mask.unsqueeze(-1)
        neutral_p = torch.tensor(BASE_POS0, device=self.device, dtype=self.dtype)
        neutral_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device,
                                 dtype=self.dtype)
        self.p = torch.where(m1, neutral_p, self.p)
        self.quat = torch.where(m1, neutral_q, self.quat)
        self.q = torch.where(m1, torch.zeros_like(self.q), self.q)
        self.v = torch.where(m1, torch.zeros_like(self.v), self.v)

    def set_state(self, q36, v35):
        """lib.rs set_joints mirror: q36 = root pos + quat wxyz + 29 joints
        (quat normalized with the same expression), v35 full velocity."""
        q36 = torch.as_tensor(q36, device=self.device, dtype=self.dtype)
        v35 = torch.as_tensor(v35, device=self.device, dtype=self.dtype)
        if q36.dim() == 1:
            q36 = q36.unsqueeze(0).expand(self.n, -1)
        if v35.dim() == 1:
            v35 = v35.unsqueeze(0).expand(self.n, -1)
        self.p = q36[:, :3].clone()
        qw, qx, qy, qz = q36[:, 3], q36[:, 4], q36[:, 5], q36[:, 6]
        qn = tsqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        self.quat = torch.stack([qw / qn, qx / qn, qy / qn, qz / qn], -1)
        self.q = q36[:, 7:].clone()
        self.v = v35.clone()

    # -- FK (fk.rs fk_all) ----------------------------------------------------
    def fk_all(self):
        qt = self.quat
        r0 = quat_to_rot(qt[:, 0], qt[:, 1], qt[:, 2], qt[:, 3])
        p0 = [self.p[:, 0], self.p[:, 1], self.p[:, 2]]
        r = [None] * NF
        pos = [None] * NF
        axw = [None] * NF
        for k, (parent, anchor, axis, coord, _m, _c, _i) in enumerate(FRAMES):
            rl, pl = (r0, p0) if parent < 0 else (r[parent], pos[parent])
            pos[k] = add3(pl, mat_vec(rl, anchor))
            if coord >= 0:
                r[k] = mat_mul(rl, rot_axis(axis, self.q[:, coord]))
            else:
                r[k] = rl
            axw[k] = mat_vec(r[k], axis)
        self.r0, self.p0, self.r, self.pos, self.axw = r0, p0, r, pos, axw

    # -- CRBA (algo.rs crba) --------------------------------------------------
    def crba(self):
        n, dev, dt = self.n, self.device, self.dtype
        Ment = {}  # (i, j) -> tensor/float

        # composite seeds
        cm = [0.0] * NF
        cc = [None] * NF
        ci = [None] * NF
        for k, fr in enumerate(FRAMES):
            mass, com, idiag = fr[4], fr[5], fr[6]
            cm[k] = mass
            if mass > 0.0:
                cc[k] = add3(self.pos[k], mat_vec(self.r[k], com))
                ci[k] = _inertia_world(self.r[k], idiag)
            else:
                cc[k] = list(self.pos[k])
                ci[k] = [[None] * 3 for _ in range(3)]
        bm = BASE_MASS
        bc = add3(self.p0, mat_vec(self.r0, BASE_COM))
        bi = _inertia_world(self.r0, BASE_INERTIA)

        for k in range(NF - 1, -1, -1):
            fr = FRAMES[k]
            coord = fr[3]
            m, c, i = cm[k], cc[k], ci[k]
            if coord >= 0 and m > 0.0:
                j = VOFF + coord
                a = self.axw[k]
                hl = scale3(cross3(a, sub3(c, self.pos[k])), m)
                ha = mat_vec(i, a)
                Ment[(j, j)] = dot3(a, add3(ha, cross3(sub3(c, self.pos[k]), hl)))
                anc = fr[0]
                while anc >= 0:
                    fa = FRAMES[anc]
                    if fa[3] >= 0:
                        ia = VOFF + fa[3]
                        val = dot3(self.axw[anc],
                                   add3(ha, cross3(sub3(c, self.pos[anc]), hl)))
                        Ment[(ia, j)] = val
                        Ment[(j, ia)] = val
                    anc = fa[0]
                lr = add3(ha, cross3(sub3(c, self.p0), hl))
                for t in range(3):
                    Ment[(t, j)] = hl[t]
                    Ment[(j, t)] = hl[t]
                    Ment[(3 + t, j)] = lr[t]
                    Ment[(j, 3 + t)] = lr[t]
            # merge composite k into its parent (or the base)
            parent = fr[0]
            if parent < 0:
                pm, pc, pi = bm, bc, bi
            else:
                pm, pc, pi = cm[parent], cc[parent], ci[parent]
            tm = pm + m
            if tm > 0.0:
                inv = 1.0 / tm
                tc = scale3(add3(scale3(pc, pm), scale3(c, m)), inv)
                _add_shift(pi, pm, sub3(pc, tc))
                _add_shift(pi, m, sub3(c, tc))
                _mat_add(pi, i)
                if parent < 0:
                    bm, bc = tm, tc
                else:
                    cm[parent], cc[parent] = tm, tc

        # armature diagonal
        for j in range(NQ):
            jj = VOFF + j
            cur = Ment.get((jj, jj))
            Ment[(jj, jj)] = sadd(cur, ARMATURE[j])

        # root 6x6 from the total composite
        d = sub3(bc, self.p0)
        rr = [[bi[i][j] for j in range(3)] for i in range(3)]
        _add_shift(rr, bm, d)
        for i in range(3):
            for j in range(3):
                Ment[(i, j)] = bm if i == j else None
                Ment[(3 + i, 3 + j)] = rr[i][j]
        sk = [[None, sneg(d[2]), d[1]], [d[2], None, sneg(d[0])],
              [sneg(d[1]), d[0], None]]
        for i in range(3):
            for j in range(3):
                val = smul(bm, sk[i][j])
                Ment[(3 + i, j)] = val
                Ment[(j, 3 + i)] = val

        # materialize (N, 35, 35)
        zero = torch.zeros(n, device=dev, dtype=dt)
        rows = []
        for i in range(NV):
            row = []
            for j in range(NV):
                e = Ment.get((i, j))
                if e is None:
                    row.append(zero)
                elif isinstance(e, float):
                    row.append(torch.full((n,), e, device=dev, dtype=dt))
                else:
                    row.append(e)
            rows.append(torch.stack(row, -1))
        self.M = torch.stack(rows, -2)

    # -- RNEA (algo.rs rnea) --------------------------------------------------
    def rnea_bias(self):
        """RNEA(q, v, 0) -> bias C (N, 35), gravity included."""
        v = self.v
        om0 = [v[:, 3], v[:, 4], v[:, 5]]
        al0 = [None, None, None]
        vo0 = [v[:, 0], v[:, 1], v[:, 2]]
        ao0 = [None, None, None]
        g = GRAVITY
        om = [None] * NF
        al = [None] * NF
        vo = [None] * NF
        ao = [None] * NF
        fb_ = [None] * NF
        nb_ = [None] * NF
        for k, fr in enumerate(FRAMES):
            parent, _anchor, _axis, coord, mass, com, idiag = fr
            if parent < 0:
                oml, all_, vol, aol, ol = om0, al0, vo0, ao0, self.p0
            else:
                oml, all_, vol, aol, ol = (om[parent], al[parent], vo[parent],
                                           ao[parent], self.pos[parent])
            dvec = sub3(self.pos[k], ol)
            vok = add3(vol, cross3(oml, dvec))
            aok = add3(add3(aol, cross3(all_, dvec)),
                       cross3(oml, cross3(oml, dvec)))
            if coord >= 0:
                qd = self.v[:, VOFF + coord]
                aw = self.axw[k]
                omk = add3(oml, scale3(aw, qd))
                alk = add3(add3(all_, scale3(aw, None)),
                           scale3(cross3(oml, aw), qd))
            else:
                omk, alk = oml, all_
            om[k], al[k], vo[k], ao[k] = omk, alk, vok, aok
            if mass > 0.0:
                e = mat_vec(self.r[k], com)
                ac = add3(add3(aok, cross3(alk, e)), cross3(omk, cross3(omk, e)))
                iw = _inertia_world(self.r[k], idiag)
                fb = scale3(sub3(ac, list(g)), mass)
                nb = add3(mat_vec(iw, alk), cross3(omk, mat_vec(iw, omk)))
                fb_[k] = fb
                nb_[k] = add3(nb, cross3(e, fb))
            else:
                fb_[k] = [None, None, None]
                nb_[k] = [None, None, None]

        # base body inertial force/moment
        e = mat_vec(self.r0, BASE_COM)
        ac = add3(add3(ao0, cross3(al0, e)), cross3(om0, cross3(om0, e)))
        iw = _inertia_world(self.r0, BASE_INERTIA)
        f0 = scale3(sub3(ac, list(g)), BASE_MASS)
        nb = add3(mat_vec(iw, al0), cross3(om0, mat_vec(iw, om0)))
        n0 = add3(nb, cross3(e, f0))

        cvec = [None] * NV
        for k in range(NF - 1, -1, -1):
            fr = FRAMES[k]
            coord, parent = fr[3], fr[0]
            if coord >= 0:
                cvec[VOFF + coord] = dot3(self.axw[k], nb_[k])
            if parent < 0:
                f0 = add3(f0, fb_[k])
                n0 = add3(n0, add3(nb_[k], cross3(sub3(self.pos[k], self.p0),
                                                  fb_[k])))
            else:
                lever = sub3(self.pos[k], self.pos[parent])
                fk_ = fb_[k]
                fb_[parent] = add3(fb_[parent], fk_)
                nb_[parent] = add3(nb_[parent],
                                   add3(nb_[k], cross3(lever, fk_)))
        cvec[0], cvec[1], cvec[2] = f0[0], f0[1], f0[2]
        cvec[3], cvec[4], cvec[5] = n0[0], n0[1], n0[2]
        zero = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.C = torch.stack(
            [zero if c is None else c for c in cvec], -1)

    # f32 stability net (the parity path never uses it): the mass
    # matrix's condition reaches ~3e5 at gimbal-adjacent poses (min
    # eigenvalue ~ the wrist armature 2.7e-5), so f32 Cholesky pivots
    # can go non-positive under contorted cap-riding poses where Rust's
    # f64 stays PD. Flooring the pivot keeps the factor finite; the
    # omega caps + step-11 clamps + step-12 envelope bound the resulting
    # velocities exactly as they bound the Coriolis artifact (SPEC-v10
    # Divergences §19). Occurrences are counted (pd_net_hits) — the
    # smoke/bench report them rather than papering over.
    PD_FLOOR_F32 = 1e-8

    # f64 pivot floor (SPEC-v10 Divergences §22): the f64 analogue of
    # the f32 net above, scaled by the machine-epsilon ratio (1e-8 *
    # eps_f64/eps_f32 ~ 1.9e-17, rounded down a decade). The smallest
    # PHYSICAL pivot (wrist armature 2.7e-5) is 12 decades above, so
    # max(pivot, FLOOR) leaves every legitimate factorization BITWISE
    # untouched — the Rust <-> torch f64 parity bar is preserved; the
    # floor only absorbs the gimbal-adjacent singular noise band that
    # used to raise here (and panic in Rust). Same constant, same
    # simple-max clamp as algo.rs PD_FLOOR_F64 / the .cu f64 path.
    PD_FLOOR_F64 = 1e-17

    # -- dense Cholesky (algo.rs chol_factor_floored, right-looking —
    # bitwise: the right-looking trailing update subtracts the same
    # l_jk^2 terms in the same k order as Rust's left-looking dots) ----
    def chol_factor(self):
        W = self.M.clone()
        L = torch.zeros_like(W)
        exact = self.dtype == torch.float64
        bad = None
        for k in range(NV):
            dcol = W[:, k, k]
            if exact:
                b = dcol < self.PD_FLOOR_F64  # §22 engagement condition
            else:
                b = dcol <= 0.0
            bad = b if bad is None else (bad | b)
            if exact:
                dj = tsqrt(torch.clamp(dcol, min=self.PD_FLOOR_F64))
            else:
                dj = tsqrt(torch.clamp(dcol, min=self.PD_FLOOR_F32))
            L[:, k, k] = dj
            if k + 1 < NV:
                col = W[:, k + 1:, k] / dj.unsqueeze(-1)
                L[:, k + 1:, k] = col
                W[:, k + 1:, k + 1:] = (W[:, k + 1:, k + 1:]
                                        - col.unsqueeze(-1) * col.unsqueeze(-2))
        self.L = L
        self._pd_bad = self._pd_bad | bad

    @staticmethod
    def chol_solve_multi(L, B, nv):
        """Solve L L^T X = B for B (N, nv, R). Forward column-oriented,
        backward row-oriented — the per-element operation sequences match
        algo.rs chol_solve bitwise (see module docstring)."""
        X = B.clone()
        for k in range(nv):
            xk = X[:, k, :] / L[:, k, k].unsqueeze(-1)
            X[:, k, :] = xk
            if k + 1 < nv:
                X[:, k + 1:, :] = (X[:, k + 1:, :]
                                   - L[:, k + 1:, k].unsqueeze(-1) * xk.unsqueeze(1))
        for i in range(nv - 1, -1, -1):
            acc = X[:, i, :]
            for kk in range(i + 1, nv):
                acc = acc - L[:, kk, i].unsqueeze(-1) * X[:, kk, :]
            X[:, i, :] = acc / L[:, i, i].unsqueeze(-1)
        return X

    # -- narrow phase (collide.rs, all 55 pairs batched) ----------------------
    def narrow_phase(self):
        """Returns (active (N,55) bool, gap, normal comps, point comps,
        endpoints-per-pair) — branch structure of seg_seg_closest mirrored
        with where ladders (identical f64 predicates)."""
        # collider endpoints to world, per collider then gathered per pair
        eps_a, eps_b = [], []
        for (frame, la, lb, _r) in COLLIDERS:
            pa = add3(self.pos[frame], mat_vec(self.r[frame], la))
            pb = add3(self.pos[frame], mat_vec(self.r[frame], lb))
            eps_a.append(_dense3(pa, self.p[:, 0]))
            eps_b.append(_dense3(pb, self.p[:, 0]))
        # (N, 55) per component
        stk = lambda f: [torch.stack([f(pid)[c] for pid in range(NUM_PAIRS)], -1)
                         for c in range(3)]
        p1 = stk(lambda pid: eps_a[PAIRS[pid][0]])
        q1 = stk(lambda pid: eps_b[PAIRS[pid][0]])
        p2 = stk(lambda pid: eps_a[PAIRS[pid][1]])
        q2 = stk(lambda pid: eps_b[PAIRS[pid][1]])

        d1 = sub3(q1, p1)
        d2 = sub3(q2, p2)
        rv = sub3(p1, p2)
        a = dot3(d1, d1)
        e = dot3(d2, d2)
        f = dot3(d2, rv)
        zero = torch.zeros_like(a)
        one = torch.ones_like(a)
        c = dot3(d1, rv)
        # general case
        b = dot3(d1, d2)
        denom = a * e - b * b
        s0 = torch.where(denom > SEG_EPS,
                         ((b * f - c * e) / denom).clamp(0.0, 1.0), zero)
        t0 = b * s0 + f
        s_gen = torch.where(
            t0 < 0.0, (-c / a).clamp(0.0, 1.0),
            torch.where(t0 > e, ((b - c) / a).clamp(0.0, 1.0), s0))
        t_gen = torch.where(t0 < 0.0, zero,
                            torch.where(t0 > e, one, t0 / e))
        # degenerate branches (priority: both, first, second, general)
        both = (a <= SEG_EPS) & (e <= SEG_EPS)
        first = a <= SEG_EPS
        second = e <= SEG_EPS
        t_sec = zero
        s_sec = (-c / a).clamp(0.0, 1.0)
        t_first = (f / e).clamp(0.0, 1.0)
        s = torch.where(both, zero,
                        torch.where(first, zero,
                                    torch.where(second, s_sec, s_gen)))
        t = torch.where(both, zero,
                        torch.where(first, t_first,
                                    torch.where(second, t_sec, t_gen)))
        c1 = add3(p1, scale3(d1, s))
        c2 = add3(p2, scale3(d2, t))
        dd = sub3(c1, c2)
        dist2 = dot3(dd, dd)
        dist = tsqrt(dist2)
        gap = dist - self._rsum
        active = gap < MARGIN
        # NOTE: scalar/tensor division in torch CPU f64 is NOT correctly
        # rounded (reciprocal path, ~28% of inputs 1 ulp off — measured
        # 2026-07-31); tensor/tensor is IEEE. Materialize the numerator.
        inv = torch.full_like(dist, 1.0) / dist
        nfall = [zero, one, zero]
        nrm = [torch.where(dist > 1e-9, (c2[k] - c1[k]) * inv, nfall[k])
               for k in range(3)]
        s1 = [c1[k] + nrm[k] * self._pair_ra for k in range(3)]
        s2 = [c2[k] - nrm[k] * self._pair_rb for k in range(3)]
        point = [(s1[k] + s2[k]) * 0.5 for k in range(3)]
        return active, gap, nrm, point

    # -- one constrained substep (step.rs + solver.rs, normative order) -------
    def substep(self, tau):
        """tau (N, 29). Mutates state; caches contact solve results."""
        n, dev, dt = self.n, self.device, self.dtype
        h = H_SUB
        hinv = 1.0 / h

        # 1-3. FK, CRBA, RNEA
        self.fk_all()
        self.crba()
        self.rnea_bias()

        # 4. rhs = tau - b*qdot - C  (root rows 0 - C)
        rhs = torch.empty(n, NV, device=dev, dtype=dt)
        rhs[:, :VOFF] = 0.0 - self.C[:, :VOFF]
        rhs[:, VOFF:] = (tau - self._damp * self.v[:, VOFF:]) - self.C[:, VOFF:]

        # 5. Cholesky
        self.chol_factor()

        # 7 (early half). narrow phase geometry — Rust runs it inside the
        # solve after weld assembly; the values are FK-pure, order-free.
        active_c, gap_c, nrm_c, point_c = self.narrow_phase()

        # weld rows: J_w (N, 12, 35) + bias
        Jw = torch.zeros(n, N_WELD, NV, device=dev, dtype=dt)
        bias_w = torch.zeros(n, N_WELD, device=dev, dtype=dt)
        for widx, (frame, target) in enumerate(zip(WELD_FRAMES, WELD_TARGET_P)):
            pf = self.pos[frame]
            for tcomp in range(3):
                row = 6 * widx + tcomp
                nvec = [1.0 if kk == tcomp else 0.0 for kk in range(3)]
                for k in CHAIN[frame]:
                    fr = FRAMES[k]
                    if fr[3] >= 0:
                        w = cross3(self.axw[k], sub3(pf, self.pos[k]))
                        val = dot3(nvec, w)
                        if val is not None:
                            Jw[:, row, VOFF + fr[3]] = val
                # root linear + angular
                Jw[:, row, tcomp] = 1.0
                rxn = cross3(sub3(pf, self.p0), nvec)
                for tt in range(3):
                    if rxn[tt] is not None:
                        Jw[:, row, 3 + tt] = (rxn[tt] if not isinstance(rxn[tt], float)
                                              else torch.full((n,), rxn[tt],
                                                              device=dev, dtype=dt))
                bias_w[:, row] = -(BETA_WELD * hinv) * (pf[tcomp] - target[tcomp])
            phi = rot_log(self.r[frame])
            for tcomp in range(3):
                row = 6 * widx + 3 + tcomp
                for k in CHAIN[frame]:
                    fr = FRAMES[k]
                    if fr[3] >= 0:
                        axc = self.axw[k][tcomp]
                        if axc is not None:
                            Jw[:, row, VOFF + fr[3]] = (
                                axc if not isinstance(axc, float)
                                else torch.full((n,), axc, device=dev, dtype=dt))
                Jw[:, row, 3 + tcomp] = 1.0
                bias_w[:, row] = -(BETA_WELD * hinv) * phi[tcomp]

        # contact rows: J_c (N, 55, 35) + bias (assembled = active mask)
        Jc = torch.zeros(n, NUM_PAIRS, NV, device=dev, dtype=dt)
        for j in range(NQ):
            k = COORD_FRAME[j]
            sgn = self._pair_sign[:, j]  # (55,)
            if bool((sgn == 0).all()):
                continue
            # w = axw_k x (point - p_k); val = n . w  (Rust dot order)
            ax = _dense3(self.axw[k], self.p[:, 0])
            pk = _dense3(self.pos[k], self.p[:, 0])
            dx = [point_c[t] - pk[t].unsqueeze(-1) for t in range(3)]
            axl = [ax[t].unsqueeze(-1) for t in range(3)]
            w0 = axl[1] * dx[2] - axl[2] * dx[1]
            w1 = axl[2] * dx[0] - axl[0] * dx[2]
            w2 = axl[0] * dx[1] - axl[1] * dx[0]
            val = nrm_c[0] * w0 + nrm_c[1] * w1 + nrm_c[2] * w2
            Jc[:, :, VOFF + j] = sgn * val
        # root columns cancel exactly (both chains reach the root) — zeros.
        bias_c = torch.where(
            gap_c >= 0.0, -gap_c * hinv,
            (BETA_CONTACT * hinv) * torch.clamp(-gap_c - SLOP, min=0.0))

        # limit rows (joint order; nearer stop; speculative activation)
        qpre = self.q
        gl = qpre - self._lo
        gh = self._hi - qpre
        near_lower = gl <= gh
        sign_l = torch.where(near_lower, torch.ones_like(gl),
                             -torch.ones_like(gl))
        gap_l = torch.where(near_lower, gl, gh)
        reach = h * self._cap
        active_l = gap_l < reach
        bias_l = torch.where(gap_l >= 0.0, -gap_l * hinv,
                             (BETA_WELD * hinv) * -gap_l)

        # -- effective masses: one multi-RHS solve against this factor --------
        # RHS layout: [rhs(1) | Jw^T(12) | Jc^T(55) | e_j(29)]
        R_RHS, R_W, R_C, R_E = 0, 1, 1 + N_WELD, 1 + N_WELD + NUM_PAIRS
        NRHS = 1 + N_WELD + NUM_PAIRS + NQ
        B = torch.zeros(n, NV, NRHS, device=dev, dtype=dt)
        B[:, :, R_RHS] = rhs
        B[:, :, R_W:R_W + N_WELD] = Jw.transpose(1, 2)
        B[:, :, R_C:R_C + NUM_PAIRS] = Jc.transpose(1, 2)
        ar = torch.arange(NQ, device=dev)
        B[:, VOFF + ar, R_E + ar] = 1.0
        X = self.chol_solve_multi(self.L, B, NV)
        sol_rhs = X[:, :, R_RHS]
        minv_w = X[:, :, R_W:R_W + N_WELD].transpose(1, 2)   # (N,12,35)
        minv_c = X[:, :, R_C:R_C + NUM_PAIRS].transpose(1, 2)  # (N,55,35)
        ecols = X[:, :, R_E:R_E + NQ].transpose(1, 2)        # (N,29,35) M^-1 e_j

        # 6. unconstrained velocity v* = v + h*M^-1(tau - C)
        v = self.v + h * sol_rhs

        # contact effective masses (accumulate over i, Rust dot order)
        diag_c = torch.zeros(n, NUM_PAIRS, device=dev, dtype=dt)
        for i in range(NV):
            diag_c = diag_c + Jc[:, :, i] * minv_c[:, :, i]

        # weld block A_ww = J_w M^-1 J_w^T; Rust computes entry (r, c<=... )
        # = sum_i J_row_c[i] * minvjt_row_r[i] and mirrors — the Cholesky
        # only reads the lower triangle, so feed acc^T (acc[r][c] = Rust's
        # upper-value) to place those bits in the lower triangle.
        acc = torch.zeros(n, N_WELD, N_WELD, device=dev, dtype=dt)
        for i in range(NV):
            acc = acc + minv_w[:, :, None, i] * Jw[:, None, :, i]
        Lww = self._chol12(acc.transpose(1, 2))

        # limit/cap directions: project e_j columns into the weld null space
        # (cap uses e_j; the limit row's signed column = sign * projected
        # column — a linear solve/projection commutes with the sign flip
        # bitwise, matching Rust's per-row computation on the signed row).
        plain_diag = ecols[:, ar, VOFF + ar]                   # (M^-1)_jj
        rw = torch.zeros(n, N_WELD, NQ, device=dev, dtype=dt)
        for i in range(NV):
            rw = rw + Jw[:, :, i].unsqueeze(-1) * ecols[:, :, i].unsqueeze(1)
        rw = self._chol12_solve(Lww, rw)
        pcols = ecols.clone()
        for w in range(N_WELD):
            pcols = pcols - minv_w[:, w, :].unsqueeze(1) * rw[:, w, :].unsqueeze(-1)
        proj_diag = pcols[:, ar, VOFF + ar]
        immobile = ~(proj_diag > COMPOSITE_EPS * plain_diag)
        inf = torch.full_like(proj_diag, float("inf"))
        capmass = torch.where(immobile, inf, proj_diag)
        # limit rows: diag replaced only when welds active (always here)
        diag_l = capmass  # sign*sign folds away: same value as Rust's dnew

        # -- step 8: 12 iterations, fixed order --------------------------------
        lam_c = torch.zeros(n, NUM_PAIRS, device=dev, dtype=dt)
        lam_l = torch.zeros(n, NQ, device=dev, dtype=dt)
        # active-contact union across the batch (one host sync per substep)
        union = torch.nonzero(active_c.any(0)).flatten().tolist()
        act_c = active_c.to(dt)
        act_l = active_l.to(dt)
        if getattr(self, "_debug", False):
            self._dbg_iter = []
        for _it in range(PGS_ITERS):
            # (a) weld block: delta = A_ww^-1 (b_w - J_w v), applied row-major
            jv = torch.zeros(n, N_WELD, device=dev, dtype=dt)
            for i in range(NV):
                jv = jv + Jw[:, :, i] * v[:, i].unsqueeze(-1)
            dvec = self._chol12_solve(Lww, (bias_w - jv).unsqueeze(-1))[:, :, 0]
            for rr in range(N_WELD):
                v = v + minv_w[:, rr, :] * dvec[:, rr].unsqueeze(-1)
            # (b) contact rows, pair-id order, projected GS lambda >= 0
            for pid in union:
                jr = Jc[:, pid, :]
                jvp = torch.zeros(n, device=dev, dtype=dt)
                for i in range(NV):
                    jvp = jvp + jr[:, i] * v[:, i]
                dl = (bias_c[:, pid] - jvp) / diag_c[:, pid]
                newl = torch.clamp(lam_c[:, pid] + dl, min=0.0)
                dd = (newl - lam_c[:, pid]) * act_c[:, pid]
                lam_c[:, pid] = torch.where(active_c[:, pid], newl,
                                            lam_c[:, pid])
                v = v + minv_c[:, pid, :] * dd.unsqueeze(-1)
            # (c) limit rows, joint order (sparse J = sign * e_j)
            for j in range(NQ):
                jvp = sign_l[:, j] * v[:, VOFF + j]
                dl = (bias_l[:, j] - jvp) / diag_l[:, j]
                newl = torch.clamp(lam_l[:, j] + dl, min=0.0)
                dd = (newl - lam_l[:, j]) * act_l[:, j]
                lam_l[:, j] = torch.where(active_l[:, j], newl, lam_l[:, j])
                # column = sign * projected e_j column
                v = v + pcols[:, j, :] * (sign_l[:, j] * dd).unsqueeze(-1)
            # (d) omega-cap pass, joint order — weld-consistent impulses
            for j in range(NQ):
                vj = v[:, VOFF + j]
                cap = OMEGA_CAP[j]
                target = vj.clamp(-cap, cap)
                need = (target != vj) & ~immobile[:, j]
                lam = torch.where(need, (target - vj) / capmass[:, j],
                                  torch.zeros_like(vj))
                v = v + pcols[:, j, :] * lam.unsqueeze(-1)
                v[:, VOFF + j] = torch.where(need, target, v[:, VOFF + j])
            if getattr(self, "_debug", False):
                self._dbg_iter.append(v.clone())

        self.v = v
        # cache contact results for the env layer (this substep): lambda
        # rows for inactive pairs were never touched (stay exactly 0)
        self._sub_contacts = (active_c, lam_c, point_c)
        if getattr(self, "_debug", False):
            self._dbg = dict(Jw=Jw, bias_w=bias_w, Jc=Jc, bias_c=bias_c,
                             active_c=active_c, sign_l=sign_l, gap_l=gap_l,
                             active_l=active_l, bias_l=bias_l, diag_c=diag_c,
                             minv_w=minv_w, minv_c=minv_c, ecols=ecols,
                             pcols=pcols, plain_diag=plain_diag,
                             proj_diag=proj_diag, capmass=capmass,
                             lam_c=lam_c, lam_l=lam_l, v_solved=v.clone())

        # 10. semi-implicit Euler positions + root exp-map quaternion
        self.q = self.q + h * self.v[:, VOFF:]
        self.p = self.p + h * self.v[:, :3]
        u0 = self.v[:, 3] * h * 0.5
        u1 = self.v[:, 4] * h * 0.5
        u2 = self.v[:, 5] * h * 0.5
        un = tsqrt(u0 * u0 + u1 * u1 + u2 * u2)
        small = un < 1e-12
        un_safe = torch.where(small, torch.ones_like(un), un)
        su = tsin(un)
        cu = tcos(un)
        one = torch.ones_like(un)
        zero = torch.zeros_like(un)
        qe = (torch.where(small, one, cu),
              torch.where(small, zero, su * u0 / un_safe),
              torch.where(small, zero, su * u1 / un_safe),
              torch.where(small, zero, su * u2 / un_safe))
        qq = quat_mul(qe, (self.quat[:, 0], self.quat[:, 1], self.quat[:, 2],
                           self.quat[:, 3]))
        qn = tsqrt(qq[0] * qq[0] + qq[1] * qq[1] + qq[2] * qq[2] + qq[3] * qq[3])
        self.quat = torch.stack([qq[0] / qn, qq[1] / qn, qq[2] / qn,
                                 qq[3] / qn], -1)

        # 11. post-solve clamps as momentum-consistent impulses
        hit = (self.q < self._lo) | (self.q > self._hi)
        self.q = torch.max(torch.min(self.q, self._hi), self._lo)
        v = self.v
        if getattr(self, "_debug", False):
            self._dbg_pass = []
        for _p in range(CLAMP_PASSES):
            for j in range(NQ):
                vj = v[:, VOFF + j]
                cap = OMEGA_CAP[j]
                target = torch.where(hit[:, j], torch.zeros_like(vj),
                                     vj.clamp(-cap, cap))
                need = target != vj
                col = ecols[:, j, :]
                lam = torch.where(need, (target - vj) / col[:, VOFF + j],
                                  torch.zeros_like(vj))
                v = v + col * lam.unsqueeze(-1)
                v[:, VOFF + j] = torch.where(need, target, v[:, VOFF + j])
            if getattr(self, "_debug", False):
                self._dbg_pass.append(v.clone())
        for j in range(NQ):
            cap = OMEGA_CAP[j]
            vj = v[:, VOFF + j].clamp(-cap, cap)
            v[:, VOFF + j] = torch.where(hit[:, j], torch.zeros_like(vj), vj)

        # 12. root velocity envelope guarantee
        w2 = v[:, 3] * v[:, 3] + v[:, 4] * v[:, 4] + v[:, 5] * v[:, 5]
        wbad = w2 > ROOT_W_MAX * ROOT_W_MAX
        # tensor/tensor division (scalar/tensor is 1-ulp-unsafe, see above)
        ws = torch.full_like(w2, ROOT_W_MAX) / tsqrt(
            torch.where(wbad, w2, torch.ones_like(w2)))
        for i in (3, 4, 5):
            v[:, i] = torch.where(wbad, v[:, i] * ws, v[:, i])
        l2 = v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2]
        lbad = l2 > ROOT_V_MAX * ROOT_V_MAX
        ls = torch.full_like(l2, ROOT_V_MAX) / tsqrt(
            torch.where(lbad, l2, torch.ones_like(l2)))
        for i in (0, 1, 2):
            v[:, i] = torch.where(lbad, v[:, i] * ls, v[:, i])
        self.v = v

    # 12x12 weld-block Cholesky helpers (same bitwise forms as above;
    # f64 uses the §22 floor too — bitwise no-op for healthy pivots,
    # and the .cu chol_factor_n template serves both call sites)
    def _chol12(self, A):
        W = A.clone()
        L = torch.zeros_like(W)
        exact = self.dtype == torch.float64
        for k in range(N_WELD):
            dcol = W[:, k, k]
            if exact:
                dj = tsqrt(torch.clamp(dcol, min=self.PD_FLOOR_F64))
            else:
                dj = tsqrt(torch.clamp(dcol, min=self.PD_FLOOR_F32))
            L[:, k, k] = dj
            if k + 1 < N_WELD:
                col = W[:, k + 1:, k] / dj.unsqueeze(-1)
                L[:, k + 1:, k] = col
                W[:, k + 1:, k + 1:] = (W[:, k + 1:, k + 1:]
                                        - col.unsqueeze(-1) * col.unsqueeze(-2))
        return L

    @staticmethod
    def _chol12_solve(L, B):
        return DynBatch.chol_solve_multi(L, B, N_WELD)

    pd_net_hits = 0  # pivot-floor engagements (reported, not hidden)

    def check_pd(self):
        """The pivot floor (f32 §20e / f64 §22) already absorbed any PD
        loss inside chol_factor — count engaged envs and continue, both
        dtypes (Rust no longer panics either: algo.rs
        chol_factor_floored counts into ws.pd_hits, popped via
        pop_pd_hits). One host sync per call."""
        if bool(self._pd_bad.any().item()):
            self.pd_net_hits += int(self._pd_bad.sum().item())
            self._pd_bad.zero_()

    # -- derived caches for the env layer -------------------------------------
    def refresh_fk(self):
        """lib.rs Body::refresh_fk — wrist/blade/head/shoulder caches
        (dense (N,·) tensors)."""
        self.fk_all()
        ref = self.p[:, 0]
        wr, bl = [], []
        for fw, fh in ((F_L_WRIST, F_L_HAND), (F_R_WRIST, F_R_HAND)):
            wr.append(torch.stack(_dense3(self.pos[fw], ref), -1))
            r = self.r[fh]
            bl.append(torch.stack(_dense3([r[0][2], r[1][2], r[2][2]], ref), -1))
        self.wrist = torch.stack(wr, 1)   # (N, 2, 3)
        self.blade = torch.stack(bl, 1)   # (N, 2, 3)
        self.head = torch.stack(_dense3(self.pos[F_HEAD], ref), -1)  # (N, 3)
        self.shoulder_z = torch.stack(
            [_dense3(self.pos[F_L_SHOULDER], ref)[2],
             _dense3(self.pos[F_R_SHOULDER], ref)[2]], -1)  # (N, 2)
