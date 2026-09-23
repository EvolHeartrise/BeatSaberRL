"""Arm / full-body forward kinematics — v10 since P6.

P6 (2026-07-31) adds the v10 FULL-BODY layer on top of the v9 arm
math:

- ``fk_body(q36)``: full-tree FK over the SPEC-v10 §1 frame table
  (mirror of rust/src/dynamics/model.rs + fk.rs) — 30 frames plus the
  free base; wrists = hand-frame origins, blades = hand-frame +z.

The v9 arm-chain FK below remains the continuity reference
(SPEC-v10 §1.3 carries the arm stacking orders verbatim): at the
reference root pose with spine 0 they agree with fk_body to 1e-12.

--- v9 doc (arm chains) ---

Joint order per hand: q = [pitch, yaw, twist, elbow, roll, bend, dev].
All-zero = arm straight forward (+z), blade the forearm's extension.

Frames: arm frame M = R_y(yaw) . R_x(-pitch) . R_z(twist) (local +z =
upper arm; pitch + = up, yaw + = toward +x). Elbow flexion rotates about
the twist-oriented local x and curls toward local +y:
forearm = M . (0, sin e, cos e). The blade lives in the forearm frame
M_f = M . R_x(-elbow); env v9 (user-caught gap) adds radial/ulnar
DEVIATION as the second wrist bending axis:
d(roll, bend, dev) = Rz(roll) . Ry(dev) . Rx(-bend) . e_z =
(cos r sin v cos b - sin r sin b,
 sin r sin v cos b + cos r sin b,
 cos v cos b).
dev = 0 reproduces the v5.2-v8 wrist exactly; at bend = 0 deviation
moves the blade at full rate (the old roll+bend parametrization had
out-of-plane speed roll_rate*sin(bend) — a pole singularity at a
straight wrist).

Used by the scenario tests, gen_crouch_table.py and the viewer mirror.
(The closed-form arm IK — ik_arm / wrist_for_blade — was removed
2026-09-22 with the scripted reference bot, its only user; so were the
bot's gravity/stance feedforward helpers — gravity_comp, stance_comp,
potential_energy, com_of.)
"""

from __future__ import annotations

import math

from .constants import L_FORE, L_UPPER, SHOULDER


def arm_matrix(pitch, yaw, twist):
    """The 3x3 arm frame M = R_y(yaw) . R_x(-pitch) . R_z(twist)."""
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    st, ct = math.sin(twist), math.cos(twist)
    return (
        (cy * ct - sy * sp * st, -cy * st - sy * sp * ct, sy * cp),
        (cp * st, cp * ct, sp),
        (-sy * ct - cy * sp * st, sy * st - cy * sp * ct, cy * cp),
    )


def fk(hand, q):
    """(elbow, wrist, blade unit dir) for one hand — mirrors Rust fk()."""
    m = arm_matrix(q[0], q[1], q[2])
    se, ce = math.sin(q[3]), math.cos(q[3])
    sh = SHOULDER[hand]
    e = tuple(sh[i] + L_UPPER * m[i][2] for i in range(3))
    f = tuple(m[i][1] * se + m[i][2] * ce for i in range(3))
    w = tuple(e[i] + L_FORE * f[i] for i in range(3))
    sr, cr = math.sin(q[4]), math.cos(q[4])
    sb, cb = math.sin(q[5]), math.cos(q[5])
    sv, cv = ((math.sin(q[6]), math.cos(q[6])) if len(q) > 6
              else (0.0, 1.0))
    d = (cr * sv * cb - sr * sb, sr * sv * cb + cr * sb, cv * cb)
    b = tuple(
        m[i][0] * d[0] + (m[i][1] * ce - m[i][2] * se) * d[1] + f[i] * d[2]
        for i in range(3))
    return e, w, b


# =============================================================================
# v10 full-body layer (P6) — mirror of rust/src/dynamics/model.rs +
# fk.rs. Frame table transcribed from the one-source-of-truth Rust
# tables (SPEC-v10 §1.2–1.3); chain (stacking) order, `coord` maps a
# frame to its ACTUATED index (§1.4). parent -1 = the free base
# (pelvis).
# =============================================================================

ROOT_POS0 = (0.0, 0.94, -0.65)

_AX = (1.0, 0.0, 0.0)
_AXN = (-1.0, 0.0, 0.0)
_AY = (0.0, 1.0, 0.0)
_AZ = (0.0, 0.0, 1.0)
_Z3 = (0.0, 0.0, 0.0)

# (parent, anchor, axis, coord, mass, com) per frame — model.rs verbatim
# (inertia diagonals omitted; mass/COM kept for table parity with model.rs).
FRAMES = (
    # spine: pelvis -> chest, anchor at the waist
    (-1, (0.0, 0.10, 0.0), _AX, 0, 0.0, _Z3),          # 0 spine_pitch
    (0, _Z3, _AY, 1, 0.0, _Z3),                        # 1 spine_yaw
    (1, _Z3, _AZ, 2, 24.20, (0.0, 0.215, 0.0)),        # 2 spine_roll + chest
    (2, (0.0, 0.66, 0.0), _AZ, -1, 5.20, _Z3),         # 3 head (WELD)
    # left arm (stack yaw -> pitch -> twist)
    (2, (-0.20, 0.43, 0.0), _AY, 4, 0.0, _Z3),         # 4 l_sh_yaw
    (4, _Z3, _AXN, 3, 0.0, _Z3),                       # 5 l_sh_pitch
    (5, _Z3, _AZ, 5, 2.00, (0.0, 0.0, 0.175)),         # 6 l_sh_twist + uarm
    (6, (0.0, 0.0, 0.35), _AXN, 6, 1.20, (0.0, 0.0, 0.15)),  # 7 l_elbow
    # left wrist (stack roll -> dev -> bend)
    (7, (0.0, 0.0, 0.30), _AZ, 7, 0.0, _Z3),           # 8 l_wr_roll
    (8, _Z3, _AY, 9, 0.0, _Z3),                        # 9 l_wr_dev
    (9, _Z3, _AXN, 8, 0.61, (0.0, 0.0, 0.06)),         # 10 l_wr_bend + hand
    # right arm
    (2, (0.20, 0.43, 0.0), _AY, 11, 0.0, _Z3),         # 11 r_sh_yaw
    (11, _Z3, _AXN, 10, 0.0, _Z3),                     # 12 r_sh_pitch
    (12, _Z3, _AZ, 12, 2.00, (0.0, 0.0, 0.175)),       # 13 r_sh_twist + uarm
    (13, (0.0, 0.0, 0.35), _AXN, 13, 1.20, (0.0, 0.0, 0.15)),  # 14 r_elbow
    (14, (0.0, 0.0, 0.30), _AZ, 14, 0.0, _Z3),         # 15 r_wr_roll
    (15, _Z3, _AY, 16, 0.0, _Z3),                      # 16 r_wr_dev
    (16, _Z3, _AXN, 15, 0.61, (0.0, 0.0, 0.06)),       # 17 r_wr_bend + hand
    # left leg (stack pitch -> yaw -> roll = actuated)
    (-1, (-0.10, 0.0, 0.0), _AXN, 17, 0.0, _Z3),       # 18 l_hip_pitch
    (18, _Z3, _AY, 18, 0.0, _Z3),                      # 19 l_hip_yaw
    (19, _Z3, _AZ, 19, 10.60, (0.0, -0.22, 0.0)),      # 20 l_hip_roll + thigh
    (20, (0.0, -0.44, 0.0), _AX, 20, 3.20, (0.0, -0.215, 0.0)),  # 21 l_knee
    (21, (0.0, -0.43, 0.0), _AXN, 21, 0.0, _Z3),       # 22 l_ankle_pitch
    (22, _Z3, _AZ, 22, 1.00, (0.0, -0.035, 0.05)),     # 23 l_ankle_roll + foot
    # right leg
    (-1, (0.10, 0.0, 0.0), _AXN, 23, 0.0, _Z3),        # 24 r_hip_pitch
    (24, _Z3, _AY, 24, 0.0, _Z3),                      # 25 r_hip_yaw
    (25, _Z3, _AZ, 25, 10.60, (0.0, -0.22, 0.0)),      # 26 r_hip_roll + thigh
    (26, (0.0, -0.44, 0.0), _AX, 26, 3.20, (0.0, -0.215, 0.0)),  # 27 r_knee
    (27, (0.0, -0.43, 0.0), _AXN, 27, 0.0, _Z3),       # 28 r_ankle_pitch
    (28, _Z3, _AZ, 28, 1.00, (0.0, -0.035, 0.05)),     # 29 r_ankle_roll + foot
)

# frame indices (model.rs constants, verbatim)
F_CHEST, F_HEAD = 2, 3
F_L_SHOULDER, F_L_ELBOW, F_L_WRIST, F_L_HAND = 4, 7, 8, 10
F_R_SHOULDER, F_R_ELBOW, F_R_WRIST, F_R_HAND = 11, 14, 15, 17
F_L_FOOT, F_R_FOOT = 23, 29


def _rot_axis(axis, angle):
    """Rodrigues for a unit axis (mirror of fk.rs rot_axis)."""
    s, c = math.sin(angle), math.cos(angle)
    k = 1.0 - c
    x, y, z = axis
    return (
        (c + k * x * x, k * x * y - s * z, k * x * z + s * y),
        (k * x * y + s * z, c + k * y * y, k * y * z - s * x),
        (k * x * z - s * y, k * y * z + s * x, c + k * z * z),
    )


def _quat_to_rot(q):
    """World-from-body rotation of a wxyz quaternion (fk.rs mirror)."""
    w, x, y, z = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def _mat_mul(a, b):
    return tuple(
        tuple(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
              for j in range(3))
        for i in range(3))


def fk_body(q36):
    """Full-tree FK from the replay-v10 position vector q36 = [root
    pos(3), root quat wxyz(4), 29 joint angles §1.4 order].

    Returns (rots, pos): per-frame world rotation (3x3 tuples) and
    world origin, in FRAMES order (30 entries). Wrist positions =
    pos[F_L_WRIST]/pos[F_R_WRIST]; blade dirs = column z of
    rots[F_L_HAND]/rots[F_R_HAND]; head = pos[F_HEAD]."""
    p0 = tuple(q36[0:3])
    r0 = _quat_to_rot(q36[3:7])
    q = q36[7:]
    rots, pos = [], []
    for parent, anchor, axis, coord, _m, _c in FRAMES:
        rl, pl = (r0, p0) if parent < 0 else (rots[parent], pos[parent])
        pos.append(tuple(pl[i] + sum(rl[i][k] * anchor[k] for k in range(3))
                         for i in range(3)))
        if coord >= 0:
            rots.append(_mat_mul(rl, _rot_axis(axis, q[coord])))
        else:
            rots.append(rl)
    return rots, pos


def blade_of(rots, hand):
    """Unit blade direction (hand-frame +z column) for hand 0/1."""
    r = rots[F_L_HAND if hand == 0 else F_R_HAND]
    return (r[0][2], r[1][2], r[2][2])
