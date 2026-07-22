"""Arm forward/inverse kinematics (env v6) — MIRROR of rust/src/lib.rs fk().

Joint order per hand: q = [pitch, yaw, twist, elbow, roll, bend].
All-zero = arm straight forward (+z), blade the forearm's extension.

Frames: arm frame M = R_y(yaw) . R_x(-pitch) . R_z(twist) (local +z =
upper arm; pitch + = up, yaw + = toward +x). Elbow flexion rotates about
the twist-oriented local x and curls toward local +y:
forearm = M . (0, sin e, cos e). The blade is the v5.2 wrist in the
forearm frame M_f = M . R_x(-elbow): d(roll, bend) =
(-sin(roll) sin(bend), cos(roll) sin(bend), cos(bend)).

Used by the scripted bot, the scenario tests, and the viewer mirror.
"""

from __future__ import annotations

import math

from .constants import (
    ELBOW_HI, L_FORE, L_UPPER, SHOULDER, TWIST_HI, TWIST_LO,
)


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
    d = (-sr * sb, cr * sb, cb)
    b = tuple(
        m[i][0] * d[0] + (m[i][1] * ce - m[i][2] * se) * d[1] + f[i] * d[2]
        for i in range(3))
    return e, w, b


def _norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def ik_arm(hand, wrist_target, swivel=0.0):
    """Closed-form (pitch, yaw, twist, elbow) putting the wrist at (or as
    near as reachable to) `wrist_target`.

    Two-link IK: the elbow sits on a circle around the shoulder->wrist
    axis; `swivel` picks the point on it (0 = the elbow pushed toward
    world -y, i.e. hanging down — the natural human pose; the string
    "back" = the elbow pushed toward -z, which keeps the FOREARM pointing
    as far forward as the pose allows — the scripted bot's choice, since
    roll-swings sweep in the plane perpendicular to the forearm). The
    required twist is solved exactly; if it exceeds TWIST_LIM the caller
    gets the clamped-arm answer (fk(result) then tells the truth).
    """
    sh = SHOULDER[hand]
    to = [wrist_target[i] - sh[i] for i in range(3)]
    r = _norm(to)
    r = min(max(r, abs(L_UPPER - L_FORE) + 1e-6), L_UPPER + L_FORE - 1e-6)
    u = [to[i] / _norm(to) for i in range(3)] if _norm(to) > 1e-9 else [0, 0, 1]

    # law of cosines: interior elbow angle; flexion = pi - interior
    cos_int = (L_UPPER ** 2 + L_FORE ** 2 - r ** 2) / (2 * L_UPPER * L_FORE)
    interior = math.acos(max(-1.0, min(1.0, cos_int)))
    elbow = math.pi - interior
    elbow = min(max(elbow, 0.0), ELBOW_HI)

    # elbow position: distance a along u, then radius h on the swivel circle
    a = (L_UPPER ** 2 - L_FORE ** 2 + r ** 2) / (2 * r)
    h = math.sqrt(max(0.0, L_UPPER ** 2 - a ** 2))
    # circle basis: n1 = normalize(u x world_y-ish) etc.; swivel 0 = -y side
    ref = (0.0, 1.0, 0.0) if abs(u[1]) < 0.99 else (1.0, 0.0, 0.0)
    n1 = [u[1] * ref[2] - u[2] * ref[1],
          u[2] * ref[0] - u[0] * ref[2],
          u[0] * ref[1] - u[1] * ref[0]]           # u x ref (horizontal-ish)
    n1n = _norm(n1)
    n1 = [v / n1n for v in n1]
    n2 = [u[1] * n1[2] - u[2] * n1[1],
          u[2] * n1[0] - u[0] * n1[2],
          u[0] * n1[1] - u[1] * n1[0]]             # u x n1 (points -y-ish)
    if swivel == "back":  # minimize the elbow's z on the circle
        swivel = math.atan2(-n1[2], -n2[2])
    cs, ss = math.cos(swivel), math.sin(swivel)
    e = [sh[i] + a * u[i] + h * (cs * n2[i] + ss * n1[i]) for i in range(3)]

    # shoulder pitch/yaw aim the upper arm at the elbow
    ud = [(e[i] - sh[i]) / L_UPPER for i in range(3)]
    pitch = math.asin(max(-1.0, min(1.0, ud[1])))
    yaw = math.atan2(ud[0], ud[2])

    # twist so that elbow flexion reaches the wrist: forearm dir in the
    # untwisted arm frame A = R_y(yaw) . R_x(-pitch) must equal
    # R_z(twist) . (0, sin e, cos e)
    fd = [(wrist_target[i] - e[i]) for i in range(3)]
    fdn = _norm(fd)
    fd = [v / fdn for v in fd] if fdn > 1e-9 else ud
    A = arm_matrix(pitch, yaw, 0.0)
    g = [sum(A[k][i] * fd[k] for k in range(3)) for i in range(3)]  # A^T fd
    se = math.hypot(g[0], g[1])
    if se < 1e-9:
        twist = 0.0
    else:
        twist = math.atan2(-g[0], g[1])
    twist = min(max(twist, TWIST_LO[hand]), TWIST_HI[hand])
    return [pitch, yaw, twist, elbow]


def wrist_for_blade(hand, q_arm, blade_target):
    """(roll, bend) pointing the blade along world `blade_target` (unit),
    given the arm pose. Inverts d(roll, bend) in the forearm frame; the
    caller clamps to the roll/bend limits."""
    m = arm_matrix(q_arm[0], q_arm[1], q_arm[2])
    se, ce = math.sin(q_arm[3]), math.cos(q_arm[3])
    # forearm frame columns: c0 = M col0, c1 = ce*Mcol1 - se*Mcol2, c2 = f
    c0 = [m[i][0] for i in range(3)]
    c1 = [m[i][1] * ce - m[i][2] * se for i in range(3)]
    c2 = [m[i][1] * se + m[i][2] * ce for i in range(3)]
    l = [sum(c[i] * blade_target[i] for i in range(3)) for c in (c0, c1, c2)]
    bend = math.acos(max(-1.0, min(1.0, l[2])))
    if math.hypot(l[0], l[1]) < 1e-9:
        roll = 0.0
    else:
        roll = math.atan2(-l[0], l[1])
    return roll, bend
