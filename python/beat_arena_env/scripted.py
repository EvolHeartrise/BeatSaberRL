"""Scripted reference bot — v9-LEGACY (BROKEN against the v10 sim).

Env v10 (P3, 2026-07-31) retired the kinematic body this bot drives:
the sim's action space is now 29 gear heads and `sim.sabers()` returns
FK wrist+blade only (no per-hand joint state). P6 re-authors the bot as
gravity-compensated PD-over-IK on the dynamic body; until then
`scripted_action` raises, and its tests are skipped pending P6.

--- original v6-v9 doc ---

The blade-level plan is unchanged from v5.2: for its color's earliest
pending note, plant so the blade lies in the hit plane perpendicular to
the required cut direction, cock back up to 130 degrees, then whip
through (the leftover rotation is the follow-through). What changed is
the body: the wrist is no longer a free point — the bot solves
closed-form two-link IK for the four arm joints (shoulder pitch/yaw/
twist + elbow) to park the wrist as near the goal as the arm reaches,
and converts desired world blade directions into (roll, bend) targets in
the live forearm frame (kinematics.wrist_for_blade). Every joint is then
steered independently: desired omega = gain * error, bang-bang action
toward it — legal because v6 joints are force-free independent
integrators. Stateless per tick; solvability proof and eyeball baseline,
not optimal.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    ALPHA_WRIST, DRAG, DT, JOINT_ALPHA, JOINT_OMEGA, KIND_BOMB,
    KIND_CHAIN_LINK, L_FORE, L_UPPER, OMEGA_WRIST, SABER_LEN, SHOULDER,
    joint_limits, note_center,
)
from .kinematics import fk, ik_arm, wrist_for_blade

CONTACT_S = 0.75          # note passes the blade at this fraction of its length
SWING_ARC = np.radians(130.0)  # ideal wind-up, past the 100 deg full credit
FOLLOW_AIM = np.radians(70.0)  # whip target past contact (follow-through room)
REACH = L_UPPER + L_FORE - 0.02   # usable wrist-reach radius around a shoulder
K_JOINT = 4.0             # per-joint arrival gain (1/s)
K_WRIST = 10.0            # wrist roll/bend tracking gain (1/s)
# env v7: shoulders sit 0.65 m behind the hit plane, so a wrist ON the
# plane is at full stretch — plant it here instead. The roll-swing sweeps
# in a plane near the wrist's z; the good hitbox (note_z - 0.75 ..
# note_z + 0.25) contains z = -0.25 while the note is near the plane.
PLANT_Z = -0.25
# whip THROUGH the contact angle this early: the elongated hitbox reaches
# the blade zone ~0.15 s before hit_time, so a plane-timed swing gets
# caught mid-wind-up ('angle'/'slow'). Cutting early inside the deep box
# is exactly what real players do.
EARLY_S = 0.10

# idle pose per hand: arm dropped and out, elbow half bent, blade up
PARK_Q = {
    0: np.array([-0.5, -0.7, 0.0, 0.8]),
    1: np.array([-0.5, 0.7, 0.0, 0.8]),
}


def _steer(q, om, target, j):
    """Ternary action driving joint j's angle toward `target`."""
    desired = np.clip(K_JOINT * (target - q), -JOINT_OMEGA[j], JOINT_OMEGA[j])
    dw = np.array([0.0, JOINT_ALPHA[j] * DT, -JOINT_ALPHA[j] * DT])
    om_next = np.clip((om + dw) * (1.0 - DRAG),
                      -JOINT_OMEGA[j], JOINT_OMEGA[j])
    return int(np.argmin(np.abs(om_next - desired)))


def _steer_wrist(q, om, target, j, gain=K_WRIST):
    desired = np.clip(gain * (target - q), -OMEGA_WRIST, OMEGA_WRIST)
    dw = np.array([0.0, ALPHA_WRIST * DT, -ALPHA_WRIST * DT])
    om_next = np.clip((om + dw) * (1.0 - DRAG), -OMEGA_WRIST, OMEGA_WRIST)
    return int(np.argmin(np.abs(om_next - desired)))


def _roll_reps(hand, roll, bend):
    """Both in-range representatives of a blade direction:
    d(roll, bend) = d(roll + pi, -bend), rolls 2*pi-shifted into the
    hand's 280-degree range. Deterministic (no dependence on the current
    state), so steering targets never thrash. Returns 1-2 (roll, bend)
    pairs; positive-bend rep first when both fit."""
    lo, hi = joint_limits(hand)[4]
    if bend < 0.0:
        roll, bend = roll + np.pi, -bend
    reps = []
    for r0, b0 in ((roll, bend), (roll + np.pi, -bend)):
        for k in (0.0, 2 * np.pi, -2 * np.pi):
            if lo - 1e-9 <= r0 + k <= hi + 1e-9:
                reps.append((float(np.clip(r0 + k, lo, hi)), b0))
                break
    return reps or [(float(np.clip(roll, lo, hi)), bend)]


def _fit_roll(hand, roll, bend):
    """First (preferred) in-range representative — see _roll_reps."""
    return _roll_reps(hand, roll, bend)[0]


def _swing_time(arc):
    """Seconds to sweep `arc` from rest at full wrist accel (+ margin)."""
    if arc <= OMEGA_WRIST ** 2 / (2 * ALPHA_WRIST):
        return float(np.sqrt(2 * arc / ALPHA_WRIST)) + 0.04
    return OMEGA_WRIST / (2 * ALPHA_WRIST) + arc / OMEGA_WRIST + 0.04


def _cut_config(hand, center3, angle):
    """(sense, theta_c, wrist_goal3) — blade perpendicular to the cut
    direction at contact, wrist ON the blade line through the note.

    For each swing sense the wrist goal slides along the ray
    center - s * (cos th_c, sin th_c, 0), planted at z = PLANT_Z: the
    search picks the s whose goal the arm can reach (nearest
    s = CONTACT_S wins), so the blade still passes through the note even
    when the ideal plant is out of reach. Candidates ranked by leftover
    overreach."""
    sh = np.array(SHOULDER[hand])
    plant = np.array([0.0, 0.0, PLANT_Z])
    best = None
    for sense, th_c in ((1.0, angle - np.pi / 2), (-1.0, angle + np.pi / 2)):
        ray = np.array([np.cos(th_c), np.sin(th_c), 0.0])
        s_best, over_best = CONTACT_S, np.inf
        for s in np.linspace(0.3, 0.95, 14):
            over = max(0.0, float(
                np.linalg.norm(center3 - s * ray + plant - sh)) - REACH)
            key = (over, abs(s - CONTACT_S))
            if key < (over_best, abs(s_best - CONTACT_S)):
                s_best, over_best = float(s), over
        if best is None or over_best < best[0]:
            best = (over_best, sense, th_c, center3 - s_best * ray + plant)
    _, sense, th_c, goal = best
    return sense, th_c, goal


def _hand_action(hand, arm, time, target):
    # env v9: 7 joints/hand — the bot drives roll+bend and PARKS the
    # deviation joint at 0 (its null DOF; wrist_for_blade solves dev=0)
    q, om = np.asarray(arm[:7]), np.asarray(arm[7:14])

    if target is None:
        acts = [_steer(q[j], om[j], PARK_Q[hand][j], j) for j in range(4)]
        r_t, b_t = _fit_roll(hand, *wrist_for_blade(
            hand, q[:4], (0.0, 1.0, 0.0)))                    # blade up
        acts.append(_steer_wrist(q[4], om[4], r_t, 4))
        acts.append(_steer_wrist(q[5], om[5], b_t, 5))
        acts.append(_steer_wrist(q[6], om[6], 0.0, 6))
        return acts

    center3 = np.array([*note_center(target), 0.0])
    # chain links carry their tangent as `angle`: swinging along the
    # chain's path sweeps through consecutive links naturally. Plain dot
    # notes get a default down-swing.
    if not target.any_dir or target.kind == KIND_CHAIN_LINK:
        angle = target.angle
    else:
        angle = 3 * np.pi / 2
    _, _, goal = _cut_config(hand, center3, angle)
    q_arm = ik_arm(hand, goal, swivel="back")
    acts = [_steer(q[j], om[j], q_arm[j], j) for j in range(4)]

    # re-aim off the pose the arm will actually reach: contact blade
    # orientation = wrist -> note (exact when the plant is reachable,
    # best-effort otherwise); the 60-degree cut tolerance absorbs the
    # residual between this and the ideal perpendicular plant.
    _, w_pred, _ = fk(hand, list(q_arm) + [0.0, 0.0, 0.0])
    th_aim = float(np.arctan2(center3[1] - w_pred[1],
                              center3[0] - w_pred[0]))
    # swing plan: over BOTH wrist representations of the contact blade
    # direction (roll +- pi with flipped bend reach the same blade) and
    # both sweep senses, require the sweep velocity within ~55 deg of the
    # cut direction (hard cap — past the 60-deg rule it's a guaranteed
    # angle-fail), then rank by roll room: wind-up arc behind the contact
    # roll, >= 45 deg of follow-through beyond it (v5.2's room ranking,
    # in roll space).
    lo, hi = joint_limits(hand)[4]
    err = lambda s: abs(np.angle(np.exp(1j * (angle - th_aim - s * np.pi / 2))))
    reps = _roll_reps(hand, *wrist_for_blade(
        hand, list(q[:4]),
        (float(np.cos(th_aim)), float(np.sin(th_aim)), 0.0)))
    best = None
    for r_c, b_c in reps:
        for sense in (1.0, -1.0):
            back = (r_c - lo) if sense > 0 else (hi - r_c)
            fwd = (hi - r_c) if sense > 0 else (r_c - lo)
            full = back >= SWING_ARC and fwd >= np.radians(45.0)
            key = (err(sense) > np.radians(55.0), not full,
                   -min(fwd, np.radians(45.0)), err(sense))
            if best is None or key < best[0]:
                best = (key, sense, back, r_c, b_c)
    _, sense, back, r_c, b_c = best
    arc = float(np.clip(back - np.radians(5.0), np.radians(20.0), SWING_ARC))

    t_until = target.hit_time - EARLY_S - time
    if t_until > _swing_time(arc):
        # wind-up: cock the roll `arc` behind the contact roll (stable
        # target — same representative every tick)
        acts.append(_steer_wrist(q[4], om[4], r_c - sense * arc, 4))
        acts.append(_steer_wrist(q[5], om[5], b_c, 5))
        acts.append(_steer_wrist(q[6], om[6], 0.0, 6))
        return acts

    # whip (v5.2 semantics): FULL roll accel in the swing sense until the
    # note resolves — timed so the blade sweeps THROUGH th_aim at contact;
    # the leftover rotation is the follow-through. +roll sweeps the blade
    # CCW in the forearm frame (the back-swivel IK keeps the forearm
    # pointing at the notes, so the world sense matches). Bend holds its
    # contact value.
    acts.append(1 if sense > 0 else 2)
    acts.append(_steer_wrist(q[5], om[5], b_c, 5))
    acts.append(_steer_wrist(q[6], om[6], 0.0, 6))
    return acts


def scripted_action(env):
    """v9-LEGACY (raises): the v9 bot drove per-hand kinematic joints;
    the v10 sim exposes neither the API nor the physics it assumed.
    P6 replaces this with the PD-over-IK bot on the dynamic body."""
    raise NotImplementedError(
        "scripted_action is v9-legacy: the v10 dynamic body needs the "
        "P6 gravity-compensated PD-over-IK bot")


def _scripted_action_v9(env):
    """The retired v9 implementation, kept for the P6 rewrite to mine."""
    status = env.sim.status()
    targets = {0: None, 1: None}
    for i, n in enumerate(env.chart):
        if n.kind == KIND_BOMB:            # bombs are never targets
            continue
        if status[i] == 0 and targets[n.color] is None:
            targets[n.color] = n
        if targets[0] is not None and targets[1] is not None:
            break
    arms = env.sim.sabers()
    time = env.sim.time
    acts = []
    for hand in (0, 1):
        acts.extend(_hand_action(hand, np.asarray(arms[hand]), time,
                                 targets[hand]))
    return tuple(acts)
