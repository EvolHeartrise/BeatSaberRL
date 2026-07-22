"""Scripted reference bot (SPEC.md §Scripted reference bot) — env v5.1.

Real swings under anatomical wrist limits: per hand, the bot fits a cut
orientation inside the hand's bounded azimuth range (at least one of the
two perpendicular orientations always fits a 280-degree arc), plants its
wrist so the blade passes through the note perpendicular to the required
direction, winds back as far as the limits allow (up to 130 degrees),
then whips through. Stateless per tick; solvability proof and eyeball
baseline, not optimal play.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    ACCEL, ALPHA_ROT, DRAG, DT, OMEGA_MAX, SABER_LEN, THETA_HI, THETA_LO,
    note_center,
)

CONTACT_S = 0.75          # note passes the blade at this fraction of its length
SWING_ARC = np.radians(130.0)  # ideal wind-up, past the 100 deg full credit
FOLLOW_ROOM = np.radians(45.0) # wanted room past contact for follow-through
CRUISE = 4.0
K_ARRIVE = 4.0            # wrist arrival gain (1/s)
K_ROT = 10.0              # angular arrival gain (1/s)

# idle hands park OUTSIDE the grid with the blade vertical: a blade
# resting inside note territory gets grazed by arriving blocks
PARK = {0: np.array([-1.45, -0.6]), 1: np.array([1.45, -0.6])}

_DIRS = np.array([[np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)] for k in range(8)])
_ACTION_DV = np.vstack([np.zeros(2), ACCEL * DT * _DIRS])
_ROT_DW = np.array([0.0, ALPHA_ROT * DT, -ALPHA_ROT * DT])  # action 0/1/2


def _steer_to_velocity(vel, desired_v):
    """Pick the move action whose one-tick velocity update lands closest."""
    v_next = (vel + _ACTION_DV) * (1.0 - DRAG)
    return int(np.argmin(np.linalg.norm(v_next - desired_v, axis=1)))


def _steer_omega(om, desired_om):
    """Pick the rot/tilt action whose one-tick omega lands closest."""
    om_next = np.clip((om + _ROT_DW) * (1.0 - DRAG), -OMEGA_MAX, OMEGA_MAX)
    return int(np.argmin(np.abs(om_next - desired_om)))


def _fit_in_range(angle, lo, hi):
    """Shift angle by 2*pi*k into [lo, hi]; None if no k works."""
    for k in (0, 1, -1):
        a = angle + 2.0 * np.pi * k
        if lo - 1e-9 <= a <= hi + 1e-9:
            return float(np.clip(a, lo, hi))
    return None


def _swing_time(arc):
    """Seconds to sweep `arc` from rest at full accel (+ a small margin)."""
    if arc <= OMEGA_MAX ** 2 / (2 * ALPHA_ROT):
        return float(np.sqrt(2 * arc / ALPHA_ROT)) + 0.02
    return OMEGA_MAX / (2 * ALPHA_ROT) + arc / OMEGA_MAX + 0.02


def _cut_config(hand, center, angle, wrist):
    """(sense, theta_c, theta_start, wrist_goal) for a note, respecting
    the hand's azimuth limits.

    The blade must be perpendicular to the cut direction at contact; a
    CCW swing (sense +1) puts the contact point's velocity along the cut
    direction when theta_c = angle - 90 deg, a CW swing when +90 deg.
    Rank candidates: full swing room first, then wrist-goal in bounds,
    then nearest. At least one candidate always fits a 280-degree range.
    """
    lo, hi = THETA_LO[hand], THETA_HI[hand]
    best = None
    for sense, th_raw in ((1.0, angle - np.pi / 2), (-1.0, angle + np.pi / 2)):
        th_c = _fit_in_range(th_raw, lo, hi)
        if th_c is None:
            continue
        room_back = (th_c - lo) if sense > 0 else (hi - th_c)
        room_fwd = (hi - th_c) if sense > 0 else (th_c - lo)
        goal = center - CONTACT_S * SABER_LEN * np.array(
            [np.cos(th_c), np.sin(th_c)])
        in_b = -1.45 <= goal[0] <= 1.45 and -1.2 <= goal[1] <= 1.2
        full = room_back >= SWING_ARC and room_fwd >= FOLLOW_ROOM
        key = (not full, not in_b, float(np.linalg.norm(goal - wrist)))
        if best is None or key < best[0]:
            best = (key, sense, th_c, goal, room_back)
    _, sense, th_c, goal, room_back = best
    arc = min(SWING_ARC, max(room_back - np.radians(5), np.radians(20)))
    return sense, th_c, th_c - sense * arc, goal, arc


def _hand_action(hand, saber, time, target):
    pos, vel = saber[0:2], saber[2:4]
    theta, phi, om_th, om_ph = saber[4], saber[5], saber[6], saber[7]
    tilt_act = _steer_omega(om_ph, np.clip(K_ROT * -phi, -OMEGA_MAX, OMEGA_MAX))
    if target is None:
        to_park = PARK[hand] - pos
        dist = float(np.linalg.norm(to_park))
        desired = to_park / dist * min(CRUISE, K_ARRIVE * dist) if dist > 0.1 else np.zeros(2)
        # blade straight up, no spin (in range for both hands)
        rot_act = _steer_omega(om_th, np.clip(
            K_ROT * (np.pi / 2 - theta), -OMEGA_MAX, OMEGA_MAX))
        return (_steer_to_velocity(vel, desired), rot_act, tilt_act)

    center = np.array(note_center(target))
    # dot notes: any direction scores — pretend it's a down-cut
    angle = target.angle if not target.any_dir else 3 * np.pi / 2
    sense, th_c, th_start, goal, arc = _cut_config(hand, center, angle, pos)
    t_until = target.hit_time - time

    if t_until > _swing_time(arc):
        # wind-up: park the wrist at the goal, cock the blade back.
        # Angles are bounded (no wrap): steer by the DIRECT error — the
        # short way around could cross a limit.
        to_goal = goal - pos
        dist = float(np.linalg.norm(to_goal))
        speed = min(CRUISE, K_ARRIVE * dist)
        desired = to_goal / dist * speed if dist > 1e-6 else np.zeros(2)
        rot_act = _steer_omega(om_th, np.clip(
            K_ROT * (th_start - theta), -OMEGA_MAX, OMEGA_MAX))
        return (_steer_to_velocity(vel, desired), rot_act, tilt_act)

    # whip: hold the wrist, full angular accel in the swing sense; the
    # rotation carried past contact is the follow-through
    move_act = _steer_to_velocity(vel, np.zeros(2))
    rot_act = 1 if sense > 0 else 2
    return (move_act, rot_act, tilt_act)


def scripted_action(env) -> tuple[int, int, int, int, int, int]:
    """env: RustBeatEnv (needs .chart, .sim.status()/sabers()/time)."""
    status = env.sim.status()
    targets = {0: None, 1: None}
    for i, n in enumerate(env.chart):
        if status[i] == 0 and targets[n.color] is None:
            targets[n.color] = n
        if targets[0] is not None and targets[1] is not None:
            break
    sabers = env.sim.sabers()
    time = env.sim.time
    acts = []
    for hand in (0, 1):
        acts.extend(_hand_action(hand, np.asarray(sabers[hand]), time,
                                 targets[hand]))
    return tuple(acts)
