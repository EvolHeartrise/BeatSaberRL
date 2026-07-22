"""Scripted reference bot (SPEC.md §Scripted reference bot) — env v5.

Real swings now: per hand, the bot plants its wrist so the blade will
pass through the note PERPENDICULAR to the required cut direction, winds
the azimuth back past the full pre-swing arc, then whips the blade
through the note. Stateless per tick; solvability proof and eyeball
baseline, not optimal play.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    ACCEL, ALPHA_ROT, DRAG, DT, OMEGA_MAX, SABER_LEN, note_center,
)

CONTACT_S = 0.75          # note passes the blade at this fraction of its length
SWING_ARC = np.radians(130.0)  # wind-up past the 100 deg full-credit arc
SWING_T = 0.18            # seconds before hit_time to start the whip
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


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _cut_config(center, angle, wrist):
    """Contact azimuth + wrist goal + swing sense for a note.

    The blade must be perpendicular to the cut direction at contact; a CCW
    swing (sense +1) puts the contact point's velocity along the cut
    direction when theta_c = angle - 90 deg, a CW swing when +90 deg.
    Prefer the candidate whose wrist goal is in bounds, then the nearer.
    """
    cands = []
    for sense, th_c in ((1.0, angle - np.pi / 2), (-1.0, angle + np.pi / 2)):
        goal = center - CONTACT_S * SABER_LEN * np.array([np.cos(th_c), np.sin(th_c)])
        in_bounds = -1.45 <= goal[0] <= 1.45 and -1.2 <= goal[1] <= 1.2
        cands.append((not in_bounds, float(np.linalg.norm(goal - wrist)),
                      sense, th_c, goal))
    cands.sort(key=lambda c: (c[0], c[1]))
    _, _, sense, th_c, goal = cands[0]
    return sense, th_c, goal


def _hand_action(hand, saber, time, target):
    pos, vel = saber[0:2], saber[2:4]
    theta, phi, om_th, om_ph = saber[4], saber[5], saber[6], saber[7]
    tilt_act = _steer_omega(om_ph, np.clip(K_ROT * -phi, -OMEGA_MAX, OMEGA_MAX))
    if target is None:
        to_park = PARK[hand] - pos
        dist = float(np.linalg.norm(to_park))
        desired = to_park / dist * min(CRUISE, K_ARRIVE * dist) if dist > 0.1 else np.zeros(2)
        # blade straight up, no spin
        rot_act = _steer_omega(om_th, np.clip(
            K_ROT * _wrap(np.pi / 2 - theta), -OMEGA_MAX, OMEGA_MAX))
        return (_steer_to_velocity(vel, desired), rot_act, tilt_act)

    center = np.array(note_center(target))
    # dot notes: any direction scores — pretend it's a down-cut
    angle = target.angle if not target.any_dir else 3 * np.pi / 2
    sense, th_c, goal = _cut_config(center, angle, pos)
    t_until = target.hit_time - time

    if t_until > SWING_T:
        # wind-up: park the wrist at the goal, cock the blade back
        to_goal = goal - pos
        dist = float(np.linalg.norm(to_goal))
        speed = min(CRUISE, K_ARRIVE * dist)
        desired = to_goal / dist * speed if dist > 1e-6 else np.zeros(2)
        th_start = th_c - sense * SWING_ARC
        rot_act = _steer_omega(om_th, np.clip(
            K_ROT * _wrap(th_start - theta), -OMEGA_MAX, OMEGA_MAX))
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
