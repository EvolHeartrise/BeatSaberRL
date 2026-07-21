"""Scripted reference bot (SPEC.md §Scripted reference bot).

Wind-up then swing, per hand. Solvability proof and eval ceiling
reference, not optimal play. Drives a RustBeatEnv via the sim's public
getters (sabers/status/time) — pure Python steering over the Rust sim.
"""

from __future__ import annotations

import numpy as np

from .constants import ACCEL, DRAG, DT, note_center

# A cut registers at first contact with the note's RADIUS (0.28 m), not
# its center — the speed check happens after only (WINDUP_DIST - 0.28) m
# of runway. 0.65 gives ~sqrt(2*30*0.37) ~ 4.7 m/s at contact.
WINDUP_DIST = 0.65
SWING_T = 0.21       # seconds before hit_time to start the swing
SWING_SPEED = 5.0
CRUISE = 4.0
K_ARRIVE = 4.0       # arrival gain (1/s): desired speed ~ K * distance

# idle hands park OUTSIDE the grid: a saber resting inside it gets hit by
# arriving blocks (stationary "crossing" = bad cut, reason wrong_hand)
PARK = {0: np.array([-1.35, 0.0]), 1: np.array([1.35, 0.0])}

_DIRS = np.array([[np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)] for k in range(8)])
# candidate per-tick velocity deltas for each of the 9 actions (0 = coast)
_ACTION_DV = np.vstack([np.zeros(2), ACCEL * DT * _DIRS])


def _steer_to_velocity(vel, desired_v):
    """Pick the action whose one-tick velocity update lands closest to desired_v."""
    v_next = (vel + _ACTION_DV) * (1.0 - DRAG)
    return int(np.argmin(np.linalg.norm(v_next - desired_v, axis=1)))


def _hand_action(hand, pos, vel, time, target):
    if target is None:
        to_park = PARK[hand] - pos
        dist = float(np.linalg.norm(to_park))
        desired = to_park / dist * min(CRUISE, K_ARRIVE * dist) if dist > 0.1 else np.zeros(2)
        return _steer_to_velocity(vel, desired)
    center = np.array(note_center(target))
    d = np.array([np.cos(target.angle), np.sin(target.angle)])
    t_until = target.hit_time - time
    if t_until > SWING_T:
        goal = center - WINDUP_DIST * d
        to_goal = goal - pos
        dist = float(np.linalg.norm(to_goal))
        speed = min(CRUISE, K_ARRIVE * dist)
        desired = to_goal / dist * speed if dist > 1e-6 else np.zeros(2)
        return _steer_to_velocity(vel, desired)
    aim = center + 0.3 * d - pos
    dist = float(np.linalg.norm(aim))
    desired = aim / dist * SWING_SPEED if dist > 1e-6 else SWING_SPEED * d
    return _steer_to_velocity(vel, desired)


def scripted_action(env) -> tuple[int, int]:
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
        s = np.asarray(sabers[hand])
        acts.append(_hand_action(hand, s[:2], s[2:], time, targets[hand]))
    return acts[0], acts[1]
