"""Scripted reference bot (SPEC.md §Scripted reference bot).

Wind-up then swing, per hand. Solvability proof and eval ceiling reference,
not optimal play. Operates on the sim directly (like kiting-arena's
scripted policy) so tests and eval can share it.
"""

from __future__ import annotations

import numpy as np

from .chart import Note
from .sim import ACCEL, DRAG, DT, BeatSim, _DIRS, dir_vec, note_center

# A cut registers at first contact with the note's RADIUS (0.55), not its
# center — so the speed check happens after only (WINDUP_DIST - 0.55) units
# of runway. 1.3 gives ~sqrt(2*60*0.75) ~ 9.5 u/s at contact.
WINDUP_DIST = 1.3
SWING_T = 0.21       # seconds before hit_time to start the swing
SWING_SPEED = 10.0
CRUISE = 8.0
K_ARRIVE = 4.0       # arrival gain: desired speed ~ K * distance near goal

# idle hands park OUTSIDE the grid: a saber resting inside it gets hit by
# arriving blocks (stationary "crossing" = bad cut, reason wrong_hand)
PARK = {0: np.array([-0.7, 1.5]), 1: np.array([4.7, 1.5])}

# candidate per-tick velocity deltas for each of the 9 actions (0 = coast)
_ACTION_DV = np.vstack([np.zeros(2), ACCEL * DT * _DIRS])


def _steer_to_velocity(vel: np.ndarray, desired_v: np.ndarray) -> int:
    """Pick the action whose one-tick velocity update lands closest to desired_v."""
    v_next = (vel + _ACTION_DV) * (1.0 - DRAG)
    return int(np.argmin(np.linalg.norm(v_next - desired_v, axis=1)))


def _hand_action(sim: BeatSim, hand: int, target: Note | None) -> int:
    pos, vel = sim.pos[hand], sim.vel[hand]
    if target is None:
        to_park = PARK[hand] - pos
        dist = float(np.linalg.norm(to_park))
        desired = to_park / dist * min(CRUISE, K_ARRIVE * dist) if dist > 0.1 else np.zeros(2)
        return _steer_to_velocity(vel, desired)
    center = note_center(target)
    d = dir_vec(target.dir)
    t_until = target.hit_time - sim.time
    if t_until > SWING_T:
        goal = center - WINDUP_DIST * d
        to_goal = goal - pos
        dist = float(np.linalg.norm(to_goal))
        speed = min(CRUISE, K_ARRIVE * dist)
        desired = to_goal / dist * speed if dist > 1e-6 else np.zeros(2)
        return _steer_to_velocity(vel, desired)
    # swing: drive velocity toward a point just past the center, along the cut dir
    aim = center + 0.3 * d - pos
    dist = float(np.linalg.norm(aim))
    desired = aim / dist * SWING_SPEED if dist > 1e-6 else SWING_SPEED * d
    return _steer_to_velocity(vel, desired)


def scripted_action(sim: BeatSim) -> tuple[int, int]:
    targets: dict[int, Note | None] = {0: None, 1: None}
    for i in sim.pending_notes():
        n = sim.chart[i]
        if targets[n.color] is None:
            targets[n.color] = n
        if targets[0] is not None and targets[1] is not None:
            break
    return (
        _hand_action(sim, 0, targets[0]),
        _hand_action(sim, 1, targets[1]),
    )
