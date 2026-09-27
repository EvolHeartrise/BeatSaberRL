"""Helpers shared by the test modules (import as `tests._helpers`)."""
import math
import os

import numpy as np
import pytest

from beat_arena_env.chart import Note
from beat_arena_env.constants import ALIVE_BONUS, N_JOINTS

try:  # the Rust-only tests import this module too and need no torch
    import torch
except ImportError:
    torch = None

cuda = pytest.mark.skipif(torch is None or not torch.cuda.is_available(),
                          reason="needs CUDA")

# a note far beyond every horizon: pure body dynamics
FAR_NOTE = Note(hit_time=200.0, col=2, row=1, x=0.3, y=1.4, color=1,
                angle=4.712, speed=4.0)

ZERO_ACT = [0] * N_JOINTS  # gear 0 = zero torque on every joint

# the per-tick survival reward every live tick collects (contact spec §7)
ALIVE = ALIVE_BONUS

# env v48 obs z lane (lanes 2/17): asinh(z / C); the rust sim reads C from
# BEAT_Z_ASINH (unset = the engines' default 8 m), so mirror it here
Z_ASINH_C = float(os.environ.get("BEAT_Z_ASINH", "8.0"))


def z_lane(z_m):
    """metres -> the obs z-lane value (above the head-z floor)."""
    return math.asinh(z_m / Z_ASINH_C)


def z_lane_m(lane):
    """obs z-lane value -> metres."""
    return math.sinh(float(lane)) * Z_ASINH_C


def rust_state(sim):
    """A RawBeatSim's dynamics state as (q36, v35) lists."""
    return list(sim.qpos()), list(np.concatenate([sim.twist(), sim.omega()]))


def resync(env, q36s, v35s):
    """Lockstep resync (the v10 parity protocol): the CUDA env re-takes
    the given per-env Rust states. set_state zeroes the finite-diff wrist
    velocity (Rust semantics), which is env-layer state Rust keeps, so the
    true velocity is carried across by hand."""
    wv = env.wrist_vel.clone()
    env.set_state(torch.tensor(q36s, dtype=torch.float64),
                  torch.tensor(v35s, dtype=torch.float64))
    env.wrist_vel.copy_(wv)


def resync_from_sims(env, sims):
    """resync() from one Rust sim per env."""
    q36s, v35s = [], []
    for sim in sims:
        q, v = rust_state(sim)
        q36s.append(q)
        v35s.append(v)
    resync(env, q36s, v35s)


# the v10 reference root pose (dynamics/model.rs base_pos0, identity quat)
REF_ROOT7 = [0.0, 0.94, -0.65, 1.0, 0.0, 0.0, 0.0]
LEG_JOINTS = range(17, 29)


def plant_reference(sim):
    """Put a RawBeatSim's body at the v10 REFERENCE stance, at rest: root
    (0, 0.94, -0.65) upright, legs neutral — all 8 sole corners exactly on
    the ground plane — spine and arms kept. This is the state the retired
    welded-feet reset started from (arm ready stance, neutral legs, root
    at the reference), so scripted scenarios keep their v10 geometry.

    The body is on unilateral ground contact, NOT welded: unactuated it
    starts to tip after ~0.5 s (tilt termination at ~0.7 s), so this is a
    platform for SHORT windows. Returns the sim."""
    q = list(sim.qpos())
    q[:7] = REF_ROOT7
    for j in LEG_JOINTS:
        q[7 + j] = 0.0
    sim.set_joints(q, [0.0] * 35)
    return sim
