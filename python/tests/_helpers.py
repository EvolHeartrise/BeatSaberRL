"""Helpers shared by the test modules (import as `tests._helpers`)."""
import numpy as np
import pytest

from beat_arena_env.chart import Note
from beat_arena_env.constants import N_JOINTS

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
