"""Child process for test_v48_z_asinh.py — BEAT_Z_ASINH is read ONCE per
process by the rust sim, so every configuration runs in its own process.
(The CUDA-vs-rust parity case needs only the default C, so it runs
in-process in the test module.)

argv: <c>   c = "default" leaves BEAT_Z_ASINH unset (the 8 m default).
Prints one JSON line."""
import json
import os
import sys

import numpy as np

from beat_arena_env.chart import Note, Wall, note_tuple, wall_tuple
from beat_arena_env.constants import (N_JOINTS, NOTE_FEAT, OFF_NOTES,
                                      OFF_NOTE_MASK)
import beat_arena_rs

if sys.argv[1] == "default":
    assert "BEAT_Z_ASINH" not in os.environ
else:
    assert os.environ.get("BEAT_Z_ASINH") == repr(float(sys.argv[1]))
SPEED = 20.0
# same x/y so the nearest-first slot order is the z order: 5, 60, 80 m
NOTES = [Note(hit_time=z / SPEED, col=2, row=1, x=0.3, y=1.4, color=1,
              angle=4.712, speed=SPEED) for z in (5.0, 60.0, 80.0)]
WALLS = [Wall(100.0 / SPEED, 0.0, 0.0, 20.0, 40.0, 30.0, SPEED)]  # front 100 m, back 130 m
MAX_STEPS = 600


def lane(obs, slot, k):
    return float(obs[OFF_NOTES + slot * NOTE_FEAT + k])


sim = beat_arena_rs.RawBeatSim(max_steps=MAX_STEPS, gears=3,
                               foot_contact=False)
sim.set_chart([note_tuple(n) for n in NOTES], [wall_tuple(w) for w in WALLS])
obs_r, _, _, _ = sim.step([0] * N_JOINTS)
obs_r = np.asarray(obs_r)
out = {"mask": [float(obs_r[OFF_NOTE_MASK + s]) for s in range(4)],
       "z": [lane(obs_r, s, 2) for s in range(4)],
       "wall_back": lane(obs_r, 3, 17), "is_wall": lane(obs_r, 3, 14)}

print(json.dumps(out))
