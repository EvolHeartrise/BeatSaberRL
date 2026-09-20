"""Child process for test_v48_z_asinh.py — BEAT_Z_ASINH is read ONCE per
process by the rust sim, so every configuration runs in its own process.

argv: <mode> <c>   mode = rust | parity
Prints one JSON line."""
import json
import math
import os
import sys

import numpy as np

from beat_arena_env.chart import Note, Wall, note_tuple, wall_tuple
from beat_arena_env.constants import (N_JOINTS, NOTE_FEAT, OBS_DIM, OFF_NOTES,
                                      OFF_NOTE_MASK)
import beat_arena_rs

mode, c = sys.argv[1], float(sys.argv[2])
assert os.environ.get("BEAT_Z_ASINH") == repr(c)
zfar = float(os.environ.get("BEAT_Z_FAR", "0"))      # env v49 split mode
SPEED = 20.0
# same x/y so the nearest-first slot order is the z order: 5, 60, 80 m
NOTES = [Note(hit_time=z / SPEED, col=2, row=1, x=0.3, y=1.4, color=1,
              angle=4.712, speed=SPEED) for z in (5.0, 60.0, 80.0)]
WALLS = [Wall(100.0 / SPEED, 0.0, 0.0, 20.0, 40.0, 30.0, SPEED)]  # front 100 m, back 130 m
MAX_STEPS = 600


def lane(obs, slot, k):
    return float(obs[OFF_NOTES + slot * NOTE_FEAT + k])


sim = beat_arena_rs.RawBeatSim(max_steps=MAX_STEPS, wide_acc=False, gears=3,
                               foot_contact=False)
sim.set_chart([note_tuple(n) for n in NOTES], [wall_tuple(w) for w in WALLS])
obs_r, _, _, _ = sim.step([0] * N_JOINTS)
obs_r = np.asarray(obs_r)
out = {"mask": [float(obs_r[OFF_NOTE_MASK + s]) for s in range(4)],
       "z": [lane(obs_r, s, 2) for s in range(4)],
       "far": [lane(obs_r, s, 13) for s in range(4)],
       "wall_back": lane(obs_r, 3, 17), "is_wall": lane(obs_r, 3, 14)}

if mode == "parity":
    import torch
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    env = CudaBeatVecEnv(1, charts=[NOTES], walls=[[wall_tuple(w) for w in WALLS]],
                         max_steps=MAX_STEPS, gears=3, wide_acc=False,
                         device="cuda", dtype=torch.float64, auto_reset=False,
                         foot_contact=False, seed=0, z_asinh=c, z_far=zfar)
    env.reset()
    sim2 = beat_arena_rs.RawBeatSim(max_steps=MAX_STEPS, wide_acc=False,
                                    gears=3, foot_contact=False)
    sim2.set_chart([note_tuple(n) for n in NOTES], [wall_tuple(w) for w in WALLS])
    worst = 0.0
    for tick in range(30):
        # lockstep-resync (v10 protocol): CUDA re-takes rust's state
        q36 = list(sim2.qpos())
        v35 = list(np.concatenate([sim2.twist(), sim2.omega()]))
        wv = env.wrist_vel.clone()
        env.set_state(torch.tensor([q36], dtype=torch.float64),
                      torch.tensor([v35], dtype=torch.float64))
        env.wrist_vel.copy_(wv)
        o_r, _, _, _ = sim2.step([0] * N_JOINTS)
        o_c, _, _, _ = env.step(torch.zeros((1, N_JOINTS), dtype=torch.int64,
                                            device="cuda"))
        o_r = np.asarray(o_r)
        o_c = o_c.cpu().numpy().reshape(OBS_DIM)
        worst = max(worst, float(np.abs(o_c[OFF_NOTES:OFF_NOTE_MASK]
                                        - o_r[OFF_NOTES:OFF_NOTE_MASK]).max()))
        assert (o_c[OFF_NOTE_MASK:OFF_NOTE_MASK + 24]
                == o_r[OFF_NOTE_MASK:OFF_NOTE_MASK + 24]).all(), tick
    out["parity_worst_abs"] = worst
print(json.dumps(out))
