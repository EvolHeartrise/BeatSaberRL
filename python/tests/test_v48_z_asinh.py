"""env v48 — asinh obs z lane (user, 2026-09-20).

Lane 2 (note z / wall front face) and lane 17 (wall back face) become
asinh(max(z, z_lo) / C) with C = --z-asinh metres (BEAT_Z_ASINH for the
rust sim): linear near the cut plane, log far out, NO upper clamp — so
far tokens stay distinguishable. It is the only z encoding (the legacy
clamp / 8 path was removed 2026-09-22); both engines default to C = 8 m
and refuse C <= 0. Each rust configuration runs in a child process
because the rust sim reads the variable once per process; the CUDA-vs-
rust parity case uses the default C = 8 and runs in-process (a CUDA
child OOMed on reset behind the parent's cached allocations).
"""
import json
import math
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = os.path.join(HERE, "_v48_z_child.py")


def run_child(c, extra_env=None, check=True):
    """c = None leaves BEAT_Z_ASINH unset (engine defaults)."""
    env = dict(os.environ)
    env.pop("BEAT_Z_ASINH", None)
    if c is not None:
        env["BEAT_Z_ASINH"] = repr(float(c))
    # the child is run by path, so its sys.path holds tests/, not python/
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(HERE)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env.update(extra_env or {})
    r = subprocess.run([sys.executable, CHILD,
                        "default" if c is None else repr(float(c))],
                       capture_output=True, text=True, env=env,
                       cwd=os.path.dirname(HERE))
    if not check:
        return r
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_rust_default_is_8m():
    assert run_child(None) == run_child(8.0)


@pytest.mark.parametrize("c", [0.0, -1.0])
def test_rust_refuses_nonpositive_c(c):
    r = run_child(c, check=False)
    assert r.returncode != 0
    assert "BEAT_Z_ASINH must be > 0" in r.stderr


def test_asinh_keeps_far_notes_distinct():
    c = 2.0
    o = run_child(c)
    assert o["mask"] == [1.0, 1.0, 1.0, 1.0]
    assert o["is_wall"] == 1.0
    # one tick in: z ~ z0 - speed/30 (+ a ~0 head offset); recover metres
    for slot, z0 in ((0, 5.0), (1, 60.0), (2, 80.0)):
        z = math.sinh(o["z"][slot]) * c
        assert z == pytest.approx(z0 - 20.0 / 30, abs=0.05), (slot, z)
    assert o["z"][2] > o["z"][1] > o["z"][0]
    # wall: front 100 m, back 130 m, both unclamped (WALL_FRONT_OFF small)
    assert math.sinh(o["z"][3]) * c == pytest.approx(100.0 - 20 / 30, abs=0.5)
    assert math.sinh(o["wall_back"]) * c == pytest.approx(130.0 - 20 / 30, abs=0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("c", [0.0, -1.0])
def test_cuda_env_refuses_nonpositive_c(c):
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    with pytest.raises(ValueError, match="z_asinh must be > 0"):
        CudaBeatVecEnv(1, device="cuda", z_asinh=c)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(os.environ.get("BEAT_Z_ASINH") not in (None, "8.0"),
                    reason="parity is pinned at the default C = 8")
def test_cuda_rust_parity():
    """Both engines at their default C (= 8 m, the recipe): the z lanes
    agree under lockstep resync. Same chart as _v48_z_child.py."""
    import numpy as np
    import beat_arena_rs
    from beat_arena_env.chart import Note, Wall, note_tuple, wall_tuple
    from beat_arena_env.constants import (N_JOINTS, OBS_DIM, OFF_NOTES,
                                          OFF_NOTE_MASK)
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    from tests._helpers import resync, rust_state
    speed, max_steps = 20.0, 600
    notes = [Note(hit_time=z / speed, col=2, row=1, x=0.3, y=1.4, color=1,
                  angle=4.712, speed=speed) for z in (5.0, 60.0, 80.0)]
    walls = [wall_tuple(Wall(100.0 / speed, 0.0, 0.0, 20.0, 40.0, 30.0,
                             speed))]
    env = CudaBeatVecEnv(1, charts=[notes], walls=[walls],
                         max_steps=max_steps, gears=3, device="cuda",
                         dtype=torch.float64, auto_reset=False,
                         foot_contact=False, seed=0)
    assert env.z_asinh == 8.0
    env.reset()
    sim = beat_arena_rs.RawBeatSim(max_steps=max_steps, gears=3,
                                   foot_contact=False)
    sim.set_chart([note_tuple(n) for n in notes], walls)
    worst = 0.0
    for tick in range(30):
        # lockstep-resync (v10 protocol): CUDA re-takes rust's state
        q36, v35 = rust_state(sim)
        resync(env, [q36], [v35])
        o_r, _, _, _ = sim.step([0] * N_JOINTS)
        o_c, _, _, _ = env.step(torch.zeros((1, N_JOINTS), dtype=torch.int64,
                                            device="cuda"))
        o_r = np.asarray(o_r)
        o_c = o_c.cpu().numpy().reshape(OBS_DIM)
        worst = max(worst, float(np.abs(o_c[OFF_NOTES:OFF_NOTE_MASK]
                                        - o_r[OFF_NOTES:OFF_NOTE_MASK]).max()))
        assert (o_c[OFF_NOTE_MASK:OFF_NOTE_MASK + 24]
                == o_r[OFF_NOTE_MASK:OFF_NOTE_MASK + 24]).all(), tick
    # f64 lockstep: the lanes agree to the FK ulp level (v17 gate: 1e-4)
    assert worst < 1e-4, worst
