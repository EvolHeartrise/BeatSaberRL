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
@pytest.mark.skipif(float(os.environ.get("BEAT_Z_ASINH", "8")) != 8.0,
                    reason="the in-process rust sim must run at C = 8")
def test_cuda_rust_parity():
    """Both engines at their default C (= 8 m, the recipe): every
    entity lane agrees under lockstep resync. Plus a C = 2 CUDA env on
    the same resynced state: its z lanes (2, 17) must be the rust C = 8
    lanes re-encoded to C = 2 and every other lane identical — pins that
    the kernel honours z_asinh (the rust side of C != 8 is pinned by the
    child-process tests above). Same chart as _v48_z_child.py."""
    import numpy as np
    import beat_arena_rs
    from beat_arena_env.chart import Note, Wall, note_tuple, wall_tuple
    from beat_arena_env.constants import (N_JOINTS, NOTE_FEAT, OBS_DIM,
                                          OFF_NOTES, OFF_NOTE_MASK)
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    from tests._helpers import resync, rust_state
    speed, max_steps = 20.0, 600
    notes = [Note(hit_time=z / speed, col=2, row=1, x=0.3, y=1.4, color=1,
                  angle=4.712, speed=speed) for z in (5.0, 60.0, 80.0)]
    walls = [wall_tuple(Wall(100.0 / speed, 0.0, 0.0, 20.0, 40.0, 30.0,
                             speed))]
    kw = dict(charts=[notes], walls=[walls], max_steps=max_steps, gears=3,
              device="cuda", dtype=torch.float64, auto_reset=False,
              seed=0)
    env = CudaBeatVecEnv(1, **kw)
    env2 = CudaBeatVecEnv(1, z_asinh=2.0, **kw)
    assert env.z_asinh == 8.0
    env.reset()
    env2.reset()
    sim = beat_arena_rs.RawBeatSim(max_steps=max_steps, gears=3)
    sim.set_chart([note_tuple(n) for n in notes], walls)
    ent = slice(OFF_NOTES, OFF_NOTE_MASK)
    zl = np.zeros((OFF_NOTE_MASK - OFF_NOTES) // NOTE_FEAT * NOTE_FEAT,
                  dtype=bool).reshape(-1, NOTE_FEAT)
    zl[:, [2, 17]] = True
    zl = zl.reshape(-1)                     # z lanes of the entity block
    act = torch.zeros((1, N_JOINTS), dtype=torch.int64, device="cuda")
    worst = worst2 = 0.0
    for tick in range(30):
        # lockstep-resync (v10 protocol): CUDA re-takes rust's state
        q36, v35 = rust_state(sim)
        resync(env, [q36], [v35])
        resync(env2, [q36], [v35])
        o_r, _, d_r, _ = sim.step([0] * N_JOINTS)
        if d_r:
            # the coasting body tipped over (ground contact since the
            # weld harness was removed, 2026-09-27): a finished env's
            # frozen obs is no longer a lockstep comparison
            break
        o_c, _, _, _ = env.step(act)
        o_2, _, _, _ = env2.step(act)
        o_r = np.asarray(o_r)
        o_c = o_c.cpu().numpy().reshape(OBS_DIM)
        o_2 = o_2.cpu().numpy().reshape(OBS_DIM)
        worst = max(worst, float(np.abs(o_c[ent] - o_r[ent]).max()))
        # rust C = 8 lanes re-encoded to C = 2 (asinh(z / C), z from sinh)
        r64 = o_r[ent].astype(np.float64)
        want2 = np.where(zl, np.arcsinh(np.sinh(r64) * 8.0 / 2.0), r64)
        worst2 = max(worst2, float(np.abs(o_2[ent] - want2).max()))
        for o in (o_c, o_2):
            assert (o[OFF_NOTE_MASK:OFF_NOTE_MASK + 24]
                    == o_r[OFF_NOTE_MASK:OFF_NOTE_MASK + 24]).all(), tick
    assert tick >= 15, tick
    # f64 lockstep: the lanes agree to the FK ulp level (v17 gate: 1e-4)
    assert worst < 1e-4, worst
    assert worst2 < 1e-4, worst2
    # and C really moved the far z lanes (40+ m out: asinh(z/8) vs z/2)
    assert float(np.abs(o_2[ent][zl] - o_c[ent][zl]).max()) > 0.5
