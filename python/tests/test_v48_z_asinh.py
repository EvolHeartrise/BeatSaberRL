"""env v48 — asinh obs z lane (user, 2026-09-20).

Lane 2 (note z / wall front face) and lane 17 (wall back face) become
asinh(max(z, z_lo) / C) with C = --z-asinh metres (BEAT_Z_ASINH for the
rust sim): linear near the cut plane, log far out, NO upper clamp — so
far tokens stay distinguishable. It is the only z encoding (the legacy
clamp / 8 path was removed 2026-09-22); both engines default to C = 8 m
and refuse C <= 0. Each configuration runs in a child process because
the rust sim reads the variable once per process.
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


def run_child(mode, c, extra_env=None, check=True):
    """c = None leaves BEAT_Z_ASINH unset (engine defaults)."""
    env = dict(os.environ)
    env.pop("BEAT_Z_ASINH", None)
    if c is not None:
        env["BEAT_Z_ASINH"] = repr(float(c))
    # the child is run by path, so its sys.path holds tests/, not python/
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(HERE)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env.update(extra_env or {})
    r = subprocess.run([sys.executable, CHILD, mode,
                        "default" if c is None else repr(float(c))],
                       capture_output=True, text=True, env=env,
                       cwd=os.path.dirname(HERE))
    if not check:
        return r
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_rust_default_is_8m():
    assert run_child("rust", None) == run_child("rust", 8.0)


@pytest.mark.parametrize("c", [0.0, -1.0])
def test_rust_refuses_nonpositive_c(c):
    r = run_child("rust", c, check=False)
    assert r.returncode != 0
    assert "BEAT_Z_ASINH must be > 0" in r.stderr


def test_asinh_keeps_far_notes_distinct():
    c = 2.0
    o = run_child("rust", c)
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
@pytest.mark.parametrize("c", [None, 2.0, 8.0])
def test_cuda_rust_parity(c):
    o = run_child("parity", c)
    # f64 lockstep: the lanes agree to the FK ulp level (v17 gate: 1e-4)
    assert o["parity_worst_abs"] < 1e-4, o
