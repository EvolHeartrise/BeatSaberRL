"""env v48 — asinh obs z lane (user, 2026-09-20).

Lane 2 (note z / wall front face) and lane 17 (wall back face) become
asinh(max(z, z_lo) / C) with C = --z-asinh metres (BEAT_Z_ASINH for the
rust sim): linear near the cut plane, log far out, NO upper clamp — so
tokens beyond the old --z-clip-hi stay distinguishable. C = 0 is the
legacy clamp(z_lo, z_clip_hi) / 8 bit for bit. Each configuration runs in
a child process because the rust sim reads the variable once per process.
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


def run_child(mode, c, extra_env=None):
    env = dict(os.environ, BEAT_Z_ASINH=repr(float(c)))
    env.pop("BEAT_Z_CLIP_HI", None)
    env.update(extra_env or {})
    r = subprocess.run([sys.executable, CHILD, mode, repr(float(c))],
                       capture_output=True, text=True, env=env,
                       cwd=os.path.dirname(HERE))
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_legacy_saturates_far_notes():
    o = run_child("rust", 0.0)
    assert o["mask"] == [1.0, 1.0, 1.0, 1.0]
    # 60 m and 80 m both clamp to 12 m -> the same lane value 1.5
    assert o["z"][1] == pytest.approx(1.5, abs=1e-6)
    assert o["z"][2] == pytest.approx(1.5, abs=1e-6)
    assert o["wall_back"] == pytest.approx(1.5, abs=1e-6)
    assert 0 < o["z"][0] < 1.5


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
@pytest.mark.parametrize("c", [0.0, 2.0])
def test_cuda_rust_parity(c):
    o = run_child("parity", c)
    # f64 lockstep: the lanes agree to the FK ulp level (v17 gate: 1e-4)
    assert o["parity_worst_abs"] < 1e-4, o
