"""Wrist-bend reversal telemetry (2026-09-22, user): per hand, a reversal
is counted when the wrist bend axis (joints 8 / 15) turns back by 20 deg
from its running extreme, on every tick. Pins (1) the state machine on
synthetic traces via a Python reference, (2) the CUDA kernel's count
against that reference run on the kernel's own integrated angles
(including episode resets), (3) the Rust sim likewise, (4) the widened
bench row. Needs torch; the sim parts need CUDA / beat_arena_rs."""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from beat_arena_env.constants import N_JOINTS  # noqa: E402

HYST = 0.34906585
L_BEND, R_BEND = 8, 15


class Ref:
    """The kernel's detector: 0 fresh, 3 armed, 1 moving +, 2 moving -."""

    def __init__(self):
        self.dir, self.ext, self.count = 0, 0.0, 0

    def reset(self):
        self.dir = 0

    def feed(self, q):
        d, e = self.dir, self.ext
        if d == 0:
            e, d = q, 3
        elif d == 3:
            if q >= e + HYST:
                d, e = 1, q
            elif q <= e - HYST:
                d, e = 2, q
        elif d == 1:
            if q > e:
                e = q
            elif q <= e - HYST:
                self.count += 1; d, e = 2, q
        else:
            if q < e:
                e = q
            elif q >= e + HYST:
                self.count += 1; d, e = 1, q
        self.dir, self.ext = d, e


def _count(trace):
    r = Ref()
    for q in trace:
        r.feed(q)
    return r.count


def test_reference_semantics():
    deg = math.radians
    assert _count([0.0] * 50) == 0
    # 30 deg flick and back: 2 reversals (out then back), the first
    # direction commitment is not a reversal
    assert _count([0, deg(10), deg(20), deg(30), deg(20), deg(10), 0]) == 2
    # 10 deg wiggle: never crosses the hysteresis -> 0
    assert _count([0, deg(10), 0, deg(10), 0, deg(10), 0]) == 0
    # 25 deg out, 15 back (not enough), then on to 40: no reversal yet;
    # then back to 0: one reversal
    assert _count([0, deg(25), deg(10), deg(40)]) == 0
    assert _count([0, deg(25), deg(10), deg(40), 0]) == 1
    # slow drift with tremor: 0
    tr = [i * deg(0.5) + (deg(3) if i % 2 else 0) for i in range(200)]
    assert _count(tr) == 0
    # a 2 Hz 30 deg oscillation over 10 s at 30 Hz: 4 reversals per
    # second minus the first commitment
    osc = [deg(15) * math.sin(2 * math.pi * 2.0 * t / 30.0) for t in range(300)]
    assert 38 <= _count(osc) <= 40


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_reference_on_its_own_angles():
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    N, T = 8, 240
    env = CudaBeatVecEnv(N, chart_len_s=6.0, seed=3, device="cuda",
                         dtype=torch.float64, gears=5)
    env.reset()
    refs = [[Ref(), Ref()] for _ in range(N)]
    rng = np.random.default_rng(0)
    # bench-row width
    ids, rows = env.pop_bench_rows()
    assert rows.shape == (0, 17)
    for t in range(T):
        acts = torch.as_tensor(rng.integers(0, 5, (N, N_JOINTS)), device="cuda")
        _obs, _r, done, _ev = env.step(acts)
        q = env.q[:, [L_BEND, R_BEND]].cpu().numpy()   # this tick's angles
        d = done.cpu().numpy()
        for i in range(N):
            for h in range(2):
                refs[i][h].feed(float(q[i, h]))
            if d[i]:                       # auto-reset: the kernel's
                for h in range(2):         # reset_env re-arms the state
                    refs[i][h].reset()
    l, r = env.pop_wrist_rev()
    exp_l = sum(refs[i][0].count for i in range(N))
    exp_r = sum(refs[i][1].count for i in range(N))
    assert (l, r) == (exp_l, exp_r), ((l, r), (exp_l, exp_r))
    assert l + r > 0                       # random gears do flap the wrists
    assert env.pop_wrist_rev() == (0, 0)   # popped + zeroed


def test_rust_matches_reference_on_its_own_angles():
    beat_arena_rs = pytest.importorskip("beat_arena_rs")
    from beat_arena_env.chart import Note, note_tuple
    sim = beat_arena_rs.RawBeatSim(max_steps=100000, wide_acc=False,
                                   gears=5, foot_contact=False)
    sim.set_chart([note_tuple(Note(hit_time=30.0, col=2, row=1, color=1, angle=0.0, speed=4.0))], [])
    rng = np.random.default_rng(1)
    refs = [Ref(), Ref()]
    n = 0
    for t in range(240):
        _obs, _r, done, _ev = sim.step([int(a) for a in rng.integers(0, 5, N_JOINTS)])
        qp = sim.qpos()
        refs[0].feed(float(qp[7 + L_BEND])); refs[1].feed(float(qp[7 + R_BEND]))
        n += 1
        if done:
            break
    l, r, ticks = sim.wrist_rev()
    assert (l, r) == (refs[0].count, refs[1].count), ((l, r), (refs[0].count, refs[1].count))
    assert ticks == n
