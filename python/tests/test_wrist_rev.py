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
    """The kernel's detector: 0 fresh, 3 armed (tracks lo and hi), 1
    moving + (tracks hi), 2 moving - (tracks lo)."""

    def __init__(self):
        self.dir, self.lo, self.hi, self.count = 0, 0.0, 0.0, 0

    def reset(self):
        self.dir = 0

    def feed(self, q):
        d, lo, hi = self.dir, self.lo, self.hi
        if d == 0:
            lo = hi = q; d = 3
        elif d == 3:
            lo = min(lo, q); hi = max(hi, q)
            if q >= lo + HYST:
                d, hi = 1, q
            elif q <= hi - HYST:
                d, lo = 2, q
        elif d == 1:
            if q > hi:
                hi = q
            elif q <= hi - HYST:
                self.count += 1; d, lo = 2, q
        else:
            if q < lo:
                lo = q
            elif q >= lo + HYST:
                self.count += 1; d, hi = 1, q
        self.dir, self.lo, self.hi = d, lo, hi


def _count(trace):
    r = Ref()
    for q in trace:
        r.feed(q)
    return r.count


def test_reference_semantics():
    deg = math.radians
    assert _count([0.0] * 50) == 0
    # 30 deg flick and back: ONE reversal (the turn at the peak); the
    # first commitment out of rest is not a direction change
    assert _count([0, deg(10), deg(20), deg(30), deg(20), deg(10), 0]) == 1
    # ... and back out again: the second turn makes 2
    assert _count([0, deg(10), deg(20), deg(30), deg(20), deg(10), 0,
                   deg(10), deg(20), deg(30)]) == 2
    # 10 deg wiggle: never crosses the hysteresis -> 0
    assert _count([0, deg(10), 0, deg(10), 0, deg(10), 0]) == 0
    # 25 deg out, 15 back (not enough), then on to 40: no reversal yet;
    # then back to 0: one reversal
    assert _count([0, deg(25), deg(10), deg(40)]) == 0
    assert _count([0, deg(25), deg(10), deg(40), 0]) == 1
    # slow drift with tremor: 0
    tr = [i * deg(0.5) + (deg(3) if i % 2 else 0) for i in range(200)]
    assert _count(tr) == 0
    # a 2 Hz 30 deg oscillation over 10 s at 30 Hz: one reversal per
    # half cycle = 4 per second, minus the first commitment
    osc = [deg(15) * math.sin(2 * math.pi * 2.0 * t / 30.0) for t in range(300)]
    assert 38 <= _count(osc) <= 40


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_reference_on_its_own_angles():
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    N, T = 8, 200
    # no auto-reset: on a terminating tick the kernel resets the pose
    # before the wrapper exposes it, so the detector's input for that
    # tick is unobservable; here only the wrists are actuated (neutral
    # gear 2 elsewhere) so the slumping body never terminates
    env = CudaBeatVecEnv(N, chart_len_s=6.0, seed=3, device="cuda",
                         dtype=torch.float64, gears=5, auto_reset=False)
    env.reset()
    refs = [[Ref(), Ref()] for _ in range(N)]
    rng = np.random.default_rng(0)
    ids, rows = env.pop_bench_rows()
    assert rows.shape == (0, 17)                      # widened bench row
    steps = 0
    for t in range(T):
        a = np.full((N, N_JOINTS), 2, dtype=np.int64)
        a[:, [L_BEND, R_BEND]] = rng.integers(0, 5, (N, 2))
        _obs, _r, done, _ev = env.step(torch.as_tensor(a, device="cuda"))
        q = env.q[:, [L_BEND, R_BEND]].cpu().numpy()   # this tick's angles
        d = done.cpu().numpy()
        for i in range(N):      # auto_reset=False: a terminated env keeps
            for h in range(2):  # its pose, so this IS the detector's input
                refs[i][h].feed(float(q[i, h]))
        steps += 1
        if d.any():             # stop before the frozen ticks
            break
    assert steps >= 100, steps
    l, r = env.pop_wrist_rev()
    exp_l = sum(refs[i][0].count for i in range(N))
    exp_r = sum(refs[i][1].count for i in range(N))
    assert (l, r) == (exp_l, exp_r), ((l, r), (exp_l, exp_r))
    assert l + r > 0                       # random wrist gears do flap
    assert env.pop_wrist_rev() == (0, 0)   # popped + zeroed


def test_rust_matches_reference_on_its_own_angles():
    beat_arena_rs = pytest.importorskip("beat_arena_rs")
    from beat_arena_env.chart import Note, note_tuple
    sim = beat_arena_rs.RawBeatSim(max_steps=100000,
                                   gears=5, foot_contact=False)
    sim.set_chart([note_tuple(Note(hit_time=30.0, col=2, row=1, color=1, angle=0.0, speed=4.0))], [])
    rng = np.random.default_rng(1)
    refs = [Ref(), Ref()]
    n = 0
    for t in range(240):
        a = [2] * N_JOINTS                  # neutral gear except the wrists
        a[L_BEND], a[R_BEND] = (int(x) for x in rng.integers(0, 5, 2))
        _obs, _r, done, _ev = sim.step(a)
        qp = sim.qpos()
        refs[0].feed(float(qp[7 + L_BEND])); refs[1].feed(float(qp[7 + R_BEND]))
        n += 1
        if done:
            break
    l, r, ticks = sim.wrist_rev()
    assert (l, r) == (refs[0].count, refs[1].count), ((l, r), (refs[0].count, refs[1].count))
    assert ticks == n
