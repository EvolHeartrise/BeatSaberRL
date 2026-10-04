"""Wrist-bend reversal telemetry (2026-09-22, user): per hand, a reversal
is counted when the wrist bend axis (joints 8 / 15) turns back by 60 deg
(20 deg before 2026-10-04)
from its running extreme, on every tick. Pins (1) the state machine on
synthetic traces via a Python reference, (2) the CUDA kernel's count
against that reference run on the kernel's own integrated angles
(including episode resets), (3) the Rust sim likewise, (4) the widened
bench row. Needs torch; the sim parts need CUDA / beat_arena_rs."""
import math

import numpy as np
import pytest
import torch

from beat_arena_env.constants import N_JOINTS

HYST = 1.0471975511965976   # 60 deg, = WREV_HYST in lib.rs / the .cu
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
    # 90 deg flick and back: ONE reversal (the turn at the peak); the
    # first commitment out of rest is not a direction change
    assert _count([0, deg(30), deg(60), deg(90), deg(60), deg(30), 0]) == 1
    # ... and back out again: the second turn makes 2
    assert _count([0, deg(30), deg(60), deg(90), deg(60), deg(30), 0,
                   deg(30), deg(60), deg(90)]) == 2
    # 30 deg wiggle (counted under the old 20 deg): under the
    # hysteresis -> 0
    assert _count([0, deg(30), 0, deg(30), 0, deg(30), 0]) == 0
    # 75 deg out, 45 back (not enough), then on to 120: no reversal yet;
    # then back to 0: one reversal
    assert _count([0, deg(75), deg(30), deg(120)]) == 0
    assert _count([0, deg(75), deg(30), deg(120), 0]) == 1
    # just under / just over the hysteresis
    assert _count([0, deg(80), deg(20.5)]) == 0
    assert _count([0, deg(80), deg(19.5)]) == 1
    # slow drift with tremor: 0
    tr = [i * deg(1.5) + (deg(9) if i % 2 else 0) for i in range(200)]
    assert _count(tr) == 0
    # a 2 Hz 90 deg (peak to peak) oscillation over 10 s at 30 Hz: one
    # reversal per half cycle = 4 per second, minus the first commitment
    osc = [deg(45) * math.sin(2 * math.pi * 2.0 * t / 30.0) for t in range(300)]
    assert 38 <= _count(osc) <= 40


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_reference_on_its_own_angles():
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    # 128 envs x 18 ticks (was 8 x 200, then 32 x 60): the cost is the
    # per-tick host sync, so extra envs buy back reversal coverage almost
    # for free. f64 stays: the Python reference compares q against
    # lo/hi +- HYST in f64, so an f32 kernel could round a boundary tick
    # differently.
    N, T = 128, 18
    # no auto-reset: on a terminating tick the kernel resets the pose
    # before the wrapper exposes it, so the detector's input for that
    # tick is unobservable; only the wrists are actuated (coast
    # elsewhere) and the window ends before the coasting body tips over
    # (~23 ticks on ground contact, 2026-09-27 — the welded body never
    # terminated; the old "neutral gear 2" was -full torque, which the
    # weld absorbed)
    env = CudaBeatVecEnv(N, chart_len_s=6.0, seed=3, device="cuda",
                         dtype=torch.float64, gears=5, auto_reset=False)
    env.reset()
    refs = [[Ref(), Ref()] for _ in range(N)]
    rng = np.random.default_rng(0)
    ids, rows = env.pop_bench_rows()
    assert rows.shape == (0, 23)                      # widened bench row (B09 + torque)
    steps = 0
    for t in range(T):
        a = np.zeros((N, N_JOINTS), dtype=np.int64)
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
    assert steps == T, steps
    l, r = env.pop_wrist_rev()
    exp_l = sum(refs[i][0].count for i in range(N))
    exp_r = sum(refs[i][1].count for i in range(N))
    assert (l, r) == (exp_l, exp_r), ((l, r), (exp_l, exp_r))
    assert l + r > 0     # random wrist gears do flap: detector exercised
    assert env.pop_wrist_rev() == (0, 0)   # popped + zeroed


def test_rust_matches_reference_on_its_own_angles():
    beat_arena_rs = pytest.importorskip("beat_arena_rs")
    from beat_arena_env.chart import Note, note_tuple
    sim = beat_arena_rs.RawBeatSim(max_steps=100000,
                                   gears=5)
    sim.set_chart([note_tuple(Note(hit_time=30.0, col=2, row=1, color=1, angle=0.0, speed=4.0))], [])
    rng = np.random.default_rng(1)
    refs = [Ref(), Ref()]
    n = 0
    for t in range(240):
        a = [0] * N_JOINTS                  # coast except the wrists
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
    assert n >= 15, n            # coasts ~23 ticks before tipping over
