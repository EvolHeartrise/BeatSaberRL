"""env v41: self-contact penalty (user direction 2026-09-12).

Contract: reward -= self_contact_coef * (tick-summed self-collision row
impulses, N*s, over all 55 capsule pairs — the same lambdas the replay
"contacts" carry). Telemetry (pop_self_contact -> (impulse, ticks with
contact, ticks)) accumulates regardless of the coefficient, and
self_contact_coef = 0 (the default everywhere) leaves the reward stream
exactly as before.

Fixture: the parity gate's "midline press" — both arms swung across the
chest from a forward-down pitch drives the forearm/fist vs torso and
head pairs into contact. The pricing identity is pinned by env pairs
(coef 0 vs c) stepped with IDENTICAL actions — the penalty does not
alter the physics, so the reward gap must equal c * (popped impulse)
exactly in f64 (Rust) / to float tolerance (CUDA).
"""
import numpy as np
import torch

import beat_arena_rs
from beat_arena_env.chart import note_tuple
from beat_arena_env.constants import N_JOINTS
from tests._helpers import FAR_NOTE, cuda

COEF = 0.05
NV = 35
TICKS = 90


def _sim(**kw):
    sim = beat_arena_rs.RawBeatSim(max_steps=6900,
                                   gears=5,
                                   **kw)
    sim.set_chart([note_tuple(FAR_NOTE)])
    return sim


def _press_pose():
    q = np.zeros(N_JOINTS)
    q[3] = -0.9    # l_arm_pitch: arm forward-down
    q[10] = -0.9   # r_arm_pitch
    q36 = [0.0, 0.94, -0.65, 1.0, 0.0, 0.0, 0.0] + [float(x) for x in q]
    return q36, [0.0] * NV


def _press_act():
    act = [0] * N_JOINTS
    act[4] = 2    # l_arm_yaw toward the midline
    act[11] = 1   # r_arm_yaw toward the midline
    return act


def _run(sim, act, ticks):
    """Returns (summed reward, ticks actually stepped) — the press pose
    ends the episode early under ground contact, so tick counts are
    measured, not assumed."""
    total, n = 0.0, 0
    for _ in range(ticks):
        _obs, r, done, _ev = sim.step(act)
        total += r
        n += 1
        if done:
            break
    return total, n


def test_press_produces_contact_and_telemetry():
    sim = _sim()
    q36, v35 = _press_pose()
    sim.set_joints(q36, v35)
    _tot, n = _run(sim, _press_act(), TICKS)
    imp, ct, t = sim.pop_self_contact()
    assert t == n and n > 0
    assert imp > 0.0 and 0 < ct <= t, (imp, ct, t)
    # popped: a second pop is empty
    assert sim.pop_self_contact() == (0.0, 0, 0)


def test_coef_zero_is_bitwise_baseline_and_pricing_identity():
    s0, s1 = _sim(), _sim(self_contact_coef=COEF)
    q36, v35 = _press_pose()
    s0.set_joints(q36, v35)
    s1.set_joints(q36, v35)
    act = _press_act()
    # per tick: the penalty does not touch the physics (identical
    # telemetry), and on every NON-terminal tick the reward gap is c x
    # that tick's impulse — a terminal (fall / tilt) tick's reward is
    # the flat -fall_const in both sims, so its impulse goes unpriced
    gap, priced, total_imp = 0.0, 0.0, 0.0
    for _ in range(TICKS):
        _o0, r0, d0, _e0 = s0.step(act)
        _o1, r1, d1, _e1 = s1.step(act)
        assert d0 == d1
        imp0, ct0, t0 = s0.pop_self_contact()
        imp1, ct1, t1 = s1.pop_self_contact()
        assert (imp0, ct0, t0) == (imp1, ct1, t1)
        total_imp += imp0
        if d0:
            assert r0 == r1
            break
        gap += r0 - r1
        priced += COEF * imp0
    assert total_imp > 0.0
    assert priced > 0.0
    # reward gap == c * impulse, f64-exact up to summation order
    assert abs(gap - priced) < 1e-9, (gap, priced)


def test_zero_contact_ticks_are_unpriced():
    """The first ticks from the reset stance carry no self-contact (the
    unactuated body only collapses into itself later): coef c and coef 0
    agree exactly while the telemetry reads zero."""
    s0, s1 = _sim(), _sim(self_contact_coef=COEF)
    act = [0] * N_JOINTS
    for _ in range(3):
        _o, r0, _d, _e = s0.step(act)
        _o, r1, _d, _e = s1.step(act)
        assert r0 == r1
    imp, ct, t = s1.pop_self_contact()
    assert t == 3 and imp == 0.0 and ct == 0, (imp, ct, t)


@cuda
def test_cuda_pricing_matches_telemetry():
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    n = 4
    charts = [[FAR_NOTE]] * n
    kw = dict(charts=charts, gears=5, max_steps=6900, device="cuda", dtype=torch.float64,
              auto_reset=False)
    e0 = CudaBeatVecEnv(n, **kw)
    e1 = CudaBeatVecEnv(n, self_contact_coef=COEF, **kw)
    q36, v35 = _press_pose()
    for env in (e0, e1):
        env.reset()
        env.set_state(torch.tensor([q36] * n, dtype=torch.float64),
                      torch.tensor([v35] * n, dtype=torch.float64))
    act = torch.as_tensor([_press_act()] * n)
    # per tick, like the Rust test above: identical telemetry, and on
    # every NON-terminal tick the reward gap is c x that tick's impulse
    # (all n envs are copies, so they terminate together). A terminal
    # tick pays the flat -fall_const in both envs, so its impulse goes
    # unpriced; stop there — with auto_reset off the kernel keeps the
    # telemetry accumulating on done envs while no reward is paid.
    gap, priced, total_imp, total_ct = 0.0, 0.0, 0.0, 0
    for _ in range(TICKS):
        _o, r0, d0, _ = e0.step(act)
        _o, r1, d1, _ = e1.step(act)
        assert torch.equal(d0, d1)
        imp0, ct0, t0 = e0.pop_self_contact()
        imp1, ct1, t1 = e1.pop_self_contact()
        assert t0 == t1 == n and ct0 == ct1 and abs(imp0 - imp1) < 1e-9
        total_imp += imp0
        total_ct += ct0
        if bool(d0.any()):
            assert bool(d0.all()) and torch.equal(r0, r1)
            break
        gap += float((r0.double() - r1.double()).sum().cpu())
        priced += COEF * imp0
    assert total_imp > 0.0 and total_ct > 0
    assert priced > 0.0
    assert abs(gap - priced) < 1e-6, (gap, priced)
