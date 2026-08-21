"""env v34: joint-limit-rest penalty (user direction 2026-08-21).

Contract: reward -= limit_coef * (tick-summed solver limit-row
impulses lambda, N*m*s, over the POSTURAL scope: spine joints 0..3 +
leg joints 17..29; the 14 arm joints are exempt — whips legitimately
slam arm stops at speed). lambda is the impulse the stop actually
carried, so resting a pose on its stops pays its gravity moment while
an active hold (hovering off the stop) pays ~nothing. Telemetry
(pop_limit_rest -> (legs, spine, ticks)) accumulates regardless of
the coefficient. limit_coef = 0 (the default everywhere) leaves the
reward stream exactly as before: the penalty term is 0 * flow and the
accumulators have no other reader.

The scripted fixtures below drive the DYNAMIC body: knees commanded to
full flexion collapse the body onto the knee/hip stops (large sustained
lambda); the same body left coasting from the reset stance carries far
less. The pricing identity is pinned by env pairs (coef 0 vs c) stepped
with IDENTICAL actions — the penalty does not alter the physics, so the
reward gap must equal c * (the popped flow) exactly in f64.
"""

import numpy as np
import pytest
import torch

import beat_arena_rs
from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import N_JOINTS

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="needs CUDA")

FAR_NOTE = Note(hit_time=200.0, col=2, row=1, x=0.3, y=1.4, color=1,
                angle=4.712, speed=4.0)
COEF = 0.01
L_KNEE, R_KNEE = 20, 26
L_HIP_P, R_HIP_P = 17, 23
R_ELBOW = 13
SPINE_P = 0


def _sim(**kw):
    sim = beat_arena_rs.RawBeatSim(max_steps=6900, effort_coef=0.0,
                                   wide_acc=False, gears=5, **kw)
    sim.set_chart([note_tuple(FAR_NOTE)])
    return sim


def _run(sim, act, ticks):
    total, t = 0.0, 0
    for _ in range(ticks):
        _obs, r, done, _ev = sim.step(act)
        total += r
        t += 1
        if done:
            break
    return total, t


def _crouch_act():
    act = [0] * N_JOINTS
    for j in (L_KNEE, R_KNEE, L_HIP_P, R_HIP_P):
        act[j] = 1                    # full flex — collapse onto stops
    return act


def test_crouch_rests_on_leg_stops_and_upright_does_not():
    sim = _sim(foot_contact=True)
    _run(sim, _crouch_act(), 60)
    legs_c, _spine_c, ticks_c = sim.pop_limit_rest()
    assert ticks_c > 0
    sim2 = _sim(foot_contact=True)
    _run(sim2, [0] * N_JOINTS, ticks_c)
    legs_u, _spine_u, ticks_u = sim2.pop_limit_rest()
    assert ticks_u > 0
    # the collapsed pose parks its weight on the knee/hip stops; the
    # coasting reset stance may graze its stops but carries far less
    assert legs_c > 1.0, f"crouch legs flow {legs_c}"
    assert legs_c > 5.0 * max(legs_u, 1e-9), \
        f"crouch {legs_c} vs upright {legs_u} (ticks {ticks_c}/{ticks_u})"


def test_pricing_gap_equals_coef_times_flow():
    # WELD mode: fall/tilt are structurally impossible, so no tick's
    # reward is replaced by the terminal override (which would erase
    # that tick's penalty gap while the pop still counts its lambda —
    # the contact-mode fixture tripped exactly that)
    act = [0] * N_JOINTS
    act[L_KNEE] = act[R_KNEE] = 1     # 150 N*m into the flexion stop
    s0 = _sim(foot_contact=False)
    s1 = _sim(foot_contact=False, limit_coef=COEF)
    r0, t0 = _run(s0, act, 60)
    r1, t1 = _run(s1, act, 60)
    assert t0 == t1               # identical physics, identical episode
    legs, spine, _ = s0.pop_limit_rest()
    assert legs + spine > 1.0
    # the penalty is the ONLY reward difference, and it prices exactly
    # the popped flow (same trajectory, f64)
    assert r0 - r1 == pytest.approx(COEF * (legs + spine), abs=1e-9)


def test_coef_default_is_off():
    act = _crouch_act()
    r_default, _ = _run(_sim(foot_contact=True), act, 40)
    r_zero, _ = _run(_sim(foot_contact=True, limit_coef=0.0), act, 40)
    assert r_default == r_zero


def test_spine_stop_lands_in_spine_bucket():
    sim = _sim(foot_contact=False)          # weld: isolate the spine
    act = [0] * N_JOINTS
    act[SPINE_P] = 2                        # full extension, -20 deg stop
    _run(sim, act, 40)
    _legs, spine, _ = sim.pop_limit_rest()
    assert spine > 0.1, f"spine flow {spine}"


def test_arm_stops_are_exempt():
    # Single-tick isolation: every POSTURAL joint starts >= 17 deg off
    # its stops, where a limit row cannot engage within one tick from
    # rest (assembly needs gap < H_SUB * omega_cap = 3.8 deg) — while
    # the right elbow starts ON its stop, pressed in by a full gear.
    # If arm rows were in scope the pop would show their lambda; the
    # contract is that it shows EXACTLY zero.
    from beat_arena_env.constants import JOINT_LIMITS
    # CALM start: the weld reset pose with zero velocity (a mid-range-
    # everything pose is violently out of equilibrium — the whole body
    # whips at the omega-cap and flings the elbow off any stop). The
    # only postural joints the weld reset rests on stops are the KNEES
    # (0-deg extension) — nudge those 17 deg off; from REST nothing
    # postural can close a >=17 deg gap within one tick. The right
    # elbow goes ON its 0-deg stop with a full gear pressing in
    # (gear id 2 = -full drives q- into the extension stop; verified
    # empirically in the calm pose — id 1 flexes away at ~13 rad/s).
    act = [0] * N_JOINTS
    act[R_ELBOW] = 2
    for coef in (0.0, 1.0):
        sim = _sim(foot_contact=False, limit_coef=coef)
        q = list(sim.qpos())
        q[7 + L_KNEE] = q[7 + R_KNEE] = 0.3
        q[7 + R_ELBOW] = JOINT_LIMITS[R_ELBOW][0]
        sim.set_joints(q, [0.0] * 35)
        sim.pop_limit_rest()                    # discard reset transient
        _obs, r, _done, _ev = sim.step(act)
        legs, spine, ticks = sim.pop_limit_rest()
        assert ticks == 1
        assert legs == 0.0 and spine == 0.0, (legs, spine)
        # elbow stayed pinned (its row genuinely carried this tick)
        assert sim.qpos()[7 + R_ELBOW] < JOINT_LIMITS[R_ELBOW][0] + 0.05
        if coef == 0.0:
            r0 = r
    # a HUGE coef changes nothing when only exempt rows carry
    assert r == r0


@cuda
def test_cuda_flow_and_pricing_consistency():
    from beat_arena_env.cuda_env import CudaBeatVecEnv

    def env(**kw):
        kw.setdefault("foot_contact", True)
        e = CudaBeatVecEnv(1, charts=[[note_tuple(FAR_NOTE)]], seed=0,
                           device="cuda", gears=5,
                           max_steps=6900, push_prob=0.0, push_vmax=0.0,
                           start_jitter=0.0, auto_reset=False, **kw)
        e.reset()
        return e

    # (a) pricing identity in WELD mode (no terminal overrides)
    act = torch.zeros((1, N_JOINTS), dtype=torch.int64, device="cuda")
    act[0, L_KNEE] = act[0, R_KNEE] = 1
    w0 = env(foot_contact=False)
    w1 = env(foot_contact=False, limit_coef=COEF)
    tot0 = tot1 = 0.0
    for _ in range(60):
        _o, r0, _d, _i = w0.step(act)
        _o, r1, _d1, _i = w1.step(act)
        tot0 += float(r0[0].item())
        tot1 += float(r1[0].item())
    legs_w, spine_w, ticks_w = w0.pop_limit_rest()
    assert ticks_w > 0 and legs_w > 1.0
    assert tot0 - tot1 == pytest.approx(COEF * (legs_w + spine_w),
                                        rel=5e-3)
    # (b) contact mode: the collapsed pose carries far more than coast
    cact = act.clone()
    cact[0, L_HIP_P] = cact[0, R_HIP_P] = 1
    e0 = env()
    ticks = 0
    for _ in range(60):
        _o, _r, d0, _i = e0.step(cact)
        ticks += 1
        if bool(d0[0].item()):
            break
    legs, spine, _t = e0.pop_limit_rest()
    assert legs > 1.0
    eu = env()
    for _ in range(ticks):
        eu.step(torch.zeros_like(act))
    legs_u, _s, _t = eu.pop_limit_rest()
    assert legs > 5.0 * max(legs_u, 1e-9)
    # cross-engine magnitude sanity vs the Rust fixture (statics
    # dominate the parked pose, so the engines agree to a factor)
    rs = _sim(foot_contact=True)
    _run(rs, _crouch_act(), 60)
    legs_r, spine_r, _ = rs.pop_limit_rest()
    ratio = (legs + spine) / max(legs_r + spine_r, 1e-9)
    assert 0.5 < ratio < 2.0, f"CUDA/Rust flow ratio {ratio}"
