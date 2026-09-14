"""env v43: torque energy price (user direction 2026-09-14).

Contract: reward -= energy_coef * sum_j |tau_j| per tick, where
tau_j = gear_fraction_j * TAU_MAX[j] is exactly the torque the sims
apply (gears-5 table: 0 coast, +-1 full, +-1/4 quarter). Replaces the
v23 fraction price: a joint pays in proportion to its torque limit
(a spine full gear at 600 N.m costs 600/1.06x a wrist full gear).
Force-priced, velocity-blind; coast is the only free command. The
charge for a given action vector is a known constant, so it is pinned
by env pairs (coef 0 vs c) stepped with IDENTICAL actions — identical
physics, reward gap == the charge == c * popped sum|tau|.
energy_coef = 0 is the unpriced reward verbatim.
"""

import numpy as np
import pytest
import torch

import beat_arena_rs
from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import N_JOINTS, TAU_MAX

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="needs CUDA")

COEF = 1e-4
FAR_NOTE = Note(hit_time=200.0, col=2, row=1, x=0.3, y=1.4, color=1,
                angle=4.712, speed=4.0)
KW = dict(push_prob=0.0, push_vmax=0.0, start_jitter=0.0)


def _rust(**kw):
    # weld mode: the charge is COMMAND-based; contact-mode terminals
    # (reward = -fall_const replaces the tick) could erase a gap
    sim = beat_arena_rs.RawBeatSim(max_steps=6900, wide_acc=False,
                                   gears=5, foot_contact=False, **kw)
    sim.set_chart([note_tuple(FAR_NOTE)])
    return sim


def _rust_gap(act, ticks=3):
    s0, s1 = _rust(), _rust(energy_coef=COEF)
    g = 0.0
    for _ in range(ticks):
        _, r0, _, _ = s0.step(act)
        _, r1, _, _ = s1.step(act)
        g += r0 - r1
    return g, s0.pop_torque(), s1.pop_torque()


def test_rust_price_identity_and_telemetry():
    act = [0] * N_JOINTS
    act[0] = 1      # spine pitch full +
    act[7] = 2      # l wrist roll full -
    act[20] = 4     # l knee -1/4
    act[10] = 3     # r arm pitch +1/4
    gap, t0, t1 = _rust_gap(act)
    assert t0 == t1, "the price does not touch the telemetry"
    tot, sp, ar, lg, n = t0
    expect_tick = (TAU_MAX[0] + TAU_MAX[7] + 0.25 * TAU_MAX[20]
                   + 0.25 * TAU_MAX[10])
    assert n == 3 and abs(tot - 3 * expect_tick) < 1e-9
    assert abs(sp - 3 * TAU_MAX[0]) < 1e-9
    assert abs(ar - 3 * (TAU_MAX[7] + 0.25 * TAU_MAX[10])) < 1e-9
    assert abs(lg - 3 * 0.25 * TAU_MAX[20]) < 1e-9
    # f64-exact identity: gap == coef * popped sum|tau|
    assert abs(gap - COEF * tot) < 1e-12, (gap, COEF * tot)


def test_rust_coast_is_free_and_wrist_vs_spine_ratio():
    gap, t0, _ = _rust_gap([0] * N_JOINTS)
    assert gap == 0.0 and t0[0] == 0.0
    w = [0] * N_JOINTS
    w[7] = 1
    s = [0] * N_JOINTS
    s[0] = 1
    gw, tw, _ = _rust_gap(w, ticks=1)
    gs, ts, _ = _rust_gap(s, ticks=1)
    assert abs(gw / gs - TAU_MAX[7] / TAU_MAX[0]) < 1e-9
    assert abs(tw[0] - TAU_MAX[7]) < 1e-12 and abs(ts[0] - TAU_MAX[0]) < 1e-12


def _cuda_pair():
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    kw = dict(chart_len_s=8.0, seed=0, device="cuda", foot_contact=False,
              gears=5, **KW)
    env0 = CudaBeatVecEnv(64, **kw)
    env1 = CudaBeatVecEnv(64, energy_coef=COEF, **kw)
    env0.reset()
    env1.reset()
    return env0, env1


def _cuda_gap(env0, env1, act):
    _, r0, _, _ = env0.step(act)
    _, r1, _, _ = env1.step(act)
    return r0 - r1


@cuda
def test_cuda_full_gear_pays_sum_tau_max():
    env0, env1 = _cuda_pair()
    act = torch.ones_like(env0._zero_act)          # id 1 = +full, all joints
    gap = _cuda_gap(env0, env1, act)
    expect = COEF * float(sum(TAU_MAX))
    assert torch.allclose(gap, torch.full_like(gap, expect), atol=1e-5), (
        gap[:3].tolist(), expect)
    tot, sp, ar, lg, n = env1.pop_torque()
    assert n == 64 and abs(tot / 64 - sum(TAU_MAX)) < 1e-3
    assert abs(sp / 64 - sum(TAU_MAX[:3])) < 1e-3
    assert abs(ar / 64 - sum(TAU_MAX[3:17])) < 1e-3
    assert abs(lg / 64 - sum(TAU_MAX[17:])) < 1e-3
    assert env0.pop_torque()[0] == pytest.approx(tot, rel=1e-6)


@cuda
def test_cuda_quarter_and_coast_and_wrist_ratio():
    env0, env1 = _cuda_pair()
    gap = _cuda_gap(env0, env1, env0._zero_act)
    assert gap.abs().max().item() < 1e-7
    act = torch.full_like(env0._zero_act, 3)
    act[:, 1::2] = 4
    gap = _cuda_gap(env0, env1, act)
    expect = COEF * 0.25 * float(sum(TAU_MAX))
    assert torch.allclose(gap, torch.full_like(gap, expect), atol=1e-5)
    w = env0._zero_act.clone(); w[:, 7] = 1
    s = env0._zero_act.clone(); s[:, 0] = 1
    gw = _cuda_gap(env0, env1, w).mean().item()
    gs = _cuda_gap(env0, env1, s).mean().item()
    assert abs(gw / gs - TAU_MAX[7] / TAU_MAX[0]) < 1e-3
