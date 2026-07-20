"""Env tests (SPEC.md §Tests 1-4, 6)."""

import numpy as np
import pytest

from beat_arena_env import (
    OBS_DIM, OFF_NOTES, OFF_NOTE_MASK, N_NOTE_SLOTS, NOTE_FEAT,
    BeatEnv, BeatSim, ChartConfig, Note, generate_chart, scripted_action,
)
from beat_arena_env.sim import DT, Z_WINDOW


def _sim_with_note(note):
    """Fresh sim whose chart is exactly [note]."""
    sim = BeatSim(ChartConfig(), seed=0)
    sim.chart = [note]
    sim.status = np.zeros(1, dtype=np.int64)
    return sim


# --- 1. obs layout -----------------------------------------------------------

def test_obs_layout():
    env = BeatEnv(seed=3)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) == (200,)
    assert OFF_NOTE_MASK == OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT == 188
    n_pending = min(len(env.sim.chart), N_NOTE_SLOTS)
    mask = obs[OFF_NOTE_MASK:]
    assert mask.sum() == n_pending
    # slot 0 = soonest note; valid flag set, features match the chart
    n = env.sim.chart[0]
    base = OFF_NOTES
    assert obs[base + 14] == 1.0
    assert obs[base + 0] == pytest.approx((n.col - 1.5) / 2.0)
    assert obs[base + 1] == pytest.approx((n.row - 1.0) / 1.5)
    z = min(n.hit_time * env.cfg.note_speed, 12.0)     # sim.time == 0 at reset
    assert obs[base + 2] == pytest.approx(z / 8.0)
    assert obs[base + 3] == pytest.approx(-env.cfg.note_speed / 8.0)
    assert obs[base + 6 + n.dir] == 1.0
    # empty slots are all-zero
    for slot in range(n_pending, N_NOTE_SLOTS):
        assert not obs[OFF_NOTES + slot * NOTE_FEAT:
                       OFF_NOTES + (slot + 1) * NOTE_FEAT].any()


# --- 2. determinism ----------------------------------------------------------

def test_determinism_scripted():
    def run():
        sim = BeatSim(ChartConfig(colors="both", note_rate=1.5), seed=7)
        rewards, events = [], []
        done = False
        while not done:
            r, done, ev = sim.step(*scripted_action(sim))
            rewards.append(r)
            events.append(ev)
        return rewards, events

    r1, e1 = run()
    r2, e2 = run()
    assert r1 == r2 and e1 == e2


# --- 3. cut detection scenarios ---------------------------------------------

def _step_events(sim, saber, pos, vel):
    sim.pos[saber] = np.array(pos, dtype=np.float64)
    sim.vel[saber] = np.array(vel, dtype=np.float64)
    _, _, events = sim.step(0, 0)
    return events


def test_good_cut():
    sim = _sim_with_note(Note(hit_time=DT, col=2, row=1, color=1, dir=6))
    ev = _step_events(sim, 1, pos=(0.25, 0.05), vel=(0.0, -5.0))
    assert ev == [{"cut": {"note": 0, "good": True, "reason": "ok"}}]
    assert sim.good_cuts == 1


def test_slow_cut_is_bad():
    sim = _sim_with_note(Note(hit_time=DT, col=2, row=1, color=1, dir=6))
    ev = _step_events(sim, 1, pos=(0.25, 0.05), vel=(0.0, -1.5))
    assert ev == [{"cut": {"note": 0, "good": False, "reason": "slow"}}]


def test_wrong_angle_is_bad():
    sim = _sim_with_note(Note(hit_time=DT, col=2, row=1, color=1, dir=6))
    ev = _step_events(sim, 1, pos=(0.25, -0.05), vel=(0.0, 5.0))  # swinging UP
    assert ev == [{"cut": {"note": 0, "good": False, "reason": "angle"}}]


def test_wrong_hand_is_bad():
    sim = _sim_with_note(Note(hit_time=DT, col=0, row=2, color=1, dir=6))
    sim.pos[1] = np.array([1.25, -1.0])  # matching (right) saber far away
    ev = _step_events(sim, 0, pos=(-0.75, 0.55), vel=(0.0, -5.0))
    assert ev == [{"cut": {"note": 0, "good": False, "reason": "wrong_hand"}}]


def test_miss_fires_exactly_at_window_expiry():
    # note far from both resting sabers, nobody moves
    sim = _sim_with_note(Note(hit_time=DT, col=0, row=2, color=1, dir=6))
    all_events = []
    for _ in range(8):
        _, _, ev = sim.step(0, 0)
        all_events.append(ev)
    miss_ticks = [i for i, ev in enumerate(all_events)
                  if any("miss" in e for e in ev)]
    assert len(miss_ticks) == 1
    t_miss = (miss_ticks[0] + 1) * DT
    w = Z_WINDOW / sim.cfg.note_speed         # slab half-depth in seconds
    assert t_miss - DT > w                     # strictly after window closes
    assert t_miss - DT <= w + DT + 1e-9        # by at most one tick
    assert sim.misses == 1


def test_early_crossing_does_nothing():
    sim = _sim_with_note(Note(hit_time=0.5, col=2, row=1, color=1, dir=6))
    ev = _step_events(sim, 1, pos=(0.25, 0.05), vel=(0.0, -5.0))
    assert ev == []
    assert sim.status[0] == 0  # still pending


# --- 4. chart generation -----------------------------------------------------

def test_chart_respects_hand_gap_and_dirs():
    cfg = ChartConfig(note_rate=3.0, colors="both", dirs=(2, 6),
                      min_hand_gap_s=0.5, chart_len_s=30.0)
    chart = generate_chart(cfg, np.random.default_rng(11))
    assert len(chart) > 20
    assert all(n.dir in (2, 6) for n in chart)
    for color in (0, 1):
        times = [n.hit_time for n in chart if n.color == color]
        assert all(b - a >= 0.5 - 1e-9 for a, b in zip(times, times[1:]))
    assert all(a.hit_time <= b.hit_time for a, b in zip(chart, chart[1:]))


# --- 6. scripted bot solvability --------------------------------------------

def test_scripted_bot_solves_easy():
    good, total = 0, 0
    for seed in range(5):
        sim = BeatSim(ChartConfig(), seed=seed)
        done = False
        while not done:
            _, done, _ = sim.step(*scripted_action(sim))
        good += sim.good_cuts
        total += len(sim.chart)
    assert good / total >= 0.8


# --- 7. shaping is potential-based: events identical, reward densified -------

def test_shaping_changes_rewards_not_events():
    def run(coef):
        sim = BeatSim(ChartConfig(shaping_coef=coef), seed=5)
        rewards, events = [], []
        done = False
        while not done:
            r, done, ev = sim.step(*scripted_action(sim))
            rewards.append(r)
            events.append(ev)
        return rewards, events

    r0, e0 = run(0.0)
    r1, e1 = run(0.2)
    assert e0 == e1                      # same trajectory, same cuts/misses
    assert r0 != r1                      # but the reward signal is denser
    # telescoping: total shaping contribution is small (bounded by phi range)
    assert abs(sum(r1) - sum(r0)) < 1.0


# --- 8. obs v2: z encodes time spatially -------------------------------------

def test_z_is_time_times_speed():
    """time_until = z / |vz| must hold at every step (the reparametrization
    that replaced the explicit time feature)."""
    from beat_arena_env.env import OFF_NOTES as ON, NOTE_FEAT as NF, Z_SCALE
    env = BeatEnv(seed=9)
    obs, _ = env.reset()
    for _ in range(60):
        obs, _, done, _, _ = env.step((0, 0))
        if done:
            break
        pending = env.sim.pending_notes()[:12]
        for slot, i in enumerate(pending):
            z_obs = obs[ON + slot * NF + 2] * Z_SCALE
            vz_obs = obs[ON + slot * NF + 3] * Z_SCALE
            t_until = env.sim.chart[i].hit_time - env.sim.time
            if -1.0 < t_until * env.cfg.note_speed < 12.0:  # inside clip range
                assert z_obs / -vz_obs == pytest.approx(t_until, abs=1e-4)


def test_note_speed_scales_the_time_window():
    """Faster notes spend less TIME in the fixed spatial slab."""
    for speed, ticks_expected in ((4.0, None), (8.0, None)):
        sim = BeatSim(ChartConfig(note_speed=speed), seed=0)
        # cell (0,2) center is far from both resting sabers
        sim.chart = [Note(hit_time=DT, col=0, row=2, color=1, dir=6)]
        sim.status = np.zeros(1, dtype=np.int64)
        t = 0
        while sim.status[0] == 0:
            sim.step(0, 0)
            t += 1
        if speed == 4.0:
            slow = t
        else:
            assert t < slow  # halves the slab-crossing time
