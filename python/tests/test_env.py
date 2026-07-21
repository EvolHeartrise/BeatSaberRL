"""Env tests against the Rust sim — THE implementation (SPEC.md §Tests).

The Python sim was deleted 2026-07-20 (surface-area reduction); the cut
scenarios that used to gate Python/Rust parity now gate the Rust sim
directly, via its set_saber test hook and event stream.
"""

import numpy as np
import pytest

from beat_arena_env import (
    OBS_DIM, OFF_NOTES, OFF_NOTE_MASK, N_NOTE_SLOTS, NOTE_FEAT,
    ChartConfig, Note, RustBeatEnv, generate_chart, scripted_action,
)
from beat_arena_env.constants import DT, Z_WINDOW

from beat_arena_rs import RawBeatSim


DOWN = 6 * np.pi / 4  # downward swing angle


def _note(hit_time, col, row, color=1, angle=DOWN, speed=4.0):
    return Note(hit_time=hit_time, col=col, row=row, color=color,
                angle=angle, speed=speed)


def _sim_with_note(note):
    sim = RawBeatSim(shaping_coef=0.0, max_steps=630)
    sim.set_chart([(note.hit_time, note.col, note.row, note.color,
                    note.angle, note.speed)])
    return sim


def _step_events(sim, saber, pos, vel):
    sim.set_saber(saber, pos[0], pos[1], vel[0], vel[1])
    _, _, _, events = sim.step(0, 0)
    return events


# --- obs layout --------------------------------------------------------------

def test_obs_layout():
    env = RustBeatEnv(seed=3)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) == (128,)
    assert OFF_NOTE_MASK == OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT == 116
    n_pending = min(len(env.chart), N_NOTE_SLOTS)
    assert obs[OFF_NOTE_MASK:].sum() == n_pending
    n = env.chart[0]
    base = OFF_NOTES
    assert obs[base + 8] == 1.0
    assert obs[base + 0] == pytest.approx((n.col - 1.5) / 2.0)
    assert obs[base + 1] == pytest.approx((n.row - 1.0) / 1.5)
    z = min(n.hit_time * n.speed, 12.0)
    assert obs[base + 2] == pytest.approx(z / 8.0)
    assert obs[base + 3] == pytest.approx(-n.speed / 8.0)
    assert obs[base + 6] == pytest.approx(np.cos(n.angle), abs=1e-6)
    assert obs[base + 7] == pytest.approx(np.sin(n.angle), abs=1e-6)
    for slot in range(n_pending, N_NOTE_SLOTS):
        assert not obs[OFF_NOTES + slot * NOTE_FEAT:
                       OFF_NOTES + (slot + 1) * NOTE_FEAT].any()


def test_z_is_time_times_speed():
    """time_until = z / |vz| at every step (spatial obs contract)."""
    env = RustBeatEnv(seed=9)
    obs, _ = env.reset()
    for _ in range(60):
        obs, _, done, _, _ = env.step((0, 0))
        if done:
            break
        status = env.sim.status()
        pending = [i for i, s in enumerate(status) if s == 0][:12]
        for slot, i in enumerate(pending):
            z_obs = obs[OFF_NOTES + slot * NOTE_FEAT + 2] * 8.0
            vz_obs = obs[OFF_NOTES + slot * NOTE_FEAT + 3] * 8.0
            t_until = env.chart[i].hit_time - env.sim.time
            if -1.0 < t_until * env.chart[i].speed < 12.0:
                assert z_obs / -vz_obs == pytest.approx(t_until, abs=1e-4)


# --- determinism -------------------------------------------------------------

def test_determinism():
    def run():
        env = RustBeatEnv(colors="both", note_rate=1.5, seed=7)
        env.reset(seed=7)
        rng = np.random.default_rng(1)
        out = []
        done = False
        while not done:
            _, r, done, _, info = env.step(rng.integers(0, 9, size=2))
            out.append((r, tuple(info["events"])))
        return out

    assert run() == run()


# --- cut detection scenarios (Rust sim, via set_saber hook) ------------------

def test_good_cut():
    sim = _sim_with_note(_note(DT, 2, 1))
    assert _step_events(sim, 1, (0.25, 0.05), (0.0, -5.0)) == [(0, 0)]  # ok
    assert sim.good_cuts == 1


def test_gentle_cut_is_good():
    """V_MIN_CUT is 1 m/s: a 1.5 m/s swing scores."""
    sim = _sim_with_note(_note(DT, 2, 1))
    assert _step_events(sim, 1, (0.25, 0.05), (0.0, -1.5)) == [(0, 0)]


def test_slow_cut_is_bad():
    sim = _sim_with_note(_note(DT, 2, 1))
    assert _step_events(sim, 1, (0.25, 0.05), (0.0, -0.5)) == [(0, 1)]  # slow


def test_wrong_angle_is_bad():
    sim = _sim_with_note(_note(DT, 2, 1))
    assert _step_events(sim, 1, (0.25, -0.05), (0.0, 5.0)) == [(0, 2)]  # angle


def test_wrong_hand_is_bad():
    sim = _sim_with_note(_note(DT, 0, 2))
    sim.set_saber(1, 1.25, -1.0, 0.0, 0.0)   # matching (right) saber far away
    assert _step_events(sim, 0, (-0.75, 0.55), (0.0, -5.0)) == [(0, 3)]


def test_miss_fires_exactly_at_slab_exit():
    sim = _sim_with_note(_note(DT, 0, 2))
    events = []
    for _ in range(8):
        _, _, _, ev = sim.step(0, 0)
        events.append(ev)
    miss_ticks = [i for i, ev in enumerate(events) if (0, 4) in ev]
    assert len(miss_ticks) == 1
    t_miss = (miss_ticks[0] + 1) * DT
    w = Z_WINDOW / 4.0
    assert w < t_miss - DT <= w + DT + 1e-9
    assert sim.misses == 1


def test_early_crossing_does_nothing():
    sim = _sim_with_note(_note(0.5, 2, 1))
    assert _step_events(sim, 1, (0.25, 0.05), (0.0, -5.0)) == []
    assert sim.status()[0] == 0


def test_note_speed_scales_the_time_window():
    """Faster notes spend less TIME in the fixed spatial slab."""
    ticks = {}
    for speed in (4.0, 8.0):
        sim = _sim_with_note(_note(DT, 0, 2, speed=speed))
        t = 0
        while sim.status()[0] == 0:
            sim.step(0, 0)
            t += 1
        ticks[speed] = t
    assert ticks[8.0] < ticks[4.0]


# --- shaping (ablation-only mechanism, still contract-tested) ---------------

def test_shaping_changes_rewards_not_events():
    def run(coef):
        sim = RawBeatSim(shaping_coef=coef, max_steps=630)
        chart = generate_chart(ChartConfig(), np.random.default_rng(5))
        sim.set_chart([(n.hit_time, n.col, n.row, n.color,
                        n.angle, n.speed) for n in chart])
        rng = np.random.default_rng(2)
        rewards, events = [], []
        done = False
        while not done:
            a = rng.integers(0, 9, size=2)
            _, r, done, ev = sim.step(int(a[0]), int(a[1]))
            rewards.append(r)
            events.append(ev)
        return rewards, events

    r0, e0 = run(0.0)
    r1, e1 = run(0.2)
    assert e0 == e1
    assert r0 != r1
    assert abs(sum(r1) - sum(r0)) < 1.0


# --- chart generation --------------------------------------------------------

def test_chart_respects_hand_gap_and_dirs():
    cfg = ChartConfig(note_rate=3.0, colors="both", dirs=(2, 6),
                      min_hand_gap_s=0.5, chart_len_s=30.0)
    chart = generate_chart(cfg, np.random.default_rng(11))
    assert len(chart) > 20
    assert all(n.angle in (2 * np.pi / 4, 6 * np.pi / 4) for n in chart)
    for color in (0, 1):
        times = [n.hit_time for n in chart if n.color == color]
        assert all(b - a >= 0.5 - 1e-9 for a, b in zip(times, times[1:]))
    assert all(a.hit_time <= b.hit_time for a, b in zip(chart, chart[1:]))


def test_chart_continuous_dirs_and_speed_range():
    cfg = ChartConfig(note_rate=2.0, colors="both", continuous_dirs=True,
                      note_speed=3.0, note_speed_max=6.0, chart_len_s=30.0)
    chart = generate_chart(cfg, np.random.default_rng(13))
    assert len(chart) > 20
    assert all(0.0 <= n.angle < 2 * np.pi for n in chart)
    assert all(3.0 <= n.speed <= 6.0 for n in chart)
    assert len({round(n.angle, 3) for n in chart}) > 10   # actually continuous
    assert len({round(n.speed, 3) for n in chart}) > 10


# --- scripted bot solvability ------------------------------------------------

def test_scripted_bot_solves_easy():
    good, total = 0, 0
    env = RustBeatEnv(seed=0)
    for seed in range(5):
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, done, _, info = env.step(scripted_action(env))
        good += info["good_cuts"]
        total += info["n_notes"]
    assert good / total >= 0.8


# --- recording (JSONL v2 from the Rust event stream) ------------------------

def test_recording_roundtrip(tmp_path):
    import json
    path = str(tmp_path / "ep.jsonl")
    env = RustBeatEnv(seed=5, chart_len_s=6.0)
    env.set_record(path)
    env.reset(seed=5)
    done = False
    while not done:
        _, _, done, _, info = env.step(scripted_action(env))
    lines = [json.loads(l) for l in open(path) if l.strip()]
    head, ticks = lines[0], lines[1:]
    assert head["header"]["version"] == 3
    assert head["header"]["config"]["note_speed"] == 4.0
    assert len(head["header"]["chart"]) == info["n_notes"]
    goods = sum(1 for t in ticks for e in t["events"]
                if "cut" in e and e["cut"]["good"])
    assert goods == info["good_cuts"]
    assert all(len(t["sabers"]) == 2 and len(t["sabers"][0]) == 4 for t in ticks)
