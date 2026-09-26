"""Flux game mode (SPEC.md §Flux; rust/src/flux.rs): generator shape,
judgment on the live sim (circle touch / wrong hand / bar crossing and
its sweet spot / misses), obs lanes, mode separation, vec == single.

The live-sim cases run in weld mode (the test harness body that never
falls) with coast actions and place each object so that its hitbox
slab holds the palm on the FIRST tick — the palm barely moves in one
1/30 s tick, so the expected points are exact."""
import math

import numpy as np
import pytest

from beat_arena_env.constants import (
    KIND_BAR, KIND_CIRCLE, N_JOINTS, NOTE_FEAT, OFF_HAND_L, OFF_HAND_R,
    OFF_NOTE_MASK, OFF_NOTES, OFF_PREV_R, PLANE_Z0, ROW_Y, X_NORM, Y_NORM,
    col_x)
from beat_arena_env.flux_chart import (FluxConfig, FluxObj, bar_axis,
                                       flux_tuple, generate_flux_chart,
                                       sweet_point)
from beat_arena_env.flux_env import FluxVecEnv, RustFluxEnv

from beat_arena_rs import RawBeatSim, VecBeatSim

ZERO = [0] * N_JOINTS
DT = 1.0 / 30.0
SPEED = 12.0


def _sim():
    return RawBeatSim(max_steps=400, foot_contact=False, flux=True)


def _arrive_now(sim, palm_z, lead=0.2):
    """hit_time putting the object centre `lead` m beyond the palm on
    tick 1 (so the palm sits inside the [-0.75, +0.25] slab)."""
    z_off = sim.head_pose()[2] - PLANE_Z0
    return DT + (palm_z + lead - z_off) / SPEED


def _run(sim, chart, ticks=30):
    sim.set_flux_chart([flux_tuple(o) for o in chart])
    events, total = [], 0.0
    for _ in range(ticks):
        _obs, r, done, ev = sim.step(ZERO)
        events += ev
        total += r
        if done:
            break
    return events, total


def _reset_palms():
    sim = _sim()
    sim.set_flux_chart([])
    return sim, sim.palms()


# -- generator ---------------------------------------------------------------

def test_generator_deterministic_sorted_and_in_range():
    cfg = FluxConfig()
    a = generate_flux_chart(cfg, np.random.default_rng(7))
    b = generate_flux_chart(cfg, np.random.default_rng(7))
    assert a == b and len(a) > 20
    assert all(x.hit_time <= y.hit_time for x, y in zip(a, a[1:]))
    kinds = {o.kind for o in a}
    assert kinds == {KIND_CIRCLE, KIND_BAR}
    x_lo, x_hi = col_x(0), col_x(3)
    y_lo, y_hi = ROW_Y[0], ROW_Y[2]
    for o in a:
        assert cfg.note_speed_min <= o.speed <= cfg.note_speed_max
        if o.kind == KIND_CIRCLE:
            assert o.color in (0, 1)
            assert x_lo <= o.x <= x_hi and y_lo <= o.y <= y_hi
        else:
            assert o.color == 0
            assert cfg.bar_len_min <= o.length <= cfg.bar_len_max
            assert -1.0 <= o.sweet <= 1.0
            # the sweet spot sits on the imaginary palms' segment, which
            # lives inside the rectangle
            sx, sy = sweet_point(o)
            assert x_lo - 1e-9 <= sx <= x_hi + 1e-9
            assert y_lo - 1e-9 <= sy <= y_hi + 1e-9


def test_generator_left_circles_stay_left_of_same_tick_right_circles():
    cfg = FluxConfig(circle_prob=1.0, bar_prob=0.0)
    ch = generate_flux_chart(cfg, np.random.default_rng(3))
    by_t = {}
    for o in ch:
        by_t.setdefault(o.hit_time, {})[o.color] = o.x
    pairs = [v for v in by_t.values() if len(v) == 2]
    assert pairs
    assert all(v[1] - v[0] >= cfg.min_sep - 1e-9 for v in pairs)


# -- judgment ------------------------------------------------------------------

def test_circle_touched_by_its_hand_scores_60():
    sim, palms = _reset_palms()
    (rx, ry, rz) = palms[1]
    c = FluxObj(hit_time=_arrive_now(sim, rz), x=rx, y=ry,
                kind=KIND_CIRCLE, speed=SPEED, color=1)
    events, total = _run(sim, [c])
    goods = [e for e in events if e[1] == 0]
    assert len(goods) == 1
    _i, _code, pts, base, acc = goods[0]
    assert (pts, base, acc) == (60.0, 50.0, 10.0)
    assert sim.good_cuts == 1 and sim.misses == 0
    assert sim.score == 60.0
    assert total == pytest.approx(60.0 / 115.0, abs=1e-9)


def test_circle_accuracy_uses_the_existing_formula():
    sim, palms = _reset_palms()
    (rx, ry, rz) = palms[1]
    # centre 0.15 m off the palm (inside the 0.175 radius):
    # round(10 * (1 - 0.15 / 0.3)) = 5
    c = FluxObj(hit_time=_arrive_now(sim, rz), x=rx + 0.15, y=ry,
                kind=KIND_CIRCLE, speed=SPEED, color=1)
    events, _ = _run(sim, [c])
    goods = [e for e in events if e[1] == 0]
    assert [g[2:] for g in goods] == [(55.0, 50.0, 5.0)]


def test_circle_touched_by_the_other_hand_is_a_bad_hit():
    sim, palms = _reset_palms()
    (rx, ry, rz) = palms[1]
    c = FluxObj(hit_time=_arrive_now(sim, rz), x=rx, y=ry,
                kind=KIND_CIRCLE, speed=SPEED, color=0)   # LEFT circle
    events, total = _run(sim, [c])
    assert [e[1] for e in events] == [3]                 # wrong_hand
    assert sim.bad_cuts == 1 and sim.good_cuts == 0
    assert total == pytest.approx(-0.5)


def test_untouched_circle_is_missed_at_the_head_plane():
    sim, palms = _reset_palms()
    c = FluxObj(hit_time=0.3, x=0.9, y=ROW_Y[2], kind=KIND_CIRCLE,
                speed=SPEED, color=1)
    events, total = _run(sim, [c])
    assert [e[1] for e in events] == [4]
    assert sim.misses == 1 and total == pytest.approx(-0.5)


def _bar_across_palms(sim, palms, length, sweet, dy=0.0):
    """A bar perpendicular to the palm-to-palm line, centred on its
    midpoint (shifted dy along the bar axis)."""
    (lx, ly, lz), (rx, ry, rz) = palms
    fx, fy = rx - lx, ry - ly
    n = math.hypot(fx, fy)
    ux, uy = -fy / n, fx / n                    # perpendicular axis
    rot = math.atan2(-ux, uy)                   # u = (-sin, cos)
    assert bar_axis(rot) == pytest.approx((ux, uy))
    mx, my = 0.5 * (lx + rx), 0.5 * (ly + ry)
    return FluxObj(hit_time=_arrive_now(sim, 0.5 * (lz + rz)),
                   x=mx + dy * ux, y=my + dy * uy, kind=KIND_BAR,
                   speed=SPEED, rot=rot, length=length, sweet=sweet)


def test_bar_crossed_at_its_sweet_spot_scores_115():
    sim, palms = _reset_palms()
    b = _bar_across_palms(sim, palms, length=0.4, sweet=0.0)
    events, total = _run(sim, [b])
    goods = [e for e in events if e[1] == 0]
    assert [g[2:] for g in goods] == [(115.0, 100.0, 15.0)]
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize("sweet, acc", [(1.0, 5.0), (-1.0, 5.0), (0.5, 10.0)])
def test_bar_sweet_offset_moves_the_full_accuracy_point(sweet, acc):
    # length 0.4: the sweet spot sits sweet * 0.2 m from the crossing;
    # 15 * (1 - 0.2 / 0.3) = 5, 15 * (1 - 0.1 / 0.3) = 10
    sim, palms = _reset_palms()
    b = _bar_across_palms(sim, palms, length=0.4, sweet=sweet)
    events, _ = _run(sim, [b])
    goods = [e for e in events if e[1] == 0]
    assert [g[2:] for g in goods] == [(100.0 + acc, 100.0, acc)]


def test_bar_the_flux_does_not_reach_is_missed():
    sim, palms = _reset_palms()
    # shifted 0.5 m along its own axis: the 0.4 m bar clears the line
    b = _bar_across_palms(sim, palms, length=0.4, sweet=0.0, dy=0.5)
    events, total = _run(sim, [b])
    assert [e[1] for e in events] == [4]
    assert total == pytest.approx(-0.5)


# -- obs -------------------------------------------------------------------------

def test_obs_hand_blocks_carry_the_palms():
    sim, palms = _reset_palms()
    obs = sim.obs()
    head = sim.head_pose()[:3]
    for base, p in ((OFF_HAND_L, palms[0]), (OFF_HAND_R, palms[1])):
        assert obs[base] == pytest.approx((p[0] - head[0]) / X_NORM, abs=1e-6)
        assert obs[base + 1] == pytest.approx((p[1] - head[1]) / Y_NORM, abs=1e-6)
        assert obs[base + 2] == pytest.approx(p[2], abs=1e-6)
        assert obs[base + 9] == 0.0 and obs[base + 10] == 0.0


def test_obs_entity_lanes_for_circles_and_bars():
    sim = _sim()
    rot, length, sweet = 0.4, 0.8, 0.5
    circ = FluxObj(hit_time=1.0, x=0.3, y=1.4, kind=KIND_CIRCLE,
                   speed=SPEED, color=1)
    bar = FluxObj(hit_time=1.2, x=-0.3, y=1.2, kind=KIND_BAR, speed=SPEED,
                  rot=rot, length=length, sweet=sweet)
    obs = sim.set_flux_chart([flux_tuple(circ), flux_tuple(bar)])
    head = sim.head_pose()[:3]
    assert obs[OFF_NOTE_MASK] == 1.0 and obs[OFF_NOTE_MASK + 1] == 1.0
    assert obs[OFF_NOTE_MASK + 2] == 0.0
    c = obs[OFF_NOTES:OFF_NOTES + NOTE_FEAT]          # nearer first
    b = obs[OFF_NOTES + NOTE_FEAT:OFF_NOTES + 2 * NOTE_FEAT]
    assert c[5] == 1.0 and c[4] == 0.0 and c[8] == 1.0 and c[9] == 0.0
    assert c[6] == 0.0 and c[7] == 0.0 and c[10] == 0.0
    ux, uy = bar_axis(rot)
    sx, sy = sweet_point(bar)
    assert b[4] == 0.0 and b[5] == 0.0 and b[8] == 0.0 and b[9] == 1.0
    assert b[6] == pytest.approx(ux, abs=1e-6)
    assert b[7] == pytest.approx(uy, abs=1e-6)
    assert b[10] == pytest.approx(length, abs=1e-6)
    assert b[11] == pytest.approx(sweet, abs=1e-6)
    assert b[12] == pytest.approx((sx - head[0]) / X_NORM, abs=1e-6)
    assert b[13] == pytest.approx((sy - head[1]) / Y_NORM, abs=1e-6)
    assert not b[14:].any()


def test_prev_block_records_a_resolved_circle():
    sim, palms = _reset_palms()
    (rx, ry, rz) = palms[1]
    c = FluxObj(hit_time=_arrive_now(sim, rz), x=rx, y=ry,
                kind=KIND_CIRCLE, speed=SPEED, color=1)
    sim.set_flux_chart([flux_tuple(c)])
    assert sim.obs()[OFF_PREV_R + 4] == 0.0
    obs, _r, _d, ev = sim.step(ZERO)
    assert [e[1] for e in ev] == [0]
    assert obs[OFF_PREV_R + 4] == 1.0
    assert obs[OFF_PREV_R + 2] == 0.0 and obs[OFF_PREV_R + 3] == 0.0


# -- mode separation / plumbing ----------------------------------------------------

def test_modes_reject_each_others_charts():
    std = RawBeatSim(max_steps=10, foot_contact=False)
    with pytest.raises(BaseException):
        std.set_flux_chart([])
    flux = _sim()
    with pytest.raises(BaseException):
        flux.set_chart([])
    with pytest.raises(BaseException):   # kind must be 4 or 5
        flux.set_flux_chart([(1.0, 0.0, 1.4, 0, 0, 0.0, 10.0, 0.0, 0.0)])


def test_vec_matches_single():
    n, steps = 3, 40
    vec = FluxVecEnv(n, seed=5, foot_contact=True)
    obs_v = vec.reset().copy()
    charts = []
    rng = np.random.default_rng(11)
    singles = []
    for i in range(n):
        ch = generate_flux_chart(vec.cfg, np.random.default_rng(100 + i))
        charts.append(ch)
        vec.sim.set_flux_chart(i, [flux_tuple(o) for o in ch])
        s = RawBeatSim(max_steps=int(20.0 * 30) + 90, foot_contact=True,
                       flux=True)
        s.set_flux_chart([flux_tuple(o) for o in ch])
        singles.append(s)
    obs_v = vec.sim.obs().reshape(n, -1)
    for i in range(n):
        assert np.array_equal(obs_v[i], singles[i].obs())
    for _ in range(steps):
        acts = rng.integers(0, 3, size=(n, N_JOINTS))
        out = np.zeros((n, obs_v.shape[1]), np.float32)
        rew, done, _ = vec.sim.step_into(acts.astype(np.int64), out)
        for i in range(n):
            o, r, d, _ev = singles[i].step([int(a) for a in acts[i]])
            assert np.array_equal(out[i], o)
            assert rew[i] == np.float32(r) and done[i] == d


def test_gym_env_rollout_and_recording(tmp_path):
    env = RustFluxEnv(foot_contact=True, seed=2)
    path = tmp_path / "flux.jsonl"
    env.set_record(str(path))
    env.reset()
    done, rng = False, np.random.default_rng(0)
    while not done:
        _o, _r, done, _t, info = env.step(rng.integers(0, 3, N_JOINTS))
    import json
    lines = path.read_text().splitlines()
    head = json.loads(lines[0])["header"]
    assert head["game"] == "flux" and len(head["chart"]) == len(env.chart)
    tick = json.loads(lines[1])
    assert len(tick["palms"]) == 2 and len(tick["palms"][0]) == 3


# -- policy ---------------------------------------------------------------------

def test_policy_game_marker_and_cross_mode_refusal():
    import torch
    from policy import NotePolicy, load_policy
    p = NotePolicy(embed=32, trunk_dim=64, n_blocks=1, n_heads=2,
                   note_geo=True, game="flux")
    ck = {"arch": dict(p.arch), "state_dict": p.state_dict()}
    assert ck["arch"]["game"] == "flux"
    q = load_policy(ck, game="flux")
    assert q.game == "flux"
    with pytest.raises(SystemExit):
        load_policy(ck)                       # standard tools refuse it
    std = NotePolicy(embed=32, trunk_dim=64, n_blocks=1, n_heads=2)
    arch = dict(std.arch)
    del arch["game"]                          # pre-Flux checkpoints
    load_policy({"arch": arch, "state_dict": std.state_dict()})
    with pytest.raises(SystemExit):
        load_policy({"arch": arch, "state_dict": std.state_dict()},
                    game="flux")
    obs = torch.from_numpy(_sim().set_flux_chart([])).float()[None]
    q.act(obs) if hasattr(q, "act") else q.forward_stacked(obs)


def test_flux_note_geo_aims_bars_at_the_sweet_spot():
    import torch
    from policy import SABER_FEAT, _note_geo
    notes = torch.zeros(1, 24, NOTE_FEAT)
    notes[0, :, 2] = 0.1
    notes[0, :, 3] = -1.0
    # token 0: circle at (0.2, 0.1); token 1: bar centred at (0.2, 0.1)
    # with its sweet spot at (-0.4, 0.3)
    notes[0, 0, [0, 1, 8]] = torch.tensor([0.2, 0.1, 1.0])
    notes[0, 1, [0, 1, 9, 12, 13]] = torch.tensor([0.2, 0.1, 1.0, -0.4, 0.3])
    sabers = torch.zeros(1, 2, SABER_FEAT)
    std = _note_geo(notes, sabers, 8.0)
    flux = _note_geo(notes, sabers, 8.0, flux=True)
    assert torch.equal(std[0, 0], flux[0, 0])             # circles: centre
    assert torch.equal(std[0, 1], std[0, 0])              # standard: centre
    # flux bar: the per-hand x/y offset lanes point at the sweet spot
    # (half-metre units: offset * 0.5)
    assert flux[0, 1, 0] == pytest.approx(-0.4 * 0.5)
    assert flux[0, 1, 1] == pytest.approx(0.3 * 0.75 * 0.5)
