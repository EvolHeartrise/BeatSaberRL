"""exp/tick-rate (2026-09-28): the Rust sim's runtime control rate
(`tick_hz`, default 30) and VecBeatSim's action repeat, on BOTH builds
(beat_arena_rs f64 and beat_arena_rs32 f32 — the bench's engine).

The Rust unit tests (rust/src/lib.rs env_tests) pin the derivation, the
bitwise equal-strength property and the per-tick rescales in f64; these
pin the FFI surface and re-check equal strength / repeat on the f32 build.
"""
import numpy as np
import pytest

import beat_arena_rs
from beat_arena_env.chart import ChartConfig, generate_chart, note_tuple
from beat_arena_env.constants import N_JOINTS, OBS_DIM, OFF_GROUND_IMP

MODS = [beat_arena_rs]
try:
    import beat_arena_rs32
    MODS.append(beat_arena_rs32)
except ImportError:
    pass

FAR = [(100.0, 0.0, 1.0, 0, 0.0, 6.0, False, 0, False, False)]


@pytest.mark.parametrize("mod", MODS)
def test_rate_derivation_and_validation(mod):
    for hz, n_sub in ((30, 8), (60, 4), (120, 2), (240, 1)):
        s = mod.RawBeatSim(max_steps=690, tick_hz=hz)
        assert (s.tick_hz, s.n_sub) == (hz, n_sub)
        assert abs(s.dt - 1.0 / hz) < 1e-7
        assert s.max_steps == 690 * hz // 30   # budget keeps its real time
    assert mod.RawBeatSim().tick_hz == 30
    assert mod.VecBeatSim(2).tick_hz == 30
    assert mod.VecBeatSim(2, tick_hz=120).tick_hz == 120
    for bad in (0, 45, 90, 480):
        with pytest.raises(ValueError):
            mod.RawBeatSim(tick_hz=bad)
        with pytest.raises(ValueError):
            mod.VecBeatSim(2, tick_hz=bad)


@pytest.mark.parametrize("mod", MODS)
def test_equal_strength_same_substeps_same_state(mod):
    """The same held action from the same state is the same substeps
    with the same torques at every rate: qpos/qvel agree bitwise after
    0.5 s (15 / 30 / 60 ticks). Full-torque patterns included; kept to
    ones that do not terminate the episode within 0.5 s (a fall/tilt is
    checked per tick, so the frozen terminal state would differ — the
    Rust Body-level test covers full-body patterns through falls)."""
    arms = [0, 0, 0] + [1, 2] * 7 + [0] * 12
    pats = {"coast": np.zeros(N_JOINTS, np.int64),
            "arms full": np.array(arms, np.int64),
            "arms full reversed": np.array([0, 0, 0] + [2, 1] * 7 + [0] * 12,
                                           np.int64),
            "arms mixed gears": np.array([0, 0, 0] + [(1, 2, 3, 4, 0)[j % 5]
                                                     for j in range(14)]
                                         + [0] * 12, np.int64)}
    for name, act in pats.items():
        out = []
        for hz in (30, 60, 120):
            s = mod.RawBeatSim(max_steps=100_000, gears=5, tick_hz=hz)
            s.set_chart(FAR)
            for _ in range(hz // 2):
                _, _, d, _ = s.step(act.tolist())
                assert not d, (name, hz)
            out.append((np.array(s.qpos()), np.array(s.twist()),
                        np.array(s.omega())))
        for k in (1, 2):
            for a, b in zip(out[0], out[k]):
                assert np.array_equal(a, b), name


@pytest.mark.parametrize("mod", MODS)
def test_ground_imp_lanes_rate_invariant(mod):
    lanes = []
    for hz in (30, 60, 120):
        s = mod.RawBeatSim(max_steps=100_000, gears=5, tick_hz=hz)
        s.set_chart(FAR)
        for _ in range(3 * hz // 30):   # 0.1 s
            o, *_ = s.step([0] * N_JOINTS)
        lanes.append(np.asarray(o)[OFF_GROUND_IMP:OFF_GROUND_IMP + 4])
    assert abs(lanes[0].sum() - 1.0) < 0.1
    for k in (1, 2):
        assert np.abs(lanes[k] - lanes[0]).max() < 0.02, lanes


@pytest.mark.parametrize("mod", MODS)
def test_action_repeat_is_held_steps(mod):
    """step_into(repeat=2) == two step_into calls with the same actions
    (obs after the 2nd tick, rewards summed, dones OR'ed, combo walk and
    counters identical), including envs that finish mid-repeat."""
    rng = np.random.default_rng(5)
    n = 4
    charts = [[note_tuple(x) for x in generate_chart(ChartConfig(), rng)][:12]
              for _ in range(n)]
    sims = []
    for _ in range(2):
        s = mod.VecBeatSim(n, max_steps=100_000, combo_walk=True,
                           return_events=True, gears=5, tick_hz=60)
        for i, c in enumerate(charts):
            s.set_chart(i, c, [])
        sims.append(s)
    oa = np.zeros((n, OBS_DIM), np.float32)
    ob = np.zeros((n, OBS_DIM), np.float32)
    done = np.zeros(n, bool)
    for t in range(2000):
        act = rng.integers(0, 5, size=(n, N_JOINTS)).astype(np.int64)
        ra, da, ea = sims[0].step_into(act, oa, repeat=2)
        r1, d1, e1 = sims[1].step_into(act, ob)
        r2, d2, e2 = sims[1].step_into(act, ob)
        ra, da, r1, d1, r2, d2 = map(np.asarray, (ra, da, r1, d1, r2, d2))
        assert np.array_equal(oa, ob), t
        assert np.array_equal(da, d1 | d2), t
        # a finished env's 2nd tick is a latched no-op (reward 0)
        assert np.allclose(ra, np.where(d1, r1, r1 + r2), atol=1e-6), t
        for i in range(n):   # per env: tick 1's events, then tick 2's
            assert ([e for e in ea if e[0] == i]
                    == [e for e in e1 + e2 if e[0] == i]), (t, i)
        done |= da
        if done.all():
            break
    assert done.all()
    for i in range(n):
        assert sims[0].combo(i) == sims[1].combo(i)
        assert sims[0].counters(i) == sims[1].counters(i)
    # repeat=1 is the default single tick
    s = mod.VecBeatSim(1, max_steps=100_000, gears=5)
    s.set_chart(0, FAR, [])
    s.step(np.zeros((1, N_JOINTS), np.int64), repeat=1)
    with pytest.raises(BaseException):
        s.step(np.zeros((1, N_JOINTS), np.int64), repeat=0)
