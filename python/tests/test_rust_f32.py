"""The f32 Rust build (rust32/, module beat_arena_rs32, 2026-09-16): same
sources as beat_arena_rs with Real = f32 — the bench's default engine.
It cannot be bitwise the f64 build; these pin that it IS the f32 build
and that it tracks f64 to f32 rounding over the first ticks."""
import numpy as np
import pytest

import beat_arena_rs
from beat_arena_env.chart import ChartConfig, generate_chart, note_tuple
from beat_arena_env.constants import N_JOINTS, OBS_DIM

rs32 = pytest.importorskip("beat_arena_rs32")


def _pair(n=4):
    rng = np.random.default_rng(3)
    charts = [[note_tuple(x) for x in generate_chart(ChartConfig(), rng)]
              for _ in range(n)]
    sims = []
    for mod in (beat_arena_rs, rs32):
        s = mod.VecBeatSim(n, max_steps=100_000, combo_walk=True,
                           return_events=False, gears=7)
        for i, c in enumerate(charts):
            s.set_chart(i, c, [])
        sims.append(s)
    return sims


def test_f32_module_is_a_distinct_build():
    assert rs32.VecBeatSim is not beat_arena_rs.VecBeatSim


def test_reset_obs_matches_to_f32_rounding():
    a, b = _pair()
    oa = np.asarray(a.obs()).reshape(-1, OBS_DIM)
    ob = np.asarray(b.obs()).reshape(-1, OBS_DIM)
    assert np.abs(oa - ob).max() < 1e-5
    # and it really is a different scalar type somewhere
    assert not np.array_equal(oa, ob)


def test_short_rollout_tracks_f64():
    a, b = _pair()
    acts = np.zeros((4, N_JOINTS), dtype=np.int64)
    acts[:, 10] = 1
    # first 4 ticks: rounding-level (measured ≤ 1.1e-5); tick 5 is where
    # the foot-contact solve first amplifies it (4e-3 on a velocity lane),
    # the same chaotic divergence the f32 CUDA kernel shows vs f64 Rust
    for t in range(4):
        ra, rb = a.step(acts), b.step(acts)
        oa = np.asarray(ra[0]).reshape(-1, OBS_DIM)
        ob = np.asarray(rb[0]).reshape(-1, OBS_DIM)
        assert np.abs(oa - ob).max() < 1e-4, t
        assert np.array_equal(np.asarray(ra[2]), np.asarray(rb[2]))
    for _ in range(300):
        rb = b.step(acts)
    assert np.isfinite(np.asarray(rb[0])).all()
