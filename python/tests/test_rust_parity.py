"""Rust sim parity: same chart + same actions => same rewards/obs/outcomes
as the Python sim, across configs (incl. shaping, for the ablation path)."""

import numpy as np
import pytest

from beat_arena_env import BeatSim, ChartConfig
from beat_arena_env.env import build_obs

beat_arena_rs = pytest.importorskip("beat_arena_rs")


def _run_pair(cfg, seed, n_steps=700):
    py = BeatSim(cfg, seed=seed)
    rs = beat_arena_rs.RawBeatSim(note_speed=cfg.note_speed,
                                  shaping_coef=cfg.shaping_coef,
                                  max_steps=py.max_steps)
    rs_obs = rs.set_chart([(n.hit_time, n.col, n.row, n.color, n.dir)
                           for n in py.chart])
    np.testing.assert_allclose(rs_obs, build_obs(py), atol=1e-6)

    rng = np.random.default_rng(seed + 999)
    for _ in range(n_steps):
        a = rng.integers(0, 9, size=2)
        py_r, py_done, _ = py.step(int(a[0]), int(a[1]))
        rs_o, rs_r, rs_done = rs.step(int(a[0]), int(a[1]))
        assert rs_r == pytest.approx(py_r, abs=1e-9)
        assert rs_done == py_done
        np.testing.assert_allclose(rs_o, build_obs(py), atol=1e-6)
        if py_done:
            break
    assert (rs.good_cuts, rs.bad_cuts, rs.misses) == \
           (py.good_cuts, py.bad_cuts, py.misses)


@pytest.mark.parametrize("cfg,seed", [
    (ChartConfig(), 0),
    (ChartConfig(colors="both", dirs=tuple(range(8))), 1),
    (ChartConfig(colors="both", dirs=tuple(range(8)), note_rate=2.5,
                 min_hand_gap_s=0.4), 2),
    (ChartConfig(shaping_coef=1.0), 3),
    (ChartConfig(colors="both", dirs=(2, 6), note_speed=8.0), 4),
])
def test_random_action_parity(cfg, seed):
    _run_pair(cfg, seed)


def test_scripted_trajectory_parity():
    """Drive BOTH sims with the scripted bot's actions (computed on the
    Python sim) — exercises the cut-heavy path, not just random flailing."""
    from beat_arena_env import scripted_action
    cfg = ChartConfig(colors="both", dirs=tuple(range(8)))
    py = BeatSim(cfg, seed=11)
    rs = beat_arena_rs.RawBeatSim(note_speed=cfg.note_speed,
                                  shaping_coef=cfg.shaping_coef,
                                  max_steps=py.max_steps)
    rs.set_chart([(n.hit_time, n.col, n.row, n.color, n.dir) for n in py.chart])
    done = False
    while not done:
        al, ar = scripted_action(py)
        py_r, done, _ = py.step(al, ar)
        _, rs_r, rs_done = rs.step(al, ar)
        assert rs_r == pytest.approx(py_r, abs=1e-9)
        assert rs_done == done
    assert (rs.good_cuts, rs.bad_cuts, rs.misses) == \
           (py.good_cuts, py.bad_cuts, py.misses)
    assert py.good_cuts > 10  # the pair actually cut things
