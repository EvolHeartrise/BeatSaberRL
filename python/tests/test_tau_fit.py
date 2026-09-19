"""tau_fit.py recovers planted per-axis slopes from synthetic ep_stats /
tau_draws files and recommends a bounded center move."""
import numpy as np

import tau_fit


def _synth(tmp_path, n=20000, r=0.25, center=None, seed=0):
    rng = np.random.default_rng(seed)
    center = np.ones(16) if center is None else np.asarray(center)
    s = center[None, :] * (1.0 + r * rng.uniform(-1, 1, (n, 16)))
    x = (s / center - 1.0) / r
    lb = rng.integers(0, 50, n)
    map_pp = rng.uniform(50, 300, 50)
    completed = (rng.random(n) < 0.8).astype(np.float32)
    # planted: axis 10 (l_hip_pitch) +8 PP per +r, axis 3 (arm pitch) -5, others 0
    pp = map_pp[lb] * 0.6 + 8.0 * x[:, 10] - 5.0 * x[:, 3] + rng.normal(0, 10, n)
    pp = np.where(completed > 0, pp, 0.0)
    E = np.zeros((n, 16), dtype=np.float32)
    E[:, 1] = lb; E[:, 4] = 300; E[:, 6] = completed; E[:, 8] = 0.9; E[:, 9] = pp
    E[:, 15] = rng.random(n)                      # ep_return: the row-align key
    T = np.zeros((n, 22), dtype=np.float32)
    T[:, :16] = s; T[:, 16] = 300; T[:, 17] = E[:, 15]
    E.tofile(tmp_path / "ep_stats.f32"); T.tofile(tmp_path / "tau_draws.f32")
    return center


def test_recovers_planted_slopes_and_bounds_the_move(tmp_path):
    center = _synth(tmp_path)
    E, T = tau_fit.load(str(tmp_path))
    res, n, nc = tau_fit.fit(E, T, center, 0.25)
    b, se, _, _ = res["pp"]
    # slopes on the demeaned-PP objective: planted x completion rate (0.8)
    # planted x completion rate (0.8): 6.4 and -4.0, within 3 HC0 se
    assert abs(b[10] - 6.4) < 3 * se[10] and abs(b[3] + 4.0) < 3 * se[3]
    assert all(abs(b[a]) < 3 * se[a] + 0.5 for a in range(16) if a not in (3, 10))
    step, t = tau_fit.recommend(b, se, cap=0.10, tmin=3.0)
    assert step[10] == 0.10 and -0.10 <= step[3] < 0
    assert all(step[a] == 0 for a in range(16) if abs(t[a]) < 3)


def test_shifted_center_is_measured_relative_to_it(tmp_path):
    center = np.ones(16); center[10] = 0.8
    _synth(tmp_path, center=center, seed=1)
    E, T = tau_fit.load(str(tmp_path))
    res, _, _ = tau_fit.fit(E, T, center, 0.25)
    b, se, _, _ = res["pp"]
    assert abs(b[10] - 6.4) < 3 * se[10]
    # the same draws expressed against the wrong center bias the slope
    res1, _, _ = tau_fit.fit(E, T, np.ones(16), 0.25)
    assert abs(res1["pp"][0][10] - b[10]) > 0.5
