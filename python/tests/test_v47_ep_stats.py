"""env v47 (user order 2026-09-19): --ep-stats — the ScoreSaber combo walk
runs in the TRAINING envs and every episode's outcome row (EP_REC = 28
cols) carries the multiplied score, max combo, breaks, completed flag,
bad cuts and misses; the trainer turns them into per-map accuracy / PP
using the pool's per-level meta (leaderboard id, max PP, max score)."""
import json

import numpy as np
import pytest
import torch

from beat_arena_env import ranked_pool as rp
from beat_arena_env.constants import N_JOINTS

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA env")


def _payload(n, dt=0.5, first=2.0, max_score=0):
    tup = [(first + i * dt, 0.3 * (-1) ** i, 1.4, i % 2, 4.71, 8.0,
            False, 0, False, False) for i in range(n)]
    return {"tuples": tup, "walls": [], "max_score": max_score,
            "duration_s": first + n * dt, "last_hit_s": tup[-1][0]}


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    import bench_scoresaber as bs
    levels = {"aaa": _payload(300, max_score=12345),
              "bbb": _payload(40, dt=1.0, max_score=678),
              "ccc": _payload(2)}                       # sparse: dropped
    cat = [{"hash": "aaa", "difficulty": "Expert", "leaderboard_id": 101,
            "stars": 5.0, "max_pp": 210.586},
           {"hash": "bbb", "difficulty": "Hard", "leaderboard_id": 202,
            "stars": 2.0},
           {"hash": "ccc", "difficulty": "Easy", "leaderboard_id": 303,
            "stars": 1.0}]
    cpath = tmp_path / "cat.json"
    cpath.write_text(json.dumps(cat))
    monkeypatch.setattr(bs, "load_or_parse",
                        lambda entry, *a, **k: levels[entry["hash"]])
    return str(cpath)


def test_build_meta_matches_sidecar_meta(catalog):
    from beat_arena_env.scoresaber import PP_PER_STAR
    _, _, meta = rp._build_segments(catalog, lambda *a, **k: None,
                                    lead_s=0.5, mirror=True)
    side = rp._level_meta(catalog, lead_s=0.5, mirror=True)
    assert meta == side
    assert len(meta) == 4                       # 2 levels x mirror
    assert meta[0] == (101, pytest.approx(210.586), 12345.0)
    assert meta[2] == (202, pytest.approx(2.0 * PP_PER_STAR), 678.0)
    arrs = rp._meta_arrays(meta)
    assert arrs["level_lb"].dtype == np.int64
    assert arrs["level_lb"].tolist() == [101, 101, 202, 202]


def test_meta_arrays_survive_subset():
    arrs = {"note_off": np.array([0, 3, 5, 9]), "wall_off": np.array([0, 0, 1, 1]),
            "n_notes": np.array([3, 2, 4]), "n_walls": np.array([0, 1, 0]),
            "hit": np.arange(9.0), "wall_t": np.array([1.0]),
            "level_lb": np.array([7, 8, 9]), "level_max_pp": np.array([1., 2., 3.]),
            "level_max_score": np.array([10., 20., 30.])}
    from beat_arena_env.cuda_env import NOTE_OBJ_FIELDS, WALL_OBJ_FIELDS  # noqa
    out = rp._subset_ragged(arrs, np.array([0, 2]))
    assert out["level_lb"].tolist() == [7, 9]
    assert out["level_max_score"].tolist() == [10.0, 30.0]


def test_pp_from_rows():
    """The trainer's row -> (acc, pp) arithmetic on a synthetic pop."""
    from beat_arena_env.scoresaber import pp_multiplier
    max_score, max_pp = 1000.0, 100.0
    for cw, completed, want in [(950.0, 1, 100.0), (1000.0, 1, 536.7394282890631),
                                (950.0, 0, 0.0)]:
        acc = min(1.0, cw / max_score)
        pp = pp_multiplier(acc) * max_pp if completed else 0.0
        assert pp == pytest.approx(want, rel=1e-9)


def _cuda_env(n, **kw):
    """Synthetic 8-s charts (no falls: foot_contact=False), like the v46
    tests — a level ends by its notes draining unless max_steps caps it."""
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    return CudaBeatVecEnv(n, chart_len_s=8.0, seed=0, device="cuda",
                          foot_contact=False, gears=5, push_prob=0.0,
                          push_vmax=0.0, start_jitter=0.0, **kw)


@cuda
def test_cuda_ep_stats_rows_completed_and_no_freeze():
    from beat_arena_env.cuda_env import EP_REC
    e = _cuda_env(16, ep_stats=True, max_steps=600)
    assert e.combo_walk and e.ep_stats
    e.reset()
    act = torch.zeros((16, N_JOINTS), dtype=torch.int64, device="cuda")
    rows = np.zeros((0, EP_REC), dtype=np.float32)
    dones = 0
    for t in range(700):
        _, _, done, _ = e.step(act)
        dones += int(done.sum().item())
        rows = np.concatenate([rows, e.pop_episodes()])
    assert rows.shape[1] == EP_REC and rows.shape[0] >= 16
    # no bench freeze: every env ended at least once and kept playing
    # (auto-reset), so more episodes than envs over 700 ticks of ~8-s levels
    assert dones >= 16 and rows.shape[0] > 16
    # levels end by their notes draining (< the 600-tick cap): completed
    assert np.all(rows[:, 16] < 600)
    assert np.all(rows[:, 25] == 1.0)
    assert np.all(rows[:, 18] == 0.0)                 # no falls (weld mode)
    # combo-walk fields: finite, non-negative, cut counts bounded by notes
    assert np.all(rows[:, 22:25] >= 0)
    assert np.all(rows[:, 19] + rows[:, 26] + rows[:, 27] >= 0)


@cuda
def test_cuda_ep_stats_tick_cap_is_not_completed():
    from beat_arena_env.cuda_env import EP_REC
    e = _cuda_env(8, ep_stats=True, max_steps=100)
    e.reset()
    act = torch.zeros((8, N_JOINTS), dtype=torch.int64, device="cuda")
    rows = np.zeros((0, EP_REC), dtype=np.float32)
    for _ in range(105):
        e.step(act)
        rows = np.concatenate([rows, e.pop_episodes()])
    assert rows.shape[0] >= 8
    assert np.all(rows[:, 16] == 100) and np.all(rows[:, 25] == 0.0)


@cuda
def test_cuda_ep_stats_off_leaves_cols_zero():
    from beat_arena_env.cuda_env import EP_REC
    e = _cuda_env(8, tau_rand=0.25, max_steps=40)
    assert not e.ep_stats
    e.reset()
    act = torch.zeros((8, N_JOINTS), dtype=torch.int64, device="cuda")
    rows = np.zeros((0, EP_REC), dtype=np.float32)
    for _ in range(45):
        e.step(act)
        rows = np.concatenate([rows, e.pop_episodes()])
    assert rows.shape[0] >= 8
    assert np.all(rows[:, 22:] == 0.0)


@cuda
def test_cuda_tau_center_shifts_the_draws():
    from beat_arena_env.constants import TAU_FEAT
    center = [1.0] * TAU_FEAT
    center[10] = 0.8        # l_hip_pitch axis centered at 80% of TAU_MAX
    e = _cuda_env(64, tau_rand=0.25, tau_center=center, max_steps=40)
    e.reset()
    s = e.tau_scale.cpu().numpy()
    assert np.all(s[:, 10] >= 0.8 * 0.75 - 1e-6) and np.all(s[:, 10] <= 0.8 * 1.25 + 1e-6)
    assert abs(s[:, 10].mean() - 0.8) < 0.05
    others = np.delete(s, 10, axis=1)
    assert np.all(others >= 0.75 - 1e-6) and np.all(others <= 1.25 + 1e-6)
    # fixed body at the center when tau_rand == 0
    f = _cuda_env(4, tau_rand=0.0, tau_center=center, max_steps=40)
    assert np.allclose(f.tau_scale.cpu().numpy()[:, 10], 0.8)
