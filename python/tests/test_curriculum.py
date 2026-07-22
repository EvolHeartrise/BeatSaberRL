"""Curriculum ladder tests — offline (network fetcher monkeypatched; charts
come from the checked-in maps/*.zip)."""

import json
import os

import numpy as np
import pytest

from beat_arena_env import curriculum as curriculum_mod
from beat_arena_env.chart import Note
from beat_arena_env.curriculum import Curriculum
from beat_arena_env.vec_env import BeatVecEnv

MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "maps")
# the fixture zips live in the (gitignored) local map cache — absent on a
# fresh clone/machine, in which case the ladder tests can't run offline
_HAVE_ZIPS = all(os.path.exists(os.path.join(MAPS_DIR, z))
                 for z in ("spectre.zip", "violin.zip"))
needs_zips = pytest.mark.skipif(
    not _HAVE_ZIPS, reason="local fixture zips missing (maps/ is a cache)")


def _stage(zip_name, song, stars, fc):
    return {"hash": zip_name, "song": song, "mapper": "m",
            "difficulty": "Easy", "stars": stars, "leaderboard_id": 0,
            "zip": zip_name, "n_notes": 0, "duration_s": 0.0, "nps": 0.0,
            "fc": fc, "target_star": stars}


def _write_state(path, stages):
    with open(path, "w") as f:
        json.dump({"version": 1, "start_star": 1.0, "star_step": 0.2,
                   "speed_scale": 0.3, "stages": stages}, f)


@pytest.fixture
def ladder(tmp_path):
    """Two-stage ladder: spectre (FC'd), violin (current)."""
    state = str(tmp_path / "curriculum.json")
    _write_state(state, [_stage("spectre.zip", "Spectre", 1.0, True),
                         _stage("violin.zip", "Violin", 1.2, False)])
    return Curriculum(MAPS_DIR, state)


@needs_zips
def test_resume_loads_charts(ladder):
    assert len(ladder.stages) == 2 and ladder.newest == 1
    assert all(len(c) > 100 for c in ladder.charts)
    assert all(isinstance(n, Note) for n in ladder.charts[0][:3])


@needs_zips
def test_sample_weighting(ladder):
    rng = np.random.default_rng(0)
    picks = [ladder.sample(0, rng) is ladder.charts[1] for _ in range(1000)]
    newest_frac = np.mean(picks)
    assert 0.42 < newest_frac < 0.58  # new_map_weight = 0.5


@needs_zips
def test_single_stage_always_sampled(tmp_path):
    state = str(tmp_path / "c.json")
    _write_state(state, [_stage("spectre.zip", "Spectre", 1.0, False)])
    c = Curriculum(MAPS_DIR, state)
    rng = np.random.default_rng(0)
    assert all(c.sample(0, rng) is c.charts[0] for _ in range(20))


@needs_zips
def test_stage_bookkeeping_and_stats(ladder):
    rng = np.random.default_rng(1)
    ladder.sample(3, rng)                      # episode 1 starts on env 3
    first = ladder.env_stage[3]
    ladder.sample(3, rng)                      # episode 1 ended -> episode 2
    assert ladder.completed_stage[3] == first
    ladder.note_result({"env": 3, "good_cuts": 5, "bad_cuts": 1,
                        "misses": 2, "n_notes": 8})
    assert list(ladder.stats[first]) == [(5, 1, 2, 8)]
    assert "current" in ladder.summary()


@needs_zips
def test_promote_appends_next_rung(ladder, monkeypatch):
    def fake_acquire(target_star, band, dest, rng, **kw):
        assert abs(target_star - 1.4) < 1e-9   # 1.2 + star_step
        return {"hash": "f", "song": "Fake", "mapper": "m",
                "difficulty": "Normal", "stars": 1.38, "leaderboard_id": 1,
                "zip": "spectre.zip", "n_notes": 300, "duration_s": 200.0,
                "nps": 1.5}
    monkeypatch.setattr(curriculum_mod.scoresaber, "acquire_random_map",
                        fake_acquire)
    ladder.promote()
    assert len(ladder.stages) == 3
    assert ladder.stages[1]["fc"] and not ladder.stages[2]["fc"]
    # state round-trips
    c2 = Curriculum(MAPS_DIR, ladder.state_path)
    assert [s["song"] for s in c2.stages] == ["Spectre", "Violin", "Fake"]


def test_vec_env_chart_sampler():
    # row 0 cells sit far below the resting forward blades (y 1.47 vs
    # 0.85), so coasting really misses them
    chart = [Note(hit_time=1.0, col=0, row=0, color=1, angle=np.pi, speed=4.0),
             Note(hit_time=2.0, col=3, row=0, color=0, angle=0.0, speed=4.0)]
    venv = BeatVecEnv(2, chart_sampler=lambda i, rng: chart,
                      max_steps=200, seed=0)
    venv.reset()
    acts = np.zeros((2, 12), dtype=np.int64)   # coast: miss everything
    done_info = None
    for _ in range(200):
        _, _, done, infos = venv.step(acts)
        if infos:
            done_info = infos[0]
            break
    assert done_info is not None
    assert done_info["n_notes"] == 2 and done_info["misses"] == 2


def test_greedy_fc_eval_shape():
    import torch
    torch.manual_seed(0)
    from policy import NotePolicy
    from train_batch import greedy_fc_eval
    chart = [Note(hit_time=1.0, col=1, row=1, color=1, angle=np.pi, speed=4.0)]
    fc, good, breaks, n = greedy_fc_eval(NotePolicy(), chart)
    assert n == 1 and isinstance(fc, bool)
    assert fc == (good == 1 and breaks == 0)
