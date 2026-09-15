"""Whole-level pool mode (2026-09-15, user order): seg_len <= 0 makes one
window per level covering [lead_s, inf), skips the MAX_OBJECTS drop, keys
its own cache, and pool_max_steps derives the episode limit from the
longest level."""
import json
import os

import numpy as np
import pytest
import torch

from beat_arena_env import ranked_pool as rp
from beat_arena_env.constants import KIND_BOMB


def _payload(n, dt=0.5, first=2.0, walls=()):
    # note_tuple order: (hit, x, y, color, angle, speed, any_dir, kind,
    # free_pre, free_post)
    tup = [(first + i * dt, 0.3 * (-1) ** i, 1.4, i % 2, 4.71, 8.0,
            False, 0, False, False) for i in range(n)]
    return {"tuples": tup, "walls": list(walls), "max_score": 0,
            "duration_s": first + n * dt, "last_hit_s": tup[-1][0]}


@pytest.fixture
def fake_catalog(tmp_path, monkeypatch):
    import bench_scoresaber as bs
    levels = {"aaa": _payload(3000, dt=0.1),          # 3,000 objects > MAX_OBJECTS
              "bbb": _payload(40, dt=1.0,
                              walls=[(0.5, 0.0, 1.0, 0.5, 1.0, 4.0, 8.0)]),
              "ccc": _payload(2, dt=0.5)}             # sparse: dropped
    cat = [{"hash": h, "difficulty": "Expert"} for h in levels]
    cpath = tmp_path / "cat.json"
    cpath.write_text(json.dumps(cat))
    monkeypatch.setattr(bs, "load_or_parse",
                        lambda entry, *a, **k: levels[entry["hash"]])
    return str(cpath)


def test_whole_level_one_window_per_level_no_cap(fake_catalog):
    segs, walls = rp._build_segments(0.0, fake_catalog, lambda *a, **k: None,
                                     lead_s=0.5)
    assert len(segs) == 2                       # 'ccc' dropped as sparse
    assert len(segs[0]) == 3000                 # no MAX_OBJECTS drop
    assert len(segs[1]) == 40
    # rebased to base 0: first note at the parser anchor (2.0 s)
    assert abs(segs[0][0].hit_time - 2.0) < 1e-9
    assert abs(segs[0][-1].hit_time - (2.0 + 2999 * 0.1)) < 1e-6
    # wall clipped to [lead_s, inf) and kept whole past that
    assert len(walls[1]) == 1
    t0, *_ = walls[1][0]
    assert abs(t0 - 0.5) < 1e-9


def test_segment_mode_unchanged(fake_catalog):
    segs, _ = rp._build_segments(25.0, fake_catalog, lambda *a, **k: None,
                                 lead_s=0.5)
    # 3,000-object level: 25-s windows hold 250 notes each -> kept
    assert all(len(s) <= rp.MAX_OBJECTS for s in segs)
    assert len(segs) > 2


def test_cache_key_distinct(fake_catalog):
    a = rp._cache_path(0.0, "ver", fake_catalog, 0.5)
    b = rp._cache_path(25.0, "ver", fake_catalog, 0.5)
    c = rp._cache_path(-1.0, "ver", fake_catalog, 0.5)
    assert a != b and a == c


def test_pool_max_steps():
    pool = {"n_notes": torch.tensor([3, 5]),
            "last_hit": torch.tensor([10.0, 200.0]),
            "tail": torch.tensor([0.5, 0.6875])}
    assert rp.pool_max_steps(pool) == int(200.6875 * 30) + 90
    empty = {"n_notes": torch.zeros(0), "last_hit": torch.zeros(0),
             "tail": torch.zeros(0)}
    assert rp.pool_max_steps(empty) == 90
