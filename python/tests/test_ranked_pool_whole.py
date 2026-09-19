"""Whole-level ranked pool (the only mode since 2026-09-16): one entry per
level covering [lead_s, inf) from song time 0, no object cap, a frozen
cache key, and pool_max_steps deriving the episode limit from the longest
level. Byte-identical to the 2026-09-15 `--seg-len 0` probe mode."""
import json

import pytest
import torch

from beat_arena_env import ranked_pool as rp


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
    levels = {"aaa": _payload(3000, dt=0.1),          # 3,000 objects: no cap
              "bbb": _payload(40, dt=1.0,
                              walls=[(0.5, 0.0, 1.0, 0.5, 1.0, 4.0, 8.0),
                                     (-1.0, 0.0, 1.0, 0.5, 1.0, 8.0, 8.0)]),
              "ccc": _payload(2, dt=0.5)}             # sparse: dropped
    cat = [{"hash": h, "difficulty": "Expert"} for h in levels]
    cpath = tmp_path / "cat.json"
    cpath.write_text(json.dumps(cat))
    monkeypatch.setattr(bs, "load_or_parse",
                        lambda entry, *a, **k: levels[entry["hash"]])
    return str(cpath)


def test_one_entry_per_level_no_cap(fake_catalog):
    segs, walls, meta = rp._build_segments(fake_catalog, lambda *a, **k: None,
                                           lead_s=0.5)
    assert len(meta) == 2
    assert len(segs) == 2                       # 'ccc' dropped as sparse
    assert len(segs[0]) == 3000                 # no object cap
    assert len(segs[1]) == 40
    # song time 0 is the episode start: the parser's 2.0-s first-note
    # anchor is the only front padding
    assert abs(segs[0][0].hit_time - 2.0) < 1e-9
    assert abs(segs[0][-1].hit_time - (2.0 + 2999 * 0.1)) < 1e-6
    # walls clipped to [lead_s, inf): the intro wall starting at -1.0 s
    # (length 8 m at 8 m/s = 1 s span, ends at 0.0) is dropped, the one
    # at 0.5 s is kept whole
    assert len(walls[1]) == 1
    t0, *_ = walls[1][0]
    assert abs(t0 - 0.5) < 1e-9


def test_mirror_doubles_the_pool(fake_catalog):
    segs, walls, meta = rp._build_segments(fake_catalog, lambda *a, **k: None,
                                           lead_s=0.5, mirror=True)
    assert len(meta) == 4 and meta[0] == meta[1]
    assert len(segs) == 4 and len(walls) == 4
    assert len(segs[0]) == len(segs[1]) == 3000
    # mirrored copy: x negated on every note
    assert all(abs(a.x + b.x) < 1e-9 for a, b in zip(segs[0], segs[1]))


def test_cache_key_is_the_frozen_whole_level_key(fake_catalog):
    """The key string is frozen at the v8 whole-level form so the pod
    caches from the v46_whole probe are reused; changing lead_s or mirror
    must still change it."""
    a = rp._cache_path("ver", fake_catalog, 0.5)
    b = rp._cache_path("ver", fake_catalog, 1.5)
    c = rp._cache_path("ver", fake_catalog, 0.5, mirror=True)
    assert a != b and a != c and b != c
    import hashlib
    cat = open(fake_catalog, "rb").read()
    expect = hashlib.sha1(
        ("whole_level_v8_ver_v7_lead0.5").encode() + cat).hexdigest()[:16]
    assert a.endswith(f"ranked_{expect}.npz")


def test_pool_max_steps():
    pool = {"n_notes": torch.tensor([3, 5]),
            "last_hit": torch.tensor([10.0, 200.0]),
            "tail": torch.tensor([0.5, 0.6875])}
    assert rp.pool_max_steps(pool) == int(200.6875 * 30) + 90
    empty = {"n_notes": torch.zeros(0), "last_hit": torch.zeros(0),
             "tail": torch.zeros(0)}
    assert rp.pool_max_steps(empty) == 90


def test_load_ranked_pool_signature_has_no_seg_len():
    import inspect
    params = list(inspect.signature(rp.load_ranked_pool).parameters)
    assert "seg_len" not in params
    assert params[:2] == ["sample", "seed"]
    assert not hasattr(rp, "window_starts") and not hasattr(rp, "MAX_OBJECTS")
