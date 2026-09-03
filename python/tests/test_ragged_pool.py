"""Ragged (CSR) chart pool == padded (P, C) pool, object for object
(2026-09-03, audit 2.7 / Pass B §4).

The kernel reads chart cc's object i at note_off[cc] + i where it read
cc * C + i; nothing else changed. This module pins the layout side of
that claim: (1) every per-object field of the ragged pool equals the
padded builder's row (p, j) for every segment p and object j < n_notes
(notes AND walls), per-segment fields equal, offsets consistent;
(2) ragged_to_padded round-trips to the padded builders' output
exactly (pads included); (3) the seeded subsample path gathers each
chosen segment's object range; (4) on CUDA, an env fed the padded dict
(converted at construction) and one fed the ragged dict step bitwise-
identically — the pad_to_ragged path the production pool takes.
"""

import numpy as np
import pytest
import torch

from beat_arena_env.chart import Note
from beat_arena_env.constants import (KIND_BOMB, KIND_CHAIN_HEAD,
                                      KIND_CHAIN_LINK, KIND_NOTE)
from beat_arena_env.cuda_env import (NOTE_OBJ_FIELDS, WALL_OBJ_FIELDS,
                                     _charts_to_padded, _charts_to_tensors,
                                     _walls_to_padded, _walls_to_tensors,
                                     pad_to_ragged, ragged_to_padded)
from beat_arena_env.ranked_pool import _subset_ragged

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="needs CUDA")


def _synthetic(n_charts=41, seed=0, max_notes=13, max_walls=4):
    """Ragged charts + walls covering every kind/flag combination,
    empty segments (notes and/or walls) included."""
    rng = np.random.default_rng(seed)
    kinds = [KIND_NOTE, KIND_BOMB, KIND_CHAIN_HEAD, KIND_CHAIN_LINK]
    charts, walls = [], []
    for p in range(n_charts):
        notes = []
        for j in range(int(rng.integers(0, max_notes))):
            kind = kinds[int(rng.integers(len(kinds)))]
            link = kind == KIND_CHAIN_LINK
            notes.append(Note(
                hit_time=float(rng.uniform(0.5, 25.0)),
                col=int(rng.integers(4)), row=int(rng.integers(3)),
                color=int(rng.integers(2)),
                angle=float(rng.uniform(-np.pi, np.pi)),
                speed=float(rng.uniform(8.0, 25.0)),
                any_dir=bool(rng.integers(2)),
                kind=kind,
                x=float(rng.uniform(-1, 1)) if link else None,
                y=float(rng.uniform(0, 2)) if link else None,
                free_pre=bool(rng.integers(2)),
                free_post=bool(rng.integers(2))))
        charts.append(sorted(notes, key=lambda n: n.hit_time))
        ws = [(float(rng.uniform(0.5, 20.0)), float(rng.uniform(-1, 1)),
               float(rng.uniform(0.5, 2.0)), float(rng.uniform(0.2, 1.0)),
               float(rng.uniform(0.2, 1.5)), float(rng.uniform(0.5, 5.0)),
               float(rng.uniform(8.0, 25.0)))
              for _ in range(int(rng.integers(0, max_walls)))]
        walls.append(sorted(ws))
    charts[3], walls[5], charts[7], walls[7] = [], [], [], []
    return charts, walls


def _build(charts, walls, device="cpu"):
    padded = _charts_to_padded(charts, device, torch.float32)
    padded.update(_walls_to_padded(walls, device, torch.float32))
    ragged = _charts_to_tensors(charts, device, torch.float32)
    ragged.update(_walls_to_tensors(walls, device, torch.float32))
    return padded, ragged


def _check_block(padded, ragged, fields, off, counts):
    P = counts.shape[0]
    assert off.shape == (P + 1,) and off.dtype == torch.int64
    assert off[0] == 0
    assert torch.equal(off[1:] - off[:-1], counts)
    total = int(off[-1])
    for k in fields:
        assert ragged[k].shape[0] == total, k
        assert ragged[k].dtype == padded[k].dtype, k
        assert ragged[k].shape[1:] == padded[k].shape[2:], k
        for p in range(P):
            n = int(counts[p])
            assert torch.equal(ragged[k][off[p]:off[p] + n],
                               padded[k][p, :n]), (k, p)


def test_ragged_equals_padded_per_object():
    charts, walls = _synthetic()
    padded, ragged = _build(charts, walls)
    # per-object fields, per segment, per object
    _check_block(padded, ragged, NOTE_OBJ_FIELDS, ragged["note_off"],
                 padded["n_notes"])
    _check_block(padded, ragged, WALL_OBJ_FIELDS, ragged["wall_off"],
                 padded["n_walls"])
    # ragged carries no pads: every kept object is valid
    assert bool(ragged["valid"].all()) and bool(ragged["wall_valid"].all())
    assert int(ragged["valid"].shape[0]) == sum(len(c) for c in charts)
    assert int(ragged["wall_valid"].shape[0]) == sum(len(w) for w in walls)
    # per-segment fields untouched
    for k in ("n_notes", "n_scoreable", "last_hit", "tail", "min_speed",
              "n_walls"):
        assert torch.equal(ragged[k], padded[k]), k
    # key set: padded keys + the two offset arrays, nothing else
    assert set(ragged) == set(padded) | {"note_off", "wall_off"}


def test_ragged_max_count_is_padded_width():
    """C / W for the per-env status stride = max objects = the padded
    width the (P, C) layout had (what CudaBeatVecEnv derives)."""
    charts, walls = _synthetic(seed=3)
    padded, ragged = _build(charts, walls)
    assert int(ragged["n_notes"].max()) == padded["hit"].shape[1]
    assert int(ragged["n_walls"].max()) == padded["wall_t"].shape[1]


def test_round_trip_padded_ragged_padded():
    charts, walls = _synthetic(seed=11)
    padded, ragged = _build(charts, walls)
    back = ragged_to_padded({k: v.clone() for k, v in ragged.items()})
    assert set(back) == set(padded)
    for k in padded:
        assert back[k].dtype == padded[k].dtype, k
        assert back[k].shape == padded[k].shape, k
        assert torch.equal(back[k], padded[k]), k
    # and the conversion is idempotent on an already-ragged pool
    again = pad_to_ragged({k: v.clone() for k, v in ragged.items()})
    for k in ragged:
        assert torch.equal(again[k], ragged[k]), k


def test_pad_to_ragged_converts_a_padded_dict():
    """A padded dict handed to pad_to_ragged (the pool= / old-cache
    path) equals the natively built ragged pool."""
    charts, walls = _synthetic(seed=5)
    padded, ragged = _build(charts, walls)
    conv = pad_to_ragged({k: v.clone() for k, v in padded.items()})
    assert set(conv) == set(ragged)
    for k in ragged:
        assert torch.equal(conv[k], ragged[k]), k


@pytest.mark.parametrize("seed", [0, 1])
def test_subset_ragged_gathers_segment_ranges(seed):
    charts, walls = _synthetic(seed=seed)
    _, ragged = _build(charts, walls)
    arrs = {k: v.numpy() for k, v in ragged.items()}
    P = len(charts)
    idx = np.sort(np.random.default_rng(seed).choice(P, 17, replace=False))
    sub = _subset_ragged(arrs, idx)
    # brute force: rebuild from the chosen segments
    ref = _charts_to_tensors([charts[i] for i in idx], "cpu", torch.float32)
    ref.update(_walls_to_tensors([walls[i] for i in idx], "cpu",
                                 torch.float32))
    assert set(sub) == set(ref)
    for k in ref:
        assert sub[k].dtype == ref[k].numpy().dtype, k
        assert np.array_equal(sub[k], ref[k].numpy()), k
    assert _subset_ragged(arrs, None) is arrs


def _ticks(env, n=120, seed=0):
    torch.manual_seed(seed)
    env.reset()
    rng = np.random.default_rng(seed)
    outs = []
    judged = 0          # notes judged over the run (auto-resets zero status)
    for _ in range(n):
        a = torch.as_tensor(rng.integers(0, 5, size=(env.num_envs, 29)),
                            device="cuda")
        obs, rew, done, _ = env.step(a)
        outs.append((obs.clone(), rew.clone(), done.clone()))
        judged += int(env.status.count_nonzero())
    return outs, [getattr(env, k).clone() for k in
                  ("status", "first_pending", "wall_passed", "first_wall",
                   "score", "good", "bad", "miss", "bombs", "walls_in")], \
        judged


@cuda
def test_env_padded_pool_equals_ragged_pool_bitwise():
    """pool= accepts either layout; both step to the same bits (the
    kernel only ever sees ragged — the padded dict is converted at
    construction)."""
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    charts, walls = _synthetic(n_charts=24, seed=2, max_notes=40,
                               max_walls=6)
    padded, ragged = _build(charts, walls)
    kw = dict(max_steps=25 * 30 + 90, foot_contact=True, gears=5, seed=0,
              device="cuda", wall_despawn=True, dtype=torch.float32)
    e_pad = CudaBeatVecEnv(64, pool=padded, **kw)
    e_rag = CudaBeatVecEnv(64, pool=ragged, **kw)
    assert (e_pad.C, e_pad.W) == (e_rag.C, e_rag.W)
    assert e_pad.C == padded["hit"].shape[1]
    assert "note_off" in e_pad.pool and e_pad.pool["hit"].dim() == 1
    a, b = _ticks(e_pad), _ticks(e_rag)
    for t, ((oa, ra, da), (ob, rb, db)) in enumerate(zip(a[0], b[0])):
        assert torch.equal(oa, ob), f"obs diverged at tick {t}"
        assert torch.equal(ra, rb), f"reward diverged at tick {t}"
        assert torch.equal(da, db), f"done diverged at tick {t}"
    for sa, sb in zip(a[1], b[1]):
        assert torch.equal(sa, sb)
    # the run actually exercised the pool: notes were judged along the way
    assert a[2] == b[2] and a[2] > 0
