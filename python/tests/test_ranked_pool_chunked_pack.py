"""Chunked chart packing == one whole-pool call, bitwise.

Pins the contract ranked_pool.load_ranked_pool relies on since
2026-08-08 (the whole-pool call OOM-killed a 14 GB laptop at 46k
segments). Padded builder (_charts_to_padded): packing chart chunks
with pad_to=<global max C> and concatenating per field must be
byte-identical to a single call over the full list — every pad value
is a per-field constant and every derived row stat (min_speed, tail,
n_notes, last_hit) is row-local. Ragged pool (2026-09-03, audit 2.7):
chunk builds stitched by cat_pools (offset rebase) must equal the
whole-list build.
"""

import numpy as np
import pytest
import torch

from beat_arena_env.chart import Note
from beat_arena_env.constants import (KIND_BOMB, KIND_CHAIN_HEAD,
                                      KIND_CHAIN_LINK, KIND_NOTE)
from beat_arena_env.cuda_env import (_charts_to_padded, _charts_to_tensors,
                                     cat_pools)


def _synthetic_charts(n_charts=37, seed=0):
    """Ragged chart list covering every kind/flag combination, including
    an empty chart (row of pure padding) and explicit-x/y chain links."""
    rng = np.random.default_rng(seed)
    kinds = [KIND_NOTE, KIND_BOMB, KIND_CHAIN_HEAD, KIND_CHAIN_LINK]
    charts = []
    for p in range(n_charts):
        notes = []
        for j in range(int(rng.integers(0, 13))):
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
    charts[3] = []                       # empty chart: all-padding row
    return charts


def _assert_same(stitched, whole):
    assert stitched.keys() == whole.keys()
    for k in whole:
        assert stitched[k].dtype == whole[k].dtype, k
        assert stitched[k].shape == whole[k].shape, k
        assert torch.equal(stitched[k], whole[k]), k


@pytest.mark.parametrize("chunk", [1, 4, 16])
def test_chunked_pack_bitwise_padded(chunk):
    charts = _synthetic_charts()
    whole = _charts_to_padded(charts, "cpu", torch.float32)
    cmax = max((len(c) for c in charts), default=1)
    parts = [_charts_to_padded(charts[lo:lo + chunk], "cpu",
                               torch.float32, pad_to=cmax)
             for lo in range(0, len(charts), chunk)]
    _assert_same({k: torch.cat([p[k] for p in parts]) for k in parts[0]},
                 whole)


@pytest.mark.parametrize("chunk", [1, 4, 16])
def test_chunked_pack_bitwise_ragged(chunk):
    """The production build path: ragged chunks (no pad_to — each chunk
    pads to ITS max and flattens) stitched by cat_pools."""
    charts = _synthetic_charts()
    whole = _charts_to_tensors(charts, "cpu", torch.float32)
    parts = [_charts_to_tensors(charts[lo:lo + chunk], "cpu", torch.float32)
             for lo in range(0, len(charts), chunk)]
    _assert_same(cat_pools(parts), whole)


def test_pad_to_default_unchanged():
    """pad_to=None must reproduce the historical local-max C exactly."""
    charts = _synthetic_charts(seed=7)
    a = _charts_to_padded(charts, "cpu", torch.float32)
    b = _charts_to_padded(charts, "cpu", torch.float32,
                          pad_to=max(len(c) for c in charts))
    for k in a:
        assert torch.equal(a[k], b[k]), k
