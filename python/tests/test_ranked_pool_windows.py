"""Ranked-pool windowing: stride = seg_len - lead_s (2026-09-06).

Pins that the kept ranges tile song time from lead_s on with no holes
and no double coverage — every note past the lead gap lands in exactly
one window — and that the last window still holds the chart's final
object. The pre-2026-09-06 stride (= seg_len) left the lead gap of
every k > 0 window as a hole: 5.77% of the catalog's scoreable notes
were in no training window.
"""

import numpy as np
import pytest

from beat_arena_env.ranked_pool import LEAD_S, window_starts


def _kept(bases, seg_len, lead_s):
    return [(b + lead_s, b + seg_len) for b in bases]


@pytest.mark.parametrize("seg_len,lead_s", [(25.0, 1.5), (25.0, 0.5),
                                            (60.0, 1.5), (600.0, 1.5)])
@pytest.mark.parametrize("last", [2.0, 23.4, 23.5, 23.6, 25.0, 48.9,
                                  49.1, 173.3, 599.0])
def test_kept_ranges_tile_without_holes(seg_len, lead_s, last):
    bases = window_starts(last, seg_len, lead_s)
    assert bases, "a chart ending past lead_s gets at least one window"
    kept = _kept(bases, seg_len, lead_s)
    assert kept[0][0] == lead_s
    for (lo0, hi0), (lo1, hi1) in zip(kept, kept[1:]):
        assert lo1 == pytest.approx(hi0)      # contiguous, no overlap
    assert kept[-1][0] <= last < kept[-1][1] + 1e-6  # last object is inside


@pytest.mark.parametrize("seed", range(5))
def test_every_note_in_exactly_one_window(seed):
    rng = np.random.default_rng(seed)
    seg_len, lead_s = 25.0, LEAD_S
    ts = np.sort(rng.uniform(2.0, 320.0, size=3000))
    bases = window_starts(float(ts[-1]), seg_len, lead_s)
    hits = np.zeros(len(ts), dtype=int)
    for b in bases:
        hits += ((ts >= b + lead_s) & (ts < b + seg_len)).astype(int)
    assert (hits == 1).all()


def test_old_stride_left_holes():
    """Documentary: the seg_len stride misses the lead gap of k > 0."""
    ts = np.array([25.2, 26.0])               # inside window 1's lead gap
    covered = any(b + 1.5 <= t < b + 25.0 for b in (0.0, 25.0) for t in ts)
    assert not covered
    bases = window_starts(27.0, 25.0, 1.5)
    assert any(b + 1.5 <= 26.0 < b + 25.0 for b in bases)


def test_short_and_degenerate_charts():
    assert window_starts(1.0, 25.0, 1.5) == []
    assert window_starts(1.5, 25.0, 1.5) == [0.0]
    with pytest.raises(ValueError):
        window_starts(10.0, 1.5, 1.5)
