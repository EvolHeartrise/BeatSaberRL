"""Top-advantage filtering pins (2026-08-31, arXiv:2606.23348 §V-E).

The paper: "we keep only the top 25% of transitions from each rollout
batch, ranked by the critic's predicted advantage, and compute the
policy-gradient update from those alone." It never specifies signed
vs absolute ranking; train_batch ranks by MAGNITUDE (two-sided —
user decision 2026-08-31; the signed one-sided variant collapsed EV
0.768 -> 0.015 in a 50-update smoke because the critic only ever saw
returns above its own predictions). These pins fix the semantics:
largest |advantage| kept, near-zero rows drop first, count exact
even on ties.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_batch import adv_filter_indices  # noqa: E402


def test_keeps_largest_magnitude_both_signs():
    adv = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0, 4.0])
    idx = adv_filter_indices(adv, 3)
    assert sorted(idx.tolist()) == [0, 4, 5]


def test_drops_near_zero_first():
    adv = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0, 4.0])
    idx = adv_filter_indices(adv, 5)
    assert sorted(idx.tolist()) == [0, 1, 3, 4, 5]


def test_exact_count_on_ties():
    adv = torch.ones(1000)
    assert adv_filter_indices(adv, 500).numel() == 500


def test_indices_unique():
    adv = torch.randn(4096)
    idx = adv_filter_indices(adv, 2048)
    assert idx.unique().numel() == 2048
