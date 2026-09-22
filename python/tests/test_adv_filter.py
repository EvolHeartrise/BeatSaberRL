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

import torch

from train_batch import adv_filter_indices


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


# --- 1.4 (2026-09-02, static audit C §1.1): normalization stats under
# the filter come from the FULL batch, once per update. The kept set's
# own sd is ~1.36x the full batch's, so per-minibatch stats on kept
# rows shrank the PG term 0.73x against the entropy/value terms — the
# 2026-08-31 probe ran with that confound. These pin the helper and
# the filter-off print format; the filter-off TRAINING path is pinned
# by the 5-update same-seed bit-identity run in the audit doc, which a
# unit test cannot reach.

from train_batch import adv_filter_stats, format_kl_line  # noqa: E402


def test_stats_are_full_batch_regardless_of_keep():
    g = torch.Generator().manual_seed(0)
    adv = torch.randn(4096, generator=g)
    for n in (64, 2048, 4096):
        idx = adv_filter_indices(adv, n)
        mu, sd, _ = adv_filter_stats(adv, idx)
        assert torch.equal(mu, adv.mean())
        assert torch.equal(sd, adv.std())


def test_kept_sd_ratio_gaussian_half_is_1p36():
    # E[A^2 | |A| > 0.6745 sd] = 1.857 sd^2 -> sd_kept = 1.363 sd
    g = torch.Generator().manual_seed(1)
    adv = torch.randn(1 << 20, generator=g)
    idx = adv_filter_indices(adv, 1 << 19)
    _, _, ratio = adv_filter_stats(adv, idx)
    assert abs(ratio.item() - 1.363) < 0.01, ratio.item()


def test_kept_sd_ratio_is_one_when_nothing_dropped():
    adv = torch.randn(1000)
    idx = adv_filter_indices(adv, 1000)
    _, _, ratio = adv_filter_stats(adv, idx)
    assert abs(ratio.item() - 1.0) < 1e-5


def test_kl_line_unchanged_when_filter_off():
    eh = [1.6] * 29
    base = format_kl_line(5, 2e-10, 1.45e-2, 0.0, 0.2365, 0.18, 1.0, eh)
    assert base == format_kl_line(5, 2e-10, 1.45e-2, 0.0, 0.2365, 0.18,
                                  1.0, eh, None)
    assert "afsd" not in base
    assert base.endswith("entfull 46.400")
    on = format_kl_line(5, 2e-10, 1.45e-2, 0.0, 0.2365, 0.18, 1.0, eh,
                        1.3627)
    assert on.startswith(base)
    assert on.endswith("entfull 46.400  afsd 1.363")
