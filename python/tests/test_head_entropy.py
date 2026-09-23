"""Per-head entropy + the [kl] diagnostic line (2026-09-02, Tier 0.3 of
PERF_AUDIT_2026-09-02_static.md, C §1.2 / §2.2 / §5).

The claims pinned here:

1. The returned `entropy` IS the sum of the stashed per-head
   entropies (`_ent_h`), so the printed `ent` and the per-head [kl]
   breakdown describe the same numbers.
2. `HEAD_GROUPS` is derived from the env's actuated-order contract, not
   invented here: policy.py duplicates it to stay a leaf module, so the
   duplication is checked against beat_arena_env.constants.
3. The reporting arithmetic (group MEANS, weakest head, 29-head sum) and
   the line format, so a later reader knows what the numbers meant.
"""

import math

import torch

from beat_arena_env.constants import (A_L_ARM, A_L_LEG, A_R_ARM, A_R_LEG,
                                      A_SPINE, JOINT_NAMES)
from policy import HEAD_GROUPS, N_ACTION_HEADS, NotePolicy, OBS_DIM
from train_batch import entropy_report, format_kl_line


# ---------------------------------------------------------------- groups

def test_head_groups_are_the_env_actuated_blocks():
    """policy.HEAD_GROUPS against the env contract: group starts are the
    env's actuated-order offsets, the groups tile all 29 heads
    contiguously in SPEC-v10 block sizes, and each group's joints carry
    its name prefix. (train_batch / metrics_log copies: see
    test_metrics_log.test_head_groups_match_trainer.)"""
    starts = {nm: a for nm, a, _ in HEAD_GROUPS}
    assert starts == {"spine": A_SPINE, "armL": A_L_ARM, "armR": A_R_ARM,
                      "legL": A_L_LEG, "legR": A_R_LEG}
    edges = [(a, b) for _, a, b in HEAD_GROUPS]
    assert edges[0][0] == 0
    assert edges[-1][1] == N_ACTION_HEADS == len(JOINT_NAMES)
    for (_, b), (a2, _) in zip(edges, edges[1:]):
        assert b == a2
    assert [b - a for a, b in edges] == [3, 7, 7, 6, 6]
    for nm, a, b in HEAD_GROUPS:
        names = JOINT_NAMES[a:b]
        if nm == "spine":
            assert all(n.startswith("spine_") for n in names)
        else:
            side = "l_" if nm.endswith("L") else "r_"
            assert all(n.startswith(side) for n in names)


# --------------------------------------------------------- entropy split

def _policy():
    torch.manual_seed(0)
    return NotePolicy(n_gears=3)


def test_stashed_per_head_entropy_sums_to_the_returned_entropy():
    pol = _policy()
    obs = torch.randn(7, OBS_DIM) * 0.7
    _, _, entropy, _ = pol.get_action_and_value(obs)
    assert pol._ent_h.shape == (7, N_ACTION_HEADS)
    # bitwise: `entropy` IS `_ent_h.sum(-1)`, the same reduction the
    # pre-split code ran second
    assert torch.equal(entropy, pol._ent_h.sum(-1))


def test_entropy_is_the_categorical_entropy_of_the_logits():
    """Correctness (not a history pin): the summed entropy matches an
    independent -(p log p) over forward_stacked's logits."""
    pol = _policy()
    obs = torch.randn(7, OBS_DIM) * 0.7
    _, _, entropy, _ = pol.get_action_and_value(obs)
    logits, _ = pol.forward_stacked(obs)
    logsm = torch.log_softmax(logits.double(), dim=-1)
    ref = -(logsm.exp() * logsm).sum(-1).sum(-1)
    assert torch.allclose(entropy.double(), ref, rtol=1e-5, atol=1e-5)


def test_need_entropy_false_leaves_no_stash():
    pol = _policy()
    obs = torch.randn(4, OBS_DIM) * 0.7
    pol._ent_h = None
    _, _, entropy, _ = pol.get_action_and_value(obs, need_entropy=False)
    assert entropy is None and pol._ent_h is None


def test_per_head_entropy_is_bounded_by_ln_k():
    pol = _policy()
    obs = torch.randn(16, OBS_DIM) * 0.7
    pol.get_action_and_value(obs)
    assert pol._ent_h.max().item() <= math.log(3) + 1e-6
    assert pol._ent_h.min().item() >= -1e-6


# ------------------------------------------------------------- reporting

def test_entropy_report_groups_are_means_not_sums():
    eh = [float(i) for i in range(29)]
    groups, _, _, _ = entropy_report(eh)
    assert dict(groups)["spine"] == (0 + 1 + 2) / 3
    assert dict(groups)["armL"] == sum(range(3, 10)) / 7
    assert dict(groups)["legR"] == sum(range(23, 29)) / 6


def test_entropy_report_total_is_the_29_head_sum():
    eh = [0.5] * 29
    _, _, _, total = entropy_report(eh)
    assert total == 14.5


def test_entropy_report_finds_the_weakest_head():
    eh = [1.0] * 29
    eh[17] = 0.25
    _, lo, lo_v, _ = entropy_report(eh)
    assert (lo, lo_v) == (17, 0.25)


def test_entropy_report_min_ties_take_the_first_head():
    eh = [1.0] * 29
    eh[4] = eh[20] = 0.1
    _, lo, _, _ = entropy_report(eh)
    assert lo == 4


def test_format_kl_line_layout():
    line = format_kl_line(5, 1.5e-4, 3.25e-3, 0.0012, 0.0431,
                          0.834, 0.712, [1.0] * 29)
    assert line.startswith("[kl] upd   5  ")
    assert "kl(mb1/mbN) 1.50e-04/3.25e-03" in line
    assert "clipfrac 0.0012/0.0431" in line
    assert "gnorm 0.834  clipc 0.712" in line
    assert "spine 1.000" in line and "legR 1.000" in line
    assert "min 1.000 (h0)" in line
    assert "entfull 29.000" in line


def test_format_kl_line_is_one_line():
    line = format_kl_line(1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                          [float(i) / 29 for i in range(29)])
    assert "\n" not in line
