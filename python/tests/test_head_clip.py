"""Per-head (sub-action) PPO clipping, --head-clip (2026-09-06).

Pinned claims:

1. `head_logp=True` returns the (B, 29) per-head log-probs whose sum is
   bit-identical to the default joint log-prob (same gather, same
   reduction order), so the stored joint log-prob / kl / clipfrac
   diagnostics are unchanged by the mode.
2. At ratio 1 (first micro-batch of an update) the per-head loss has
   EXACTLY the joint loss's gradient w.r.t. every head's log-prob
   (-adv for each head): lr / ent_coef / vf_coef keep their meaning.
3. One head outside 1 +/- CLIP silences only ITS gradient under the
   per-head loss; under the joint ratio it silences all 29.
4. Dual-clip (adv < 0) binds per head too: a head with ratio > c gets
   the flat ceiling (zero gradient) while its siblings keep theirs.
5. The [kl] line carries `cfh` only when given.
"""

import torch

from policy import N_ACTION_HEADS, NotePolicy, OBS_DIM
from train_batch import (CLIP, DUAL_CLIP_C, format_kl_line, pg_loss_fn,
                         pg_loss_heads)

H = N_ACTION_HEADS


def _lp(B, seed):
    g = torch.Generator().manual_seed(seed)
    return -torch.rand(B, H, generator=g, dtype=torch.float64) - 0.1


def test_head_logp_sum_is_the_joint_logprob():
    torch.manual_seed(0)
    pol = NotePolicy(n_gears=5).double()
    obs = torch.randn(8, OBS_DIM, dtype=torch.float64)
    with torch.no_grad():
        a, lp_h, _, _ = pol.get_action_and_value(obs, head_logp=True)
        _, lp, _, _ = pol.get_action_and_value(obs, a, head_logp=False)
    assert lp_h.shape == (8, H)
    assert torch.equal(lp_h.sum(-1), lp)


def test_gradient_matches_joint_loss_at_ratio_one():
    B = 64
    old = _lp(B, 1)
    adv = torch.randn(B, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    new_j = old.clone().requires_grad_(True)
    new_h = old.clone().requires_grad_(True)
    pg_loss_fn((new_j - old).sum(-1).exp(), adv).backward()
    pg_loss_heads((new_h - old).exp(), adv).backward()
    # both: d/dlogp_h = -adv / B for every head
    expect = (-adv / B).unsqueeze(-1).expand(B, H)
    assert torch.allclose(new_j.grad, expect)
    assert torch.allclose(new_h.grad, expect)
    assert torch.allclose(new_j.grad, new_h.grad)


def test_one_head_out_of_region_silences_only_itself():
    B = 4
    old = _lp(B, 3)
    adv = torch.ones(B, dtype=torch.float64)          # positive advantages
    new = old.clone()
    new[:, 7] += 0.4                                   # head 7 ratio e^0.4 = 1.49 > 1.2
    nh = new.clone().requires_grad_(True)
    pg_loss_heads((nh - old).exp(), adv).backward()
    g = nh.grad
    assert torch.all(g[:, 7] == 0)                     # clipped head: no gradient
    others = [h for h in range(H) if h != 7]
    assert torch.all(g[:, others] != 0)                # every sibling still learns
    # joint ratio: the same perturbation zeroes ALL 29 heads' gradient
    nj = new.clone().requires_grad_(True)
    pg_loss_fn((nj - old).sum(-1).exp(), adv).backward()
    assert torch.all(nj.grad == 0)


def test_dual_clip_ceiling_is_per_head():
    B = 4
    old = _lp(B, 4)
    adv = -torch.ones(B, dtype=torch.float64)         # negative advantages
    new = old.clone()
    new[:, 3] += 1.5                                   # ratio e^1.5 = 4.48 > c = 3
    nh = new.clone().requires_grad_(True)
    loss = pg_loss_heads((nh - old).exp(), adv)
    loss.backward()
    assert torch.all(nh.grad[:, 3] == 0)               # ceiling binds for head 3
    others = [h for h in range(H) if h != 3]
    assert torch.all(nh.grad[:, others] != 0)
    # the loss VALUE for that head is the flat ceiling c*|adv|
    per_head = -adv.unsqueeze(-1) * torch.clamp((new - old).exp(), 1 - CLIP, 1 + CLIP)
    per_head = torch.min(torch.max(-adv.unsqueeze(-1) * (new - old).exp(), per_head),
                         -DUAL_CLIP_C * adv.unsqueeze(-1))
    assert torch.allclose(loss, per_head.sum(-1).mean())


def test_kl_line_carries_cfh_only_when_given():
    eh = [1.0] * H
    base = format_kl_line(3, 1e-3, 2e-3, 0.1, 0.2, 0.5, 1.0, eh)
    assert "cfh" not in base
    line = format_kl_line(3, 1e-3, 2e-3, 0.1, 0.2, 0.5, 1.0, eh, cfh=0.0123)
    assert "cfh 0.0123" in line
    assert line.index("cfh") > line.index("clipfrac")
