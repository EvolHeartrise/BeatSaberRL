"""HL-Gauss critic head (Farebrother et al. 2024; --value-head hl,
2026-09-19): histogram over the normalized return, Gaussian-smoothed
target truncated to the support and renormalized, value = expectation."""
import math

import torch

import policy as P
import train_batch as tb


def _centers(k=101, vmax=6.0):
    return torch.linspace(-vmax, vmax, k)


def test_target_rows_sum_to_one_and_center_on_v():
    c = _centers()
    width = float(c[1] - c[0])
    v = torch.tensor([-2.0, 0.0, 0.37, 3.3])
    t = P.hl_target(v, c, 0.75 * width)
    assert torch.allclose(t.sum(-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(t @ c, v, atol=width * 0.05)       # expectation ~ v
    # sigma = 0.75 widths spreads mass over ~6 bins (paper)
    assert 4 <= int((t[1] > 1e-3).sum()) <= 8


def test_target_is_truncated_and_renormalized_at_the_edge():
    c = _centers()
    width = float(c[1] - c[0])
    t = P.hl_target(torch.tensor([50.0, -50.0]), c, 0.75 * width)
    assert torch.allclose(t.sum(-1), torch.ones(2), atol=1e-6)
    assert t[0].argmax() == len(c) - 1 and t[1].argmax() == 0
    assert torch.isfinite(t).all()


def test_loss_gradient_moves_expectation_toward_target():
    c = _centers()
    width = float(c[1] - c[0])
    logits = torch.zeros(8, len(c), requires_grad=True)
    v = torch.full((8,), 1.5)
    loss = P.hl_loss(logits, v, c, 0.75 * width)
    loss.backward()
    with torch.no_grad():
        new = logits - 500.0 * logits.grad
    e = torch.softmax(new, -1) @ c
    assert (e - 1.5).abs().max() < 1.0 and (e > 0.5).all()


def test_policy_hl_head_value_shape_logits_and_muon_split():
    pol = P.NotePolicy(embed=32, trunk_dim=48, n_blocks=1, n_heads=2,
                       n_gears=5, ffn_mult=2, note_geo=True,
                       trunk_arch="simba", value_bins=51, value_vmax=6.0)
    obs = torch.randn(4, P.OBS_DIM)
    with torch.no_grad():
        _, _, _, v = pol.get_action_and_value(obs)
    assert v.shape == (4,) and pol._value_logits is None
    _, _, _, v = pol.get_action_and_value(obs)
    assert pol._value_logits.shape == (4, 51)
    assert torch.allclose(torch.softmax(pol._value_logits, -1) @ pol.value_centers, v)
    # the 51-row head stays on Adam (Jordan's rule: output layers)
    names = tb.muon_param_names(pol)
    assert f"critic_mlp.{len(pol.critic_mlp) - 1}.weight" not in names
    assert any(n.startswith("critic_mlp.0") for n in names)
    # arch round trip through load_policy
    ck = {"arch": pol.arch, "state_dict": pol.state_dict()}
    pol2 = P.load_policy(ck)
    assert pol2.value_bins == 51 and pol2.critic_mlp[-1].out_features == 51


def test_mse_head_unchanged():
    pol = P.NotePolicy(embed=32, trunk_dim=48, n_blocks=1, n_heads=2,
                       n_gears=5, ffn_mult=2, note_geo=True, trunk_arch="simba")
    assert pol.value_bins == 0 and pol.critic_mlp[-1].out_features == 1
    assert pol.arch["value_bins"] == 0
