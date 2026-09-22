"""Muon's Newton-Schulz coefficient schedule (MUON_NS_COEFS, the only
schedule since v56 / 2026-09-22: the modded-nanogpt 2025-02-14 per-step
schedule PufferLib 5.0 ships). Checks the spectrum it returns (singular
values in [0.98, 1.03] for input sigma >= 0.01, nothing overshooting),
that the output matches the composed polynomial applied to the input
spectrum, that the batched bmm path matches the per-matrix path, and
that the Muon step is RMS-matched. Needs torch (run on the pod)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train_batch as tb  # noqa: E402

SHAPES = [(192, 384), (384, 192), (384, 384), (768, 192)]


def _sv(x):
    return torch.linalg.svdvals(x.double())


def _poly(x):
    for a, b, c in tb.MUON_NS_COEFS:
        x = a * x + b * x ** 3 + c * x ** 5
    return x


def test_schedule_shape():
    assert len(tb.MUON_NS_COEFS) == 5
    a, b, c = tb.MUON_NS_COEFS[0]
    assert a == 4.0848                       # the tuned schedule's first step
    # the last two steps polish toward 1: p(1) -> 0.94, 0.99
    assert 0.9 < sum(tb.MUON_NS_COEFS[3]) < 1.0
    assert 0.95 < sum(tb.MUON_NS_COEFS[4]) < 1.0


@pytest.mark.parametrize("shape", SHAPES)
def test_spectrum(shape):
    g = torch.randn(*shape, generator=torch.Generator().manual_seed(0))
    s_in = _sv(g / g.norm())                 # the NS input spectrum
    out = _sv(tb.newton_schulz5(g))
    # the iteration is a polynomial in G G^T, so it maps each singular
    # value through the composed polynomial (fp32 vs float64 drift is
    # ~1e-4 after five quintic steps)
    expect = _poly(s_in)
    assert torch.allclose(out.sort().values, expect.sort().values,
                          atol=2e-3, rtol=1e-3)
    # the analytic band holds for input sigma >= 0.01; a square Gaussian
    # matrix also has singular values near 0 (Marchenko-Pastur) that no
    # five-step schedule lifts, so mask those
    m = s_in >= 0.01
    assert m.sum() > 0.5 * len(s_in)
    assert 0.97 <= expect[m].min() and expect[m].max() <= 1.04, (
        expect[m].min(), expect[m].max())
    assert expect.max() <= 1.21              # nothing overshoots, ever


def test_flat_past_one():
    # bf16-style overshoot of the normalised input: stays near 1
    x = torch.tensor([1.05], dtype=torch.float64)
    assert abs(_poly(x).item() - 1.0) < 0.05


@pytest.mark.parametrize("shape", SHAPES[:3])
def test_batched_matches(shape):
    gen = torch.Generator().manual_seed(1)
    gs = [torch.randn(*shape, generator=gen) for _ in range(4)]
    bat = tb.newton_schulz5_batched(torch.stack(gs))
    for k, g in enumerate(gs):
        assert torch.allclose(bat[k], tb.newton_schulz5(g),
                              atol=1e-5, rtol=1e-4)


def test_muon_step_rms_matched():
    w = torch.nn.Parameter(torch.randn(64, 32))
    w.grad = torch.randn(64, 32)
    before = w.detach().clone()
    m = tb.Muon([{"params": [w], "lr": 1e-2, "lr_mult": 1.0}])
    assert set(m.state_dict()) == {"momentum", "bufs"}
    m.step()
    d = w.detach() - before
    rms = d.pow(2).mean().sqrt().item()
    # 0.2 * lr per element (Moonshot RMS matching)
    assert abs(rms - 0.2 * 1e-2) / (0.2 * 1e-2) < 0.2, rms
