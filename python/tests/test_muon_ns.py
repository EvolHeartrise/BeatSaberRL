"""Muon's Newton-Schulz coefficient schedule (MUON_NS_COEFS, the only
schedule since v56 / 2026-09-22: the modded-nanogpt 2025-02-14 per-step
schedule PufferLib 5.0 ships). Checks the spectrum it returns (singular
values in [0.98, 1.03] for input sigma >= 0.01, nothing overshooting),
that the output matches the composed polynomial applied to the input
spectrum, that the batched bmm Newton-Schulz (the only path since
2026-09-22) and the Muon step match an independent per-matrix reference
defined here, and that the Muon step is RMS-matched. Needs torch (run
on the pod)."""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train_batch as tb  # noqa: E402

SHAPES = [(192, 384), (384, 192), (384, 384), (768, 192)]


def _sv(x):
    return torch.linalg.svdvals(x.double())


def _ns_ref(g, eps=1e-7):
    """Per-matrix reference: the same five quintic NS steps, one matrix
    at a time with plain mm (independent of the trainer's bmm path)."""
    x = g / (g.norm() + eps)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for a, b, c in tb.MUON_NS_COEFS:
        s = x @ x.T
        x = a * x + (b * s + c * (s @ s)) @ x
    return x.T if tall else x


def _muon_ref_step(groups, bufs, mu):
    """Per-matrix reference Muon step: Nesterov momentum -> _ns_ref ->
    0.2*sqrt(max(A, B)) RMS-matched SGD step."""
    for g in groups:
        for p in g["params"]:
            if p.grad is None:
                continue
            buf = bufs[id(p)]
            buf.mul_(mu).add_(p.grad)
            o = _ns_ref(p.grad.add(buf, alpha=mu))
            p.add_(o, alpha=-g["lr"] * 0.2 * math.sqrt(max(p.shape)))


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
    out = _sv(tb.newton_schulz5_batched(g[None])[0])
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


@pytest.mark.parametrize("shape", SHAPES)
def test_batched_matches_per_matrix(shape):
    gen = torch.Generator().manual_seed(1)
    G = torch.randn(4, *shape, generator=gen)
    bat = tb.newton_schulz5_batched(G)
    for k in range(4):
        assert (bat[k] - _ns_ref(G[k])).abs().max() < 1e-6


def _groups(seed):
    g = torch.Generator().manual_seed(seed)
    shapes = ([(192, 192)] * 4 + [(768, 192)] * 2 + [(192, 768)] * 2
              + [(384, 384)] + [(384, 576)])
    ps = [torch.randn(*s, generator=g) for s in shapes]
    for k, p in enumerate(ps):          # a few params without a grad
        p.grad = None if k in (1, 5, 8) else torch.randn(p.shape,
                                                         generator=g)
    return [{"params": ps[:7], "lr": 1e-3, "lr_mult": 1.0},
            {"params": ps[7:], "lr": 3e-3, "lr_mult": 3.0}]


def test_muon_step_matches_per_matrix_reference():
    ga, gb = _groups(0), _groups(0)
    ref_bufs = {id(p): torch.zeros_like(p) for g in ga for p in g["params"]}
    muon = tb.Muon(gb, momentum=0.95)
    pa_all = [p for g in ga for p in g["params"]]
    for _ in range(3):
        _muon_ref_step(ga, ref_bufs, 0.95)
        muon.step()
        for pa, pb in zip(pa_all, muon._params()):
            if pa.grad is None:
                assert torch.equal(pa, pb)
                continue
            assert torch.allclose(pa, pb, rtol=0, atol=2e-6), \
                (pa - pb).abs().max()
            assert torch.allclose(ref_bufs[id(pa)], muon.state[pb],
                                  rtol=0, atol=1e-6)


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
