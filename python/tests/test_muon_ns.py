"""--muon-ns (2026-09-21): the Newton-Schulz coefficient schedules.

jordan = Keller Jordan's single quintic x 5 (every Muon run before v56);
tuned = the modded-nanogpt 2025-02-14 per-step schedule PufferLib 5.0
ships. Checks the spectrum each schedule returns, that the batched bmm
path matches the per-matrix path, that jordan is unchanged by the
refactor (coefs=None == the jordan schedule, bit for bit), and that the
Muon optimizer wires the flag through. Needs torch (run on the pod)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train_batch as tb  # noqa: E402

SHAPES = [(192, 384), (384, 192), (384, 384), (768, 192)]


def _sv(x):
    return torch.linalg.svdvals(x.double())


def _poly(coefs, x):
    for a, b, c in coefs:
        x = a * x + b * x ** 3 + c * x ** 5
    return x


@pytest.mark.parametrize("shape", SHAPES)
def test_spectra(shape):
    g = torch.randn(*shape, generator=torch.Generator().manual_seed(0))
    s_in = _sv(g / g.norm())                 # the NS input spectrum
    for name, lo, hi in (("jordan", 0.65, 1.16), ("tuned", 0.97, 1.04)):
        co = tb.MUON_NS_SCHEDULES[name]
        out = _sv(tb.newton_schulz5(g, coefs=co))
        # the iteration is a polynomial in G G^T, so it maps each singular
        # value through the schedule's composed polynomial (fp32 vs
        # float64 drift is ~1e-4 after five quintic steps)
        expect = _poly(co, s_in)
        assert torch.allclose(out.sort().values, expect.sort().values,
                              atol=2e-3, rtol=1e-3), name
        # the analytic bands (jordan [0.68, 1.13], tuned [0.98, 1.03]) hold
        # for input sigma >= 0.01; a square Gaussian matrix also has
        # singular values near 0 (Marchenko-Pastur) that neither schedule
        # lifts (both reach ~2e-3), so mask those
        m = s_in >= 0.01
        assert m.sum() > 0.5 * len(s_in)
        assert lo <= expect[m].min() and expect[m].max() <= hi, (
            name, expect[m].min(), expect[m].max())
        assert expect.max() <= 1.21          # nothing overshoots, ever
    sj, st = (_poly(tb.MUON_NS_SCHEDULES[n], s_in) for n in ("jordan", "tuned"))
    m = s_in >= 0.01
    assert (st[m].max() - st[m].min()) < (sj[m].max() - sj[m].min()) / 3


@pytest.mark.parametrize("shape", SHAPES[:3])
def test_jordan_unchanged_and_batched_matches(shape):
    gen = torch.Generator().manual_seed(1)
    gs = [torch.randn(*shape, generator=gen) for _ in range(4)]
    for g in gs:
        a = tb.newton_schulz5(g)                       # refactor default
        b = tb.newton_schulz5(g, coefs=tb.MUON_NS_SCHEDULES["jordan"])
        assert torch.equal(a, b)
    for name in ("jordan", "tuned"):
        co = tb.MUON_NS_SCHEDULES[name]
        bat = tb.newton_schulz5_batched(torch.stack(gs), coefs=co)
        for k, g in enumerate(gs):
            assert torch.allclose(bat[k], tb.newton_schulz5(g, coefs=co),
                                  atol=1e-5, rtol=1e-4)


def test_muon_wires_schedule():
    w = torch.nn.Parameter(torch.randn(64, 32))
    w.grad = torch.randn(64, 32)
    outs = {}
    for name in ("jordan", "tuned"):
        p = torch.nn.Parameter(w.detach().clone())
        p.grad = w.grad.clone()
        m = tb.Muon([{"params": [p], "lr": 1e-2, "lr_mult": 1.0}], ns=name)
        assert m.state_dict()["ns"] == name
        m.step()
        outs[name] = p.detach() - w.detach()
    # both schedules move the weight by an RMS-matched 0.2 * lr per element
    for name, d in outs.items():
        rms = d.pow(2).mean().sqrt().item()
        assert abs(rms - 0.2 * 1e-2) / (0.2 * 1e-2) < 0.2, (name, rms)
    assert not torch.allclose(outs["jordan"], outs["tuned"])
    with pytest.raises(KeyError):
        tb.Muon([{"params": [w], "lr": 1e-2, "lr_mult": 1.0}], ns="polar")


def test_flag_default_is_jordan():
    sys.argv = ["train_batch.py"]
    assert tb.parse_args().muon_ns == "jordan"
