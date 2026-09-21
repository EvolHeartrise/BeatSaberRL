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


@pytest.mark.parametrize("shape", SHAPES)
def test_spectra(shape):
    g = torch.randn(*shape, generator=torch.Generator().manual_seed(0))
    sj = _sv(tb.newton_schulz5(g, coefs=tb.MUON_NS_SCHEDULES["jordan"]))
    st = _sv(tb.newton_schulz5(g, coefs=tb.MUON_NS_SCHEDULES["tuned"]))
    # a Gaussian matrix's normalised singular values sit well above 1e-2,
    # so the analytic bands apply: jordan [0.68, 1.13], tuned [0.98, 1.03]
    assert 0.65 <= sj.min() and sj.max() <= 1.16, (sj.min(), sj.max())
    assert 0.97 <= st.min() and st.max() <= 1.04, (st.min(), st.max())
    assert (st.max() - st.min()) < (sj.max() - sj.min()) / 3


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
