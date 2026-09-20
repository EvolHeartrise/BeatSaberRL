"""Batched Muon step (2026-09-19) == the per-matrix step to fp rounding."""
import math
import torch

import train_batch as tb


def _groups(seed):
    g = torch.Generator().manual_seed(seed)
    shapes = [(192, 192)] * 4 + [(768, 192)] * 2 + [(192, 768)] * 2 + [(384, 384)] + [(384, 576)]
    ps = [torch.randn(*s, generator=g) for s in shapes]
    for p in ps:
        p.grad = torch.randn(p.shape, generator=g) if hash(tuple(p.shape)) % 7 else None
    return [{"params": ps[:7], "lr": 1e-3, "lr_mult": 1.0},
            {"params": ps[7:], "lr": 3e-3, "lr_mult": 3.0}]


def test_batched_matches_eager_over_steps():
    ga, gb = _groups(0), _groups(0)
    eager = tb.Muon(ga, momentum=0.95, batched=False)
    batched = tb.Muon(gb, momentum=0.95, batched=True)
    for _ in range(3):
        eager.step()
        batched.step()
        for pa, pb in zip(eager._params(), batched._params()):
            if pa.grad is None:
                assert torch.equal(pa, pb)
                continue
            assert torch.allclose(pa, pb, rtol=0, atol=2e-6), \
                (pa - pb).abs().max()
            assert torch.allclose(eager.state[pa], batched.state[pb], rtol=0, atol=1e-6)


def test_batched_ns_is_the_per_matrix_ns():
    g = torch.Generator().manual_seed(1)
    for shape in [(192, 192), (768, 192), (192, 768)]:
        G = torch.randn(3, *shape, generator=g)
        B = tb.newton_schulz5_batched(G)
        for k in range(3):
            E = tb.newton_schulz5(G[k])
            assert (B[k] - E).abs().max() < 1e-6
