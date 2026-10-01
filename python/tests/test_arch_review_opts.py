"""--muon-enc-hidden and --depth-init (architecture review, 2026-10-01)."""
import torch

import policy as P
import train_batch as tb


def _pol(n_blocks=3):
    torch.manual_seed(0)
    return P.NotePolicy(embed=64, trunk_dim=96, n_blocks=n_blocks,
                        n_heads=4, n_gears=5, ffn_mult=4)


def test_muon_enc_hidden_moves_only_encoder_hidden_layers():
    pol = _pol()
    base = set(tb.muon_param_names(pol))
    more = set(tb.muon_param_names(pol, enc_hidden=True))
    enc = ("note_mlp", "saber_mlp", "torso_mlp", "leg_mlp")
    assert not any(n.startswith(enc) for n in base)
    assert more - base == {f"{e}.2.weight" for e in enc}
    assert not any(n.startswith(enc) and ".0." in n for n in more)
    assert not any(n.startswith("heads.") for n in more)


def test_depth_init_scales_residual_writes_and_heads():
    pol, ref = _pol(4), _pol(4)
    msg = P.apply_depth_init(pol)
    f = (2 * 4) ** -0.5
    for b, r in zip(pol.blocks, ref.blocks):
        for lin, rl in ((b.out_proj, r.out_proj), (b.ffn[-1], r.ffn[-1])):
            assert torch.allclose(lin.weight, rl.weight * f)
            assert torch.count_nonzero(lin.bias) == 0
        # everything else in the block is untouched
        assert torch.equal(b.q_proj.weight, r.q_proj.weight)
        assert torch.equal(b.ffn[0].weight, r.ffn[0].weight)
    assert torch.allclose(pol.trunk_fc2.weight, ref.trunk_fc2.weight * f)
    for h, rh in zip(pol.heads, ref.heads):
        assert torch.allclose(h.weight, rh.weight * 0.01)
        assert torch.count_nonzero(h.bias) == 0
    assert "x0.01" in msg


def test_flags_default_off():
    a = tb.build_parser().parse_args([])
    assert a.muon_enc_hidden is False and a.depth_init is False
