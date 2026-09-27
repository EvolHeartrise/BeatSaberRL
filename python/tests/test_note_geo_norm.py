"""note-geo normalisation (2026-09-21, user; the only layout since
2026-09-22): every geometry lane is O(1) for notes at any distance and
the z decode uses the lane's true asinh encoding."""
import math

import torch

from policy import _note_geo, NOTE_GEO_FEAT, SABER_FEAT, _Z_SCALE


def _notes(zs, c):
    B, N = 1, len(zs)
    notes = torch.zeros(B, 24, 18)
    for i, z in enumerate(zs):
        notes[0, i, 0] = 0.3          # x / X_NORM
        notes[0, i, 1] = 0.2          # (y - center) / Y_NORM
        notes[0, i, 2] = math.asinh(z / c)
        notes[0, i, 3] = -20.0 / _Z_SCALE   # speed 20 m/s
        notes[0, i, 6] = 0.0; notes[0, i, 7] = -1.0
    sabers = torch.zeros(B, 2, SABER_FEAT)
    sabers[:, :, 15:18] = torch.tensor([0.1, 1.2, -0.65])
    sabers[:, :, 21:24] = torch.tensor([0.0, -1.0, 0.0])
    return notes, sabers


def test_norm_lanes_are_order_one_at_any_distance():
    zs = [0.0, 0.6, 2.0, 12.0, 40.0, 100.0, 2000.0]
    notes, sabers = _notes(zs, c=8.0)
    g = _note_geo(notes, sabers, 8.0)
    assert g.shape == (1, 24, NOTE_GEO_FEAT)
    assert g[0, :len(zs)].abs().max() < 8.0
    # per hand: dp(3) dist(1) tpl(1) vdot(1) bdot(1); z offset lane = asinh((z - wz)/8)
    for i, z in enumerate(zs):
        dz = (z - (-0.65))
        assert abs(g[0, i, 2].item() - math.asinh(dz / 8)) < 1e-3
        assert -0.17 <= g[0, i, 4].item() <= 1.0 + 1e-6          # tpl / 3
    # far notes stay distinct
    assert g[0, 5, 2] > g[0, 4, 2] > g[0, 3, 2]
    assert g[0, 6, 2] > g[0, 5, 2]


def test_norm_true_decodes_true_z():
    notes, sabers = _notes([100.0], c=8.0)
    g = _note_geo(notes, sabers, 8.0)
    # tpl uses the TRUE 100 m (clipped at 3 s -> lane 1.0), not 8*asinh(100/8) = 25.8 m (1.3 s)
    assert abs(g[0, 0, 4].item() - 1.0) < 1e-6


def test_load_policy_refuses_raw_metre_note_geo():
    # a note_geo checkpoint without the marker (pre-v55 raw-metre lanes)
    # is refused; one without note_geo needs no marker
    import pytest
    from policy import NotePolicy, load_policy
    torch.manual_seed(0)
    p = NotePolicy(embed=64, trunk_dim=128, n_blocks=1, n_gears=5,
                   note_geo=True)
    assert p.arch["note_geo_norm"] is True
    for bad in (None, False):
        arch = dict(p.arch)
        if bad is None:
            arch.pop("note_geo_norm")
        else:
            arch["note_geo_norm"] = bad
        with pytest.raises(SystemExit, match="raw-metre"):
            load_policy({"arch": arch, "state_dict": p.state_dict()})
    q = NotePolicy(embed=64, trunk_dim=128, n_blocks=1, n_gears=5)
    arch = dict(q.arch); arch.pop("note_geo_norm")
    load_policy({"arch": arch, "state_dict": q.state_dict()})


def test_fp16_stream_rollout_and_update_forwards_agree_bitwise():
    """--fp16 all: the rollout forwards fp32 obs, the update forwards the
    half b_obs copy of the SAME obs. Every consumer rounds to fp16 at use,
    so the two logit sets must be bit-identical — the note-geo lanes
    included (2026-09-26 fix: they used to be computed from fp32 lanes in
    the rollout and from fp16 lanes in fp16 arithmetic in the update)."""
    import pytest
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (the fp16 stream is CUDA-only)")
    from policy import NotePolicy, OBS_DIM
    torch.manual_seed(0)
    p = NotePolicy(embed=64, trunk_dim=128, n_blocks=2, n_heads=4,
                   note_geo=True).cuda().eval()
    p.fp16 = True
    obs = torch.randn(256, OBS_DIM, device="cuda") * 0.7
    with torch.no_grad():
        l32, v32 = p.forward_stacked(obs)
        l16, v16 = p.forward_stacked(obs.half())
    assert torch.equal(l32, l16), (l32 - l16).abs().max()
    assert torch.equal(v32, v16), (v32 - v16).abs().max()
