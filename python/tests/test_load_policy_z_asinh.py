"""load_policy carries the obs z encoding (2026-09-22 eval bug): the
note-geo lanes decode each note's distance as C sinh(lane) with C =
the policy's z_asinh. load_policy used to rebuild the policy without it,
so every v55+ bench / replay decoded far notes linearly (lane x 8) while
training used C sinh(lane). A policy built the way train_batch.py builds
it must forward bit-identically after a checkpoint round trip through
load_policy. CPU only."""
import math

import pytest
import torch

import policy as P
from beat_arena_env import constants as C


def _trainer_policy(z_asinh, seed=0):
    # train_batch.py's fresh-run construction (recipe markers: note_geo
    # + simba + HL critic), small widths for speed
    torch.manual_seed(seed)
    return P.NotePolicy(embed=64, trunk_dim=128, n_blocks=2, n_heads=4,
                        n_gears=5, ffn_mult=4, note_geo=True,
                        trunk_arch="simba",
                        value_bins=101, value_vmax=6.0, value_sigma=0.75,
                        z_asinh=z_asinh)


def _obs_far_notes(B, c, seed=1):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(B, C.OBS_DIM, generator=g) * 0.5
    notes = obs[:, C.OFF_NOTES:C.OFF_NOTE_MASK].view(
        B, C.N_NOTE_SLOTS, C.NOTE_FEAT)
    # z lane = asinh(z / c): near notes AND far ones (up to 2 km -> lane
    # ~6), where c sinh(lane) and the linear lane x 8 disagree by metres
    z = torch.rand(B, C.N_NOTE_SLOTS, generator=g) ** 3 * 2000.0
    notes[..., 2] = torch.asinh(z / c)
    notes[..., 3] = -(10.0 + 20.0 * torch.rand(B, C.N_NOTE_SLOTS,
                                                generator=g)) / 8.0
    obs[:, C.OFF_NOTE_MASK:C.OFF_NOTE_MASK + C.N_NOTE_SLOTS] = 1.0
    assert notes[..., 2].max() > math.asinh(500.0 / c)
    return obs


@pytest.mark.parametrize("c", [8.0, 3.0])
def test_round_trip_forward_identical(c):
    p = _trainer_policy(c).eval()
    assert p.arch["z_asinh"] == c
    ck = {"arch": dict(p.arch), "state_dict": p.state_dict()}
    q = P.load_policy(ck).eval()
    assert q.arch["z_asinh"] == c and q.z_asinh == c
    obs = _obs_far_notes(16, c)
    with torch.no_grad():
        a, va = p.forward_stacked(obs)
        b, vb = q.forward_stacked(obs)
    assert torch.equal(a, b) and torch.equal(va, vb)


def test_obs_exercise_the_z_decode():
    # non-vacuity: the same weights with another z_asinh must forward
    # differently on these obs (else the round trip above proves nothing)
    p = _trainer_policy(3.0).eval()
    q = _trainer_policy(8.0).eval()
    q.load_state_dict(p.state_dict())
    obs = _obs_far_notes(16, 3.0)
    with torch.no_grad():
        a, _ = p.forward_stacked(obs)
        b, _ = q.forward_stacked(obs)
    assert not torch.allclose(a, b)


def test_refuses_missing_or_bad_z_asinh():
    p = _trainer_policy(8.0)
    for bad in (None, 0.0, -8.0):
        arch = dict(p.arch)
        if bad is None:
            arch.pop("z_asinh")
        else:
            arch["z_asinh"] = bad
        with pytest.raises(SystemExit, match="z_asinh"):
            P.load_policy({"arch": arch, "state_dict": p.state_dict()})
    with pytest.raises(ValueError, match="z_asinh"):
        _trainer_policy(0.0)
