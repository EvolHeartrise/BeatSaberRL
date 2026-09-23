"""note-geo normalisation (2026-09-21, user): with norm=True every geometry
lane is O(1) for notes at any distance, the z decode uses the lane's true
asinh encoding, and norm=False reproduces the pre-2026-09-21 lanes exactly
(z decoded as lane x 8, whatever the lane's encoding)."""
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
    g = _note_geo(notes, sabers, norm=True, z_asinh=8.0)
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


def test_norm_false_matches_legacy_lanes():
    zs = [0.0, 2.0, 12.0, 40.0]
    notes, sabers = _notes(zs, c=8.0)
    g_old = _note_geo(notes, sabers)
    g_ref = _note_geo(notes, sabers, norm=False, z_asinh=8.0)
    assert torch.equal(g_old, g_ref)
    # legacy lanes are raw half-metres of lane x 8: 40 m note -> the
    # decoded z is 8 asinh(40 / 8), dp_z = (that + 0.65) / 2
    assert abs(g_old[0, 3, 2].item()
               - (8.0 * math.asinh(40.0 / 8.0) + 0.65) * 0.5) < 1e-4


def test_norm_true_decodes_true_z():
    notes, sabers = _notes([100.0], c=8.0)
    g = _note_geo(notes, sabers, norm=True, z_asinh=8.0)
    # tpl uses the TRUE 100 m (clipped at 3 s -> lane 1.0), not 8*asinh(100/8) = 25.8 m (1.3 s)
    assert abs(g[0, 0, 4].item() - 1.0) < 1e-6
