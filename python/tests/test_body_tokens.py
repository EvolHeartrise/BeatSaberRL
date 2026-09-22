"""--body-tokens 3 (2026-09-22, user): the single body token decomposed
into [torso, leg_L, leg_R]. Pins the lane bookkeeping (the three tokens
see exactly the old body token's lanes, legs mirror each other joint for
joint, the torso holds spine/root/head), the token/readout shapes, the
arch marker round trip through load_policy, that the CUDA fp16 fused
path agrees with the fp32 CPU path, and that the old layout is
untouched (body_tokens=1 == pre-2026-09-22 bit for bit). Needs torch."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import policy as P  # noqa: E402
from beat_arena_env import constants as C  # noqa: E402


def _joint_q_lane(j):
    # q lanes: joints in order, the two wrist rolls (7, 14) take cos/sin
    return C.OFF_Q + j + sum(1 for w in C.WRAP_JOINTS if w < j)


def test_lane_bookkeeping():
    q = lambda j: _joint_q_lane(j)
    qd = lambda j: C.OFF_QD + j
    # torso: root pose/vel block, spine q/qd, head, contact, spine tau
    torso = set(P._TORSO_IDX)
    assert set(range(0, 15)) <= torso                       # root pos/rot6/vlin/vang
    assert {q(0), q(1), q(2), qd(0), qd(1), qd(2)} <= torso
    assert set(range(C.OFF_HEAD, C.OFF_HEAD + 3)) <= torso
    assert set(range(C.OFF_CONTACT, C.OFF_CONTACT + 2)) <= torso
    assert set(range(P._TAU, P._TAU + 3)) <= torso
    assert len(torso) == P.TORSO_FEAT == 29
    # legs: hip pitch/yaw/roll, knee, ankle pitch/roll q + qd, that
    # foot's heel/toe impulse, its knee + foot position and velocity,
    # the 6 shared leg tau axes; left and right mirror lane for lane
    L, R = P._LEG_L_IDX, P._LEG_R_IDX
    assert len(L) == len(R) == P.LEG_FEAT == 32
    for a in range(6):
        assert L[a] == q(C.A_L_LEG + a) and R[a] == q(C.A_R_LEG + a)
        assert L[6 + a] == qd(C.A_L_LEG + a) and R[6 + a] == qd(C.A_R_LEG + a)
    gi = C.OBS_OFF["ground_imp"]; bp = C.OBS_OFF["body_pts"]; bv = C.OBS_OFF["body_pt_vel"]
    assert L[12:14] == (gi, gi + 1) and R[12:14] == (gi + 2, gi + 3)
    assert L[14:17] == tuple(range(bp + 6, bp + 9))      # l_knee
    assert R[14:17] == tuple(range(bp + 9, bp + 12))     # r_knee
    assert L[17:20] == tuple(range(bp + 12, bp + 15))    # l_foot
    assert R[17:20] == tuple(range(bp + 15, bp + 18))    # r_foot
    assert L[20:23] == tuple(range(bv + 6, bv + 9)) and R[20:23] == tuple(range(bv + 9, bv + 12))
    assert L[23:26] == tuple(range(bv + 12, bv + 15)) and R[23:26] == tuple(range(bv + 15, bv + 18))
    assert L[26:] == R[26:] == tuple(range(P._TAU + 10, P._TAU + 16))
    # together = the old body token exactly (leg tau axes twice)
    assert set(P._TORSO_IDX) | set(L) | set(R) == set(P._BODY_IDX)
    assert len(P._TORSO_IDX) + len(L) + len(R) == P.BODY_FEAT + 6
    # nothing from the saber tokens leaks in
    assert not (set(P._BODY3_IDX) & set(P._SABER_L_IDX + P._SABER_R_IDX))


def _policy(body_tokens, seed=0, **kw):
    torch.manual_seed(seed)
    return P.NotePolicy(embed=64, trunk_dim=128, n_blocks=2, n_heads=4,
                        n_gears=5, ffn_mult=4, critic="adapool_v1",
                        note_geo=True, note_geo_norm=True, trunk_arch="simba",
                        value_bins=101, body_tokens=body_tokens, **kw)


def _obs(B, seed=1):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(B, C.OBS_DIM, generator=g) * 0.5
    # note mask: some slots empty
    obs[:, C.OFF_NOTE_MASK:C.OFF_NOTE_MASK + C.N_NOTE_SLOTS] = (
        torch.rand(B, C.N_NOTE_SLOTS, generator=g) < 0.7).float()
    return obs


def test_shapes_and_marker():
    p3 = _policy(3)
    assert p3.arch["body_tokens"] == 3 and p3.n_lead == 5
    assert p3.arch["body_token"] is True
    assert hasattr(p3, "torso_mlp") and hasattr(p3, "leg_mlp") and not hasattr(p3, "body_mlp")
    assert p3.trunk_proj.in_features == 5 * 64
    p1 = _policy(1)
    assert p1.arch["body_tokens"] == 1 and p1.n_lead == 3 and hasattr(p1, "body_mlp")
    assert p1.trunk_proj.in_features == 3 * 64
    obs = _obs(8)
    for p in (p1, p3):
        logits, value = p.forward_stacked(obs)
        assert logits.shape == (8, 29, 5) and value.shape == (8,)
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    # parameter delta: -body_mlp(87->64->64) + torso(29->64->64) + leg(34->64->64)
    n1 = sum(q.numel() for q in p1.parameters()); n3 = sum(q.numel() for q in p3.parameters())
    d = (29 * 64 + 64 + 64 * 64 + 64) + (34 * 64 + 64 + 64 * 64 + 64) - (87 * 64 + 64 + 64 * 64 + 64) + (2 * 64) * 128
    assert n3 - n1 == d, (n3 - n1, d)


def test_load_policy_round_trip_and_legacy_default():
    p3 = _policy(3)
    ck = {"arch": dict(p3.arch), "model": p3.state_dict()}
    q = P.load_policy(ck)
    assert q.body_tokens == 3 and q.n_lead == 5
    q.load_state_dict(ck["model"])
    obs = _obs(4)
    a, va = p3.forward_stacked(obs); b, vb = q.forward_stacked(obs)
    assert torch.equal(a, b) and torch.equal(va, vb)
    # a checkpoint without the marker (every one before 2026-09-22)
    p1 = _policy(1)
    arch = dict(p1.arch); arch.pop("body_tokens")
    q1 = P.load_policy({"arch": arch, "model": p1.state_dict()})
    assert q1.body_tokens == 1 and hasattr(q1, "body_mlp")


def test_legs_are_side_aware():
    # swapping the two leg feature blocks (and nothing else) must change
    # the output: the side one-hot distinguishes them
    p3 = _policy(3)
    obs = _obs(4)
    a, _ = p3.forward_stacked(obs)
    sw = obs.clone()
    L = torch.tensor(P._LEG_L_IDX); R = torch.tensor(P._LEG_R_IDX)
    sw[:, L], sw[:, R] = obs[:, R], obs[:, L]
    b, _ = p3.forward_stacked(sw)
    assert not torch.allclose(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_fp16_matches_cpu_fp32():
    p3 = _policy(3)
    obs = _obs(64)
    ref, vref = p3.forward_stacked(obs)
    g = _policy(3).cuda(); g.load_state_dict(p3.state_dict()); g.fp16 = True
    out, vout = g.forward_stacked(obs.cuda())
    assert torch.allclose(out.float().cpu(), ref, atol=3e-2, rtol=3e-2)
    assert torch.allclose(vout.float().cpu(), vref, atol=5e-2, rtol=5e-2)
    # fp32 CUDA path (no fp16) matches CPU tightly
    g.fp16 = False
    out32, v32 = g.forward_stacked(obs.cuda())
    assert torch.allclose(out32.cpu(), ref, atol=1e-4, rtol=1e-4)
