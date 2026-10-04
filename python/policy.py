"""Note-sequence policy for Beat Arena (SPEC.md §Policy).

Current production shape (v18, 2026-08-16): the 24-slot shared entity
window (notes + walls, distance-sorted), two saber tokens and one body
token form a 27-token sequence run through 3 full pre-LN transformer
blocks (LayerNorm -> 4-head self-attention -> residual; LayerNorm ->
4x FFN -> residual) at embed 128, then a 256-wide trunk (a plain MLP
then; the SimBa residual + LN trunk since 2026-09-15) and 29
factored action heads (~863k actor params, the 2026-08-13
user-approved 6.2x scale-up) plus the adapool_v1 critic (2026-08-19,
2026-08-19 critic redesign): a body-token-derived attention-pool query
over the resolved token stream + a 3-layer value MLP, ~231k critic
params, ~1.09M total. The critic reads; it is never read.

Each entity token carries its z position along the approach axis —
with constant note speed that IS the temporal ordering, expressed
spatially (time_until = z / |vz|), so no separate positional encoding
is needed. Readout = the two saber tokens' output embeddings (a
CLS-token-style readout, one per hand) plus the body token.

No action masks exist in this env: every action is always legal, so the
logprob is the unconditional sum over the 29 factored heads (env v10
full-body gear control, SPEC-v10 §1.4 joint order; pure independent
factorization).
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import fp16_linear as _fl
from fp16_linear import (fp16_linear, fp16_linear_nobias, fp16_linear_relu,
                         fp16_relu_chain)
from fused_ln import FusedLayerNorm

# (The round-4 fused single-head attention kernel was DELETED
# 2026-08-16: it only ever ran at n_heads == 1 and d == 64, both
# impossible since the 6.2x scale-up — see PERF_AUDIT_2026-08-16.md.)
# The fp16 stream runs the 2026-08-15 ladder WINNERS unconditionally
# (isolated-pod A/B, numerics-gated): SDPA attention, packed q/k/v
# GEMMs, the packed head GEMM, the bias_relu / cublasLt-epilogue
# Linear+ReLU sites and the ffn.2 bias fold into the next block's fused
# residual+LN pass. sdpa+qkv+heads measured 49.2 -> 40.3 ms/update
# (-18%); r1 (rollout/update consistency) is EXACTLY 0 because both
# phases run the same SDPA kernels. Numerics class: fp16 last-rounding
# (worst 5.9e-3 rel on grad probes). The env-var bisect hatches that
# reinstated the pre-ladder kernels were removed 2026-09-22 (last used
# 2026-09-02); the fp32 record path is the reference and stays bitwise
# stable regardless.
# Since 2026-09-27 every fp16 Linear-ReLU-...-Linear chain with an
# 8-aligned input (block FFNs, note encoder, SimBa trunk, critic MLP)
# runs its ReLUs in cuBLASLt epilogues (fp16_linear.fp16_relu_chain:
# RELU_AUX_BIAS forward, DRELU on the next Linear's dgrad) — no
# standalone relu_bwd pass. And at the production attention shape
# (29 keys, d 192, 4 heads) every attention site — the blocks' packed
# self-attention, the final block's lead queries, the critic's pool —
# runs the fused T = 29 kernel (fp16_linear.attn29_*, csrc/
# small_attn.cu) in BOTH phases instead of SDPA; SDPA remains for any
# other shape.


def _lin(m: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Linear that follows the stream dtype: fp16 activations route to
    the fp16-storage path (half addmm forward, GemmEx fp32-accumulated
    weight grads against the fp32 masters); fp32 runs the module as-is."""
    if x.dtype is torch.float16:
        return fp16_linear(x, m.weight, m.bias)
    return m(x)


def _seq(seq: nn.Sequential, x: torch.Tensor, fold_tail: bool = False):
    """Run a Sequential on the current stream dtype.

    fold_tail=True returns (out, tail_bias): on the fp16 path the FINAL
    Linear runs BIAS-FREE and its bias comes back for the consumer to
    fold into its own fused pass (round-4 #2 — the sm_61 addmm bias
    epilogue is 33-59% of the GEMM). tail_bias is None when the fold
    does not apply (fp32 record path, missing extension, env toggle),
    in which case out already carries the bias as before."""
    ms = list(seq)
    if fold_tail:
        tail = ms[-1]
        if (x.dtype is torch.float16 and isinstance(tail, nn.Linear)
                and _fl.ext_ok()):
            # whole Sequential is one Linear-ReLU-...-Linear chain (the
            # block FFN): ReLUs in the GEMM epilogues, tail bias-free
            chain = _relu_chain_at(ms, 0)
            if chain is not None and len(chain) * 2 - 1 == len(ms):
                out = fp16_relu_chain(x, chain, tail_bias=False)
                if out is not None:
                    return out, tail.bias
            return (fp16_linear_nobias(_run(ms[:-1], x), tail.weight,
                                       tail.bias),
                    tail.bias)
        return _run(ms, x), None
    return _run(ms, x)


def _relu_chain_at(ms: list, i: int):
    """The maximal Linear (ReLU Linear)+ run starting at ms[i], as the
    list of its Linears, or None when ms[i] does not start one."""
    chain = []
    while (i < len(ms) and isinstance(ms[i], nn.Linear)):
        chain.append(ms[i])
        if (i + 2 < len(ms) and isinstance(ms[i + 1], nn.ReLU)
                and isinstance(ms[i + 2], nn.Linear)):
            i += 2
        else:
            break
    return chain if len(chain) >= 2 else None


def _run(ms: list, x: torch.Tensor) -> torch.Tensor:
    i = 0
    while i < len(ms):
        m = ms[i]
        if isinstance(m, nn.Linear):
            # fp16 Linear-ReLU-...-Linear chains: every ReLU in a GEMM
            # epilogue (fp16_linear.fp16_relu_chain, 2026-09-27)
            if x.dtype is torch.float16 and _fl.ext_ok():
                chain = _relu_chain_at(ms, i)
                if chain is not None:
                    out = fp16_relu_chain(x, chain)
                    if out is not None:
                        x = out
                        i += 2 * len(chain) - 1
                        continue
            # fp16 Linear+ReLU pairs take the bias_relu epilogue path
            # (bias-free mm + in-place relu(y + b_fp32)) — round-4 #2
            if (x.dtype is torch.float16 and i + 1 < len(ms)
                    and isinstance(ms[i + 1], nn.ReLU)
                    and _fl.ext_ok()):
                x = fp16_linear_relu(x, m.weight, m.bias)
                i += 2
                continue
            x = _lin(m, x)
        else:
            x = m(x)
        i += 1
    return x

# Observation layout (contract; mirrors beat_arena_env/constants.py,
# obs v17 = unilateral-contact spec §12: root pose/vel (15), q block
# (31, wrist rolls cos/sin), qd (29), two 10-lane derived hand blocks,
# head (3), contact bits (2), then the 16x18 ENTITY block + mask (env
# v17: notes AND walls share the slots, nearest-first by head distance;
# per-token = the v9 note lanes then [is_wall, width, height, length])
# — plus the env v12 (spec §7) 4-lane per-foot ground-contact block at
# 558..562 (BODY token) and the env v13 (spec §8) 18-lane body-point
# block at 562..580: elbows to their saber tokens, knees + feet to the
# BODY token. Window 16 -> 24 in env v18 (eviction study); hand blocks
# 10 -> 11 lanes in env v19 (remaining follow-through credit),
# OBS_DIM 578 -> 580, offsets past the hands +2. Env v24: 18-lane
# body-point VELOCITY block at 580..598, routed like the positions
# (elbow vels to their saber tokens, knee + foot vels to BODY),
# OBS_DIM 580 -> 598. Env v45 (2026-09-15, user order; replaces the
# same-day v44 block): two 5-lane per-hand PREVIOUS-NOTE blocks
# appended at 598..608 — for each hand the last resolved note OF THAT
# HAND'S COLOUR regardless of who cut it: head-relative x/y, cut
# direction cos/sin, valid — routed into the saber tokens, OBS_DIM
# 598 -> 608, SABER_FEAT 32 -> 37. Env v46 (2026-09-16, user design):
# 16 torque-limit SCALE lanes appended at 608..624 — one per axis
# (spine 3, arm 7 shared L/R, leg 6 shared L/R), (s − 1)/0.25; the 7
# arm lanes go into BOTH saber tokens (37 -> 44), spine + leg lanes
# into the body token (78 -> 87). OBS_DIM 608 -> 624. Env v49
# (2026-09-30, review item I07): 6 blade-TIP velocity lanes appended at
# 624..630 ([L xyz, R xyz] / V_NORM, the judge's tip_move / DT); each
# hand's 3 go LAST into its saber token (44 -> 47). OBS_DIM 624 -> 630.
OBS_DIM = 630
N_NOTE_SLOTS = 24
NOTE_FEAT = 18
OFF_NOTES = 102
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT  # 534
# Token layout (SPEC-v10 §5 "informative", finalized here at P4):
# [saber_L, saber_R, body, note_0..23] = 27 tokens. Each saber token =
# that hand's q lanes (8, incl. the cos/sin roll pair) + qd lanes (7) +
# derived block (11, env v19: + remaining follow-through credit) = 26
# features + the 2-lane hand one-hot; the ONE body token = root
# pose/vel block (15) + spine/leg q (15) + spine/leg qd (15) + head (3)
# + contact bits (2) + ground-contact lanes (4, env v12) + knee/foot
# positions (12, env v13) + knee/foot velocities (12, env v24) = 78
# features. Each saber token gains its arm's elbow position (3, env
# v13) and elbow velocity (3, env v24): 26 -> 32; and its colour's
# previous-note block (5, env v45): 32 -> 37, appended LAST so the
# note-geo lane constants (_S_WPOS/_S_WVEL/_S_BLADE) stay valid; then
# the 7 arm tau-scale lanes (env v46): 37 -> 44; then that hand's
# blade-tip velocity (3, env v49): 44 -> 47 (_S_TIPV).
SABER_FEAT = 47
PREV_FEAT = 5
TAU_FEAT = 16
TIP_FEAT = 3            # per hand (env v49)
BODY_FEAT = 87
# derived from the window size so a future slot change can't skew them
_GI = OFF_NOTE_MASK + N_NOTE_SLOTS      # ground_imp block (4)
_BP = _GI + 4                           # body_pts block (18)
_BPV = _BP + 18                         # body_pt_vel block (18, env v24)
_PREV_L = _BPV + 18                     # prev-note L block (5, env v45)
_PREV_R = _PREV_L + PREV_FEAT           # prev-note R block (5, env v45)
_TAU = _PREV_R + PREV_FEAT              # tau-scale lanes (16, env v46)
_TIP = _TAU + TAU_FEAT                  # blade-tip velocity [L, R] (6, env v49)
assert _TIP + 2 * TIP_FEAT == OBS_DIM
# env v46 routing: arm axes (7) to both saber tokens; spine (3) + leg (6)
# axes to the body token — all appended LAST in their tokens
_TAU_ARM = tuple(range(_TAU + 3, _TAU + 10))
_TAU_BODY = tuple(range(_TAU, _TAU + 3)) + tuple(range(_TAU + 10, _TAU + 16))
_SABER_L_IDX = (tuple(range(18, 26)) + tuple(range(49, 56))
                + tuple(range(75, 86)) + tuple(range(_BP, _BP + 3))
                + tuple(range(_BPV, _BPV + 3))
                + tuple(range(_PREV_L, _PREV_L + PREV_FEAT))
                + _TAU_ARM + tuple(range(_TIP, _TIP + TIP_FEAT)))
_SABER_R_IDX = (tuple(range(26, 34)) + tuple(range(56, 63))
                + tuple(range(86, 97)) + tuple(range(_BP + 3, _BP + 6))
                + tuple(range(_BPV + 3, _BPV + 6))
                + tuple(range(_PREV_R, _PREV_R + PREV_FEAT))
                + _TAU_ARM + tuple(range(_TIP + TIP_FEAT, _TIP + 2 * TIP_FEAT)))
_BODY_IDX = (tuple(range(0, 18)) + tuple(range(34, 49))
             + tuple(range(63, 75)) + tuple(range(97, 102))
             + tuple(range(_GI, _GI + 4))
             + tuple(range(_BP + 6, _BP + 18))
             + tuple(range(_BPV + 6, _BPV + 18))
             + _TAU_BODY)
assert len(_SABER_L_IDX) == len(_SABER_R_IDX) == SABER_FEAT
assert len(_BODY_IDX) == BODY_FEAT

# --- 3-token body (2026-09-22, user order; the ONLY layout since the
# same day — the single body token was removed): the 87 body lanes
# (_BODY_IDX, kept as the coverage reference) as [torso, leg_L, leg_R],
# mirroring the two saber tokens
# (a shared leg encoder + a 2-lane side one-hot, like saber_mlp + the
# hand one-hot). Lead tokens become [saber_L, saber_R, torso, leg_L,
# leg_R] (5) + 24 notes = 29; the torso sits at index 2, where the
# single body token used to be, so the critic's body-derived query
# reads the torso. Arch marker "body_tokens" = 3; load_policy refuses
# checkpoints without it.
#   torso  root pose/vel (15) + spine q (3) + spine qd (3) + head (3)
#          + contact bits (2) + spine tau-scale lanes (3)        = 29
#   leg_s  hip/knee/ankle q (6) + qd (6) + that foot's ground-contact
#          lanes [heel, toe] (2) + knee + foot position (6) + knee +
#          foot velocity (6) + the 6 leg tau-scale axes (shared L/R,
#          routed to BOTH leg tokens like the arm axes to both sabers)
#                                                                  = 32
# Lane bookkeeping: q lanes 15..45 = spine 15..17, arm_L 18..25 (8: the
# wrist roll is cos/sin), arm_R 26..33, leg_L 34..39, leg_R 40..45; qd
# lanes 46..74 = spine 46..48, arm_L 49..55, arm_R 56..62, leg_L 63..68,
# leg_R 69..74. body_pts / body_pt_vel order: l_elbow, r_elbow, l_knee,
# r_knee, l_foot, r_foot (3 each). ground_imp = [L_heel, L_toe, R_heel,
# R_toe].
_TAU_SPINE = tuple(range(_TAU, _TAU + 3))
_TAU_LEG = tuple(range(_TAU + 10, _TAU + 16))
_TORSO_IDX = (tuple(range(0, 18)) + tuple(range(46, 49))
              + tuple(range(97, 102)) + _TAU_SPINE)
_LEG_L_IDX = (tuple(range(34, 40)) + tuple(range(63, 69))
              + (_GI, _GI + 1)
              + tuple(range(_BP + 6, _BP + 9)) + tuple(range(_BP + 12, _BP + 15))
              + tuple(range(_BPV + 6, _BPV + 9)) + tuple(range(_BPV + 12, _BPV + 15))
              + _TAU_LEG)
_LEG_R_IDX = (tuple(range(40, 46)) + tuple(range(69, 75))
              + (_GI + 2, _GI + 3)
              + tuple(range(_BP + 9, _BP + 12)) + tuple(range(_BP + 15, _BP + 18))
              + tuple(range(_BPV + 9, _BPV + 12)) + tuple(range(_BPV + 15, _BPV + 18))
              + _TAU_LEG)
TORSO_FEAT = 29
LEG_FEAT = 32
assert len(_TORSO_IDX) == TORSO_FEAT and len(_LEG_L_IDX) == len(_LEG_R_IDX) == LEG_FEAT
# the three tokens together see exactly the single body token's lanes
# (the 6 leg tau axes twice), and overlap only on those axes
assert set(_TORSO_IDX) | set(_LEG_L_IDX) | set(_LEG_R_IDX) == set(_BODY_IDX)
assert set(_TORSO_IDX).isdisjoint(_LEG_L_IDX) and set(_TORSO_IDX).isdisjoint(_LEG_R_IDX)
assert set(_LEG_L_IDX) & set(_LEG_R_IDX) == set(_TAU_LEG)
_BODY3_IDX = _TORSO_IDX + _LEG_L_IDX + _LEG_R_IDX

# --- hand-relative note geometry (2026-09-14, research/2026-09-14-
# architecture-brainstorm.md #1; user order) ---------------------------
# Per (hand, note) pair, computed from lanes the obs ALREADY carries and
# appended to the 18 note lanes before note_mlp, so the wrist->note
# relation is an input instead of something attention has to assemble:
#   dp (3)   wrist->note offset: x, y in METRES / 2 (both head-relative
#            in the obs since env v38 — same frame, same normalizers), z
#            (world z on both sides) as asinh(m / 8)
#   |dp| (1) asinh(m / 8)
#   t_plane (1)  (note z − z_off) / speed = seconds until the note
#            reaches the plane it is judged at (z_off = head_z − PLANE_Z0
#            = obs lane 99, lib.rs step_inner; B12 2026-09-30 — it was
#            z / speed, the time to world z = 0), clipped to [-0.5, 3], / 3
#   v.d (1)  unit wrist-velocity . required cut direction (cos_a, sin_a, 0)
#   b.d (1)  blade direction . required cut direction
# = 7 per hand -> 14 lanes; the dot lanes are masked to 0 on wall rows
# (walls have no cut direction). Only note_mlp.0.weight widens.
# I07 (2026-09-30, env v49; review item I07 — the geometry the game
# actually SCORES): 4 more per hand, appended AFTER the 14 as one block
# [hand L (4), hand R (4)] so a pre-I07 note_mlp's 32 columns keep their
# indices (load_policy zero-fills the 8 new ones):
#   along (1)  dp·b − HANDLE_LEN, asinh(m / 2): where along the blade
#              the note sits (the cutting span is [0, SABER_LEN])
#   perp (1)   |dp − (dp·b) b|, asinh(m / 2): distance from the note to
#              the blade LINE
#   cut (1)    dp·n̂, n̂ = (b × v_tip) / (|b × v_tip| + 0.05 m/s),
#              asinh(m / 2): the SIGNED distance to the plane the
#              blade sweeps — the judge's accuracy d_min = |cw·(b ×
#              tip_move)| / |b × tip_move| (lib.rs step_inner), signed
#              positive on the b × v_tip side (pinned in lib.rs
#              cut_plane_signed_distance_matches_the_judge); soft-
#              normalised so a still blade (no plane) reads ~0
#   vt.d (1)   v_tip · cut direction / V_NORM: the blade-TIP velocity
#              (obs env v49) along the required direction — the direction
#              check reads the contact point's velocity, not the wrist's;
#              masked on walls like the other dots
# = 11 per hand -> 22 lanes (note_mlp input 18 -> 40). The arch marker
# "blade_geo" = 1 records the I07 layout (these lanes + obs 630).
NOTE_GEO_FEAT_V55 = 14
NOTE_GEO_FEAT = 22
# obs normalizers, pinned against beat_arena_env.constants in
# tests/test_policy.py (the obs is written by the Rust/CUDA sims)
_X_NORM, _Y_NORM, _Z_SCALE, _V_NORM = 1.0, 0.75, 8.0, 6.0
_HANDLE_LEN = 0.16       # constants.HANDLE_LEN (env v42 hilt grip)
_OBS_HEAD_Z = 99         # obs lane (head_z − PLANE_Z0) / X_NORM = z_off
# saber_feat lane order: q(8) qd(7) then the 11-lane hand block
# [wrist pos(3), wrist vel(3), blade dir(3), meter, post_rem], ... and
# the blade-tip velocity LAST (env v49)
_S_WPOS, _S_WVEL, _S_BLADE = 15, 18, 21
_S_TIPV = SABER_FEAT - TIP_FEAT          # 44


def _note_geo(notes: torch.Tensor, sabers: torch.Tensor,
              z_asinh: float, flux: bool = False, *,
              z_off: torch.Tensor) -> torch.Tensor:
    """(B, 24, 18) fp32 note lanes + (B, 2, 47) fp32 saber lanes
    (SABER_FEAT) + (B,) head z offset z_off = obs lane 99 (metres) ->
    (B, 24, 22) geometry lanes (fp32; the caller casts with the notes):
    [hand L v55 (7), hand R v55 (7), hand L I07 (4), hand R I07 (4)].

    Every lane O(1) (user 2026-09-21, v55; the only layout since
    2026-09-22 — the raw-metre lanes of earlier checkpoints were
    removed): z decoded from the lane's true encoding c sinh(lane) (c =
    z_asinh, the policy's recorded --z-asinh), the per-hand offset's z and
    the distance compressed as asinh(m / 8) (the lane-2 shape), x/y
    offsets in half-metres, seconds-to-plane / 3; the I07 blade-frame
    lengths as asinh(m / 2) (half-metres near the blade, like the x/y
    offsets; < 8 out to 2 km). Motivation: [gnorm-lanes] on
    v52-v54 put the note encoder's excess gradient in its second layer,
    fed by far tokens' large hidden activations from raw-metre lanes.

    Flux (SPEC.md §Flux obs): the hand block holds the PALM, lanes 6/7
    are a bar's axis (so vdot/bdot read palm-velocity / hand-axis · bar
    axis), and a bar's target point is its SWEET SPOT (lanes 12/13)
    rather than its centre — the per-hand offsets aim the palms at
    circles and at the point the Flux must cross. The tip-velocity obs
    lanes are zero there, so the cut / vt.d lanes read 0."""
    assert z_asinh > 0, f"the obs z encoding C must be > 0, got {z_asinh}"
    z_m = torch.sinh(notes[..., 2]) * float(z_asinh)
    tx, ty = notes[..., 0], notes[..., 1]
    if flux:
        is_bar = notes[..., 9] > 0.5
        tx = torch.where(is_bar, notes[..., 12], tx)
        ty = torch.where(is_bar, notes[..., 13], ty)
    npos = torch.stack([tx * _X_NORM, ty * _Y_NORM,
                        z_m], -1)                                # (B,24,3) m
    cutd = torch.stack([notes[..., 6], notes[..., 7],
                        torch.zeros_like(notes[..., 6])], -1)   # (B,24,3)
    not_wall = 1.0 - notes[..., 14:15]                          # (B,24,1)
    # seconds to the judged plane: (z − z_off) / speed with speed =
    # -lane3 * Z_SCALE (B12: the note is cut at z_off, not world z = 0)
    speed = (-notes[..., 3:4] * _Z_SCALE).clamp_min(1e-3)
    zrel = z_m - (z_off.to(z_m.dtype) * _X_NORM)[:, None]
    tpl = (zrel[..., None] / speed).clamp(-0.5, 3.0)
    tpl = tpl / 3.0
    feats, blade_feats = [], []
    for hnd in range(2):
        w = sabers[:, hnd]                                       # (B,47)
        wpos = torch.stack([w[:, _S_WPOS] * _X_NORM,
                            w[:, _S_WPOS + 1] * _Y_NORM,
                            w[:, _S_WPOS + 2]], -1)              # (B,3) m
        dpm = npos - wpos[:, None]                               # (B,24,3) m
        dp = dpm * 0.5
        dist = dp.norm(dim=-1, keepdim=True)
        # z offset and distance: asinh(m / 8) keeps the near field
        # linear (2 dp units = 1 m) and bounds far tokens (~3 at
        # 100 m, ~7 at 2 km) instead of raw half-metres
        dp = torch.cat([dp[..., :2], torch.asinh(dp[..., 2:3] * 2.0 / 8.0)], -1)
        dist = torch.asinh(dist * 2.0 / 8.0)
        v = w[:, _S_WVEL:_S_WVEL + 3] * _V_NORM                  # (B,3) m/s
        vhat = v / (v.norm(dim=-1, keepdim=True) + 0.05)
        vdot = (vhat[:, None] * cutd).sum(-1, keepdim=True) * not_wall
        b = w[:, _S_BLADE:_S_BLADE + 3]                          # (B,3) unit
        bdot = (b[:, None] * cutd).sum(-1, keepdim=True) * not_wall
        feats += [dp, dist, tpl, vdot, bdot]
        # I07: the blade frame. along/perp decompose dpm on the blade
        # line; cut = signed distance to the swept plane
        ab = (dpm * b[:, None]).sum(-1, keepdim=True)            # (B,24,1) m
        along = torch.asinh((ab - _HANDLE_LEN) * 0.5)
        perp = torch.asinh(
            (dpm - ab * b[:, None]).norm(dim=-1, keepdim=True) * 0.5)
        vt = w[:, _S_TIPV:_S_TIPV + 3]                           # (B,3) /V_NORM
        pn = torch.linalg.cross(b, vt * _V_NORM)                 # (B,3) m/s
        nhat = pn / (pn.norm(dim=-1, keepdim=True) + 0.05)
        cut = torch.asinh((dpm * nhat[:, None]).sum(-1, keepdim=True) * 0.5)
        vtdot = (vt[:, None] * cutd).sum(-1, keepdim=True) * not_wall
        blade_feats += [along, perp, cut, vtdot]
    return torch.cat(feats + blade_feats, -1)
# env v10: 29 factored joint heads (SPEC-v10 §1.4 actuated order —
# spine 3, arm_l 7, arm_r 7, leg_l 6, leg_r 6); K = n_gears options
# per head (env v8 gear tables). This module constant is the K=3
# default; a policy's real sizes live in policy.head_sizes.
N_ACTION_HEADS = 29
HEAD_SIZES = (3,) * N_ACTION_HEADS
# Head -> limb grouping for diagnostics (2026-09-02). The network has
# NO explicit grouping — the 29 heads are one packed (145, 256) GEMM
# and head h is simply action lane h — so this is DERIVED from the
# actuated order above, which is the action-index contract shared with
# the env (beat_arena_env/constants.py JOINT_NAMES, A_SPINE=0,
# A_L_ARM=3, A_R_ARM=10, A_L_LEG=17, A_R_LEG=23). Duplicated rather
# than imported to keep policy.py a leaf module; the duplication is
# pinned against constants.py by tests/test_head_entropy.py.
HEAD_GROUPS = (("spine", 0, 3), ("armL", 3, 10), ("armR", 10, 17),
               ("legL", 17, 23), ("legR", 23, 29))

# Additive attention-mask magnitude. 2^14: exactly representable in fp16
# (the fp16-storage path needs the bias to survive an fp16 round trip;
# 1e9 is inf in fp16), cannot overflow the score epilogue (|score| < ~30,
# fp16 max 65504), and lands in expf's exact-zero regime after row-max
# subtraction just like 1e9 did — masked softmax weights are bitwise 0.0
# under either constant.
MASK_BIAS = 16384.0

# Logit of a head's lanes past its gear count (per-joint gear counts,
# 2026-10-04), the fp32 fill the head outputs are copied into: finite (so
# the entropy term p*log p is 0 * -1e9 = 0, not 0 * -inf = nan), and
# exp() of it underflows to exactly 0 — the lane has no probability.
MASKED_LOGIT = -1e9


# Block.forward's delta_bias sentinel: the previous block's fused FFN
# kernel (fp16_linear.fused_ffn, --fused-ffn) already ran THIS block's
# residual add + ln_attn — x arrives resolved and delta IS ln_attn(x).
_LN_DONE = object()


def _mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


@torch.no_grad()
def apply_depth_init(policy):
    """Depth-safe init for a FRESH policy (train_batch --depth-init,
    architecture review 2026-10-01). Every write into a residual stream
    (each block's attention out_proj and FFN second layer, the SimBa
    trunk's fc2) is scaled by 1/sqrt(2 * n_blocks) with zero bias, so a
    deep pre-LN stack starts near identity instead of growing the
    stream with depth (GPT-2's residual scaling); the 29 action heads
    are scaled by 0.01 with zero bias, so the initial policy is ~uniform
    over the gears. Returns a one-line summary."""
    n = len(policy.blocks)
    f = (2.0 * n) ** -0.5
    layers = []
    for blk in policy.blocks:
        if blk.out_proj is not None:
            layers.append(blk.out_proj)
        layers.append(blk.ffn[-1])
    # the SimBa trunk's residual write; a SimBaV2 trunk (--trunk-arch
    # simbav2) has no fc2 and keeps its unit-norm rows (projected)
    if getattr(policy, "trunk_fc2", None) is not None:
        layers.append(policy.trunk_fc2)
    for lin in layers:
        lin.weight.mul_(f)
        lin.bias.zero_()
    for h in policy.heads:
        h.weight.mul_(0.01)
        h.bias.zero_()
    return (f"{len(layers)} residual writes x{f:.4f} (1/sqrt(2*{n})), "
            f"{len(policy.heads)} heads x0.01, biases 0")


class Block(nn.Module):
    """Full pre-LN transformer block.

    n_heads == 1 keeps the legacy layout (v added straight into the
    residual stream, no output projection) so pre-2026-07-21 checkpoints
    load unchanged. n_heads > 1 uses standard multi-head attention with
    a concat-heads output projection.
    """

    def __init__(self, d: int, n_heads: int = 1, ffn_mult: int = 2,
                 value_residual: bool = False):
        super().__init__()
        assert d % n_heads == 0, f"embed {d} not divisible by {n_heads} heads"
        # FusedLayerNorm == nn.LayerNorm in state_dict/params; on CUDA
        # fp32 it runs the warp-per-row kernel (csrc/layer_norm.cu)
        self.ln_attn = FusedLayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d) if n_heads > 1 else None
        self.ln_ffn = FusedLayerNorm(d)
        # ffn_mult (v17 scale-up): hidden width ratio. The historical
        # layout is 2x (the 138k-param arch and the fused-kernel
        # contract); the scaled arch uses the standard 4x.
        self.ffn = _mlp([d, ffn_mult * d, d])
        self.d = d
        self.n_heads = n_heads
        self.dh = d // n_heads
        # --value-residual (ResFormer, arXiv 2410.17897; arch
        # "value_residual"): every block after the first mixes block 0's
        # attention values into its own, v <- a v + b v_0, with learnable
        # scalars initialised to (a, b) = (1, 0) so training starts at the
        # baseline. Registered LAST in the block (the diag groups slice the
        # flat grad buffer at each block's ln_attn.weight) and only when
        # on, so an off policy's parameters() order is untouched.
        if value_residual:
            self.vres_a = nn.Parameter(torch.ones(1))
            self.vres_b = nn.Parameter(torch.zeros(1))
        else:
            self.vres_a = self.vres_b = None

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor,
                n_queries: int = 0, attn_bias: torch.Tensor | None = None,
                delta: torch.Tensor | None = None,
                delta_bias: torch.Tensor | None = None,
                want_resolved: bool = False,
                next_ln: nn.Module | None = None,
                v0: torch.Tensor | None = None, want_v: bool = False):
        # x: (B, T, d); key_mask: (B, T), 1 = attendable
        #
        # n_queries > 0 computes the block's output for only the first
        # n_queries tokens (keys/values still span all T). Because the
        # attention mask is applied on the KEY dim only, token i's output
        # never depends on any other token's q/FFN path — so this is
        # EXACTLY the full block restricted to those rows, not an
        # approximation. Used by the readout (tokens[:, :2]) to skip the
        # 16 note tokens' final-block compute (~29% of forward MACs).
        #
        # attn_bias: the additive key mask (key_mask - 1) * MASK_BIAS, (B, 1, T)
        # — block-invariant, so the caller hoists it out of the loop.
        # delta: the previous block's un-added FFN output; its residual
        # add rides this block's ln_attn pass (FusedLayerNorm.forward_add)
        # instead of being a separate full-size kernel. Both fusions are
        # bit-exact. Blocks called with n_queries == 0 return
        # (x_without_ffn, ffn_out, ffn_bias) for the next block's delta;
        # the final block (n_queries > 0) returns the resolved rows.
        # delta_bias: the producer's ffn.2 bias when that Linear ran
        # bias-free (fp16 only) — it is added to delta inside the fused
        # ln_attn pass. NEAR-TIE numerics (the bias rounds in the LN
        # pass's fp32 sum instead of the GEMM epilogue); the fp32 record
        # path never sets it (see _seq's fold_tail).
        # want_resolved (final block only, 2026-09-02 Tier 1.1): ALSO
        # return the full (B, T, d) resolved residual stream entering
        # this block — the x this pass writes (delta + delta_bias
        # folded), BEFORE the n_queries slice — for the adapool critic,
        # which pools over exactly the token embeddings this block's
        # own k/v see. Returning it here is free (the tensor exists);
        # the caller used to RECOMPUTE it as tokens + delta + delta_b,
        # which promoted the fp16 stream to fp32 (fp32 bias Parameter).
        # next_ln (non-final blocks): the NEXT block's ln_attn. Under
        # --fused-ffn (fp16_linear.fused_ffn, 2026-09-30) this block's FFN,
        # the residual add and that LN run as ONE kernel; the block then
        # returns (x_resolved, ln_attn(x_resolved), _LN_DONE) and the next
        # block skips its forward_add.
        # v0 (--value-residual, blocks 1..L-1): block 0's attention values
        # (B, T, d), a view of its projection output so the gradient
        # reaches it; mixed into this block's v as a v + b v0
        # (fp16_linear.vres_mix: fp32 arithmetic, rounded once to the
        # stream dtype — bitwise v at (a, b) = (1, 0)), out of place,
        # BEFORE the attention. On the fused T = 29 kernels the mix runs
        # inside fp16_linear.attn29_packed_vres / attn29_qkv_vres (mixed v
        # as its own buffer, q/k still views of the packed projection, ONE
        # packed gradient back); elsewhere the packed q|k|v / k|v buffer is
        # rebuilt with the mixed v.
        # want_v (block 0, non-final): also return its v as a 4th element.
        B, T, _ = x.shape
        if delta_bias is _LN_DONE:
            h = delta
        elif delta is not None:
            x, h = self.ln_attn.forward_add(x, delta, delta_bias)
        else:
            h = self.ln_attn(x)
        qkv = kv = None
        if h.dtype is torch.float16:
            # Tier 1.4: q/k/v share input h -> one GEMM against the cat
            # replicas. The final block's q runs on the SLICED h, so
            # only k/v pack there — their cat is the qkv group's tail
            # rows (layout [wq wk wv], flatten_fp16_replicas groups=).
            wh, bh = _fl.packed_halves(
                [self.q_proj, self.k_proj, self.v_proj])
            if n_queries:
                kv = _fl.fp16_packed_linear(h, wh[self.d:], bh[self.d:],
                                            [self.k_proj, self.v_proj])
                k, v = kv.split(self.d, dim=-1)
            else:
                qkv = _fl.fp16_packed_linear(
                    h, wh, bh, [self.q_proj, self.k_proj, self.v_proj])
                q, k, v = qkv.split(self.d, dim=-1)
        else:
            k, v = _lin(self.k_proj, h), _lin(self.v_proj, h)
        v_own = v
        # value residual on the fused T = 29 kernels: the mix runs inside
        # fp16_linear.attn29_*_vres (packed gradient, no rebuilt buffer);
        # every other path mixes here and rebuilds the packed buffer
        vres_fused = (v0 is not None and self.n_heads > 1
                      and (qkv is not None or kv is not None)
                      and _fl.attn29_ok(T, self.d, self.n_heads))
        if v0 is not None and not vres_fused:
            v = _VMix.apply(v, v0, self.vres_a, self.vres_b)
            if qkv is not None:
                qkv = torch.cat([qkv[..., :2 * self.d], v], dim=-1)
            elif kv is not None:
                kv = torch.cat([k, v], dim=-1)
        if n_queries:
            # x.contiguous() is load-bearing, not hygiene: a (B, T, d)
            # row slice is NOT contiguous, and FusedLayerNorm._eligible
            # requires it — without it the ln_ffn.forward_add below
            # silently took the 5-kernel fallback (2026-08-02).
            # Verified bit-identical on BOTH streams: the fused
            # ln_fwd_add computes its stats on the once-rounded a + b,
            # exactly like the fallback's add-then-LN.
            # h is deliberately left as a view: its only consumer is the
            # q projection, whose reshape pays the same copy anyway
            # (fp16), while an explicit .contiguous() flips fp32's
            # F.linear onto a different cuBLAS path and SHIFTS the
            # record path's bits (measured 1e-4 on the logits).
            x_full = x
            x = x[:, :n_queries].contiguous()
            h = h[:, :n_queries]
        if qkv is None:
            q = _lin(self.q_proj, h)
        Tq = q.shape[1]
        if attn_bias is None:
            attn_bias = (key_mask.unsqueeze(1) - 1.0) * MASK_BIAS  # (B, 1, T)
        if self.n_heads == 1:
            # scale + mask ride the GEMM epilogue: one kernel
            # instead of three (matmul, div, add). Bitwise-identical
            # here because d ** -0.5 is a power of two at d = 64/128
            # (exact multiply)
            scores = torch.baddbmm(attn_bias, q, k.transpose(1, 2),
                                   alpha=self.d ** -0.5)      # (B, Tq, T)
            att = torch.softmax(scores, dim=-1) @ v
        elif qkv is not None and _fl.attn29_ok(T, self.d, self.n_heads):
            # non-final fp16 block: the T = 29 fused attention kernel
            # (fp16_linear.attn29_packed, 2026-09-27) straight off the
            # packed projection -> (B, T, d) in the out-projection's
            # layout; gradient = ONE packed (B, T, 3d) tensor
            if vres_fused:
                att = _fl.attn29_packed_vres(qkv, v0, self.vres_a,
                                             self.vres_b, attn_bias)
            else:
                att = _fl.attn29_packed(qkv, attn_bias)
            att = _lin(self.out_proj, att)
        elif qkv is not None:
            # non-final fp16 block: fused attention straight off the
            # packed projection; the backward writes dq|dk|dv as ONE
            # packed gradient (no split-backward cat copy, 2026-09-26) —
            # bitwise identical to the F.sdpa call below
            # (fp16_linear.packed_sdpa)
            att = _fl.packed_sdpa(qkv, attn_bias.unsqueeze(1),
                                  self.n_heads)
            att = att.transpose(1, 2).reshape(B, Tq, self.d)
            att = _lin(self.out_proj, att)
        elif (q.dtype is torch.float16 and n_queries
                and _fl.attn29_ok(T, self.d, self.n_heads)):
            # final fp16 block: the lead-token queries over the packed
            # k|v (same T = 29 kernel as every other attention site)
            if vres_fused:
                att = _fl.attn29_qkv_vres(q, kv, v0, self.vres_a,
                                          self.vres_b, attn_bias)
            else:
                att = _fl.attn29_qkv(q, kv, attn_bias)
            att = _lin(self.out_proj, att)
        else:
            q = q.view(B, Tq, self.n_heads, self.dh).transpose(1, 2)
            k = k.view(B, T, self.n_heads, self.dh).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.dh).transpose(1, 2)
            if q.dtype is torch.float16:
                # Tier 1.3b: fused attention. The additive key mask is
                # legal here — no row is ever all-masked (the lead lanes
                # are always attendable), so no NaN softmax rows.
                att = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias.unsqueeze(1))
            else:
                # fp32 record path / CPU: the reference arithmetic
                scores = q @ k.transpose(-2, -1) / (self.dh ** 0.5)
                scores = scores + attn_bias.unsqueeze(1)         # (B,1,1,T)
                att = torch.softmax(scores, dim=-1) @ v          # (B,H,Tq,dh)
            att = att.transpose(1, 2).reshape(B, Tq, self.d)
            att = _lin(self.out_proj, att)
        x, h2 = self.ln_ffn.forward_add(x, att)
        if n_queries:
            # final block: nothing downstream fuses the residual add, so
            # ffn.2 keeps its own bias (fused: the kernel's residual-only
            # variant)
            if _fl.fused_ffn_ok(h2, self.ffn):
                out = _fl.fused_ffn(x, h2, self.ffn)
            else:
                out = x + _seq(self.ffn, h2)
            return (out, x_full) if want_resolved else out
        if next_ln is not None and _fl.fused_ffn_ok(h2, self.ffn, next_ln):
            # --fused-ffn: FFN + residual + the next block's ln_attn in one
            # kernel; the hidden never reaches memory in the rollout
            x, h_next = _fl.fused_ffn(x, h2, self.ffn, next_ln)
            if want_v:
                return x, h_next, _LN_DONE, v_own
            return x, h_next, _LN_DONE
        # non-final block: the consumer is the NEXT block's
        # ln_attn.forward_add, which can absorb ffn.2's bias
        ffn_out, ffn_bias = _seq(self.ffn, h2, fold_tail=True)
        if want_v:
            return x, ffn_out, ffn_bias, v_own
        return x, ffn_out, ffn_bias


class _VMix(torch.autograd.Function):
    """--value-residual mix a v + b v0 -> v's dtype: fp32 arithmetic,
    rounded once (bitwise v at (a, b) = (1, 0)). A Function rather than
    autograd ops so the backward saves only v and v0 — views of the
    blocks' q|k|v projections the attention already keeps — instead of
    two fp32 (B, T, d) upcasts per block (+23% peak memory measured with
    plain ops)."""

    @staticmethod
    def forward(ctx, v, v0, a, b):
        ctx.save_for_backward(v, v0, a, b)
        return _fl.vres_mix(v, v0, a, b)

    @staticmethod
    def backward(ctx, g):
        v, v0, a, b = ctx.saved_tensors
        gf = g.float()
        dv = (gf * a).to(v.dtype) if ctx.needs_input_grad[0] else None
        dv0 = (gf * b).to(v0.dtype) if ctx.needs_input_grad[1] else None
        da = (gf * v.float()).sum().view(1) if ctx.needs_input_grad[2] else None
        db = (gf * v0.float()).sum().view(1) if ctx.needs_input_grad[3] else None
        return dv, dv0, da, db


def _l2n(x: torch.Tensor) -> torch.Tensor:
    """Row l2 normalisation (SimBaV2's hypersphere projection)."""
    return F.normalize(x, dim=-1, eps=1e-6)


# readout-trunk layouts (arch "trunk_arch"): "simba" = residual + LN
# block (the production trunk); "simbav2" = the hyperspherical SimBaV2
# trunk (arXiv 2502.15280, --trunk-arch simbav2, NotePolicy._trunk_v2).
# Its scalers use the paper's parametrisation: a learnable vector p with
# s = p * (s_init / s_scale), p initialised to s_scale — s_init sets the
# value, s_scale the effective step size under Adam. Scalers: s_init =
# s_scale = sqrt(2 / width) (the reference config); the residual LERP
# alpha: init 1 / (L + 1) = 0.5 (L = 1 block), scale 1 / sqrt(trunk_dim).
TRUNK_ARCHS = ("simba", "simbav2")


def hl_target(v: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    """HL-Gauss target (Farebrother et al. 2024, "Stop Regressing", eq.
    for p_i): the mass of N(v, sigma^2) inside each histogram bin,
    truncated to the support [v_min, v_max] and renormalized by Z so
    every row sums to 1 (mass outside the support is dropped, as in the
    paper). The scalar target is first clamped into the support so a
    far-out return does not make Z underflow. v (B,), centers (K,)
    equally spaced -> (B, K)."""
    v = v.float()
    width = centers[1] - centers[0]
    edges = torch.cat([centers[:1] - 0.5 * width, centers + 0.5 * width])
    v = v.clamp(edges[0], edges[-1])
    z = (edges.unsqueeze(0) - v.unsqueeze(1)) / (sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))                           # (B, K+1)
    p = cdf[:, 1:] - cdf[:, :-1]
    return p / p.sum(-1, keepdim=True)


def hl_loss(logits: torch.Tensor, v_tgt: torch.Tensor, centers: torch.Tensor,
            sigma: float) -> torch.Tensor:
    """Cross-entropy between the critic's bin distribution and the
    HL-Gauss target; mean over the batch. logits (B, K) fp32."""
    tgt = hl_target(v_tgt, centers, sigma)
    return -(tgt * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


class NotePolicy(nn.Module):
    # v17 scaled defaults (2026-08-13, user-approved 6.2x): embed 128,
    # 4 attention heads (multi-entity reasoning — a saber can attend to
    # its next note AND an incoming wall separately), 3 blocks, 4x FFN,
    # trunk 256 -> ~861k params at gears 5 (was 138,642). The 138k arch
    # and its fused update kernels were retired with it (no conversion,
    # house pattern); training runs the eager fp16 path.
    # Recipe since 2026-09-15: the TRAINER defaults are embed 192 /
    # trunk 384 (train_batch.py); the ctor keeps the historical 128/256
    # widths. The readout trunk is always SimBa (the plain MLP trunk
    # was removed 2026-09-22).
    def __init__(self, embed: int = 128, trunk_dim: int = 256,
                 encoder: str = "transformer", n_blocks: int = 3,
                 n_heads: int = 4, n_gears: int = 3, ffn_mult: int = 4,
                 note_geo: bool = False,
                 value_bins: int = 101, value_vmax: float = 6.0,
                 value_sigma: float = 0.75,
                 trunk_hidden: int | None = None,
                 z_asinh: float = 8.0, game: str = "standard",
                 fp16_accum: str = "off", fused_ffn: str = "off",
                 emb_ln: bool = False, value_residual: bool = False,
                 final_ln: bool = False, trunk_arch: str = "simba",
                 head_gears: tuple | list | None = None):
        super().__init__()
        if trunk_arch not in TRUNK_ARCHS:
            raise ValueError(f"trunk_arch must be one of {TRUNK_ARCHS}, "
                             f"got {trunk_arch!r}")
        # game mode the policy plays (SPEC.md §Flux): same network,
        # different obs lane meanings — never cross-loaded
        if game not in ("standard", "flux"):
            raise ValueError(f"game must be 'standard' or 'flux', got {game!r}")
        # obs z lane encoding asinh(z / C), C = z_asinh metres (env v48;
        # the only encoding since 2026-09-22). The note-geo lanes decode
        # the note distance with it, so it is part of the forward and lives
        # in the arch record. Default = the env/trainer default C = 8.
        if not float(z_asinh) > 0:
            raise ValueError(f"z_asinh must be > 0 (metres), got {z_asinh}")
        if fp16_accum not in _fl.ACCUM_MODES:
            raise ValueError(f"fp16_accum must be one of {_fl.ACCUM_MODES}, "
                             f"got {fp16_accum!r}")
        if fused_ffn not in _fl.FFN_MODES:
            raise ValueError(f"fused_ffn must be one of {_fl.FFN_MODES}, "
                             f"got {fused_ffn!r}")
        assert encoder == "transformer", (
            f"unknown encoder: {encoder} (pool was removed 2026-07-20)")
        assert n_gears in (3, 5, 7), f"n_gears must be 3/5/7, got {n_gears}"
        assert value_bins > 1, (
            f"value_bins must be > 1 (HL-Gauss is the only critic head; "
            f"the scalar MSE head was removed 2026-09-22), got {value_bins}")
        self.encoder = encoder
        self.embed = embed
        self.n_heads = n_heads
        # n_gears = the env's BASE gear table (constants.gear_table); the
        # per-head action counts are head_gears (2026-10-04, per-joint
        # gear counts). None = THIS build's layout,
        # constants.joint_gear_counts(n_gears) — every head n_gears on
        # main (today's policy exactly), the branch's counts on a body
        # branch with extra gears. load_policy passes a checkpoint's own
        # record instead (absent = uniform) and checks it.
        self.n_gears = n_gears
        if head_gears is None:
            head_gears = build_head_gears(n_gears)
        head_gears = tuple(int(k) for k in head_gears)
        assert len(head_gears) == N_ACTION_HEADS and all(
            3 <= k <= 7 for k in head_gears), (
            f"head_gears must be {N_ACTION_HEADS} counts in [3, 7], got "
            f"{head_gears}")
        self.head_sizes = head_gears
        # stacked logits are (B, 29, k_max): a head with fewer ids gets
        # k_max outputs whose extra lanes are masked (_gear_mask)
        self.k_max = max(head_gears)
        # arch record: load_policy REFUSES checkpoints whose markers
        # don't match the current era (obs 426 / ffn_mult present).
        self.arch = {"embed": embed, "trunk_dim": trunk_dim,
                     "n_blocks": n_blocks, "n_heads": n_heads,
                     "n_gears": n_gears, "obs_dim": OBS_DIM,
                     "ffn_mult": ffn_mult, "critic": "adapool_v1",
                     "n_action_heads": N_ACTION_HEADS, "body_token": True,
                     # 2026-09-22 (user): the body is ALWAYS the three
                     # tokens [torso, leg_L, leg_R] (_TORSO_IDX /
                     # _LEG_*_IDX); the single 87-lane body token of every
                     # checkpoint before v58 is gone (no back-compat)
                     "body_tokens": 3,
                     "note_geo": bool(note_geo),
                     # 2026-09-21 (v55): note-geo lanes normalised to
                     # O(1); the only layout since 2026-09-22, still
                     # stamped because load_policy refuses a note_geo
                     # checkpoint without it (raw-metre lanes)
                     "note_geo_norm": True,
                     # 2026-09-30 (review item I07, env v49): obs 630
                     # with the blade-tip velocity lanes into the saber
                     # tokens (SABER_FEAT 47) and the 22-lane note geo
                     # (the I07 blade-frame block). Absent = a pre-I07
                     # checkpoint (obs 624, SABER_FEAT 44, 14 geo
                     # lanes), which load_policy UPGRADES in place:
                     # zero input columns for every new lane
                     # (upgrade_pre_i07_state_dict)
                     "blade_geo": 1,
                     "z_asinh": float(z_asinh),
                     # 2026-09-15 (brainstorm item 2): readout-trunk
                     # layout. "simba" = residual + LayerNorm block,
                     # below — the only layout since 2026-09-22 (the
                     # plain MLP trunk is gone); still stamped because
                     # load_policy refuses anything else.
                     # "simbav2" (2026-10-03, --trunk-arch simbav2): the
                     # hyperspherical SimBaV2 trunk (_trunk_v2)
                     "trunk_arch": trunk_arch,
                     "trunk_hidden": (int(trunk_hidden) if trunk_hidden
                                      else trunk_dim),
                     # HL-Gauss critic head (2026-09-19; the only head
                     # since 2026-09-22, the scalar MSE head is gone):
                     # K-bin histogram over the normalized return in
                     # [-vmax, vmax], target width sigma (in bin
                     # widths), value = expectation
                     "value_bins": int(value_bins),
                     "value_vmax": float(value_vmax),
                     "value_sigma": float(value_sigma),
                     "game": game,
                     # GEMM accumulate mode of the fp16 stream (issue #3,
                     # fp16_linear.fp16_accum): part of the forward the
                     # policy was TRAINED with, so every fp16 consumer
                     # (rollout, update, fp16 benches) runs the same one.
                     # The fp32 record path ignores it. Absent = "off".
                     "fp16_accum": fp16_accum,
                     # fused block-FFN kernel of the fp16 stream (review
                     # I13, fp16_linear.fused_ffn): near-tie numerics vs
                     # the unfused path, so recorded like fp16_accum.
                     # Ignored off the fp16 CUDA stream / at other widths.
                     # Absent = "off".
                     "fused_ffn": fused_ffn,
                     # architecture A/B options (2026-10-03; each off by
                     # default, absent = off): LayerNorm on the 29 input
                     # tokens (--emb-ln), ResFormer value residual
                     # (--value-residual), LayerNorm on the 5 lead
                     # outputs before the trunk (--final-ln). (--dwa,
                     # DenseFormer, lost the A/B and was removed
                     # 2026-10-04; load_policy refuses its checkpoints.)
                     "emb_ln": bool(emb_ln),
                     "value_residual": bool(value_residual),
                     "final_ln": bool(final_ln)}
        # per-joint gear counts (2026-10-04): recorded only when they
        # differ from n_gears everywhere, so a uniform policy's arch is
        # unchanged; load_policy rebuilds the heads from it
        if head_gears != (n_gears,) * N_ACTION_HEADS:
            self.arch["head_gears"] = list(head_gears)
        self.value_bins = int(value_bins)
        self.value_sigma_bins = float(value_sigma)
        centers = torch.linspace(-float(value_vmax), float(value_vmax),
                                 self.value_bins)
        self.register_buffer("value_centers", centers, persistent=False)
        # sigma in return units: sigma_bins x bin width
        self.value_sigma = float(value_sigma) * float(centers[1] - centers[0])
        self._value_logits = None
        self.note_geo = bool(note_geo)
        self.z_asinh = float(z_asinh)
        self.game = game
        self.note_mlp = _mlp([NOTE_FEAT + (NOTE_GEO_FEAT if note_geo else 0),
                              embed, embed])
        # saber token features: the 25 obs-v10 hand features + [is_left,
        # is_right]; the body token gets its own encoder
        self.saber_mlp = _mlp([SABER_FEAT + 2, embed, embed])
        self.body_tokens = 3
        # lead tokens [saber_L, saber_R, torso, leg_L, leg_R]; the notes follow
        self.n_lead = 5
        # torso gets its own encoder; the two legs share one (+ the 2-lane
        # side one-hot), like the sabers. Registered before the blocks so
        # the trainer's "enc" diag group covers them.
        self.torso_mlp = _mlp([TORSO_FEAT, embed, embed])
        self.leg_mlp = _mlp([LEG_FEAT + 2, embed, embed])
        # architecture A/B options (2026-10-03). Every new module is
        # created ONLY when its option is on (LayerNorms and the 1-D
        # scalars draw no RNG), so an all-off policy has today's
        # parameters, order and init stream exactly.
        self.value_residual = bool(value_residual)
        # --emb-ln: LayerNorm (learnable gain/bias) on the 29 input tokens
        # before block 0 — block 0 writes 15-27x the embeddings' RMS, so
        # the raw input barely survives in the stream. Registered before
        # the blocks: the trainer's "enc" diag group covers it.
        self.emb_ln = FusedLayerNorm(embed) if emb_ln else None
        self.blocks = nn.ModuleList(
            Block(embed, n_heads, ffn_mult,
                  value_residual=bool(value_residual) and i > 0)
            for i in range(n_blocks))
        # --final-ln: LayerNorm on the 5 lead tokens' final-block outputs
        # before trunk_proj (they enter it at RMS ~30 otherwise)
        self.final_ln = FusedLayerNorm(embed) if final_ln else None
        self.trunk_arch = trunk_arch
        # readout = the 5 lead tokens' output embeddings
        # SimBa-style readout trunk (2026-09-15, brainstorm item 2;
        # SimBa 2410.09754 / SimBaV2 2502.15280, Creus Castanyer
        # 2506.15544, Kumar 2010.14498): the old plain trunk was the
        # ONE stage with neither normalization nor a skip, and it
        # produces the feature h whose effective rank the [diag]
        # line tracks. Layout: projection -> one residual block
        # x + fc2(relu(fc1(LN1 x))) -> LN2 -> h. H = trunk_hidden
        # (default trunk_dim). Registered between the blocks and the
        # heads so the flat-grad diag groups (encoder -> blocks ->
        # trunk -> heads -> critic) keep their order; the trainer's
        # diag looks the trunk group up by name prefix.
        H = int(trunk_hidden) if trunk_hidden else trunk_dim
        if trunk_arch == "simba":
            self.trunk_proj = nn.Linear(self.n_lead * embed, trunk_dim)
            self.trunk_ln1 = FusedLayerNorm(trunk_dim)
            self.trunk_fc1 = nn.Linear(trunk_dim, H)
            self.trunk_fc2 = nn.Linear(H, trunk_dim)
            self.trunk_ln2 = FusedLayerNorm(trunk_dim)
        else:
            # SimBaV2 (arXiv 2502.15280), one block, hidden H:
            #   x = l2n(s_in * W_in z)
            #   y = l2n(W2 relu(s_1 * W1 x)); x = l2n(x + alpha (y - x))
            #   h = sqrt(trunk_dim) x   (RMS 1 per element, like LN2's)
            # W_in/W1/W2 rows are kept UNIT-NORM by project_trunk_weights
            # (here at init, and by the trainer after every optimizer
            # step). The paper's layers have no bias: these are nn.Linear
            # (the fp16 machinery — flat replicas, fp16_linear — wants a
            # bias tensor) with the bias zeroed and FROZEN
            # (requires_grad False; the trainer leaves frozen params out
            # of its optimizers). Orthogonal init, as the paper.
            D = trunk_dim
            self.trunk_in = nn.Linear(self.n_lead * embed, D)
            self.trunk_w1 = nn.Linear(D, H)
            self.trunk_w2 = nn.Linear(H, D)
            for lin in (self.trunk_in, self.trunk_w1, self.trunk_w2):
                nn.init.orthogonal_(lin.weight)
                nn.init.zeros_(lin.bias)
                lin.bias.requires_grad_(False)
            # scalers: p init s_scale, forward factor s_init / s_scale
            # (= 1 for these, kept explicit for the parametrisation)
            self._v2_c_in = 1.0
            self._v2_c_1 = 1.0
            self.trunk_s_in = nn.Parameter(
                torch.full((D,), math.sqrt(2.0 / D)))
            self.trunk_s_1 = nn.Parameter(
                torch.full((H,), math.sqrt(2.0 / H)))
            a_init, a_scale = 0.5, 1.0 / math.sqrt(D)
            self._v2_c_alpha = a_init / a_scale
            self.trunk_alpha = nn.Parameter(torch.full((D,), a_scale))
            self.project_trunk_weights()
        # 29 factored joint heads, K = n_gears options each (env v8);
        # with per-joint counts (head_gears) head j has its own n_j rows
        # (the packed GEMM is sum(n_j) wide) and its outputs are copied
        # into the (B, 29, k_max) logits over a MASKED_LOGIT fill: the
        # lanes past a head's count get exp() = 0 exactly (no
        # probability, no entropy, never sampled) and have no weights.
        # Padding every head to k_max rows instead made the packed
        # GEMM 29 x 7 = 203 wide, a width at which cuBLAS picks
        # different fp16-accumulate kernels for the rollout and the
        # update shapes (r1 2e-5 at init vs 6e-7; 149 and 145 are
        # bitwise consistent) — pinned by test_joint_gears on the pod.
        self.heads = nn.ModuleList(
            nn.Linear(trunk_dim, n) for n in self.head_sizes)
        if min(self.head_sizes) < self.k_max:
            valid = torch.tensor([[a < k for a in range(self.k_max)]
                                  for k in self.head_sizes])
            self.register_buffer("_gear_valid", valid, persistent=False)
            self.register_buffer(
                "_gear_idx", torch.tensor(
                    [j * self.k_max + a
                     for j, k in enumerate(self.head_sizes)
                     for a in range(k)], dtype=torch.long),
                persistent=False)
        else:
            self._gear_valid = None
            self._gear_idx = None
        # Critic (2026-08-19 redesign):
        # adapool_v1 = query-derived attention pooling (Brothers
        # 2506.09215: a state-DERIVED query wins over a free learned
        # CLS token in exactly our low-SNR entity regime) + value MLP:
        # the critic's query is a projection of the body token's
        # embedding entering the final block; it pools READ-ONLY over
        # all resolved tokens (nothing attends back), then a 3-layer MLP on
        # [pooled, trunk] predicts the NORMALIZED value (the trainer
        # owns the return-normalization stats). Registered AFTER the
        # heads so the critic occupies the tail of parameters() order
        # (the trainer's flat-grad diagnostics slice on that).
        # (the pre-redesign Linear(trunk,1) critic was removed 2026-09-22
        # — no loadable checkpoint carries it)
        self.critic_ln = FusedLayerNorm(embed)
        self.critic_q = nn.Linear(embed, embed)
        self.critic_k = nn.Linear(embed, embed)
        self.critic_v = nn.Linear(embed, embed)
        self.critic_out = nn.Linear(embed, embed)
        self.critic_mlp = _mlp([embed + trunk_dim, trunk_dim,
                                trunk_dim, self.value_bins])
        # trainer diagnostics: _diag_stash=True makes the next forward
        # keep its trunk output on self._diag_h (forward hooks can't
        # see it — the fp16 paths bypass module __call__). Plain
        # attrs; the graphed rollout captures with the flag False.
        self._diag_stash = False
        self._diag_h = None
        # per-head entropy of the last need_entropy=True forward, (B, 29)
        # (2026-09-02). Written unconditionally when entropy is asked
        # for — it is the same tensor the sum reduces, not extra work.
        self._ent_h = None
        # token-slice index buffers (device-tracked, not in state_dict)
        self.register_buffer(
            "_saber_idx", torch.tensor(_SABER_L_IDX + _SABER_R_IDX,
                                       dtype=torch.long), persistent=False)
        self.register_buffer(
            "_body_idx", torch.tensor(_BODY3_IDX, dtype=torch.long),
            persistent=False)
        # fp16-storage mode (2026-07-25): activations flow fp16 between
        # the obs split and the head output; GEMMs are fp16-in with fp32
        # cuBLAS compute; LN/softmax stats, log_softmax, sampling, and
        # all PPO math stay fp32. Weights stay fp32 masters (state_dict
        # unchanged). Plain attribute, NOT in state_dict.
        self.fp16 = False
        # accumulate mode of the fp16 stream's GEMMs (arch "fp16_accum")
        self.fp16_accum = fp16_accum
        # load_policy: "pre-i07" when the weights were upgraded from a
        # pre-I07 checkpoint (zero-filled new input columns)
        self.upgraded_from = None
        # fused block-FFN kernel mode of the fp16 stream (arch "fused_ffn")
        self.fused_ffn = fused_ffn

    def _split(self, obs: torch.Tensor):
        sabers = obs.index_select(1, self._saber_idx)                # (B, 50)
        body = obs.index_select(1, self._body_idx)                   # (B, 50)
        notes = obs[:, OFF_NOTES:OFF_NOTE_MASK].reshape(
            -1, N_NOTE_SLOTS, NOTE_FEAT)                             # (B, 24, 18)
        note_mask = obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS]
        return sabers, body, notes, note_mask

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        """SimBa readout trunk: (B, n_lead*embed) -> (B, trunk_dim)
        feature h. proj -> LN1 -> fc1 -> ReLU -> fc2 -> residual add + LN2, with
        the add+LN2 taken through FusedLayerNorm.forward_add (one fused
        pass on the CUDA path; since 2026-09-26 the fp16 kernels take
        rows up to 384 wide, so at trunk_dim 384 both LNs run the fused
        kernel on the fp16 stream — the fp32 record path keeps the stock
        fallback above 256). Same helpers as the blocks, so the fp16 stream (fp16
        activations, fp32 LN stats/affine) is honoured end to end."""
        if self.trunk_arch == "simbav2":
            return self._trunk_v2(x)
        x = _lin(self.trunk_proj, x)
        # fc1 + ReLU: on the fp16 stream fp16_linear_relu (cublasLt
        # RELU_BIAS epilogue forward, fused relu_bwd backward —
        # 2026-09-26; it was F.relu(_lin(...)), i.e. a separate ReLU
        # pass + threshold_backward + a gm.sum(0) bias grad; fp16
        # last-rounding class, like every other Linear+ReLU site).
        # fp32 / no extension: _lin then F.relu, bitwise as before.
        u = self.trunk_ln1(x)
        d = None
        if u.dtype is torch.float16 and _fl.ext_ok():
            # fc1 -> ReLU -> fc2 as one epilogue chain (2026-09-27)
            d = fp16_relu_chain(u, [self.trunk_fc1, self.trunk_fc2])
            if d is None:
                r = fp16_linear_relu(u, self.trunk_fc1.weight,
                                     self.trunk_fc1.bias)
        else:
            r = F.relu(_lin(self.trunk_fc1, u))
        if d is None:
            d = _lin(self.trunk_fc2, r)
        _, h = self.trunk_ln2.forward_add(x, d)
        return h

    def _trunk_v2(self, z: torch.Tensor) -> torch.Tensor:
        """SimBaV2 readout trunk (--trunk-arch simbav2): (B, n_lead*embed)
        -> (B, trunk_dim). GEMMs on the stream dtype (fp16_linear on the
        fp16 stream, the zero frozen bias adds exactly 0); scalers, ReLU,
        normalisations and the LERP in fp32, rounded once before each
        GEMM and at the output. The same arithmetic in rollout and update
        (no grad-mode branches), so r1 is unaffected."""
        dt = z.dtype
        x = _l2n(_lin(self.trunk_in, z).float()
                 * (self.trunk_s_in * self._v2_c_in))
        u = _lin(self.trunk_w1, x.to(dt)).float()
        u = F.relu(u * (self.trunk_s_1 * self._v2_c_1))
        y = _l2n(_lin(self.trunk_w2, u.to(dt)).float())
        alpha = self.trunk_alpha * self._v2_c_alpha
        x = _l2n(x + alpha * (y - x))
        return (x * math.sqrt(x.shape[-1])).to(dt)

    def trunk_projected(self):
        """The (W_in, W1, W2) Linears whose rows project_trunk_weights
        keeps unit-norm ([] unless --trunk-arch simbav2)."""
        if self.trunk_arch != "simbav2":
            return []
        return [self.trunk_in, self.trunk_w1, self.trunk_w2]

    @torch.no_grad()
    def project_trunk_weights(self) -> bool:
        """SimBaV2 weight projection: every output row of W_in / W1 / W2
        back onto the unit sphere (and their frozen biases at 0). In place
        on the fp32 masters, so flat-buffer views and CUDA-graph-captured
        storages stay valid; the CALLER bumps the fp16 weights version
        (fp16_linear.bump_weights_version) afterwards, exactly as after
        an optimizer step. No-op (False) for the SimBa trunk."""
        lins = self.trunk_projected()
        for lin in lins:
            w = lin.weight
            w.div_(w.norm(dim=1, keepdim=True).clamp_min(1e-12))
            lin.bias.zero_()
        return bool(lins)

    def forward_stacked(self, obs: torch.Tensor):
        """Trunk + ONE fused (trunk_dim -> 29K) head GEMM: stacked
        logits (B, 29, K) + the adapool critic's value (B,). The 29
        per-head Linears stay separate modules (state_dict unchanged);
        their weights are concatenated per call. On the fp16 stream the
        GEMMs run under the policy's accumulate mode (self.fp16_accum,
        fp16_linear.fp16_accum); the backward of a grad-mode call
        inherits it (the Functions latch it at forward time)."""
        if not (self.fp16 and obs.is_cuda):
            return self._forward_stacked(obs)
        with contextlib.ExitStack() as modes:
            if self.fp16_accum != "off":
                modes.enter_context(_fl.fp16_accum(self.fp16_accum))
            if self.fused_ffn != "off":
                # the fused block FFN (--fused-ffn); fused_ffn_ok falls
                # back per block wherever the kernel does not apply
                modes.enter_context(_fl.ffn_fusion(self.fused_ffn))
            return self._forward_stacked(obs)

    def _forward_stacked(self, obs: torch.Tensor):
        fp16 = self.fp16 and obs.is_cuda
        if fp16 and obs.dtype != torch.float16:
            # the fp16 stream consumes the fp16-ROUNDED obs in BOTH phases
            # (2026-09-26): `--fp16 all` stores b_obs half, so the update
            # forwards rounded obs while the rollout forwarded fp32 obs —
            # and several consumers (the note-geo lanes, and fp32 paths
            # before their casts) read the fp32 values at full precision,
            # so the acting and the updated policy disagreed at identical
            # weights (the fp16 diag r1; up to ~1 logit on a random net).
            # Rounding once here makes the two forwards bit-identical.
            obs = obs.half()
        sabers, body, notes, note_mask = self._split(obs)
        B = obs.shape[0]
        if self.note_geo:
            # geometry in fp32 from the (possibly fp16) lanes, then one
            # cast+copy
            notes = torch.cat(
                [notes, _note_geo(notes.float(),
                                  sabers.float().reshape(B, 2, SABER_FEAT),
                                  self.z_asinh, self.game == "flux",
                                  z_off=obs[:, _OBS_HEAD_Z].float())
                 .to(notes.dtype)],
                dim=-1)                                              # (B,24,40)
        if fp16:
            notes = notes.half()   # C1: the cast IS the copy F.linear
        emb = _seq(self.note_mlp, notes)                             # (B, 16, 64)
        dt = torch.float16 if fp16 else obs.dtype
        # hand-id one-hot is a constant: cached per (device, dtype)
        # instead of a fresh eye() alloc every forward (2026-07-28)
        hid = self.__dict__.get("_hand_id_cache")
        if hid is None or hid[0] != (obs.device, dt):
            hid = ((obs.device, dt),
                   torch.eye(2, device=obs.device, dtype=dt))
            self._hand_id_cache = hid
        hand_id = hid[1].expand(B, 2, 2)
        saber_feat = torch.cat(
            [sabers.reshape(B, 2, SABER_FEAT).to(dt), hand_id], dim=-1)
        # column slices of the (B, 93) body block are non-contiguous
        # views; the fp16 wgrad kernel needs contiguous inputs, so each
        # token's features are materialised (B x 29 / B x 64 fp16 copies
        # — negligible)
        body = body.to(dt)
        torso_tok = _seq(self.torso_mlp,
                         body[:, :TORSO_FEAT].contiguous()).unsqueeze(1)
        leg_feat = torch.cat([body[:, TORSO_FEAT:].contiguous()
                              .view(B, 2, LEG_FEAT),
                              hand_id], dim=-1)            # side one-hot
        body_tok = torch.cat([torso_tok, _seq(self.leg_mlp, leg_feat)],
                             dim=1)                                  # (B, 3, d)
        tokens = torch.cat([_seq(self.saber_mlp, saber_feat), body_tok, emb],
                           dim=1)                                    # (B, 29, d)
        # additive attention mask is block-invariant: build it once.
        # The lead lanes (sabers + body token(s)) are always attendable
        # -> bias exactly +0.0 = F.pad's zero fill (bit-identical).
        # The fused fp16 attention kernel accepts odd T since
        # 2026-08-02 (plain-__half bias load) — the 19-token v10
        # layout runs it again; the even-T fallback era was
        # 2026-07-31..08-02.
        attn_bias = F.pad((note_mask.unsqueeze(1) - 1.0) * MASK_BIAS,
                          (self.n_lead, 0))
        if fp16:
            attn_bias = attn_bias.half()  # C3: 2^14 is fp16-exact
        if self.emb_ln is not None:
            tokens = self.emb_ln(tokens)                 # --emb-ln
        delta = delta_b = None
        # --value-residual: block 0 hands back its attention values v_0;
        # every later block mixes them in (Block.forward v0)
        v0 = None
        for i, blk in enumerate(self.blocks[:-1]):
            if self.value_residual:
                r = blk(tokens, None, attn_bias=attn_bias, delta=delta,
                        delta_bias=delta_b,
                        next_ln=self.blocks[i + 1].ln_attn,
                        v0=v0, want_v=i == 0)
                if len(r) == 4:
                    tokens, delta, delta_b, v0 = r
                else:
                    tokens, delta, delta_b = r
                continue
            tokens, delta, delta_b = blk(tokens, None, attn_bias=attn_bias,
                                         delta=delta, delta_bias=delta_b,
                                         next_ln=self.blocks[i + 1].ln_attn)
        # final block: only the lead tokens' outputs are consumed.
        # adapool critic: it pools over the RESOLVED token stream
        # entering the final block — the same embeddings that block's
        # k/v see — which the block's fused ln_attn pass writes anyway,
        # so the block hands it back (want_resolved) instead of this
        # function recomputing `tokens + delta + delta_b` (2026-09-02,
        # Tier 1.1 of the 09-02 static audit, A §1). That recompute was
        # the fp32-critic bug: delta_b is the fp32 ffn.2.bias
        # PARAMETER, and dimensioned fp16 + dimensioned fp32 promotes
        # to fp32, so on the fp16 stream x_res — and with it critic_ln,
        # the four critic Linears (stock F.linear, non-tensor-core fp32
        # SGEMM), two hidden reshape copies in the pooling matmuls and
        # critic_mlp — silently ran in fp32 through stock torch since
        # the adapool commit (1f1c2a0). Now: 2 fewer (B,27,d) kernels,
        # the critic input is bit-consistent with the block's own k/v
        # input, and _critic rides the fp16 machinery. fp32 record path
        # (--fp16 off / CPU): bit-identical — the fused ln_fwd_add's x
        # IS `a + b` in fp32, exactly the old sum (delta_b is None
        # there). fp16: the value moves at fp16-rounding level, the
        # class already accepted for the actor; r1 is untouched (the
        # value is not in the ratio). Pinned by tests/test_critic_fp16.
        tokens, x_res = self.blocks[-1](tokens, None, n_queries=self.n_lead,
                                        attn_bias=attn_bias, delta=delta,
                                        delta_bias=delta_b,
                                        want_resolved=True, v0=v0)
        if self.final_ln is not None:
            tokens = self.final_ln(tokens)               # --final-ln
        h = self._trunk(tokens.reshape(B, -1))          # lead-token readout
        if self._diag_stash:
            self._diag_h = h
        tail = list(self.heads)
        if fp16 and not torch.is_grad_enabled():
            # rollout fast path: weights are frozen between opt.steps,
            # so the tail-tensor cat + fp16 cast is cached per weight
            # EPOCH (fp16_linear.bump_weights_version — tensor _version
            # is not bumped by fused Adam) instead of rebuilt 128 times
            c = self._refresh_head_cache()
            out = _fl._linear_f(h, c[1], c[2]).float()
        elif fp16:
            # Tier 1.5: the update path reuses the rollout's fused-head
            # cache — no per-head weight/bias cats (or their backward
            # split nodes) per minibatch; grads route to the individual
            # head masters via the packed Function's row slices.
            c = self._refresh_head_cache()
            out = _fl.fp16_packed_linear(h, c[1], c[2], tail).float()
        else:
            w = torch.cat([m.weight for m in tail])
            b = torch.cat([m.bias for m in tail])
            if fp16:
                # C5: the hard fp32 boundary — everything after
                # (log_softmax, sampling, PPO math) is exactly today's
                # fp32 code
                out = fp16_linear(h, w, b).float()
            else:
                out = F.linear(h, w, b)                  # (B, 29K)
        value = self._critic(x_res, h, attn_bias)
        if self._gear_idx is not None:
            # per-joint counts: (B, sum n_j) -> the padded (B, 29 k_max)
            # over the MASKED_LOGIT fill (fp32; grad flows back through
            # index_copy to the live lanes only)
            out = out.new_full((B, N_ACTION_HEADS * self.k_max),
                               MASKED_LOGIT).index_copy(1, self._gear_idx,
                                                        out)
        return out.view(B, N_ACTION_HEADS, self.k_max), value

    def _critic(self, x_res: torch.Tensor, h: torch.Tensor,
                attn_bias: torch.Tensor) -> torch.Tensor:
        """adapool_v1 value path. Query = a learned projection of the
        BODY token's post-LN embedding (state-derived, per Brothers
        2506.09215 — never a free CLS parameter); multi-head attention
        pool over all resolved tokens under the same additive note
        mask; value MLP on [pooled, trunk readout]. Read-only by
        construction: nothing here feeds back into the actor path.
        Output is the NORMALIZED value prediction (fp32) — the trainer
        owns the return-normalization stats."""
        B, T, d = x_res.shape
        # On the fp16 stream x_res is fp16 (Tier 1.1, see
        # forward_stacked), so from here every stage takes the same
        # fp16 path as the actor blocks: fp16 ln_fwd kernel, fp16_linear
        # (custom wgrad) for q/out, packed k/v, SDPA, and critic_mlp's
        # bias_relu / relu_bwd epilogues.
        hv = self.critic_ln(x_res)
        q = _lin(self.critic_q, hv[:, 2:3])   # body / torso token's query
        if hv.dtype is torch.float16:
            # A §9 / B §7: k and v share input hv -> ONE GEMM against
            # the [wk wv] cat replicas (a flat group when flattened,
            # else a version-keyed cat cache), one wgrad + one bias
            # sum in backward. Same mechanism as the blocks' qkv.
            wh, bh = _fl.packed_halves([self.critic_k, self.critic_v])
            kv = _fl.fp16_packed_linear(hv, wh, bh,
                                        [self.critic_k, self.critic_v])
            k, v = kv.split(d, dim=-1)
        else:
            k = _lin(self.critic_k, hv)
            v = _lin(self.critic_v, hv)
        nh, dh = self.n_heads, d // self.n_heads
        if hv.dtype is torch.float16 and _fl.attn29_ok(T, d, nh):
            # Tq = 1 on the T = 29 fused attention kernel (2026-09-27,
            # fp16_linear.attn29_qkv): (B, 1, d), heads concatenated
            a = _fl.attn29_qkv(q, kv, attn_bias).view(B, d)
        else:
            q = q.view(B, 1, nh, dh).transpose(1, 2)
            k = k.view(B, T, nh, dh).transpose(1, 2)
            v = v.view(B, T, nh, dh).transpose(1, 2)
            if q.dtype is torch.float16:
                # Tq = 1 fused attention under the same additive key mask
                # the blocks use (no all-masked rows: the 3 lead lanes
                # are always attendable). Replaces two batched matmuls
                # whose non-collapsible (B, nh) batch strides forced
                # hidden reshape COPIES of k and v, forward and backward.
                a = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias.unsqueeze(1))  # (B,nh,1,dh)
            else:
                scores = (q * dh ** -0.5) @ k.transpose(-2, -1)
                scores = scores + attn_bias.unsqueeze(1)    # (B, nh, 1, T)
                a = torch.softmax(scores, dim=-1) @ v
            a = a.transpose(1, 2).reshape(B, d)
        a = _lin(self.critic_out, a)
        logits = _seq(self.critic_mlp, torch.cat([a, h], dim=-1)).float()
        # HL-Gauss: bins -> expectation; the (B, K) logits are kept for
        # the update's cross-entropy (grad-enabled calls only)
        self._value_logits = logits if torch.is_grad_enabled() else None
        return torch.softmax(logits, dim=-1) @ self.value_centers

    def flatten_fp16(self):
        """Repack all Linear masters into one flat fp32 buffer with a
        flat fp16 replica mirror (fp16_linear.flatten_fp16_replicas).
        The heads + value head are laid out LAST in cat order, so the
        fused-head cache becomes a pair of static VIEWS — never
        rebuilt, refreshed by the same single flat cast as everything
        else. Trainer-only; MUST run before CUDA-graph capture."""
        tail = list(self.heads)
        # qkv groups (Tier 1.4): laid out [wq wk wv | bq bk bv] per
        # block so packed_halves resolves to static views.
        groups = [[blk.q_proj, blk.k_proj, blk.v_proj]
                  for blk in self.blocks]
        # critic k/v pack the same way (Tier 1.1 / A §9); critic_q runs
        # on the single torso row and stays a standalone Linear
        groups.append([self.critic_k, self.critic_v])
        w16, b16 = _fl.flatten_fp16_replicas(self, tail_group=tail,
                                             groups=groups)
        self._head_cache = [None, w16, b16]
        self._flat_fp16 = True

    def _refresh_head_cache(self):
        """Version-keyed fused-head weight cache, refreshed IN PLACE so
        the storages never move (CUDA-graph contract — see
        fp16_linear._half_params). Flat mode: the cache IS a view pair
        into the flat fp16 buffer; only the global refresh runs."""
        if getattr(self, "_flat_fp16", False):
            _fl.flat_refresh()
            return self._head_cache
        vers = _fl._weights_version
        c = getattr(self, "_head_cache", None)
        if c is None or c[0] != vers:
            tail = list(self.heads)
            w = torch.cat([m.weight for m in tail])
            b = torch.cat([m.bias for m in tail])
            if c is None:
                c = [vers, w.half(), b.half()]
                self._head_cache = c
            else:
                c[1].copy_(w)
                c[2].copy_(b)
                c[0] = vers
        return c

    def refresh_fp16_caches(self):
        """Force-refresh every fp16 replica cache (13 Linears + fused
        head) for the current weights version. A graphed rollout MUST
        call this after opt.steps — its captured kernels read the cache
        storages directly and never run the lazy rebuild logic. Flat
        mode collapses the whole refresh to ONE cast kernel."""
        if getattr(self, "_flat_fp16", False):
            _fl.flat_refresh()
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _fl._half_params(m.weight, m.bias)
        self._refresh_head_cache()
        _fl.refresh_packed()   # non-flat packed qkv cat caches, if any

    def forward(self, obs: torch.Tensor):
        logits, value = self.forward_stacked(obs)
        return list(logits.unbind(1)), value

    def get_action_and_value(self, obs, action=None, need_entropy=True):
        """need_entropy=False skips the entropy reduction (~5 kernels)
        for callers that discard it — the rollout and the GAE bootstrap.
        No RNG is consumed by entropy, so the sampling stream is
        untouched either way (2026-07-28)."""
        logits, value = self.forward_stacked(obs)
        logsm = torch.log_softmax(logits, dim=-1)
        probs = None
        if action is None:
            # Gumbel-max (2026-07-28): argmax(logp - log E), E~Exp(1),
            # samples the same categorical distribution as the previous
            # multinomial at a fraction of the kernel cost (no
            # renormalize pass, no block-per-row kernel), rides the
            # same default philox generator (graph-safe), and skips
            # the probs exp() entirely. Different RNG STREAM than
            # multinomial — the same accepted caveat the graphed
            # rollout capture already carried.
            g = torch.empty_like(logsm).exponential_()
            score = logsm - g.log()
            if self._gear_valid is not None:
                # an E = 0 draw is +inf on ANY lane (log 0): pin the lanes
                # past a head's count to -inf so they can never win
                score = score.masked_fill(~self._gear_valid, -math.inf)
            action = score.argmax(-1)
        elif action.dtype is not torch.long:
            # the trainer stores rollout actions as uint8 (2026-09-02,
            # Tier 1.3); gather needs an int64 index. Exact (integers),
            # (B, 29) — a tiny per-micro-batch cast.
            action = action.long()
        logprob = logsm.gather(-1, action.unsqueeze(-1)).squeeze(-1).sum(-1)
        entropy = None
        if need_entropy:
            if probs is None:
                probs = logsm.exp()
            # PER-HEAD first, then the 29-head sum. Bit-identical to the
            # old `-(probs*logsm).sum(-1).sum(-1)`: that was already two
            # reductions in exactly this order, so splitting the
            # intermediate out is a rename, not a numerics change (pinned
            # by tests/test_head_entropy.py). `_ent_h` is stashed for the
            # trainer's per-group / min-head entropy diagnostic
            # (2026-09-02, Tier 0.3 of the 09-02 static audit) and costs
            # zero extra kernels; the caller clears it.
            ent_h = -(probs * logsm).sum(-1)                  # (B, 29)
            # .detach(): the stash outlives the backward (the trainer
            # clears it at the next rollout tick 0), and a live grad_fn
            # would pin that micro-batch's autograd nodes for the rest
            # of the update. Shares storage, so it is a view, not a copy.
            self._ent_h = ent_h.detach()
            entropy = ent_h.sum(-1)
        return action, logprob, entropy, value


# pre-I07 (env v46..v48) layout — the one a checkpoint without the
# "blade_geo" arch marker was trained on; upgrade_pre_i07_state_dict
# maps it onto the current one
OBS_DIM_PRE_I07 = 624
SABER_FEAT_PRE_I07 = 44


def upgrade_pre_i07_state_dict(sd: dict, note_geo: bool) -> dict:
    """Map a pre-I07 state_dict (obs 624: SABER_FEAT 44, 14 note-geo
    lanes) onto the I07 layout with ZERO weights on every new input
    column, so the upgraded policy's output is the old one's (up to the
    B12 t_plane fix, which changes the forward by design):
      saber_mlp.0.weight (E, 44 + 2) -> (E, 47 + 2): the 3 tip-velocity
        columns are inserted before the hand one-hot (old cols 44:46 ->
        47:49);
      note_mlp.0.weight (E, 18 + 14) -> (E, 18 + 22) when note_geo: the
        8 I07 blade-frame columns are appended.
    Biases and every other tensor are untouched. Returns a new dict."""
    sd = dict(sd)
    w = sd["saber_mlp.0.weight"]
    assert w.shape[1] == SABER_FEAT_PRE_I07 + 2, (
        f"saber_mlp.0.weight {tuple(w.shape)} is not the pre-I07 layout")
    nw = w.new_zeros(w.shape[0], SABER_FEAT + 2)
    nw[:, :SABER_FEAT_PRE_I07] = w[:, :SABER_FEAT_PRE_I07]
    nw[:, SABER_FEAT:] = w[:, SABER_FEAT_PRE_I07:]
    sd["saber_mlp.0.weight"] = nw
    if note_geo:
        w = sd["note_mlp.0.weight"]
        assert w.shape[1] == NOTE_FEAT + NOTE_GEO_FEAT_V55, (
            f"note_mlp.0.weight {tuple(w.shape)} is not the pre-I07 layout")
        nw = w.new_zeros(w.shape[0], NOTE_FEAT + NOTE_GEO_FEAT)
        nw[:, :w.shape[1]] = w
        sd["note_mlp.0.weight"] = nw
    return sd


def build_head_gears(n_gears: int) -> tuple:
    """This build's per-joint gear counts at base table n_gears
    (constants.joint_gear_counts; raises ValueError when the build's
    extra gears do not fit on that base). Imported lazily so policy.py
    stays a leaf module; standalone (no beat_arena_env) = uniform."""
    try:
        from beat_arena_env.constants import joint_gear_counts
    except ImportError:
        return (n_gears,) * N_ACTION_HEADS
    return joint_gear_counts(n_gears)


def check_gear_layout(policy) -> None:
    """Refuse a policy whose per-head action counts are not this build's
    per-joint gear counts (constants.joint_gear_counts(n_gears): the base
    table plus the build's JOINT_EXTRA_GEARS). The env decodes gear ids
    with the build's tables, so a checkpoint trained on another gear
    layout (a body branch's, e.g. exp/bend7's 7-gear wrist bends) would
    play the wrong torques silently — bench / replay it from its branch."""
    from beat_arena_env import constants as _C
    try:
        build = _C.joint_gear_counts(policy.n_gears)
    except ValueError as e:
        raise SystemExit(f"n_gears={policy.n_gears} does not fit this "
                         f"build's per-joint gears: {e}") from None
    if tuple(policy.head_sizes) != build:
        raise SystemExit(
            f"the policy's per-joint gear counts {list(policy.head_sizes)} "
            f"(n_gears={policy.n_gears}) are not this build's "
            f"{list(build)} (constants.joint_gear_counts): the env would "
            "decode its actions with different gear tables. Bench / replay "
            "it from the branch it was trained on.")


def load_policy(ckpt, game: str = "standard") -> NotePolicy:
    """Rebuild a NotePolicy from a checkpoint dict. v10 checkpoints
    carry obs_dim/n_action_heads/body_token arch markers; anything else
    (v9 and earlier: obs 290 / 14 heads, no body token) is REFUSED —
    the v10 body retired every earlier checkpoint (house pattern).
    Also refused: an arch without z_asinh > 0 (the obs z encoding the
    note-geo lanes decode with; never defaulted), and a note_geo arch
    without the note_geo_norm marker (the raw-metre lanes, removed),
    and a checkpoint of the other GAME MODE (arch "game"; absent =
    "standard") — the caller names the mode it will play.

    Pre-I07 checkpoints (no "blade_geo" marker, obs 624; env v46..v48)
    are UPGRADED, not refused: the returned policy has the current
    layout (obs 630) with zero weights on the new input columns
    (upgrade_pre_i07_state_dict), so it plays the v49 env exactly as
    the old net played v48 except for the B12 t_plane fix — the
    --init-from warm start. policy.upgraded_from = "pre-i07" marks it
    (its optimizer state no longer matches: --resume refuses it)."""
    arch = ckpt.get("arch", {})
    blade_geo = arch.get("blade_geo", 0)
    if blade_geo not in (0, 1):
        raise SystemExit(
            f"checkpoint has blade_geo {blade_geo!r}; this policy knows "
            "the pre-I07 layout (absent) and I07 (1)")
    want_obs = OBS_DIM if blade_geo else OBS_DIM_PRE_I07
    if arch.get("game", "standard") != game:
        raise SystemExit(
            f"checkpoint plays game {arch.get('game', 'standard')!r} but "
            f"this tool plays {game!r}; the two modes share the network "
            "shape but not the obs lane meanings (SPEC.md §Flux)")
    if arch.get("body_tokens", 1) != 3:
        raise SystemExit(
            "checkpoint has the single body token (pre-v58, 2026-09-22); "
            "the body is now always [torso, leg_L, leg_R] and the old "
            "layout was removed (user: no back-compat). Check out a "
            "commit before 2026-09-22 to bench or replay it.")
    if (arch.get("obs_dim") != want_obs
            or arch.get("n_action_heads") != N_ACTION_HEADS
            or "ffn_mult" not in arch
            or not arch.get("body_token", False)):
        raise SystemExit(
            f"checkpoint is not env v46 era (needs arch obs_dim="
            f"{want_obs} (blade_geo {blade_geo}), n_action_heads=29, "
            f"ffn_mult, body_token=True; "
            f"got {dict(arch)}). Env v46 (2026-09-16) appended the 16 "
            "torque-limit scale lanes (obs 608 -> 624; v45 was 608, "
            "the same-day v44 block 626); every earlier checkpoint "
            "plays a different obs — retrain, no conversion.")
    # the obs z encoding feeds the note-geo distance decode: it must
    # come from the checkpoint (before 2026-09-22 it was dropped here and
    # every v55+ bench/replay decoded far notes linearly — train/eval skew)
    z_asinh = arch.get("z_asinh")
    if z_asinh is None or not float(z_asinh) > 0:
        raise SystemExit(
            f"checkpoint arch has z_asinh {z_asinh!r}: every loadable "
            "checkpoint records its obs z encoding (> 0 m, env v48); "
            "refusing to guess the note-distance decode")
    if arch.get("note_geo", False) and not arch.get("note_geo_norm", False):
        raise SystemExit(
            "checkpoint has the raw-metre note-geo lanes (no note_geo_norm "
            "marker, pre-v55); the normalised lanes are the only layout "
            "since 2026-09-22 (user: no back-compat). Check out a commit "
            "before 2026-09-22 to bench or replay it.")
    if arch.get("trunk_arch") not in TRUNK_ARCHS:
        raise SystemExit(
            f"checkpoint has readout trunk {arch.get('trunk_arch')!r}; the "
            "SimBa trunk is the only layout since 2026-09-22 (plus the "
            "opt-in 'simbav2', 2026-10-03; the plain MLP trunk was "
            "removed, no back-compat). Check out a commit before "
            "2026-09-22 to bench or replay it.")
    if arch.get("dwa"):
        raise ValueError(
            "checkpoint uses --dwa (DenseFormer depth-weighted averages), "
            "which lost the 2026-10-03 architecture A/B and was removed "
            "2026-10-04; no code path can rebuild it")
    policy = NotePolicy(
        embed=arch["embed"], trunk_dim=arch["trunk_dim"],
        n_blocks=arch["n_blocks"], n_heads=arch["n_heads"],
        n_gears=arch["n_gears"], ffn_mult=arch["ffn_mult"],
        # pre-2026-09-14 checkpoints carry no note_geo marker: 18 lanes
        note_geo=arch.get("note_geo", False),
        trunk_hidden=arch.get("trunk_hidden"),
        value_bins=arch["value_bins"],
        value_vmax=arch.get("value_vmax", 6.0),
        value_sigma=arch.get("value_sigma", 0.75),
        z_asinh=float(z_asinh), game=game,
        # absent (every checkpoint before issue #3) = fp32 accumulate
        fp16_accum=arch.get("fp16_accum", "off"),
        # absent (every checkpoint before review I13) = unfused FFN
        fused_ffn=arch.get("fused_ffn", "off"),
        # architecture A/B options (2026-10-03): absent = off
        emb_ln=arch.get("emb_ln", False),
        value_residual=arch.get("value_residual", False),
        final_ln=arch.get("final_ln", False),
        trunk_arch=arch["trunk_arch"],
        # per-joint gear counts (2026-10-04): absent = n_gears everywhere
        # (never the build default: a uniform checkpoint stays uniform,
        # and check_gear_layout refuses it on a build with extras)
        head_gears=(arch.get("head_gears")
                    or (arch["n_gears"],) * N_ACTION_HEADS))
    check_gear_layout(policy)
    sd = ckpt["state_dict"]
    policy.upgraded_from = None
    if not blade_geo:
        sd = upgrade_pre_i07_state_dict(sd, policy.note_geo)
        policy.upgraded_from = "pre-i07"
    policy.load_state_dict(sd)
    return policy
