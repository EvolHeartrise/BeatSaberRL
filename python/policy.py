"""Note-sequence policy for Beat Arena (SPEC.md §Policy).

Current production shape (v18, 2026-08-16): the 24-slot shared entity
window (notes + walls, distance-sorted), two saber tokens and one body
token form a 27-token sequence run through 3 full pre-LN transformer
blocks (LayerNorm -> 4-head self-attention -> residual; LayerNorm ->
4x FFN -> residual) at embed 128, then a 256-wide trunk MLP and 29
factored action heads (~863k actor params, the 2026-08-13
user-approved 6.2x scale-up) plus the adapool_v1 critic (2026-08-19,
topics/critic-design.md): a body-token-derived attention-pool query
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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import fp16_linear as _fl
from fp16_linear import fp16_linear, fp16_linear_nobias, fp16_linear_relu
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
            return (fp16_linear_nobias(_run(ms[:-1], x), tail.weight,
                                       tail.bias),
                    tail.bias)
        return _run(ms, x), None
    return _run(ms, x)


def _run(ms: list, x: torch.Tensor) -> torch.Tensor:
    i = 0
    while i < len(ms):
        m = ms[i]
        if isinstance(m, nn.Linear):
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
# into the body token (78 -> 87). OBS_DIM 608 -> 624.
OBS_DIM = 624
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
# note-geo lane constants (_S_WPOS/_S_WVEL/_S_BLADE) stay valid.
SABER_FEAT = 44
PREV_FEAT = 5
TAU_FEAT = 16
BODY_FEAT = 87
# derived from the window size so a future slot change can't skew them
_GI = OFF_NOTE_MASK + N_NOTE_SLOTS      # ground_imp block (4)
_BP = _GI + 4                           # body_pts block (18)
_BPV = _BP + 18                         # body_pt_vel block (18, env v24)
_PREV_L = _BPV + 18                     # prev-note L block (5, env v45)
_PREV_R = _PREV_L + PREV_FEAT           # prev-note R block (5, env v45)
_TAU = _PREV_R + PREV_FEAT              # tau-scale lanes (16, env v46)
assert _TAU + TAU_FEAT == OBS_DIM
# env v46 routing: arm axes (7) to both saber tokens; spine (3) + leg (6)
# axes to the body token — all appended LAST in their tokens
_TAU_ARM = tuple(range(_TAU + 3, _TAU + 10))
_TAU_BODY = tuple(range(_TAU, _TAU + 3)) + tuple(range(_TAU + 10, _TAU + 16))
_SABER_L_IDX = (tuple(range(18, 26)) + tuple(range(49, 56))
                + tuple(range(75, 86)) + tuple(range(_BP, _BP + 3))
                + tuple(range(_BPV, _BPV + 3))
                + tuple(range(_PREV_L, _PREV_L + PREV_FEAT))
                + _TAU_ARM)
_SABER_R_IDX = (tuple(range(26, 34)) + tuple(range(56, 63))
                + tuple(range(86, 97)) + tuple(range(_BP + 3, _BP + 6))
                + tuple(range(_BPV + 3, _BPV + 6))
                + tuple(range(_PREV_R, _PREV_R + PREV_FEAT))
                + _TAU_ARM)
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
#   dp (3)   wrist->note offset in METRES / 2  (x, y are both head-relative
#            in the obs since env v38 — same frame, same normalizers; z
#            is world z on both sides)
#   |dp| (1) / 2
#   t_plane (1)  note z / speed = seconds until the note reaches the head
#            plane, clipped to [-0.5, 3], / 1.0
#   v.d (1)  unit wrist-velocity . required cut direction (cos_a, sin_a, 0)
#   b.d (1)  blade direction . required cut direction
# = 7 per hand -> 14 lanes; the dot lanes are masked to 0 on wall rows
# (walls have no cut direction). Obs, env, kernel untouched; only
# note_mlp.0.weight widens (18 -> 32 input columns).
NOTE_GEO_FEAT = 14
# obs normalizers, pinned against beat_arena_env.constants in
# tests/test_policy.py (the obs is written by the Rust/CUDA sims)
_X_NORM, _Y_NORM, _Z_SCALE, _V_NORM = 1.0, 0.75, 8.0, 6.0
# saber_feat lane order: q(8) qd(7) then the 11-lane hand block
# [wrist pos(3), wrist vel(3), blade dir(3), meter, post_rem]
_S_WPOS, _S_WVEL, _S_BLADE = 15, 18, 21


def _note_geo(notes: torch.Tensor, sabers: torch.Tensor, norm: bool = False,
              z_asinh: float = 0.0) -> torch.Tensor:
    """(B, 24, 18) fp32 note lanes + (B, 2, 32) fp32 saber lanes ->
    (B, 24, 14) geometry lanes (fp32; the caller casts with the notes).

    norm=False (every checkpoint before 2026-09-21): metres and seconds,
    z decoded as lane 2 x 8 (wrong beyond ~8 m under --z-asinh).
    norm=True (arch marker 'note_geo_norm', user 2026-09-21): every lane
    O(1) — z decoded from the lane's true encoding (c sinh(lane) under
    --z-asinh c, the linear lane otherwise), the per-hand offset's z and
    the distance compressed as asinh(m / 8) (the lane-2 shape), x/y
    offsets in half-metres as before, seconds-to-plane / 3. Motivation:
    [gnorm-lanes] on v52-v54 put the note encoder's excess gradient in
    its second layer, fed by far tokens' large hidden activations from
    these raw-metre lanes."""
    if norm and z_asinh > 0:
        z_m = torch.sinh(notes[..., 2]) * float(z_asinh)
    else:
        z_m = notes[..., 2] * _Z_SCALE
    npos = torch.stack([notes[..., 0] * _X_NORM, notes[..., 1] * _Y_NORM,
                        z_m], -1)                                # (B,24,3) m
    cutd = torch.stack([notes[..., 6], notes[..., 7],
                        torch.zeros_like(notes[..., 6])], -1)   # (B,24,3)
    not_wall = 1.0 - notes[..., 14:15]                          # (B,24,1)
    # seconds to the head plane: z / speed with speed = -lane3 * Z_SCALE
    speed = (-notes[..., 3:4] * _Z_SCALE).clamp_min(1e-3)
    tpl = (z_m[..., None] / speed).clamp(-0.5, 3.0)
    if norm:
        tpl = tpl / 3.0
    feats = []
    for hnd in range(2):
        w = sabers[:, hnd]                                       # (B,32)
        wpos = torch.stack([w[:, _S_WPOS] * _X_NORM,
                            w[:, _S_WPOS + 1] * _Y_NORM,
                            w[:, _S_WPOS + 2]], -1)              # (B,3) m
        dp = (npos - wpos[:, None]) * 0.5                        # (B,24,3)
        dist = dp.norm(dim=-1, keepdim=True)
        if norm:
            # z offset and distance: asinh(m / 8) keeps the near field
            # linear (2 dp units = 1 m) and bounds far tokens (~3 at
            # 100 m, ~7 at 2 km) instead of raw half-metres
            dp = torch.cat([dp[..., :2], torch.asinh(dp[..., 2:3] * 2.0 / 8.0)], -1)
            dist = torch.asinh(dist * 2.0 / 8.0)
        v = w[:, _S_WVEL:_S_WVEL + 3] * _V_NORM                  # (B,3) m/s
        vhat = v / (v.norm(dim=-1, keepdim=True) + 0.05)
        vdot = (vhat[:, None] * cutd).sum(-1, keepdim=True) * not_wall
        bdot = (w[:, _S_BLADE:_S_BLADE + 3][:, None] * cutd).sum(
            -1, keepdim=True) * not_wall
        feats += [dp, dist, tpl, vdot, bdot]
    return torch.cat(feats, -1)
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


def _mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class Block(nn.Module):
    """Full pre-LN transformer block.

    n_heads == 1 keeps the legacy layout (v added straight into the
    residual stream, no output projection) so pre-2026-07-21 checkpoints
    load unchanged. n_heads > 1 uses standard multi-head attention with
    a concat-heads output projection.
    """

    def __init__(self, d: int, n_heads: int = 1, ffn_mult: int = 2):
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

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor,
                n_queries: int = 0, attn_bias: torch.Tensor | None = None,
                delta: torch.Tensor | None = None,
                delta_bias: torch.Tensor | None = None,
                want_resolved: bool = False):
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
        B, T, _ = x.shape
        if delta is not None:
            x, h = self.ln_attn.forward_add(x, delta, delta_bias)
        else:
            h = self.ln_attn(x)
        qkv = None
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
            # ffn.2 keeps its own bias
            out = x + _seq(self.ffn, h2)
            return (out, x_full) if want_resolved else out
        # non-final block: the consumer is the NEXT block's
        # ln_attn.forward_add, which can absorb ffn.2's bias
        ffn_out, ffn_bias = _seq(self.ffn, h2, fold_tail=True)
        return x, ffn_out, ffn_bias


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
    # trunk 384 and trunk_arch "simba" (train_batch.py); the ctor keeps
    # the historical defaults (128/256, "mlp") so checkpoint loading and
    # tests that construct explicitly are unaffected.
    def __init__(self, embed: int = 128, trunk_dim: int = 256,
                 encoder: str = "transformer", n_blocks: int = 3,
                 n_heads: int = 4, n_gears: int = 3, ffn_mult: int = 4,
                 note_geo: bool = False, note_geo_norm: bool = False,
                 value_bins: int = 0, value_vmax: float = 6.0,
                 value_sigma: float = 0.75,
                 trunk_arch: str = "mlp", trunk_hidden: int | None = None):
        super().__init__()
        assert encoder == "transformer", (
            f"unknown encoder: {encoder} (pool was removed 2026-07-20)")
        assert n_gears in (3, 5, 7), f"n_gears must be 3/5/7, got {n_gears}"
        assert trunk_arch in ("mlp", "simba"), trunk_arch
        self.encoder = encoder
        self.embed = embed
        self.n_heads = n_heads
        self.n_gears = n_gears
        self.head_sizes = (n_gears,) * N_ACTION_HEADS
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
                     "note_geo_norm": bool(note_geo_norm),
                     # 2026-09-15 (brainstorm item 2): readout-trunk
                     # layout. "mlp" = the plain Linear-ReLU-Linear every
                     # checkpoint before this date used (absent marker =
                     # mlp); "simba" = residual + LayerNorm block, below.
                     "trunk_arch": trunk_arch,
                     "trunk_hidden": (int(trunk_hidden) if trunk_hidden
                                      else trunk_dim),
                     # HL-Gauss critic head (2026-09-19): 0 = scalar
                     # regression; K > 0 = K-bin histogram over the
                     # normalized return in [-vmax, vmax], target width
                     # sigma (in bin widths), value = expectation
                     "value_bins": int(value_bins),
                     "value_vmax": float(value_vmax),
                     "value_sigma": float(value_sigma)}
        self.value_bins = int(value_bins)
        self.value_sigma_bins = float(value_sigma)
        if self.value_bins:
            centers = torch.linspace(-float(value_vmax), float(value_vmax),
                                     self.value_bins)
            self.register_buffer("value_centers", centers, persistent=False)
            # sigma in return units: sigma_bins x bin width
            self.value_sigma = float(value_sigma) * float(centers[1] - centers[0])
        self._value_logits = None
        self.note_geo = bool(note_geo)
        self.note_geo_norm = bool(note_geo_norm)
        self.trunk_arch = trunk_arch
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
        self.blocks = nn.ModuleList(
            Block(embed, n_heads, ffn_mult) for _ in range(n_blocks))
        # readout = the 5 lead tokens' output embeddings
        if trunk_arch == "mlp":
            self.trunk = _mlp([self.n_lead * embed, trunk_dim, trunk_dim])
        else:
            # SimBa-style readout trunk (2026-09-15, brainstorm item 2;
            # SimBa 2410.09754 / SimBaV2 2502.15280, Creus Castanyer
            # 2506.15544, Kumar 2010.14498): the plain trunk was the
            # ONE stage with neither normalization nor a skip, and it
            # produces the feature h whose effective rank the [diag]
            # line tracks. Layout: projection -> one residual block
            # x + fc2(relu(fc1(LN1 x))) -> LN2 -> h. H = trunk_hidden
            # (default trunk_dim: +149,376 params at 192/384; 1536 =
            # +1,035,264). Registered in the same parameters() slot as
            # `trunk` so the flat-grad diag groups (encoder -> blocks ->
            # trunk -> heads -> critic) keep their order; the trainer's
            # diag looks the trunk group up by name prefix.
            H = int(trunk_hidden) if trunk_hidden else trunk_dim
            self.trunk_proj = nn.Linear(self.n_lead * embed, trunk_dim)
            self.trunk_ln1 = FusedLayerNorm(trunk_dim)
            self.trunk_fc1 = nn.Linear(trunk_dim, H)
            self.trunk_fc2 = nn.Linear(H, trunk_dim)
            self.trunk_ln2 = FusedLayerNorm(trunk_dim)
        # 29 factored joint heads, K = n_gears options each (env v8)
        self.heads = nn.ModuleList(
            nn.Linear(trunk_dim, n) for n in self.head_sizes)
        # Critic (2026-08-19 redesign, topics/critic-design.md):
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
                                trunk_dim, self.value_bins or 1])
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

    def _split(self, obs: torch.Tensor):
        sabers = obs.index_select(1, self._saber_idx)                # (B, 50)
        body = obs.index_select(1, self._body_idx)                   # (B, 50)
        notes = obs[:, OFF_NOTES:OFF_NOTE_MASK].reshape(
            -1, N_NOTE_SLOTS, NOTE_FEAT)                             # (B, 24, 18)
        note_mask = obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS]
        return sabers, body, notes, note_mask

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Readout trunk: (B, 3*embed) -> (B, trunk_dim) feature h.
        mlp: the plain Linear-ReLU-Linear (unchanged bits). simba:
        proj -> LN1 -> fc1 -> ReLU -> fc2 -> residual add + LN2, with
        the add+LN2 taken through FusedLayerNorm.forward_add (one fused
        pass on the CUDA path when the kernel is eligible — its row
        width must be <= 256, so at trunk_dim 384 both LNs run the
        stock fp32-stats fallback; the residual sum is then a plain
        add). Same helpers as the blocks, so the fp16 stream (fp16
        activations, fp32 LN stats/affine) is honoured end to end."""
        if self.trunk_arch == "mlp":
            return _seq(self.trunk, x)
        x = _lin(self.trunk_proj, x)
        d = _lin(self.trunk_fc2, F.relu(_lin(self.trunk_fc1,
                                              self.trunk_ln1(x))))
        _, h = self.trunk_ln2.forward_add(x, d)
        return h

    def forward_stacked(self, obs: torch.Tensor):
        """Trunk + ONE fused (trunk_dim -> 29K) head GEMM: stacked
        logits (B, 29, K) + the adapool critic's value (B,). The 29
        per-head Linears stay separate modules (state_dict unchanged);
        their weights are concatenated per call."""
        fp16 = self.fp16 and obs.is_cuda
        sabers, body, notes, note_mask = self._split(obs)
        B = obs.shape[0]
        if self.note_geo:
            # fp32 geometry from the fp32 obs slices, then one cast+copy
            notes = torch.cat(
                [notes, _note_geo(notes, sabers.reshape(B, 2, SABER_FEAT),
                                  norm=self.note_geo_norm,
                                  z_asinh=float(self.arch.get("z_asinh", 0.0)))],
                dim=-1)                                              # (B,24,32)
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
        delta = delta_b = None
        for blk in self.blocks[:-1]:
            tokens, delta, delta_b = blk(tokens, None, attn_bias=attn_bias,
                                         delta=delta, delta_bias=delta_b)
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
        # the adapool commit (8d36d47). Now: 2 fewer (B,27,d) kernels,
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
                                        want_resolved=True)
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
            out = F.linear(h, c[1], c[2]).float()
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
        return out.view(B, N_ACTION_HEADS, self.n_gears), value

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
        q = q.view(B, 1, nh, dh).transpose(1, 2)
        k = k.view(B, T, nh, dh).transpose(1, 2)
        v = v.view(B, T, nh, dh).transpose(1, 2)
        if q.dtype is torch.float16:
            # Tq = 1 fused attention under the same additive key mask
            # the blocks use (no all-masked rows: the 3 lead lanes are
            # always attendable). Replaces two batched matmuls whose
            # non-collapsible (B, nh) batch strides forced hidden
            # reshape COPIES of k and v, forward and backward.
            a = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias.unsqueeze(1))   # (B,nh,1,dh)
        else:
            scores = (q * dh ** -0.5) @ k.transpose(-2, -1)
            scores = scores + attn_bias.unsqueeze(1)     # (B, nh, 1, T)
            a = torch.softmax(scores, dim=-1) @ v
        a = _lin(self.critic_out, a.transpose(1, 2).reshape(B, d))
        val = _seq(self.critic_mlp, torch.cat([a, h], dim=-1))
        if self.value_bins:
            # HL-Gauss: bins -> expectation; the (B, K) logits are kept
            # for the update's cross-entropy (grad-enabled calls only)
            logits = val.float()
            self._value_logits = logits if torch.is_grad_enabled() else None
            return torch.softmax(logits, dim=-1) @ self.value_centers
        return val.float().squeeze(-1)

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
            action = (logsm - g.log()).argmax(-1)
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


def load_policy(ckpt) -> NotePolicy:
    """Rebuild a NotePolicy from a checkpoint dict. v10 checkpoints
    carry obs_dim/n_action_heads/body_token arch markers; anything else
    (v9 and earlier: obs 290 / 14 heads, no body token) is REFUSED —
    the v10 body retired every earlier checkpoint (house pattern)."""
    arch = ckpt.get("arch", {})
    if arch.get("body_tokens", 1) != 3:
        raise SystemExit(
            "checkpoint has the single body token (pre-v58, 2026-09-22); "
            "the body is now always [torso, leg_L, leg_R] and the old "
            "layout was removed (user: no back-compat). Check out a "
            "commit before 2026-09-22 to bench or replay it.")
    if (arch.get("obs_dim") != OBS_DIM
            or arch.get("n_action_heads") != N_ACTION_HEADS
            or "ffn_mult" not in arch
            or not arch.get("body_token", False)):
        raise SystemExit(
            f"checkpoint is not env v46 era (needs arch obs_dim="
            f"{OBS_DIM}, n_action_heads=29, ffn_mult, body_token=True; "
            f"got {dict(arch)}). Env v46 (2026-09-16) appended the 16 "
            "torque-limit scale lanes (obs 608 -> 624; v45 was 608, "
            "the same-day v44 block 626); every earlier checkpoint "
            "plays a different obs — retrain, no conversion.")
    policy = NotePolicy(
        embed=arch["embed"], trunk_dim=arch["trunk_dim"],
        n_blocks=arch["n_blocks"], n_heads=arch["n_heads"],
        n_gears=arch["n_gears"], ffn_mult=arch["ffn_mult"],
        # pre-2026-09-14 checkpoints carry no note_geo marker: 18 lanes
        note_geo=arch.get("note_geo", False),
        note_geo_norm=arch.get("note_geo_norm", False),   # 2026-09-21
        # pre-2026-09-15 checkpoints carry no trunk marker: plain mlp
        trunk_arch=arch.get("trunk_arch", "mlp"),
        trunk_hidden=arch.get("trunk_hidden"),
        value_bins=arch.get("value_bins", 0),
        value_vmax=arch.get("value_vmax", 6.0),
        value_sigma=arch.get("value_sigma", 0.75))
    policy.load_state_dict(ckpt["state_dict"])
    return policy
