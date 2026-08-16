"""Note-sequence policy for Beat Arena (SPEC.md §Policy).

Current production shape (v18, 2026-08-16): the 24-slot shared entity
window (notes + walls, distance-sorted), two saber tokens and one body
token form a 27-token sequence run through 3 full pre-LN transformer
blocks (LayerNorm -> 4-head self-attention -> residual; LayerNorm ->
4x FFN -> residual) at embed 128, then a 256-wide trunk MLP and 29
factored action heads + a value head (~861k params, the 2026-08-13
user-approved 6.2x scale-up).

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

import torch
import torch.nn as nn
import torch.nn.functional as F

import fp16_linear as _fl
from fp16_linear import fp16_linear, fp16_linear_nobias, fp16_linear_relu
from fused_ln import FusedLayerNorm

# (The round-4 fused single-head attention kernel was DELETED
# 2026-08-16: it only ever ran at n_heads == 1 and d == 64, both
# impossible since the 6.2x scale-up — see PERF_AUDIT_2026-08-16.md.)
# Env toggles (debug/bisect) for the LIVE fp16 epilogue paths:
#   BEAT_NO_BIAS_RELU=1   -> eager addmm+ReLU in _seq
#   BEAT_NO_BIAS_FOLD=1   -> ffn.2 keeps its addmm bias epilogue
#                            instead of folding into the next block's
#                            fused residual+LN pass
import os as _os
_NO_BIAS_RELU = _os.environ.get("BEAT_NO_BIAS_RELU", "0") == "1"
_NO_BIAS_FOLD = _os.environ.get("BEAT_NO_BIAS_FOLD", "0") == "1"
# Perf toggles (audit Tier 1.3-1.5), fp16 stream ONLY — the fp32
# record path stays bitwise stable under every setting. Defaults are
# the 2026-08-15 ladder WINNERS (isolated-pod A/B, numerics-gated):
# sdpa+qkv+heads measured 49.2 -> 40.3 ms/update (-18%), graphed
# rollout forward -23% at N=16384, production train --bench +5.4-6.9%
# steps/s — and r1 (rollout/update consistency) is EXACTLY 0 because
# both phases now run the same SDPA kernels. Numerics class: fp16
# last-rounding (worst 5.9e-3 rel on grad probes), same accepted class
# as the delta_bias fold. Escape hatches for bisects:
#   BEAT_ATTN=legacy    hand-rolled scores/softmax/@v (pre-ladder)
#   BEAT_ATTN=fold      pre-scaled-q reassociation (measured mid)
#   BEAT_QKV_PACKED=0   per-projection q/k/v GEMMs
#   BEAT_PACKED_HEADS=0 30-way head cat per update minibatch
# (The old "SDPA slower" refutation was sm_61/d64/T19/1-head — it did
# not transfer to the 4090/4-head/27-token config, as the audit
# predicted.)
_ATTN_MODE = _os.environ.get("BEAT_ATTN", "sdpa")
assert _ATTN_MODE in ("legacy", "fold", "sdpa"), _ATTN_MODE
_QKV_PACKED = _os.environ.get("BEAT_QKV_PACKED", "1") == "1"
_PACKED_HEADS = _os.environ.get("BEAT_PACKED_HEADS", "1") == "1"


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
                and not _NO_BIAS_FOLD and _fl.ext_ok()):
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
                    and not _NO_BIAS_RELU and _fl.ext_ok()):
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
# OBS_DIM 578 -> 580, offsets past the hands +2.
OBS_DIM = 580
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
# positions (12, env v13) = 66 features. Each saber token gains its
# arm's elbow position (3, env v13): 26 -> 29.
SABER_FEAT = 29
BODY_FEAT = 66
# derived from the window size so a future slot change can't skew them
_GI = OFF_NOTE_MASK + N_NOTE_SLOTS      # ground_imp block (4)
_BP = _GI + 4                           # body_pts block (18)
_SABER_L_IDX = (tuple(range(18, 26)) + tuple(range(49, 56))
                + tuple(range(75, 86)) + tuple(range(_BP, _BP + 3)))
_SABER_R_IDX = (tuple(range(26, 34)) + tuple(range(56, 63))
                + tuple(range(86, 97)) + tuple(range(_BP + 3, _BP + 6)))
_BODY_IDX = (tuple(range(0, 18)) + tuple(range(34, 49))
             + tuple(range(63, 75)) + tuple(range(97, 102))
             + tuple(range(_GI, _GI + 4))
             + tuple(range(_BP + 6, _BP + 18)))
assert len(_SABER_L_IDX) == len(_SABER_R_IDX) == SABER_FEAT
assert len(_BODY_IDX) == BODY_FEAT
# env v10: 29 factored joint heads (SPEC-v10 §1.4 actuated order —
# spine 3, arm_l 7, arm_r 7, leg_l 6, leg_r 6); K = n_gears options
# per head (env v8 gear tables). This module constant is the K=3
# default; a policy's real sizes live in policy.head_sizes.
N_ACTION_HEADS = 29
HEAD_SIZES = (3,) * N_ACTION_HEADS

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
                delta_bias: torch.Tensor | None = None):
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
        B, T, _ = x.shape
        if delta is not None:
            x, h = self.ln_attn.forward_add(x, delta, delta_bias)
        else:
            h = self.ln_attn(x)
        qkv = None
        if _QKV_PACKED and h.dtype is torch.float16:
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
            if _ATTN_MODE == "sdpa" and q.dtype is torch.float16:
                # Tier 1.3b: fused attention. The additive key mask is
                # legal here — no row is ever all-masked (3 lead lanes
                # are always attendable), so no NaN softmax rows.
                att = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias.unsqueeze(1))
            else:
                if _ATTN_MODE == "fold" and q.dtype is torch.float16:
                    # Tier 1.3a: pre-scale q (tiny tensor) instead of
                    # dividing the (B,H,Tq,T) scores, and add the bias
                    # in place — one full-scores kernel dropped, one
                    # alloc dropped. Near-tie numerics (mul-by-
                    # reciprocal on q vs divide on scores).
                    scores = (q * self.dh ** -0.5) @ k.transpose(-2, -1)
                    scores += attn_bias.unsqueeze(1)
                else:
                    scores = q @ k.transpose(-2, -1) / (self.dh ** 0.5)
                    scores = scores + attn_bias.unsqueeze(1)     # (B,1,1,T)
                att = torch.softmax(scores, dim=-1) @ v          # (B,H,Tq,dh)
            att = att.transpose(1, 2).reshape(B, Tq, self.d)
            att = _lin(self.out_proj, att)
        x, h2 = self.ln_ffn.forward_add(x, att)
        if n_queries:
            # final block: nothing downstream fuses the residual add, so
            # ffn.2 keeps its own bias
            return x + _seq(self.ffn, h2)
        # non-final block: the consumer is the NEXT block's
        # ln_attn.forward_add, which can absorb ffn.2's bias
        ffn_out, ffn_bias = _seq(self.ffn, h2, fold_tail=True)
        return x, ffn_out, ffn_bias


class NotePolicy(nn.Module):
    # v17 scaled defaults (2026-08-13, user-approved 6.2x): embed 128,
    # 4 attention heads (multi-entity reasoning — a saber can attend to
    # its next note AND an incoming wall separately), 3 blocks, 4x FFN,
    # trunk 256 -> ~861k params at gears 5 (was 138,642). The 138k arch
    # and its fused update kernels were retired with it (no conversion,
    # house pattern); training runs the eager fp16 path.
    def __init__(self, embed: int = 128, trunk_dim: int = 256,
                 encoder: str = "transformer", n_blocks: int = 3,
                 n_heads: int = 4, n_gears: int = 3, ffn_mult: int = 4):
        super().__init__()
        assert encoder == "transformer", (
            f"unknown encoder: {encoder} (pool was removed 2026-07-20)")
        assert n_gears in (3, 5, 7), f"n_gears must be 3/5/7, got {n_gears}"
        self.encoder = encoder
        self.embed = embed
        self.n_gears = n_gears
        self.head_sizes = (n_gears,) * N_ACTION_HEADS
        # arch record: load_policy REFUSES checkpoints whose markers
        # don't match the current era (obs 426 / ffn_mult present).
        self.arch = {"embed": embed, "trunk_dim": trunk_dim,
                     "n_blocks": n_blocks, "n_heads": n_heads,
                     "n_gears": n_gears, "obs_dim": OBS_DIM,
                     "ffn_mult": ffn_mult,
                     "n_action_heads": N_ACTION_HEADS, "body_token": True}
        self.note_mlp = _mlp([NOTE_FEAT, embed, embed])
        # saber token features: the 25 obs-v10 hand features + [is_left,
        # is_right]; the body token gets its own encoder
        self.saber_mlp = _mlp([SABER_FEAT + 2, embed, embed])
        self.body_mlp = _mlp([BODY_FEAT, embed, embed])
        self.blocks = nn.ModuleList(
            Block(embed, n_heads, ffn_mult) for _ in range(n_blocks))
        # readout = [saber_L, saber_R, body] output embeddings
        self.trunk = _mlp([3 * embed, trunk_dim, trunk_dim])
        # 29 factored joint heads, K = n_gears options each (env v8)
        self.heads = nn.ModuleList(
            nn.Linear(trunk_dim, n) for n in self.head_sizes)
        self.value_head = nn.Linear(trunk_dim, 1)
        # token-slice index buffers (device-tracked, not in state_dict)
        self.register_buffer(
            "_saber_idx", torch.tensor(_SABER_L_IDX + _SABER_R_IDX,
                                       dtype=torch.long), persistent=False)
        self.register_buffer(
            "_body_idx", torch.tensor(_BODY_IDX, dtype=torch.long),
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

    def forward_stacked(self, obs: torch.Tensor):
        """Trunk + ONE fused (trunk_dim -> 29K + 1) head GEMM: stacked
        logits (B, 29, K) + value (B,). The 29 per-head Linears + value
        head stay separate modules (state_dict unchanged); their weights
        are concatenated per call."""
        fp16 = self.fp16 and obs.is_cuda
        sabers, body, notes, note_mask = self._split(obs)
        if fp16:
            notes = notes.half()   # C1: the cast IS the copy F.linear
        emb = _seq(self.note_mlp, notes)                             # (B, 16, 64)
        B = obs.shape[0]
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
        body_tok = _seq(self.body_mlp, body.to(dt)).unsqueeze(1)     # (B, 1, 64)
        tokens = torch.cat([_seq(self.saber_mlp, saber_feat), body_tok, emb],
                           dim=1)                                    # (B, 19, 64)
        # additive attention mask is block-invariant: build it once.
        # The 3 lead lanes (2 sabers + body) are always attendable ->
        # bias exactly +0.0 = F.pad's zero fill (bit-identical).
        # The fused fp16 attention kernel accepts odd T since
        # 2026-08-02 (plain-__half bias load) — the 19-token v10
        # layout runs it again; the even-T fallback era was
        # 2026-07-31..08-02.
        attn_bias = F.pad((note_mask.unsqueeze(1) - 1.0) * MASK_BIAS,
                          (3, 0))
        if fp16:
            attn_bias = attn_bias.half()  # C3: 2^14 is fp16-exact
        delta = delta_b = None
        for blk in self.blocks[:-1]:
            tokens, delta, delta_b = blk(tokens, None, attn_bias=attn_bias,
                                         delta=delta, delta_bias=delta_b)
        # final block: only the 3 readout tokens' outputs are consumed
        tokens = self.blocks[-1](tokens, None, n_queries=3,
                                 attn_bias=attn_bias, delta=delta,
                                 delta_bias=delta_b)
        h = _seq(self.trunk, tokens.reshape(B, -1))     # saber+body readout
        if fp16 and not torch.is_grad_enabled():
            # rollout fast path: weights are frozen between opt.steps,
            # so the 30-tensor cat + fp16 cast is cached per weight
            # EPOCH (fp16_linear.bump_weights_version — tensor _version
            # is not bumped by fused Adam) instead of rebuilt 128 times
            c = self._refresh_head_cache()
            out = F.linear(h, c[1], c[2]).float()
            return (out[:, :-1].view(B, N_ACTION_HEADS, self.n_gears),
                    out[:, -1])
        if fp16 and _PACKED_HEADS:
            # Tier 1.5: the update path reuses the rollout's fused-head
            # cache — no 30-way weight/bias cats (or their backward
            # split nodes) per minibatch; grads route to the individual
            # head masters via the packed Function's row slices.
            c = self._refresh_head_cache()
            out = _fl.fp16_packed_linear(
                h, c[1], c[2], list(self.heads) + [self.value_head]
            ).float()
            return (out[:, :-1].view(B, N_ACTION_HEADS, self.n_gears),
                    out[:, -1])
        w = torch.cat([hd.weight for hd in self.heads]
                      + [self.value_head.weight])
        b = torch.cat([hd.bias for hd in self.heads]
                      + [self.value_head.bias])
        if fp16:
            # C5: the hard fp32 boundary — everything after (log_softmax,
            # sampling, PPO math) is exactly today's fp32 code
            out = fp16_linear(h, w, b).float()
        else:
            out = F.linear(h, w, b)                          # (B, 29K + 1)
        return out[:, :-1].view(B, N_ACTION_HEADS, self.n_gears), out[:, -1]

    def flatten_fp16(self):
        """Repack all Linear masters into one flat fp32 buffer with a
        flat fp16 replica mirror (fp16_linear.flatten_fp16_replicas).
        The heads + value head are laid out LAST in cat order, so the
        fused-head cache becomes a pair of static VIEWS — never
        rebuilt, refreshed by the same single flat cast as everything
        else. Trainer-only; MUST run before CUDA-graph capture."""
        tail = list(self.heads) + [self.value_head]
        # qkv groups (Tier 1.4): laid out [wq wk wv | bq bk bv] per
        # block so packed_halves resolves to static views. Grouping is
        # harmless when the toggle is off (layout-only), but kept
        # toggle-gated so the flat buffer layout — and therefore any
        # bit-comparison of flat-order-dependent consumers — is
        # unchanged in the default configuration.
        groups = ([[blk.q_proj, blk.k_proj, blk.v_proj]
                   for blk in self.blocks] if _QKV_PACKED else None)
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
            w = torch.cat([hd.weight for hd in self.heads]
                          + [self.value_head.weight])
            b = torch.cat([hd.bias for hd in self.heads]
                          + [self.value_head.bias])
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
        logits, value = self.forward_stacked(obs)                # (B, 12, 3)
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
        logprob = logsm.gather(-1, action.unsqueeze(-1)).squeeze(-1).sum(-1)
        entropy = None
        if need_entropy:
            if probs is None:
                probs = logsm.exp()
            entropy = -(probs * logsm).sum(-1).sum(-1)
        return action, logprob, entropy, value


def load_policy(ckpt) -> NotePolicy:
    """Rebuild a NotePolicy from a checkpoint dict. v10 checkpoints
    carry obs_dim/n_action_heads/body_token arch markers; anything else
    (v9 and earlier: obs 290 / 14 heads, no body token) is REFUSED —
    the v10 body retired every earlier checkpoint (house pattern)."""
    arch = ckpt.get("arch", {})
    if (arch.get("obs_dim") != OBS_DIM
            or arch.get("n_action_heads") != N_ACTION_HEADS
            or "ffn_mult" not in arch
            or not arch.get("body_token", False)):
        raise SystemExit(
            f"checkpoint is not v17-scaled era (needs arch obs_dim="
            f"{OBS_DIM}, n_action_heads=29, ffn_mult, body_token=True; "
            f"got {dict(arch)}). Pre-scale checkpoints (obs 362 / the "
            "138k arch) were retired — retrain, no conversion.")
    policy = NotePolicy(
        embed=arch["embed"], trunk_dim=arch["trunk_dim"],
        n_blocks=arch["n_blocks"], n_heads=arch["n_heads"],
        n_gears=arch["n_gears"], ffn_mult=arch["ffn_mult"])
    policy.load_state_dict(ckpt["state_dict"])
    return policy
