"""Note-sequence policy for Beat Arena (SPEC.md §Policy).

Two encoder arms over the 12-slot upcoming-note window:

- ``transformer`` (the only encoder since 2026-07-20 — the pool
  baseline was deleted at user direction after the easy-config studies
  concluded): the notes, two saber tokens AND (env v10) one body token
  form a 19-token sequence run through 2 full pre-LN transformer blocks
  (LayerNorm -> single-head self-attention -> residual; LayerNorm ->
  FFN -> residual).
  Each note token carries its z position along the approach axis (obs v2)
  — with constant note speed that IS the temporal ordering, expressed
  spatially (time_until = z / |vz|), so no separate positional encoding
  is needed. Readout = the two saber tokens' output embeddings, i.e. each
  hand's token gathers what it needs from the note stream via attention
  (a CLS-token-style readout, one per hand).

Unlike kiting-arena's deliberately minimal attention arm, the transformer
here gets its standard optimization aids (LayerNorm, FFN, residuals) —
that A/B's live hypothesis (b) was that stripping them handicaps training.

No action masks exist in this env: every action is always legal, so the
logprob is the unconditional sum over the 29 factored heads (env v10
full-body gear control, SPEC-v10 §1.4 joint order; pure independent
factorization, no kind-gating like kiting).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import fp16_linear as _fl
from fp16_linear import fp16_linear, fp16_linear_nobias, fp16_linear_relu
from fused_attn import eligible as _fused_attn_ok, fused_attn as _fused_attn
from fused_ln import FusedLayerNorm

# Fused single-head attention on the fp16 update path (round-4 #15,
# default ON since 2026-07-25: 1.76x on the attention slice fwd+bwd,
# grads closer to fp64 than eager's on every input). Module-level switch
# so experiment arms can pin it; near-tie numerics class (never touches
# the fp32 record path). Env toggles (debug/bisect):
#   BEAT_NO_FUSED_ATTN=1  -> eager attention (also disables qkv bias
#                            folding, which rides the fused kernel)
#   BEAT_NO_BIAS_RELU=1   -> eager addmm+ReLU in _seq
#   BEAT_NO_BIAS_FOLD=1   -> ffn.2 keeps its addmm bias epilogue
#                            instead of folding into the next block's
#                            fused residual+LN pass
import os as _os
USE_FUSED_ATTN = _os.environ.get("BEAT_NO_FUSED_ATTN", "0") != "1"
_NO_BIAS_RELU = _os.environ.get("BEAT_NO_BIAS_RELU", "0") == "1"
_NO_BIAS_FOLD = _os.environ.get("BEAT_NO_BIAS_FOLD", "0") == "1"


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
# obs v10 = SPEC-v10 §5: 340 lanes — root pose/vel (15), q block (31,
# wrist rolls cos/sin), qd (29), two 10-lane derived hand blocks, head
# (3), contact bits (2), then the v9-verbatim 16x14 note block + mask).
OBS_DIM = 340
N_NOTE_SLOTS = 16
NOTE_FEAT = 14
OFF_NOTES = 100
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT  # 324
# Token layout (SPEC-v10 §5 "informative", finalized here at P4):
# [saber_L, saber_R, body, note_0..15] = 19 tokens. Each saber token =
# that hand's q lanes (8, incl. the cos/sin roll pair) + qd lanes (7) +
# derived block (10) = 25 features + the 2-lane hand one-hot; the ONE
# body token = root pose/vel block (15) + spine/leg q (15) + spine/leg
# qd (15) + head (3) + contact bits (2) = 50 features.
SABER_FEAT = 25
BODY_FEAT = 50
_SABER_L_IDX = tuple(range(18, 26)) + tuple(range(49, 56)) + tuple(range(75, 85))
_SABER_R_IDX = tuple(range(26, 34)) + tuple(range(56, 63)) + tuple(range(85, 95))
_BODY_IDX = (tuple(range(0, 18)) + tuple(range(34, 49))
             + tuple(range(63, 75)) + tuple(range(95, 100)))
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

    def __init__(self, d: int, n_heads: int = 1):
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
        self.ffn = _mlp([d, 2 * d, d])
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
        # fused-attn eligibility decides the PROJECTION form too: the
        # fused path runs bias-free mms and folds the q/k/v biases into
        # the attention kernel's loads (round-4 #2: the sm_61 addmm
        # bias epilogue costs 33-59% of each GEMM)
        use_fused = (USE_FUSED_ATTN and self.n_heads == 1
                     and h.dtype is torch.float16
                     and _fused_attn_ok(h, attn_bias))
        if use_fused:
            k = fp16_linear_nobias(h, self.k_proj.weight, self.k_proj.bias)
            v = fp16_linear_nobias(h, self.v_proj.weight, self.v_proj.bias)
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
        q = (fp16_linear_nobias(h, self.q_proj.weight, self.q_proj.bias)
             if use_fused else _lin(self.q_proj, h))
        Tq = q.shape[1]
        if attn_bias is None:
            attn_bias = (key_mask.unsqueeze(1) - 1.0) * MASK_BIAS  # (B, 1, T)
        if self.n_heads == 1:
            if use_fused:
                # fp16 path: one fused kernel per direction (round-4
                # #15) — the scores tensor never touches global memory.
                # Near-tie tolerance class, so fp16-only: the fp32
                # update path below stays bitwise for the record runs.
                att = _fused_attn(q, k, v, attn_bias, self.d ** -0.5,
                                  self.q_proj.bias, self.k_proj.bias,
                                  self.v_proj.bias)
            else:
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
            scores = q @ k.transpose(-2, -1) / (self.dh ** 0.5)   # (B,H,Tq,T)
            scores = scores + attn_bias.unsqueeze(1)              # (B,1,1,T)
            att = torch.softmax(scores, dim=-1) @ v               # (B,H,Tq,dh)
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
    def __init__(self, embed: int = 64, trunk_dim: int = 128,
                 encoder: str = "transformer", n_blocks: int = 2,
                 n_heads: int = 1, n_gears: int = 3):
        super().__init__()
        assert encoder == "transformer", (
            f"unknown encoder: {encoder} (pool was removed 2026-07-20)")
        assert n_gears in (3, 5, 7), f"n_gears must be 3/5/7, got {n_gears}"
        self.encoder = encoder
        self.embed = embed
        self.n_gears = n_gears
        self.head_sizes = (n_gears,) * N_ACTION_HEADS
        # v10 arch record: obs 340, 29 heads, body token — load_policy
        # REFUSES checkpoints without these markers (v9 and earlier).
        self.arch = {"embed": embed, "trunk_dim": trunk_dim,
                     "n_blocks": n_blocks, "n_heads": n_heads,
                     "n_gears": n_gears, "obs_dim": OBS_DIM,
                     "n_action_heads": N_ACTION_HEADS, "body_token": True}
        self.note_mlp = _mlp([NOTE_FEAT, embed, embed])
        # saber token features: the 25 obs-v10 hand features + [is_left,
        # is_right]; the body token gets its own encoder
        self.saber_mlp = _mlp([SABER_FEAT + 2, embed, embed])
        self.body_mlp = _mlp([BODY_FEAT, embed, embed])
        self.blocks = nn.ModuleList(
            Block(embed, n_heads) for _ in range(n_blocks))
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
            -1, N_NOTE_SLOTS, NOTE_FEAT)                             # (B, 16, 14)
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
        w16, b16 = _fl.flatten_fp16_replicas(self, tail_group=tail)
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
            or not arch.get("body_token", False)):
        raise SystemExit(
            "checkpoint is not env v10 (needs arch obs_dim=340, "
            f"n_action_heads=29, body_token=True; got {dict(arch)}). "
            "v9 and earlier checkpoints were retired with the v10 body — "
            "retrain, there is no conversion.")
    policy = NotePolicy(
        embed=arch.get("embed", 64), trunk_dim=arch.get("trunk_dim", 128),
        n_blocks=arch.get("n_blocks", 2), n_heads=arch.get("n_heads", 1),
        n_gears=arch.get("n_gears", 3))
    policy.load_state_dict(ckpt["state_dict"])
    return policy
