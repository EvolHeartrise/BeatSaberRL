"""Note-sequence policy for Beat Arena (SPEC.md §Policy).

Two encoder arms over the 12-slot upcoming-note window:

- ``transformer`` (the only encoder since 2026-07-20 — the pool
  baseline was deleted at user direction after the easy-config studies
  concluded): the notes AND two saber tokens form a 14-token
  sequence run through 2 full pre-LN transformer blocks (LayerNorm ->
  single-head self-attention -> residual; LayerNorm -> FFN -> residual).
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
logprob is the unconditional sum over the twelve factored heads (per
hand: pitch, yaw, twist, elbow, roll, bend — env v6 joint control, all
ternary; pure independent factorization, no kind-gating like kiting).
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
import os as _os
USE_FUSED_ATTN = _os.environ.get("BEAT_NO_FUSED_ATTN", "0") != "1"
_NO_BIAS_RELU = _os.environ.get("BEAT_NO_BIAS_RELU", "0") == "1"


def _lin(m: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Linear that follows the stream dtype: fp16 activations route to
    the fp16-storage path (half addmm forward, GemmEx fp32-accumulated
    weight grads against the fp32 masters); fp32 runs the module as-is."""
    if x.dtype is torch.float16:
        return fp16_linear(x, m.weight, m.bias)
    return m(x)


def _seq(seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    ms = list(seq)
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

# Observation layout (contract; mirrors beat_arena_env/constants.py, obs
# v9: env v7's 16 object slots x 14 features incl. kind flags and arc
# grants; saber block gains the v9 wrist-deviation angle + omega -> 25).
SABER_FEAT = 25
OFF_NOTES = 2 * SABER_FEAT                            # 50
N_NOTE_SLOTS = 16
NOTE_FEAT = 14
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT  # 274
OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS                # 290
# env v6: per hand 6 joint heads (pitch, yaw, twist, elbow, roll, bend)
# — twelve factored heads, left hand first. Env v8: heads have K =
# n_gears options (3/5/7); this module constant is the K=3 DEFAULT
# (legacy imports use it for the head count / ternary-era shapes) —
# a policy's real sizes live in policy.head_sizes / arch["n_gears"].
# Env v9: 14 heads (7 joints/hand — wrist deviation added).
HEAD_SIZES = (3,) * 14

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
                delta: torch.Tensor | None = None):
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
        # (x_without_ffn, ffn_out) for the next block's delta; the final
        # block (n_queries > 0) returns the resolved output rows.
        B, T, _ = x.shape
        if delta is not None:
            x, h = self.ln_attn.forward_add(x, delta)
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
            x = x[:, :n_queries]
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
        ffn_out = _seq(self.ffn, h2)
        if n_queries:
            return x + ffn_out
        return x, ffn_out


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
        self.head_sizes = (n_gears,) * 14
        self.arch = {"embed": embed, "trunk_dim": trunk_dim,
                     "n_blocks": n_blocks, "n_heads": n_heads,
                     "n_gears": n_gears}
        self.note_mlp = _mlp([NOTE_FEAT, embed, embed])
        # saber token features: the 10 obs-v4 features + [is_left, is_right]
        self.saber_mlp = _mlp([SABER_FEAT + 2, embed, embed])
        self.blocks = nn.ModuleList(
            Block(embed, n_heads) for _ in range(n_blocks))
        self.trunk = _mlp([2 * embed, trunk_dim, trunk_dim])
        # 12 factored joint heads, K = n_gears options each (env v8)
        self.heads = nn.ModuleList(
            nn.Linear(trunk_dim, n) for n in self.head_sizes)
        self.value_head = nn.Linear(trunk_dim, 1)
        # fp16-storage mode (2026-07-25): activations flow fp16 between
        # the obs split and the head output; GEMMs are fp16-in with fp32
        # cuBLAS compute; LN/softmax stats, log_softmax, sampling, and
        # all PPO math stay fp32. Weights stay fp32 masters (state_dict
        # unchanged). Plain attribute, NOT in state_dict.
        self.fp16 = False

    def _split(self, obs: torch.Tensor):
        sabers = obs[:, :OFF_NOTES]                                  # (B, 46)
        notes = obs[:, OFF_NOTES:OFF_NOTE_MASK].reshape(
            -1, N_NOTE_SLOTS, NOTE_FEAT)                             # (B, 16, 14)
        note_mask = obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS]
        return sabers, notes, note_mask

    def forward_stacked(self, obs: torch.Tensor):
        """Trunk + ONE fused (trunk_dim -> 37) head GEMM: stacked logits
        (B, 12, 3) + value (B,). The 12 per-head Linears + value head
        stay separate modules (state_dict unchanged); their weights are
        concatenated per call (13 tiny tensors — negligible next to the
        13 skinny GEMMs and ~120 Categorical kernels this replaces)."""
        fp16 = self.fp16 and obs.is_cuda
        sabers, notes, note_mask = self._split(obs)
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
        tokens = torch.cat([_seq(self.saber_mlp, saber_feat), emb], dim=1)
        # additive attention mask is block-invariant: build it once.
        # The 2 saber lanes are always attendable -> bias exactly +0.0,
        # which is what F.pad's zero fill writes — so the ones+cat
        # key_mask build is skipped entirely (blocks only consume
        # key_mask when attn_bias is None; bit-identical, 2026-07-28)
        attn_bias = F.pad((note_mask.unsqueeze(1) - 1.0) * MASK_BIAS,
                          (2, 0))
        if fp16:
            attn_bias = attn_bias.half()  # C3: 2^14 is fp16-exact
        delta = None
        for blk in self.blocks[:-1]:
            tokens, delta = blk(tokens, None, attn_bias=attn_bias,
                                delta=delta)
        # final block: only the 2 readout tokens' outputs are consumed
        tokens = self.blocks[-1](tokens, None, n_queries=2,
                                 attn_bias=attn_bias, delta=delta)
        h = _seq(self.trunk, tokens.reshape(B, -1))                  # saber readout
        if fp16 and not torch.is_grad_enabled():
            # rollout fast path: weights are frozen between opt.steps,
            # so the 13-tensor cat + fp16 cast is cached per weight
            # EPOCH (fp16_linear.bump_weights_version — tensor _version
            # is not bumped by fused Adam) instead of rebuilt 128 times
            c = self._refresh_head_cache()
            out = F.linear(h, c[1], c[2]).float()
            return out[:, :-1].view(B, 14, self.n_gears), out[:, -1]
        w = torch.cat([hd.weight for hd in self.heads]
                      + [self.value_head.weight])
        b = torch.cat([hd.bias for hd in self.heads]
                      + [self.value_head.bias])
        if fp16:
            # C5: the hard fp32 boundary — everything after (log_softmax,
            # sampling, PPO math) is exactly today's fp32 code
            out = fp16_linear(h, w, b).float()
        else:
            out = F.linear(h, w, b)                          # (B, 12K + 1)
        return out[:, :-1].view(B, 14, self.n_gears), out[:, -1]

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
    """Rebuild a NotePolicy from a checkpoint dict. Checkpoints since
    2026-07-21 carry an "arch" dict; older ones fall back to the original
    d64/1-head/2-block defaults."""
    arch = ckpt.get("arch", {})
    policy = NotePolicy(
        embed=arch.get("embed", 64), trunk_dim=arch.get("trunk_dim", 128),
        n_blocks=arch.get("n_blocks", 2), n_heads=arch.get("n_heads", 1),
        n_gears=arch.get("n_gears", 3))
    policy.load_state_dict(ckpt["state_dict"])
    return policy
