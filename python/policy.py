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

# Observation layout (contract; mirrors beat_arena_env/constants.py, obs v6:
# env v7 — 16 object slots x 14 features incl. kind flags (bomb, chain
# head/link) and arc pre/post-swing grants; sabers unchanged from v6).
SABER_FEAT = 23
OFF_NOTES = 2 * SABER_FEAT                            # 46
N_NOTE_SLOTS = 16
NOTE_FEAT = 14
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT  # 270
OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS                # 286
# env v6: per hand 6 ternary joint heads (pitch, yaw, twist, elbow, roll,
# bend) — twelve factored heads, left hand first
HEAD_SIZES = (3,) * 12

NEG_INF = -1e9


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
        self.ln_attn = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d) if n_heads > 1 else None
        self.ln_ffn = nn.LayerNorm(d)
        self.ffn = _mlp([d, 2 * d, d])
        self.d = d
        self.n_heads = n_heads
        self.dh = d // n_heads

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor,
                n_queries: int = 0) -> torch.Tensor:
        # x: (B, T, d); key_mask: (B, T), 1 = attendable
        #
        # n_queries > 0 computes the block's output for only the first
        # n_queries tokens (keys/values still span all T). Because the
        # attention mask is applied on the KEY dim only, token i's output
        # never depends on any other token's q/FFN path — so this is
        # EXACTLY the full block restricted to those rows, not an
        # approximation. Used by the readout (tokens[:, :2]) to skip the
        # 16 note tokens' final-block compute (~29% of forward MACs).
        B, T, _ = x.shape
        h = self.ln_attn(x)
        k, v = self.k_proj(h), self.v_proj(h)
        if n_queries:
            x = x[:, :n_queries]
            h = h[:, :n_queries]
        q = self.q_proj(h)
        Tq = q.shape[1]
        if self.n_heads == 1:
            scores = q @ k.transpose(1, 2) / (self.d ** 0.5)      # (B, Tq, T)
            scores = scores + (key_mask.unsqueeze(1) - 1.0) * 1e9
            x = x + torch.softmax(scores, dim=-1) @ v
        else:
            q = q.view(B, Tq, self.n_heads, self.dh).transpose(1, 2)
            k = k.view(B, T, self.n_heads, self.dh).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.dh).transpose(1, 2)
            scores = q @ k.transpose(-2, -1) / (self.dh ** 0.5)   # (B,H,Tq,T)
            scores = scores + (key_mask[:, None, None, :] - 1.0) * 1e9
            att = torch.softmax(scores, dim=-1) @ v               # (B,H,Tq,dh)
            att = att.transpose(1, 2).reshape(B, Tq, self.d)
            x = x + self.out_proj(att)
        x = x + self.ffn(self.ln_ffn(x))
        return x


class NotePolicy(nn.Module):
    def __init__(self, embed: int = 64, trunk_dim: int = 128,
                 encoder: str = "transformer", n_blocks: int = 2,
                 n_heads: int = 1):
        super().__init__()
        assert encoder == "transformer", (
            f"unknown encoder: {encoder} (pool was removed 2026-07-20)")
        self.encoder = encoder
        self.embed = embed
        self.arch = {"embed": embed, "trunk_dim": trunk_dim,
                     "n_blocks": n_blocks, "n_heads": n_heads}
        self.note_mlp = _mlp([NOTE_FEAT, embed, embed])
        # saber token features: the 10 obs-v4 features + [is_left, is_right]
        self.saber_mlp = _mlp([SABER_FEAT + 2, embed, embed])
        self.blocks = nn.ModuleList(
            Block(embed, n_heads) for _ in range(n_blocks))
        self.trunk = _mlp([2 * embed, trunk_dim, trunk_dim])
        # six factored heads (env v5): per hand move(9), rot(3), tilt(3)
        self.heads = nn.ModuleList(
            nn.Linear(trunk_dim, n) for n in HEAD_SIZES)
        self.value_head = nn.Linear(trunk_dim, 1)

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
        sabers, notes, note_mask = self._split(obs)
        emb = self.note_mlp(notes)                                   # (B, 16, 64)
        B = obs.shape[0]
        hand_id = torch.eye(2, device=obs.device).expand(B, 2, 2)
        saber_feat = torch.cat(
            [sabers.reshape(B, 2, SABER_FEAT), hand_id], dim=-1)
        tokens = torch.cat([self.saber_mlp(saber_feat), emb], dim=1)  # (B, 18, 64)
        key_mask = torch.cat(
            [torch.ones(B, 2, device=obs.device), note_mask], dim=1)
        for blk in self.blocks[:-1]:
            tokens = blk(tokens, key_mask)
        # final block: only the 2 readout tokens' outputs are consumed
        tokens = self.blocks[-1](tokens, key_mask, n_queries=2)
        h = self.trunk(tokens.reshape(B, -1))                        # saber readout
        w = torch.cat([hd.weight for hd in self.heads]
                      + [self.value_head.weight])
        b = torch.cat([hd.bias for hd in self.heads]
                      + [self.value_head.bias])
        out = F.linear(h, w, b)                                      # (B, 37)
        return out[:, :-1].view(B, len(HEAD_SIZES), 3), out[:, -1]

    def forward(self, obs: torch.Tensor):
        logits, value = self.forward_stacked(obs)
        return list(logits.unbind(1)), value

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward_stacked(obs)                # (B, 12, 3)
        logsm = torch.log_softmax(logits, dim=-1)
        if action is None:
            # one multinomial over all B*12 rows — same distribution as 12
            # per-head Categorical.sample()s, different RNG STREAM (the
            # same caveat the graphed-rollout capture carried)
            action = torch.multinomial(
                logsm.exp().reshape(-1, 3), 1).view(-1, len(HEAD_SIZES))
        logprob = logsm.gather(-1, action.unsqueeze(-1)).squeeze(-1).sum(-1)
        entropy = -(logsm.exp() * logsm).sum(-1).sum(-1)
        return action, logprob, entropy, value


def load_policy(ckpt) -> NotePolicy:
    """Rebuild a NotePolicy from a checkpoint dict. Checkpoints since
    2026-07-21 carry an "arch" dict; older ones fall back to the original
    d64/1-head/2-block defaults."""
    arch = ckpt.get("arch", {})
    policy = NotePolicy(
        embed=arch.get("embed", 64), trunk_dim=arch.get("trunk_dim", 128),
        n_blocks=arch.get("n_blocks", 2), n_heads=arch.get("n_heads", 1))
    policy.load_state_dict(ckpt["state_dict"])
    return policy
