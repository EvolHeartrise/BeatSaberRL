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
from torch.distributions import Categorical

# Observation layout (contract; mirrors beat_arena_env/constants.py, obs v5:
# env v6 arm chains — 23 features per hand; notes unchanged from v3).
SABER_FEAT = 23
OFF_NOTES = 2 * SABER_FEAT                            # 46
N_NOTE_SLOTS = 12
NOTE_FEAT = 9
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT  # 154
OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS                # 166
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

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d); key_mask: (B, T), 1 = attendable
        B, T, _ = x.shape
        h = self.ln_attn(x)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        if self.n_heads == 1:
            scores = q @ k.transpose(1, 2) / (self.d ** 0.5)      # (B, T, T)
            scores = scores + (key_mask.unsqueeze(1) - 1.0) * 1e9
            x = x + torch.softmax(scores, dim=-1) @ v
        else:
            q = q.view(B, T, self.n_heads, self.dh).transpose(1, 2)  # (B,H,T,dh)
            k = k.view(B, T, self.n_heads, self.dh).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.dh).transpose(1, 2)
            scores = q @ k.transpose(-2, -1) / (self.dh ** 0.5)      # (B,H,T,T)
            scores = scores + (key_mask[:, None, None, :] - 1.0) * 1e9
            att = torch.softmax(scores, dim=-1) @ v                  # (B,H,T,dh)
            att = att.transpose(1, 2).reshape(B, T, self.d)
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
        sabers = obs[:, :OFF_NOTES]                                  # (B, 20)
        notes = obs[:, OFF_NOTES:OFF_NOTE_MASK].reshape(
            -1, N_NOTE_SLOTS, NOTE_FEAT)                             # (B, 12, 9)
        note_mask = obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS]
        return sabers, notes, note_mask

    def forward(self, obs: torch.Tensor):
        sabers, notes, note_mask = self._split(obs)
        emb = self.note_mlp(notes)                                   # (B, 12, 64)
        B = obs.shape[0]
        hand_id = torch.eye(2, device=obs.device).expand(B, 2, 2)
        saber_feat = torch.cat(
            [sabers.reshape(B, 2, SABER_FEAT), hand_id], dim=-1)
        tokens = torch.cat([self.saber_mlp(saber_feat), emb], dim=1)  # (B, 14, 64)
        key_mask = torch.cat(
            [torch.ones(B, 2, device=obs.device), note_mask], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens, key_mask)
        h = self.trunk(tokens[:, :2].reshape(B, -1))                 # saber readout
        return [head(h) for head in self.heads], self.value_head(h).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dists = [Categorical(logits=l) for l in logits]
        if action is None:
            action = torch.stack([d.sample() for d in dists], dim=1)
        logprob = sum(d.log_prob(action[:, i]) for i, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
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
