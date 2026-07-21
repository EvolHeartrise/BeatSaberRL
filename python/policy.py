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

No action masks exist in this env: every acceleration is always legal, so
logprob = logp_left + logp_right unconditionally (pure independent
factorization — no kind-gating like kiting).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

# Observation layout (contract; mirrors beat_arena_env/constants.py, obs v3:
# direction = (cos, sin) unit vector, continuous-angle ready).
OFF_NOTES = 8
N_NOTE_SLOTS = 12
NOTE_FEAT = 9
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT  # 116
OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS                # 128
N_ACTIONS = 9

NEG_INF = -1e9


def _mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class Block(nn.Module):
    """Full pre-LN transformer block, single-head."""

    def __init__(self, d: int):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.ln_ffn = nn.LayerNorm(d)
        self.ffn = _mlp([d, 2 * d, d])
        self.d = d

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d); key_mask: (B, T), 1 = attendable
        h = self.ln_attn(x)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        scores = q @ k.transpose(1, 2) / (self.d ** 0.5)      # (B, T, T)
        scores = scores + (key_mask.unsqueeze(1) - 1.0) * 1e9  # mask key columns
        x = x + torch.softmax(scores, dim=-1) @ v
        x = x + self.ffn(self.ln_ffn(x))
        return x


class NotePolicy(nn.Module):
    def __init__(self, embed: int = 64, trunk_dim: int = 128,
                 encoder: str = "transformer", n_blocks: int = 2):
        super().__init__()
        assert encoder == "transformer", (
            f"unknown encoder: {encoder} (pool was removed 2026-07-20)")
        self.encoder = encoder
        self.embed = embed
        self.note_mlp = _mlp([NOTE_FEAT, embed, embed])
        # saber token features: [x, y, vx, vy, is_left, is_right]
        self.saber_mlp = _mlp([6, embed, embed])
        self.blocks = nn.ModuleList(Block(embed) for _ in range(n_blocks))
        self.trunk = _mlp([2 * embed, trunk_dim, trunk_dim])
        self.left_head = nn.Linear(trunk_dim, N_ACTIONS)
        self.right_head = nn.Linear(trunk_dim, N_ACTIONS)
        self.value_head = nn.Linear(trunk_dim, 1)

    def _split(self, obs: torch.Tensor):
        sabers = obs[:, :OFF_NOTES]                                  # (B, 8)
        notes = obs[:, OFF_NOTES:OFF_NOTE_MASK].reshape(
            -1, N_NOTE_SLOTS, NOTE_FEAT)                             # (B, 12, 14)
        note_mask = obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS]
        return sabers, notes, note_mask

    def forward(self, obs: torch.Tensor):
        sabers, notes, note_mask = self._split(obs)
        emb = self.note_mlp(notes)                                   # (B, 12, 64)
        B = obs.shape[0]
        hand_id = torch.eye(2, device=obs.device).expand(B, 2, 2)
        saber_feat = torch.cat([sabers.reshape(B, 2, 4), hand_id], dim=-1)
        tokens = torch.cat([self.saber_mlp(saber_feat), emb], dim=1)  # (B, 14, 64)
        key_mask = torch.cat(
            [torch.ones(B, 2, device=obs.device), note_mask], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens, key_mask)
        h = self.trunk(tokens[:, :2].reshape(B, -1))                 # saber readout
        return (self.left_head(h), self.right_head(h),
                self.value_head(h).squeeze(-1))

    def get_action_and_value(self, obs, action=None):
        left_l, right_l, value = self.forward(obs)
        d_left, d_right = Categorical(logits=left_l), Categorical(logits=right_l)
        if action is None:
            action = torch.stack([d_left.sample(), d_right.sample()], dim=1)
        logprob = d_left.log_prob(action[:, 0]) + d_right.log_prob(action[:, 1])
        entropy = d_left.entropy() + d_right.entropy()
        return action, logprob, entropy, value
