"""Fused CUDA inference for NotePolicy (csrc/policy_forward.cu).

The whole forward — note/saber MLPs, both pre-LN transformer blocks
(masked single-head attention, FFN, residuals), trunk, 12 action heads +
value head, and the per-head argmax — runs as ONE kernel launch, one
thread block per sample, activations in shared memory. Inference only
(no autograd), fp32, arch fixed at embed=64 / trunk_dim=128 / n_heads=1
(the only production arch); n_blocks is free. Other archs raise.

Usage::

    fused = FusedNotePolicy(policy_or_state_dict)   # packs weights once
    logits, value = fused.fused_forward(obs)        # (N, 12, 3), (N,)
    actions = fused.argmax(obs)                     # (N, 12) int64
    fused.repack(policy)                            # after weight updates

Output tensors are preallocated per batch size and REUSED across calls
(clone if you need to keep them past the next call) — this also makes
the call CUDA-graph-capturable, though it hardly matters: it is a
single launch already.

Built like beat_arena_env/cuda_env.py: torch.utils.cpp_extension.load
with the same Windows toolchain discovery, ninja-cached under
csrc/build_policy (its own dir — load() writes one build.ninja per dir),
-arch=sm_61.
"""

from __future__ import annotations

import os

import torch

from beat_arena_env.cuda_env import _toolchain_env
from policy import HEAD_SIZES, NOTE_FEAT, OBS_DIM, SABER_FEAT

EMBED = 64
TRUNK_DIM = 128
N_ACTION_HEADS = len(HEAD_SIZES)      # 12, all ternary
N_PARAMS_2BLK = 107365                # layout checksum for the default arch

_ext = None


def _load_ext():
    """Compile-or-load the cached extension (warm cache loads in ~1 s)."""
    global _ext
    if _ext is None:
        _toolchain_env()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "beat_arena_env", "csrc")
        bdir = os.path.join(src, "build_policy")
        os.makedirs(bdir, exist_ok=True)
        _ext = load(name="policy_forward_cuda",
                    sources=[os.path.join(src, "policy_forward.cu")],
                    build_directory=bdir,
                    extra_cuda_cflags=["-arch=sm_61"])
    return _ext


def _state_dict(source):
    if isinstance(source, torch.nn.Module):
        return source.state_dict()
    return source


def _validate(sd) -> int:
    """Check the arch the kernel hard-codes; return n_blocks."""
    def shape(k):
        return tuple(sd[k].shape)
    if shape("note_mlp.0.weight") != (EMBED, NOTE_FEAT):
        raise NotImplementedError(
            f"fused kernel supports embed={EMBED} only, got "
            f"note_mlp.0.weight {shape('note_mlp.0.weight')}")
    if shape("saber_mlp.0.weight") != (EMBED, SABER_FEAT + 2):
        raise NotImplementedError(
            f"unexpected saber_mlp.0.weight {shape('saber_mlp.0.weight')}")
    if shape("trunk.0.weight") != (TRUNK_DIM, 2 * EMBED):
        raise NotImplementedError(
            f"fused kernel supports trunk_dim={TRUNK_DIM} only, got "
            f"trunk.0.weight {shape('trunk.0.weight')}")
    if any("out_proj" in k for k in sd):
        raise NotImplementedError(
            "fused kernel supports n_heads=1 only (found out_proj weights)")
    n_blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    if n_blocks == 0:
        raise NotImplementedError("no transformer blocks in state_dict")
    for b in range(n_blocks):
        if shape(f"blocks.{b}.ffn.0.weight") != (2 * EMBED, EMBED):
            raise NotImplementedError(
                f"unexpected ffn shape in block {b}: "
                f"{shape(f'blocks.{b}.ffn.0.weight')}")
    if len(sd["heads.0.bias"]) != 3 or not all(
            tuple(sd[f"heads.{i}.weight"].shape) == (3, TRUNK_DIM)
            for i in range(N_ACTION_HEADS)):
        raise NotImplementedError("unexpected head shapes")
    return n_blocks


def _pack(sd, n_blocks) -> torch.Tensor:
    """Flatten the state_dict into the kernel's packed layout: every
    matrix transposed to [in][out] (so the output-owning thread reads
    coalesced), biases inline after their matrix, 12 heads + value head
    fused into one (128 -> 37) matrix. Order is a contract with the
    constexpr offsets in policy_forward.cu."""
    parts = []

    def t(name):
        return sd[name].detach().float().cpu()

    def lin(prefix):
        parts.append(t(prefix + ".weight").t().contiguous().reshape(-1))
        parts.append(t(prefix + ".bias").reshape(-1))

    lin("note_mlp.0")
    lin("note_mlp.2")
    lin("saber_mlp.0")
    lin("saber_mlp.2")
    for b in range(n_blocks):
        p = f"blocks.{b}."
        parts += [t(p + "ln_attn.weight"), t(p + "ln_attn.bias")]
        lin(p + "q_proj")
        lin(p + "k_proj")
        lin(p + "v_proj")
        parts += [t(p + "ln_ffn.weight"), t(p + "ln_ffn.bias")]
        lin(p + "ffn.0")
        lin(p + "ffn.2")
    lin("trunk.0")
    lin("trunk.2")
    hw = torch.cat([t(f"heads.{i}.weight") for i in range(N_ACTION_HEADS)]
                   + [t("value_head.weight")], dim=0)      # (37, 128)
    parts.append(hw.t().contiguous().reshape(-1))
    parts.append(torch.cat(
        [t(f"heads.{i}.bias") for i in range(N_ACTION_HEADS)]
        + [t("value_head.bias")]))
    flat = torch.cat(parts)
    if n_blocks == 2:
        assert flat.numel() == N_PARAMS_2BLK, flat.numel()
    return flat


class FusedNotePolicy:
    """Packs a NotePolicy's weights once and serves single-launch
    forwards. Accepts a NotePolicy module or its state_dict."""

    def __init__(self, source, device="cuda"):
        sd = _state_dict(source)
        self.n_blocks = _validate(sd)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("FusedNotePolicy requires a CUDA device")
        self._ext = _load_ext()
        self.weights = _pack(sd, self.n_blocks).to(self.device)
        self._bufs = {}

    def repack(self, source):
        """Re-upload weights in place (same buffer — pointers, and any
        capture of the call, stay valid). Arch must match."""
        sd = _state_dict(source)
        if _validate(sd) != self.n_blocks:
            raise ValueError("repack with a different n_blocks")
        self.weights.copy_(_pack(sd, self.n_blocks).to(self.device))

    def _run(self, obs):
        if not (torch.is_tensor(obs) and obs.is_cuda):
            obs = torch.as_tensor(obs, device=self.device)
        if obs.dtype != torch.float32:
            obs = obs.float()
        obs = obs.contiguous()
        n = obs.shape[0]
        bufs = self._bufs.get(n)
        if bufs is None:
            bufs = (torch.empty(n, N_ACTION_HEADS, 3, device=self.device),
                    torch.empty(n, device=self.device),
                    torch.empty(n, N_ACTION_HEADS, device=self.device,
                                dtype=torch.int64))
            self._bufs[n] = bufs
        self._ext.forward(self.weights, obs, bufs[0], bufs[1], bufs[2],
                          self.n_blocks)
        return bufs

    def fused_forward(self, obs):
        """logits (N, 12, 3) — torch.stack(policy(obs)[0], dim=1) layout —
        and value (N,). Buffers are reused across calls."""
        logits, value, _ = self._run(obs)
        return logits, value

    __call__ = fused_forward

    def argmax(self, obs):
        """Greedy actions (N, 12) int64, computed inside the same kernel."""
        return self._run(obs)[2]
