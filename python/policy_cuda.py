"""Fused CUDA inference AND training path for NotePolicy — env v10
(csrc/policy_forward.cu + csrc/policy_backward.cu).

Inference (`FusedNotePolicy`): the whole forward — note/saber/body MLPs,
both pre-LN transformer blocks (masked single-head attention, FFN,
residuals), trunk (192 -> 128 over the 3-token readout), the fused
29xK-head + value GEMM, and the per-head argmax — runs as ONE kernel
launch, one thread block per sample, activations in shared memory. No
autograd, fp32, arch fixed at embed=64 / trunk_dim=128 / n_heads=1 (the
production arch); n_blocks and n_gears (K in 3/5/7) are runtime.
Other archs raise.

Training (`FusedPolicyTrainer` + `FusedPolicyFn`): the packed flat
weight tensor is itself the single trainable leaf. FusedPolicyFn.apply
runs the fused forward with grad tracking; its backward hands
(grad_logits, grad_value) — produced by the PPO loss, which stays in
plain torch on the tiny (mb,29,K)/(mb,) outputs — to the fused backward
kernel, which RECOMPUTES the forward per sample (saving only the obs
minibatch — the 2026-07-25 d128 OOM lesson) and accumulates every
parameter gradient into the packed grad via striped atomics + a reduce.
fp32 atomic accumulation reassociates across rows, so gradients carry
fp32-level run-to-run jitter (quantified vs torch fp32 and a float64
reference in tests/test_policy_backward.py; ncopies >= mb makes the
backward fully deterministic).

Usage::

    fused = FusedNotePolicy(policy_or_state_dict)   # packs weights once
    logits, value = fused.fused_forward(obs)        # (N, 29, K), (N,)
    actions = fused.argmax(obs)                     # (N, 29) int64
    fused.repack(policy)                            # after weight updates

    trainer = FusedPolicyTrainer(policy)            # training path
    opt = torch.optim.Adam(trainer.parameters(), lr=...)
    logits, value = trainer(obs_minibatch)          # autograd-tracked
    loss.backward()                                 # fills packed .grad
    opt.step(); trainer.unpack_to(policy)           # sync for rollout

Inference output tensors are preallocated per batch size and REUSED
across calls (clone if you need to keep them past the next call); the
training path allocates per call (tiny, and autograd owns them). Both
are CUDA-graph-capturable.

Built like beat_arena_env/cuda_env.py: torch.utils.cpp_extension.load
with the same Windows toolchain discovery, ninja-cached under
csrc/build_policy and csrc/build_policy_bwd, -arch from
jit_cache.cuda_flags() (the current device's compute capability).
"""

from __future__ import annotations

import os

import torch

from beat_arena_env.cuda_env import _toolchain_env
from policy import (
    BODY_FEAT, N_ACTION_HEADS, NOTE_FEAT, OBS_DIM, SABER_FEAT,
)

EMBED = 64
TRUNK_DIM = 128
# v10 packed-layout section sizes (contract with the constexpr offsets
# in the .cu files)
_FRONT = (NOTE_FEAT * EMBED + EMBED + EMBED * EMBED + EMBED          # note
          + (SABER_FEAT + 2) * EMBED + EMBED + EMBED * EMBED + EMBED  # saber
          + BODY_FEAT * EMBED + EMBED + EMBED * EMBED + EMBED)        # body
_BLK = (2 * EMBED + 3 * (EMBED * EMBED + EMBED) + 2 * EMBED
        + EMBED * (2 * EMBED) + 2 * EMBED + (2 * EMBED) * EMBED + EMBED)
_TRUNK_SZ = (3 * EMBED * TRUNK_DIM + TRUNK_DIM
             + TRUNK_DIM * TRUNK_DIM + TRUNK_DIM)
assert _FRONT == 18752 and _BLK == 29312 and _TRUNK_SZ == 41216  # v12


def expected_numel(n_blocks, n_gears):
    nout = N_ACTION_HEADS * n_gears + 1
    return _FRONT + n_blocks * _BLK + _TRUNK_SZ + (TRUNK_DIM + 1) * nout


_ext = None
_ext_bwd = None


def _load_one(name, src_file, build_subdir):
    from beat_arena_env.jit_cache import arch_list, cuda_flags, fast_load
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "beat_arena_env", "csrc")
    bdir = os.path.join(src, build_subdir)
    os.makedirs(bdir, exist_ok=True)
    srcs = [os.path.join(src, src_file)]

    def _full():
        _toolchain_env()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list())
        from torch.utils.cpp_extension import load
        return load(name=name, sources=srcs, build_directory=bdir,
                    extra_cuda_cflags=cuda_flags(), verbose=True)

    return fast_load(name, srcs, bdir, cuda_flags(), _full)


def _load_ext():
    """Compile-or-load the cached forward extension (~1 s warm)."""
    global _ext
    if _ext is None:
        _ext = _load_one("policy_forward_cuda", "policy_forward.cu",
                         "build_policy")
    return _ext


def _load_bwd_ext():
    """Compile-or-load the cached backward extension (~1 s warm)."""
    global _ext_bwd
    if _ext_bwd is None:
        _ext_bwd = _load_one("policy_backward_cuda", "policy_backward.cu",
                             "build_policy_bwd")
    return _ext_bwd


def _state_dict(source):
    if isinstance(source, torch.nn.Module):
        return source.state_dict()
    return source


def _validate(sd):
    """Check the arch the kernel hard-codes; return (n_blocks, n_gears).
    The v10 kernels REFUSE pre-v10 state_dicts (no body_mlp / wrong obs
    slicing / 12- or 14-head layouts)."""
    def shape(k):
        return tuple(sd[k].shape)
    if "body_mlp.0.weight" not in sd:
        raise NotImplementedError(
            "fused kernels bake the env v10 layout (obs 340, body token, "
            "29 heads); this state_dict has no body_mlp — pre-v10 "
            "checkpoints were retired with the v10 body")
    n_heads_sd = sum(1 for k in sd
                     if k.startswith("heads.") and k.endswith(".weight"))
    if n_heads_sd != N_ACTION_HEADS:
        raise NotImplementedError(
            f"fused kernels bake the 29-head (env v10 §1.4) layout; got "
            f"{n_heads_sd} heads — use the torch paths")
    n_gears = shape("heads.0.weight")[0]
    if n_gears not in (3, 5, 7):
        raise NotImplementedError(
            f"fused kernels support K in (3, 5, 7), got {n_gears} "
            "logits/head")
    if not all(tuple(sd[f"heads.{i}.weight"].shape)
               == (n_gears, TRUNK_DIM) for i in range(n_heads_sd)):
        raise NotImplementedError("unexpected head shapes")
    if shape("note_mlp.0.weight") != (EMBED, NOTE_FEAT):
        raise NotImplementedError(
            f"fused kernel supports embed={EMBED} only, got "
            f"note_mlp.0.weight {shape('note_mlp.0.weight')}")
    if shape("saber_mlp.0.weight") != (EMBED, SABER_FEAT + 2):
        raise NotImplementedError(
            f"unexpected saber_mlp.0.weight {shape('saber_mlp.0.weight')}")
    if shape("body_mlp.0.weight") != (EMBED, BODY_FEAT):
        raise NotImplementedError(
            f"unexpected body_mlp.0.weight {shape('body_mlp.0.weight')}")
    if shape("trunk.0.weight") != (TRUNK_DIM, 3 * EMBED):
        raise NotImplementedError(
            f"fused kernel supports trunk_dim={TRUNK_DIM} over the "
            f"3-token readout only, got trunk.0.weight "
            f"{shape('trunk.0.weight')}")
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
    return n_blocks, n_gears


def _pack(sd, n_blocks, n_gears) -> torch.Tensor:
    """Flatten the state_dict into the kernel's packed layout: every
    matrix transposed to [in][out] (so the output-owning thread reads
    coalesced), biases inline after their matrix, 29 heads + value head
    fused into one (128 -> 29K+1) matrix. Order is a contract with the
    constexpr offsets in policy_forward.cu / policy_backward.cu."""
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
    lin("body_mlp.0")
    lin("body_mlp.2")
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
                   + [t("value_head.weight")], dim=0)     # (29K+1, 128)
    parts.append(hw.t().contiguous().reshape(-1))
    parts.append(torch.cat(
        [t(f"heads.{i}.bias") for i in range(N_ACTION_HEADS)]
        + [t("value_head.bias")]))
    flat = torch.cat(parts)
    assert flat.numel() == expected_numel(n_blocks, n_gears), flat.numel()
    return flat


class FusedNotePolicy:
    """Packs a NotePolicy's weights once and serves single-launch
    forwards. Accepts a NotePolicy module or its state_dict."""

    def __init__(self, source, device="cuda"):
        sd = _state_dict(source)
        self.n_blocks, self.n_gears = _validate(sd)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("FusedNotePolicy requires a CUDA device")
        self._ext = _load_ext()
        self.weights = _pack(sd, self.n_blocks, self.n_gears).to(self.device)
        self._bufs = {}

    def repack(self, source):
        """Re-upload weights in place (same buffer — pointers, and any
        capture of the call, stay valid). Arch must match."""
        sd = _state_dict(source)
        if _validate(sd) != (self.n_blocks, self.n_gears):
            raise ValueError("repack with a different arch")
        self.weights.copy_(
            _pack(sd, self.n_blocks, self.n_gears).to(self.device))

    def _run(self, obs):
        if not (torch.is_tensor(obs) and obs.is_cuda):
            obs = torch.as_tensor(obs, device=self.device)
        if obs.dtype != torch.float32:
            obs = obs.float()
        obs = obs.contiguous()
        if obs.dim() != 2 or obs.shape[1] != OBS_DIM:
            raise ValueError(                # kernel indexes obs by OBS_DIM
                f"FusedNotePolicy expects (N, {OBS_DIM}) obs, "
                f"got {tuple(obs.shape)}")
        n = obs.shape[0]
        bufs = self._bufs.get(n)
        if bufs is None:
            bufs = (torch.empty(n, N_ACTION_HEADS, self.n_gears,
                                device=self.device),
                    torch.empty(n, device=self.device),
                    torch.empty(n, N_ACTION_HEADS, device=self.device,
                                dtype=torch.int64))
            self._bufs[n] = bufs
        self._ext.forward(self.weights, obs, bufs[0], bufs[1], bufs[2],
                          self.n_blocks, self.n_gears)
        return bufs

    def fused_forward(self, obs):
        """logits (N, 29, K) — torch.stack(policy(obs)[0], dim=1) layout —
        and value (N,). Buffers are reused across calls."""
        logits, value, _ = self._run(obs)
        return logits, value

    __call__ = fused_forward

    def argmax(self, obs):
        """Greedy actions (N, 29) int64, computed inside the same kernel."""
        return self._run(obs)[2]


def unpack(flat, n_blocks, n_gears):
    """Inverse of _pack: packed flat tensor -> {param_name: tensor} in
    torch NotePolicy shapes. Works on weights AND on the packed grad
    (same layout). Returned tensors are views/copies on flat's device."""
    out = {}
    pos = 0

    def take(n):
        nonlocal pos
        seg = flat[pos:pos + n]
        pos += n
        return seg

    def lin(prefix, n_in, n_out):
        out[prefix + ".weight"] = take(n_in * n_out).view(
            n_in, n_out).t().contiguous()
        out[prefix + ".bias"] = take(n_out).clone()

    lin("note_mlp.0", NOTE_FEAT, EMBED)
    lin("note_mlp.2", EMBED, EMBED)
    lin("saber_mlp.0", SABER_FEAT + 2, EMBED)
    lin("saber_mlp.2", EMBED, EMBED)
    lin("body_mlp.0", BODY_FEAT, EMBED)
    lin("body_mlp.2", EMBED, EMBED)
    for b in range(n_blocks):
        p = f"blocks.{b}."
        out[p + "ln_attn.weight"] = take(EMBED).clone()
        out[p + "ln_attn.bias"] = take(EMBED).clone()
        lin(p + "q_proj", EMBED, EMBED)
        lin(p + "k_proj", EMBED, EMBED)
        lin(p + "v_proj", EMBED, EMBED)
        out[p + "ln_ffn.weight"] = take(EMBED).clone()
        out[p + "ln_ffn.bias"] = take(EMBED).clone()
        lin(p + "ffn.0", EMBED, 2 * EMBED)
        lin(p + "ffn.2", 2 * EMBED, EMBED)
    lin("trunk.0", 3 * EMBED, TRUNK_DIM)
    lin("trunk.2", TRUNK_DIM, TRUNK_DIM)
    nlogits = N_ACTION_HEADS * n_gears
    nout = nlogits + 1
    hw = take(TRUNK_DIM * nout).view(TRUNK_DIM, nout).t()
    hb = take(nout)
    for i in range(N_ACTION_HEADS):
        out[f"heads.{i}.weight"] = hw[n_gears * i:n_gears * (i + 1)] \
            .contiguous()
        out[f"heads.{i}.bias"] = hb[n_gears * i:n_gears * (i + 1)].clone()
    out["value_head.weight"] = hw[nlogits:nout].contiguous()
    out["value_head.bias"] = hb[nlogits:nout].clone()
    assert pos == flat.numel(), (pos, flat.numel())
    return out


class FusedPolicyFn(torch.autograd.Function):
    """Autograd boundary of the fused training path. forward = the fused
    forward kernel (logits (mb,29,K) + value (mb,), grad-tracked);
    backward = the fused recompute-backward kernel, which turns
    (grad_logits, grad_value) into the full packed parameter gradient.
    The PPO loss math stays in plain torch between the two. No grad
    w.r.t. obs (inputs are data)."""

    @staticmethod
    def forward(ctx, packed, obs, trainer):
        n = obs.shape[0]
        logits = torch.empty(n, N_ACTION_HEADS, trainer.n_gears,
                             device=obs.device)
        value = torch.empty(n, device=obs.device)
        trainer._ext.forward(packed.detach(), obs, logits, value,
                             trainer._actions_buf(n), trainer.n_blocks,
                             trainer.n_gears)
        ctx.save_for_backward(packed, obs)
        ctx.trainer = trainer
        return logits, value

    @staticmethod
    def backward(ctx, grad_logits, grad_value):
        packed, obs = ctx.saved_tensors
        tr = ctx.trainer
        n = obs.shape[0]
        gl = grad_logits.reshape(n, N_ACTION_HEADS * tr.n_gears).contiguous()
        gv = (grad_value.contiguous() if grad_value is not None
              else torch.zeros(n, device=obs.device))
        tr.partial.zero_()
        grad = torch.empty(packed.numel(), device=obs.device)
        tr._ext_bwd.backward(packed.detach(), obs, gl, gv, tr.partial,
                             grad, tr.n_blocks, tr.n_gears)
        return grad, None, None


class FusedPolicyTrainer:
    """Owns the packed flat weight tensor as THE trainable leaf: build
    the optimizer over `.parameters()` (one tensor — Adam's update and
    clip_grad_norm_'s global norm are mathematically identical packed or
    per-parameter). `unpack_to(policy)` mirrors the weights back into a
    NotePolicy for rollout sampling / checkpointing.

    ncopies = striped partial-grad buffers the backward accumulates into
    (2 measured optimal on the GTX 1080 — L2 residency vs same-address
    atomic serialization; ncopies >= mb makes the backward deterministic,
    used by the bitwise tests)."""

    def __init__(self, source, device="cuda", ncopies=2):
        sd = _state_dict(source)
        self.n_blocks, self.n_gears = _validate(sd)
        if self.n_blocks > 2:
            raise NotImplementedError(
                "fused backward saves block inputs in shared memory and "
                f"supports n_blocks <= 2, got {self.n_blocks}")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("FusedPolicyTrainer requires a CUDA device")
        self._ext = _load_ext()
        self._ext_bwd = _load_bwd_ext()
        self.packed = torch.nn.Parameter(
            _pack(sd, self.n_blocks, self.n_gears).to(self.device))
        self.partial = torch.zeros(ncopies * self.packed.numel(),
                                   device=self.device)
        self._act_bufs = {}

    def parameters(self):
        return [self.packed]

    def _actions_buf(self, n):
        buf = self._act_bufs.get(n)
        if buf is None:
            buf = torch.empty(n, N_ACTION_HEADS, dtype=torch.int64,
                              device=self.device)
            self._act_bufs[n] = buf
        return buf

    def forward(self, obs):
        """logits (N, 29, K) + value (N,), tracked by autograd; backward
        fills self.packed.grad via the fused backward kernel."""
        if not (torch.is_tensor(obs) and obs.is_cuda
                and obs.dtype == torch.float32 and obs.is_contiguous()):
            raise ValueError("obs must be a contiguous CUDA f32 tensor")
        return FusedPolicyFn.apply(self.packed, obs, self)

    __call__ = forward

    def unpack_to(self, policy):
        """Copy the packed weights back into a NotePolicy (state_dict
        must match this arch)."""
        sd = unpack(self.packed.detach(), self.n_blocks, self.n_gears)
        with torch.no_grad():
            for k, v in policy.state_dict().items():
                v.copy_(sd[k])

    def grads_by_param(self):
        """The packed gradient unpacked to {param_name: grad} — for
        parity tests and inspection."""
        assert self.packed.grad is not None, "no gradient accumulated"
        return unpack(self.packed.grad, self.n_blocks, self.n_gears)
