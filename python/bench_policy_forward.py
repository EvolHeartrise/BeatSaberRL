"""Benchmark the NotePolicy forward: eager vs CUDA-graphed PyTorch.

    python bench_policy_forward.py [--ns 2048 16384] [--ckpt runs/...]

(Restored 2026-08-15 from 68fbc13^ as the perf-ladder A/B instrument;
the fused-kernel column died with the 138k arch — the fused stack only
ever ran at n_heads == 1 / d == 64, see PERF_AUDIT_2026-08-16.md.)

Two paths, two shapes (production rollout N=16384, ScoreSaber bench
N=2048):
  eager    torch no_grad forward (launch-bound path)
  graphed  the same forward captured as one CUDA graph (the mechanism
           the ScoreSaber bench uses for its argmax forward; the
           trainer's --graph-rollout was removed 2026-08-31 after this
           very instrument measured it at 0.99x eager at N=32,768)

Each measurement: warmup, then best of --reps timed loops of --iters
forwards (cuda-synchronized), reported as µs/forward. Two operations
are timed: "forward" (logits + value on device) and "argmax" (greedy
(N, 29) actions on device — what bench_scoresaber consumes). Inputs
live on device; no host<->device copies are inside the timed region.

(The perf-ladder env toggles were removed 2026-09-22; run one
process per configuration.)

--fused-ffn off on [up16] (2026-09-30, review I13) times the fused
block-FFN kernel against the unfused path in ONE process: the mode is a
runtime policy attribute (arch "fused_ffn"), each mode gets freshly
captured graphs, and the table's `ffn` column names the mode. The line
before each mode's rows reports how many fused-kernel calls one forward
made (0 = the kernel does not apply: no extension / other widths / no
--fp16). Random init now defaults to the train_batch recipe (192 / 384 /
6 blocks, the only shape the kernel is built for); --legacy-arch keeps
the NotePolicy ctor defaults.
"""

from __future__ import annotations

import argparse
import time

import torch

import fp16_linear as _fl
from policy import (N_NOTE_SLOTS, NotePolicy, OBS_DIM, OFF_NOTE_MASK,
                    load_policy)
from cuda_graphs import _Graphed


class _GraphedForward(_Graphed):
    def __init__(self, policy, n, device, dtype):
        super().__init__()
        self.policy = policy
        self.s_obs = torch.zeros(n, OBS_DIM, device=device, dtype=dtype)

    def _compute(self):
        with torch.no_grad():
            return self.policy.forward_stacked(self.s_obs)

    def __call__(self, obs):
        self._copy_static(self.s_obs, obs)
        return self._run()


class _GraphedArgmax(_GraphedForward):
    def _compute(self):
        with torch.no_grad():
            logits, _ = self.policy.forward_stacked(self.s_obs)
        return logits.argmax(-1)


def bench(fn, iters, reps, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best * 1e6  # µs


def make_obs(n, device):
    g = torch.Generator().manual_seed(0)
    obs = (torch.randn(n, OBS_DIM, generator=g) * 0.7).to(device)
    # the mask block is interior now (ground_imp/body_pts follow it)
    obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS] = (
        torch.rand(n, N_NOTE_SLOTS, generator=g) < 0.6).float().to(device)
    return obs


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", default="runs/perf_ref.pt")
    p.add_argument("--ns", type=int, nargs="+", default=[2048, 16384])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--fp16-accum", default=None,
                   choices=("off", "fwd", "all"),
                   help="GEMM accumulate mode of the fp16 stream (issue "
                        "#3); default = the checkpoint arch's mode")
    p.add_argument("--fp16", action="store_true",
                   help="production rollout config: fp16 stream via the "
                        "flat replica path")
    p.add_argument("--fused-ffn", nargs="+", default=None,
                   choices=_fl.FFN_MODES,
                   help="fused block-FFN kernel mode(s) to time, one table "
                        "row set each (review I13); default = the "
                        "checkpoint arch's mode")
    p.add_argument("--legacy-arch", action="store_true",
                   help="random init at the NotePolicy ctor defaults "
                        "(embed 128 / trunk 256 / 3 blocks) instead of the "
                        "train_batch recipe (192 / 384 / 6 blocks, "
                        "note_geo)")
    args = p.parse_args()

    device = torch.device("cuda")
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        policy = load_policy(ckpt)
        print(f"weights: {args.ckpt}")
    except (FileNotFoundError, SystemExit):
        torch.manual_seed(0)
        if args.legacy_arch:
            policy = NotePolicy(n_gears=5)
        else:                        # train_batch.py recipe defaults
            policy = NotePolicy(embed=192, trunk_dim=384, n_blocks=6,
                                n_heads=4, n_gears=5, ffn_mult=4,
                                note_geo=True)
        print("weights: random init (checkpoint not found)")
    policy = policy.to(device).eval()
    if args.fp16_accum is not None:
        policy.fp16_accum = policy.arch["fp16_accum"] = args.fp16_accum
    print(f"fp16 accumulate: {policy.fp16_accum}")
    if args.fp16:
        policy.fp16 = True
        policy.flatten_fp16()
    print(f"gpu: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"best of {args.reps} x {args.iters} iters, µs/forward\n")

    modes = args.fused_ffn or [policy.fused_ffn]
    hdr = (f"{'ffn':<6}{'op':<8}{'N':>6}{'eager':>10}{'graphed':>10}"
           f"{'vs eager':>10}")
    print(hdr)
    print("-" * len(hdr))
    for mode in modes:
        policy.fused_ffn = policy.arch["fused_ffn"] = mode
        print(f"[{mode}] fused FFN kernel calls / forward: "
              f"{_count_fused(policy, device)}")
        for n in args.ns:
            _bench_n(policy, n, device, args, mode)
    print()


def _count_fused(policy, device):
    """How many fused-FFN kernel calls one eager forward makes."""
    calls = [0]
    orig = _fl.fused_ffn

    def counted(*a, **k):
        calls[0] += 1
        return orig(*a, **k)
    _fl.fused_ffn = counted
    try:
        with torch.no_grad():
            policy.forward_stacked(make_obs(64, device))
    finally:
        _fl.fused_ffn = orig
    return calls[0]


def _bench_n(policy, n, device, args, mode):
    """eager vs graphed forward / argmax rows at batch n."""
    obs = make_obs(n, device)

    def eager_fwd():
        with torch.no_grad():
            return policy.forward_stacked(obs)

    def eager_arg():
        with torch.no_grad():
            logits, _ = policy.forward_stacked(obs)
        return logits.argmax(-1)

    gf = _GraphedForward(policy, n, device, obs.dtype)
    ga = _GraphedArgmax(policy, n, device, obs.dtype)
    for op, e_fn, g_fn in (("forward", eager_fwd, lambda: gf(obs)),
                           ("argmax", eager_arg, lambda: ga(obs))):
        te = bench(e_fn, args.iters, args.reps)
        tg = bench(g_fn, args.iters, args.reps)
        print(f"{mode:<6}{op:<8}{n:>6}{te:>10.1f}{tg:>10.1f}"
              f"{te / tg:>9.2f}x")


if __name__ == "__main__":
    main()
