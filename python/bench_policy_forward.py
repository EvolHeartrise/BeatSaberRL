"""Benchmark the NotePolicy forward: eager vs CUDA-graphed PyTorch.

    python bench_policy_forward.py [--ns 2048 16384] [--ckpt runs/...]

(Restored 2026-08-15 from bd07669^ as the perf-ladder A/B instrument;
the fused-kernel column died with the 138k arch — the fused stack only
ever ran at n_heads == 1 / d == 64, see PERF_AUDIT_2026-08-16.md.)

Two paths, two shapes (production rollout N=16384, ScoreSaber bench
N=2048):
  eager    torch no_grad forward (launch-bound path)
  graphed  the same forward captured as one CUDA graph (train_batch.py
           --graph-rollout, the production default)

Each measurement: warmup, then best of --reps timed loops of --iters
forwards (cuda-synchronized), reported as µs/forward. Two operations
are timed: "forward" (logits + value on device) and "argmax" (greedy
(N, 29) actions on device — what bench_scoresaber consumes). Inputs
live on device; no host<->device copies are inside the timed region.

Perf-ladder toggles (BEAT_ATTN / BEAT_QKV_PACKED / BEAT_PACKED_HEADS /
BEAT_GEMM_LEGACY / BEAT_NO_WIDE_RELU_BWD) are read at import — run one
process per configuration.
"""

from __future__ import annotations

import argparse
import time

import torch

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
    p.add_argument("--fp16", action="store_true",
                   help="production rollout config: fp16 stream via the "
                        "flat replica path")
    args = p.parse_args()

    device = torch.device("cuda")
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        policy = load_policy(ckpt)
        print(f"weights: {args.ckpt}")
    except (FileNotFoundError, SystemExit):
        torch.manual_seed(0)
        policy = NotePolicy(n_gears=5)
        print("weights: random init (checkpoint not found)")
    policy = policy.to(device).eval()
    if args.fp16:
        policy.fp16 = True
        policy.flatten_fp16()
    print(f"gpu: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"best of {args.reps} x {args.iters} iters, µs/forward\n")

    hdr = f"{'op':<8}{'N':>6}{'eager':>10}{'graphed':>10}{'vs eager':>10}"
    print(hdr)
    print("-" * len(hdr))
    for n in args.ns:
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
            print(f"{op:<8}{n:>6}{te:>10.1f}{tg:>10.1f}{te / tg:>9.2f}x")
    print()


if __name__ == "__main__":
    main()
