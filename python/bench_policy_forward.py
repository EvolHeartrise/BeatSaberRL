"""Benchmark the fused CUDA NotePolicy forward vs eager and CUDA-graphed
PyTorch.

    python bench_policy_forward.py [--ns 512 4096] [--ckpt runs/...]

Three paths, two shapes (rollout N=4096, ScoreSaber bench N=512):
  eager    torch no_grad forward (the ~100-kernel launch-bound path)
  graphed  the same forward captured as one CUDA graph (train_batch.py
           _Graphed protocol — the honest incumbent)
  fused    policy_cuda.FusedNotePolicy (one kernel launch)

Each measurement: warmup, then best of --reps timed loops of --iters
forwards (cuda-synchronized), reported as µs/forward. Two operations
are timed: "forward" (logits + value on device) and "argmax" (greedy
(N, 12) actions on device — what bench_scoresaber consumes). Inputs
live on device; no host<->device copies are inside the timed region.
"""

from __future__ import annotations

import argparse
import time

import torch

from policy import NotePolicy, OBS_DIM, load_policy
from policy_cuda import FusedNotePolicy
from cuda_graphs import _Graphed


class _GraphedForward(_Graphed):
    def __init__(self, policy, n, device):
        super().__init__()
        self.policy = policy
        self.s_obs = torch.zeros(n, OBS_DIM, device=device)

    def _compute(self):
        with torch.no_grad():
            logits, value = self.policy(self.s_obs)
        return torch.stack(logits, dim=1), value

    def __call__(self, obs):
        self.s_obs.copy_(obs)
        return self._run()


class _GraphedArgmax(_GraphedForward):
    def _compute(self):
        with torch.no_grad():
            logits, _ = self.policy(self.s_obs)
        return torch.stack(logits, dim=1).argmax(-1)


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


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", default="runs/gen2_ep2_50m/final.pt")
    p.add_argument("--ns", type=int, nargs="+", default=[512, 4096])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()

    device = torch.device("cuda")
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        policy = load_policy(ckpt)
        print(f"weights: {args.ckpt}")
    except FileNotFoundError:
        policy = NotePolicy()
        print("weights: random init (checkpoint not found)")
    policy = policy.to(device).eval()
    fused = FusedNotePolicy(policy)
    print(f"gpu: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"best of {args.reps} x {args.iters} iters, µs/forward\n")

    hdr = f"{'op':<8}{'N':>6}{'eager':>10}{'graphed':>10}{'fused':>10}" \
          f"{'vs eager':>10}{'vs graphed':>12}"
    print(hdr)
    print("-" * len(hdr))
    for n in args.ns:
        g = torch.Generator().manual_seed(0)
        obs = (torch.randn(n, OBS_DIM, generator=g) * 0.7).to(device)
        obs[:, 270:] = (torch.rand(n, 16, generator=g) < 0.6).float().to(device)

        def eager_fwd():
            with torch.no_grad():
                logits, value = policy(obs)
            return torch.stack(logits, dim=1), value

        def eager_arg():
            with torch.no_grad():
                logits, _ = policy(obs)
            return torch.stack(logits, dim=1).argmax(-1)

        gf = _GraphedForward(policy, n, device)
        ga = _GraphedArgmax(policy, n, device)
        for op, e_fn, g_fn, f_fn in (
                ("forward", eager_fwd, lambda: gf(obs),
                 lambda: fused.fused_forward(obs)),
                ("argmax", eager_arg, lambda: ga(obs),
                 lambda: fused.argmax(obs))):
            te = bench(e_fn, args.iters, args.reps)
            tg = bench(g_fn, args.iters, args.reps)
            tf = bench(f_fn, args.iters, args.reps)
            print(f"{op:<8}{n:>6}{te:>10.1f}{tg:>10.1f}{tf:>10.1f}"
                  f"{te / tf:>9.2f}x{tg / tf:>11.2f}x")
    print()


if __name__ == "__main__":
    main()
