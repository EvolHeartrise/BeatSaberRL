"""Benchmark the PPO minibatch UPDATE (fwd + losses + bwd + clip + Adam)
across the update paths, plus peak-memory probes.

    python bench_policy_update.py [--mbs 16384 32768] [--ckpt runs/...]

Paths (the graphed columns died with the trainer's CUDA-graphs removal,
2026-07-25):
  eager    the train_batch.py eager minibatch body (torch autograd)
  fused    --fused-update body: fused fwd/bwd kernels + torch losses +
           Adam over the packed param (policy_cuda.FusedPolicyTrainer)

Timing: warmup, then best of --reps timed loops of --iters updates
(cuda-synchronized), reported as ms/update. The updates REALLY step Adam
(weights drift during timing — same work every iteration, and PPO
minibatch updates are exactly this workload).

Memory: peak torch.cuda.max_memory_allocated during one eager-path vs
one fused-path update at each minibatch size, minus the resting
footprint — approximately the activation/workspace cost of the update.
The fused backward saves only the obs minibatch and recomputes
activations in-kernel, so its overhead stays O(obs mb) + the fixed
striped-grad workspace, while autograd's saved activations scale with
depth x width x rows.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from policy import NotePolicy, OBS_DIM, load_policy
from policy_cuda import FusedPolicyTrainer
from train_batch import (MAX_GRAD_NORM, VF_COEF, fused_minibatch_losses,
                         pg_loss_fn)

ENT_COEF = 0.01
LR = 3e-4


def make_minibatch(n, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = (torch.randn(n, OBS_DIM, generator=g) * 0.7).to(device)
    obs[:, 270:] = (torch.rand(n, 16, generator=g) < 0.6).float().to(device)
    act = torch.randint(0, 3, (n, 12), generator=g).to(device)
    logp = (torch.randn(n, generator=g) * 0.3 - 13.0).to(device)
    adv = torch.randn(n, generator=g).to(device)
    ret = torch.randn(n, generator=g).to(device)
    return obs, act, logp, adv, ret


def eager_update(policy, opt, obs, act, logp, adv, ret):
    _, newlogp, entropy, newval = policy.get_action_and_value(obs, act)
    ratio = (newlogp - logp).exp()
    a_mb = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg_loss, _ = pg_loss_fn(ratio, a_mb)
    v_loss = 0.5 * ((newval - ret) ** 2).mean()
    loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
    opt.step()


def fused_update(trainer, opt, obs, act, logp, adv, ret):
    loss, _ = fused_minibatch_losses(trainer, ENT_COEF, obs, act, logp,
                                     adv, ret)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(trainer.parameters(), MAX_GRAD_NORM)
    opt.step()


def bench(fn, iters, reps, warmup=5):
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
    return best * 1e3  # ms


def peak_mem_mb(fn, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def fresh_policy(src, device):
    p = NotePolicy().to(device)
    p.load_state_dict(src.state_dict())
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default="runs/gen2_ep2_50m/final.pt")
    ap.add_argument("--mbs", type=int, nargs="+", default=[16384, 32768])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.distributions.Distribution.set_default_validate_args(False)
    try:
        src = load_policy(torch.load(args.ckpt, map_location="cpu",
                                     weights_only=True))
        print(f"weights: {args.ckpt}")
    except FileNotFoundError:
        torch.manual_seed(0)
        src = NotePolicy()
        print("weights: random init (checkpoint not found)")
    print(f"gpu: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"best of {args.reps} x {args.iters} updates, ms/update "
          f"(update = fwd + losses + bwd + clip + Adam)\n")

    hdr = f"{'mb':>7}{'eager':>10}{'fused':>10}{'vs eager':>10}"
    print(hdr)
    print("-" * len(hdr))
    mem_rows = []
    for n in args.mbs:
        data = make_minibatch(n, device)

        pol_e = fresh_policy(src, device)
        opt_e = torch.optim.Adam(pol_e.parameters(), lr=LR)
        t_eager = bench(lambda: eager_update(pol_e, opt_e, *data),
                        args.iters, args.reps)
        mem_eager = peak_mem_mb(lambda: eager_update(pol_e, opt_e, *data))

        tr = FusedPolicyTrainer(fresh_policy(src, device))
        opt_f = torch.optim.Adam(tr.parameters(), lr=LR)
        t_fused = bench(lambda: fused_update(tr, opt_f, *data),
                        args.iters, args.reps)
        mem_fused = peak_mem_mb(lambda: fused_update(tr, opt_f, *data))

        print(f"{n:>7}{t_eager:>10.2f}{t_fused:>10.2f}"
              f"{t_eager / t_fused:>9.2f}x")
        mem_rows.append((n, mem_eager, mem_fused))

    print("\npeak memory over resting footprint during ONE update (MB):")
    print(f"{'mb':>7}{'eager autograd':>16}{'fused recompute':>17}")
    for n, me, mf in mem_rows:
        print(f"{n:>7}{me:>16.1f}{mf:>17.1f}")
    print("\n(the fused path's obs minibatch itself is "
          f"{OBS_DIM * 4 / 2**20:.4f} MB/kRow; its striped-grad workspace "
          "is a fixed ~0.9 MB)")


if __name__ == "__main__":
    main()
