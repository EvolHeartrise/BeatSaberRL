"""Benchmark the PPO minibatch UPDATE (fwd + losses + bwd + clip + Adam)
on the eager path, plus a peak-memory probe.

    python bench_policy_update.py [--mbs 16384] [--ckpt runs/...]

(Restored 2026-08-15 from bd07669^ as the perf-ladder A/B instrument;
the fused-trainer column died with the 138k arch, and the eager body
below mirrors train_batch.py's minibatch loop minus the loss-scale /
flat-grad machinery — a few fixed elementwise kernels that are
identical across ladder configurations. Confirm absolute wins with
train_batch --bench.)

Timing: warmup, then best of --reps timed loops of --iters updates
(cuda-synchronized), reported as ms/update. The updates REALLY step
Adam and bump the fp16 weights-version each step (weights drift during
timing — same work every iteration, and the per-minibatch replica
refresh is exactly production's).

Perf-ladder toggles (BEAT_ATTN / BEAT_QKV_PACKED / BEAT_PACKED_HEADS /
BEAT_GEMM_LEGACY / BEAT_NO_WIDE_RELU_BWD) are read at import — run one
process per configuration.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn

from fp16_linear import bump_weights_version
from policy import (N_ACTION_HEADS, N_NOTE_SLOTS, NotePolicy, OBS_DIM,
                    OFF_NOTE_MASK, load_policy)
from train_batch import MAX_GRAD_NORM, VF_COEF, pg_loss_fn

ENT_COEF = 0.03
LR = 7e-4


def make_minibatch(n, device, n_gears, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = (torch.randn(n, OBS_DIM, generator=g) * 0.7).to(device)
    obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS] = (
        torch.rand(n, N_NOTE_SLOTS, generator=g) < 0.6).float().to(device)
    act = torch.randint(0, n_gears, (n, N_ACTION_HEADS),
                        generator=g).to(device)
    logp0 = -N_ACTION_HEADS * math.log(n_gears)
    logp = (torch.randn(n, generator=g) * 0.3 + logp0).to(device)
    adv = torch.randn(n, generator=g).to(device)
    ret = torch.randn(n, generator=g).to(device)
    return obs, act, logp, adv, ret


def eager_update(policy, opt, obs, act, logp, adv, ret):
    _, newlogp, entropy, newval = policy.get_action_and_value(obs, act)
    ratio = (newlogp - logp).exp()
    a_mb = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg_loss = pg_loss_fn(ratio, a_mb)
    v_loss = 0.5 * ((newval - ret) ** 2).mean()
    loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss
    opt.zero_grad(set_to_none=False)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
    opt.step()
    bump_weights_version()


def fresh_policy(src, device):
    """One policy per bench row: flatten_fp16 moves p.data pointers
    exactly once per module, so a shared module can't be reflattened."""
    a = src.arch
    p = NotePolicy(embed=a["embed"], trunk_dim=a["trunk_dim"],
                   n_blocks=a["n_blocks"], n_heads=a["n_heads"],
                   n_gears=a["n_gears"], ffn_mult=a["ffn_mult"]).to(device)
    p.load_state_dict(src.state_dict())
    return p


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default="runs/perf_ref.pt")
    ap.add_argument("--mbs", type=int, nargs="+", default=[16384])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--fp16", action="store_true",
                    help="production config: fp16 stream via the flat "
                         "replica path + fused Adam")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.distributions.Distribution.set_default_validate_args(False)
    try:
        src = load_policy(torch.load(args.ckpt, map_location="cpu",
                                     weights_only=True))
        print(f"weights: {args.ckpt}")
    except (FileNotFoundError, SystemExit):
        torch.manual_seed(0)
        src = NotePolicy(n_gears=5)  # production default (--gears 5)
        print("weights: random init (checkpoint not found)")
    print(f"gpu: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"best of {args.reps} x {args.iters} updates, ms/update "
          f"(update = fwd + losses + bwd + clip + Adam)\n")

    hdr = f"{'mb':>7}{'ms/update':>12}{'peak MB':>10}"
    print(hdr)
    print("-" * len(hdr))
    for n in args.mbs:
        pol = fresh_policy(src, device)
        if args.fp16:                    # mirror train_batch.py order:
            pol.fp16 = True              # flatten BEFORE the optimizer is
            pol.flatten_fp16()           # built (p.data pointers move once)
        opt = torch.optim.Adam(pol.parameters(), lr=LR,
                               fused=args.fp16 or None)
        data = make_minibatch(n, device, src.n_gears)
        t = bench(lambda: eager_update(pol, opt, *data),
                  args.iters, args.reps)
        mem = peak_mem_mb(lambda: eager_update(pol, opt, *data))
        print(f"{n:>7}{t:>12.2f}{mem:>10.1f}")
    print()


if __name__ == "__main__":
    main()
