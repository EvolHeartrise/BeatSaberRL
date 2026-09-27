"""Torch-profiler kernel census at production shape (audit Tier 0.3
proxy — works where Nsight's performance counters are locked down,
which is the norm on shared-cloud pods).

    python perf_profile.py [--phase env|update|both] [--num-envs 16384]

env    : the CUDA env step loop (ranked pool, production flags) driven
         by random actions + a policy rollout forward per tick — the
         rollout phase minus GAE.
update : the eager PPO minibatch update (bench_policy_update body).

Prints the top kernels by total CUDA time per phase. This decides the
audit's Tier 3 fork: if step_kernel dominates the env phase, the
Nsight bandwidth-vs-latency question is worth pursuing; if the obs
writer or GEMMs dominate, layout work targets those instead.
"""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile


def kernel_table(prof, title, rows=20):
    print(f"\n=== {title}: top {rows} kernels by CUDA time ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=rows,
        max_name_column_width=80))


def env_phase(num_envs, iters, policy, pool_sample=999999):
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    from beat_arena_env.ranked_pool import load_ranked_pool
    device = torch.device("cuda")
    # pool_sample small (e.g. 64) makes every pool read L2-resident —
    # the pool-residency probe that decides whether pool-layout work
    # (audit Tier 3.1/3.2) can matter at all
    pool = load_ranked_pool(sample=pool_sample, seed=0,
                            device=device, lead_s=0.5, mirror=True)
    from beat_arena_env.ranked_pool import pool_max_steps
    venv = CudaBeatVecEnv(num_envs, pool=pool, max_steps=pool_max_steps(pool),
                          gears=5, seed=0,
                          device=device)
    venv.reset()
    obs = venv._obs

    def tick():
        with torch.no_grad():
            a, _, _, _ = policy.get_action_and_value(
                obs.float(), need_entropy=False)
        venv.step(a)

    for _ in range(20):        # warmup: JIT loads, graphless steady state
        tick()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            tick()
        torch.cuda.synchronize()
    kernel_table(prof, f"env+rollout ({num_envs} envs x {iters} ticks)")


def update_phase(mb, iters, policy):
    import bench_policy_update as bpu
    device = torch.device("cuda")
    opt = torch.optim.Adam(policy.parameters(), lr=7e-4, fused=True)
    data = bpu.make_minibatch(mb, device, policy.n_gears)
    for _ in range(5):
        bpu.eager_update(policy, opt, *data)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            bpu.eager_update(policy, opt, *data)
        torch.cuda.synchronize()
    kernel_table(prof, f"update (mb {mb} x {iters})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default="both",
                    choices=("env", "update", "both"))
    ap.add_argument("--num-envs", type=int, default=16384)
    ap.add_argument("--mb", type=int, default=16384)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--ckpt", default="runs/perf_ref.pt")
    ap.add_argument("--pool-sample", type=int, default=999999,
                    help="ranked-pool subsample (small => L2-resident "
                         "pool: the Tier 3.1/3.2 residency probe)")
    args = ap.parse_args()

    from policy import NotePolicy, load_policy
    device = torch.device("cuda")
    try:
        policy = load_policy(torch.load(args.ckpt, map_location="cpu",
                                        weights_only=True))
        print(f"weights: {args.ckpt}")
    except (FileNotFoundError, SystemExit):
        torch.manual_seed(0)
        policy = NotePolicy(n_gears=5)
        print("weights: random init")
    policy = policy.to(device)
    policy.fp16 = True
    policy.flatten_fp16()
    print(f"gpu: {torch.cuda.get_device_name(0)}")

    if args.phase in ("env", "both"):
        env_phase(args.num_envs, args.iters, policy, args.pool_sample)
    if args.phase in ("update", "both"):
        update_phase(args.mb, args.iters, policy)


if __name__ == "__main__":
    main()
