"""CUDA env-kernel A/B harness: step timing + bitwise trajectory identity.

    # time the env step (production contact flags, 32768 envs)
    python tools/bench_env_kernel.py time [--num-envs 32768] [--ticks 300]
    # record a reference trajectory from THIS tree's kernel
    python tools/bench_env_kernel.py record --out base.npz
    # replay the same inputs on THIS tree's kernel, require bit equality
    python tools/bench_env_kernel.py compare --ref base.npz

Workloads (both use the production contact-mode recipe flags: pushes,
start jitter, crouch starts, torque randomization, energy / limit /
foot terms):
  flux     — Flux charts driven GREEDILY by a trained Flux policy
             (--ckpt), so the body actually stands and moves: both feet
             loaded, i.e. the ground block's real workload.
  standard — procedural standard charts with a fixed random action
             stream (bodies fall and auto-reset: limit/contact coverage,
             not a realistic posture mix).
  stdtr    — the TRAINING standard workload: ranked whole-level pool
             (the v58 recipe env: mirror, lead 0.5, wall despawn,
             ep_stats) driven by a trained standard trunk (--std-ckpt,
             a 20-min probe that stands at 88% acc), so the arms really
             swing sabers — limit rows and speed caps under the load
             training puts on them.
Actions are the policy's argmax by default; --sample (time) draws them
from softmax(logits) instead (Gumbel-max off a seeded CUDA generator,
so a sampled rollout is as reproducible as a greedy one) — the training
regime, where the policy's entropy keeps the joints busier. The *_s
identity tiers are the sampled twins.
The identity tiers run every workload in f32 and f64 (the weld-mode tier
went with the weld harness, 2026-09-27), compare obs / reward / done /
qpos / twist of every tick with array_equal. Everything is seeded; any
bit difference is a kernel change. Timing uses CUDA events around
env.step only (the policy forward is excluded) after a warmup.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beat_arena_env.cuda_env import CudaBeatVecEnv  # noqa: E402

RECIPE = dict(gears=5, fall_const=5.0,
              push_prob=0.00333, push_vmax=0.15, start_jitter=0.15,
              start_pose_jitter=0.15, crouch_start_prob=0.25,
              foot_z_coef=0.01, foot_x_coef=0.01, foot_hinge_r=0.12,
              foot_shape_coef=0.05, energy_coef=7.5e-6, limit_coef=5e-3,
              limit_arms=True, tau_rand=0.25, z_asinh=8.0, env_order=True)
DEFAULT_CKPT = "runs/flux_v1_cd300/leaf_from_upd00457.pt"
DEFAULT_STD_CKPT = "runs/std_probe/trunk_final.pt"
_RANKED = {}


def ranked_pool(n_levels):
    """The training pool (train_batch --ranked-maps, seed 0, lead 0.5,
    mirror augment); n_levels None = every level, as training uses."""
    if n_levels not in _RANKED:
        from beat_arena_env.ranked_pool import load_ranked_pool
        _RANKED[n_levels] = load_ranked_pool(
            sample=n_levels, seed=0, device="cuda", lead_s=0.5,
            mirror=True, log=lambda *a, **k: None)
    return _RANKED[n_levels]


def make_env(workload, n, dtype, pool=2048, std_levels=None):
    kw = dict(RECIPE)
    if workload == "flux":
        kw["game"] = "flux"
    if workload == "stdtr":
        # the v58 standard recipe env (runs/std_probe/args.json) on the
        # ranked pool; z_asinh 8 is the checkpoint's arch value
        from beat_arena_env.ranked_pool import pool_max_steps
        rp = ranked_pool(std_levels)
        kw.update(wall_despawn=True, ep_stats=True)
        return CudaBeatVecEnv(n, pool=rp, max_steps=pool_max_steps(rp),
                              seed=0, device="cuda", dtype=dtype, **kw)
    return CudaBeatVecEnv(n, chart_pool=pool, seed=0, device="cuda",
                          dtype=dtype, **kw)


def load_policy_for(workload, ckpt, std_ckpt=DEFAULT_STD_CKPT):
    if workload == "standard":
        return None
    from policy import load_policy
    game = "flux" if workload == "flux" else "standard"
    path = ckpt if workload == "flux" else std_ckpt
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return load_policy(ck, game=game).cuda().eval()


def actions_for(policy, obs, gen, n, sample=False):
    if policy is None:
        return torch.randint(0, 5, (n, 29), device="cuda", generator=gen)
    with torch.no_grad():
        logits, _ = policy.forward_stacked(obs.float())
        if sample:   # Gumbel-max == a draw from softmax(logits)
            u = torch.rand(logits.shape, device=logits.device,
                           generator=gen).clamp_(1e-10, 1.0)
            return (logits.float() - torch.log(-torch.log(u))).argmax(-1)
    return logits.argmax(-1)


def rollout(workload, n, ticks, dtype, ckpt, record=False, timing=False, warmup=0, sample=False,
            std_ckpt=DEFAULT_STD_CKPT, std_levels=None):
    torch.manual_seed(0)                       # pushes use the global RNG
    gen = torch.Generator(device="cuda").manual_seed(1)
    env = make_env(workload, n, dtype, std_levels=std_levels)
    policy = load_policy_for(workload, ckpt, std_ckpt)
    obs = env.reset()
    rec = {k: [] for k in ("obs", "rew", "done", "qpos", "twist")}
    ms = []
    for t in range(warmup + ticks):
        a = actions_for(policy, obs, gen, n, sample)
        if timing and t >= warmup:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            obs, rew, done, _ = env.step(a)
            e1.record()
            e1.synchronize()
            ms.append(e0.elapsed_time(e1))
        else:
            obs, rew, done, _ = env.step(a)
        if record:
            rec["obs"].append(obs.cpu().numpy().copy())
            rec["rew"].append(rew.cpu().numpy().copy())
            rec["done"].append(done.cpu().numpy().copy())
            rec["qpos"].append(env.qpos().cpu().numpy().copy())
            rec["twist"].append(env.twist().cpu().numpy().copy())
    return rec, ms


TIERS = [  # (tag, workload, dtype, n, ticks[, sample])
    ("flux_f32", "flux", torch.float32, 2048, 240),
    ("flux_f64", "flux", torch.float64, 512, 120),
    ("std_f32", "standard", torch.float32, 2048, 240),
    ("std_f64", "standard", torch.float64, 512, 120),
    # the training regimes: trained standard trunk (greedy + sampled) and
    # sampled Flux; a reference without them compares the tiers it has
    ("stdtr_f32", "stdtr", torch.float32, 2048, 240, False),
    ("stdtr_s_f32", "stdtr", torch.float32, 2048, 240, True),
    ("stdtr_s_f64", "stdtr", torch.float64, 512, 120, True),
    ("flux_s_f32", "flux", torch.float32, 2048, 240, True),
]
STD_LEVELS_ID = 1024   # identity tiers: a seeded 1,024-level subset


def tiers(a):
    for t in TIERS:
        tag, wl, dt, n, ticks = t[:5]
        if a.tiers and tag not in a.tiers:
            continue
        yield tag, wl, dt, n, ticks, (t[5] if len(t) > 5 else False)


def cmd_record(a):
    out = {}
    for tag, wl, dt, n, ticks, smp in tiers(a):
        rec, _ = rollout(wl, n, ticks, dt, a.ckpt, record=True,
                         sample=smp, std_ckpt=a.std_ckpt,
                         std_levels=STD_LEVELS_ID)
        for k, v in rec.items():
            out[f"{tag}/{k}"] = np.stack(v)
        print(f"recorded {tag}: {n} envs x {ticks} ticks", flush=True)
    np.savez_compressed(a.out, **out)
    print(f"-> {a.out}")


def cmd_compare(a):
    ref = np.load(a.ref)
    bad = 0
    for tag, wl, dt, n, ticks, smp in tiers(a):
        if f"{tag}/obs" not in ref.files:
            print(f"skipped {tag} (not in the reference)", flush=True)
            continue
        rec, _ = rollout(wl, n, ticks, dt, a.ckpt, record=True,
                         sample=smp, std_ckpt=a.std_ckpt,
                         std_levels=STD_LEVELS_ID)
        for k, v in rec.items():
            got, want = np.stack(v), ref[f"{tag}/{k}"]
            if got.shape != want.shape or not np.array_equal(
                    got.view(np.uint8), want.view(np.uint8)):
                diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
                first = int(np.argmax(diff.reshape(len(diff), -1).max(1) > 0))
                print(f"MISMATCH {tag}/{k}: max |d| {diff.max():.3e}, "
                      f"first differing tick {first}", flush=True)
                bad += 1
        print(f"compared {tag}", flush=True)
    print("BITWISE-IDENTICAL" if bad == 0 else f"{bad} MISMATCHES")
    sys.exit(1 if bad else 0)


def cmd_time(a):
    for wl in a.workloads:
        _, ms = rollout(wl, a.num_envs, a.ticks, torch.float32, a.ckpt,
                        record=False, timing=True, warmup=a.warmup,
                        sample=a.sample, std_ckpt=a.std_ckpt,
                        std_levels=a.std_levels)
        ms = np.array(ms)
        tag = wl + ("/sampled" if a.sample and wl != "standard" else "")
        print(f"TIME {tag} N={a.num_envs}: mean {ms.mean():.2f} ms/tick "
              f"median {np.median(ms):.2f} p90 {np.percentile(ms, 90):.2f} "
              f"-> {a.num_envs / (ms.mean() / 1e3):,.0f} env-steps/s "
              f"({len(ms)} ticks)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["time", "record", "compare"])
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--std-ckpt", default=DEFAULT_STD_CKPT)
    ap.add_argument("--sample", action="store_true",
                    help="time: sample actions from the policy (seeded) "
                         "instead of argmax")
    ap.add_argument("--std-levels", type=int, default=None,
                    help="time: ranked levels for stdtr (default all)")
    ap.add_argument("--tiers", nargs="+", default=None,
                    help="record/compare: only these tier tags")
    ap.add_argument("--num-envs", type=int, default=32768)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--workloads", nargs="+", default=["flux", "standard"],
                    choices=["flux", "standard", "stdtr"])
    ap.add_argument("--out", default="env_kernel_ref.npz")
    ap.add_argument("--ref", default="env_kernel_ref.npz")
    a = ap.parse_args()
    t0 = time.time()
    {"time": cmd_time, "record": cmd_record, "compare": cmd_compare}[a.cmd](a)
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
