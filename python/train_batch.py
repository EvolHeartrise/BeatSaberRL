"""Batched-rollout CPU PPO: N parallel Rust envs, ONE policy forward per
vector step. Same PPO math as train_smoke.py (GAE, dual-clip); the batch
shape is the variable under study — batch = num_envs x horizon.

Throughput study:  python train_batch.py --bench --num-envs 256
Real training:     python train_batch.py --num-envs 128 --total-steps 5000000

NOTE on science: changing num_envs changes the optimization regime
(updates-per-sample, minibatch size). Throughput comparisons are clean;
learning-curve comparisons across different batch shapes are NOT
apples-to-apples without retuning (LR, epochs).
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from beat_arena_env.vec_env import BeatVecEnv
from policy import NotePolicy, OBS_DIM

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.2
DUAL_CLIP_C = 3.0
VF_COEF = 0.5
LR = 3e-4
MAX_GRAD_NORM = 0.5


def pg_loss_fn(ratio, adv, clip=CLIP, dual_clip_c=DUAL_CLIP_C):
    """Dual-clip PPO policy loss (arXiv:1912.09729)."""
    pg1 = -adv * ratio
    pg2 = -adv * torch.clamp(ratio, 1 - clip, 1 + clip)
    standard = torch.max(pg1, pg2)
    ceiling = -dual_clip_c * adv
    dual = torch.min(standard, ceiling)
    loss = torch.where(adv < 0, dual, standard)
    frac_active = ((adv < 0) & (ceiling < standard)).float().mean()
    return loss.mean(), frac_active


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--time-cap", type=float, default=0.0)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=0)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--init-from", type=str, default=None)
    p.add_argument("--bench", action="store_true",
                   help="short run, prints steps/s split (rollout vs update)")
    # env config (mirrors train_smoke)
    p.add_argument("--note-rate", type=float, default=1.0)
    p.add_argument("--colors", choices=["right", "both"], default="right")
    p.add_argument("--dirs", type=int, nargs="+", default=[6])
    p.add_argument("--cols", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--chart-len", type=float, default=20.0)
    p.add_argument("--min-hand-gap", type=float, default=0.5)
    p.add_argument("--note-speed", type=float, default=4.0,
                   help="approach speed; MIN of the range if --note-speed-max set")
    p.add_argument("--note-speed-max", type=float, default=None)
    p.add_argument("--continuous-dirs", action="store_true",
                   help="angles ~ U(0, 2pi) instead of the --dirs subset")
    p.add_argument("--shaping-coef", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.bench:
        args.total_steps = args.num_envs * args.horizon * 4
    if args.run_dir is None:
        args.run_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "runs", "batch_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.run_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    N, T = args.num_envs, args.horizon
    batch = N * T
    mb_size = batch // args.minibatches

    venv = BeatVecEnv(
        N, note_rate=args.note_rate, colors=args.colors, dirs=tuple(args.dirs),
        cols=tuple(args.cols), rows=tuple(args.rows), chart_len_s=args.chart_len,
        min_hand_gap_s=args.min_hand_gap, note_speed=args.note_speed, note_speed_max=args.note_speed_max,
              continuous_dirs=args.continuous_dirs,
        shaping_coef=args.shaping_coef, seed=args.seed)

    policy = NotePolicy()
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        policy.load_state_dict(ckpt["state_dict"])
        print(f"warm-started from {args.init_from}", flush=True)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    obs = torch.as_tensor(venv.reset(), dtype=torch.float32)

    b_obs = torch.zeros((T, N, OBS_DIM))
    b_act = torch.zeros((T, N, 2), dtype=torch.long)
    b_logp = torch.zeros((T, N))
    b_rew = torch.zeros((T, N))
    b_done = torch.zeros((T, N))
    b_val = torch.zeros((T, N))

    ep_returns = []
    num_updates = max(1, args.total_steps // batch)
    global_step = 0
    t_rollout = t_update = 0.0
    start = time.time()

    print(f"=== batched PPO: {N} envs x {T} horizon = {batch} steps/update, "
          f"minibatch {mb_size} ===", flush=True)

    for update in range(num_updates):
        t0 = time.perf_counter()
        for t in range(T):
            b_obs[t] = obs
            with torch.no_grad():
                a, logp, _, v = policy.get_action_and_value(obs)
            b_act[t], b_logp[t], b_val[t] = a, logp, v
            nobs, rew, done, infos = venv.step(a.numpy())
            b_rew[t] = torch.as_tensor(rew)
            b_done[t] = torch.as_tensor(done.astype(np.float32))
            for info in infos:
                ep_returns.append(info["episode_return"])
            obs = torch.as_tensor(nobs, dtype=torch.float32)
            global_step += N
        t_rollout += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            _, _, _, next_val = policy.get_action_and_value(obs)
            adv = torch.zeros((T, N))
            lastgae = torch.zeros(N)
            for t in reversed(range(T)):
                nonterminal = 1.0 - b_done[t]
                nextvalue = next_val if t == T - 1 else b_val[t + 1]
                delta = b_rew[t] + GAMMA * nextvalue * nonterminal - b_val[t]
                lastgae = delta + GAMMA * GAE_LAMBDA * nonterminal * lastgae
                adv[t] = lastgae
            returns = adv + b_val

        f_obs = b_obs.reshape(-1, OBS_DIM)
        f_act = b_act.reshape(-1, 2)
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = returns.reshape(-1)

        idx = np.arange(batch)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for s in range(0, batch, mb_size):
                mb = idx[s:s + mb_size]
                _, newlogp, entropy, newval = policy.get_action_and_value(
                    f_obs[mb], f_act[mb])
                ratio = (newlogp - f_logp[mb]).exp()
                a_mb = f_adv[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg_loss, dual_frac = pg_loss_fn(ratio, a_mb)
                v_loss = 0.5 * ((newval - f_ret[mb]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * ent_loss + VF_COEF * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                opt.step()
        t_update += time.perf_counter() - t0

        recent = np.mean(ep_returns[-100:]) if ep_returns else float("nan")
        elapsed = time.time() - start
        rate = global_step / elapsed
        print(f"update {update+1:>3}/{num_updates}  steps {global_step:>9}  "
              f"ep_return(last100) {recent:8.3f}  ent {ent_loss.item():.3f}  "
              f"{rate:8,.0f} steps/s  {elapsed:6.1f}s", flush=True)

        if args.ckpt_every > 0 and (update + 1) % args.ckpt_every == 0:
            torch.save({"state_dict": policy.state_dict(), "encoder": policy.encoder,
                        "update": update + 1, "global_step": global_step,
                        "recent_return": float(recent)},
                       os.path.join(args.run_dir, f"ckpt_upd{update+1:04d}.pt"))
        if args.time_cap > 0 and elapsed > args.time_cap:
            print(f"Reached --time-cap {args.time_cap:.0f}s.", flush=True)
            break

    recent = np.mean(ep_returns[-100:]) if ep_returns else float("nan")
    torch.save({"state_dict": policy.state_dict(), "encoder": policy.encoder,
                "update": update + 1, "global_step": global_step,
                "recent_return": float(recent)},
               os.path.join(args.run_dir, "final.pt"))
    tot = t_rollout + t_update
    print(f"THROUGHPUT: {global_step / (time.time() - start):,.0f} steps/s total | "
          f"rollout {100*t_rollout/tot:.0f}% update {100*t_update/tot:.0f}% | "
          f"N={N} T={T}", flush=True)


if __name__ == "__main__":
    main()
