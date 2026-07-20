"""CPU PPO training for Beat Arena (SPEC.md §Training).

Mirrors kiting-arena's harness: PufferLib emulation + Serial vectorization,
compact PPO with GAE and dual-clip. Reports random / trained / scripted
mean episodic return — scripted is the solvability ceiling reference.

Default run:  python train_smoke.py            (~easy chart, pool encoder)
Transformer:  python train_smoke.py --encoder transformer
Longer:       python train_smoke.py --total-steps 1000000 --time-cap 0
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

import pufferlib.emulation as em
import pufferlib.vector as vec

from beat_arena_env import BeatEnv, BeatSim, ChartConfig, scripted_action
from policy import NotePolicy

NUM_ENVS = 8
HORIZON = 128
EPOCHS = 3
MINIBATCHES = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.2
DUAL_CLIP_C = 3.0
VF_COEF = 0.5
LR = 3e-4
MAX_GRAD_NORM = 0.5

DEVICE = "cpu"


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
    p.add_argument("--total-steps", type=int, default=120_000)
    p.add_argument("--time-cap", type=float, default=240.0,
                   help="stop training after this many seconds; 0 disables")
    p.add_argument("--ckpt-every", type=int, default=10,
                   help="save a checkpoint every N updates; 0 disables")
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--encoder", choices=["pool", "transformer"], default="pool")
    p.add_argument("--note-rate", type=float, default=1.0)
    p.add_argument("--colors", choices=["right", "both"], default="right")
    p.add_argument("--dirs", type=int, nargs="+", default=[6],
                   help="allowed cut directions 0..7 (default: 6 = down)")
    p.add_argument("--chart-len", type=float, default=20.0)
    p.add_argument("--min-hand-gap", type=float, default=0.5)
    p.add_argument("--shaping-coef", type=float, default=0.0,
                   help="potential-based shaping strength; 0 (default) = pure "
                        "sim rewards; nonzero only for ablation studies")
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="entropy bonus coefficient")
    p.add_argument("--cols", type=int, nargs="+", default=[0, 1, 2, 3],
                   help="allowed note columns (curriculum: restrict)")
    p.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2],
                   help="allowed note rows")
    p.add_argument("--init-from", type=str, default=None)
    return p.parse_args()


def env_kwargs(args):
    return dict(note_rate=args.note_rate, colors=args.colors,
                dirs=tuple(args.dirs), chart_len_s=args.chart_len,
                min_hand_gap_s=args.min_hand_gap,
                shaping_coef=args.shaping_coef,
                cols=tuple(args.cols), rows=tuple(args.rows))


def make_venv(num_envs, args, base_seed=0):
    kw = env_kwargs(args)

    def make_env(buf=None, seed=0, **_):
        return em.GymnasiumPufferEnv(env=BeatEnv(seed=seed, **kw), buf=buf, seed=seed)

    return vec.make(make_env, num_envs=num_envs, backend=vec.Serial, seed=base_seed)


def save_ckpt(policy, run_dir, name, update, global_step, recent_return):
    path = os.path.join(run_dir, name)
    torch.save({
        "state_dict": policy.state_dict(),
        "encoder": policy.encoder,
        "update": update,
        "global_step": global_step,
        "recent_return": float(recent_return),
    }, path)
    return path


def evaluate(policy, args, n_episodes=30, base_seed=10_000, random_policy=False):
    venv = make_venv(8, args, base_seed=base_seed)
    obs, _ = venv.reset(seed=base_seed)
    ep_returns = []
    cur = np.zeros(8, dtype=np.float64)
    act_space = venv.single_action_space
    while len(ep_returns) < n_episodes:
        if random_policy:
            acts = np.stack([act_space.sample() for _ in range(8)])
        else:
            with torch.no_grad():
                t = torch.as_tensor(obs, dtype=torch.float32)
                a, _, _, _ = policy.get_action_and_value(t)
                acts = a.cpu().numpy().astype(np.int64)
        obs, rew, term, trunc, info = venv.step(acts)
        cur += rew
        for i in np.flatnonzero(np.logical_or(term, trunc)):
            ep_returns.append(cur[i])
            cur[i] = 0.0
    venv.close()
    return float(np.mean(ep_returns[:n_episodes])), ep_returns[:n_episodes]


def evaluate_scripted(args, n_episodes=30, base_seed=10_000):
    cfg = ChartConfig(**{k: v for k, v in env_kwargs(args).items()})
    rets, goods, notes = [], 0, 0
    for ep in range(n_episodes):
        sim = BeatSim(cfg, seed=base_seed + ep)
        done, tot = False, 0.0
        while not done:
            al, ar = scripted_action(sim)
            r, done, _ = sim.step(al, ar)
            tot += r
        rets.append(tot)
        goods += sim.good_cuts
        notes += len(sim.chart)
    return float(np.mean(rets)), goods / max(notes, 1)


def train(args):
    venv = make_venv(NUM_ENVS, args, base_seed=args.seed)
    obs_dim = venv.single_observation_space.shape[0]
    policy = NotePolicy(encoder=args.encoder).to(DEVICE)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        policy.load_state_dict(sd)
        print(f"warm-started from {args.init_from} "
              f"(update {ckpt.get('update', '?')}, "
              f"recent return {ckpt.get('recent_return', float('nan')):.2f})",
              flush=True)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    obs, _ = venv.reset(seed=args.seed)
    obs = torch.as_tensor(obs, dtype=torch.float32)

    batch = NUM_ENVS * HORIZON
    mb_size = batch // MINIBATCHES
    num_updates = max(1, args.total_steps // batch)

    b_obs = torch.zeros((HORIZON, NUM_ENVS, obs_dim))
    b_act = torch.zeros((HORIZON, NUM_ENVS, 2), dtype=torch.long)
    b_logp = torch.zeros((HORIZON, NUM_ENVS))
    b_rew = torch.zeros((HORIZON, NUM_ENVS))
    b_done = torch.zeros((HORIZON, NUM_ENVS))
    b_val = torch.zeros((HORIZON, NUM_ENVS))

    ep_return_hist = []
    cur_ret = np.zeros(NUM_ENVS, dtype=np.float64)

    start = time.time()
    global_step = 0
    for update in range(num_updates):
        for t in range(HORIZON):
            b_obs[t] = obs
            with torch.no_grad():
                a, logp, _, v = policy.get_action_and_value(obs)
            b_act[t], b_logp[t], b_val[t] = a, logp, v

            acts = a.cpu().numpy().astype(np.int64)
            nobs, rew, term, trunc, info = venv.step(acts)
            done = np.logical_or(term, trunc)

            b_rew[t] = torch.as_tensor(rew, dtype=torch.float32)
            b_done[t] = torch.as_tensor(done.astype(np.float32))

            cur_ret += rew
            for i in np.flatnonzero(done):
                ep_return_hist.append(cur_ret[i])
                cur_ret[i] = 0.0

            obs = torch.as_tensor(nobs, dtype=torch.float32)
            global_step += NUM_ENVS

        with torch.no_grad():
            _, _, _, next_val = policy.get_action_and_value(obs)
            adv = torch.zeros((HORIZON, NUM_ENVS))
            lastgae = torch.zeros(NUM_ENVS)
            for t in reversed(range(HORIZON)):
                nonterminal = 1.0 - b_done[t]
                nextvalue = next_val if t == HORIZON - 1 else b_val[t + 1]
                delta = b_rew[t] + GAMMA * nextvalue * nonterminal - b_val[t]
                lastgae = delta + GAMMA * GAE_LAMBDA * nonterminal * lastgae
                adv[t] = lastgae
            returns = adv + b_val

        f_obs = b_obs.reshape(-1, obs_dim)
        f_act = b_act.reshape(-1, 2)
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = returns.reshape(-1)

        idx = np.arange(batch)
        for _ in range(EPOCHS):
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

        recent = np.mean(ep_return_hist[-50:]) if ep_return_hist else float("nan")
        elapsed = time.time() - start
        print(f"update {update+1:>3}/{num_updates}  steps {global_step:>7}  "
              f"ep_return(last50) {recent:8.3f}  "
              f"pg {pg_loss.item():+.3f} v {v_loss.item():.3f} "
              f"ent {ent_loss.item():.3f} dual {dual_frac.item():.4f}  "
              f"{elapsed:5.1f}s", flush=True)

        if args.ckpt_every > 0 and (update + 1) % args.ckpt_every == 0:
            path = save_ckpt(policy, args.run_dir, f"ckpt_upd{update+1:04d}.pt",
                             update + 1, global_step, recent)
            print(f"  saved {path}", flush=True)

        if args.time_cap > 0 and elapsed > args.time_cap:
            print(f"Reached --time-cap {args.time_cap:.0f}s; stopping early.",
                  flush=True)
            break

    recent = np.mean(ep_return_hist[-50:]) if ep_return_hist else float("nan")
    path = save_ckpt(policy, args.run_dir, "final.pt", update + 1, global_step, recent)
    print(f"saved {path}", flush=True)
    venv.close()
    return policy


def main():
    args = parse_args()
    if args.run_dir is None:
        args.run_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.run_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=== Beat Arena PPO run ===", flush=True)
    print(f"run dir: {args.run_dir}", flush=True)
    print(f"total steps: {args.total_steps}  time cap: {args.time_cap or 'none'}s  "
          f"encoder: {args.encoder}  rate: {args.note_rate}  colors: {args.colors}  "
          f"dirs: {args.dirs}  chart: {args.chart_len}s", flush=True)

    rand_mean = scripted_mean = None
    if not args.skip_eval:
        print("\nEvaluating RANDOM policy (floor)...", flush=True)
        rand_mean, _ = evaluate(None, args, args.eval_episodes, random_policy=True)
        print(f"random mean return: {rand_mean:.3f}", flush=True)
        print("Evaluating SCRIPTED bot (ceiling reference)...", flush=True)
        scripted_mean, good_rate = evaluate_scripted(args, args.eval_episodes)
        print(f"scripted mean return: {scripted_mean:.3f} "
              f"(good-cut rate {100*good_rate:.0f}%)", flush=True)

    policy = train(args)

    if not args.skip_eval:
        print("\nEvaluating TRAINED policy...", flush=True)
        policy.eval()
        tr_mean, _ = evaluate(policy, args, args.eval_episodes)
        print("\n=== RESULT ===", flush=True)
        print(f"random   mean return: {rand_mean:8.3f}", flush=True)
        print(f"trained  mean return: {tr_mean:8.3f}", flush=True)
        print(f"scripted mean return: {scripted_mean:8.3f}", flush=True)
        if tr_mean > rand_mean:
            print("PASS: trained policy beats random.", flush=True)
        else:
            print("WARN: trained policy did not beat random.", flush=True)


if __name__ == "__main__":
    main()
