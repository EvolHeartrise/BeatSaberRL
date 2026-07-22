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
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from beat_arena_env.rust_env import RustBeatEnv
from beat_arena_env.vec_env import BeatVecEnv
from policy import NotePolicy, OBS_DIM, load_policy

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
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--lr-decay", choices=["none", "cosine", "linear"],
                   default="none",
                   help="anneal lr from --lr to --lr-final over the run "
                        "(motivated by the 2026-07-21 finding: 3e-4 "
                        "discovers, ~1e-4 refines)")
    p.add_argument("--lr-final", type=float, default=1e-4,
                   help="end-of-run lr for --lr-decay")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=0)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--init-from", type=str, default=None)
    p.add_argument("--bench", action="store_true",
                   help="short run, prints steps/s split (rollout vs update)")
    p.add_argument("--device", type=str, default=None,
                   help="cuda|cpu (default: cuda if available)")
    p.add_argument("--env", choices=["rust", "torch"], default="rust",
                   help="torch = GPU-resident sim (obs born on device; "
                        "procedural charts only, no curriculum)")
    p.add_argument("--chart-pool", type=int, default=1024,
                   help="pregenerated chart pool size for --env torch")
    # model arch (ignored with --init-from: the checkpoint's arch wins)
    p.add_argument("--embed", type=int, default=64)
    p.add_argument("--trunk-dim", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=1)
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
    # curriculum mode (SPEC §Curriculum): real ScoreSaber-ranked maps on a
    # star ladder instead of the procedural generator
    c = p.add_argument_group("curriculum")
    c.add_argument("--curriculum", action="store_true",
                   help="train on a ScoreSaber map ladder (ignores the "
                        "procedural chart flags above)")
    c.add_argument("--map-dir", type=str, default="maps/curriculum",
                   help="shared zip cache for downloaded maps")
    c.add_argument("--curriculum-state", type=str, default=None,
                   help="ladder state JSON (default <run-dir>/curriculum.json; "
                        "point at a previous run's file to resume its ladder)")
    c.add_argument("--start-star", type=float, default=1.0)
    c.add_argument("--star-step", type=float, default=0.2)
    c.add_argument("--star-band", type=float, default=0.1,
                   help="accept maps within ±band of the target star")
    c.add_argument("--speed-scale", type=float, default=0.3,
                   help="note speed = NJS * this (matches sightread.py)")
    c.add_argument("--new-map-weight", type=float, default=0.5,
                   help="fraction of episodes on the newest (un-FC'd) map; "
                        "the rest spread uniformly over the FC'd rungs")
    c.add_argument("--fc-eval-every", type=int, default=10,
                   help="updates between greedy full-combo checks")
    c.add_argument("--max-song-s", type=float, default=400.0,
                   help="skip maps longer than this; also sets the tick cap")
    return p.parse_args()


def cpu_state(policy):
    """Checkpoint state_dict on CPU regardless of the training device."""
    return {k: v.detach().cpu() for k, v in policy.state_dict().items()}


def greedy_fc_eval(policy, notes):
    """Play a fixed chart once with argmax actions (deterministic since the
    sim is deterministic). Aborts on the first combo break — FC needs zero.
    Returns (fc: bool, good, breaks, n_notes)."""
    dev = next(policy.parameters()).device
    env = RustBeatEnv(external_chart=notes, shaping_coef=0.0, seed=0)
    obs, _ = env.reset()
    good = breaks = 0
    done = False
    while not done and breaks == 0:
        with torch.no_grad():
            logits, _ = policy(
                torch.as_tensor(obs, dtype=torch.float32,
                                device=dev).unsqueeze(0))
        act = tuple(int(l.argmax()) for l in logits)
        obs, _, done, _, info = env.step(act)
        for _, reason, _pts in info["events"]:
            if reason == "ok":
                good += 1
            elif reason != "post":   # follow-throughs aren't note outcomes
                breaks += 1
    return breaks == 0 and good == len(notes), good, breaks, len(notes)


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
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    N, T = args.num_envs, args.horizon
    batch = N * T
    mb_size = batch // args.minibatches

    assert not (args.curriculum and args.env == "torch"), (
        "--curriculum needs --env rust (real-map sampler is Python-side)")
    torch_env = args.env == "torch"
    curriculum = None
    if args.curriculum:
        from beat_arena_env.curriculum import Curriculum
        state_path = args.curriculum_state or os.path.join(
            args.run_dir, "curriculum.json")
        curriculum = Curriculum(
            args.map_dir, state_path, start_star=args.start_star,
            star_step=args.star_step, star_band=args.star_band,
            speed_scale=args.speed_scale, new_map_weight=args.new_map_weight,
            max_song_s=args.max_song_s,
            acquire_rng=np.random.default_rng(args.seed))
        venv = BeatVecEnv(
            N, shaping_coef=args.shaping_coef, seed=args.seed,
            chart_sampler=curriculum.sample,
            max_steps=int(args.max_song_s * 30) + 30)
    elif args.env == "torch":
        from beat_arena_env.torch_env import TorchBeatVecEnv
        venv = TorchBeatVecEnv(
            N, note_rate=args.note_rate, colors=args.colors, dirs=tuple(args.dirs),
            cols=tuple(args.cols), rows=tuple(args.rows), chart_len_s=args.chart_len,
            min_hand_gap_s=args.min_hand_gap, note_speed=args.note_speed,
            note_speed_max=args.note_speed_max,
            continuous_dirs=args.continuous_dirs,
            shaping_coef=args.shaping_coef, seed=args.seed,
            chart_pool=args.chart_pool, device=device)
    else:
        venv = BeatVecEnv(
            N, note_rate=args.note_rate, colors=args.colors, dirs=tuple(args.dirs),
            cols=tuple(args.cols), rows=tuple(args.rows), chart_len_s=args.chart_len,
            min_hand_gap_s=args.min_hand_gap, note_speed=args.note_speed,
            note_speed_max=args.note_speed_max,
            continuous_dirs=args.continuous_dirs,
            shaping_coef=args.shaping_coef, seed=args.seed)

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        policy = load_policy(ckpt)
        print(f"warm-started from {args.init_from} (arch {policy.arch})",
              flush=True)
    else:
        policy = NotePolicy(embed=args.embed, trunk_dim=args.trunk_dim,
                            n_blocks=args.n_blocks, n_heads=args.n_heads)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy arch {policy.arch} — {n_params:,} params", flush=True)
    policy = policy.to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    obs = torch.as_tensor(venv.reset(), dtype=torch.float32, device=device)

    b_obs = torch.zeros((T, N, OBS_DIM), device=device)
    b_act = torch.zeros((T, N, 6), dtype=torch.long, device=device)
    b_logp = torch.zeros((T, N), device=device)
    b_rew = torch.zeros((T, N), device=device)
    b_done = torch.zeros((T, N), device=device)
    b_val = torch.zeros((T, N), device=device)

    ep_returns = []
    num_updates = max(1, args.total_steps // batch)
    global_step = 0
    t_rollout = t_update = 0.0
    start = time.time()

    print(f"=== batched PPO: {N} envs x {T} horizon = {batch} steps/update, "
          f"minibatch {mb_size}, device {device} ===", flush=True)

    for update in range(num_updates):
        if args.lr_decay != "none":
            frac = update / max(1, num_updates - 1)
            if args.lr_decay == "cosine":
                lr_now = args.lr_final + (args.lr - args.lr_final) * 0.5 * (
                    1.0 + math.cos(math.pi * frac))
            else:
                lr_now = args.lr + (args.lr_final - args.lr) * frac
            for g in opt.param_groups:
                g["lr"] = lr_now
        t0 = time.perf_counter()
        for t in range(T):
            b_obs[t] = obs
            with torch.no_grad():
                a, logp, _, v = policy.get_action_and_value(obs)
            b_act[t], b_logp[t], b_val[t] = a, logp, v
            if torch_env:
                nobs, rew, done, infos = venv.step(a)
                b_rew[t] = rew.to(b_rew.dtype)
                b_done[t] = done.to(b_done.dtype)
                obs = nobs
            else:
                nobs, rew, done, infos = venv.step(a.cpu().numpy())
                b_rew[t] = torch.as_tensor(rew, device=device)
                b_done[t] = torch.as_tensor(
                    done.astype(np.float32), device=device)
                obs = torch.as_tensor(nobs, dtype=torch.float32, device=device)
            for info in infos:
                ep_returns.append(info["episode_return"])
                if curriculum is not None:
                    curriculum.note_result(info)
            global_step += N
        if torch_env:
            ep_returns.extend(venv.pop_stats())
        t_rollout += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            _, _, _, next_val = policy.get_action_and_value(obs)
            adv = torch.zeros((T, N), device=device)
            lastgae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nonterminal = 1.0 - b_done[t]
                nextvalue = next_val if t == T - 1 else b_val[t + 1]
                delta = b_rew[t] + GAMMA * nextvalue * nonterminal - b_val[t]
                lastgae = delta + GAMMA * GAE_LAMBDA * nonterminal * lastgae
                adv[t] = lastgae
            returns = adv + b_val

        f_obs = b_obs.reshape(-1, OBS_DIM)
        f_act = b_act.reshape(-1, 6)
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = returns.reshape(-1)

        idx = np.arange(batch)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for s in range(0, batch, mb_size):
                mb = torch.as_tensor(idx[s:s + mb_size], device=device)
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

        if curriculum is not None and (update + 1) % args.fc_eval_every == 0:
            newest = curriculum.stages[curriculum.newest]
            fc, good, breaks, n = greedy_fc_eval(
                policy, curriculum.charts[curriculum.newest])
            print(f"[curriculum] FC check {newest['song']!r} "
                  f"{newest['stars']:.2f}★: good {good}/{n}, "
                  f"breaks {breaks}{'  *** FULL COMBO ***' if fc else ''}",
                  flush=True)
            print(curriculum.summary(), flush=True)
            if fc:
                torch.save({"state_dict": cpu_state(policy), "arch": policy.arch,
                            "encoder": policy.encoder, "update": update + 1,
                            "global_step": global_step,
                            "fc_map": newest["song"], "stars": newest["stars"]},
                           os.path.join(args.run_dir,
                                        f"fc_{newest['stars']:.2f}star.pt"))
                try:
                    curriculum.promote()
                except Exception as e:   # network down: retry next eval tick
                    print(f"[curriculum] promote failed ({e}); "
                          f"will retry next FC check", flush=True)

        if args.ckpt_every > 0 and (update + 1) % args.ckpt_every == 0:
            torch.save({"state_dict": cpu_state(policy), "arch": policy.arch,
                        "encoder": policy.encoder,
                        "update": update + 1, "global_step": global_step,
                        "recent_return": float(recent)},
                       os.path.join(args.run_dir, f"ckpt_upd{update+1:04d}.pt"))
        if args.time_cap > 0 and elapsed > args.time_cap:
            print(f"Reached --time-cap {args.time_cap:.0f}s.", flush=True)
            break

    recent = np.mean(ep_returns[-100:]) if ep_returns else float("nan")
    torch.save({"state_dict": cpu_state(policy), "arch": policy.arch,
                "encoder": policy.encoder,
                "update": update + 1, "global_step": global_step,
                "recent_return": float(recent)},
               os.path.join(args.run_dir, "final.pt"))
    tot = t_rollout + t_update
    print(f"THROUGHPUT: {global_step / (time.time() - start):,.0f} steps/s total | "
          f"rollout {100*t_rollout/tot:.0f}% update {100*t_update/tot:.0f}% | "
          f"N={N} T={T}", flush=True)


if __name__ == "__main__":
    main()
