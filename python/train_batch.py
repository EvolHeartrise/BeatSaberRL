"""Batched-rollout CPU PPO: N parallel Rust envs, ONE policy forward per
vector step. Same PPO math as train_smoke.py (GAE, dual-clip); the batch
shape is the variable under study — batch = num_envs x horizon.

Throughput study:  python train_batch.py --bench --num-envs 256
Real training:     python train_batch.py --num-envs 128 --total-steps 5000000

NOTE on science: changing num_envs changes the optimization regime
(updates-per-sample, minibatch size). Throughput comparisons are clean;
learning-curve comparisons across different batch shapes are NOT
apples-to-apples without retuning (LR, epochs).

CUDA graphs were REMOVED 2026-07-25: the launch storm they hid (12
separate heads + 12 Categoricals, ~120 tiny kernels per forward) was
eliminated at the source by the fused action head + last-block pruning
in policy.py, after which capture measured as pure overhead (static-
buffer copies + capturable Adam) at every shape tried. Checkpoints
saved by graphs-era runs load unchanged (they were normalized to the
eager Adam form at save time).

--fused-update replaces the minibatch forward+backward with the fused
CUDA kernels (policy_cuda.FusedPolicyTrainer): one launch each way,
recompute-based backward (activation memory O(obs minibatch)), packed
weights as the single Adam leaf; PPO loss math stays in torch on the
tiny (mb,12,3)/(mb,) outputs, so semantics cannot drift. d64/1-head
archs only. Weights are mirrored back into the torch policy once per
PPO update for rollout sampling and checkpoints. Checkpoints remain
load-compatible EXCEPT the Adam state (one packed param vs ~50):
--resume of a --fused-update checkpoint needs --fused-update again,
and vice versa. NOTE (2026-07-25): on speed it now trails the
optimized eager path — its value is the ~160x activation-memory
reduction (large-arch / large-minibatch headroom).
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

from beat_arena_env.vec_env import BeatVecEnv
from fp16_linear import bump_weights_version
from policy import HEAD_SIZES, NotePolicy, OBS_DIM, load_policy

A_DIM = len(HEAD_SIZES)  # 12 factored joint heads (env v6)

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


def fused_minibatch_losses(trainer, ent_coef, obs, act, logp, adv, ret):
    """The PPO minibatch losses over the fused forward. Identical math
    to the eager body: log_softmax gather == Categorical.log_prob,
    -(p * log p).sum == Categorical.entropy (policy.py's factored
    heads), then the same dual-clip PPO + value + entropy losses."""
    logits, value = trainer(obs)                    # (mb,12,3), (mb,)
    logsm = torch.log_softmax(logits, dim=-1)
    newlogp = logsm.gather(-1, act.unsqueeze(-1)).squeeze(-1).sum(-1)
    entropy = -(logsm.exp() * logsm).sum(-1).sum(-1)
    ratio = (newlogp - logp).exp()
    a_mb = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg_loss, _ = pg_loss_fn(ratio, a_mb)
    v_loss = 0.5 * ((value - ret) ** 2).mean()
    ent_loss = entropy.mean()
    loss = pg_loss - ent_coef * ent_loss + VF_COEF * v_loss
    return loss, ent_loss


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
    p.add_argument("--init-from", type=str, default=None,
                   help="warm-start WEIGHTS only; fresh optimizer + schedule")
    p.add_argument("--resume", type=str, default=None,
                   help="EXACT resume from a checkpoint of this trainer: "
                        "restores weights, Adam state, and the original LR "
                        "schedule position (lr/lr-final/lr-decay/total-steps "
                        "come from the checkpoint, CLI values ignored). The "
                        "env/chart stream restarts fresh — only the "
                        "optimization state is exact.")
    p.add_argument("--bench", action="store_true",
                   help="short run, prints steps/s split (rollout vs update)")
    p.add_argument("--device", type=str, default=None,
                   help="cuda|cpu (default: cuda if available)")
    p.add_argument("--env", choices=["rust", "cuda"], default=None,
                   help="cuda = custom CUDA-kernel sim (one launch per "
                        "step, obs stay on device). Default: cuda when "
                        "training on a CUDA device (the production path; "
                        "rust is ~4.6x slower there), else rust. (--env "
                        "torch, the tensor-op sim, was removed 2026-07-25)")
    p.add_argument("--chart-pool", type=int, default=65536,
                   help="pregenerated chart pool size for --env cuda "
                        "(default 65536 since 2026-07-26, user decision "
                        "after two paired 50M gates: 1024->8192 +91.3 "
                        "wpp (t=+8.31), 8192->65536 +80.9 wpp (t=+12.55, "
                        "score_acc +21%%, bad_cuts -31%%, FCs +38%%) — "
                        "chart diversity was the binding constraint. "
                        "Costs ~96 s pool gen + ~242 MB VRAM per leg; "
                        "~1.2 repeats/chart on a 50M leg, ~16 on 700M)")
    p.add_argument("--fp16", choices=["off", "update", "all"],
                   default="all",
                   help="fp16-storage activations (fp32 cuBLAS compute, "
                        "fp32 master weights, fp32 PPO math; 2026-07-25). "
                        "'all' (the DEFAULT since 2026-07-26, user "
                        "decision after the graph gate: +72.4 wpp over "
                        "'update', r1 2.0e-3 -> 6.7e-4 — rollout/update "
                        "consistency beats rollout precision) = rollout "
                        "AND update; 'update' = minibatch fwd+bwd only "
                        "(the pre-gate default); 'off' = full fp32 "
                        "(record-attempt fallback). Checkpoints stay "
                        "fp32 in every mode. Forced off on CPU devices")
    p.add_argument("--graph-rollout", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="capture the rollout step (policy fwd + sample + "
                        "env kernel) as ONE CUDA graph (round-4 #9; "
                        "DEFAULT since 2026-07-26 — gate-cleared with "
                        "--fp16 all: 1.20x rollout, +72.4 wpp paired). "
                        "Changes the sampling RNG stream vs eager. "
                        "Auto-downgrades (with a note) on CPU/rust env, "
                        "--fused-update, or capture failure")
    p.add_argument("--fp16-scale", type=float, default=131072.0,
                   help="static loss scale for --fp16 (2^17 from the "
                        "2026-07-25 grad probe: max|g| 2.2e-2, 26%% of "
                        "grad elements below fp16 normal range unscaled). "
                        "Halves automatically on overflow, never grows")
    p.add_argument("--fused-update", action="store_true",
                   help="run the minibatch forward+backward as fused CUDA "
                        "kernels (policy_cuda) with a recompute-based "
                        "backward — activation memory O(obs minibatch) — "
                        "and Adam over the packed weights. d64/1-head arch "
                        "only; --resume only onto another --fused-update "
                        "checkpoint. Memory feature, not a speed feature")
    # model arch (ignored with --init-from: the checkpoint's arch wins)
    p.add_argument("--embed", type=int, default=64)
    p.add_argument("--trunk-dim", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=1)
    # env config (generator v2 dials, SPEC.md §Chart generation)
    p.add_argument("--colors", choices=["right", "both"], default="right")
    p.add_argument("--cols", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--chart-len", type=float, default=20.0)
    p.add_argument("--note-speed-min", type=float, default=2.0,
                   help="speed random-walk clamp (m/s)")
    p.add_argument("--note-speed-max", type=float, default=8.0)
    p.add_argument("--note-prob", type=float, default=0.25,
                   help="P(note) per hand path per ~0.15 s generation tick")
    p.add_argument("--extra-prob", type=float, default=0.125,
                   help="P(a spawned note brings an extra along its swing line)")
    p.add_argument("--dot-prob", type=float, default=0.1,
                   help="P(a generated note turns into a dot)")
    p.add_argument("--bomb-prob", type=float, default=0.125,
                   help="P(bomb) per generation tick")
    p.add_argument("--effort-coef", type=float, default=1.0,
                   help="scales the tiered per-joint effort cost "
                        "(EFFORT_COST; 0 = free movement)")
    return p.parse_args()


def cpu_state(policy):
    """Checkpoint state_dict on CPU regardless of the training device."""
    return {k: v.detach().cpu() for k, v in policy.state_dict().items()}


def cpu_opt_state(opt):
    """Adam state_dict with tensors on CPU (load_state_dict moves them
    back to the params' device on resume). The lr/capturable
    normalization survives from the CUDA-graphs era (removed
    2026-07-25) — a no-op for eager runs, and it keeps this trainer
    byte-compatible with graphs-era checkpoints on file."""
    s = opt.state_dict()
    s["state"] = {k: {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
                      for kk, vv in st.items()}
                  for k, st in s["state"].items()}
    s["param_groups"] = [
        {**g, "lr": (float(g["lr"]) if torch.is_tensor(g["lr"]) else g["lr"]),
         "capturable": False}
        for g in s["param_groups"]]
    return s


def main():
    args = parse_args()
    if args.bench:
        args.total_steps = args.num_envs * args.horizon * 4
    resume_ckpt = None
    if args.resume:
        assert not args.init_from, "--resume and --init-from are exclusive"
        resume_ckpt = torch.load(args.resume, map_location="cpu",
                                 weights_only=True)
        if "optimizer" not in resume_ckpt:
            raise SystemExit(
                "--resume needs a checkpoint carrying optimizer + schedule "
                "state (saved by this trainer version); older checkpoints "
                "can only be --init-from warm-starts")
        sched = resume_ckpt["schedule"]
        args.lr, args.lr_final = sched["lr"], sched["lr_final"]
        args.lr_decay = sched["lr_decay"]
        if args.run_dir is None:      # continue in the checkpoint's run dir
            args.run_dir = os.path.dirname(os.path.abspath(args.resume))
    if args.run_dir is None:
        args.run_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "runs", "batch_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.run_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.env is None:
        args.env = "cuda" if device.type == "cuda" else "rust"
        print(f"--env not given: defaulting to '{args.env}'", flush=True)

    N, T = args.num_envs, args.horizon
    batch = N * T
    mb_size = batch // args.minibatches

    # --fp16 update is the DEFAULT (2026-07-25); downgrade quietly where
    # it cannot apply instead of crashing paths that never asked for it
    if args.fp16 != "off" and device.type != "cuda":
        print("--fp16: CPU device, forcing off", flush=True)
        args.fp16 = "off"
    if args.fp16 != "off" and args.fused_update:
        print("--fp16: incompatible with --fused-update (fp32 packed "
              "kernels), forcing off", flush=True)
        args.fp16 = "off"
    fp16_update = args.fp16 in ("update", "all")
    fp16_scale = args.fp16_scale
    # device-side loss-scaling state: NO host syncs in the minibatch
    # loop (an overflow zeroes the grads device-side; the audit rides
    # the existing once-per-update print sync)
    found_inf = torch.zeros(1, device=device)
    inv_scale = torch.zeros(1, device=device)
    inf_count = torch.zeros(1, device=device)
    ratio_diag = torch.zeros((), device=device)

    tensor_env = args.env == "cuda"
    # --graph-rollout is the DEFAULT (2026-07-26); downgrade with a note
    # where it cannot apply (same policy as the fp16 default above)
    if args.graph_rollout and (not tensor_env or device.type != "cuda"):
        print("--graph-rollout: needs --env cuda on a CUDA device, "
              "forcing off", flush=True)
        args.graph_rollout = False
    if args.graph_rollout and args.fused_update:
        print("--graph-rollout: incompatible with --fused-update, "
              "forcing off", flush=True)
        args.graph_rollout = False
    if tensor_env:
        from beat_arena_env.cuda_env import CudaBeatVecEnv as VecCls
        venv = VecCls(
            N, colors=args.colors, cols=tuple(args.cols),
            rows=tuple(args.rows), chart_len_s=args.chart_len,
            note_speed_min=args.note_speed_min,
            note_speed_max=args.note_speed_max,
            note_prob=args.note_prob, extra_prob=args.extra_prob,
            dot_prob=args.dot_prob, bomb_prob=args.bomb_prob,
            effort_coef=args.effort_coef,
            seed=args.seed, chart_pool=args.chart_pool, device=device)
    else:
        venv = BeatVecEnv(
            N, colors=args.colors, cols=tuple(args.cols),
            rows=tuple(args.rows), chart_len_s=args.chart_len,
            note_speed_min=args.note_speed_min,
            note_speed_max=args.note_speed_max,
            note_prob=args.note_prob, extra_prob=args.extra_prob,
            dot_prob=args.dot_prob, bomb_prob=args.bomb_prob,
            effort_coef=args.effort_coef, seed=args.seed)

    if resume_ckpt is not None:
        policy = load_policy(resume_ckpt)
    elif args.init_from:
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
    policy.fp16 = args.fp16 == "all"
    if args.fp16 != "off":
        print(f"--fp16 {args.fp16}: fp16-storage activations, fp32 "
              f"masters, static loss scale {fp16_scale:.0f}", flush=True)
    trainer = None
    if args.fused_update:
        assert device.type == "cuda", "--fused-update needs a CUDA device"
        from policy_cuda import FusedPolicyTrainer
        trainer = FusedPolicyTrainer(policy)  # raises on unsupported arch
        print("--fused-update: packed 107k-param leaf, fused fwd/bwd kernels",
              flush=True)
    params = trainer.parameters() if trainer is not None \
        else policy.parameters()
    # fused=True: one multi-tensor kernel instead of the foreach path's
    # ~10 launches per step (77 elementwise calls/minibatch measured in
    # the 2026-07-25 profile). Same per-element math, not guaranteed
    # bitwise vs foreach.
    fused_adam = device.type == "cuda"
    opt = torch.optim.Adam(params, lr=args.lr, fused=fused_adam)
    if resume_ckpt is not None:
        try:
            # stamp the fused flag into the SAVED groups before loading:
            # load_state_dict places the 'step' state tensors by the
            # saved group's fused/capturable flags, and fused Adam needs
            # them on the param device (a foreach-era checkpoint would
            # otherwise leave them on CPU and crash the first step)
            opt_sd = resume_ckpt["optimizer"]
            for g in opt_sd["param_groups"]:
                g["fused"] = fused_adam
            opt.load_state_dict(opt_sd)
        except ValueError as e:
            raise SystemExit(
                "--resume optimizer state does not match this update path "
                "(a --fused-update checkpoint stores Adam state over ONE "
                "packed param; the eager path over per-module params). "
                f"Re-run with the matching flag. [{e}]")

    obs = torch.as_tensor(venv.reset(), dtype=torch.float32, device=device)

    b_obs = torch.zeros((T, N, OBS_DIM), device=device)
    b_act = torch.zeros((T, N, A_DIM), dtype=torch.long, device=device)
    b_logp = torch.zeros((T, N), device=device)
    b_rew = torch.zeros((T, N), device=device)
    b_done = torch.zeros((T, N), device=device)
    b_val = torch.zeros((T, N), device=device)

    graph_roll = None
    if args.graph_rollout:
        try:
            from rollout_graph import GraphedRollout
            graph_roll = GraphedRollout(policy, venv)
            print(f"--graph-rollout: rollout step captured as one CUDA "
                  f"graph ({'fp16' if policy.fp16 else 'fp32'} rollout; "
                  f"sampling RNG stream differs from eager)", flush=True)
        except Exception as e:  # driver/WDDM capture failure: eager fallback
            graph_roll = None
            print(f"--graph-rollout: capture failed, falling back to the "
                  f"eager rollout [{type(e).__name__}: {e}]", flush=True)

    ep_returns = []
    if resume_ckpt is not None:
        num_updates = resume_ckpt["schedule"]["num_updates"]
        start_update = resume_ckpt["update"]
        global_step = resume_ckpt["global_step"]
        print(f"RESUMED {args.resume} at update {start_update}/{num_updates} "
              f"(step {global_step:,}; lr/schedule/Adam state restored; "
              f"env stream fresh)", flush=True)
    else:
        num_updates = max(1, args.total_steps // batch)
        start_update = 0
        global_step = 0
    steps_before = global_step
    t_rollout = t_update = 0.0
    start = time.time()

    print(f"=== batched PPO: {N} envs x {T} horizon = {batch} steps/update, "
          f"minibatch {mb_size}, device {device} ===", flush=True)

    for update in range(start_update, num_updates):
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
        if graph_roll is not None:
            graph_roll.refresh()  # in-place fp16 caches after opt.steps
            for t in range(T):
                b_obs[t] = venv._obs
                a, logp, v, rew, done = graph_roll.step()
                b_act[t], b_logp[t], b_val[t] = a, logp, v
                b_rew[t] = rew
                b_done[t] = done.to(b_done.dtype)
                global_step += N
            obs = venv._obs
        for t in range(T if graph_roll is None else 0):
            b_obs[t] = obs
            with torch.no_grad():
                a, logp, _, v = policy.get_action_and_value(obs)
            b_act[t], b_logp[t], b_val[t] = a, logp, v
            if tensor_env:
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
            global_step += N
        if tensor_env:
            ep_returns.extend(venv.pop_stats())
        # drain before reading the phase timer: kernel launches are
        # async, so without this the section boundary charges queued
        # work to whichever section blocks next (the 2026-07-25
        # attribution found the printed split ~inverted: real
        # production split is ~16% rollout / ~84% update). --bench only:
        # the drain serializes CPU enqueue with GPU execution and costs
        # ~2-8% wall, so production keeps async timers (and says so)
        if args.bench and device.type == "cuda":
            torch.cuda.synchronize()
        t_rollout += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            _, _, _, next_val = policy.get_action_and_value(obs)
            # everything non-recursive hoisted out of the T-loop (same
            # elementwise op order as the per-t version, so bit-exact);
            # only the scan itself stays sequential
            nonterm = 1.0 - b_done
            nextv = torch.cat([b_val[1:], next_val.unsqueeze(0)])
            deltas = b_rew + GAMMA * nextv * nonterm - b_val
            coef = (GAMMA * GAE_LAMBDA) * nonterm
            adv = torch.zeros((T, N), device=device)
            lastgae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                lastgae = deltas[t] + coef[t] * lastgae
                adv[t] = lastgae
            returns = adv + b_val

        f_obs = b_obs.reshape(-1, OBS_DIM)
        f_act = b_act.reshape(-1, A_DIM)
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = returns.reshape(-1)

        if fp16_update:
            policy.fp16 = True  # 'update' mode: rollout stayed fp32
            inv_scale.fill_(1.0 / fp16_scale)
        for ep_i in range(args.epochs):
            # device-side shuffle: a host-side index array costs a
            # synchronizing pageable H2D copy per minibatch (~26 ms/update
            # measured on WDDM)
            perm = torch.randperm(batch, device=device)
            for s in range(0, batch, mb_size):
                mb = perm[s:s + mb_size]
                if trainer is not None:  # fused minibatch body
                    loss, ent_loss = fused_minibatch_losses(
                        trainer, args.ent_coef, f_obs[mb], f_act[mb],
                        f_logp[mb], f_adv[mb], f_ret[mb])
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(trainer.parameters(),
                                             MAX_GRAD_NORM)
                    opt.step()
                    bump_weights_version()
                    continue
                _, newlogp, entropy, newval = policy.get_action_and_value(
                    f_obs[mb], f_act[mb])
                ratio = (newlogp - f_logp[mb]).exp()
                if fp16_update and ep_i == 0 and s == 0:
                    # "same policy" fixed point: |ratio-1| at epoch 1
                    # measures rollout/update precision consistency
                    # (device-side; read at the per-update print sync)
                    ratio_diag = (ratio - 1.0).abs().mean().detach()
                a_mb = f_adv[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg_loss, dual_frac = pg_loss_fn(ratio, a_mb)
                v_loss = 0.5 * ((newval - f_ret[mb]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * ent_loss + VF_COEF * v_loss
                opt.zero_grad(set_to_none=True)
                if fp16_update:
                    # static loss scale, ZERO host syncs: unscale + flag
                    # non-finites in one multi-tensor kernel, then zero
                    # every grad on overflow device-side. MUST run
                    # before clip_grad_norm_ (an inf grad would turn
                    # clip's 0.5/inf coef into 0*inf = NaN everywhere).
                    # Overflow steps become momentum-only Adam steps —
                    # bounded by lr, audited at the print.
                    (loss * fp16_scale).backward()
                    grads = [p.grad for p in policy.parameters()
                             if p.grad is not None]
                    torch._amp_foreach_non_finite_check_and_unscale_(
                        grads, found_inf, inv_scale)
                    ok = 1.0 - found_inf[0].clamp(max=1.0)  # 0-dim
                    torch._foreach_mul_(grads, ok)
                    inf_count += found_inf
                    found_inf.zero_()
                else:
                    loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                opt.step()
                bump_weights_version()  # invalidate fp16 replica caches
        if fp16_update:
            policy.fp16 = args.fp16 == "all"
        if trainer is not None:
            # mirror the packed weights into the torch policy: the next
            # rollout and any checkpoint below read the module's params
            trainer.unpack_to(policy)
        if args.bench and device.type == "cuda":
            torch.cuda.synchronize()  # see rollout-timer note
        t_update += time.perf_counter() - t0

        recent = np.mean(ep_returns[-100:]) if ep_returns else float("nan")
        elapsed = time.time() - start
        rate = (global_step - steps_before) / elapsed
        if fp16_update:
            # ONE host sync (same as the .item() it replaces) reads the
            # loss, the overflow audit, and the ratio diagnostic together
            ent_v, inf_v, r1_v = torch.stack(
                [ent_loss.detach(), inf_count[0], ratio_diag]).tolist()
            extra = f"  r1 {r1_v:.1e}"
            if inf_v > 0:
                fp16_scale = max(fp16_scale / 2, 1.0)  # halve, never grow
                inf_count.zero_()
                extra += (f"  [fp16] {int(inf_v)} overflow step(s) "
                          f"zeroed; scale -> {fp16_scale:.0f}")
            print(f"update {update+1:>3}/{num_updates}  steps "
                  f"{global_step:>9}  ep_return(last100) {recent:8.3f}  "
                  f"ent {ent_v:.3f}  {rate:8,.0f} steps/s  "
                  f"{elapsed:6.1f}s{extra}", flush=True)
        else:
            print(f"update {update+1:>3}/{num_updates}  steps {global_step:>9}  "
                  f"ep_return(last100) {recent:8.3f}  ent {ent_loss.item():.3f}  "
                  f"{rate:8,.0f} steps/s  {elapsed:6.1f}s", flush=True)

        if args.ckpt_every > 0 and (update + 1) % args.ckpt_every == 0:
            torch.save({"state_dict": cpu_state(policy), "arch": policy.arch,
                        "encoder": policy.encoder,
                        "update": update + 1, "global_step": global_step,
                        "recent_return": float(recent),
                        "optimizer": cpu_opt_state(opt),
                        "schedule": {"lr": args.lr, "lr_final": args.lr_final,
                                     "lr_decay": args.lr_decay,
                                     "num_updates": num_updates}},
                       os.path.join(args.run_dir, f"ckpt_upd{update+1:04d}.pt"))
        if args.time_cap > 0 and elapsed > args.time_cap:
            print(f"Reached --time-cap {args.time_cap:.0f}s.", flush=True)
            break

    recent = np.mean(ep_returns[-100:]) if ep_returns else float("nan")
    torch.save({"state_dict": cpu_state(policy), "arch": policy.arch,
                "encoder": policy.encoder,
                "update": update + 1, "global_step": global_step,
                "recent_return": float(recent),
                "optimizer": cpu_opt_state(opt),
                "schedule": {"lr": args.lr, "lr_final": args.lr_final,
                             "lr_decay": args.lr_decay,
                             "num_updates": num_updates}},
               os.path.join(args.run_dir, "final.pt"))
    tot = t_rollout + t_update
    split_kind = "exec-true" if args.bench else \
        "enqueue-attributed, NOT execution time — use --bench"
    print(f"THROUGHPUT: {(global_step - steps_before) / (time.time() - start):,.0f} steps/s total | "
          f"rollout {100*t_rollout/tot:.0f}% update {100*t_update/tot:.0f}% "
          f"({split_kind}) | N={N} T={T}", flush=True)


if __name__ == "__main__":
    main()
