"""Batched-rollout CPU PPO: N parallel Rust envs, ONE policy forward per
vector step. Same PPO math as train_smoke.py (GAE, dual-clip); the batch
shape is the variable under study — batch = num_envs x horizon.

Throughput study:  python train_batch.py --bench --num-envs 256
Real training:     python train_batch.py --num-envs 128 --total-steps 5000000

NOTE on science: changing num_envs changes the optimization regime
(updates-per-sample, minibatch size). Throughput comparisons are clean;
learning-curve comparisons across different batch shapes are NOT
apples-to-apples without retuning (LR, epochs).

On GPU both halves are LAUNCH-overhead-bound (107k-param policy, ~100
tiny kernels per forward): --cuda-graphs captures the whole minibatch
update (fwd+bwd+clip+Adam) as one CUDA graph, and --cuda-graphs-rollout
additionally captures the rollout forward + sampling. Runtime feature,
Pascal-safe (no Triton/compile); flag off = the eager path unchanged.

--fused-update replaces the minibatch forward+backward with the fused
CUDA kernels (policy_cuda.FusedPolicyTrainer): one launch each way,
recompute-based backward (activation memory O(obs minibatch)), packed
weights as the single Adam leaf; PPO loss math stays in torch on the
tiny (mb,12,3)/(mb,) outputs, so semantics cannot drift. d64/1-head
archs only; composes with --cuda-graphs (the fused minibatch body is
captured like the eager one). Weights are mirrored back into the torch
policy once per PPO update for rollout sampling and checkpoints.
Checkpoints remain load-compatible EXCEPT the Adam state (one packed
param vs ~50): --resume of a --fused-update checkpoint needs
--fused-update again, and vice versa.
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


_GRAPH_WARMUP = 3  # eager iterations before capture (torch's documented recipe)


class _Graphed:
    """CUDA-graph capture driver (the torch.cuda.graphs protocol):
    subclasses copy their inputs into static buffers and call _run(),
    which runs _GRAPH_WARMUP eager iterations on a side stream, then
    captures _compute() into a CUDAGraph, then replays it forever.
    _out holds the outputs (static tensors once captured)."""

    def __init__(self):
        self.graph = None
        self._out = None
        self._iters = 0
        self._side = torch.cuda.Stream()

    def _compute(self):
        raise NotImplementedError

    def _run(self):
        if self.graph is not None:
            self.graph.replay()
        elif self._iters < _GRAPH_WARMUP:
            self._side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._side):
                self._out = self._compute()
            torch.cuda.current_stream().wait_stream(self._side)
            self._iters += 1
        else:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):  # records, does NOT execute
                self._out = self._compute()
            self.graph.replay()
        return self._out


class GraphedMinibatchStep(_Graphed):
    """The whole PPO minibatch update — forward + losses + backward +
    grad clip + Adam step — as ONE CUDA graph. Warmup iterations are
    REAL optimization steps on the real minibatches (same math, side
    stream), so training semantics match the eager path throughout.
    Requires Adam(capturable=True) with a tensor lr the schedule mutates
    in place. Returns the entropy loss (static tensor after capture)."""

    def __init__(self, policy, opt, ent_coef, mb_size, device):
        super().__init__()
        self.policy, self.opt, self.ent_coef = policy, opt, ent_coef
        self.s_obs = torch.zeros(mb_size, OBS_DIM, device=device)
        self.s_act = torch.zeros(mb_size, A_DIM, dtype=torch.long,
                                 device=device)
        self.s_logp = torch.zeros(mb_size, device=device)
        self.s_adv = torch.zeros(mb_size, device=device)
        self.s_ret = torch.zeros(mb_size, device=device)

    def _compute(self):
        # the eager minibatch body, verbatim, over the static buffers
        # (set_to_none=False keeps .grad tensors at fixed addresses)
        self.opt.zero_grad(set_to_none=False)
        _, newlogp, entropy, newval = self.policy.get_action_and_value(
            self.s_obs, self.s_act)
        ratio = (newlogp - self.s_logp).exp()
        a_mb = (self.s_adv - self.s_adv.mean()) / (self.s_adv.std() + 1e-8)
        pg_loss, _ = pg_loss_fn(ratio, a_mb)
        v_loss = 0.5 * ((newval - self.s_ret) ** 2).mean()
        ent_loss = entropy.mean()
        loss = pg_loss - self.ent_coef * ent_loss + VF_COEF * v_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
        self.opt.step()
        # detach: returning ent_loss with its grad_fn would keep each
        # warmup iteration's autograd graph (and its side-stream
        # AccumulateGrad nodes) alive into capture — a stream mismatch
        return ent_loss.detach()

    def step(self, obs, act, logp, adv, ret):
        self.s_obs.copy_(obs)
        self.s_act.copy_(act)
        self.s_logp.copy_(logp)
        self.s_adv.copy_(adv)
        self.s_ret.copy_(ret)
        return self._run()


class GraphedFusedMinibatchStep(GraphedMinibatchStep):
    """--fused-update x --cuda-graphs: the fused minibatch body (fused
    fwd kernel + torch losses + fused bwd kernels + clip + Adam over the
    single packed param) captured exactly like the eager one."""

    def __init__(self, trainer, opt, ent_coef, mb_size, device):
        super().__init__(None, opt, ent_coef, mb_size, device)
        self.trainer = trainer

    def _compute(self):
        self.opt.zero_grad(set_to_none=False)
        loss, ent_loss = fused_minibatch_losses(
            self.trainer, self.ent_coef, self.s_obs, self.s_act,
            self.s_logp, self.s_adv, self.s_ret)
        loss.backward()
        nn.utils.clip_grad_norm_(self.trainer.parameters(), MAX_GRAD_NORM)
        self.opt.step()
        return ent_loss.detach()


class GraphedRolloutForward(_Graphed):
    """policy.get_action_and_value (forward + per-head multinomial
    sampling) over the fixed (N, obs) rollout batch as one CUDA graph.
    Sampling captures via torch's graph-safe Philox RNG: each replay
    advances the generator offset, so replays draw fresh actions (the
    random STREAM differs from eager sampling; the distribution and the
    on-policy math do not)."""

    def __init__(self, policy, num_envs, device):
        super().__init__()
        self.policy = policy
        self.s_obs = torch.zeros(num_envs, OBS_DIM, device=device)

    def _compute(self):
        with torch.no_grad():
            a, logp, _, v = self.policy.get_action_and_value(self.s_obs)
        return a, logp, v

    def __call__(self, obs):
        self.s_obs.copy_(obs)
        return self._run()


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
    p.add_argument("--env", choices=["rust", "torch", "cuda"], default="rust",
                   help="torch = GPU-resident tensor sim; cuda = custom "
                        "CUDA-kernel sim (one launch per step); both keep "
                        "obs on device")
    p.add_argument("--chart-pool", type=int, default=1024,
                   help="pregenerated chart pool size for --env torch/cuda")
    p.add_argument("--cuda-graphs", action="store_true",
                   help="capture the PPO minibatch update (fwd+bwd+clip+Adam) "
                        "as one CUDA graph — kills per-kernel launch overhead "
                        "(the 107k-param update is launch-bound). Runtime "
                        "feature, Pascal-safe; off = the eager path unchanged")
    p.add_argument("--cuda-graphs-rollout", action="store_true",
                   help="with --cuda-graphs: also capture the rollout "
                        "forward + action sampling (graph-safe Philox RNG)")
    p.add_argument("--fused-update", action="store_true",
                   help="run the minibatch forward+backward as fused CUDA "
                        "kernels (policy_cuda) with a recompute-based "
                        "backward — activation memory O(obs minibatch) — "
                        "and Adam over the packed weights. d64/1-head arch "
                        "only; composes with --cuda-graphs; --resume only "
                        "onto another --fused-update checkpoint")
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
    back to the params' device on resume). --cuda-graphs runs use a
    capturable Adam with a tensor lr; normalize both back to the plain
    eager form so checkpoints stay loadable either way (no-op for eager
    runs — lr is already a float and capturable already False)."""
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

    N, T = args.num_envs, args.horizon
    batch = N * T
    mb_size = batch // args.minibatches

    if args.cuda_graphs_rollout:
        assert args.cuda_graphs, "--cuda-graphs-rollout needs --cuda-graphs"
    if args.cuda_graphs:
        assert device.type == "cuda", "--cuda-graphs needs a CUDA device"
        assert batch % args.minibatches == 0, (
            "--cuda-graphs captures a fixed minibatch shape: num_envs x "
            "horizon must divide evenly by --minibatches")
        # distributions arg validation host-syncs (support checks) — illegal
        # inside graph capture; the math is unchanged without it
        torch.distributions.Distribution.set_default_validate_args(False)

    tensor_env = args.env in ("torch", "cuda")
    if tensor_env:
        if args.env == "cuda":
            from beat_arena_env.cuda_env import CudaBeatVecEnv as VecCls
        else:
            from beat_arena_env.torch_env import TorchBeatVecEnv as VecCls
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
    trainer = None
    if args.fused_update:
        assert device.type == "cuda", "--fused-update needs a CUDA device"
        from policy_cuda import FusedPolicyTrainer
        trainer = FusedPolicyTrainer(policy)  # raises on unsupported arch
        print("--fused-update: packed 107k-param leaf, fused fwd/bwd kernels",
              flush=True)
    params = trainer.parameters() if trainer is not None \
        else policy.parameters()
    if args.cuda_graphs:
        # graph replay re-runs recorded kernels, so Adam must READ its
        # hyperparams from device memory: capturable Adam + a tensor lr
        # the schedule fills in place (never rebinds)
        opt = torch.optim.Adam(params,
                               lr=torch.tensor(args.lr, device=device),
                               capturable=True)
    else:
        opt = torch.optim.Adam(params, lr=args.lr)
    if resume_ckpt is not None:
        try:
            opt.load_state_dict(resume_ckpt["optimizer"])
        except ValueError as e:
            raise SystemExit(
                "--resume optimizer state does not match this update path "
                "(a --fused-update checkpoint stores Adam state over ONE "
                "packed param; the eager path over per-module params). "
                f"Re-run with the matching flag. [{e}]")
        if args.cuda_graphs:  # checkpoints store the plain eager form
            for g in opt.param_groups:
                g["capturable"] = True
                g["lr"] = torch.tensor(float(g["lr"]), device=device)
            for st in opt.state.values():
                st["step"] = st["step"].to(device)

    obs = torch.as_tensor(venv.reset(), dtype=torch.float32, device=device)

    b_obs = torch.zeros((T, N, OBS_DIM), device=device)
    b_act = torch.zeros((T, N, A_DIM), dtype=torch.long, device=device)
    b_logp = torch.zeros((T, N), device=device)
    b_rew = torch.zeros((T, N), device=device)
    b_done = torch.zeros((T, N), device=device)
    b_val = torch.zeros((T, N), device=device)

    g_update = g_rollout = None
    if args.cuda_graphs:
        if trainer is not None:
            g_update = GraphedFusedMinibatchStep(trainer, opt, args.ent_coef,
                                                 mb_size, device)
        else:
            g_update = GraphedMinibatchStep(policy, opt, args.ent_coef,
                                            mb_size, device)
        if args.cuda_graphs_rollout:
            g_rollout = GraphedRolloutForward(policy, N, device)

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
            if args.cuda_graphs:
                for g in opt.param_groups:
                    g["lr"].fill_(lr_now)  # in place: captured kernels read it
            else:
                for g in opt.param_groups:
                    g["lr"] = lr_now
        t0 = time.perf_counter()
        for t in range(T):
            b_obs[t] = obs
            if g_rollout is not None:
                a, logp, v = g_rollout(obs)
            else:
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
        t_rollout += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            if g_rollout is not None:
                _, _, next_val = g_rollout(obs)
            else:
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
        f_act = b_act.reshape(-1, A_DIM)
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = returns.reshape(-1)

        idx = np.arange(batch)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for s in range(0, batch, mb_size):
                mb = torch.as_tensor(idx[s:s + mb_size], device=device)
                if g_update is not None:
                    ent_loss = g_update.step(
                        f_obs[mb], f_act[mb], f_logp[mb], f_adv[mb], f_ret[mb])
                    continue
                if trainer is not None:  # eager fused minibatch body
                    loss, ent_loss = fused_minibatch_losses(
                        trainer, args.ent_coef, f_obs[mb], f_act[mb],
                        f_logp[mb], f_adv[mb], f_ret[mb])
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(trainer.parameters(),
                                             MAX_GRAD_NORM)
                    opt.step()
                    continue
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
        if trainer is not None:
            # mirror the packed weights into the torch policy: the next
            # rollout (eager or graphed — copies land in-place) and any
            # checkpoint below read the module's params
            trainer.unpack_to(policy)
        t_update += time.perf_counter() - t0

        recent = np.mean(ep_returns[-100:]) if ep_returns else float("nan")
        elapsed = time.time() - start
        rate = (global_step - steps_before) / elapsed
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
    print(f"THROUGHPUT: {(global_step - steps_before) / (time.time() - start):,.0f} steps/s total | "
          f"rollout {100*t_rollout/tot:.0f}% update {100*t_update/tot:.0f}% | "
          f"N={N} T={T}", flush=True)


if __name__ == "__main__":
    main()
