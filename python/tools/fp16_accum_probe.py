"""fp16-accumulate numerics probe (issue #3): headroom, output error,
gradient error and rollout/update consistency of the --fp16-accum modes
on a trained checkpoint and REAL observations.

    # 1. harvest real obs: the trained trunk plays the v58 training env
    python tools/fp16_accum_probe.py harvest --ckpt runs/std_probe/trunk_final.pt \
        --out obs_std.pt
    # 2. per-site headroom (max |partial sum| / 65504) of every GEMM the
    #    modes flip, forward and loss-scaled dgrad
    python tools/fp16_accum_probe.py headroom --ckpt ... --obs obs_std.pt
    # 3. logits / value / logp error and gradient error vs an fp32
    #    reference, per mode, over several minibatch seeds; r1 check
    python tools/fp16_accum_probe.py error --ckpt ... --obs obs_std.pt

Headroom: an fp16 accumulator saturates at 65504. For every flipped call
(fp16_linear._acc16 sites: the torch forward mm/addmm/F.linear and the
dgrad mm) the probe recomputes, in fp32 on a row sample, every PREFIX sum
of the K-length dot products (the natural-order proxy for the running
accumulator; cuBLAS tiles K in 16/32-wide steps, so its partials are a
subset of these orderings up to split-k) and reports max |prefix| and
max |output|. Dgrad runs at the production loss scale (--scale, 2^17 =
the train_batch --fp16-scale cap; the dynamic scaler only ever lowers
it).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import fp16_linear as fl  # noqa: E402
from policy import hl_loss, load_policy  # noqa: E402

FP16_MAX = 65504.0


def cmd_harvest(a):
    import bench_env_kernel as bek
    torch.manual_seed(0)
    gen = torch.Generator(device="cuda").manual_seed(1)
    env = bek.make_env("stdtr", a.num_envs, torch.float32,
                       std_levels=a.levels)
    pol = load_policy(torch.load(a.ckpt, map_location="cpu",
                                 weights_only=True)).cuda().eval()
    obs = env.reset()
    keep = []
    for t in range(a.ticks):
        act = bek.actions_for(pol, obs, gen, a.num_envs, sample=True)
        obs, _, _, _ = env.step(act)
        if t >= a.ticks - a.keep * a.every and (a.ticks - 1 - t) % a.every == 0:
            keep.append(obs.clone())
    obs = torch.cat(keep)
    torch.save(obs.cpu(), a.out)
    print(f"harvested {tuple(obs.shape)} obs -> {a.out}")


def build(ckpt, mode, fp16=True):
    pol = load_policy(torch.load(ckpt, map_location="cpu",
                                 weights_only=True)).cuda()
    pol.fp16_accum = pol.arch["fp16_accum"] = mode
    if fp16:
        pol.fp16 = True
        pol.flatten_fp16()
    return pol


def load_obs(path, n, seed):
    obs = torch.load(path)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(obs.shape[0], generator=g)[:n]
    return obs[idx].cuda()


def minibatch(pol, obs, seed):
    """Real obs, the policy's own sampled actions and logp (ratio 1 at
    the loaded weights, like the first epoch of an update), N(0,1)
    advantages and normalized returns."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        logits, _ = pol.forward_stacked(obs)
        u = torch.rand(logits.shape, device="cuda", generator=g).clamp_(1e-10, 1)
        act = (logits.float() - torch.log(-torch.log(u))).argmax(-1)
        logp = torch.log_softmax(logits.float(), -1).gather(
            -1, act.unsqueeze(-1)).squeeze(-1).sum(-1)
    adv = torch.randn(obs.shape[0], device="cuda", generator=g)
    ret = torch.randn(obs.shape[0], device="cuda", generator=g)
    return obs, act, logp, adv, ret


def loss_backward(pol, data, scale):
    import bench_policy_update as bpu
    for p in pol.parameters():
        p.grad = None
    bpu.eager_update(pol, None, *data, step=False, scale=scale)
    return {n: p.grad.detach().float() / scale
            for n, p in pol.named_parameters() if p.grad is not None}


# ---- headroom --------------------------------------------------------------

def _prefix_max(A, Bm, rows):
    """max |prefix sum over k| of A[rows] @ Bm, in fp32."""
    best = 0.0
    Bf = Bm.float()
    K, N = Bf.shape
    step = max(1, (1 << 25) // (K * N))
    for i in range(0, rows.numel(), step):
        r = rows[i:i + step]
        P = (A[r].float().unsqueeze(-1) * Bf.unsqueeze(0)).cumsum(1)
        best = max(best, P.abs().max().item())
    return best


def cmd_headroom(a):
    pol = build(a.ckpt, "all")
    obs = load_obs(a.obs, a.mb, 0)
    data = minibatch(pol, obs, 0)
    stats = {}
    orig = fl._acc16
    phase = {"p": "fwd"}

    def hook(on, fn, *args):
        out = orig(on, fn, *args)
        if fn is torch.addmm:
            A, Bm = args[1], args[2]
        elif fn is torch.mm:
            A, Bm = args[0], args[1]
        else:                                  # F.linear(x, w, b)
            A, Bm = args[0].reshape(-1, args[0].shape[-1]), args[1].t()
        kind = "dgrad" if phase["p"] == "bwd" else "fwd"
        key = (kind, A.shape[1], Bm.shape[1], A.shape[0])
        g = torch.Generator(device="cuda").manual_seed(len(stats))
        rows = torch.randint(0, A.shape[0], (min(a.rows, A.shape[0]),),
                             device="cuda", generator=g)
        s = stats.setdefault(key, [0, 0.0, 0.0, 0.0])
        s[0] += 1
        s[1] = max(s[1], A.abs().max().item())
        s[2] = max(s[2], _prefix_max(A, Bm, rows))
        s[3] = max(s[3], out.abs().max().item())
        return out

    fl._acc16 = hook
    try:
        import bench_policy_update as bpu
        for p in pol.parameters():
            p.grad = None
        # grad-mode forward (records fwd sites), then the scaled backward
        _, newlogp, entropy, _ = pol.get_action_and_value(data[0], data[1])
        ratio = (newlogp - data[2]).exp()
        a_mb = (data[3] - data[3].mean()) / (data[3].std() + 1e-8)
        loss = (bpu.pg_loss_fn(ratio, a_mb) - bpu.ENT_COEF * entropy.mean()
                + bpu.VF_COEF * hl_loss(pol._value_logits, data[4],
                                        pol.value_centers, pol.value_sigma))
        phase["p"] = "bwd"
        (loss * a.scale).backward()
    finally:
        fl._acc16 = orig
    print(f"headroom at loss scale {a.scale:.0f} (mb {a.mb}, {a.rows} "
          f"sampled rows per call; fp16 max {FP16_MAX:.0f})")
    print(f"{'site':6} {'K':>5} {'N':>5} {'rows':>8} {'calls':>5} "
          f"{'max|in|':>10} {'max|prefix|':>12} {'max|out|':>10} "
          f"{'prefix/65504':>13}")
    worst = 0.0
    for (kind, K, N, R), (c, mi, mp, mo) in sorted(stats.items()):
        worst = max(worst, mp / FP16_MAX)
        print(f"{kind:6} {K:>5} {N:>5} {R:>8} {c:>5} {mi:>10.3g} "
              f"{mp:>12.4g} {mo:>10.4g} {mp / FP16_MAX:>13.2e}")
    print(f"worst prefix/65504: {worst:.2e} "
          f"(headroom x{1 / max(worst, 1e-30):.0f})")


# ---- error -----------------------------------------------------------------

def _fwd(pol, obs):
    with torch.no_grad():
        logits, value = pol.forward_stacked(obs)
    return logits.float(), value.float()


def _groups(name):
    for pre, g in (("blocks", "blocks"), ("trunk", "trunk"),
                   ("heads", "heads"), ("critic", "critic")):
        if name.startswith(pre):
            return g
    return "enc"


def cmd_error(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = build(a.ckpt, "off", fp16=False)
    pols = {m: build(a.ckpt, m) for m in a.modes}
    obs_all = load_obs(a.obs, a.mb, 0)
    # the fp32 reference forwards the fp16-ROUNDED obs (what every fp16
    # consumer reads), so the error is the network arithmetic only
    lr, vr = _fwd(ref, obs_all.half().float())
    lsm_r = torch.log_softmax(lr, -1)
    act = lsm_r.argmax(-1)
    print(f"forward error vs fp32 reference (mb {a.mb}, real obs)")
    print(f"{'mode':5} {'logit max':>10} {'logit mean':>11} {'logp max':>9} "
          f"{'logp mean':>10} {'KL mean':>9} {'value max':>10} {'value mean':>11}"
          f"   rollout(2M rows) vs update(M rows)")
    for m, pol in pols.items():
        l, v = _fwd(pol, obs_all)
        lsm = torch.log_softmax(l, -1)
        dl = (l - lr).abs()
        lp = lsm.gather(-1, act.unsqueeze(-1)).squeeze(-1).sum(-1)
        lpr = lsm_r.gather(-1, act.unsqueeze(-1)).squeeze(-1).sum(-1)
        kl = (lsm_r.exp() * (lsm_r - lsm)).sum(-1).sum(-1)
        dv = (v - vr).abs()
        # r1 at the production shapes: the rollout's grad-free forward
        # at 2x rows (N envs = 2 x minibatch) on fp32 obs vs the update's
        # grad-mode forward on the half obs copy of the first half
        n = a.mb // 2
        lg, _ = pol.forward_stacked(obs_all[:n].half())
        lpu = torch.log_softmax(lg.detach().float(), -1).gather(
            -1, act[:n].unsqueeze(-1)).squeeze(-1).sum(-1)
        del lg
        pol._value_logits = None
        r1 = (lpu - lp[:n]).abs().max().item()
        # production's r1 diag: mean |ratio - 1| (train_batch ratio_diag)
        r1m = (lpu - lp[:n]).exp().sub(1).abs().mean().item()
        # same-M: the grad-free forward at the update's row count must
        # be bitwise the grad-mode one (the r1 contract of the kernels)
        ls, _ = _fwd(pol, obs_all[:n])
        lps = torch.log_softmax(ls, -1).gather(
            -1, act[:n].unsqueeze(-1)).squeeze(-1).sum(-1)
        same = torch.equal(lps, lpu)
        print(f"{m:5} {dl.max().item():>10.3e} {dl.mean().item():>11.3e} "
              f"{(lp - lpr).abs().max().item():>9.3e} "
              f"{(lp - lpr).abs().mean().item():>10.3e} "
              f"{kl.mean().item():>9.2e} {dv.max().item():>10.3e} "
              f"{dv.mean().item():>11.3e}   r1 max|dlogp| {r1:.3g} "
              f"mean|ratio-1| {r1m:.3g} same-M bitwise {same}")
        torch.cuda.empty_cache()
    print(f"\ngradient error vs fp32 reference (relative Frobenius, "
          f"mb {a.gmb}, loss scale {a.scale:.0f})")
    groups = ["enc", "blocks", "trunk", "heads", "critic"]
    print(f"{'seed':>4} {'mode':5} {'all':>9} {'cos':>9} " +
          " ".join(f"{g:>9}" for g in groups))
    for seed in a.seeds:
        obs = load_obs(a.obs, a.gmb, 100 + seed)
        data = minibatch(ref, obs.half().float(), seed)
        gr = loss_backward(ref, data, 1.0)
        for m, pol in pols.items():
            gm = loss_backward(pol, data, a.scale)
            num = {g: 0.0 for g in groups}
            den = {g: 0.0 for g in groups}
            dot = nr = nm = 0.0
            for n, x in gr.items():
                y = gm[n]
                g = _groups(n)
                num[g] += (y - x).square().sum().item()
                den[g] += x.square().sum().item()
                dot += (x * y).sum().item()
                nr += x.square().sum().item()
                nm += y.square().sum().item()
            tot = math.sqrt(sum(num.values()) / sum(den.values()))
            print(f"{seed:>4} {m:5} {tot:>9.3e} "
                  f"{dot / math.sqrt(nr * nm):>9.6f} " +
                  " ".join(f"{math.sqrt(num[g] / den[g]):>9.3e}"
                           for g in groups), flush=True)
        del gr


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["harvest", "headroom", "error"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--obs", default="obs_std.pt")
    ap.add_argument("--out", default="obs_std.pt")
    ap.add_argument("--num-envs", type=int, default=16384)
    ap.add_argument("--levels", type=int, default=2048)
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--mb", type=int, default=16384)
    ap.add_argument("--gmb", type=int, default=8192)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--scale", type=float, default=131072.0)
    ap.add_argument("--modes", nargs="+", default=["off", "fwd", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    {"harvest": cmd_harvest, "headroom": cmd_headroom,
     "error": cmd_error}[a.cmd](a)


if __name__ == "__main__":
    main()
