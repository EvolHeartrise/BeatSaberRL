"""Numerics gate for the perf ladder: dump a forward + backward
fingerprint under the current toggle configuration, or compare two
dumps.

    python perf_numerics.py dump out.pt [--ckpt runs/...]
    python perf_numerics.py compare base.pt other.pt [--tol 0.02]

The fingerprint (production fp16 config: fp16 stream, flat replicas):
  logits/value   no-grad forward on a fixed obs batch (rollout path)
  g_*            full grads of representative masters after one
                 grad-mode forward + synthetic loss backward (update
                 path — exercises packed backward routing, relu_bwd
                 widths, attention backward)

compare prints max-abs diffs and fails past --tol. Expected scales:
bit-identical for the gemm-width dispatch, ~1e-3 (fp16 last-rounding)
for fold/sdpa/packed. A grad diff at O(1) means a ROUTING bug (grads
landing on the wrong master), not rounding — that is what this gate
exists to catch.
"""

from __future__ import annotations

import argparse
import sys

import torch


def build(ckpt_path):
    from policy import NotePolicy, load_policy
    device = torch.device("cuda")
    try:
        policy = load_policy(torch.load(ckpt_path, map_location="cpu",
                                        weights_only=True))
        print(f"weights: {ckpt_path}")
    except FileNotFoundError:
        torch.manual_seed(0)
        policy = NotePolicy(n_gears=5)
        print("weights: random init")
    policy = policy.to(device)
    policy.fp16 = True
    policy.flatten_fp16()
    return policy, device


def dump(args):
    import bench_policy_update as bpu
    policy, device = build(args.ckpt)
    obs, act, logp, adv, ret = bpu.make_minibatch(4096, device,
                                                  policy.n_gears)
    out = {}
    with torch.no_grad():
        logits, value = policy.forward_stacked(obs)
    out["logits"] = logits.float().cpu()
    out["value"] = value.float().cpu()

    _, newlogp, entropy, newval = policy.get_action_and_value(obs, act)
    loss = (newlogp.mean() + 0.1 * entropy.mean()
            + 0.5 * ((newval - ret) ** 2).mean())
    loss.backward()
    out["loss"] = loss.detach().float().cpu()
    blk = policy.blocks[0]
    probes = {"g_q0": blk.q_proj.weight, "g_k0": blk.k_proj.weight,
              "g_v0": blk.v_proj.weight, "g_ffn0": blk.ffn[0].weight,
              "g_ffn2": blk.ffn[2].weight,
              "g_trunk": policy.trunk[0].weight,
              "g_head0": policy.heads[0].weight,
              "g_head28": policy.heads[28].weight,
              "g_value": policy.value_head.weight,
              # bias grads ride relu_bwd's fixed-order partials at the
              # NEW widths (512/256) — probe them explicitly
              "g_ffn0_b": blk.ffn[0].bias,
              "g_trunk_b": policy.trunk[0].bias,
              "g_q0_b": blk.q_proj.bias}
    for k, p in probes.items():
        out[k] = (p.grad.detach().float().cpu() if p.grad is not None
                  else torch.zeros(1))
    torch.save(out, args.out)
    print(f"dumped {args.out}")


def compare(args):
    a = torch.load(args.base, weights_only=True)
    b = torch.load(args.other, weights_only=True)
    worst = 0.0
    for k in a:
        if a[k].shape != b[k].shape:
            print(f"{k:>10}: SHAPE {tuple(a[k].shape)} vs "
                  f"{tuple(b[k].shape)}")
            worst = float("inf")
            continue
        d = (a[k] - b[k]).abs().max().item()
        scale = a[k].abs().max().item() or 1.0
        print(f"{k:>10}: max|diff| {d:.3e}  (max|base| {scale:.3e})")
        worst = max(worst, d / scale)
    verdict = "PASS" if worst <= args.tol else "FAIL"
    print(f"NUMERICS {verdict} worst_rel {worst:.3e} tol {args.tol}")
    sys.exit(0 if worst <= args.tol else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dump")
    d.add_argument("out")
    d.add_argument("--ckpt", default="runs/perf_ref.pt")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("other")
    c.add_argument("--tol", type=float, default=0.02)
    args = ap.parse_args()
    dump(args) if args.mode == "dump" else compare(args)


if __name__ == "__main__":
    main()
