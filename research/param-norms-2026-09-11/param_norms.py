"""Parameter-norm census over Beat Arena checkpoints (CPU)."""
import glob, json, os, re, sys, csv
import torch

ROOT = os.path.expanduser("~/beat-arena-volume-backup")
RUNS = [
    ("v38t32", "beat-arena-v38t32/runs/v38t32_trunk"),
    ("v38t32", "beat-arena-v38t32/runs/v38t32_cut02250_cd200_s1"),
    ("v38t32", "beat-arena-v38t32/runs/v38t32_cut04500_cd200_s1"),
    ("af50n", "beat-arena-af50/runs/v38t32af50_trunk"),
    ("af50n", "beat-arena-af50n/runs/af50n_trunk"),
    ("af50n", "beat-arena-af50n/runs/ctrl_trunk"),
    ("t64", "beat-arena-t64/runs/t64_g965"),
    ("t64", "beat-arena-t64/runs/t64_g98"),
    ("t64", "beat-arena-t64/runs/t64_g98_leaf_cd200"),
]

def group_of(name):
    if name.startswith(("note_mlp", "saber_mlp", "body_mlp")):
        return "enc"
    if name.startswith("blocks."):
        return "blocks"
    if name.startswith("trunk."):
        return "trunk"
    if name.startswith("heads."):
        return "heads"
    if name.startswith(("critic_", "value_head")):
        return "critic"
    return "other"

def upd_of(fname):
    m = re.search(r"upd(\d+)", fname)
    return int(m.group(1)) if m else None

rows = []
for camp, rel in RUNS:
    d = os.path.join(ROOT, rel)
    try:
        args = json.load(open(os.path.join(d, "args.json")))
    except Exception:
        args = {}
    files = sorted(glob.glob(os.path.join(d, "*.pt")),
                   key=lambda f: (upd_of(f) is None, upd_of(f) or 0))
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        sq = {"enc": 0.0, "blocks": 0.0, "trunk": 0.0, "heads": 0.0,
              "critic": 0.0, "other": 0.0}
        ln_gain = []
        qk = []
        n_total = 0
        for k, v in sd.items():
            v = v.float()
            sq[group_of(k)] += float((v * v).sum())
            n_total += v.numel()
            if k.endswith("ln_attn.weight") or k.endswith("ln_ffn.weight") \
                    or k == "critic_ln.weight":
                ln_gain.append(float(v.mean()))
        for b in range(8):
            q = sd.get(f"blocks.{b}.q_proj.weight")
            kk = sd.get(f"blocks.{b}.k_proj.weight")
            if q is None:
                break
            sq_q = float(torch.linalg.matrix_norm(q.float(), ord=2))
            sq_k = float(torch.linalg.matrix_norm(kk.float(), ord=2))
            qk.append(sq_q * sq_k)
        total = sum(sq.values()) ** 0.5
        rows.append({
            "campaign": camp, "run": os.path.basename(d),
            "file": os.path.basename(f),
            "role": ck.get("wsd_role", "?"),
            "update": ck.get("update"), "global_step": ck.get("global_step"),
            "acc": ck.get("recent_acc"), "ret": ck.get("recent_return"),
            "n_params": n_total,
            "total": total, "rms": total / n_total ** 0.5,
            "enc": sq["enc"] ** 0.5, "blocks": sq["blocks"] ** 0.5,
            "trunk": sq["trunk"] ** 0.5, "heads": sq["heads"] ** 0.5,
            "critic": sq["critic"] ** 0.5,
            "ln_gain_mean": sum(ln_gain) / len(ln_gain) if ln_gain else None,
            "qk_spec_max": max(qk) if qk else None,
            "lr": args.get("lr"), "T": args.get("horizon", args.get("T")),
        })
        print(f"{camp:7s} {os.path.basename(d):24s} {os.path.basename(f):32s} "
              f"upd {rows[-1]['update']!s:>6} step {rows[-1]['global_step']!s:>14} "
              f"acc {rows[-1]['acc'] if rows[-1]['acc'] is None else round(rows[-1]['acc'],4)!s:>7} "
              f"|θ| {total:8.2f} enc {sq['enc']**0.5:6.2f} blk {sq['blocks']**0.5:6.2f} "
              f"trunk {sq['trunk']**0.5:6.2f} heads {sq['heads']**0.5:6.2f} "
              f"critic {sq['critic']**0.5:6.2f} ln {rows[-1]['ln_gain_mean']:.3f} "
              f"qk {rows[-1]['qk_spec_max']:6.2f}", flush=True)

out = sys.argv[1] if len(sys.argv) > 1 else "param_norms.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("args keys sample:", sorted(args.keys())[:40])
print("wrote", out, len(rows), "rows")
