"""Rollout-shape ladder gate (2026-08-08, RTX 4090): 50M x 3 arms.

The 4090 N-scaling curve (order-balanced, exec-true --bench, ranked
instrument confirmed) is 116k steps/s at 4096/32, 171k at 8192/16,
218k at 16384/8 — the knee sits at 8 blocks/SM on 128 SMs. All three
ladder arms preserve batch = 131,072, so the Protein-tuned hypers,
LR/entropy schedules, minibatch shape, and checkpoint indices are
inherited legitimately; ONLY the rollout shape changes. The science
delta is GAE truncation: at these hypers gamma*lambda = 0.89669, so
trace weight past the horizon is 3.05% at T=32, 17.4% at T=16, 41.8%
at T=8 — the T=8 arm leans hardest on the value bootstrap and is the
arm this gate exists to test. Physics is bitwise N-invariant (one
thread per env).

The 4096/32 baseline arm is trained FRESH on this pod rather than
reusing the G1 numbers: those came from the GTX 1080, and cross-
instrument comparisons are banned (2026-08-08 rule). Judged like the
wide gate: bench upp on the frozen catalog, final checkpoints, same
seed. Adopt the fastest non-inferior arm.

  python v10_shape_ladder.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = sys.executable
BASE = os.path.join("runs", "v10_shape_ladder")
TOTAL_STEPS = 50_000_000
CKPT_EVERY = 95            # same update indices as the smoke/gate line

ARMS = [                   # (name, num_envs, horizon) — batch 131,072 each
    ("a4096x32", 4096, 32),
    ("b8192x16", 8192, 16),
    ("c16384x8", 16384, 8),
]

HYPERS = {"lr": 0.00296092,
          "ent_coef": 0.0331038, "ent_final": 0.013929,
          "gamma": 0.914958, "gae_lambda": 0.980024,
          "effort_coef": 4e-4}


def bench(ck, out):
    try:
        r = subprocess.run([PY, "bench_scoresaber.py", "--ckpt", ck,
                            "--out", out, "--num-envs", "2048"],
                           capture_output=True, text=True, timeout=10800)
    except subprocess.TimeoutExpired:
        print(f"BENCH TIMEOUT {ck} (10800s)", flush=True)
        return None
    if r.returncode != 0 or not os.path.exists(out):
        print(f"BENCH FAILED {ck}: {(r.stderr or '')[-300:]}", flush=True)
        return None
    b = json.load(open(out, encoding="utf-8"))
    fcs = sum(1 for lv in b["levels"] if lv["full_combo"])
    acc = [lv["accuracy"] for lv in b["levels"] if "accuracy" in lv]
    return dict(upp=b["total_unweighted_pp"], wpp=b["total_weighted_pp"],
                fcs=fcs, acc=sum(acc) / len(acc) if acc else 0.0)


def train_arm(name, n, t):
    h = HYPERS
    run_dir = os.path.join(BASE, name)
    ck = os.path.join(run_dir, "final.pt")
    if os.path.exists(ck):
        print(f"[{name}] final.pt exists — skipping train", flush=True)
        return ck
    os.makedirs(run_dir, exist_ok=True)
    cmd = [PY, "train_batch.py",
           "--num-envs", str(n), "--horizon", str(t),
           "--lr-decay", "cosine", "--seed", "0",
           "--ranked-maps", "--seg-len", "25",
           "--chart-pool", "24576",
           "--gears", "5",
           "--lr", f"{h['lr']:.6g}",
           "--ent-coef", f"{h['ent_coef']:.6g}",
           "--ent-final", f"{h['ent_final']:.6g}",
           "--gamma", f"{h['gamma']:.6g}",
           "--gae-lambda", f"{h['gae_lambda']:.6g}",
           "--effort-coef", f"{h['effort_coef']:.6g}",
           "--total-steps", str(TOTAL_STEPS),
           "--ckpt-every", str(CKPT_EVERY),
           "--run-dir", run_dir]
    print(f"[{name}] train 50M ({n}/{t}) -> {run_dir}", flush=True)
    t0 = time.time()
    with open(os.path.join(run_dir, "train.log"), "w",
              encoding="utf-8") as log:
        try:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            print(f"[{name}] TRAIN TIMEOUT (3600s)", flush=True)
            return None
    wall = time.time() - t0
    if r.returncode != 0 or not os.path.exists(ck):
        tail = open(os.path.join(run_dir, "train.log"),
                    encoding="utf-8").read()[-800:]
        print(f"[{name}] TRAIN FAILED ({wall:.0f}s): {tail}", flush=True)
        return None
    print(f"[{name}] trained in {wall/60:.1f} min "
          f"({TOTAL_STEPS/wall:,.0f} steps/s wall)", flush=True)
    return ck


def main():
    table = {}
    for name, n, t in ARMS:            # train all, then bench all: the
        ck = train_arm(name, n, t)     # GPU is idle during benches (Rust
        if ck:                         # engine), so keep its queue full
            table[name] = {"ck": ck}   # first
    for name, row in table.items():
        res = bench(row["ck"],
                    os.path.join("bench", f"v10_shape_ladder_{name}.json"))
        if res:
            row.update(res)
            print(f"[{name}] upp {res['upp']:,.0f} / wpp {res['wpp']:,.2f}"
                  f" / FCs {res['fcs']} / acc {res['acc']:.4f}", flush=True)

    print("\n=== shape ladder — finals ===", flush=True)
    for (name, n, t) in ARMS:
        rr = table.get(name, {})
        if "upp" in rr:
            print(f"{name} ({n}x{t}): upp {rr['upp']:>12,.0f}  "
                  f"wpp {rr['wpp']:>9,.2f}  FCs {rr['fcs']}  "
                  f"acc {rr['acc']:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
