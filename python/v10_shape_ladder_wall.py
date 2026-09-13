"""Equal-WALL extension of the shape ladder (2026-08-08, user-directed).

The equal-steps ladder (v10_shape_ladder.py) was the dominance test and
the wide arms failed it: at 50M steps, 8192/16 = -47% upp and 16384/8 =
-57% vs 4096/32. But steps are manufactured, not scarce — the
operational question is score per pod-HOUR, and at equal wall the wide
shapes run 1.8-2.1x the steps. This driver gives each wide arm the
baseline arm's measured wall budget (7.3 min = 438 s of train wall,
114,067 steps/s x 438 s = 50M) at its own measured rate:

  b: 206,122 steps/s x 438 s = 90.3M -> 687 updates x 131,072 = 90.05M
  c: 243,510 steps/s x 438 s = 106.7M -> 816 updates x 131,072 = 106.95M

Each arm is configured as a self-contained run (lr cosine + entropy
schedules stretch over its own total — the configuration a real
wide-shape production run would use). Anchor: the equal-steps ladder's
a4096x32 final (upp 55,047 / wpp 485.33 / acc 0.1077, bench
v10_shape_ladder_a4096x32.json). Beat the anchor => the wide shape wins
per pod-hour despite the per-sample tax; lose => the N question closes
completely at today's scale.

  python v10_shape_ladder_wall.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = sys.executable
BASE = os.path.join("runs", "v10_shape_ladder")
CKPT_EVERY = 95

ARMS = [  # (name, num_envs, horizon, total_steps = equal-wall budget)
    ("b8192x16_wall", 8192, 16, 687 * 131072),
    ("c16384x8_wall", 16384, 8, 816 * 131072),
]

ANCHOR = ("a4096x32 @ 50M (equal wall)",
          dict(upp=55047.0, wpp=485.33, acc=0.1077))

HYPERS = {"lr": 0.00296092,
          "ent_coef": 0.0331038, "ent_final": 0.013929,
          "gamma": 0.914958, "gae_lambda": 0.980024}


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


def train_arm(name, n, t, steps):
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
           "--total-steps", str(steps),
           "--ckpt-every", str(CKPT_EVERY),
           "--run-dir", run_dir]
    print(f"[{name}] train {steps/1e6:.1f}M ({n}/{t}) -> {run_dir}",
          flush=True)
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
          f"({steps/wall:,.0f} steps/s wall)", flush=True)
    return ck


def main():
    table = {}
    for name, n, t, steps in ARMS:
        ck = train_arm(name, n, t, steps)
        if ck:
            table[name] = {"ck": ck}
    for name, row in table.items():
        res = bench(row["ck"],
                    os.path.join("bench", f"v10_shape_ladder_{name}.json"))
        if res:
            row.update(res)
            print(f"[{name}] upp {res['upp']:,.0f} / wpp {res['wpp']:,.2f}"
                  f" / FCs {res['fcs']} / acc {res['acc']:.4f}", flush=True)

    print("\n=== equal-wall ladder — vs anchor ===", flush=True)
    aname, a = ANCHOR
    print(f"{aname}: upp {a['upp']:>12,.0f}  wpp {a['wpp']:>9,.2f}  "
          f"acc {a['acc']:.4f}", flush=True)
    for name, n, t, steps in ARMS:
        rr = table.get(name, {})
        if "upp" in rr:
            print(f"{name} ({n}x{t}, {steps/1e6:.0f}M): "
                  f"upp {rr['upp']:>12,.0f}  wpp {rr['wpp']:>9,.2f}  "
                  f"FCs {rr['fcs']}  acc {rr['acc']:.4f}  "
                  f"[{rr['upp']/a['upp']:+.1%} vs anchor]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
