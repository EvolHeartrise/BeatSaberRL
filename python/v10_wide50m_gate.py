"""G1/G3 recipe gate (2026-08-07): 50M at N=4096/T=32, batch-preserving.

Round-5 audit found production trains the env kernel at ~20% of its
resident-thread budget (N=1024 -> 32 blocks over 20 SMs vs the 5,120-
thread ceiling). Measured 2026-08-07 (exec-true, --bench): 12,423
steps/s at 1024/128 vs 24,422 at 4096/32 = 1.97x, split 96/4 vs 93/7
rollout/update. This driver is the learning-science arm of that gate:
identical recipe to v10_gears5_smoke50m.py (same hypers, seed 0,
batch = 131,072 preserved so LR/entropy schedules and checkpoint
cadence land on identical update indices) with ONLY the rollout shape
changed. Physics is bitwise N-invariant (one thread per env); the
science deltas are the GAE truncation horizon (T=128 -> 32; at these
hypers gamma*lambda = 0.89669 so only 3.05%% of trace weight lies past
32 steps, and the value bootstrap covers it) and the chart-draw /
sampling stream (4x envs). Baseline arm = the recorded
runs/v10_gears5_smoke50m results (valid across kernel versions: every
landed kernel change since was bitwise). Compare upp/wpp/FCs/acc at
the same update indices. NOTE: ep_return(last100) prints a ~4x
shorter time window at N=4096 (deque maxlen 200, more episodes/update)
— expected, cosmetic, not a regime change.

  python v10_wide50m_gate.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v10_wide50m_gate")
TOTAL_STEPS = 50_000_000
CKPT_EVERY = 95  # updates; same indices as the smoke baseline

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


def main():
    h = HYPERS
    os.makedirs(RUN_DIR, exist_ok=True)
    ck = os.path.join(RUN_DIR, "final.pt")
    if os.path.exists(ck):
        print(f"{RUN_DIR}/final.pt exists — skipping train", flush=True)
    else:
        cmd = [PY, "train_batch.py", "--num-envs", "4096", "--horizon", "32",
               "--lr-decay", "cosine", "--seed", "0",
               "--ranked-maps", "--seg-len", "25",
               "--chart-pool", "24576",
               "--gears", "5",
               "--lr", f"{h['lr']:.6g}",
               "--ent-coef", f"{h['ent_coef']:.6g}",
               "--ent-final", f"{h['ent_final']:.6g}",
               "--gamma", f"{h['gamma']:.6g}",
               "--gae-lambda", f"{h['gae_lambda']:.6g}",
               "--total-steps", str(TOTAL_STEPS),
               "--ckpt-every", str(CKPT_EVERY),
               "--run-dir", RUN_DIR]
        print(f"[train] v10 WIDE gate 50M (4096/32) -> {RUN_DIR}",
              flush=True)
        t0 = time.time()
        log = open(os.path.join(RUN_DIR, "train.log"), "w",
                   encoding="utf-8")
        try:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               text=True, timeout=25200)
        except subprocess.TimeoutExpired:
            log.close()
            print("[train] TIMEOUT (25200s)", flush=True)
            return
        log.close()
        wall = time.time() - t0
        if r.returncode != 0 or not os.path.exists(ck):
            tail = open(os.path.join(RUN_DIR, "train.log"),
                        encoding="utf-8").read()[-800:]
            print(f"[train] FAILED ({wall:.0f}s): {tail}", flush=True)
            return
        print(f"[train] done in {wall/60:.1f} min", flush=True)

    table = {}
    res = bench(ck, os.path.join(RUN_DIR, "bench.json"))
    if res is None:
        return
    table["final.pt"] = res
    print(f"[final] upp {res['upp']:,.0f} / wpp {res['wpp']:,.2f} / "
          f"FCs {res['fcs']} / acc {res['acc']:.4f}", flush=True)
    for c in sorted(glob.glob(os.path.join(RUN_DIR, "ckpt_upd*.pt")),
                    key=lambda p: int(re.search(r"upd(\d+)", p).group(1))):
        name = os.path.basename(c)[:-3]
        rr = bench(c, os.path.join("bench", f"v10_wide50m_gate_{name}.json"))
        if rr is None:
            continue
        upd = int(re.search(r"upd(\d+)", c).group(1))
        table[name] = rr
        print(f"[{name}] ({upd*131072/1e6:.1f}M) upp {rr['upp']:,.0f} / "
              f"wpp {rr['wpp']:,.2f} / FCs {rr['fcs']} / "
              f"acc {rr['acc']:.4f}", flush=True)

    best = max(table.items(), key=lambda kv: kv[1]["upp"])
    print(f"\nBEST upp: {best[0]} — {best[1]['upp']:,.0f}", flush=True)
    print("\n=== wide-gate — checkpoint trajectory (both pp totals) ===",
          flush=True)
    for name, rr in table.items():
        print(f"{name}: upp {rr['upp']:>12,.0f}  wpp {rr['wpp']:>9,.2f}"
              f"  FCs {rr['fcs']}  acc {rr['acc']:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
