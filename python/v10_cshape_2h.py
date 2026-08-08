"""c-shape 2-hour production run (2026-08-08, user-locked shape).

Shape: N=16384 / T=8 — winner of the equal-WALL ladder (upp 479,296 at
107M steps in 438 s vs the 4096/32 baseline's 55,047 at 50M in the same
wall; +871%). User locked the shape 2026-08-08 without a mid-scale gate.
Batch stays 131,072, so the frozen wide hypers are inherited unchanged;
NOTE the known upside left on the table: gamma/gae_lambda were tuned
for longer horizons, never for T=8 (a T-aware sweep is the recorded
next lever).

Budget: 2 h of train wall at the measured 246,638 steps/s wall rate
=> 13,500 updates x 131,072 = 1,769,472,000 steps (~1.77B, ~1.99 h).
Checkpoint every 954 updates (125M steps, the 1B-driver cadence) => 14
checkpoints + final, every one benched on the frozen catalog afterward
(~8 min each on this pod's 256-core host, CPU-bound, GPU idle).

  python v10_cshape_2h.py
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
RUN_DIR = os.path.join("runs", "v10_cshape_2h")
NUM_ENVS, HORIZON = 16384, 8
TOTAL_STEPS = 13_500 * 131_072          # 1,769,472,000
CKPT_EVERY = 954                        # updates; 1B-driver cadence
TRAIN_TIMEOUT = TOTAL_STEPS // 200_000 + 1_800   # generous vs 246k rate

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


def main():
    h = HYPERS
    os.makedirs(RUN_DIR, exist_ok=True)
    ck = os.path.join(RUN_DIR, "final.pt")
    if os.path.exists(ck):
        print(f"{RUN_DIR}/final.pt exists — skipping train", flush=True)
    else:
        cmd = [PY, "train_batch.py",
               "--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
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
               "--run-dir", RUN_DIR]
        print(f"[train] c-shape 2h ({NUM_ENVS}/{HORIZON}, "
              f"{TOTAL_STEPS/1e9:.2f}B) -> {RUN_DIR}", flush=True)
        t0 = time.time()
        with open(os.path.join(RUN_DIR, "train.log"), "w",
                  encoding="utf-8") as log:
            try:
                r = subprocess.run(cmd, stdout=log,
                                   stderr=subprocess.STDOUT,
                                   text=True, timeout=TRAIN_TIMEOUT)
            except subprocess.TimeoutExpired:
                print(f"[train] TIMEOUT ({TRAIN_TIMEOUT}s)", flush=True)
                return
        wall = time.time() - t0
        if r.returncode != 0 or not os.path.exists(ck):
            tail = open(os.path.join(RUN_DIR, "train.log"),
                        encoding="utf-8").read()[-800:]
            print(f"[train] FAILED ({wall:.0f}s): {tail}", flush=True)
            return
        print(f"[train] done in {wall/3600:.2f} h "
              f"({TOTAL_STEPS/wall:,.0f} steps/s wall)", flush=True)

    table = {}
    res = bench(ck, os.path.join("bench", "v10_cshape_2h_final.json"))
    if res:
        table["final"] = res
        print(f"[final] upp {res['upp']:,.0f} / wpp {res['wpp']:,.2f} / "
              f"FCs {res['fcs']} / acc {res['acc']:.4f}", flush=True)
    # bench the LAST 3 numbered checkpoints only (user decision
    # 2026-08-08: 4 bench points total incl. final — the late curve is
    # the question; the early trajectory isn't worth 11 x 8 min of CPU)
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "ckpt_upd*.pt")),
                      key=lambda p: int(re.search(r"upd(\d+)", p).group(1)))
    for c in numbered[-3:]:
        name = os.path.basename(c)[:-3]
        rr = bench(c, os.path.join("bench", f"v10_cshape_2h_{name}.json"))
        if rr is None:
            continue
        upd = int(re.search(r"upd(\d+)", c).group(1))
        table[name] = rr
        print(f"[{name}] ({upd*131072/1e6:.0f}M) upp {rr['upp']:,.0f} / "
              f"wpp {rr['wpp']:,.2f} / FCs {rr['fcs']} / "
              f"acc {rr['acc']:.4f}", flush=True)

    if table:
        best = max(table.items(), key=lambda kv: kv[1]["upp"])
        print(f"\nBEST upp: {best[0]} — {best[1]['upp']:,.0f}", flush=True)
        print("\n=== c-shape 2h — checkpoint trajectory ===", flush=True)
        for name, rr in table.items():
            print(f"{name}: upp {rr['upp']:>12,.0f}  "
                  f"wpp {rr['wpp']:>9,.2f}  FCs {rr['fcs']}  "
                  f"acc {rr['acc']:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
