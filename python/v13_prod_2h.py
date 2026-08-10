"""env v13 production: 2 hours, c-shape, K=5 — the s4-locked recipe.
Hypers locked from sweep s4 (71 legs, 2026-08-10 read; sweeps/
s4_contact_ret): gamma 0.965 — the supported band is 0.91..0.995
(phase-2 gamma-score r=+0.52, floor 0.72 firmly rejected; the user's
balance-horizon hypothesis holds); lr 5e-3 (sweep band 4e-3..1.2e-2,
tempered for the 4x-longer schedule); ent 0.03 -> 3e-6 (top legs
pinned the low floor); lambda 0.98 (the gamma*lambda ridge); effort
1e-4 (sweep-pinned ~= off — REWARD-PARAMETERIZED under the sweep
objective, real tax to be decided externally from bench/replay
evidence).

Purpose: the sweep's objective cannot see note-hitting (plateau
-64..-68 at every affordable budget); this run is the pp-benched
confirmation at the scale where hits actually accrue. 2 h at ~148k
steps/s => 8,100 updates x 131,072 = 1,061,683,200 steps. Fresh init.
Checkpoints every 2,000; bench final + last 3, all --foot-contact.

Compare against: v12 1h (upp 123 @ 524M, falls 2,031 at end) — this
run should beat it on both by scale alone; the recipe question is by
how much.

  python v13_prod_2h.py
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
RUN_DIR = os.path.join("runs", "v13_prod_2h")
NUM_ENVS, HORIZON = 16384, 8
TOTAL_STEPS = 8_100 * 131_072           # 1,061,683,200 (~2.0 h @ 148k)
CKPT_EVERY = 2_000
TRAIN_TIMEOUT = TOTAL_STEPS // 100_000 + 1_800

HYPERS = {"lr": 5e-3,
          "ent_coef": 0.03, "ent_final": 3e-6,
          "gamma": 0.965, "gae_lambda": 0.98,
          "effort_coef": 1e-4}


def bench(ck, out):
    try:
        r = subprocess.run([PY, "bench_scoresaber.py", "--ckpt", ck,
                            "--out", out, "--num-envs", "2048",
                            "--foot-contact"],
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
               "--foot-contact",
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
        print(f"[train] v13 PROD 2h ({NUM_ENVS}/{HORIZON}, "
              f"{TOTAL_STEPS/1e6:.0f}M, s4-locked recipe, K=5) "
              f"-> {RUN_DIR}", flush=True)
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
        print(f"[train] done in {wall/60:.1f} min "
              f"({TOTAL_STEPS/wall:,.0f} steps/s wall)", flush=True)

    table = {}
    res = bench(ck, os.path.join("bench", "v13_prod_2h_final.json"))
    if res:
        table["final"] = res
        print(f"[final] upp {res['upp']:,.0f} / wpp {res['wpp']:,.2f} / "
              f"FCs {res['fcs']} / acc {res['acc']:.4f}", flush=True)
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "ckpt_upd*.pt")),
                      key=lambda p: int(re.search(r"upd(\d+)", p).group(1)))
    for c in numbered[-3:]:
        name = os.path.basename(c)[:-3]
        rr = bench(c, os.path.join("bench", f"v13_prod_2h_{name}.json"))
        if rr is None:
            continue
        table[name] = rr
        print(f"[{name}] upp {rr['upp']:,.0f} / wpp {rr['wpp']:,.2f} / "
              f"FCs {rr['fcs']} / acc {rr['acc']:.4f}", flush=True)
    print("DONE " + json.dumps(table), flush=True)


if __name__ == "__main__":
    main()
