"""v23 iso probe (user direction 2026-08-17): activation cost REPLACES
the positive-work effort cost. Motivated by the replay measurement
that the policy commands torque nearly everywhere nearly always
(whole-body mean Σ|gear frac| 12.0 v21 / 15.2 v22 of a 29 max; v21
knees at FULL gear 30% of ticks holding the lunge, v22 hip rolls at
full gear 30% holding the wide stance, coast only 3%) — all free
under max(tau*qd, 0) work pricing because a held pose has qd = 0.
Real muscle pays for force whether or not it moves; this term is
that: reward -= iso_coef * Σ_j |commanded gear fraction| per tick.
Coast becomes the only free command.

    python v23_iso_probe.py

Changes vs the v22 recipe (user's call: activation INSTEAD of work):
       the work price is OFF for this probe
  --iso-coef 5e-4      ~6.5e-3/tick at the measured v22 baseline
                       activation (~13) ~ 5 return per 25 s segment —
                       the same order as the x-rent that moved the
                       stance. Pruning gratuitous torque halves it.

From scratch (obs 580), v22 recipe verbatim otherwise: 3k trunk +
cd400 leaf + full bench. Judges vs the v22 shaping 3k leaf baseline
(+103.5 ret / 1,299,122 upp / 10,148 wpp / 534 FCs / 0 zeros / flail
0.99971), leaf-end return ruling, upp crater veto. Note the return
includes the new -iso charge (up to ~-5/ep at unpruned activation),
so a small return deficit vs v22 is bookkeeping, not regression.
Also watch: flail (activation should price idle-arm torque), falls
(the stance hold gets more expensive), and the replay activation
stat (Σ|frac| and coast%) — the direct judge of whether the policy
learns to relax.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v23_iso")
NUM_ENVS, HORIZON = 16384, 8
TRUNK_UPDATES = 3_000
COOLDOWN = 400

# the v22 recipe verbatim, except: work cost off, activation cost on
COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", "--seg-len", "25",
          "--chart-pool", "999999", "--mirror-augment",
          "--gamma", "0.965", "--gae-lambda", "0.98",
          "--energy-coef", "7.5e-6",  # env v43: was --iso-coef 5e-4 (retired 2026-09-14)
          "--ent-coef", "0.03", "--ent-tau", "1000", "--ent-final", "1e-3",
          "--lead-s", "0.5", "--lr", "7e-4",
          "--start-pose-jitter", "0.15", "--foot-z-coef", "0.01",
          "--foot-x-coef", "0.01",
          "--foot-hinge-r", "0.12", "--foot-shape-coef", "0.05"]


def run(cmd, log_name, timeout):
    t0 = time.time()
    with open(os.path.join(RUN_DIR, log_name), "a", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
    if r.returncode != 0:
        tail = open(os.path.join(RUN_DIR, log_name),
                    encoding="utf-8").read()[-800:]
        print(f"[iso-probe] FAILED {log_name}: {tail}", flush=True)
        sys.exit(1)
    return time.time() - t0


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    done_ck = os.path.join(RUN_DIR, f"trunk_upd{TRUNK_UPDATES:05d}.pt")
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "trunk_upd*.pt")))
    resume = (["--resume", numbered[-1]]
              if numbered and not os.path.exists(done_ck) else [])
    if not os.path.exists(done_ck):
        print(f"[iso-probe] trunk {TRUNK_UPDATES} updates"
              + (f" (resuming {os.path.basename(numbered[-1])})"
                 if resume else " (fresh, v23 iso arm)"), flush=True)
        w = run([PY, "train_batch.py", *COMMON, *resume,
                 "--total-steps", str(TRUNK_UPDATES * NUM_ENVS * HORIZON),
                 "--ckpt-every", "1000", "--run-dir", RUN_DIR],
                "trunk.log", 21_600)
        print(f"[iso-probe] trunk done in {w/60:.1f} min", flush=True)

    w = run([PY, "train_batch.py", *COMMON,
             "--resume", done_ck,
             "--cooldown-updates", str(COOLDOWN),
             "--ent-final", "3e-6",
             "--run-dir", RUN_DIR], "leaf.log", 7_200)
    print(f"[iso-probe] leaf done in {w/60:.1f} min", flush=True)

    leaf = [f for f in os.listdir(RUN_DIR)
            if f.startswith("leaf_from") and f.endswith(".pt")][0]
    out = os.path.join("bench", "v23_iso_leaf03000.json")
    run([PY, "bench_scoresaber.py",
         "--ckpt", os.path.join(RUN_DIR, leaf),
         "--num-envs", "2048", "--foot-contact", "--out", out],
        "bench.log", 10_800)

    b = json.load(open(out, encoding="utf-8"))
    lv = [x for x in b["levels"] if "skipped" not in x]
    fcs = sum(1 for x in lv if x.get("full_combo"))
    zeros = sum(1 for x in lv if x.get("accuracy", 0.0) < 0.01)
    ret = "?"
    with open(os.path.join(RUN_DIR, "leaf.log"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("update"):
                ret = line.split()[5]
    print(f"PROBE_RESULT v23_iso@3000 leaf_return {ret} "
          f"upp {b['total_unweighted_pp']:.2f} "
          f"wpp {b['total_weighted_pp']:.2f} fcs {fcs} zeros {zeros} "
          f"flail {b.get('flail_rad_per_tick', -1):.5f}",
          flush=True)


if __name__ == "__main__":
    main()
