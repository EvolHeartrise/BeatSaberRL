"""v17 stable-lr ladder rung: 3,000-update trunk at a given lr, then a
cd400 leaf + bench.

    python v17_lr_probe.py <lr> [tag]

One rung of the 861k-arch stable-phase lr calibration (2026-08-13).
The stable lr is THE number WSD hangs off (warmup ramps into it,
leaves decay from it), and the river-valley/functional-scaling-law
literature says to run the largest ADMISSIBLE value — so the ladder
(7e-4 / 1e-3 / 1.4e-3 / 2e-3) brackets the 1/width prior (2e-3 at
embed 64 -> ~1e-3 at 128) and keeps the old recipe as the edge
control. Judges, in order: the leaf's bench upp, leaf-end ep_return,
trunk maxlr/entropy stability (WSD r2 taught us inadmissible looks
like maxlr in the hundreds, permanent).

v17 recipe otherwise = the v16 winner: gears 5, foot contact, ranked
segments 25 s / lead 0.5, gamma 0.965, lambda 0.98, effort 1e-4,
ent 0.03 tau 1000 floor 1e-3, batch 16384 x 8.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = sys.executable
LR = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else f"lr{LR}"
RUN_DIR = os.path.join("runs", f"v17_lr_{TAG}")
NUM_ENVS, HORIZON = 16384, 8
TRUNK_UPDATES = 3_000
COOLDOWN = 400

COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", "--chart-pool", "24576",
          "--gamma", "0.965", "--gae-lambda", "0.98",
          "--ent-coef", "0.03", "--ent-tau", "1000", "--ent-final", "1e-3",
          "--lead-s", "0.5"]


def run(cmd, log_name, timeout):
    t0 = time.time()
    # "a" not "w": a crash-resumed rung must not clobber the pre-crash log
    with open(os.path.join(RUN_DIR, log_name), "a", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
    if r.returncode != 0:
        tail = open(os.path.join(RUN_DIR, log_name),
                    encoding="utf-8").read()[-800:]
        print(f"[lr-probe {TAG}] FAILED {log_name}: {tail}", flush=True)
        sys.exit(1)
    return time.time() - t0


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    batch = NUM_ENVS * HORIZON
    # Crash-resume from the highest NUMBERED ckpt (trunk_final.pt only
    # exists on clean completion — the 2026-08-14 balance-outage pod
    # death mid-rung is exactly the case this covers)
    import glob
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "trunk_upd*.pt")))
    resume = (["--resume", numbered[-1]] if numbered
              and not os.path.exists(os.path.join(
                  RUN_DIR, f"trunk_upd{TRUNK_UPDATES:05d}.pt")) else [])
    done = os.path.exists(os.path.join(RUN_DIR,
                                       f"trunk_upd{TRUNK_UPDATES:05d}.pt"))
    print(f"[lr-probe {TAG}] trunk {TRUNK_UPDATES} updates @ lr {LR}"
          + (f" (resuming {os.path.basename(numbered[-1])})" if resume
             else " (already complete)" if done else ""), flush=True)
    if not done:
        w = run([PY, "train_batch.py", *COMMON, "--lr", LR, *resume,
                 "--total-steps", str(TRUNK_UPDATES * batch),
                 "--ckpt-every", "1000", "--run-dir", RUN_DIR],
                "trunk.log", 21_600)
    else:
        w = 0.0
    print(f"[lr-probe {TAG}] trunk done in {w/60:.1f} min", flush=True)

    trunk_ck = os.path.join(RUN_DIR, f"trunk_upd{TRUNK_UPDATES:05d}.pt")
    w = run([PY, "train_batch.py", *COMMON, "--lr", LR,
             "--resume", trunk_ck,
             "--cooldown-updates", str(COOLDOWN), "--ent-final", "3e-6",
             "--run-dir", RUN_DIR], "leaf.log", 7_200)
    print(f"[lr-probe {TAG}] leaf done in {w/60:.1f} min", flush=True)

    leaf = [f for f in os.listdir(RUN_DIR)
            if f.startswith("leaf_from") and f.endswith(".pt")][0]
    out = os.path.join("bench", f"v17_lr_{TAG}.json")
    run([PY, "bench_scoresaber.py", "--ckpt", os.path.join(RUN_DIR, leaf),
         "--num-envs", "2048", "--foot-contact", "--out", out],
        "bench.log", 10_800)
    b = json.load(open(out, encoding="utf-8"))
    fcs = sum(1 for lv in b["levels"] if lv.get("full_combo"))
    print(f"PROBE_RESULT {TAG} lr {LR} "
          f"upp {b['total_unweighted_pp']:.2f} "
          f"wpp {b['total_weighted_pp']:.2f} fcs {fcs}", flush=True)


if __name__ == "__main__":
    main()
