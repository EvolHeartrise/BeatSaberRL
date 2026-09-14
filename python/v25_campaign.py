"""v25 campaign (user direction 2026-08-18): extend the v25 hands
trunk past its 3k probe on a second pod, while the probe pod finishes
its own leaf + bench.

Resumes runs/v25_hands/trunk_upd03000.pt into ITS OWN run dir
(runs/v25_campaign — no file collisions with the probe's leaf/bench),
recipe verbatim (v24/v25: iso 5e-4, effort 0, foot terms, jitter,
wide-stance rents), constant lr 7e-4 WSD trunk, ckpt every 1000.
Target 24,000 updates; leaves get cut from checkpoints on demand
(launch_leafcut pattern), selection by leaf-end return as always.

    python v25_campaign.py
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v25_campaign")
SEED_CKPT = os.path.join("runs", "v25_hands", "trunk_upd03000.pt")
NUM_ENVS, HORIZON = 16384, 8
TARGET_UPDATES = 24_000

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


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "trunk_upd*.pt")))
    resume = numbered[-1] if numbered else SEED_CKPT
    if not os.path.exists(resume):
        print(f"[campaign] FATAL: seed ckpt {resume} missing", flush=True)
        sys.exit(1)
    print(f"[campaign] resuming {resume} -> {TARGET_UPDATES} updates",
          flush=True)
    t0 = time.time()
    with open(os.path.join(RUN_DIR, "trunk.log"), "a",
              encoding="utf-8") as log:
        r = subprocess.run(
            [PY, "train_batch.py", *COMMON,
             "--resume", resume,
             "--total-steps", str(TARGET_UPDATES * NUM_ENVS * HORIZON),
             "--ckpt-every", "1000", "--run-dir", RUN_DIR],
            stdout=log, stderr=subprocess.STDOUT, text=True)
    print(f"[campaign] exited rc={r.returncode} after "
          f"{(time.time() - t0)/3600:.2f} h", flush=True)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
