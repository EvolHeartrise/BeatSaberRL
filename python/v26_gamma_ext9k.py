"""v26 gamma trunk extension to 9k updates (user direction
2026-08-18). Trunk ONLY — leaf cuts happen on their own throwaway
pods via v26gamma_leafcut.py (user: keep the trunk pod training
uninterrupted).

    python v26_gamma_ext9k.py
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v26_gamma")
NUM_ENVS, HORIZON = 16384, 8
TARGET_UPDATES = 9_000

# the v26 gamma recipe (v25 verbatim except gamma 0.99)
COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", "--seg-len", "25",
          "--chart-pool", "999999", "--mirror-augment",
          "--gamma", "0.99", "--gae-lambda", "0.98",
          "--effort-coef", "0", "--iso-coef", "5e-4",
          "--ent-coef", "0.03", "--ent-tau", "1000", "--ent-final", "1e-3",
          "--lead-s", "0.5", "--lr", "7e-4",
          "--start-pose-jitter", "0.15", "--foot-z-coef", "0.01",
          "--foot-x-coef", "0.01",
          "--foot-hinge-r", "0.12", "--foot-shape-coef", "0.05"]


def main():
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "trunk_upd*.pt")))
    assert numbered, f"no trunk ckpts in {RUN_DIR}"
    resume = numbered[-1]
    print(f"[gamma-ext9k] resuming {os.path.basename(resume)} "
          f"-> {TARGET_UPDATES} updates", flush=True)
    t0 = time.time()
    with open(os.path.join(RUN_DIR, "trunk.log"), "a",
              encoding="utf-8") as log:
        r = subprocess.run(
            [PY, "train_batch.py", *COMMON,
             "--resume", resume,
             "--total-steps", str(TARGET_UPDATES * NUM_ENVS * HORIZON),
             "--ckpt-every", "1000", "--run-dir", RUN_DIR],
            stdout=log, stderr=subprocess.STDOUT, text=True)
    print(f"[gamma-ext9k] exited rc={r.returncode} after "
          f"{(time.time() - t0)/3600:.2f} h", flush=True)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
