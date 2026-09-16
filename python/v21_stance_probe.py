"""v21 stance probe (user direction 2026-08-16): wide stance (hip_roll
+-8 deg, ankle counter-roll, flat soles, exact root ground-fit at
every contact reset) + per-side foot-X penalty (targets +-FOOT_X0 =
+-0.2211, the stance's own foot x) + jitter now CENTERED on the
stance. One combined arm on the v20 recipe.

    python v21_stance_probe.py

The two changes under test (both built this session, defaults OFF in
train_batch — this probe turns them on):
  --start-pose-jitter 0.15  episode-start leg/foot joint jitter
                            (~8.6 deg scale, knees |draw|, exact
                            sole-corner ground fix). Hypothesis:
                            teaches knee bend, helps balance.
  --foot-z-coef 0.01        per-tick linear penalty on each foot's
                            |world_z - reference_z|. Hypothesis: kills
                            the toward/away-from-notes shimmy.
Sizing: a sustained 0.3 m two-foot offset costs ~0.006/tick ~ 4.5
return over a 25 s segment — real pressure, well under the fall price.

From scratch (obs 580), v20 recipe otherwise: 3k trunk + cd400 leaf +
full bench. Judges vs the v20 legs 3k leaf baseline (+104.1 ret /
1,360,206 upp / 10,231 wpp / 432 FCs / 0 zeros / flail 0.9104),
leaf-end return ruling, upp crater veto; watch falls and flail.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v21_stance")
NUM_ENVS, HORIZON = 16384, 8
TRUNK_UPDATES = 3_000
COOLDOWN = 400

# the v19/v18-campaign recipe verbatim + the two v20 flags
COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", 
          "--chart-pool", "999999", "--mirror-augment",
          "--gamma", "0.965", "--gae-lambda", "0.98",
          "--ent-coef", "0.03", "--ent-tau", "1000", "--ent-final", "1e-3",
          "--lead-s", "0.5", "--lr", "7e-4",
          "--start-pose-jitter", "0.15", "--foot-z-coef", "0.01",
          "--foot-x-coef", "0.01"]


def run(cmd, log_name, timeout):
    t0 = time.time()
    with open(os.path.join(RUN_DIR, log_name), "a", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
    if r.returncode != 0:
        tail = open(os.path.join(RUN_DIR, log_name),
                    encoding="utf-8").read()[-800:]
        print(f"[stance-probe] FAILED {log_name}: {tail}", flush=True)
        sys.exit(1)
    return time.time() - t0


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    done_ck = os.path.join(RUN_DIR, f"trunk_upd{TRUNK_UPDATES:05d}.pt")
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "trunk_upd*.pt")))
    resume = (["--resume", numbered[-1]]
              if numbered and not os.path.exists(done_ck) else [])
    if not os.path.exists(done_ck):
        print(f"[stance-probe] trunk {TRUNK_UPDATES} updates"
              + (f" (resuming {os.path.basename(numbered[-1])})"
                 if resume else " (fresh, v21 stance arm)"), flush=True)
        w = run([PY, "train_batch.py", *COMMON, *resume,
                 "--total-steps", str(TRUNK_UPDATES * NUM_ENVS * HORIZON),
                 "--ckpt-every", "1000", "--run-dir", RUN_DIR],
                "trunk.log", 21_600)
        print(f"[stance-probe] trunk done in {w/60:.1f} min", flush=True)

    w = run([PY, "train_batch.py", *COMMON,
             "--resume", done_ck,
             "--cooldown-updates", str(COOLDOWN),
             "--ent-final", "3e-6",
             "--run-dir", RUN_DIR], "leaf.log", 7_200)
    print(f"[stance-probe] leaf done in {w/60:.1f} min", flush=True)

    leaf = [f for f in os.listdir(RUN_DIR)
            if f.startswith("leaf_from") and f.endswith(".pt")][0]
    out = os.path.join("bench", "v21_stance_leaf03000.json")
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
    print(f"PROBE_RESULT v21_stance@3000 leaf_return {ret} "
          f"upp {b['total_unweighted_pp']:.2f} "
          f"wpp {b['total_weighted_pp']:.2f} fcs {fcs} zeros {zeros} "
          f"flail {b.get('flail_rad_per_tick', -1):.5f}",
          flush=True)


if __name__ == "__main__":
    main()
