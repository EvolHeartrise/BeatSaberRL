"""v19 post-swing-remaining (psr) obs probe (user go 2026-08-16).

    python v19_psr_probe.py

One arm, from scratch (obs 578 -> 580 invalidates every checkpoint):
the full v18 campaign recipe verbatim (train_batch defaults: window
24, flat fall 5.0, pushes; full mirrored pool, lr 7e-4) with the env
v19 change — a remaining-follow-through-credit lane per hand
(hand_base+10). 3k trunk + cd400 leaf (ent -> 3e-6) + full bench.

The change under test closes the post-swing credit-assignment hole:
follow-through reward lands after the note leaves the obs, so those
frames alias with genuinely empty air and the policy learns that
swinging at nothing pays (the flail bug, user diagnosis 2026-08-16).

Judges vs the SAME-RECIPE obs-578 baselines
  w24 probe 3k leaf:     +118.2 ret / 1,330,255 upp / 10,348 wpp
                         / 417 FC / 14 zeros
  v18 campaign 3k leaf:  +107.4 ret / 1,118,088 upp /  8,966 wpp
                         / 101 FC /  7 zeros   (seed-0 rerun spread)
  (1) leaf-end ep_return (ruling), (2) bench flail_rad_per_tick — the
  direct phantom-swing judge, expected to DROP hard (no 578 baseline
  number exists; the [flail] trunk-log trajectory is the within-run
  signal), (3) upp crater veto, (4) FCs / zeros.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v19_psr")
NUM_ENVS, HORIZON = 16384, 8
TRUNK_UPDATES = 3_000
COOLDOWN = 400

# the v18 campaign recipe verbatim (v17_trunk_long.py COMMON + lr)
COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", "--seg-len", "25",
          "--chart-pool", "999999", "--mirror-augment",
          "--gamma", "0.965", "--gae-lambda", "0.98",
          "--ent-coef", "0.03", "--ent-tau", "1000", "--ent-final", "1e-3",
          "--lead-s", "0.5", "--lr", "7e-4"]


def run(cmd, log_name, timeout):
    t0 = time.time()
    with open(os.path.join(RUN_DIR, log_name), "a", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
    if r.returncode != 0:
        tail = open(os.path.join(RUN_DIR, log_name),
                    encoding="utf-8").read()[-800:]
        print(f"[psr-probe] FAILED {log_name}: {tail}", flush=True)
        sys.exit(1)
    return time.time() - t0


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    done_ck = os.path.join(RUN_DIR, f"trunk_upd{TRUNK_UPDATES:05d}.pt")
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "trunk_upd*.pt")))
    resume = (["--resume", numbered[-1]]
              if numbered and not os.path.exists(done_ck) else [])
    if not os.path.exists(done_ck):
        print(f"[psr-probe] trunk {TRUNK_UPDATES} updates"
              + (f" (resuming {os.path.basename(numbered[-1])})"
                 if resume else " (fresh, obs 580)"), flush=True)
        w = run([PY, "train_batch.py", *COMMON, *resume,
                 "--total-steps", str(TRUNK_UPDATES * NUM_ENVS * HORIZON),
                 "--ckpt-every", "1000", "--run-dir", RUN_DIR],
                "trunk.log", 21_600)
        print(f"[psr-probe] trunk done in {w/60:.1f} min", flush=True)

    w = run([PY, "train_batch.py", *COMMON,
             "--resume", done_ck,
             "--cooldown-updates", str(COOLDOWN),
             "--ent-final", "3e-6",
             "--run-dir", RUN_DIR], "leaf.log", 7_200)
    print(f"[psr-probe] leaf done in {w/60:.1f} min", flush=True)

    leaf = [f for f in os.listdir(RUN_DIR)
            if f.startswith("leaf_from") and f.endswith(".pt")][0]
    out = os.path.join("bench", "v19_psr_leaf03000.json")
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
    print(f"PROBE_RESULT v19_psr@3000 leaf_return {ret} "
          f"upp {b['total_unweighted_pp']:.2f} "
          f"wpp {b['total_weighted_pp']:.2f} fcs {fcs} zeros {zeros} "
          f"flail {b.get('flail_rad_per_tick', -1):.5f}",
          flush=True)


if __name__ == "__main__":
    main()
