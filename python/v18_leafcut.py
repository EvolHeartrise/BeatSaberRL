"""v18 campaign leaf cut: cd400 cooldown + full bench from a numbered
campaign trunk checkpoint, on a THROWAWAY pod sharing the volume with
the live trunk pod.

    python v18_leafcut.py <cut_updates>      # e.g. 3000

Reads  runs/v18_campaign/trunk_upd{cut:05d}.pt  (write-once, safe to
read while the trunk trains on) and writes everything to its own dirs
so the two pods never touch the same file:
  leaf run dir : runs/v18_campaign_leaf{cut:05d}
  bench json   : bench/v18_campaign_leaf{cut:05d}.json

Recipe = the campaign/w24-probe COMMON verbatim (v17_trunk_long.py) +
the standard leaf cooldown (cd400, ent -> 3e-6). Judged on leaf-end
ep_return (ruling metric), upp crater veto, FCs/zeros vs the w24
baseline (+118.2 / 1,330,255 upp / 10,348 wpp / 417 FCs / 14 zeros).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = sys.executable
CUT = int(sys.argv[1])
TRUNK_DIR = os.path.join("runs", "v18_campaign")
LEAF_DIR = os.path.join("runs", f"v18_campaign_leaf{CUT:05d}")
COOLDOWN = 400

# v17_trunk_long.py COMMON verbatim + the trunk lr
COMMON = ["--num-envs", "16384", "--horizon", "8",
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", "--seg-len", "25",
          "--chart-pool", "999999", "--mirror-augment",
          "--gamma", "0.965", "--gae-lambda", "0.98",
          "--effort-coef", "1e-4",
          "--ent-coef", "0.03", "--ent-tau", "1000", "--ent-final", "1e-3",
          "--lead-s", "0.5", "--lr", "7e-4"]


def run(cmd, log_name, timeout):
    t0 = time.time()
    with open(os.path.join(LEAF_DIR, log_name), "a", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
    if r.returncode != 0:
        tail = open(os.path.join(LEAF_DIR, log_name),
                    encoding="utf-8").read()[-800:]
        print(f"[leafcut {CUT}] FAILED {log_name}: {tail}", flush=True)
        sys.exit(1)
    return time.time() - t0


def main():
    os.makedirs(LEAF_DIR, exist_ok=True)
    cut_ck = os.path.join(TRUNK_DIR, f"trunk_upd{CUT:05d}.pt")
    if not os.path.exists(cut_ck):
        print(f"[leafcut {CUT}] MISSING {cut_ck}", flush=True)
        sys.exit(1)

    w = run([PY, "train_batch.py", *COMMON,
             "--resume", cut_ck,
             "--cooldown-updates", str(COOLDOWN),
             "--ent-final", "3e-6",
             "--run-dir", LEAF_DIR], "leaf_train.log", 7_200)
    print(f"[leafcut {CUT}] leaf done in {w/60:.1f} min", flush=True)

    leaf = [f for f in os.listdir(LEAF_DIR)
            if f.startswith("leaf_from") and f.endswith(".pt")][0]
    out = os.path.join("bench", f"v18_campaign_leaf{CUT:05d}.json")
    run([PY, "bench_scoresaber.py",
         "--ckpt", os.path.join(LEAF_DIR, leaf),
         "--num-envs", "2048", "--foot-contact", "--out", out],
        "bench.log", 10_800)

    b = json.load(open(out, encoding="utf-8"))
    lv = [x for x in b["levels"] if "skipped" not in x]
    fcs = sum(1 for x in lv if x.get("full_combo"))
    zeros = sum(1 for x in lv if x.get("accuracy", 0.0) < 0.01)
    ret = "?"
    with open(os.path.join(LEAF_DIR, "leaf_train.log"),
              encoding="utf-8") as f:
        for line in f:
            if line.startswith("update"):
                ret = line.split()[5]
    print(f"LEAF_RESULT v18@{CUT} leaf_return {ret} "
          f"upp {b['total_unweighted_pp']:.2f} "
          f"wpp {b['total_weighted_pp']:.2f} fcs {fcs} zeros {zeros}",
          flush=True)


if __name__ == "__main__":
    main()
