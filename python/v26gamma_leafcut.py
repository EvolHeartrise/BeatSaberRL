"""v26 gamma leaf cutter: cd400 leaf + full bench + top-10 pp elbow
replays from a v26_gamma trunk checkpoint, on a throwaway pod, in its
OWN run dir (the trunk pod keeps writing runs/v26_gamma — no
collisions). Replays ship per cut because the elbow-at-contact
distribution is this arm's pass/fail metric alongside return/pp.

    python v26gamma_leafcut.py 6000
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = sys.executable
CUT = int(sys.argv[1])
CKPT = os.path.join("runs", "v26_gamma", f"trunk_upd{CUT:05d}.pt")
RUN_DIR = os.path.join("runs", f"v26gamma_leaf{CUT // 1000}k")
NUM_ENVS, HORIZON = 16384, 8
COOLDOWN = 400

# the v26 gamma recipe (v25 verbatim except gamma 0.99)
COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", 
          "--chart-pool", "999999", "--mirror-augment",
          "--gamma", "0.99", "--gae-lambda", "0.98",
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
        print(f"[leafcut] FAILED {log_name}: {tail}", flush=True)
        sys.exit(1)
    return time.time() - t0


def main():
    assert os.path.exists(CKPT), f"missing {CKPT}"
    os.makedirs(RUN_DIR, exist_ok=True)
    w = run([PY, "train_batch.py", *COMMON,
             "--resume", CKPT,
             "--cooldown-updates", str(COOLDOWN),
             "--ent-final", "3e-6",
             "--run-dir", RUN_DIR], "leaf.log", 7_200)
    print(f"[leafcut] leaf done in {w/60:.1f} min", flush=True)

    leaf = [f for f in os.listdir(RUN_DIR)
            if f.startswith("leaf_from") and f.endswith(".pt")][0]
    leaf_path = os.path.join(RUN_DIR, leaf)
    out = os.path.join("bench", f"v26_gamma_leaf{CUT:05d}.json")
    run([PY, "bench_scoresaber.py",
         "--ckpt", leaf_path,
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
    print(f"LEAF_RESULT v26gamma@{CUT} leaf_return {ret} "
          f"upp {b['total_unweighted_pp']:.2f} "
          f"wpp {b['total_weighted_pp']:.2f} fcs {fcs} zeros {zeros} "
          f"flail {b.get('flail_rad_per_tick', -1):.5f}",
          flush=True)

    run([PY, "make_top10_replays.py", "--ckpt", leaf_path,
         "--bench", out, "--tag", f"v26gamma{CUT // 1000}k",
         "--top", "10", "--workers", "8", "--foot-contact"],
        "replays.log", 7_200)
    print("[leafcut] replays done", flush=True)


if __name__ == "__main__":
    main()
