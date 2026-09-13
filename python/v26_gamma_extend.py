"""v26 gamma probe autoextension (user direction 2026-08-18): after
the 3k probe pipeline, extend the trunk 3k -> 6k on the same pod,
then cut the standard artifact set at 6k — cd400 leaf (own run dir,
no leaf_from collision with the 3k leaf), full bench, top-10 pp
replays. Same judge line as the probe (PROBE_RESULT v26_gamma@6000).

    python v26_gamma_extend.py
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v26_gamma")
LEAF_DIR = os.path.join("runs", "v26gamma_leaf6k")
NUM_ENVS, HORIZON = 16384, 8
TARGET_UPDATES = 6_000
COOLDOWN = 400

# the v26 gamma recipe (v25 verbatim except gamma 0.99)
COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", "--seg-len", "25",
          "--chart-pool", "999999", "--mirror-augment",
          "--gamma", "0.99", "--gae-lambda", "0.98",
          "--iso-coef", "5e-4",
          "--ent-coef", "0.03", "--ent-tau", "1000", "--ent-final", "1e-3",
          "--lead-s", "0.5", "--lr", "7e-4",
          "--start-pose-jitter", "0.15", "--foot-z-coef", "0.01",
          "--foot-x-coef", "0.01",
          "--foot-hinge-r", "0.12", "--foot-shape-coef", "0.05"]


def run(cmd, log_dir, log_name, timeout):
    t0 = time.time()
    with open(os.path.join(log_dir, log_name), "a", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
    if r.returncode != 0:
        tail = open(os.path.join(log_dir, log_name),
                    encoding="utf-8").read()[-800:]
        print(f"[gamma-ext] FAILED {log_name}: {tail}", flush=True)
        sys.exit(1)
    return time.time() - t0


def main():
    done_ck = os.path.join(RUN_DIR, f"trunk_upd{TARGET_UPDATES:05d}.pt")
    if not os.path.exists(done_ck):
        numbered = sorted(glob.glob(os.path.join(RUN_DIR, "trunk_upd*.pt")))
        assert numbered, f"no trunk ckpts in {RUN_DIR}"
        print(f"[gamma-ext] extending {os.path.basename(numbered[-1])} "
              f"-> {TARGET_UPDATES} updates", flush=True)
        w = run([PY, "train_batch.py", *COMMON,
                 "--resume", numbered[-1],
                 "--total-steps", str(TARGET_UPDATES * NUM_ENVS * HORIZON),
                 "--ckpt-every", "1000", "--run-dir", RUN_DIR],
                RUN_DIR, "trunk.log", 21_600)
        print(f"[gamma-ext] trunk 6k done in {w/60:.1f} min", flush=True)

    os.makedirs(LEAF_DIR, exist_ok=True)
    w = run([PY, "train_batch.py", *COMMON,
             "--resume", done_ck,
             "--cooldown-updates", str(COOLDOWN),
             "--ent-final", "3e-6",
             "--run-dir", LEAF_DIR], LEAF_DIR, "leaf.log", 7_200)
    print(f"[gamma-ext] leaf done in {w/60:.1f} min", flush=True)

    leaf = [f for f in os.listdir(LEAF_DIR)
            if f.startswith("leaf_from") and f.endswith(".pt")][0]
    leaf_path = os.path.join(LEAF_DIR, leaf)
    out = os.path.join("bench", "v26_gamma_leaf06000.json")
    run([PY, "bench_scoresaber.py",
         "--ckpt", leaf_path,
         "--num-envs", "2048", "--foot-contact", "--out", out],
        LEAF_DIR, "bench.log", 10_800)

    b = json.load(open(out, encoding="utf-8"))
    lv = [x for x in b["levels"] if "skipped" not in x]
    fcs = sum(1 for x in lv if x.get("full_combo"))
    zeros = sum(1 for x in lv if x.get("accuracy", 0.0) < 0.01)
    ret = "?"
    with open(os.path.join(LEAF_DIR, "leaf.log"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("update"):
                ret = line.split()[5]
    print(f"PROBE_RESULT v26_gamma@6000 leaf_return {ret} "
          f"upp {b['total_unweighted_pp']:.2f} "
          f"wpp {b['total_weighted_pp']:.2f} fcs {fcs} zeros {zeros} "
          f"flail {b.get('flail_rad_per_tick', -1):.5f}",
          flush=True)

    run([PY, "make_top10_replays.py", "--ckpt", leaf_path,
         "--bench", out, "--tag", "v26gamma6k",
         "--top", "10", "--workers", "8", "--foot-contact"],
        LEAF_DIR, "replays.log", 7_200)
    print("[gamma-ext] replays done", flush=True)


if __name__ == "__main__":
    main()
