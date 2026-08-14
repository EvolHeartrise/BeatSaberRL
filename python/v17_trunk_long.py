"""Long v17 trunk: extend the lr-ladder winner's trunk (7e-4, 861k
arch, env v17 walls) from 3,000 -> 24,000 updates, extendable further
under the standing extend-while-climbing rule.

7e-4 won the 5-rung stable-lr ladder on leaf-end return (ruling
metric, user decision 2026-08-14): 95.5 (1e-3) -> 114.3 (7e-4) ->
97.6 (5e-4) — a peak inside the bracket — and was second-best on upp
(468k) with 85 FCs, so no crater-flag. First run seeds RUN_DIR from
the ladder rung's final trunk ckpt (probe trunks are the exact
production recipe, so the 3k updates are banked); after that it
crash-resumes from the highest NUMBERED ckpt in RUN_DIR. Leaf cuts
happen OFF-POD (cut_leaf.sh throwaway pods, every 3k updates); this
driver only extends the trunk and benches the raw end.

  python v17_trunk_long.py [total_updates]
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v17_trunk")
SEED_CK = os.path.join("runs", "v17_lr_lr7e-4b", "trunk_upd03000.pt")
NUM_ENVS, HORIZON = 16384, 8
BATCH = NUM_ENVS * HORIZON
LR = "7e-4"
TRUNK_TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 24_000
TRAIN_TIMEOUT = 42_000

# The ladder-rung recipe verbatim (= v16 winner + v17 walls env).
# ent-tau/ent-final: the trunk's entropy coef decays to the 1e-3 floor
# by ~3k updates and holds there — the trunk-ified schedule; leaves
# take it to 3e-6.
COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", "--seg-len", "25", "--chart-pool", "24576",
          "--gamma", "0.965", "--gae-lambda", "0.98",
          "--effort-coef", "1e-4",
          "--ent-coef", "0.03", "--ent-tau", "1000", "--ent-final", "1e-3",
          "--lead-s", "0.5"]


def bench(ck, out):
    try:
        r = subprocess.run([PY, "bench_scoresaber.py", "--ckpt", ck,
                            "--out", out, "--num-envs", "2048",
                            "--foot-contact"],
                           capture_output=True, text=True, timeout=10800)
    except subprocess.TimeoutExpired:
        print(f"BENCH TIMEOUT {ck}", flush=True)
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
    os.makedirs(RUN_DIR, exist_ok=True)
    # Resume from the highest NUMBERED checkpoint: trunk_final.pt is only
    # written on clean completion, so after a pod death it is stale — the
    # numbered ckpts (every 500) are the real frontier.
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "trunk_upd*.pt")))
    if not numbered:
        assert os.path.exists(SEED_CK), f"seed ckpt {SEED_CK} missing"
        seeded = os.path.join(RUN_DIR, os.path.basename(SEED_CK))
        shutil.copy2(SEED_CK, seeded)
        numbered = [seeded]
        print(f"[trunk-ext] seeded {RUN_DIR} from ladder rung {SEED_CK}",
              flush=True)
    trunk_ck = numbered[-1]
    print(f"[trunk-ext] resuming from {trunk_ck}", flush=True)
    cmd = [PY, "train_batch.py", *COMMON, "--lr", LR,
           "--resume", trunk_ck,
           "--total-steps", str(TRUNK_TOTAL * BATCH),
           "--ckpt-every", "500",
           "--run-dir", RUN_DIR]
    print(f"[trunk-ext] v17 trunk -> {TRUNK_TOTAL:,} updates @ lr {LR}",
          flush=True)
    t0 = time.time()
    with open(os.path.join(RUN_DIR, "trunk_ext.log"), "a",
              encoding="utf-8") as log:
        try:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               text=True, timeout=TRAIN_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"[trunk-ext] TIMEOUT ({TRAIN_TIMEOUT}s)", flush=True)
            return
    wall = time.time() - t0
    if r.returncode != 0:
        tail = open(os.path.join(RUN_DIR, "trunk_ext.log"),
                    encoding="utf-8").read()[-800:]
        print(f"[trunk-ext] FAILED ({wall:.0f}s): {tail}", flush=True)
        return
    print(f"[trunk-ext] done in {wall/60:.1f} min", flush=True)

    final_ck = os.path.join(RUN_DIR, "trunk_final.pt")
    res = bench(final_ck,
                os.path.join("bench",
                             f"v17_trunk{TRUNK_TOTAL // 1000}k_raw.json"))
    table = {f"trunk_raw_{TRUNK_TOTAL // 1000}k": res} if res else {}
    table["ladder_7e-4_leaf"] = {"upp": 468551.11, "wpp": 7250.50,
                                 "leaf_return": 114.3}
    print("DONE " + json.dumps(table), flush=True)


if __name__ == "__main__":
    main()
