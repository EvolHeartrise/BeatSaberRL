"""v18 CAMPAIGN trunk (user go 2026-08-16): window 24 + constant fall
price + perturbation kicks (all train_batch defaults now) on the full
mirrored pool — the recipe validated end-to-end by the w24 probe
(+118.2 leaf return / wpp ATH 10,348 / 417 FCs vs constpush w16).
Fresh RUN_DIR: the old fpm trunk is obs-426 era and must not be
resumed. Originally: long v17 trunk, full-pool + mirror campaign
(user decision 2026-08-14): train FROM SCRATCH on ALL ~46k ranked segments (the old
--chart-pool 24576 was a fixed seed-0 subset — 47% of the catalog
never trained) doubled by left-right mirror augmentation (~92k
segments, ~5.9 GB pool). Diversity is the medicine for the
entropy-collapse/specialization failure: the first full-pool-less
trunk hit ent 2.6 by update 4,876 (16.7 at resume) while returns
climbed — the v16 6k-leaf movie.

lr 7e-4 carried over from the 5-rung ladder (leaf-end return ruling
metric: 95.5/114.3/97.6 peak inside the bracket). Calibrated on the
subsampled pool — a prior, not a re-measurement. Crash-resumes from
the highest NUMBERED ckpt in RUN_DIR. Leaf cuts happen OFF-POD
(cut_leaf.sh throwaway pods, every 3k updates); this driver only
runs the trunk and benches the raw end.

  python v17_trunk_long.py [total_updates]
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v18_campaign")   # w24+constpush era
NUM_ENVS, HORIZON = 16384, 8
BATCH = NUM_ENVS * HORIZON
LR = "7e-4"
TRUNK_TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 24_000
TRAIN_TIMEOUT = 50_000   # from-scratch: 24k upds ~10.2h + pool build

# The ladder-rung recipe verbatim (= v16 winner + v17 walls env).
# ent-tau/ent-final: the trunk's entropy coef decays to the 1e-3 floor
# by ~3k updates and holds there — the trunk-ified schedule; leaves
# take it to 3e-6.
COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", 
          "--chart-pool", "999999", "--mirror-augment",   # FULL doubled pool
          "--gamma", "0.965", "--gae-lambda", "0.98",
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
    resume = ["--resume", numbered[-1]] if numbered else []
    print(f"[trunk-ext] " + (f"resuming from {numbered[-1]}" if numbered
                             else "fresh start (full mirrored pool)"),
          flush=True)
    cmd = [PY, "train_batch.py", *COMMON, "--lr", LR, *resume,
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
                             f"v18_campaign_trunk{TRUNK_TOTAL // 1000}k_raw.json"))
    table = {f"trunk_raw_{TRUNK_TOTAL // 1000}k": res} if res else {}
    # the w24 validation-probe leaf is the standing baseline
    table["w24_probe_3k_leaf"] = {"upp": 1330255.25, "wpp": 10348.00}
    print("DONE " + json.dumps(table), flush=True)


if __name__ == "__main__":
    main()
