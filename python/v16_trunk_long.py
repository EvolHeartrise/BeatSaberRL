"""Long v16 trunk: extend the breakthrough probe's trunk 3,000 ->
24,000 updates (~8h). The 1h probe (69,658 upp leaf, 419x the
old-reward baseline, wave churn gone) earned the drift test proper:
does leaf value still peak-and-decay under discounted fall pricing +
0.5s lead, or does it keep compounding? Leaf cuts happen OFF-POD
(cut_leaf.sh throwaway pods, laptop-orchestrated, every 3k updates);
this driver only extends the trunk and benches the raw end.

  python v16_trunk_long.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v16_probe_1h")   # the ongoing v16 trunk
NUM_ENVS, HORIZON = 16384, 8
BATCH = NUM_ENVS * HORIZON
TRUNK_TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 24_000
TRAIN_TIMEOUT = 42_000

COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--foot-contact", "--gears", "5",
          "--seed", "0",
          "--ranked-maps", "--seg-len", "25",
          "--chart-pool", "24576",
          "--gamma", "0.965", "--gae-lambda", "0.98",
          "--effort-coef", "1e-4",
          "--ent-coef", "0.03",
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
    trunk_ck = os.path.join(RUN_DIR, "trunk_final.pt")
    assert os.path.exists(trunk_ck), f"{trunk_ck} missing"
    cmd = [PY, "train_batch.py", *COMMON,
           "--resume", trunk_ck,
           "--total-steps", str(TRUNK_TOTAL * BATCH),
           "--ckpt-every", "500",
           "--run-dir", RUN_DIR]
    print(f"[trunk-ext] v16 trunk -> {TRUNK_TOTAL:,} updates", flush=True)
    t0 = time.time()
    with open(os.path.join(RUN_DIR, "trunk_ext.log"), "w",
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

    res = bench(trunk_ck, os.path.join("bench", "v16_trunk24k_raw.json"))
    table = {"trunk_raw_24k": res} if res else {}
    table["baseline_1h_leaf"] = {"upp": 69658.43, "wpp": 3105.6}
    print("DONE " + json.dumps(table), flush=True)


if __name__ == "__main__":
    main()
