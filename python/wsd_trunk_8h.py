"""8h WSD trunk extension: continue the r3 trunk (NOT from scratch).

The r3 A/B trunk (runs/wsd_probe_1h_r3, 3,000 updates at lr 2e-3,
ent tau 1000 -> floor 1e-3) ended with falls still declining (2,475)
and benched 166.4 upp through a 400-update leaf — parity with the
cosine baseline plus 6x better balance (149 falls vs 859). Extension
banks those 3,000 updates: --resume continues the exact lr/ent curves
and Adam state (2026-08-11 user decision: extend, don't restart).

Plan: trunk 3,000 -> 21,000 updates (+18,000 = 2.36B new steps,
~7.1h at ~92k steps/s), trunk ckpts every 500 (each an extension /
branch point), then a 2,000-update leaf (~9.5% of trunk, ~48 min),
then bench raw trunk end + leaf.

  python wsd_trunk_8h.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "wsd_probe_1h_r3")   # the ongoing main trunk
NUM_ENVS, HORIZON = 16384, 8
BATCH = NUM_ENVS * HORIZON
TRUNK_TOTAL, LEAF_UPDATES = 21_000, 2_000
TRAIN_TIMEOUT = 42_000          # 11.6h hard stop for a ~7h phase

COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--foot-contact", "--gears", "5",
          "--seed", "0",
          "--ranked-maps", "--seg-len", "25",
          "--chart-pool", "24576",
          "--gamma", "0.965", "--gae-lambda", "0.98",
          "--effort-coef", "1e-4",
          "--ent-coef", "0.03"]


def run_phase(label, cmd, log_name):
    print(f"[{label}] -> {RUN_DIR}/{log_name}", flush=True)
    t0 = time.time()
    with open(os.path.join(RUN_DIR, log_name), "w", encoding="utf-8") as log:
        try:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               text=True, timeout=TRAIN_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"[{label}] TIMEOUT ({TRAIN_TIMEOUT}s)", flush=True)
            return False
    wall = time.time() - t0
    if r.returncode != 0:
        tail = open(os.path.join(RUN_DIR, log_name),
                    encoding="utf-8").read()[-800:]
        print(f"[{label}] FAILED ({wall:.0f}s): {tail}", flush=True)
        return False
    print(f"[{label}] done in {wall/60:.1f} min", flush=True)
    return True


def bench(ck, out):
    try:
        r = subprocess.run([PY, "bench_scoresaber.py", "--ckpt", ck,
                            "--out", out, "--num-envs", "2048",
                            "--foot-contact"],
                           capture_output=True, text=True, timeout=10800)
    except subprocess.TimeoutExpired:
        print(f"BENCH TIMEOUT {ck} (10800s)", flush=True)
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
    leaf_ck = os.path.join(RUN_DIR, f"leaf_from_upd{TRUNK_TOTAL:05d}.pt")
    assert os.path.exists(trunk_ck), f"{trunk_ck} missing"

    import torch
    upd = torch.load(trunk_ck, map_location="cpu",
                     weights_only=True)["update"]
    if upd >= TRUNK_TOTAL:
        print(f"trunk already at update {upd:,} — skipping extension",
              flush=True)
    else:
        ok = run_phase("trunk-ext", [PY, "train_batch.py", *COMMON,
                                     "--resume", trunk_ck,
                                     "--total-steps",
                                     str(TRUNK_TOTAL * BATCH),
                                     "--ckpt-every", "500",
                                     "--run-dir", RUN_DIR], "trunk_ext.log")
        if not ok:
            return

    if os.path.exists(leaf_ck):
        print(f"{leaf_ck} exists — skipping leaf", flush=True)
    else:
        ok = run_phase("leaf", [PY, "train_batch.py", *COMMON,
                                "--resume", trunk_ck,
                                "--cooldown-updates", str(LEAF_UPDATES),
                                "--ent-final", "3e-6"], "leaf_8h.log")
        if not ok:
            return

    table = {}
    for name, ck in (("trunk_raw", trunk_ck), ("leaf", leaf_ck)):
        res = bench(ck, os.path.join("bench", f"wsd_8h_{name}.json"))
        if res:
            table[name] = res
            print(f"[{name}] upp {res['upp']:,.0f} / wpp {res['wpp']:,.2f} / "
                  f"FCs {res['fcs']} / acc {res['acc']:.4f}", flush=True)
    table["baseline_1h_wsd_leaf"] = {"upp": 166.42, "wpp": 9.23}
    table["baseline_1h_cosine"] = {"upp": 174.57, "wpp": 10.69}
    print("DONE " + json.dumps(table), flush=True)


if __name__ == "__main__":
    main()
