"""v27 cooldown-length sweep (user direction 2026-08-19): cd400 was
set once at the v25 shape and never swept. Branch cd50 / cd100 /
cd400 leaves from the SAME v27 trunk checkpoint (trunk_upd01500) and
bench each; the v27 probe's own cd200 leaf is the fourth arm (its
bench already exists). In recipe-equivalent env-step units the arms
are cd100/200/400/800 — brackets the EPFL 2405.18392 10-20%-of-trunk
guidance from both sides (trunk 1,500 updates: 10-20% = 150-300).

Everything held except cooldown length: same 1-sqrt lr decay to 1e-4,
same ent anneal to 3e-6 (a shorter cd IS a faster ent crash — that
coupling is part of what "cooldown length" means here). Judged by
leaf-end ep_return (house metric), upp crater veto, EV co-read.

    python v27_cd_sweep.py     # requires runs/v27_critic/trunk_upd01500.pt
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from v27_critic_probe import COMMON, RUN_DIR as PROBE_DIR

PY = sys.executable
ARMS = [50, 100, 400]          # cd200 = the probe's own leaf
TRUNK_CK = os.path.join(PROBE_DIR, "trunk_upd01500.pt")


def run_logged(cmd, log_path, timeout):
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
    if r.returncode != 0:
        tail = open(log_path, encoding="utf-8").read()[-800:]
        sys.exit(f"[cd-sweep] FAILED {log_path}: {tail}")
    return time.time() - t0


def leaf_stats(run_dir, bench_json):
    b = json.load(open(bench_json, encoding="utf-8"))
    lv = [x for x in b["levels"] if "skipped" not in x]
    fcs = sum(1 for x in lv if x.get("full_combo"))
    zeros = sum(1 for x in lv if x.get("accuracy", 0.0) < 0.01)
    ret = ev = "?"
    with open(os.path.join(run_dir, "leaf.log"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("update"):
                parts = line.split()
                ret = parts[5]
                if "ev" in parts:
                    ev = parts[parts.index("ev") + 1]
    return (f"leaf_return {ret} leaf_ev {ev} "
            f"upp {b['total_unweighted_pp']:.2f} "
            f"wpp {b['total_weighted_pp']:.2f} fcs {fcs} zeros {zeros} "
            f"flail {b.get('flail_rad_per_tick', -1):.5f}")


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    if not os.path.exists(TRUNK_CK):
        sys.exit(f"[cd-sweep] missing {TRUNK_CK} — run the probe first")
    for cd in ARMS:
        rd = os.path.join("runs", f"v27_cd{cd:03d}")
        out = os.path.join("bench", f"v27_critic_leaf01500_cd{cd:03d}.json")
        if os.path.exists(out):
            print(f"[cd-sweep] cd{cd} already benched, skipping", flush=True)
            continue
        os.makedirs(rd, exist_ok=True)
        if not [f for f in os.listdir(rd) if f.startswith("leaf_from")]:
            print(f"[cd-sweep] cutting cd{cd}", flush=True)
            w = run_logged([PY, "train_batch.py", *COMMON,
                            "--resume", TRUNK_CK,
                            "--cooldown-updates", str(cd),
                            "--ent-final", "3e-6", "--run-dir", rd],
                           os.path.join(rd, "leaf.log"), 7_200)
            print(f"[cd-sweep] cd{cd} leaf done in {w/60:.1f} min",
                  flush=True)
        leaf = [f for f in os.listdir(rd)
                if f.startswith("leaf_from") and f.endswith(".pt")][0]
        run_logged([PY, "bench_scoresaber.py",
                    "--ckpt", os.path.join(rd, leaf),
                    "--num-envs", "2048", "--foot-contact", "--out", out],
                   os.path.join(rd, "bench.log"), 10_800)
        print(f"SWEEP_RESULT v27_cd{cd:03d} {leaf_stats(rd, out)}",
              flush=True)

    # the probe's cd200 arm, restated for the summary table
    probe_bench = os.path.join("bench", "v27_critic_leaf01500.json")
    if os.path.exists(probe_bench):
        print(f"SWEEP_RESULT v27_cd200 {leaf_stats(PROBE_DIR, probe_bench)}",
              flush=True)
    print("SWEEP-DONE", flush=True)


if __name__ == "__main__":
    main()
