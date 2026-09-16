"""v26 gamma probe (user direction 2026-08-18): --gamma 0.965 -> 0.99,
everything else the v25 recipe verbatim. No env change.

Motivation (elbow-at-contact measurement, 2026-08-18): the 19k
champion plays with both elbows commanded full-gear into the 0-deg
extension stop 71% of ALL ticks (median elbow at cut = 0.0 deg;
humans play ~90-100 deg bent). The activation cost already charges
this at the maximum rate and the policy pays — the purchase motive
is the discount earliness premium: at gamma 0.965 a cut landed one
tick earlier is worth +3.6%, the full 0.5 s lead window +70%, and
the straight arm is the longest lever = earliest contact. At 0.99
the premium falls to +1.0%/tick (+16% window) and the horizon grows
0.95 s -> 3.3 s, long enough to credit posture/stability economics.
gae-lambda stays 0.98 (gamma*lambda 0.97, credit length ~33 ticks).

    python v26_gamma_probe.py

Judges vs the v25 hands 3k leaf baseline (+108.2 ret / 1,293,592
upp / 9,823 wpp / 461 FCs / 6 zeros / flail 0.7623), leaf-end
return ruling, upp crater veto. NOTE the ruling metric is not
comparable across gamma at the trunk level (different objective);
the LEAF-END return printed here is still the raw undiscounted
ep_return train_batch logs, so it stays comparable. Pass/fail
beyond return/pp: the elbow-at-contact distribution on the top-10
replays (recorded below) must shift away from the 0-deg limit park.
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
NUM_ENVS, HORIZON = 16384, 8
TRUNK_UPDATES = 3_000
COOLDOWN = 400

# the v25 recipe verbatim, except gamma 0.965 -> 0.99
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
        print(f"[gamma-probe] FAILED {log_name}: {tail}", flush=True)
        sys.exit(1)
    return time.time() - t0


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    done_ck = os.path.join(RUN_DIR, f"trunk_upd{TRUNK_UPDATES:05d}.pt")
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "trunk_upd*.pt")))
    resume = (["--resume", numbered[-1]]
              if numbered and not os.path.exists(done_ck) else [])
    if not os.path.exists(done_ck):
        print(f"[gamma-probe] trunk {TRUNK_UPDATES} updates"
              + (f" (resuming {os.path.basename(numbered[-1])})"
                 if resume else " (fresh, gamma 0.99 arm)"), flush=True)
        w = run([PY, "train_batch.py", *COMMON, *resume,
                 "--total-steps", str(TRUNK_UPDATES * NUM_ENVS * HORIZON),
                 "--ckpt-every", "1000", "--run-dir", RUN_DIR],
                "trunk.log", 21_600)
        print(f"[gamma-probe] trunk done in {w/60:.1f} min", flush=True)

    w = run([PY, "train_batch.py", *COMMON,
             "--resume", done_ck,
             "--cooldown-updates", str(COOLDOWN),
             "--ent-final", "3e-6",
             "--run-dir", RUN_DIR], "leaf.log", 7_200)
    print(f"[gamma-probe] leaf done in {w/60:.1f} min", flush=True)

    leaf = [f for f in os.listdir(RUN_DIR)
            if f.startswith("leaf_from") and f.endswith(".pt")][0]
    leaf_path = os.path.join(RUN_DIR, leaf)
    out = os.path.join("bench", "v26_gamma_leaf03000.json")
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
    print(f"PROBE_RESULT v26_gamma@3000 leaf_return {ret} "
          f"upp {b['total_unweighted_pp']:.2f} "
          f"wpp {b['total_weighted_pp']:.2f} fcs {fcs} zeros {zeros} "
          f"flail {b.get('flail_rad_per_tick', -1):.5f}",
          flush=True)

    # elbow pass/fail material: top-10 pp replays of this leaf, same
    # protocol as the 19k-champion measurement set (v25c19k)
    run([PY, "make_top10_replays.py", "--ckpt", leaf_path,
         "--bench", out, "--tag", "v26gamma3k",
         "--top", "10", "--workers", "8", "--foot-contact"],
        "replays.log", 7_200)
    print("[gamma-probe] replays done", flush=True)


if __name__ == "__main__":
    main()
