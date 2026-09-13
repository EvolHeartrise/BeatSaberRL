"""v18 fall-robustness probe: one arm of the 3-arm A/B (2026-08-15).

    python v18_fall_probe.py <arm>     # arm: const | constpush

Control arm = the fpm campaign's existing 3k leaf (return +96.8, upp
300,132, wpp 8,115 — runs/v17_fpm_leaf03000); these arms rerun the
same fpm recipe (full mirrored pool, lr 7e-4, 3k trunk + cd400 leaf +
bench) with the v18 changes:

  const:     --fall-pen 5.0 (flat terminal reward; kills the
             sparse-state fall-indifference exploit — replay analysis
             showed zero-acc maps are engage-and-topple falls at
             median 4.4 s, priced near-zero by remaining-notes charge)
  constpush: const + random horizontal root kicks (push-recovery lit:
             ~1 per 10 s per env, |dv| <= 0.5 m/s) + 0.2 m/s
             episode-start jitter (RSI-flavored start variety)

Judges: leaf-end return (ruling metric), bench ZERO-COUNT (levels at
acc < 1% — the failure this targets), fall telemetry in the trunk
log, upp/wpp as sanity.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = sys.executable
ARM = sys.argv[1]
ARMS = {
    # const/constpush flags are now the train_batch DEFAULTS; kept for
    # provenance. w24 = the env v18 window-24 validation arm (obs 578):
    # pure defaults, compared against the constpush 3k baseline.
    "const": ["--fall-pen", "5.0"],
    "constpush": ["--fall-pen", "5.0", "--push-prob", "0.00333",
                  "--push-vmax", "0.5", "--start-jitter", "0.2"],
    "w24": [],
    # N learning gate (2026-08-15): same PPO batch (65,536 x 2 =
    # 131,072) and update count, but the rollout pays 2 env ticks per
    # update instead of 8 — the wall-clock arm. MUST run with
    # BEAT_CHOL_RL=1 BEAT_MIN_BLOCKS=16 exported (the flag-pair kernel:
    # 352.8 ms @65k, the +39%/env occupancy win). Judged on leaf-end
    # return vs the w24 baseline (+118.2) AND trunk wall minutes; the
    # learning risk under test is the T=2 GAE bootstrap horizon.
    "n65": ["--num-envs", "65536", "--horizon", "2"],
}
EXTRA = ARMS[ARM]
RUN_DIR = os.path.join("runs", f"v18_fall_{ARM}")
NUM_ENVS, HORIZON = 16384, 8
# n65's two-sided win condition (user, 2026-08-15): leaf at 3,000 =
# equal-optimization comparison; leaf at 3,750 = equal-WALL comparison
# (3,750 cheaper updates ~ the baseline's 77-min trunk).
TRUNK_UPDATES = 3_750 if ARM == "n65" else 3_000
LEAF_CUTS = [3_000, 3_750] if ARM == "n65" else [3_000]
COOLDOWN = 400

# the fpm campaign recipe verbatim (v17_trunk_long.py COMMON)
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
        print(f"[fall-probe {ARM}] FAILED {log_name}: {tail}", flush=True)
        sys.exit(1)
    return time.time() - t0


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    batch = NUM_ENVS * HORIZON
    import glob
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "trunk_upd*.pt")))
    done_ck = os.path.join(RUN_DIR, f"trunk_upd{TRUNK_UPDATES:05d}.pt")
    resume = (["--resume", numbered[-1]]
              if numbered and not os.path.exists(done_ck) else [])
    print(f"[fall-probe {ARM}] trunk {TRUNK_UPDATES} updates"
          + (f" (resuming {os.path.basename(numbered[-1])})" if resume
             else " (already complete)" if os.path.exists(done_ck) else ""),
          flush=True)
    if not os.path.exists(done_ck):
        w = run([PY, "train_batch.py", *COMMON, *EXTRA, *resume,
                 "--total-steps", str(TRUNK_UPDATES * batch),
                 "--ckpt-every", "1000", "--run-dir", RUN_DIR],
                "trunk.log", 21_600)
        print(f"[fall-probe {ARM}] trunk done in {w/60:.1f} min", flush=True)

    for cut in LEAF_CUTS:
        cut_ck = os.path.join(RUN_DIR, f"trunk_upd{cut:05d}.pt")
        leaf_log = f"leaf_{cut:05d}.log"
        w = run([PY, "train_batch.py", *COMMON, *EXTRA,
                 "--resume", cut_ck,
                 "--cooldown-updates", str(COOLDOWN),
                 "--ent-final", "3e-6",
                 "--run-dir", RUN_DIR], leaf_log, 7_200)
        print(f"[fall-probe {ARM}] leaf@{cut} done in {w/60:.1f} min",
              flush=True)

        leaf = [f for f in os.listdir(RUN_DIR)
                if f.startswith("leaf_from") and f"{cut:05d}" in f
                and f.endswith(".pt")][0]
        out = os.path.join("bench", f"v18_fall_{ARM}_{cut:05d}.json")
        run([PY, "bench_scoresaber.py",
             "--ckpt", os.path.join(RUN_DIR, leaf),
             "--num-envs", "2048", "--foot-contact", "--out", out],
            f"bench_{cut:05d}.log", 10_800)
        b = json.load(open(out, encoding="utf-8"))
        lv = [x for x in b["levels"] if "skipped" not in x]
        fcs = sum(1 for x in lv if x.get("full_combo"))
        zeros = sum(1 for x in lv if x.get("accuracy", 0.0) < 0.01)
        ret = "?"
        with open(os.path.join(RUN_DIR, leaf_log), encoding="utf-8") as f:
            for line in f:
                if line.startswith("update"):
                    ret = line.split()[5]
        print(f"PROBE_RESULT {ARM}@{cut} leaf_return {ret} "
              f"upp {b['total_unweighted_pp']:.2f} "
              f"wpp {b['total_weighted_pp']:.2f} fcs {fcs} zeros {zeros}",
              flush=True)


if __name__ == "__main__":
    main()
