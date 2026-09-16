"""v27 critic+N32k probe (user direction 2026-08-19): the adapool_v1
critic redesign AND the N=32,768 batch scale-up, evaluated as ONE
package. Gamma back to 0.965 — this is NOT a reward-signal probe.

What changed vs the v25 recipe (everything else verbatim):
  - N 16,384 -> 32,768 (5090 N-ladder peak: 114.3k steps/s exec-true,
    +31.5% over the 4090 at N=16,384; needs --mb-accum 2 +
    expandable_segments — plain N=32k OOMs on graph pools + the
    env's non-torch state)
  - minibatches 8 -> 16: mb_size HELD at 16,384 rows (the batch
    doubled; literature says add gradient steps, never grow the
    minibatch). Steps-per-gradient-step ratio is IDENTICAL to the
    recipe; each update consumes 2x env steps.
  - update-indexed schedules retimed /2 to keep the same env-step
    curves: ent-tau 1000 -> 500, trunk 3,000 -> 1,500 updates
    (= the same 393M env steps as a 3k-update recipe trunk),
    cd400 -> cd200, ckpt 1000 -> 500.
  - trainer defaults new since 2026-08-19 ride along: adapool_v1
    critic (~231k params, 1,093,906 total), value-target
    normalization, critic lr x3 group, vf-coef 0.5, [diag] every 50
    updates (grad cosine / dormant / srank land in trunk.log).

    python v27_critic_probe.py

Judges vs the v25 3k leaf (+108.2 ret / 1,293,592 upp / 461 FCs /
6 zeros) and the v26 gamma 3k leaf (1,008,371 upp). Leaf-end
ep_return rules (raw undiscounted — comparable across all lines),
upp crater veto. Also read: EV at leaf end (new critic quality) and
the [diag] gradcos trace (interference check at depth).
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v27_critic")
NUM_ENVS, HORIZON = 32768, 8
TRUNK_UPDATES = 1_500          # = 393M env steps = recipe-3k depth
COOLDOWN = 200                 # cd400 retimed for the doubled batch

# v25 recipe verbatim except the N32k package (see module docstring)
COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--minibatches", "16", "--mb-accum", "2",
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", 
          "--chart-pool", "999999", "--mirror-augment",
          "--gamma", "0.965", "--gae-lambda", "0.98",
          "--energy-coef", "7.5e-6",  # env v43: was --iso-coef 5e-4 (retired 2026-09-14)
          "--ent-coef", "0.03", "--ent-tau", "500", "--ent-final", "1e-3",
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
        print(f"[v27-probe] FAILED {log_name}: {tail}", flush=True)
        sys.exit(1)
    return time.time() - t0


def main():
    # N=32k does not fit without this (fragmentation between the env's
    # non-torch allocations, the graph pools, and the update phase)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    os.makedirs(RUN_DIR, exist_ok=True)
    done_ck = os.path.join(RUN_DIR, f"trunk_upd{TRUNK_UPDATES:05d}.pt")
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "trunk_upd*.pt")))
    resume = (["--resume", numbered[-1]]
              if numbered and not os.path.exists(done_ck) else [])
    if not os.path.exists(done_ck):
        print(f"[v27-probe] trunk {TRUNK_UPDATES} updates"
              + (f" (resuming {os.path.basename(numbered[-1])})"
                 if resume else " (fresh: adapool critic + N32k)"),
              flush=True)
        w = run([PY, "train_batch.py", *COMMON, *resume,
                 "--total-steps", str(TRUNK_UPDATES * NUM_ENVS * HORIZON),
                 "--ckpt-every", "500", "--run-dir", RUN_DIR],
                "trunk.log", 21_600)
        print(f"[v27-probe] trunk done in {w/60:.1f} min", flush=True)

    w = run([PY, "train_batch.py", *COMMON,
             "--resume", done_ck,
             "--cooldown-updates", str(COOLDOWN),
             "--ent-final", "3e-6",
             "--run-dir", RUN_DIR], "leaf.log", 7_200)
    print(f"[v27-probe] leaf done in {w/60:.1f} min", flush=True)

    leaf = [f for f in os.listdir(RUN_DIR)
            if f.startswith("leaf_from") and f.endswith(".pt")][0]
    leaf_path = os.path.join(RUN_DIR, leaf)
    out = os.path.join("bench", "v27_critic_leaf01500.json")
    run([PY, "bench_scoresaber.py",
         "--ckpt", leaf_path,
         "--num-envs", "2048", "--foot-contact", "--out", out],
        "bench.log", 10_800)

    b = json.load(open(out, encoding="utf-8"))
    lv = [x for x in b["levels"] if "skipped" not in x]
    fcs = sum(1 for x in lv if x.get("full_combo"))
    zeros = sum(1 for x in lv if x.get("accuracy", 0.0) < 0.01)
    ret = ev = "?"
    with open(os.path.join(RUN_DIR, "leaf.log"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("update"):
                parts = line.split()
                ret = parts[5]
                if "ev" in parts:
                    ev = parts[parts.index("ev") + 1]
    print(f"PROBE_RESULT v27_critic@{TRUNK_UPDATES} leaf_return {ret} "
          f"leaf_ev {ev} "
          f"upp {b['total_unweighted_pp']:.2f} "
          f"wpp {b['total_weighted_pp']:.2f} fcs {fcs} zeros {zeros} "
          f"flail {b.get('flail_rad_per_tick', -1):.5f}",
          flush=True)

    # top-10 pp replays: elbow-at-contact protocol continuity
    run([PY, "make_top10_replays.py", "--ckpt", leaf_path,
         "--bench", out, "--tag", "v27crit3k",
         "--top", "10", "--workers", "8", "--foot-contact"],
        "replays.log", 7_200)
    print("[v27-probe] replays done", flush=True)


if __name__ == "__main__":
    main()
