"""v28 z-clip + long-gamma probe (user direction 2026-08-20): fresh
trunk with obs z-lane clamp 12 -> 40 m AND gamma 0.965 -> 0.99333
(value horizon 28.6 -> 150 ticks = 5.0 s), everything else the v27
recipe verbatim (N=32,768, mb16, mb-accum 2, adapool critic, value
normalization — which removes the v26 confound where a gamma change
silently rescaled the value loss into the shared trunk).

Design rationale: the ranked pool's real note speeds (13.4M notes:
p50 19 m/s, 71% in 16-22) put the 12 m clip's exact-timing reach at
~0.63 s — fine for the 0.95 s horizon, structurally blind for longer
ones. 40 m at p50 = 2.1 s of exact timing; gamma 0.99333 prices the
full 0.5 s earliness window at +10.6% (vs +70.6% at 0.965, +16.2% at
v26's 0.99 which still bought one arm's elbow park). The probe asks:
with the obs grown to (partially) match, does a 5 s horizon pay for
itself — and does the +10.6% residual premium still buy the park?

Trunk 7,500 updates (recipe-15k, 1.97B steps; extended from 4,500 by
user direction 2026-08-20), ckpt every 500; cd800 leaves (user
direction; the two-depth cd law: longer won everywhere, plateau
400-800) every 1,500 updates: upd 1500 / 3000 / 4500 / 6000 / 7500,
each benched with BEAT_Z_CLIP_HI=40 (the rust bench backend reads the
env var). Idempotent: re-running resumes the trunk from the latest
stored ckpt and skips already-benched cuts, so the 7.5k extension is
just this same script run again after the 4.5k pass.

Judges vs the v27 line at matched depth: cd400@1500 1,325,969;
cd400@3000 1,432,843; cd800@4500 1,491,015; cd800@6000 1,528,494
(all-time record); 7500 is past v27's deepest point. acc is THE
training metric; elbow-at-contact on the winner decides the posture
claim (v27 12k-equiv parked at L 3.0 / R 4.5 despite acc records).

    python v28_zgamma_probe.py
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v28_zgamma")
NUM_ENVS, HORIZON = 32768, 8
TRUNK_UPDATES = 7_500
CUTS = [1_500, 3_000, 4_500, 6_000, 7_500]
COOLDOWN = 800
Z_CLIP_HI = "40"

COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--minibatches", "16", "--mb-accum", "2",
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", "--seg-len", "25",
          "--chart-pool", "999999", "--mirror-augment",
          "--gamma", "0.99333", "--gae-lambda", "0.98",
          "--z-clip-hi", Z_CLIP_HI,
          "--effort-coef", "0", "--iso-coef", "5e-4",
          "--ent-coef", "0.03", "--ent-tau", "500", "--ent-final", "1e-3",
          "--lead-s", "0.5", "--lr", "7e-4",
          "--start-pose-jitter", "0.15", "--foot-z-coef", "0.01",
          "--foot-x-coef", "0.01",
          "--foot-hinge-r", "0.12", "--foot-shape-coef", "0.05"]


def run(cmd, log_path, timeout, env_extra=None):
    t0 = time.time()
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    with open(log_path, "a", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        tail = open(log_path, encoding="utf-8").read()[-800:]
        print(f"[v28-probe] FAILED {log_path}: {tail}", flush=True)
        sys.exit(1)
    return time.time() - t0


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    os.makedirs(RUN_DIR, exist_ok=True)
    from v27_cd_sweep import leaf_stats

    done_ck = os.path.join(RUN_DIR, f"trunk_upd{TRUNK_UPDATES:05d}.pt")
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "trunk_upd*.pt")))
    resume = (["--resume", numbered[-1]]
              if numbered and not os.path.exists(done_ck) else [])
    if not os.path.exists(done_ck):
        print(f"[v28-probe] trunk {TRUNK_UPDATES} updates"
              + (f" (resuming {os.path.basename(numbered[-1])})"
                 if resume else " (FRESH: z-clip 40, gamma 0.99333)"),
              flush=True)
        w = run([PY, "train_batch.py", *COMMON, *resume,
                 "--total-steps", str(TRUNK_UPDATES * NUM_ENVS * HORIZON),
                 "--ckpt-every", "500", "--run-dir", RUN_DIR],
                os.path.join(RUN_DIR, "trunk.log"), 21_600)
        print(f"[v28-probe] trunk done in {w/60:.1f} min", flush=True)

    for cut in CUTS:
        ck = os.path.join(RUN_DIR, f"trunk_upd{cut:05d}.pt")
        rd = os.path.join("runs", f"v28_cut{cut:05d}_cd{COOLDOWN}")
        out = os.path.join("bench", f"v28_zgamma_leaf{cut:05d}.json")
        if os.path.exists(out):
            print(f"[v28-probe] cut {cut} already benched", flush=True)
            continue
        os.makedirs(rd, exist_ok=True)
        if not [f for f in os.listdir(rd) if f.startswith("leaf_from")]:
            print(f"[v28-probe] cutting cd{COOLDOWN} from upd {cut}",
                  flush=True)
            w = run([PY, "train_batch.py", *COMMON,
                     "--resume", ck,
                     "--cooldown-updates", str(COOLDOWN),
                     "--ent-final", "3e-6", "--run-dir", rd],
                    os.path.join(rd, "leaf.log"), 14_400)
            print(f"[v28-probe] cut {cut} leaf done in {w/60:.1f} min",
                  flush=True)
        leaf = [f for f in os.listdir(rd)
                if f.startswith("leaf_from") and f.endswith(".pt")][0]
        # the bench backend is the RUST sim: it reads BEAT_Z_CLIP_HI
        run([PY, "bench_scoresaber.py",
             "--ckpt", os.path.join(rd, leaf),
             "--num-envs", "2048", "--foot-contact", "--out", out],
            os.path.join(rd, "bench.log"), 10_800,
            env_extra={"BEAT_Z_CLIP_HI": Z_CLIP_HI})
        print(f"PROBE_RESULT v28_zgamma@{cut} {leaf_stats(rd, out)}",
              flush=True)
    print("V28-PROBE-DONE", flush=True)


if __name__ == "__main__":
    main()
