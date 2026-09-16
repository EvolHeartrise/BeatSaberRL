"""v29 gamma-isolation probe (user direction 2026-08-20): fresh trunk,
the v28 recipe verbatim (z-clip 40, N=32,768, mb16, mb-accum 2,
adapool critic, value normalization) with ONLY gamma moved 0.99333 ->
0.99 (value horizon 150 -> 100 ticks = 3.33 s).

Context: v28 (5 s horizon) opened -25% vs v27 at 1500, tracked ~12%
under the old v26-gamma line at 3000, recovered to within 3% of it at
4500 (+13% rung), then went flat through the extension window with a
psrank<srank inversion (19 vs 24) at ~6800 — the representation-
contraction signature, at the same depth where v26-gamma regressed.
User stopped the 7.5k extension and ordered this probe instead.

What it isolates: v26-gamma ran gamma 0.99 WITHOUT the fixes (blind
12 m clip, linear critic, no value norm — the v_loss-scale confound).
v28 has the fixes but a longer horizon. v29 = fixes + v26's exact
gamma, so at matched depth: v29 vs v26-gamma measures what the fixes
are worth at 0.99, and v29 vs v28 measures what horizon length alone
costs. v26-gamma anchors (recipe-unit benches 3000/6000 = these cuts):
upp 1,008,371 @1500 / 1,154,713 @3000. v28: 991,685 / 1,015,402.
v27 (gamma 0.965): 1,325,969 / 1,432,843.

Trunk 3,000 updates (recipe-6k), ckpt every 500; cd800 leaf from
upd 3000 only (user skipped the 1500 cut mid-run), benched with
BEAT_Z_CLIP_HI=40. Idempotent: re-run resumes the trunk from the
latest ckpt and skips benched cuts.

    python v29_g99_probe.py
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v29_g99")
NUM_ENVS, HORIZON = 32768, 8
TRUNK_UPDATES = 3_000
CUTS = [3_000]
COOLDOWN = 800
Z_CLIP_HI = "40"

COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--minibatches", "16", "--mb-accum", "2",
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", 
          "--chart-pool", "999999", "--mirror-augment",
          "--gamma", "0.99", "--gae-lambda", "0.98",
          "--z-clip-hi", Z_CLIP_HI,
          "--energy-coef", "7.5e-6",  # env v43: was --iso-coef 5e-4 (retired 2026-09-14)
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
        print(f"[v29-probe] FAILED {log_path}: {tail}", flush=True)
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
        print(f"[v29-probe] trunk {TRUNK_UPDATES} updates"
              + (f" (resuming {os.path.basename(numbered[-1])})"
                 if resume else " (FRESH: z-clip 40, gamma 0.99)"),
              flush=True)
        w = run([PY, "train_batch.py", *COMMON, *resume,
                 "--total-steps", str(TRUNK_UPDATES * NUM_ENVS * HORIZON),
                 "--ckpt-every", "500", "--run-dir", RUN_DIR],
                os.path.join(RUN_DIR, "trunk.log"), 21_600)
        print(f"[v29-probe] trunk done in {w/60:.1f} min", flush=True)

    for cut in CUTS:
        ck = os.path.join(RUN_DIR, f"trunk_upd{cut:05d}.pt")
        rd = os.path.join("runs", f"v29_cut{cut:05d}_cd{COOLDOWN}")
        out = os.path.join("bench", f"v29_g99_leaf{cut:05d}.json")
        if os.path.exists(out):
            print(f"[v29-probe] cut {cut} already benched", flush=True)
            continue
        os.makedirs(rd, exist_ok=True)
        if not [f for f in os.listdir(rd) if f.startswith("leaf_from")]:
            print(f"[v29-probe] cutting cd{COOLDOWN} from upd {cut}",
                  flush=True)
            w = run([PY, "train_batch.py", *COMMON,
                     "--resume", ck,
                     "--cooldown-updates", str(COOLDOWN),
                     "--ent-final", "3e-6", "--run-dir", rd],
                    os.path.join(rd, "leaf.log"), 14_400)
            print(f"[v29-probe] cut {cut} leaf done in {w/60:.1f} min",
                  flush=True)
        leaf = [f for f in os.listdir(rd)
                if f.startswith("leaf_from") and f.endswith(".pt")][0]
        # the bench backend is the RUST sim: it reads BEAT_Z_CLIP_HI
        run([PY, "bench_scoresaber.py",
             "--ckpt", os.path.join(rd, leaf),
             "--num-envs", "2048", "--foot-contact", "--out", out],
            os.path.join(rd, "bench.log"), 10_800,
            env_extra={"BEAT_Z_CLIP_HI": Z_CLIP_HI})
        print(f"PROBE_RESULT v29_g99@{cut} {leaf_stats(rd, out)}",
              flush=True)
    print("V29-PROBE-DONE", flush=True)


if __name__ == "__main__":
    main()
