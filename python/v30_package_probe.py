"""v30 posture/physics package probe (user direction 2026-08-20):
fresh trunk, gamma BACK to 0.965 (the gamma ladder closed: 0.965 >>
0.99 > 0.99333 at matched depth — v29 settled it), z-clip 40 kept
(era standard, no cost), carrying the full 2026-08-20 package:

  reward/env: WALL_PEN -0.5 once on head-entry edge (was -0.05/tick);
    FALL_ROOT_Y 0.55 -> 0.45; TILT termination (root up-axis > 30 deg
    sagittal / > 40 deg lateral at tick end, flat -fall_const);
    foot-lift -0.1 on the plant->lift edge (arm-after-first-plant);
    push/start kicks capped U(0, 0.15) (were 0.5 / 0.2).
  body: knee tau 250 -> 150; other leg joints -25% (262.5/75/135/
    135/37.5); shoulder pitch/yaw 113.84 -> 85.38.

Package probe: attribution is deliberately AT THE SET level, judged
against the stored v27 line at matched depth (cd400@1500 1,325,969 /
cd400@3000 1,432,843 / cd800@4500 1,491,015) — those numbers were
measured under the OLD physics; never re-bench old-era ckpts on this
binary. Watch [tilt]/[lift]/[fall] early: a fresh random policy under
the tilt gates is the first live read on whether the posture rules
and exploration coexist. Duck-vs-eat-the-wall is the behavioral
question; walls_in on the benches is the tell.

Trunk 4,500 updates (recipe-9k), ckpt every 500; cd800 leaves from
upd 1500 / 3000 / 4500, benched with BEAT_Z_CLIP_HI=40. Idempotent:
re-run resumes the trunk from the latest ckpt, skips benched cuts.

    python v30_package_probe.py
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v30_package")
NUM_ENVS, HORIZON = 32768, 8
TRUNK_UPDATES = 4_500
CUTS = [1_500, 3_000, 4_500]
# user 2026-08-20: benches skipped for the 1500/3000 leaves — leaf
# training acc is the read there; only the terminal leaf gets the
# full bench (leaves are still cut at every rung)
BENCH_CUTS = [4_500]
COOLDOWN = 800
Z_CLIP_HI = "40"

COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--minibatches", "16", "--mb-accum", "2",
          "--foot-contact", "--gears", "5", "--seed", "0",
          "--ranked-maps", "--seg-len", "25",
          "--chart-pool", "999999", "--mirror-augment",
          "--gamma", "0.965", "--gae-lambda", "0.98",
          "--z-clip-hi", Z_CLIP_HI,
          "--iso-coef", "5e-4",
          "--ent-coef", "0.03", "--ent-tau", "500", "--ent-final", "1e-3",
          "--lead-s", "0.5", "--lr", "7e-4",
          "--start-pose-jitter", "0.15", "--foot-z-coef", "0.01",
          "--foot-x-coef", "0.01",
          "--foot-hinge-r", "0.12", "--foot-shape-coef", "0.05"]


def leaf_only_stats(run_dir):
    """leaf_stats minus the bench fields — field-name parsing (never
    positional) of the leaf log's last update line."""
    ret = ev = acc = "?"
    with open(os.path.join(run_dir, "leaf.log"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("update"):
                parts = line.split()

                def fld(name, parts=parts):
                    return (parts[parts.index(name) + 1]
                            if name in parts else "?")
                ret = fld("ep_return(last100)")
                ev = fld("ev")
                acc = fld("acc")
    return f"leaf_return {ret} leaf_acc {acc} leaf_ev {ev} bench SKIPPED"


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
        print(f"[v30-probe] FAILED {log_path}: {tail}", flush=True)
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
        print(f"[v30-probe] trunk {TRUNK_UPDATES} updates"
              + (f" (resuming {os.path.basename(numbered[-1])})"
                 if resume else " (FRESH: 2026-08-20 package, g0.965)"),
              flush=True)
        w = run([PY, "train_batch.py", *COMMON, *resume,
                 "--total-steps", str(TRUNK_UPDATES * NUM_ENVS * HORIZON),
                 "--ckpt-every", "500", "--run-dir", RUN_DIR],
                os.path.join(RUN_DIR, "trunk.log"), 21_600)
        print(f"[v30-probe] trunk done in {w/60:.1f} min", flush=True)

    for cut in CUTS:
        ck = os.path.join(RUN_DIR, f"trunk_upd{cut:05d}.pt")
        rd = os.path.join("runs", f"v30_cut{cut:05d}_cd{COOLDOWN}")
        out = os.path.join("bench", f"v30_package_leaf{cut:05d}.json")
        if cut in BENCH_CUTS and os.path.exists(out):
            print(f"[v30-probe] cut {cut} already benched", flush=True)
            continue
        os.makedirs(rd, exist_ok=True)
        if not [f for f in os.listdir(rd) if f.startswith("leaf_from")]:
            print(f"[v30-probe] cutting cd{COOLDOWN} from upd {cut}",
                  flush=True)
            w = run([PY, "train_batch.py", *COMMON,
                     "--resume", ck,
                     "--cooldown-updates", str(COOLDOWN),
                     "--ent-final", "3e-6", "--run-dir", rd],
                    os.path.join(rd, "leaf.log"), 14_400)
            print(f"[v30-probe] cut {cut} leaf done in {w/60:.1f} min",
                  flush=True)
        if cut not in BENCH_CUTS:
            print(f"PROBE_RESULT v30_package@{cut} {leaf_only_stats(rd)}",
                  flush=True)
            continue
        leaf = [f for f in os.listdir(rd)
                if f.startswith("leaf_from") and f.endswith(".pt")][0]
        # the bench backend is the RUST sim: it reads BEAT_Z_CLIP_HI
        run([PY, "bench_scoresaber.py",
             "--ckpt", os.path.join(rd, leaf),
             "--num-envs", "2048", "--foot-contact", "--out", out],
            os.path.join(rd, "bench.log"), 10_800,
            env_extra={"BEAT_Z_CLIP_HI": Z_CLIP_HI})
        print(f"PROBE_RESULT v30_package@{cut} {leaf_stats(rd, out)}",
              flush=True)
    print("V30-PROBE-DONE", flush=True)


if __name__ == "__main__":
    main()
