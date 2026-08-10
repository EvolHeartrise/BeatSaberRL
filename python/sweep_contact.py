"""Protein sweep s4: the CONTACT regime, from scratch (2026-08-09).

Fresh profile — sweeps/s3_cutplane_upp is weld-physics obs-340
observations and is FROZEN for the same reason s2 was at the cut-plane
break: the objective changed underneath (env v11 unilateral contact +
v11.1 fall pricing + v12 contact obs/alive bonus + v13 body points =
a different game). This driver owns sweeps/s4_contact_ret and refuses
the old dirs.

Differences from sweep_ranked.py, each deliberate:

- **Env**: --foot-contact, --gears 5, v13 obs-362, and the c-shape
  16384/8 (the 2026-08-08 equal-wall winner; the old 1024/128 predates
  the N-scaling finding and runs 12x slower).
- **Objective**: mean ep_return(last100) over the final 10% of update
  lines, parsed from train.log — NOT catalog upp. At sweep-leg budgets
  a contact policy scores upp ~0-150 (v12 1h: 123), which is a
  zero-inflated landscape the GP cannot rank; ep_return discriminates
  at every budget (falls price the remaining chart, standing collects
  the alive bonus, hits climb from there) and is the exact scalar PPO
  optimizes. No bench per leg — that alone saves 10-40 min/leg.
  Failure score -120 (below any real return; 0.0 would be a GOOD
  ep_return on this scale — the s1-s3 convention of scoring failures 0
  is upp-specific and would poison this GP).
- **Space**: s3 bounds for lr/ent/gamma/lambda. gamma's 0.72..0.997
  box straddles both hypotheses (weld-tuned 0.915 vs the balance-
  horizon argument for ~0.97+ — 2026-08-09 session note); let the GP
  settle it. effort_coef re-bounded 1e-4..8.0: the s3 ceiling of 128
  was cut-plane weld economics; under contact the min-gear balance
  dither costs ~4.5 W, and at effort 128 that is -0.019/tick against
  the +0.002 alive bonus — standing itself would price negative, so
  the high decades would only buy dead legs.
- **Cost axis**: total_steps 6e7..1.2e9 (at the measured ~148k
  steps/s: ~7 min .. ~2.2 h/leg).

  python sweep_contact.py --max-runs 100 [--dir sweeps/s4_contact_ret]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy

from protein_sweep import Protein

PY = sys.executable
SEG_LEN = 25.0
SEG_POOL = 24576
NUM_ENVS, HORIZON = 16384, 8
FAIL_SCORE = -120.0

SPACE = {
    "metric": "final_ep_return",
    "goal": "maximize",
    "metric_distribution": "linear",
    "downsample": 1,
    "early_stop_quantile": 0.3,
    "use_gpu": False,
    # seconds; 1.2e9 steps at the 148k steps/s contact floor + startup
    "max_suggestion_cost": 8500,
    "train": {
        # widening #1 (2026-08-09, legs 000-038): lr ceiling 8e-3 ->
        # 1.5e-2 (run 028 EXACTLY on 8e-3, top-half median 6.2e-3),
        # gae_lambda ceiling 0.995 -> 0.999 (7 of top 10 on 0.995),
        # ent_final floor 1e-5 -> 1e-6 (top legs pinned). effort floor
        # NOT lowered: under the ep_return objective the effort axis is
        # REWARD-PARAMETERIZED (the tax is inside the score), so it pins
        # low mechanically and its true optimum is unidentifiable here —
        # pinned-at-floor == tax-off, decide the real tax externally.
        "lr":          dict(distribution="log_normal", min=1e-4, max=1.5e-2, scale=0.5),
        "ent_coef":    dict(distribution="log_normal", min=1e-4, max=0.3, scale=0.5),
        "ent_final":   dict(distribution="log_normal", min=1e-6, max=0.1, scale=0.5),
        "gamma":       dict(distribution="logit_normal", min=0.72, max=0.997, scale=0.5),
        "gae_lambda":  dict(distribution="logit_normal", min=0.65, max=0.999, scale=0.5),
        "effort_coef": dict(distribution="log_normal", min=1e-4, max=8.0, scale=0.5),
        "total_steps": dict(distribution="log_normal", min=6e7, max=1.2e9, scale="time"),
    },
}
FILL = {"train": {k: None for k in SPACE["train"]}}

_UPD_RE = re.compile(r"^update\s+\d+/\d+\s+steps\s+\d+\s+"
                     r"ep_return\(last100\)\s+(-?[\d.]+)")


def final_ep_return(log_path):
    """Mean ep_return(last100) over the final 10% of update lines
    (min 20). None if no update lines survived."""
    vals = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = _UPD_RE.match(line)
            if m:
                vals.append(float(m.group(1)))
    if not vals:
        return None
    tail = max(20, len(vals) // 10)
    tail_vals = vals[-tail:]
    return sum(tail_vals) / len(tail_vals)


def run_leg(idx, h, run_dir):
    """One contact-regime training leg. Returns (score, wall_s) or None."""
    shutil.rmtree(run_dir, ignore_errors=True)  # leg = always fresh model
    cmd = [PY, "train_batch.py",
           "--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
           "--foot-contact", "--gears", "5",
           "--lr-decay", "cosine", "--seed", "0",
           "--ranked-maps", "--seg-len", f"{SEG_LEN:g}",
           "--chart-pool", str(SEG_POOL),
           "--lr", f"{h['lr']:.6g}",
           "--ent-coef", f"{h['ent_coef']:.6g}",
           "--ent-final", f"{h['ent_final']:.6g}",
           "--gamma", f"{h['gamma']:.6g}",
           "--gae-lambda", f"{h['gae_lambda']:.6g}",
           "--effort-coef", f"{h['effort_coef']:.6g}",
           "--total-steps", str(int(h["total_steps"])),
           "--run-dir", run_dir]
    t0 = time.time()
    # 100k steps/s floor (contact measures ~148k at 16384/8) + startup
    tmo = int(h["total_steps"]) // 100_000 + 1800
    log_path = os.path.join(run_dir, "train.log")
    os.makedirs(run_dir, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               text=True, timeout=tmo)
        except subprocess.TimeoutExpired:
            print(f"[run {idx:03d}] TRAIN TIMEOUT ({tmo}s)", flush=True)
            return None
    wall = time.time() - t0
    ck = os.path.join(run_dir, "final.pt")
    if r.returncode != 0 or not os.path.exists(ck):
        tail = open(log_path, encoding="utf-8").read()[-400:]
        print(f"[run {idx:03d}] TRAIN FAILED ({wall:.0f}s): {tail}",
              flush=True)
        return None
    score = final_ep_return(log_path)
    if score is None:
        print(f"[run {idx:03d}] no update lines in train.log", flush=True)
        return None
    return score, wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-runs", type=int, default=100)
    p.add_argument("--dir", type=str, default="sweeps/s4_contact_ret")
    args = p.parse_args()
    norm = os.path.normpath(args.dir).replace("\\", "/")
    if norm.endswith(("sweeps/s2_ranked_upp", "sweeps/s3_cutplane_upp",
                      "sweeps/s1_upp")):
        sys.exit("s1/s2/s3 are FROZEN weld-physics observations; the "
                 "contact regime is a different objective. Use the s4 "
                 "default or a fresh dir.")
    os.makedirs(args.dir, exist_ok=True)
    log_path = os.path.join(args.dir, "observations.jsonl")

    protein = Protein(SPACE, cost_param="train/total_steps")

    start = 0
    if os.path.exists(log_path):
        for line in open(log_path, encoding="utf-8"):
            o = json.loads(line)
            protein.observe(o["hypers"], o["score"], o["cost"],
                            o["is_failure"])
            protein.suggestion_idx += 1
            start += 1
        print(f"resumed: replayed {start} observations", flush=True)

    for idx in range(start, args.max_runs):
        hypers, info = protein.suggest(deepcopy(FILL))
        h = hypers["train"]
        run_dir = os.path.join(args.dir, f"run_{idx:03d}")
        print(f"[run {idx:03d}] lr {h['lr']:.2e} ent {h['ent_coef']:.2e}"
              f"->{h['ent_final']:.2e} gamma {h['gamma']:.4f} "
              f"lam {h['gae_lambda']:.4f} "
              f"effort {h['effort_coef']:.4g} "
              f"steps {h['total_steps']/1e6:.0f}M", flush=True)
        res = run_leg(idx, h, run_dir)
        if res is None:
            score, cost, fail = FAIL_SCORE, 1.0, True
        else:
            score, cost = res
            fail = False
        protein.observe(hypers, score, cost, fail)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"idx": idx, "hypers": hypers,
                                "score": score, "cost": cost,
                                "is_failure": fail}) + "\n")
        print(f"[run {idx:03d}] -> ep_return {score:,.2f} "
              f"({cost/60:.1f} min){' FAILED' if fail else ''}", flush=True)

    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
