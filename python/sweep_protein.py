"""Protein hyperparameter sweep runner for train_batch.py (2026-07-27).

Sweep 1 ("entropy/effort/gamma"): the flailing + bomb-economics + horizon
cluster, from scratch, judged on TOTAL UNWEIGHTED PP over the FULL
ScoreSaber catalog (user decision 2026-07-27 — smoother low-end
gradient than band wpp and measures breadth; costs ~3-4 min/bench warm
vs ~40 s for the band). Cost axis = total_steps (Protein's multi-budget
scaling-law trick); cost signal = wall seconds of the training leg.
downsample=1 (one bench per leg) which also enables Protein's
success-probability classifier.

Space (7 dims):
  lr          log     1e-4 .. 3e-3   (cosine to 1e-4 fixed)
  ent_coef    log     1e-4 .. 0.1    initial entropy coef
  ent_final   log     1e-5 .. 0.1    linear anneal target (new flag)
  gamma       logit   0.90 .. 0.997  user hypothesis: ~1 s horizon (~0.967)
  effort_coef log     0.01 .. 8      flailing tax (log cannot express 0)
  bomb_prob   log     0.01 .. 0.4    bomb density (72%-of-reward question)
  total_steps log     5e7  .. 4e8    THE COST AXIS (3.3-27 min at ~250k/s)

State: every observation appended to <dir>/observations.jsonl and
replayed through observe() on restart (observe() is pure, so this fully
reconstructs the GP state — Protein itself has no save/load).

  python sweep_protein.py --max-runs 100 [--dir sweeps/s1_entropy_gamma]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy

from protein_sweep import Protein

PY = sys.executable
SPACE = {
    "metric": "catalog_unweighted_pp",
    "goal": "maximize",
    "metric_distribution": "linear",   # wpp is unbounded
    "downsample": 1,                   # one bench per leg
    "early_stop_quantile": 0.3,
    "use_gpu": False,                  # GP on CPU; the 1080 belongs to training
    "max_suggestion_cost": 2200,       # ~4e8 steps + startup, seconds
    "train": {
        "lr":          dict(distribution="log_normal", min=1e-4, max=3e-3, scale=0.5),
        "ent_coef":    dict(distribution="log_normal", min=1e-4, max=0.1, scale=0.5),
        "ent_final":   dict(distribution="log_normal", min=1e-5, max=0.1, scale=0.5),
        "gamma":       dict(distribution="logit_normal", min=0.90, max=0.997, scale=0.5),
        "effort_coef": dict(distribution="log_normal", min=0.01, max=8.0, scale=0.5),
        "bomb_prob":   dict(distribution="log_normal", min=0.01, max=0.4, scale=0.5),
        "total_steps": dict(distribution="log_normal", min=5e7, max=4e8, scale="time"),
    },
}
FILL = {"train": {k: None for k in SPACE["train"]}}


def run_leg(idx, h, run_dir):
    """One training leg + band bench. Returns (band_wpp, wall_s) or None."""
    cmd = [PY, "train_batch.py", "--num-envs", "1024",
           "--lr-decay", "cosine", "--seed", "0",
           "--lr", f"{h['lr']:.6g}",
           "--ent-coef", f"{h['ent_coef']:.6g}",
           "--ent-final", f"{h['ent_final']:.6g}",
           "--gamma", f"{h['gamma']:.6g}",
           "--effort-coef", f"{h['effort_coef']:.6g}",
           "--bomb-prob", f"{h['bomb_prob']:.6g}",
           "--total-steps", str(int(h["total_steps"])),
           "--run-dir", run_dir]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    wall = time.time() - t0
    ck = os.path.join(run_dir, "final.pt")
    if r.returncode != 0 or not os.path.exists(ck):
        tail = (r.stdout or "")[-400:] + (r.stderr or "")[-400:]
        print(f"[run {idx:03d}] TRAIN FAILED ({wall:.0f}s): {tail}",
              flush=True)
        return None
    out = os.path.join(run_dir, "bench.json")
    b = subprocess.run([PY, "bench_scoresaber.py", "--ckpt", ck,
                        "--out", out],
                       capture_output=True, text=True, timeout=3000)
    if b.returncode != 0 or not os.path.exists(out):
        print(f"[run {idx:03d}] BENCH FAILED: {(b.stderr or '')[-300:]}",
              flush=True)
        return None
    upp = json.load(open(out))["total_unweighted_pp"]
    return upp, wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-runs", type=int, default=100)
    p.add_argument("--dir", type=str, default="sweeps/s1_upp")
    args = p.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    log_path = os.path.join(args.dir, "observations.jsonl")

    protein = Protein(SPACE, cost_param="train/total_steps")

    start = 0
    if os.path.exists(log_path):
        for line in open(log_path, encoding="utf-8"):
            o = json.loads(line)
            protein.observe(o["hypers"], o["score"], o["cost"],
                            o["is_failure"])
            protein.suggestion_idx += 1  # keep Sobol phase consumed
            start += 1
        print(f"resumed: replayed {start} observations", flush=True)

    for idx in range(start, args.max_runs):
        hypers, info = protein.suggest(deepcopy(FILL))
        h = hypers["train"]
        run_dir = os.path.join(args.dir, f"run_{idx:03d}")
        print(f"[run {idx:03d}] lr {h['lr']:.2e} ent {h['ent_coef']:.2e}"
              f"->{h['ent_final']:.2e} gamma {h['gamma']:.4f} "
              f"effort {h['effort_coef']:.2f} bomb {h['bomb_prob']:.3f} "
              f"steps {h['total_steps']/1e6:.0f}M", flush=True)
        res = run_leg(idx, h, run_dir)
        if res is None:
            score, cost, fail = 0.0, 1.0, True
        else:
            score, cost = res
            fail = False
        protein.observe(hypers, score, cost, fail)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"idx": idx, "hypers": hypers,
                                "score": score, "cost": cost,
                                "is_failure": fail}) + "\n")
        print(f"[run {idx:03d}] -> catalog upp {score:,.0f} "
              f"({cost/60:.1f} min){' FAILED' if fail else ''}", flush=True)

    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
