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

**WSD port (2026-09-30)**: see sweep_protein.py's WSD PORT note (trunk
+ leaf instead of the removed `--lr-decay cosine`; honest failure cost;
sub-minute crashes abort). Contact is the only physics now, so the
deprecated no-op `--foot-contact` is no longer passed. The objective's
`ep_return(last100)` field was dropped from the trainer's update line
on 2026-09-12 (too noisy); the score is now the mean `return` (reward
per env over the rollout, the trainer's stable population measure)
over the final 10% (min 20) of the leg's update records, read from the
trunk's and the leaf's metrics.jsonl — a different scale from s4, so
s4 is FROZEN and the default --dir is sweeps/s4_contact_ret_wsd.
FAIL_SCORE is only what the jsonl records: Protein refits every
failure at the running min score. effort_coef (above) left the space
with the v10 effort price.

  python sweep_contact.py --max-runs 100 [--dir sweeps/s4_contact_ret_wsd]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy

from sweep_protein import (log_observation, replay, run_wsd_train,
                           wsd_cmds)

# SEG_LEN retired 2026-09-16: whole-level episodes are the only pool mode
SEG_POOL = 24576
NUM_ENVS, HORIZON = 16384, 8
ENV_FLAGS = ["--gears", "5", "--ranked-maps", "--chart-pool", str(SEG_POOL)]
FAIL_SCORE = -120.0

SPACE = {
    "metric": "final_return",
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
        "total_steps": dict(distribution="log_normal", min=6e7, max=1.2e9, scale="time"),
    },
}
FILL = {"train": {k: None for k in SPACE["train"]}}

def final_return(metrics_paths):
    """Mean per-update `return` over the final 10% (min 20) of the update
    records in these metrics.jsonl files, in order (trunk, then leaf).
    None if no record carries a return."""
    vals = []
    for path in metrics_paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    v = json.loads(line).get("return")
                except json.JSONDecodeError:
                    continue   # a torn last line from a killed trainer
                if v is not None:
                    vals.append(float(v))
    if not vals:
        return None
    tail = max(20, len(vals) // 10)
    tail_vals = vals[-tail:]
    return sum(tail_vals) / len(tail_vals)


def train_cmds(h, run_dir):
    """(trunk_cmd, leaf_cmd, leaf_dir) for one WSD leg."""
    return wsd_cmds(h, run_dir, NUM_ENVS, HORIZON, ENV_FLAGS)


def run_leg(idx, h, run_dir):
    """One contact-regime WSD leg (trunk + leaf).
    Returns (score or None, train_wall_s)."""
    shutil.rmtree(run_dir, ignore_errors=True)  # leg = always fresh model
    # 100k steps/s floor (contact measures ~148k at 16384/8) + startup
    cmds = train_cmds(h, run_dir)
    ck, wall = run_wsd_train(idx, h, run_dir, NUM_ENVS, HORIZON, cmds,
                             100_000)
    if ck is None:
        return None, wall
    score = final_return([os.path.join(run_dir, "metrics.jsonl"),
                          os.path.join(cmds[2], "metrics.jsonl")])
    if score is None:
        print(f"[run {idx:03d}] no `return` in metrics.jsonl", flush=True)
    return score, wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-runs", type=int, default=100)
    p.add_argument("--dir", type=str, default="sweeps/s4_contact_ret_wsd")
    args = p.parse_args()
    norm = os.path.normpath(args.dir).replace("\\", "/")
    if norm.endswith(("sweeps/s2_ranked_upp", "sweeps/s3_cutplane_upp",
                      "sweeps/s1_upp", "sweeps/s4_contact_ret")):
        sys.exit("s1-s4 are FROZEN observations of older objectives "
                 "(weld physics; s4 = ep_return(last100) on the pre-WSD "
                 "env). Use the default or a fresh dir.")
    os.makedirs(args.dir, exist_ok=True)
    log_path = os.path.join(args.dir, "observations.jsonl")

    from protein_sweep import Protein   # gpytorch: sweep host only
    protein = Protein(SPACE, cost_param="train/total_steps")

    start = replay(protein, log_path)
    if start:
        print(f"resumed: replayed {start} observations", flush=True)

    for idx in range(start, args.max_runs):
        hypers, info = protein.suggest(deepcopy(FILL))
        h = hypers["train"]
        # phase 2 (2026-08-10, after legs 000-051): discrimination floor.
        # The 60-130M regime is mapped — a flat -64.5..-67 stand-and-miss
        # plateau where the objective barely separates configs, so the
        # cost-aware GP rationally kept buying cheap legs forever.
        # Note-hitting (where gamma/lambda/lr actually differentiate)
        # starts ~300M+ (v12 1h: first pp by 262M, upp 123 at 524M).
        # Clamp suggested budgets up; the clamped value is what gets
        # observed, so the GP fits the leg that actually ran.
        h["total_steps"] = max(h["total_steps"], 3e8)
        run_dir = os.path.join(args.dir, f"run_{idx:03d}")
        print(f"[run {idx:03d}] lr {h['lr']:.2e} ent {h['ent_coef']:.2e}"
              f"->{h['ent_final']:.2e} gamma {h['gamma']:.4f} "
              f"lam {h['gae_lambda']:.4f} "
              f"steps {h['total_steps']/1e6:.0f}M", flush=True)
        score, cost = run_leg(idx, h, run_dir)
        fail = score is None
        if fail:
            score = FAIL_SCORE
        protein.observe(hypers, score, cost, fail)
        log_observation(log_path, idx, hypers, score, cost, fail)
        print(f"[run {idx:03d}] -> return {score:,.3f} "
              f"({cost/60:.1f} min){' FAILED' if fail else ''}", flush=True)

    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
