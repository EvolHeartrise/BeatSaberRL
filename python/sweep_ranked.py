"""Protein sweep s2: train on RANKED-MAP SEGMENTS (2026-07-29).

Independent successor to sweep_protein.py (s1) — that file and
sweeps/s1_upp are untouched; this driver owns sweeps/s2_ranked_upp
with its own observations.jsonl. Motivation: gen5b's ~87% accuracy
ceiling holds at EVERY star range and its final-leg slump showed
procedural-chart overfit — s2 removes the procedural->real gap by
training directly on 25 s segments of the full 6101-level catalog
(beat_arena_env/ranked_pool.py; bench parsing protocol, so training
and evaluation see identical note streams).

Space = s1 minus bomb_prob (real maps carry their own bombs/density),
same objective: TOTAL UNWEIGHTED PP over the full catalog. Bounds
started as s1's widened box; widened AGAIN 2026-07-29 after legs
010-019 (gamma floor 0.87 -> 0.80, gae_lambda ceiling 0.99 -> 0.995):
the GP pinned gamma at the old floor with BOTH front-runners sitting
on it (run 017 692k, run 019 738k) — the exact s1 run-018 pattern.
Widened a SECOND time 2026-07-29 (gamma floor 0.80 -> 0.72) after run
045 set the sweep best (969k, 128M) EXACTLY on the 0.80 floor with
048 (957k) beside it. gamma 0.72 at 30 Hz is a ~3-step credit
half-life — approaching the physical limit where credit cannot span
note approach time, so a floor THERE would be a finding, not a bounds
artifact. Logged observations replay fine into the wider box. NOTE: with no holdout (user decision
2026-07-29), the bench measures performance on the training
distribution — cross-check any winner on procedural charts / old
gens before claiming generalization.

**s3 (2026-07-29): fresh sweep under the CUT-PLANE accuracy rules**
(scoring v2 — the user-caught 2026-07-29 fix that unlocked the acc
component). The fix changed both
the bench scale and the training reward, so s2's observations are a
different objective: sweeps/s2_ranked_upp is FROZEN (103 v1-scored
observations, never extend), the default --dir now points at
sweeps/s3_cutplane_upp, and this driver refuses the s2 dir outright.
Space/bounds/protocol unchanged from s2 — its widened box is the
prior; the GP restarts from Sobol on the new objective. First sweep
where policies train with reachable accuracy gradient.

**WSD port (2026-09-30)**: see sweep_protein.py's WSD PORT note — a
leg is a WSD trunk + leaf (not `--lr-decay cosine`, removed), the leaf
checkpoint is benched without the retired `--fused-forward`, failures
cost their real wall and a sub-minute crash aborts the sweep. Env
flags: `--ranked-maps` (now the trainer default, kept explicit) with
the recipe's other defaults (mirror augment, ep-stats, ...) — the env
is the current one, not the 2026-07-29 cut-plane env, so s3 is FROZEN
too and the default --dir is sweeps/s3_ranked_upp_wsd.

  python sweep_ranked.py --max-runs 100 [--dir sweeps/s3_ranked_upp_wsd]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy

from sweep_protein import (PY, log_observation, replay, run_step,
                           run_wsd_train, wsd_cmds)

NUM_ENVS, HORIZON = 1024, 128
# SEG_LEN retired 2026-09-16: whole-level episodes are the only pool mode
SEG_POOL = 24576   # seeded segment subsample per leg (~0.6 GB VRAM;
#                    identical across legs -> the pool npz caches once)
ENV_FLAGS = ["--ranked-maps", "--chart-pool", str(SEG_POOL)]
SPACE = {
    "metric": "catalog_unweighted_pp",
    "goal": "maximize",
    "metric_distribution": "linear",
    "downsample": 1,
    "early_stop_quantile": 0.3,
    "use_gpu": False,
    # seconds; ~4e8 steps + startup. Re-derived 2026-08-07 for the v10
    # env: 12,000 steps/s production floor (measured 12,423 at N=1024/
    # T=128) -> 4e8 / 12e3 = 33,300 s + startup ~= 35,000. The old 2,200
    # came from the pre-v10 ~250k steps/s env.
    "max_suggestion_cost": 35000,
    "train": {
        "lr":          dict(distribution="log_normal", min=1e-4, max=8e-3, scale=0.5),
        "ent_coef":    dict(distribution="log_normal", min=1e-4, max=0.3, scale=0.5),
        "ent_final":   dict(distribution="log_normal", min=1e-5, max=0.1, scale=0.5),
        "gamma":       dict(distribution="logit_normal", min=0.72, max=0.997, scale=0.5),
        "gae_lambda":  dict(distribution="logit_normal", min=0.65, max=0.995, scale=0.5),
        # s3 widening #1 (2026-07-29, after legs 000-048): effort ceiling
        # 8.0 -> 32.0 — three of the top five (incl. best run 048 at
        # 1,305,759) sat EXACTLY on the 8.0 bound, and the in-flight 049
        # suggestion was there too. Under cut-plane scoring the effort
        # tax flipped: centering adjustments pay for themselves (s2's
        # optimum was ~0.2-0.7 with 5+ catastrophic). s1/s2 protocol:
        # stop, widen, relaunch (observations replay into the wider box).
        # s3 widening #2 (2026-07-29, legs 049-075): 32.0 -> 128.0 — the
        # new best 075 (1,648,925, first leg past the 400M reference)
        # sat EXACTLY on 32.0 with two more of the top six, rest at
        # 20-30. Economy-of-motion is a live regularizer under
        # cut-plane scoring; keep widening until the optimum is
        # interior. At 128 a full-motion tick costs ~4x a note's
        # reward, so a floor above that would be physics, not bounds.
        # (effort_coef itself left the space when the v10 effort price
        # was retired — 737931c; the notes above are its history.)
        "total_steps": dict(distribution="log_normal", min=5e7, max=4e8, scale="time"),
    },
}
FILL = {"train": {k: None for k in SPACE["train"]}}


def train_cmds(h, run_dir):
    """(trunk_cmd, leaf_cmd, leaf_dir) for one WSD leg."""
    return wsd_cmds(h, run_dir, NUM_ENVS, HORIZON, ENV_FLAGS)


def bench_cmd(ck, out):
    return [PY, "bench_scoresaber.py", "--ckpt", ck, "--out", out,
            "--num-envs", "2048"]


def run_leg(idx, h, run_dir):
    """One ranked-maps WSD leg (trunk + leaf) + full-catalog bench of the
    leaf. Returns (catalog_upp or None, train_wall_s)."""
    shutil.rmtree(run_dir, ignore_errors=True)  # leg = always fresh model
    # conservative 8,000 steps/s floor (v10 measures 12.4k at 1024/128,
    # 24.4k at 4096/32 — 2026-08-07) plus 1800 s startup per step
    ck, wall = run_wsd_train(idx, h, run_dir, NUM_ENVS, HORIZON,
                             train_cmds(h, run_dir), 8000)
    if ck is None:
        return None, wall
    out = os.path.join(run_dir, "bench.json")
    # 7200 s: the v10 catalog bench runs ~1,900-2,400 s (2026-08-07),
    # so this is ~3x margin. The old 3000 s was a pre-v10 constant.
    ok, _ = run_step(idx, "BENCH", bench_cmd(ck, out),
                     os.path.join(run_dir, "bench.log"), 7200)
    if not ok or not os.path.exists(out):
        return None, wall
    upp = json.load(open(out))["total_unweighted_pp"]
    return upp, wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-runs", type=int, default=100)
    p.add_argument("--dir", type=str, default="sweeps/s3_ranked_upp_wsd")
    args = p.parse_args()
    if os.path.normpath(args.dir).replace("\\", "/").endswith(
            ("sweeps/s2_ranked_upp", "sweeps/s3_cutplane_upp")):
        sys.exit("sweeps/s2_ranked_upp and s3_cutplane_upp are FROZEN: "
                 "their observations are older objectives (scoring v1; "
                 "the pre-contact, pre-WSD env) and must not mix with "
                 "current legs. Use the default or a fresh dir.")
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
        run_dir = os.path.join(args.dir, f"run_{idx:03d}")
        print(f"[run {idx:03d}] lr {h['lr']:.2e} ent {h['ent_coef']:.2e}"
              f"->{h['ent_final']:.2e} gamma {h['gamma']:.4f} "
              f"lam {h['gae_lambda']:.4f} "
              f"steps {h['total_steps']/1e6:.0f}M", flush=True)
        score, cost = run_leg(idx, h, run_dir)
        fail = score is None
        if fail:
            score = 0.0   # Protein refits failures at the min score
        protein.observe(hypers, score, cost, fail)
        log_observation(log_path, idx, hypers, score, cost, fail)
        print(f"[run {idx:03d}] -> catalog upp {score:,.0f} "
              f"({cost/60:.1f} min){' FAILED' if fail else ''}", flush=True)

    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
