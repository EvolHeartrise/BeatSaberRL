"""Protein hyperparameter sweep runner for train_batch.py (2026-07-27).

Sweep 1 ("entropy/effort/gamma"): the flailing + bomb-economics + horizon
cluster, from scratch, judged on TOTAL UNWEIGHTED PP over the FULL
ScoreSaber catalog (user decision 2026-07-27 — smoother low-end
gradient than band wpp and measures breadth; costs ~3-4 min/bench warm
vs ~40 s for the band). Cost axis = total_steps (Protein's multi-budget
scaling-law trick); cost signal = wall seconds of the training leg.
downsample=1 (one bench per leg) which also enables Protein's
success-probability classifier.

Space (7 dims; gae_lambda added 2026-07-27 — gamma and lambda jointly
set the credit window; effort_coef left the space with the v10 effort
price's retirement, see WSD PORT below):
  lr          log     1e-4 .. 8e-3   trunk lr; the leaf decays it to 1e-4
  ent_coef    log     1e-4 .. 0.3    initial entropy coef
  ent_final   log     1e-5 .. 0.1    trunk anneal floor + leaf target
  gamma       logit   0.87 .. 0.997  user hypothesis: ~1 s horizon (~0.967)
  gae_lambda  logit   0.65 .. 0.99   trace decay (gamma*lambda = PG window)
  bomb_prob   log     0.003 .. 0.4   bomb density (72%-of-reward question)
  total_steps log     5e7  .. 4e8    THE COST AXIS (1.2-9.3 h at the
                                     v10 env's ~12k steps/s, measured
                                     2026-08-07; it was 3.3-27 min at
                                     the pre-v10 ~250k/s)

Bounds widened 2026-07-27 after runs 010-020: the GP pinned lr and
ent_coef at their ceilings and gae_lambda/effort/bomb at their floors
WITH the winners sitting there (run 018, 778k upp: lam exactly 0.80,
lr 3e-3, ent 0.1); gamma got margin too (best runs 0.908-0.918, just
above the old 0.90 floor). Observations replay fine — raw values
re-normalize into the wider box.

State: every observation appended to <dir>/observations.jsonl and
replayed through observe() on restart (observe() is pure, so this fully
reconstructs the GP state — Protein itself has no save/load).

**WSD PORT (2026-09-30).** All three drivers died on argparse in
seconds against the current trainer/bench and were logged as failures
at cost 1.0 s. What changed, per the CLIs as of this date:
  - `--lr-decay cosine` is gone (WSD is the only schedule since
    2026-08-11). A leg is now a WSD TRUNK (warmup -> constant --lr,
    exponential ent anneal ent_coef -> floor ent_final, --ent-tau
    recipe default) for (1 - COOLDOWN_FRAC) of total_steps, then a
    LEAF (`--resume trunk_final.pt --cooldown-updates N`: lr 1-sqrt to
    --lr-final 1e-4, ent linear to ent_final, `--no-wall-despawn` as
    the README leaf launch) for the rest. total_steps and the cost
    (trunk + leaf wall) keep their meaning; the leaf checkpoint is
    what gets benched. The trainer writes trunk_final.pt /
    leaf_from_updNNNNN.pt — `final.pt` no longer exists, so even a
    working leg would have been scored a failure.
  - `--ranked-maps` is the trainer default now; this s1 driver is the
    PROCEDURAL sweep (bomb_prob is ignored on real maps), so it passes
    `--no-ranked-maps --no-ep-stats --chart-pool 65536`.
  - bench: `--fused-forward` is gone (the fused forward is the only
    path); the bench keeps --num-envs 2048 explicit.
  - failures are honest: cost = the real wall seconds spent, and a
    step that dies in < FAST_FAIL_S is a config/CLI error — the sweep
    ABORTS with the log tail instead of observing it.
  - observations carry `"era": ERA`; replay refuses pre-port lines
    (a different trainer, schedule and env — the s1-s4 dirs are frozen).

  python sweep_protein.py --max-runs 100 [--dir sweeps/s1_upp_wsd]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy

PY = sys.executable
# ---- shared leg plumbing (sweep_ranked.py / sweep_contact.py import it)
ERA = "wsd-2026-09-30"   # observations.jsonl tag; replay refuses others
FAST_FAIL_S = 60.0       # non-zero exit this fast = config/CLI error
COOLDOWN_FRAC = 0.2      # leaf share of a leg's updates (EPFL
#                          2405.18392: 10-20% of trunk; recipe cd300)
LR_FINAL = 1e-4          # the old cosine floor, now the leaf's floor


def split_budget(total_steps, num_envs, horizon):
    """(trunk_steps, cooldown_updates) for a leg of total_steps."""
    batch = num_envs * horizon
    n_upd = max(2, int(total_steps) // batch)
    cd = max(1, round(COOLDOWN_FRAC * n_upd))
    return (n_upd - cd) * batch, cd


def wsd_cmds(h, run_dir, num_envs, horizon, env_flags):
    """(trunk_cmd, leaf_cmd, leaf_dir) for one WSD leg. A --resume
    takes only lr/ent/schedule/z/arch from the checkpoint, so the leaf
    repeats every env/shape flag of the trunk."""
    trunk_steps, cd = split_budget(h["total_steps"], num_envs, horizon)
    common = [PY, "train_batch.py",
              "--num-envs", str(num_envs), "--horizon", str(horizon),
              "--seed", "0", *env_flags,
              "--gamma", f"{h['gamma']:.6g}",
              "--gae-lambda", f"{h['gae_lambda']:.6g}",
              "--ent-final", f"{h['ent_final']:.6g}"]
    trunk = common + ["--lr", f"{h['lr']:.6g}",
                      "--lr-final", f"{LR_FINAL:g}",
                      "--ent-coef", f"{h['ent_coef']:.6g}",
                      "--total-steps", str(trunk_steps),
                      "--run-dir", run_dir]
    leaf_dir = os.path.join(run_dir, "leaf")
    leaf = common + ["--resume", os.path.join(run_dir, "trunk_final.pt"),
                     "--cooldown-updates", str(cd),
                     "--no-wall-despawn",   # walls solid, as in the bench
                     "--run-dir", leaf_dir]
    return trunk, leaf, leaf_dir


def run_step(idx, what, cmd, log_path, timeout):
    """Run one subprocess with stdout+stderr to log_path. Returns
    (ok, wall_s). A non-zero exit inside FAST_FAIL_S is a CLI/config
    error, not a result: abort the sweep with the log tail rather than
    teach the optimizer that the config is a (free) failure."""
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            wall = time.time() - t0
            print(f"[run {idx:03d}] {what} TIMEOUT ({timeout}s)", flush=True)
            return False, wall
    wall = time.time() - t0
    if r.returncode == 0:
        return True, wall
    with open(log_path, encoding="utf-8", errors="replace") as f:
        tail = f.read()[-1500:]
    if wall < FAST_FAIL_S:
        sys.exit(f"[run {idx:03d}] {what} exited {r.returncode} after "
                 f"{wall:.1f}s (< {FAST_FAIL_S:.0f}s: a CLI/config error, "
                 f"not a result) — sweep ABORTED, nothing observed.\n"
                 f"cmd: {' '.join(cmd)}\n--- {log_path} tail ---\n{tail}")
    print(f"[run {idx:03d}] {what} FAILED rc {r.returncode} ({wall:.0f}s): "
          f"{tail[-400:]}", flush=True)
    return False, wall


def run_wsd_train(idx, h, run_dir, num_envs, horizon, cmds,
                  steps_per_s):
    """Trunk then leaf (cmds = a driver's train_cmds(h, run_dir)).
    Returns (leaf_ckpt or None, train_wall_s); the wall is the real
    time spent even when a step failed."""
    trunk, leaf, leaf_dir = cmds
    trunk_steps, cd = split_budget(h["total_steps"], num_envs, horizon)
    os.makedirs(run_dir, exist_ok=True)
    ok, wall = run_step(idx, "TRUNK", trunk,
                        os.path.join(run_dir, "train.log"),
                        trunk_steps // steps_per_s + 1800)
    if not ok:
        return None, wall
    if not os.path.exists(os.path.join(run_dir, "trunk_final.pt")):
        print(f"[run {idx:03d}] TRUNK exited 0 without trunk_final.pt",
              flush=True)
        return None, wall
    os.makedirs(leaf_dir, exist_ok=True)
    ok, w = run_step(idx, "LEAF", leaf,
                     os.path.join(leaf_dir, "leaf_train.log"),
                     cd * num_envs * horizon // steps_per_s + 1800)
    wall += w
    cks = sorted(glob.glob(os.path.join(leaf_dir, "leaf_from_upd*.pt")))
    if not ok or not cks:
        if ok:
            print(f"[run {idx:03d}] LEAF exited 0 without a leaf ckpt",
                  flush=True)
        return None, wall
    return cks[-1], wall


def replay(protein, log_path):
    """Replay observations.jsonl into Protein; refuses pre-port lines.
    Returns the number replayed."""
    n = 0
    if not os.path.exists(log_path):
        return 0
    for line in open(log_path, encoding="utf-8"):
        o = json.loads(line)
        if o.get("era") != ERA:
            sys.exit(f"{log_path}: observation idx {o.get('idx')} is era "
                     f"{o.get('era')!r}, this driver writes {ERA!r} — a "
                     f"different trainer/schedule/env. Use a fresh --dir.")
        protein.observe(o["hypers"], o["score"], o["cost"],
                        o["is_failure"])
        protein.suggestion_idx += 1  # keep Sobol phase consumed
        n += 1
    return n


def log_observation(log_path, idx, hypers, score, cost, fail):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"idx": idx, "era": ERA, "hypers": hypers,
                            "score": score, "cost": cost,
                            "is_failure": fail}) + "\n")


# ---- s1 (this driver)
NUM_ENVS, HORIZON = 1024, 128
# procedural charts: bomb_prob is a generator dial, ignored on real maps
ENV_FLAGS = ["--no-ranked-maps", "--no-ep-stats", "--chart-pool", "65536"]
SPACE = {
    "metric": "catalog_unweighted_pp",
    "goal": "maximize",
    "metric_distribution": "linear",   # wpp is unbounded
    "downsample": 1,                   # one bench per leg
    "early_stop_quantile": 0.3,
    "use_gpu": False,                  # GP on CPU; the 1080 belongs to training
    # seconds; ~4e8 steps + startup. Re-derived 2026-08-07 for the v10
    # env: 12,000 steps/s production floor (measured 12,423 at N=1024/
    # T=128) -> 4e8 / 12e3 = 33,300 s + startup ~= 35,000. The old 2,200
    # came from the pre-v10 ~250k steps/s env.
    "max_suggestion_cost": 35000,
    "train": {
        "lr":          dict(distribution="log_normal", min=1e-4, max=8e-3, scale=0.5),
        "ent_coef":    dict(distribution="log_normal", min=1e-4, max=0.3, scale=0.5),
        "ent_final":   dict(distribution="log_normal", min=1e-5, max=0.1, scale=0.5),
        "gamma":       dict(distribution="logit_normal", min=0.87, max=0.997, scale=0.5),
        "gae_lambda":  dict(distribution="logit_normal", min=0.65, max=0.99, scale=0.5),
        "bomb_prob":   dict(distribution="log_normal", min=0.003, max=0.4, scale=0.5),
        "total_steps": dict(distribution="log_normal", min=5e7, max=4e8, scale="time"),
    },
}
FILL = {"train": {k: None for k in SPACE["train"]}}


def train_cmds(h, run_dir):
    """(trunk_cmd, leaf_cmd, leaf_dir); bomb_prob rides on both."""
    flags = ENV_FLAGS + ["--bomb-prob", f"{h['bomb_prob']:.6g}"]
    return wsd_cmds(h, run_dir, NUM_ENVS, HORIZON, flags)


def bench_cmd(ck, out):
    # explicit width (2026-07-27): don't let a future default change
    # silently slow or re-class the sweep. --fused-forward retired: the
    # fused forward is the bench's only path now.
    return [PY, "bench_scoresaber.py", "--ckpt", ck, "--out", out,
            "--num-envs", "2048"]


def run_leg(idx, h, run_dir):
    """One WSD leg (trunk + leaf) + catalog bench of the leaf.
    Returns (catalog_upp or None, train_wall_s)."""
    # a leg is always a fresh model — a stale dir (e.g. from a run that
    # was in-flight at a pause) would make the bench refuse the
    # different-model overwrite and record a spurious failure
    shutil.rmtree(run_dir, ignore_errors=True)
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
    p.add_argument("--dir", type=str, default="sweeps/s1_upp_wsd")
    args = p.parse_args()
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
              f"bomb {h['bomb_prob']:.3f} "
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
