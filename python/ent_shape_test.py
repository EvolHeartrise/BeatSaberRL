"""Entropy-schedule SHAPE test (2026-07-29).

Question from the rising-entropy discussion (sessions/
2026-07-29-rising-entropy-discussion.md): did sweep s2's winners win
because entropy ROSE, or because it stayed in a moderate band and
never annealed to ~0? Three shapes on the exact run-085 recipe, with
the same time-AVERAGED entropy coefficient (the trainer interpolates
linearly, so the flat control is the arithmetic mean):

  rising : 0.013929  -> 0.0331038   (= run 085 verbatim)
  flat   : 0.0235164 -> 0.0235164   (arithmetic mean of the two)
  falling: 0.0331038 -> 0.013929    (mirror)

Two seeds per shape. rise_s0 is NOT re-run: training is deterministic
given code+seed, so run 085's bench (upp 1,218,348.82) IS that arm.
NOTE: --seed also seeds the 24,576-segment pool subsample
(ranked_pool.load_ranked_pool), so seed 1 arms train on a different
pool draw — shape comparisons are paired WITHIN a seed; the two seeds
are full replicate triplets, not pure init-noise replicates.

Registered predictions (Claude, before launch): ~60% flat matches
rising within seed noise; falling worst.

  python ent_shape_test.py            # ~1 h: 5 legs x (~10.6 min train + bench)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

PY = sys.executable
OUT_DIR = os.path.join("sweeps", "ent_shape")
TOTAL_STEPS = 176_508_300   # run 085's budget, verbatim

# run 085 hypers exactly as the sweep driver passed them (%.6g)
BASE = ["--num-envs", "1024", "--horizon", "128",
        "--lr-decay", "cosine",
        "--ranked-maps", "--seg-len", "25",
        "--chart-pool", "24576",
        "--lr", "0.00296092",
        "--gamma", "0.914958",
        "--gae-lambda", "0.980024",
        "--effort-coef", "0.240844",
        "--total-steps", str(TOTAL_STEPS)]

ENT = {"rise": ("0.013929", "0.0331038"),
       "flat": ("0.0235164", "0.0235164"),
       "fall": ("0.0331038", "0.013929")}

# seed-0 rising = sweep run 085 (deterministic trainer, same code path)
KNOWN = {"rise_s0": dict(upp=1218348.82, wpp=9947.20, fcs=1496,
                         source="sweeps/s2_ranked_upp/run_085")}

ARMS = [("fall", 0), ("flat", 0), ("rise", 1), ("flat", 1), ("fall", 1)]


def summarize(bench_path):
    b = json.load(open(bench_path, encoding="utf-8"))
    fcs = sum(1 for lv in b["levels"] if lv["full_combo"])
    return dict(upp=b["total_unweighted_pp"], wpp=b["total_weighted_pp"],
                fcs=fcs)


def run_leg(name, shape, seed):
    run_dir = os.path.join(OUT_DIR, name)
    shutil.rmtree(run_dir, ignore_errors=True)  # leg = always fresh model
    e0, e1 = ENT[shape]
    cmd = [PY, "train_batch.py", *BASE, "--seed", str(seed),
           "--ent-coef", e0, "--ent-final", e1, "--run-dir", run_dir]
    t0 = time.time()
    # total_steps at a conservative 8,000 steps/s floor (v10 measures
    # 12.4k at 1024/128, 24.4k at 4096/32 — 2026-08-07) + 1800 s startup
    tmo = TOTAL_STEPS // 8000 + 1800
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=tmo)
    except subprocess.TimeoutExpired:
        print(f"[{name}] TRAIN TIMEOUT ({tmo}s)", flush=True)
        return None
    wall = time.time() - t0
    ck = os.path.join(run_dir, "final.pt")
    if r.returncode != 0 or not os.path.exists(ck):
        tail = (r.stdout or "")[-400:] + (r.stderr or "")[-400:]
        print(f"[{name}] TRAIN FAILED ({wall:.0f}s): {tail}", flush=True)
        return None
    out = os.path.join(run_dir, "bench.json")
    # 7200 s: the v10 catalog bench runs ~1,900-2,400 s (2026-08-07),
    # so this is ~3x margin. The old 3000 s was a pre-v10 constant.
    try:
        b = subprocess.run([PY, "bench_scoresaber.py", "--ckpt", ck,
                            "--out", out,
                            "--num-envs", "2048", "--fused-forward"],
                           capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        print(f"[{name}] BENCH TIMEOUT (7200s)", flush=True)
        return None
    if b.returncode != 0 or not os.path.exists(out):
        print(f"[{name}] BENCH FAILED: {(b.stderr or '')[-300:]}",
              flush=True)
        return None
    return dict(**summarize(out), train_wall_s=round(wall, 1))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res_path = os.path.join(OUT_DIR, "results.json")
    results = dict(KNOWN)
    if os.path.exists(res_path):
        results.update(json.load(open(res_path, encoding="utf-8")))

    for shape, seed in ARMS:
        name = f"{shape}_s{seed}"
        if name in results:
            print(f"[{name}] already done, skipping", flush=True)
            continue
        e0, e1 = ENT[shape]
        print(f"[{name}] ent {e0} -> {e1}, seed {seed}", flush=True)
        r = run_leg(name, shape, seed)
        if r is None:
            r = dict(failed=True)
        results[name] = r
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        if "upp" in r:
            print(f"[{name}] -> upp {r['upp']:,.0f} / wpp {r['wpp']:,.2f}"
                  f" / FCs {r['fcs']} ({r['train_wall_s']/60:.1f} min)",
                  flush=True)

    print("\n=== ENT SHAPE TEST SUMMARY (both pp totals) ===", flush=True)
    for name in ["rise_s0", "flat_s0", "fall_s0",
                 "rise_s1", "flat_s1", "fall_s1"]:
        r = results.get(name)
        if r is None or "upp" not in r:
            print(f"{name}: FAILED/missing", flush=True)
        else:
            print(f"{name}: upp {r['upp']:>12,.0f}  wpp {r['wpp']:>9,.2f}"
                  f"  FCs {r['fcs']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
