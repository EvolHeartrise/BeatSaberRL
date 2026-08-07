"""Harvest a finished training run into the s2 sweep observations.

Any seed-0 ranked-maps run that matches the sweep-leg protocol can
serve as GP data for future sweep_ranked.py legs (2026-07-29 user
request: don't lose protocol-valid runs). This tool VALIDATES instead
of trusting: it diffs the run's resolved args.json against a
reference sweep leg (run_085), allowing only the swept dimensions
(+ run_dir/ckpt_every, which don't affect the training trajectory)
to differ — so a warm-start, a different seed (seed also redraws the
segment pool!), a different pool size, etc. are refused with the
offending keys listed.

Score = final.pt's catalog upp (sweep convention: FINAL only, never
best-checkpoint — mixing best-of-run scores into a final-scored
dataset biases the GP optimistic on long budgets). Cost = train wall,
estimated as mtime(final.pt) - mtime(args.json) (both written at the
run's ends; matches the sweep's subprocess-wall definition) unless
--cost overrides. Duplicate hypers already in the log are skipped
(--force to append anyway).

  python harvest_obs.py --run-dir runs/<run> [--dry-run] [--cost S]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

from sweep_ranked import SPACE

OBS = os.path.join("sweeps", "s2_ranked_upp", "observations.jsonl")
REF = os.path.join("sweeps", "s2_ranked_upp", "run_085", "args.json")
SWEPT = list(SPACE["train"].keys())          # the 7 GP dimensions
IGNORE = SWEPT + ["run_dir", "ckpt_every"]   # allowed to differ
# trainer args added AFTER the reference leg: a run still matches the
# protocol iff the new key holds the value that reproduces the
# reference-era behavior (e.g. wide_acc=True is a DIFFERENT objective —
# refused; wide_acc=False is the old scoring — accepted)
NEW_KEY_DEFAULTS = {"wide_acc": False}


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cost", type=float, default=None,
                   help="train wall seconds (default: file-mtime estimate)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="append even if identical hypers already logged")
    args = p.parse_args()

    ck = os.path.join(args.run_dir, "final.pt")
    aj = os.path.join(args.run_dir, "args.json")
    if not os.path.exists(ck):
        sys.exit(f"REFUSED: {ck} missing — run not finished")
    run = json.load(open(aj, encoding="utf-8"))["resolved"]
    ref = json.load(open(REF, encoding="utf-8"))["resolved"]

    bad = []
    for k in sorted(set(run) | set(ref)):
        if k in IGNORE:
            continue
        ref_v = ref[k] if k in ref else NEW_KEY_DEFAULTS.get(k)
        if run.get(k) != ref_v:
            bad.append(f"  {k}: run={run.get(k)!r} protocol={ref_v!r}")
    if bad:
        sys.exit("REFUSED: run deviates from the sweep protocol on non-"
                 "swept keys:\n" + "\n".join(bad))

    hypers = {"train": {k: float(run[k]) for k in SWEPT}}
    for k, spec in SPACE["train"].items():
        v = hypers["train"][k]
        if not (spec["min"] <= v <= spec["max"]):
            print(f"WARNING: {k}={v:g} outside sweep bounds "
                  f"[{spec['min']:g}, {spec['max']:g}] — the GP will see "
                  "an out-of-box observation", flush=True)

    lines = [json.loads(l) for l in open(OBS, encoding="utf-8")]
    for o in lines:
        if all(close(o["hypers"]["train"][k], hypers["train"][k])
               for k in SWEPT):
            msg = (f"duplicate of idx {o['idx']} "
                   f"(score {o['score']:,.0f})")
            if not args.force:
                sys.exit(f"SKIPPED: {msg} — use --force to append anyway")
            print(f"WARNING: appending {msg}", flush=True)

    bench_path = os.path.join(args.run_dir, "bench.json")
    if not os.path.exists(bench_path):
        if args.dry_run:
            print("[dry-run] would bench final.pt (no bench.json yet)",
                  flush=True)
            score = None
        else:
            # 7200 s: the v10 catalog bench runs ~1,900-2,400 s
            # (2026-08-07), ~3x margin; 3000 was a pre-v10 constant
            try:
                r = subprocess.run(
                    [sys.executable, "bench_scoresaber.py", "--ckpt", ck,
                     "--out", bench_path,
                     "--num-envs", "2048", "--fused-forward"],
                    capture_output=True, text=True, timeout=7200)
            except subprocess.TimeoutExpired:
                sys.exit("BENCH TIMEOUT (7200s)")
            if r.returncode != 0 or not os.path.exists(bench_path):
                sys.exit(f"BENCH FAILED: {(r.stderr or '')[-300:]}")
    if os.path.exists(bench_path):
        b = json.load(open(bench_path, encoding="utf-8"))
        if b["checkpoint_name"] != "final.pt":
            sys.exit(f"REFUSED: bench.json is for {b['checkpoint_name']}, "
                     "not final.pt (sweep scores FINAL only)")
        # s2 is an OLD-RULES dataset: every observation was scored under
        # swept-surface accuracy (scoring v1). The 2026-07-29 cut-plane
        # fix changed both the bench scale AND the training reward, so
        # post-fix runs are a different objective with incomparable
        # scores — they need a fresh sweep namespace (s3), not s2.
        if b.get("scoring_version", 1) != 1:
            sys.exit(f"REFUSED: {OBS} is a scoring-v1 (pre-cut-plane) "
                     f"dataset but this bench is scoring-v"
                     f"{b['scoring_version']}. s2 is frozen; start a new "
                     f"sweep namespace for cut-plane-rule observations.")
        score = b["total_unweighted_pp"]

    cost = args.cost
    if cost is None:
        cost = round(os.path.getmtime(ck) - os.path.getmtime(aj), 1)
        if cost <= 0:
            sys.exit("REFUSED: mtime cost estimate non-positive — pass "
                     "--cost explicitly")

    obs = {"idx": len(lines), "hypers": hypers, "score": score,
           "cost": cost, "is_failure": False}
    if args.dry_run:
        print(f"[dry-run] would append: {json.dumps(obs)}", flush=True)
        return
    with open(OBS, "a", encoding="utf-8") as f:
        f.write(json.dumps(obs) + "\n")
    print(f"appended idx {obs['idx']}: score {score:,.0f}, "
          f"cost {cost:.0f}s -> {OBS}", flush=True)


if __name__ == "__main__":
    main()
