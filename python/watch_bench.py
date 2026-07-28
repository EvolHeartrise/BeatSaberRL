"""Checkpoint-bench watcher: polls a training run dir and benches every
new ckpt_upd*.pt (and final.pt) through bench_scoresaber as it lands.

    python watch_bench.py --run-dir runs/gen5b_5b --prefix gen5b_5b_

Exits after final.pt has been benched (training complete). Skips files
younger than --min-age (a checkpoint mid-torch.save); a failed bench is
retried on the next poll. Prints BOTH pp totals per result (user rule:
weighted AND unweighted in every report).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--prefix", required=True,
                    help="bench/<prefix><ckpt-stem>.json")
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--min-age", type=float, default=30.0)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "bench"), exist_ok=True)

    done = set()
    while True:
        cks = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_upd*.pt")))
        fin = os.path.join(args.run_dir, "final.pt")
        if os.path.exists(fin):
            cks.append(fin)
        for ck in cks:
            stem = os.path.splitext(os.path.basename(ck))[0]
            out = os.path.join(here, "bench", f"{args.prefix}{stem}.json")
            if out in done or os.path.exists(out):
                done.add(out)
                continue
            if time.time() - os.path.getmtime(ck) < args.min_age:
                continue                      # possibly mid-write
            print(f"[watch] benching {stem}", flush=True)
            r = subprocess.run(
                [sys.executable, os.path.join(here, "bench_scoresaber.py"),
                 "--ckpt", ck, "--out", out,
                 "--num-envs", str(args.num_envs), "--fused-forward"],
                capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(out):
                print(f"[watch] bench FAILED for {stem} (retry next poll)"
                      f"\n{r.stdout[-800:]}\n{r.stderr[-800:]}", flush=True)
                continue
            with open(out, encoding="utf-8") as f:
                d = json.load(f)
            print(f"[watch] {stem}: unweighted pp "
                  f"{d['total_unweighted_pp']:,.0f} | weighted pp "
                  f"{d['total_weighted_pp']:,.2f} | levels "
                  f"{d['n_evaluated']}", flush=True)
            done.add(out)
        if os.path.join(args.run_dir, "final.pt") in cks and \
                all(os.path.exists(os.path.join(
                    here, "bench",
                    f"{args.prefix}{os.path.splitext(os.path.basename(c))[0]}"
                    f".json")) for c in cks):
            print("[watch] final.pt benched — watcher done", flush=True)
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
