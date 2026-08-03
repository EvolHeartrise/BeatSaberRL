"""400M wide-acc twin of confirm_fall_400m (2026-07-29).

Identical recipe and falling entropy schedule, but trained under the
wide-acc reward (30*clamp(1 - d/0.6), unrounded — train_batch default
since e391d34). Head-to-head vs confirm_fall_400m answers whether the
dead-zone fix moves the accuracy component (registered predictions in
sessions/2026-07-29-wide-acc-reward.md: 75% procedural-eval acc >=
2/15, 60% wpp record falls, risk = early FC cost).

NOT harvested into sweep s2 (different objective; harvest_obs refuses
wide runs). Bench is unchanged real scoring, so every number below is
directly comparable to confirm_fall_400m's trajectory table. Interval
benches run AFTER training. Ends with the 30-ep procedural eval of
final.pt (real scoring) for the acc-component prediction.

  python wideacc_fall_400m.py     # ~26 min train + ~10 min benches
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "wideacc_fall_400m")
TOTAL_STEPS = 400_000_000
CKPT_EVERY = 381  # updates; 381*131072 = 49.9M steps


def bench(ck, out):
    r = subprocess.run([PY, "bench_scoresaber.py", "--ckpt", ck,
                        "--out", out,
                        "--num-envs", "2048", "--fused-forward"],
                       capture_output=True, text=True, timeout=3000)
    if r.returncode != 0 or not os.path.exists(out):
        print(f"BENCH FAILED {ck}: {(r.stderr or '')[-300:]}", flush=True)
        return None
    b = json.load(open(out, encoding="utf-8"))
    lv = b["levels"]
    return dict(upp=b["total_unweighted_pp"], wpp=b["total_weighted_pp"],
                fcs=sum(1 for x in lv if x["full_combo"]),
                macc=sum(x["accuracy"] for x in lv) / len(lv))


def fmt(name, r):
    return (f"{name}: upp {r['upp']:>12,.0f}  wpp {r['wpp']:>9,.2f}"
            f"  FCs {r['fcs']}  mean-acc {r['macc']:.4f}")


def main():
    if os.path.exists(os.path.join(RUN_DIR, "final.pt")):
        print(f"{RUN_DIR}/final.pt already exists — refusing to retrain "
              "(delete the run dir to redo)", flush=True)
        return
    cmd = [PY, "train_batch.py", "--num-envs", "1024", "--horizon", "128",
           "--lr-decay", "cosine", "--seed", "0",
           "--ranked-maps", "--seg-len", "25",
           "--chart-pool", "24576",
           "--wide-acc",  # the default, explicit for the record
           "--lr", "0.00296092",
           "--ent-coef", "0.0331038", "--ent-final", "0.013929",
           "--gamma", "0.914958", "--gae-lambda", "0.980024",
           "--effort-coef", "0.240844",
           "--total-steps", str(TOTAL_STEPS),
           "--ckpt-every", str(CKPT_EVERY),
           "--run-dir", RUN_DIR]
    print(f"[train] 400M wide-acc fall-entropy leg -> {RUN_DIR}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
    wall = time.time() - t0
    ck = os.path.join(RUN_DIR, "final.pt")
    if r.returncode != 0 or not os.path.exists(ck):
        tail = (r.stdout or "")[-400:] + (r.stderr or "")[-400:]
        print(f"[train] FAILED ({wall:.0f}s): {tail}", flush=True)
        return
    print(f"[train] done in {wall/60:.1f} min", flush=True)

    res = bench(ck, os.path.join(RUN_DIR, "bench.json"))
    if res is None:
        return
    print("[final] " + fmt("final.pt", res), flush=True)

    ckpts = sorted(glob.glob(os.path.join(RUN_DIR, "ckpt_upd*.pt")),
                   key=lambda p: int(re.search(r"upd(\d+)", p).group(1)))
    table = {"final.pt": res}
    for c in ckpts:
        name = os.path.basename(c)[:-3]
        rr = bench(c, os.path.join("bench", f"wideacc_fall_400m_{name}.json"))
        if rr is None:
            continue
        upd = int(re.search(r"upd(\d+)", c).group(1))
        table[name] = rr
        print(f"[{name}] ({upd*131072/1e6:.0f}M) " + fmt("", rr), flush=True)

    print("\n=== wideacc_fall_400m — checkpoint trajectory "
          "(both pp totals) ===", flush=True)
    for name, rr in table.items():
        print(fmt(name, rr), flush=True)

    # procedural 30-ep greedy eval (REAL scoring) — the acc-component
    # prediction is judged on this line's "acc x.x/15"
    e = subprocess.run([PY, "eval_ckpt.py", "--ckpt", ck],
                       capture_output=True, text=True, timeout=1200)
    print("[procedural eval] " + (e.stdout or "").strip(), flush=True)
    if e.returncode != 0:
        print(f"[procedural eval] FAILED: {(e.stderr or '')[-300:]}",
              flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
