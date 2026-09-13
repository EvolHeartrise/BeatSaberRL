"""Env v8 gears head-to-head: 5-gear vs 7-gear at 400M (2026-07-30).

Two sequential fullspeed-protocol legs on the v8 body (alpha/omega
doubled; user-directed) differing ONLY in the action table: --gears 5
({0, +/-1/4, +/-1}) vs --gears 7 ({0, +/-1/8, +/-1/3, +/-1}). Recipe =
the fall-entropy winner unchanged. Probes whether torque resolution
moves the centering deficit (acc component 8.4/15 on the worst maps,
min position nudge 12.7 deg at full torque).

Both legs bench on the full catalog after training (scoring v3 —
comparable to each other; the fullspeed_400m baseline was the WEAKER
v7.9 body, reference only). Not appended to any sweep observations.

  python gears_head2head.py
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
TOTAL_STEPS = 400_000_000
CKPT_EVERY = 381  # updates; 381*131072 = 49.9M steps

HYPERS = {"lr": 0.00296092,
          "ent_coef": 0.0331038, "ent_final": 0.013929,
          "gamma": 0.914958, "gae_lambda": 0.980024}


def bench(ck, out):
    # 7200 s: the v10 catalog bench runs ~1,900-2,400 s (2026-08-07),
    # so this is ~3x margin. The old 3000 s was a pre-v10 constant.
    try:
        r = subprocess.run([PY, "bench_scoresaber.py", "--ckpt", ck,
                            "--out", out,
                            "--num-envs", "2048"],
                           capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        print(f"BENCH TIMEOUT {ck} (7200s)", flush=True)
        return None
    if r.returncode != 0 or not os.path.exists(out):
        print(f"BENCH FAILED {ck}: {(r.stderr or '')[-300:]}", flush=True)
        return None
    b = json.load(open(out, encoding="utf-8"))
    fcs = sum(1 for lv in b["levels"] if lv["full_combo"])
    acc = [lv["accuracy"] for lv in b["levels"] if "accuracy" in lv]
    return dict(upp=b["total_unweighted_pp"], wpp=b["total_weighted_pp"],
                fcs=fcs, acc=sum(acc) / len(acc) if acc else 0.0)


def leg(gears):
    h = HYPERS
    run_dir = os.path.join("runs", f"gears{gears}_400m")
    ck = os.path.join(run_dir, "final.pt")
    if os.path.exists(ck):
        print(f"[gears{gears}] final.pt exists — skipping train", flush=True)
    else:
        cmd = [PY, "train_batch.py", "--num-envs", "1024", "--horizon", "128",
               "--lr-decay", "cosine", "--seed", "0",
               "--ranked-maps", "--seg-len", "25",
               "--chart-pool", "24576",
               "--gears", str(gears),
               "--lr", f"{h['lr']:.6g}",
               "--ent-coef", f"{h['ent_coef']:.6g}",
               "--ent-final", f"{h['ent_final']:.6g}",
               "--gamma", f"{h['gamma']:.6g}",
               "--gae-lambda", f"{h['gae_lambda']:.6g}",
               "--total-steps", str(TOTAL_STEPS),
               "--ckpt-every", str(CKPT_EVERY),
               "--run-dir", run_dir]
        print(f"[gears{gears}] training 400M -> {run_dir}", flush=True)
        t0 = time.time()
        # total_steps at a conservative 8,000 steps/s floor (v10 measures
        # 12.4k at 1024/128, 24.4k at 4096/32 — 2026-08-07) + 1800 s
        # startup
        tmo = TOTAL_STEPS // 8000 + 1800
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=tmo)
        except subprocess.TimeoutExpired:
            print(f"[gears{gears}] TRAIN TIMEOUT ({tmo}s)", flush=True)
            return None
        wall = time.time() - t0
        if r.returncode != 0 or not os.path.exists(ck):
            tail = (r.stdout or "")[-400:] + (r.stderr or "")[-400:]
            print(f"[gears{gears}] TRAIN FAILED ({wall:.0f}s): {tail}",
                  flush=True)
            return None
        print(f"[gears{gears}] trained in {wall/60:.1f} min", flush=True)

    table = {}
    res = bench(ck, os.path.join(run_dir, "bench.json"))
    if res is None:
        return None
    table["final.pt"] = res
    print(f"[gears{gears} final] upp {res['upp']:,.0f} / "
          f"wpp {res['wpp']:,.2f} / FCs {res['fcs']} / "
          f"acc {res['acc']:.4f}", flush=True)
    for c in sorted(glob.glob(os.path.join(run_dir, "ckpt_upd*.pt")),
                    key=lambda p: int(re.search(r"upd(\d+)", p).group(1))):
        name = os.path.basename(c)[:-3]
        rr = bench(c, os.path.join("bench", f"gears{gears}_400m_{name}.json"))
        if rr is None:
            continue
        upd = int(re.search(r"upd(\d+)", c).group(1))
        table[name] = rr
        print(f"[gears{gears} {name}] ({upd*131072/1e6:.0f}M) "
              f"upp {rr['upp']:,.0f} / wpp {rr['wpp']:,.2f} / "
              f"FCs {rr['fcs']} / acc {rr['acc']:.4f}", flush=True)
    return table


def main():
    results = {}
    for gears in (5, 7):
        t = leg(gears)
        if t is not None:
            results[gears] = t
    print("\n=== gears head-to-head (v8 body, both pp totals) ===",
          flush=True)
    for gears, table in results.items():
        print(f"\n--- {gears}-gear ---", flush=True)
        for name, rr in table.items():
            print(f"{name}: upp {rr['upp']:>12,.0f}  wpp {rr['wpp']:>9,.2f}"
                  f"  FCs {rr['fcs']}  acc {rr['acc']:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
