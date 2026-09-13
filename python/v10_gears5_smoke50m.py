"""FIRST env v10 training run: 50M smoke on the full dynamic body
(2026-07-31). v10 replaces the fixed-shoulder kinematic chains with a
29-joint humanoid under full rigid-body dynamics (floating pelvis,
welded feet, gravity, capsule self-collision, torque actuation) — see
SPEC-v10.md. Nothing has ever trained on it. Protocol mirrors the
v8/v9 record-run recipe (fall-entropy, ranked 25 s segments, gears 5,
N=1024) with ONE deliberate change: effort_coef 4e-4 — v10 effort is
physical work in joules (SPEC calibration), not the v9 per-action
table; the v9 value (0.2408) would tax a typical swing ~600x and
train a statue.

Smoke gates (not competence): run completes, no NaN, pd_net hits ~0,
entropy tracks schedule, reward stream live, checkpoints bench.
Registered prediction (Claude, pre-run): stable but minimally
competent at 50M — bench upp <= 100k, FCs <= 5, engagement below the
PD bot's 33-37%%; rationale: gravity+balance+torque exploration depth
dwarfs the v6 precedent, and gen-v2 alone doubled surge onset.

  python v10_gears5_smoke50m.py
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
RUN_DIR = os.path.join("runs", "v10_gears5_smoke50m")
TOTAL_STEPS = 50_000_000
CKPT_EVERY = 95  # updates; 95*131072 = 12.45M steps -> 3 ckpts + final

HYPERS = {"lr": 0.00296092,
          "ent_coef": 0.0331038, "ent_final": 0.013929,
          "gamma": 0.914958, "gae_lambda": 0.980024}


def bench(ck, out):
    try:
        r = subprocess.run([PY, "bench_scoresaber.py", "--ckpt", ck,
                            "--out", out, "--num-envs", "2048"],
                           capture_output=True, text=True, timeout=10800)
    except subprocess.TimeoutExpired:
        print(f"BENCH TIMEOUT {ck} (10800s)", flush=True)
        return None
    if r.returncode != 0 or not os.path.exists(out):
        print(f"BENCH FAILED {ck}: {(r.stderr or '')[-300:]}", flush=True)
        return None
    b = json.load(open(out, encoding="utf-8"))
    fcs = sum(1 for lv in b["levels"] if lv["full_combo"])
    acc = [lv["accuracy"] for lv in b["levels"] if "accuracy" in lv]
    return dict(upp=b["total_unweighted_pp"], wpp=b["total_weighted_pp"],
                fcs=fcs, acc=sum(acc) / len(acc) if acc else 0.0)


def main():
    h = HYPERS
    os.makedirs(RUN_DIR, exist_ok=True)
    ck = os.path.join(RUN_DIR, "final.pt")
    if os.path.exists(ck):
        print(f"{RUN_DIR}/final.pt exists — skipping train", flush=True)
    else:
        cmd = [PY, "train_batch.py", "--num-envs", "1024", "--horizon", "128",
               "--lr-decay", "cosine", "--seed", "0",
               "--ranked-maps", "--seg-len", "25",
               "--chart-pool", "24576",
               "--gears", "5",
               "--lr", f"{h['lr']:.6g}",
               "--ent-coef", f"{h['ent_coef']:.6g}",
               "--ent-final", f"{h['ent_final']:.6g}",
               "--gamma", f"{h['gamma']:.6g}",
               "--gae-lambda", f"{h['gae_lambda']:.6g}",
               "--total-steps", str(TOTAL_STEPS),
               "--ckpt-every", str(CKPT_EVERY),
               "--run-dir", RUN_DIR]
        print(f"[train] v10 smoke 50M -> {RUN_DIR}", flush=True)
        t0 = time.time()
        log = open(os.path.join(RUN_DIR, "train.log"), "w",
                   encoding="utf-8")
        try:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               text=True, timeout=25200)
        except subprocess.TimeoutExpired:
            log.close()
            print("[train] TIMEOUT (25200s)", flush=True)
            return
        log.close()
        wall = time.time() - t0
        if r.returncode != 0 or not os.path.exists(ck):
            tail = open(os.path.join(RUN_DIR, "train.log"),
                        encoding="utf-8").read()[-800:]
            print(f"[train] FAILED ({wall:.0f}s): {tail}", flush=True)
            return
        print(f"[train] done in {wall/60:.1f} min", flush=True)

    table = {}
    res = bench(ck, os.path.join(RUN_DIR, "bench.json"))
    if res is None:
        return
    table["final.pt"] = res
    print(f"[final] upp {res['upp']:,.0f} / wpp {res['wpp']:,.2f} / "
          f"FCs {res['fcs']} / acc {res['acc']:.4f}", flush=True)
    for c in sorted(glob.glob(os.path.join(RUN_DIR, "ckpt_upd*.pt")),
                    key=lambda p: int(re.search(r"upd(\d+)", p).group(1))):
        name = os.path.basename(c)[:-3]
        rr = bench(c, os.path.join("bench", f"v10_smoke50m_{name}.json"))
        if rr is None:
            continue
        upd = int(re.search(r"upd(\d+)", c).group(1))
        table[name] = rr
        print(f"[{name}] ({upd*131072/1e6:.1f}M) upp {rr['upp']:,.0f} / "
              f"wpp {rr['wpp']:,.2f} / FCs {rr['fcs']} / "
              f"acc {rr['acc']:.4f}", flush=True)

    best = max(table.items(), key=lambda kv: kv[1]["upp"])
    print(f"\nBEST upp: {best[0]} — {best[1]['upp']:,.0f}", flush=True)
    print("\n=== v10 smoke — checkpoint trajectory (both pp totals) ===",
          flush=True)
    for name, rr in table.items():
        print(f"{name}: upp {rr['upp']:>12,.0f}  wpp {rr['wpp']:>9,.2f}"
              f"  FCs {rr['fcs']}  acc {rr['acc']:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
