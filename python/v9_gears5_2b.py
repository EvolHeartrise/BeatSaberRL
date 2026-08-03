"""FIRST env v9 training run: gears5 at 2B on the 7-joint body
(2026-07-30). v9 adds wrist radial/ulnar deviation (right [-15,+30],
left mirrored) — the orthogonal bending axis the roll+bend wrist
lacked; nothing has trained on it before this run. Protocol = the v8
gears5_2b record run unchanged (fall-entropy recipe, ranked 25 s
segments, ~100M rungs, all benched post-train) so the body is the only
variable. v8 reference (RETIRED body, comparison indicative only):
2,535,389 upp / 19,395.60 wpp / 4,702 FCs / .9739 acc.

  python v9_gears5_2b.py
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
RUN_DIR = os.path.join("runs", "v9_gears5_2b")
TOTAL_STEPS = 2_000_000_000
CKPT_EVERY = 762  # updates; 762*131072 = 99.9M steps

HYPERS = {"lr": 0.00296092,
          "ent_coef": 0.0331038, "ent_final": 0.013929,
          "gamma": 0.914958, "gae_lambda": 0.980024,
          "effort_coef": 0.240844}


def bench(ck, out):
    r = subprocess.run([PY, "bench_scoresaber.py", "--ckpt", ck,
                        "--out", out, "--num-envs", "2048"],
                       capture_output=True, text=True, timeout=3000)
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
               "--effort-coef", f"{h['effort_coef']:.6g}",
               "--total-steps", str(TOTAL_STEPS),
               "--ckpt-every", str(CKPT_EVERY),
               "--run-dir", RUN_DIR]
        print(f"[train] gears5 2B -> {RUN_DIR}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=28800)
        wall = time.time() - t0
        if r.returncode != 0 or not os.path.exists(ck):
            tail = (r.stdout or "")[-400:] + (r.stderr or "")[-400:]
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
        rr = bench(c, os.path.join("bench", f"v9_gears5_2b_{name}.json"))
        if rr is None:
            continue
        upd = int(re.search(r"upd(\d+)", c).group(1))
        table[name] = rr
        print(f"[{name}] ({upd*131072/1e9:.2f}B) upp {rr['upp']:,.0f} / "
              f"wpp {rr['wpp']:,.2f} / FCs {rr['fcs']} / "
              f"acc {rr['acc']:.4f}", flush=True)

    best = max(table.items(), key=lambda kv: kv[1]["upp"])
    print(f"\nBEST upp: {best[0]} — {best[1]['upp']:,.0f}", flush=True)
    print("\n=== gears5_2b — checkpoint trajectory (both pp totals) ===",
          flush=True)
    for name, rr in table.items():
        print(f"{name}: upp {rr['upp']:>12,.0f}  wpp {rr['wpp']:>9,.2f}"
              f"  FCs {rr['fcs']}  acc {rr['acc']:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
