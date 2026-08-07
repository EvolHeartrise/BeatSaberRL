"""First real-note-speed training run: 400M on ranked segments (2026-07-30).

speed_scale is REMOVED (user decision): ranked maps now play at their
real NJS (10-23 m/s vs the historical NJS*0.3 = 3-7). This is the first
training probe of the real physics regime at 30 Hz — contact windows
are 1-2 ticks (vs 4-10 at 0.3x), the early-cut allowance is 33-75 ms
(vs 110-250), and timing errors cost ~3.3x more accuracy under the
cut-plane rule. Recipe = the s2/s3 fall-entropy winner (confirm_fall
protocol) unchanged — deliberately, so the regime change is the only
variable. NOT appended to the s2 sweep observations: different physics,
would poison the GP.

Benches run sequentially after training (scoring_version 3 — real-speed
results never merge with 0.3-era files; the first bench also pays the
full chart_cache reparse, the parser-source hash changed).

  python fullspeed_400m.py
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
RUN_DIR = os.path.join("runs", "fullspeed_400m")
TOTAL_STEPS = 400_000_000
CKPT_EVERY = 381  # updates; 381*131072 = 49.9M steps

HYPERS = {"lr": 0.00296092,
          "ent_coef": 0.0331038, "ent_final": 0.013929,
          "gamma": 0.914958, "gae_lambda": 0.980024,
          "effort_coef": 0.240844}


def bench(ck, out):
    # 7200 s: the v10 catalog bench runs ~1,900-2,400 s (2026-08-07),
    # so this is ~3x margin. The old 3000 s was a pre-v10 constant.
    try:
        r = subprocess.run([PY, "bench_scoresaber.py", "--ckpt", ck,
                            "--out", out,
                            "--num-envs", "2048", "--fused-forward"],
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


def main():
    h = HYPERS
    if os.path.exists(os.path.join(RUN_DIR, "final.pt")):
        print(f"{RUN_DIR}/final.pt already exists — refusing to retrain "
              "(delete the run dir to redo)", flush=True)
        return
    cmd = [PY, "train_batch.py", "--num-envs", "1024", "--horizon", "128",
           "--lr-decay", "cosine", "--seed", "0",
           "--ranked-maps", "--seg-len", "25",
           "--chart-pool", "24576",
           "--lr", f"{h['lr']:.6g}",
           "--ent-coef", f"{h['ent_coef']:.6g}",
           "--ent-final", f"{h['ent_final']:.6g}",
           "--gamma", f"{h['gamma']:.6g}",
           "--gae-lambda", f"{h['gae_lambda']:.6g}",
           "--effort-coef", f"{h['effort_coef']:.6g}",
           "--total-steps", str(TOTAL_STEPS),
           "--ckpt-every", str(CKPT_EVERY),
           "--run-dir", RUN_DIR]
    print(f"[train] 400M full-speed leg -> {RUN_DIR}", flush=True)
    t0 = time.time()
    # total_steps at a conservative 8,000 steps/s floor (v10 measures
    # 12.4k at 1024/128, 24.4k at 4096/32 — 2026-08-07) + 1800 s startup
    tmo = TOTAL_STEPS // 8000 + 1800
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=tmo)
    except subprocess.TimeoutExpired:
        print(f"[train] TIMEOUT ({tmo}s)", flush=True)
        return
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
    print(f"[final] upp {res['upp']:,.0f} / wpp {res['wpp']:,.2f} / "
          f"FCs {res['fcs']} / acc {res['acc']:.4f}", flush=True)

    ckpts = sorted(glob.glob(os.path.join(RUN_DIR, "ckpt_upd*.pt")),
                   key=lambda p: int(re.search(r"upd(\d+)", p).group(1)))
    table = {"final.pt": res}
    for c in ckpts:
        name = os.path.basename(c)[:-3]
        out = os.path.join("bench", f"fullspeed_400m_{name}.json")
        rr = bench(c, out)
        if rr is None:
            continue
        upd = int(re.search(r"upd(\d+)", c).group(1))
        table[name] = rr
        print(f"[{name}] ({upd*131072/1e6:.0f}M) upp {rr['upp']:,.0f} / "
              f"wpp {rr['wpp']:,.2f} / FCs {rr['fcs']} / "
              f"acc {rr['acc']:.4f}", flush=True)

    print("\n=== fullspeed_400m — checkpoint trajectory "
          "(both pp totals) ===", flush=True)
    for name, rr in table.items():
        print(f"{name}: upp {rr['upp']:>12,.0f}  wpp {rr['wpp']:>9,.2f}"
              f"  FCs {rr['fcs']}  acc {rr['acc']:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
