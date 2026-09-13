"""Sweep-s2 confirmation run: fall-entropy recipe at 400M (2026-07-29).

Fresh 400M-step leg of the run-085 recipe with the FALLING entropy
schedule (0.0331038 -> 0.013929) that won the shape test
(sessions/2026-07-29-ent-shape-test.md). Questions it answers:
does the ranked-segment recipe scale with budget (gamma-ridge drift
said yes), does it close the wpp/FC gap vs gen5b's 4B checkpoint, and
where does the peak sit (gen5b peaked at 80% of budget)?

Protocol = sweep leg (seed 0, same pool draw, same train/bench
commands) so the final.pt result is appended to
sweeps/s2_ranked_upp/observations.jsonl as observation idx 102 —
sweep-valid data for future GP legs. Interval checkpoints every 381
updates (~50M steps at N=1024, T=128) are benched AFTER training
(sequential — no GPU contention, so the logged cost stays pure train
wall like every other observation).

  python confirm_fall_400m.py     # ~26 min train + ~9 min benches
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
RUN_DIR = os.path.join("runs", "confirm_fall_400m")
OBS = os.path.join("sweeps", "s2_ranked_upp", "observations.jsonl")
TOTAL_STEPS = 400_000_000
CKPT_EVERY = 381  # updates; 381*131072 = 49.9M steps

HYPERS = {"train": {"lr": 0.00296092,
                    "ent_coef": 0.0331038, "ent_final": 0.013929,
                    "gamma": 0.914958, "gae_lambda": 0.980024,
                    "total_steps": float(TOTAL_STEPS)}}


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
    return dict(upp=b["total_unweighted_pp"], wpp=b["total_weighted_pp"],
                fcs=fcs)


def main():
    h = HYPERS["train"]
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
           "--total-steps", str(TOTAL_STEPS),
           "--ckpt-every", str(CKPT_EVERY),
           "--run-dir", RUN_DIR]
    print(f"[train] 400M fall-entropy leg -> {RUN_DIR}", flush=True)
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

    # final.pt first (the sweep observation), then interval ckpts
    res = bench(ck, os.path.join(RUN_DIR, "bench.json"))
    if res is None:
        return
    print(f"[final] upp {res['upp']:,.0f} / wpp {res['wpp']:,.2f} / "
          f"FCs {res['fcs']}", flush=True)

    with open(OBS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"idx": 102, "hypers": HYPERS,
                            "score": res["upp"], "cost": round(wall, 1),
                            "is_failure": False}) + "\n")
    print(f"[obs] appended idx 102 (score {res['upp']:,.0f}, "
          f"cost {wall:.0f}s) -> {OBS}", flush=True)

    ckpts = sorted(glob.glob(os.path.join(RUN_DIR, "ckpt_upd*.pt")),
                   key=lambda p: int(re.search(r"upd(\d+)", p).group(1)))
    table = {"final.pt": res}
    for c in ckpts:
        name = os.path.basename(c)[:-3]
        out = os.path.join("bench", f"confirm_fall_400m_{name}.json")
        rr = bench(c, out)
        if rr is None:
            continue
        upd = int(re.search(r"upd(\d+)", c).group(1))
        table[name] = rr
        print(f"[{name}] ({upd*131072/1e6:.0f}M) upp {rr['upp']:,.0f} / "
              f"wpp {rr['wpp']:,.2f} / FCs {rr['fcs']}", flush=True)

    print("\n=== confirm_fall_400m — checkpoint trajectory "
          "(both pp totals) ===", flush=True)
    for name, rr in table.items():
        print(f"{name}: upp {rr['upp']:>12,.0f}  wpp {rr['wpp']:>9,.2f}"
              f"  FCs {rr['fcs']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
