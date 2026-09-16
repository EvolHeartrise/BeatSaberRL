"""env v13 run: 1 hour, c-shape, K=5 — body-point obs + longer gamma.

Two changes over the v12 recipe (user, 2026-08-09):
- env v13 obs (contact spec §8): explicit elbow/knee/foot world
  positions — FK shortcuts so the net spends parameters on control,
  not kinematics.
- gamma 0.914958 -> 0.97. The wide-sweep gamma was tuned on WELD
  physics, where balance was free and note credit is short (~0.4 s).
  Under contact, the causal gap between a fatal lean and the fall is
  ~0.5-1.5 s (pendulum doubling 216 ms, ~5 doublings from wobble to
  FALL_ROOT_Y); at gamma 0.915 a fall 1 s away is discounted to 7%.
  0.97 puts the value horizon at ~1.1 s (33 ticks) while gamma*lambda
  = 0.951 keeps the credit trace at ~0.67 s — long enough to see falls
  coming, short enough not to drown note credit. Lambda unchanged.

Everything else inherited: 16384/8, K=5, wide lr/ent schedule, alive
bonus + fall pricing (env v12 reward, unchanged by v13). Fresh init.

Budget: ~1 h at an estimated ~145k steps/s (v12 ran 150.7k; v13 adds
18 obs lanes and 12 policy-input features) => 4,000 updates x 131,072
= 524,288,000 steps. Checkpoints every 1,000; bench final + last 3,
all --foot-contact.

Watch: falls/update vs the v12 curve AT EQUAL UPDATE COUNTS (v12: 5,008
@400, 4,686 @1200, 3,411 @2400, 2,625 @3200, 2,031 @4300 — note v12's
anneals were stretched over 4,300 updates, this run's over 4,000, so
the comparison is nearly schedule-fair), then bench pp vs v12's 123.

  python v13_body_pts_1h.py
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
RUN_DIR = os.path.join("runs", "v13_body_pts_1h")
NUM_ENVS, HORIZON = 16384, 8
TOTAL_STEPS = 4_000 * 131_072           # 524,288,000 (~1.0 h @ ~145k)
CKPT_EVERY = 1_000
TRAIN_TIMEOUT = TOTAL_STEPS // 100_000 + 1_800

HYPERS = {"lr": 0.00296092,
          "ent_coef": 0.0331038, "ent_final": 0.013929,
          "gamma": 0.97, "gae_lambda": 0.980024}


def bench(ck, out):
    try:
        r = subprocess.run([PY, "bench_scoresaber.py", "--ckpt", ck,
                            "--out", out, "--num-envs", "2048",
                            "--foot-contact"],
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
        cmd = [PY, "train_batch.py",
               "--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
               "--foot-contact",
               "--lr-decay", "cosine", "--seed", "0",
               "--ranked-maps", 
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
        print(f"[train] v13 BODY-PTS 1h ({NUM_ENVS}/{HORIZON}, "
              f"{TOTAL_STEPS/1e6:.0f}M, +body points, gamma 0.97, K=5) "
              f"-> {RUN_DIR}", flush=True)
        t0 = time.time()
        with open(os.path.join(RUN_DIR, "train.log"), "w",
                  encoding="utf-8") as log:
            try:
                r = subprocess.run(cmd, stdout=log,
                                   stderr=subprocess.STDOUT,
                                   text=True, timeout=TRAIN_TIMEOUT)
            except subprocess.TimeoutExpired:
                print(f"[train] TIMEOUT ({TRAIN_TIMEOUT}s)", flush=True)
                return
        wall = time.time() - t0
        if r.returncode != 0 or not os.path.exists(ck):
            tail = open(os.path.join(RUN_DIR, "train.log"),
                        encoding="utf-8").read()[-800:]
            print(f"[train] FAILED ({wall:.0f}s): {tail}", flush=True)
            return
        print(f"[train] done in {wall/60:.1f} min "
              f"({TOTAL_STEPS/wall:,.0f} steps/s wall)", flush=True)

    table = {}
    res = bench(ck, os.path.join("bench", "v13_body_pts_1h_final.json"))
    if res:
        table["final"] = res
        print(f"[final] upp {res['upp']:,.0f} / wpp {res['wpp']:,.2f} / "
              f"FCs {res['fcs']} / acc {res['acc']:.4f}", flush=True)
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "ckpt_upd*.pt")),
                      key=lambda p: int(re.search(r"upd(\d+)", p).group(1)))
    for c in numbered[-3:]:
        name = os.path.basename(c)[:-3]
        rr = bench(c, os.path.join("bench", f"v13_body_pts_1h_{name}.json"))
        if rr is None:
            continue
        table[name] = rr
        print(f"[{name}] upp {rr['upp']:,.0f} / wpp {rr['wpp']:,.2f} / "
              f"FCs {rr['fcs']} / acc {rr['acc']:.4f}", flush=True)
    print("DONE " + json.dumps(table), flush=True)


if __name__ == "__main__":
    main()
