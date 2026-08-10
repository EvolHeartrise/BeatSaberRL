"""env v14 probe: 1 hour, c-shape, K=5 — first training on honest physics.
First run since the phantom-force repair (contact spec §9): stops are
reciprocal-impulse springs, the free bracing crutch is gone, and cold-
start falls are WORSE than v13 (~10k/update vs ~7k) because standing
must now be earned. This probe answers one question: does balance
learning progress at all on real physics with the s4-locked recipe
(gamma 0.965, lr 5e-3, ent 0.03->3e-6, lambda 0.98, effort 1e-4)?

v13-era numbers are NOT comparable baselines (that policy trained on
+250-335 N of phantom support). The probe's own falls curve is the
verdict: monotone decline = learnable, plateau at ~10k = the task
needs help (curriculum / v10 warm-start / gear review).

1 h at the measured v14 rate (124k steps/s): 3,400 updates x 131,072
= 445,644,800 steps. Checkpoints every 1,000; bench final + last 3.

  python v14_probe_1h.py
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
RUN_DIR = os.path.join("runs", "v14_probe_1h")
NUM_ENVS, HORIZON = 16384, 8
TOTAL_STEPS = 3_400 * 131_072           # 445,644,800 (~1.0 h @ 124k)
CKPT_EVERY = 1_000
TRAIN_TIMEOUT = TOTAL_STEPS // 100_000 + 1_800

HYPERS = {"lr": 5e-3,
          "ent_coef": 0.03, "ent_final": 3e-6,
          "gamma": 0.965, "gae_lambda": 0.98,
          "effort_coef": 1e-4}


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
        print(f"[train] v14 PROBE 1h ({NUM_ENVS}/{HORIZON}, "
              f"{TOTAL_STEPS/1e6:.0f}M, honest physics, K=5) "
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
    res = bench(ck, os.path.join("bench", "v14_probe_1h_final.json"))
    if res:
        table["final"] = res
        print(f"[final] upp {res['upp']:,.0f} / wpp {res['wpp']:,.2f} / "
              f"FCs {res['fcs']} / acc {res['acc']:.4f}", flush=True)
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "ckpt_upd*.pt")),
                      key=lambda p: int(re.search(r"upd(\d+)", p).group(1)))
    for c in numbered[-3:]:
        name = os.path.basename(c)[:-3]
        rr = bench(c, os.path.join("bench", f"v14_probe_1h_{name}.json"))
        if rr is None:
            continue
        table[name] = rr
        print(f"[{name}] upp {rr['upp']:,.0f} / wpp {rr['wpp']:,.2f} / "
              f"FCs {rr['fcs']} / acc {rr['acc']:.4f}", flush=True)
    print("DONE " + json.dumps(table), flush=True)


if __name__ == "__main__":
    main()
