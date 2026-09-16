"""env v12 unblinded-feet run: 8 hours, c-shape, K=5, contact obs +
survival gradient.

Scale-up of the 1h run (user, 2026-08-09): falls fell 5,008 -> 2,031
MONOTONE with no plateau in sight, episodes crossed the 2.0 s first
note, first nonzero contact pp (upp 15 -> 123 across ckpts). Same
recipe, fresh init, 8x the wall to find the fall-curve floor and see
whether hit rate compounds once survival is routine.

The v11.1 8h run's falls plateaued ~5,100/update (episodes ~0.86 s,
first note at 2.0 s). v12 (contact spec §7) attacks the two most likely
causes at once: the policy now FEELS the ground (four per-foot
heel/toe normal-impulse lanes into the body token) and survival time
has a gradient (ALIVE_BONUS 0.002/tick, terminal ticks excluded).
Everything else inherited from the v11 recipe: 16384/8, K=5, frozen
wide hypers. Fresh init — the obs change orphans every earlier
checkpoint anyway.

Budget: 8 h of train wall at the 1h run's measured rate (148.2k
steps/s) => 32,500 updates x 131,072 = 4,259,840,000 steps (~7.99 h).
Checkpoints every 2,400 updates; bench final + last 3, all
--foot-contact.

Watch: where the fall curve bottoms out (1h ended at 2,031 and still
declining), ep_return unpinning from ~-70 as the hit rate grows, and
bench pp against the 1h ladder (upp 123 at upd4000).

  python v12_contact_obs_8h.py
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
RUN_DIR = os.path.join("runs", "v12_contact_obs_8h")
NUM_ENVS, HORIZON = 16384, 8
TOTAL_STEPS = 32_500 * 131_072          # 4,259,840,000 (~7.99 h @ 148.2k)
CKPT_EVERY = 2_400
TRAIN_TIMEOUT = TOTAL_STEPS // 100_000 + 1_800

HYPERS = {"lr": 0.00296092,
          "ent_coef": 0.0331038, "ent_final": 0.013929,
          "gamma": 0.914958, "gae_lambda": 0.980024}


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
        print(f"[train] v12 CONTACT-OBS 8h ({NUM_ENVS}/{HORIZON}, "
              f"{TOTAL_STEPS/1e6:.0f}M, unblinded feet + alive bonus, "
              f"K=5) -> {RUN_DIR}", flush=True)
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
    res = bench(ck, os.path.join("bench", "v12_contact_obs_8h_final.json"))
    if res:
        table["final"] = res
        print(f"[final] upp {res['upp']:,.0f} / wpp {res['wpp']:,.2f} / "
              f"FCs {res['fcs']} / acc {res['acc']:.4f}", flush=True)
    numbered = sorted(glob.glob(os.path.join(RUN_DIR, "ckpt_upd*.pt")),
                      key=lambda p: int(re.search(r"upd(\d+)", p).group(1)))
    for c in numbered[-3:]:
        name = os.path.basename(c)[:-3]
        rr = bench(c, os.path.join("bench", f"v12_contact_obs_8h_{name}.json"))
        if rr is None:
            continue
        table[name] = rr
        print(f"[{name}] upp {rr['upp']:,.0f} / wpp {rr['wpp']:,.2f} / "
              f"FCs {rr['fcs']} / acc {rr['acc']:.4f}", flush=True)
    print("DONE " + json.dumps(table), flush=True)


if __name__ == "__main__":
    main()
