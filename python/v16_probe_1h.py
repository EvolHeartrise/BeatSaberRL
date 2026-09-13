"""env v16 1h probe: discounted fall pricing + 0.5s segment lead.

The survival-drift fixes (user suspects #1 + #4, contact spec §11):
falls now charge the gamma-DISCOUNTED remaining miss stream (trainer
gamma 0.965 flows into fall_gamma), and ranked segments start notes
at 0.5 s instead of 1.5 s. Everything else is byte-identical to the
r3 1h probe (trunk 3,000 at lr 2e-3, ent tau 1000 -> floor 1e-3,
cd400 leaf) whose leaf benched 166.4 — THE baseline. Drift itself
needs a long trunk to show; this probe gates the fixes (leaf >= r3's
166 = fixes don't hurt; the falls/return curves show the new
incentives' early shape).

Same total compute as the v15 probe (3,400 updates x 131,072 =
445,644,800 steps), same env (v15 head-anchored track), same s4 recipe
values — only the SCHEDULES differ:

  cosine baseline (v15 probe): lr cosine 5e-3 -> 1e-4, ent linear
      0.03 -> 3e-6, both over the full 3,400.
  WSD arm (this): trunk = warmup (auto ~45) then CONSTANT lr 5e-3 and
      CONSTANT ent 0.03 for 3,000 updates; leaf = 400-update cooldown
      (11.8%), lr 1-sqrt to 1e-4, ent linear to 3e-6.

Scoreboard: leaf bench vs the v15 probe's final 174.6 upp. The trunk
end is benched too (raw, un-decayed) to measure the trunk->leaf gap —
the river-valley picture predicts it is LARGE. Trunk checkpoints every
500 stay on disk: any of them can later be extended or branched.

  python wsd_probe_1h.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = sys.executable
RUN_DIR = os.path.join("runs", "v16_probe_1h")
NUM_ENVS, HORIZON = 16384, 8
TRUNK_UPDATES, LEAF_UPDATES = 3_000, 400
BATCH = NUM_ENVS * HORIZON
TRAIN_TIMEOUT = (TRUNK_UPDATES + LEAF_UPDATES) * BATCH // 100_000 + 1_800

COMMON = ["--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
          "--foot-contact", "--gears", "5",
          "--seed", "0",
          "--ranked-maps", "--seg-len", "25",
          "--chart-pool", "24576",
          "--gamma", "0.965", "--gae-lambda", "0.98",
          "--ent-coef", "0.03",
          "--lead-s", "0.5"]


def run_phase(label, cmd, log_name):
    print(f"[{label}] {' '.join(cmd[1:3])} ... -> {RUN_DIR}", flush=True)
    t0 = time.time()
    with open(os.path.join(RUN_DIR, log_name), "w", encoding="utf-8") as log:
        try:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               text=True, timeout=TRAIN_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"[{label}] TIMEOUT ({TRAIN_TIMEOUT}s)", flush=True)
            return False
    wall = time.time() - t0
    if r.returncode != 0:
        tail = open(os.path.join(RUN_DIR, log_name),
                    encoding="utf-8").read()[-800:]
        print(f"[{label}] FAILED ({wall:.0f}s): {tail}", flush=True)
        return False
    print(f"[{label}] done in {wall/60:.1f} min", flush=True)
    return True


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
    os.makedirs(RUN_DIR, exist_ok=True)
    trunk_ck = os.path.join(RUN_DIR, "trunk_final.pt")
    leaf_ck = os.path.join(RUN_DIR, f"leaf_from_upd{TRUNK_UPDATES:05d}.pt")

    if os.path.exists(trunk_ck):
        print(f"{trunk_ck} exists — skipping trunk", flush=True)
    else:
        ok = run_phase("trunk", [PY, "train_batch.py", *COMMON,
                                 "--lr", "2e-3",
                                 "--ent-final", "1e-3", "--ent-tau", "1000",
                                 "--total-steps", str(TRUNK_UPDATES * BATCH),
                                 "--ckpt-every", "500",
                                 "--run-dir", RUN_DIR], "trunk.log")
        if not ok:
            return

    if os.path.exists(leaf_ck):
        print(f"{leaf_ck} exists — skipping leaf", flush=True)
    else:
        ok = run_phase("leaf", [PY, "train_batch.py", *COMMON,
                                "--resume", trunk_ck,
                                "--cooldown-updates", str(LEAF_UPDATES),
                                "--ent-final", "3e-6"], "leaf.log")
        if not ok:
            return

    table = {}
    for name, ck in (("trunk_raw", trunk_ck), ("leaf", leaf_ck)):
        res = bench(ck, os.path.join("bench", f"v16_probe_1h_{name}.json"))
        if res:
            table[name] = res
            print(f"[{name}] upp {res['upp']:,.0f} / wpp {res['wpp']:,.2f} / "
                  f"FCs {res['fcs']} / acc {res['acc']:.4f}", flush=True)
    table["baseline_r3_1h_leaf_v15env"] = {"upp": 166.42, "wpp": 9.23}
    table["baseline_crown_8k_leaf"] = {"upp": 6318.04, "wpp": 99.53}
    print("DONE " + json.dumps(table), flush=True)


if __name__ == "__main__":
    main()
