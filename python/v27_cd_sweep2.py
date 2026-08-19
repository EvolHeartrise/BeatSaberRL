"""v27 cooldown sweep, point 2 (user direction 2026-08-19): repeat
the cd-length sweep from the 4,500-update trunk (= recipe-9k depth) so we have optimal-cd measurements at
TWO depths. Arms cd100/200/400/800: cd50 was refuted at the 1,500
sweep (+54.3 return — the 30->3e-6 entropy crash in 50 updates breaks
the policy); cd800 added because EPFL 2405.18392's 10-20%-of-trunk
guidance predicts the optimum GROWS with trunk length (10-20% of
4,500 = 450-900 — cd800 probes past the bracket top).

    python v27_cd_sweep2.py    # requires runs/v27_critic/trunk_upd04500.pt
"""

from __future__ import annotations

import os
import sys

from v27_critic_probe import COMMON, RUN_DIR as PROBE_DIR
from v27_cd_sweep import leaf_stats, run_logged

PY = sys.executable
ARMS = [100, 200, 400, 800]
TRUNK_CK = os.path.join(PROBE_DIR, "trunk_upd04500.pt")


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    if not os.path.exists(TRUNK_CK):
        sys.exit(f"[cd-sweep2] missing {TRUNK_CK} — extend the trunk first")
    for cd in ARMS:
        rd = os.path.join("runs", f"v27_45k_cd{cd:03d}")
        out = os.path.join("bench", f"v27_critic_leaf04500_cd{cd:03d}.json")
        if os.path.exists(out):
            print(f"[cd-sweep2] cd{cd} already benched, skipping",
                  flush=True)
            continue
        os.makedirs(rd, exist_ok=True)
        if not [f for f in os.listdir(rd) if f.startswith("leaf_from")]:
            print(f"[cd-sweep2] cutting cd{cd} from upd 4500", flush=True)
            w = run_logged([PY, "train_batch.py", *COMMON,
                            "--resume", TRUNK_CK,
                            "--cooldown-updates", str(cd),
                            "--ent-final", "3e-6", "--run-dir", rd],
                           os.path.join(rd, "leaf.log"), 14_400)
            print(f"[cd-sweep2] cd{cd} leaf done in {w/60:.1f} min",
                  flush=True)
        leaf = [f for f in os.listdir(rd)
                if f.startswith("leaf_from") and f.endswith(".pt")][0]
        run_logged([PY, "bench_scoresaber.py",
                    "--ckpt", os.path.join(rd, leaf),
                    "--num-envs", "2048", "--foot-contact", "--out", out],
                   os.path.join(rd, "bench.log"), 10_800)
        print(f"SWEEP2_RESULT v27_45k_cd{cd:03d} {leaf_stats(rd, out)}",
              flush=True)
    print("SWEEP2-DONE", flush=True)


if __name__ == "__main__":
    main()
