"""Dead-gear census over replay .jsonl files (the Dead Gear Census artifact's
data). Per joint: share of ticks on each gear id (0 coast, 1 +1.0, 2 -1.0,
3 +1/4, 4 -1/4 of tau_max), the mean |tau| the gear choices imply (gear
fraction x TAU_MAX x the replay's tau_scale for that axis), group summaries,
and the count of (joint, gear) cells under 1% use. Cross-checks the
gear-derived total against the replay's own torque telemetry.

  python tools/gear_census.py --out census.json replays/v51_cd400_lf2_toppp*.jsonl
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "_ba_constants", os.path.join(PY, "beat_arena_env", "constants.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)
TAU_MAX = np.asarray(_c.TAU_MAX, dtype=np.float64)
TAU_AXIS_OF_JOINT = list(_c.TAU_AXIS_OF_JOINT)
GEAR = np.asarray(_c.gear_table(5))

JOINTS = (["spine pitch", "spine yaw", "spine roll"]
          + [f"{s} {n}" for s in ("L", "R") for n in
             ("arm pitch", "arm yaw", "arm twist", "elbow", "wrist roll",
              "wrist bend", "wrist dev")]
          + [f"{s} {n}" for s in ("L", "R") for n in
             ("hip pitch", "hip yaw", "hip roll", "knee", "ankle pitch",
              "ankle roll")])
GROUPS = {"spine": range(0, 3), "arms": [3, 4, 5, 6, 10, 11, 12, 13],
          "wrists": [7, 8, 9, 14, 15, 16], "hips": [17, 18, 19, 23, 24, 25],
          "knees": [20, 26], "ankles": [21, 22, 27, 28]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    counts = np.zeros((29, 5), dtype=np.int64)
    tau_scale = None
    body = None
    tele = np.zeros(4)
    n_ticks = 0
    per_file = []
    for f in a.files:
        acts = []
        with open(f) as fh:
            hdr = json.loads(fh.readline())["header"]
            ts = hdr.get("tau_scale") or [1.0] * 16
            if tau_scale is None:
                tau_scale = ts
            assert list(ts) == list(tau_scale), (f, ts)
            # run-only body options (record_rev 6; absent = default)
            b = _c.env_body(hdr.get("env_body"))
            if body is None:
                body = b
            assert b == body, (f, b)
            for line in fh:
                r = json.loads(line)
                acts.append(r["action"])
                tele += np.asarray(r["torque"], dtype=np.float64)
        acts = np.asarray(acts, dtype=np.int64)
        for j in range(29):
            counts[j] += np.bincount(acts[:, j], minlength=5)
        n_ticks += len(acts)
        per_file.append({"file": os.path.basename(f), "ticks": int(len(acts))})
    share = counts / max(n_ticks, 1)
    # the body's per-joint tau_max and gear table (wrist bends may differ)
    tmax = TAU_MAX.copy()
    gear = np.tile(np.abs(GEAR), (29, 1))
    for j in _c.WRIST_BEND_JOINTS:
        tmax[j] *= body["wrist_bend_tau_mult"]
        if body["wrist_bend_half_gear"]:
            gear[j, 3:5] = 0.5
    lim = tmax * np.asarray([tau_scale[TAU_AXIS_OF_JOINT[j]] for j in range(29)])
    mean_tau = (share * gear).sum(axis=1) * lim
    dead = int((share < 0.01).sum())
    groups = {}
    for g, js in GROUPS.items():
        js = list(js)
        s = share[js]
        groups[g] = {"coast": float(s[:, 0].mean()),
                     "full": float(s[:, 1:3].sum(axis=1).mean()),
                     "quarter": float(s[:, 3:5].sum(axis=1).mean()),
                     "mean_tau_sum": float(mean_tau[js].sum())}
    tele /= max(n_ticks, 1)
    out = {"label": a.label, "ticks": n_ticks, "files": per_file,
           "tau_scale": tau_scale, "env_body": body, "dead_cells": dead,
           "joints": [{"name": JOINTS[j], "share": share[j].tolist(),
                       "tau_max": float(tmax[j]), "tau_lim": float(lim[j]),
                       "mean_tau": float(mean_tau[j])} for j in range(29)],
           "groups": groups,
           "gear_total_tau": float(mean_tau.sum()),
           "telemetry_tau": tele.tolist()}
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print(f"{a.label}: {n_ticks:,} ticks, {dead} of 145 cells < 1%; gear-derived "
          f"sum|tau| {mean_tau.sum():.1f} N·m/tick vs telemetry {tele.round(1).tolist()}")
    for g, v in groups.items():
        print(f"  {g:7s} coast {v['coast']:.1%} full {v['full']:.1%} quarter "
              f"{v['quarter']:.1%}  sum|tau| {v['mean_tau_sum']:.1f}")


if __name__ == "__main__":
    main()
