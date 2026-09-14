"""What is +x points of the 15-pt accuracy component per good cut worth
on the bench (upp / wpp), overall and per star bin?  Applies a uniform
+x to score_acc on every level (capped at 15 per cut), recomputes the
multiplied score assuming the level's own combo structure (score scales
with unmultiplied points), and re-evaluates the pp curve.

usage: python3 acc_component_value.py <bench.json> [x ...]
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ppcurve import pp_multiplier, PP_DECAY


def total(levels, dx=0.0, stars=(0, 99), dq=0.0):
    pps = []
    for e in levels:
        g = e["good_cuts"]; U = e["score_pre"] + e["score_acc"] + e["score_post"]
        if stars[0] <= e["stars"] < stars[1] and g:
            gain = min(15.0 * g, e["score_acc"] + dx * g) - e["score_acc"]
            gain += dq * g
            new_score = e["score"] * (U + gain) / U if U else e["score"]
        else:
            new_score = e["score"]
        acc = min(new_score / e["max_score"], 1.0)
        pps.append(pp_multiplier(acc) * e["max_pp"])
    pps = np.array(pps)
    w = PP_DECAY ** np.arange(len(pps))
    return pps.sum(), (np.sort(pps)[::-1] * w).sum()


if __name__ == "__main__":
    L = json.load(open(sys.argv[1]))["levels"]
    base_u, base_w = total(L)
    print(f"base upp {base_u:.0f} wpp {base_w:.0f}")
    print("| +acc pts per good cut (all levels) | upp | Δupp | wpp | Δwpp |")
    print("|---|---|---|---|---|")
    for dx in [0.25, 0.5, 1.0, 1.5, 2.0, 3.13]:
        u, w = total(L, dx)
        print(f"| +{dx:g} | {u:.0f} | {u-base_u:+.0f} | {w:.0f} | {w-base_w:+.0f} |")
    print("\n+1.0 acc pt per cut applied to ONE star bin only:")
    print("| stars | Δupp | Δwpp | notes share |")
    print("|---|---|---|---|")
    tn = sum(e["n_notes"] for e in L)
    for lo, hi in zip([0, 2, 4, 6, 8, 10, 12, 14], [2, 4, 6, 8, 10, 12, 14, 20]):
        u, w = total(L, 1.0, (lo, hi))
        nn = sum(e["n_notes"] for e in L if lo <= e["stars"] < hi)
        print(f"| {lo}–{hi} | {u-base_u:+.0f} | {w-base_w:+.0f} | {100*nn/tn:.1f}% |")
    print("\nfor scale: post-swing +0.22/cut (to 30.0 on every plain note) and pre +0.06:")
    u, w = total(L, 0.0, dq=0.28)
    print(f"  Δupp {u-base_u:+.0f}")
