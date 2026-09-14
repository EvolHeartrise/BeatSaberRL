"""Training exposure vs bench pp weight.

The ranked pool (ranked_pool.py) slices every catalog level into 25 s
windows (stride 23.5 s at lead 1.5) and the CUDA env draws a window
UNIFORMLY AT RANDOM per reset (cuda_env._cand -> torch.randint).  So a
level's training share is ~ its number of windows ~ duration, and a
note's training share is ~ uniform over notes (every note in exactly one
window).  The bench weights a level by max_pp * curve(acc) - i.e. by
stars and by how close to 100% the play is.  Compare the two.

usage: python3 train_vs_bench_weight.py <bench.json>
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ppcurve import pp_multiplier, PP_CURVE  # noqa: E402


def window_starts(last, seg_len, lead_s=1.5):
    """Copy of ranked_pool.window_starts (pure; the package __init__ needs gymnasium)."""
    stride = seg_len - lead_s
    if last < lead_s:
        return []
    n = int(np.floor((last - lead_s + 1e-6) / stride)) + 1
    return [k * stride for k in range(n)]


FEATS = json.load(open(os.path.join(HERE, "level_features.json")))


def main(path):
    b = json.load(open(path)); L = b["levels"]
    rows = []
    for e in L:
        f = FEATS[f"{e['song_hash']}_{e['difficulty']}"]
        n_win = len(window_starts(f["dur_s"] + 2.0, 25.0, 1.5))   # last object ≈ dur + 2 s parser shift
        pp = e["pp"]; mpp = e["max_pp"]
        eps = 1e-4
        slope = (pp_multiplier(min(e["accuracy"] + eps, 1)) - pp_multiplier(e["accuracy"] - eps)) / (2 * eps) * mpp
        rows.append(dict(e=e, f=f, n_win=n_win, pp=pp, headroom=PP_CURVE[0][1] * mpp - pp,
                         marg=slope * 0.005))          # pp for +0.5 pt on this level
    tw = sum(r["n_win"] for r in rows); tn = sum(r["e"]["n_notes"] for r in rows)
    tpp = sum(r["pp"] for r in rows); th = sum(r["headroom"] for r in rows); tm = sum(r["marg"] for r in rows)
    print(f"windows {tw} (segments in the pool at seg 25 / lead 1.5), notes {tn}, levels {len(rows)}")
    print("\n| stars | levels | train share: episodes (windows) | train share: notes | bench share: pp | "
          "bench share: headroom to 100% | bench share: pp for +0.5 pt acc | bench/train ratio (pp / notes) |")
    print("|---|" * 8)
    bins = [0, 2, 4, 6, 8, 10, 12, 14, 20]
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = [r for r in rows if lo <= r["e"]["stars"] < hi]
        if not s:
            continue
        w = sum(r["n_win"] for r in s) / tw; nn = sum(r["e"]["n_notes"] for r in s) / tn
        p = sum(r["pp"] for r in s) / tpp; h = sum(r["headroom"] for r in s) / th; m = sum(r["marg"] for r in s) / tm
        print(f"| {lo}–{hi} | {len(s)} | {100*w:.1f}% | {100*nn:.1f}% | {100*p:.1f}% | {100*h:.1f}% | {100*m:.1f}% | {p/max(nn,1e-9):.2f} |")
    print("\nSame by NJS:")
    print("| njs | levels | train notes | bench pp | bench +0.5pt | ratio |")
    print("|---|---|---|---|---|---|")
    for lo, hi in zip([0, 12, 14, 16, 18, 20, 22, 24], [12, 14, 16, 18, 20, 22, 24, 40]):
        s = [r for r in rows if lo <= r["f"]["njs"] < hi]
        if not s:
            continue
        nn = sum(r["e"]["n_notes"] for r in s) / tn; p = sum(r["pp"] for r in s) / tpp; m = sum(r["marg"] for r in s) / tm
        print(f"| {lo}–{hi} | {len(s)} | {100*nn:.1f}% | {100*p:.1f}% | {100*m:.1f}% | {p/max(nn,1e-9):.2f} |")
    print("\nSame by NPS:")
    print("| nps | levels | train notes | bench pp | bench +0.5pt | ratio |")
    print("|---|---|---|---|---|---|")
    for lo, hi in zip([0, 2, 3, 4, 5, 6, 7, 8, 10], [2, 3, 4, 5, 6, 7, 8, 10, 30]):
        s = [r for r in rows if lo <= r["f"]["nps"] < hi]
        if not s:
            continue
        nn = sum(r["e"]["n_notes"] for r in s) / tn; p = sum(r["pp"] for r in s) / tpp; m = sum(r["marg"] for r in s) / tm
        print(f"| {lo}–{hi} | {len(s)} | {100*nn:.1f}% | {100*p:.1f}% | {100*m:.1f}% | {p/max(nn,1e-9):.2f} |")
    # effective sample size / how many times each level is seen per update at 32768 envs, 25 s windows
    print(f"\nAt N=32,768 envs, T=32 ticks (1.07 s) per update and ~25 s episodes, ~{32768*32/(25*30):.0f} "
          f"episodes start per update -> each of the {tw} windows is drawn once every ~{tw/(32768*32/(25*30)):.1f} updates; "
          f"over 1,068 updates each window is played ~{1068*(32768*32/(25*30))/tw:.0f} times.")
    # top-pp-share levels
    print("\nTop 10 levels by bench pp:")
    for r in sorted(rows, key=lambda r: -r["pp"])[:10]:
        e = r["e"]
        print(f"  {e['song'][:30]:30} {e['difficulty'][:6]:6} {e['stars']:5.2f}★ acc {e['accuracy']:.4f} pp {e['pp']:.0f} windows {r['n_win']}")
    # pp concentration
    pps = np.sort([r["pp"] for r in rows])[::-1]
    c = np.cumsum(pps) / pps.sum()
    for q in (100, 300, 1000, 2000):
        print(f"top {q} levels carry {100*c[q-1]:.1f}% of upp")
    # wpp: 0.965^rank weighting
    w = 0.965 ** np.arange(len(pps))
    print(f"wpp = {(pps*w).sum():.0f}; top 100 levels carry {100*(pps*w)[:100].sum()/(pps*w).sum():.1f}% of wpp")


if __name__ == "__main__":
    main(sys.argv[1])
