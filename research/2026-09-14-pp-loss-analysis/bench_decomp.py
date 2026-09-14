"""Where is the bench pp lost?  Exact score-deficit decomposition per level
from a bench JSON + level_features.json, converted to pp through the
official ScoreSaber curve (counterfactual per category), then binned by
stars / NPS / NJS / duration / difficulty / mapper, with a linear fit of
accuracy on chart features and its outliers.

usage: python3 bench_decomp.py <bench.json> [--ref <other_bench.json>]
"""
import argparse, collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ppcurve import pp_multiplier, PP_CURVE  # noqa: E402

FEATS = json.load(open(os.path.join(HERE, "level_features.json")))
MULT = 8.0
PP_MAX_MULT = PP_CURVE[0][1]


def pp_of(acc, max_pp):
    return pp_multiplier(min(acc, 1.0)) * max_pp


def decompose(e):
    """Exact split of max_score - score into score-point categories."""
    f = FEATS[f"{e['song_hash']}_{e['difficulty']}"]
    n = e["n_notes"]; good, bad, miss = e["good_cuts"], e["bad_cuts"], e["misses"]
    unplayed = n - good - bad - miss
    U = e["score_pre"] + e["score_acc"] + e["score_post"]
    mpn = f["max_unmult"] / n                     # max unmultiplied pts / note
    ramp_fc = MULT * f["max_unmult"] - e["max_score"]   # FC ramp deficit (exact)
    d = {}
    d["unplayed"] = MULT * mpn * unplayed
    d["missbad"] = MULT * mpn * (miss + bad)
    d["combo"] = max(0.0, (MULT * U - ramp_fc * (U / max(f["max_unmult"], 1))) - e["score"])
    q = MULT * (mpn * good - U)                     # cut-quality loss on good cuts
    # split quality by component using the level's per-kind maxima
    gfrac = good / max(n, 1)
    lp = MULT * (f["max_pre"] * gfrac - e["score_pre"])
    la = MULT * (f["max_acc"] * gfrac - e["score_acc"])
    lo = MULT * (f["max_post"] * gfrac - e["score_post"])
    s = lp + la + lo
    if s > 0:
        d["q_pre"], d["q_acc"], d["q_post"] = q * lp / s, q * la / s, q * lo / s
    else:
        d["q_pre"] = d["q_acc"] = d["q_post"] = q / 3
    d["deficit"] = e["max_score"] - e["score"]
    # combo attribution among break causes. The bench does not count wall
    # head-entries, but they break combo (lib.rs combo walk, code 8), so
    # estimate the break count from the combo loss itself: one isolated
    # break at 8x costs 4 x mpn x 8 recovery notes = 32 mpn (multiplier
    # halves to 4x, 8 good cuts to regain 8x); breaks beyond the observed
    # miss+bad+bomb count are attributed to walls.
    breaks = miss + bad + e["bombs_hit"]
    per_break = 32.0 * mpn
    est_breaks = d["combo"] / per_break if per_break > 0 else 0.0
    wall_breaks = max(0.0, est_breaks - breaks)
    d["est_breaks"] = est_breaks; d["wall_breaks_est"] = wall_breaks
    tot = breaks + wall_breaks
    if tot > 0:
        d["combo_wall"] = d["combo"] * wall_breaks / tot
        d["combo_bomb"] = d["combo"] * e["bombs_hit"] / tot
        d["combo_missbad"] = d["combo"] - d["combo_wall"] - d["combo_bomb"]
    else:
        d["combo_wall"] = d["combo_bomb"] = d["combo_missbad"] = 0.0
    return d, f


CATS = ["unplayed", "missbad", "combo_missbad", "combo_bomb", "combo_wall", "q_pre", "q_acc", "q_post"]


def analyze(path, tag):
    b = json.load(open(path))
    L = b["levels"]
    rows = []
    for e in L:
        d, f = decompose(e)
        acc = e["accuracy"]; mpp = e["max_pp"]
        pp = e["pp"]
        cf = {c: pp_of((e["score"] + d[c]) / e["max_score"], mpp) - pp for c in CATS}
        # first-order marginal: slope of the curve at the current acc
        eps = 1e-4
        slope = (pp_of(acc + eps, mpp) - pp_of(acc - eps, mpp)) / (2 * eps)
        fo = {c: slope * d[c] / e["max_score"] for c in CATS}
        rows.append(dict(e=e, f=f, d=d, cf=cf, fo=fo, slope=slope,
                         headroom=PP_MAX_MULT * mpp - pp))
    tot_def = sum(r["d"]["deficit"] for r in rows)
    print(f"\n# {tag}: {os.path.basename(path)}  levels {len(L)}  upp {b['total_unweighted_pp']:.0f}  "
          f"wpp {b['total_weighted_pp']:.1f}  mean acc {np.mean([r['e']['accuracy'] for r in rows]):.4f}")
    print(f"score deficit {tot_def/1e6:.1f}M of {sum(r['e']['max_score'] for r in rows)/1e6:.0f}M "
          f"({100*tot_def/sum(r['e']['max_score'] for r in rows):.2f}%); headroom to 100% acc: "
          f"{sum(r['headroom'] for r in rows)/1e6:.3f}M pp")
    print("\n## Deficit decomposition (whole bench)")
    print("| category | score pts lost | share of deficit | pp (counterfactual, category zeroed) | pp (first-order) |")
    print("|---|---|---|---|---|")
    for c in CATS:
        s = sum(r["d"][c] for r in rows)
        print(f"| {c} | {s/1e6:.1f}M | {100*s/tot_def:.1f}% | {sum(r['cf'][c] for r in rows)/1e3:.0f}k | "
              f"{sum(r['fo'][c] for r in rows)/1e3:.0f}k |")
    print(f"| all | {tot_def/1e6:.1f}M | 100% | {sum(r['headroom'] for r in rows)/1e3:.0f}k | - |")
    print("\ncounts: good %d bad %d miss %d bombs %d unplayed %d FC %d; "
          "per-good-cut pre/acc/post %.2f / %.2f / %.2f (max 70/15/30 for plain notes)" % (
              sum(r["e"]["good_cuts"] for r in rows), sum(r["e"]["bad_cuts"] for r in rows),
              sum(r["e"]["misses"] for r in rows), sum(r["e"]["bombs_hit"] for r in rows),
              sum(r["e"]["n_notes"] - r["e"]["good_cuts"] - r["e"]["bad_cuts"] - r["e"]["misses"] for r in rows),
              sum(r["e"]["full_combo"] for r in rows),
              sum(r["e"]["score_pre"] for r in rows) / sum(r["e"]["good_cuts"] for r in rows),
              sum(r["e"]["score_acc"] for r in rows) / sum(r["e"]["good_cuts"] for r in rows),
              sum(r["e"]["score_post"] for r in rows) / sum(r["e"]["good_cuts"] for r in rows)))

    def table(key, bins, label):
        print(f"\n## By {label}")
        print("| bin | levels | notes | mean acc | pp | pp/level | headroom pp | "
              + " | ".join(f"cf {c}" for c in CATS) + " |")
        print("|---|" * (7 + len(CATS)) + "|")
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = [r for r in rows if lo <= key(r) < hi]
            if not sel:
                continue
            print(f"| {lo:g}–{hi:g} | {len(sel)} | {sum(r['e']['n_notes'] for r in sel)/1e3:.0f}k | "
                  f"{np.mean([r['e']['accuracy'] for r in sel]):.4f} | {sum(r['e']['pp'] for r in sel)/1e3:.0f}k | "
                  f"{sum(r['e']['pp'] for r in sel)/len(sel):.0f} | {sum(r['headroom'] for r in sel)/1e3:.0f}k | "
                  + " | ".join(f"{sum(r['cf'][c] for r in sel)/1e3:.0f}k" for c in CATS) + " |")

    table(lambda r: r["e"]["stars"], [0, 2, 4, 6, 8, 10, 12, 14, 20], "stars")
    table(lambda r: r["f"]["nps"], [0, 2, 3, 4, 5, 6, 7, 8, 10, 30], "NPS (scoreable notes / note-span s)")
    table(lambda r: r["f"]["peak_nps5"], [0, 4, 6, 8, 10, 12, 14, 30], "peak NPS (densest 5 s)")
    table(lambda r: r["f"]["njs"], [0, 12, 14, 16, 18, 20, 22, 24, 40], "NJS (median note speed m/s)")
    table(lambda r: r["e"]["duration_s"], [0, 60, 120, 180, 240, 300, 400, 700], "duration (s)")

    print("\n## By difficulty label")
    print("| difficulty | levels | mean stars | mean acc | pp | headroom | cf unplayed | cf missbad | cf q_acc |")
    print("|---|---|---|---|---|---|---|---|---|")
    for dif in ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]:
        sel = [r for r in rows if r["e"]["difficulty"] == dif]
        print(f"| {dif} | {len(sel)} | {np.mean([r['e']['stars'] for r in sel]):.2f} | "
              f"{np.mean([r['e']['accuracy'] for r in sel]):.4f} | {sum(r['e']['pp'] for r in sel)/1e3:.0f}k | "
              f"{sum(r['headroom'] for r in sel)/1e3:.0f}k | {sum(r['cf']['unplayed'] for r in sel)/1e3:.0f}k | "
              f"{sum(r['cf']['missbad'] for r in sel)/1e3:.0f}k | {sum(r['cf']['q_acc'] for r in sel)/1e3:.0f}k |")

    print("\n## Cut-quality components per good cut, by stars (max 70 / 15 / 30)")
    print("| stars | good cuts | pre | acc | post | acc as % of 15 | good-cut share of notes |")
    print("|---|---|---|---|---|---|---|")
    for lo, hi in zip([0, 2, 4, 6, 8, 10, 12, 14], [2, 4, 6, 8, 10, 12, 14, 20]):
        s = [r for r in rows if lo <= r["e"]["stars"] < hi]
        g = sum(r["e"]["good_cuts"] for r in s)
        if not g:
            continue
        print(f"| {lo}–{hi} | {g/1e3:.0f}k | {sum(r['e']['score_pre'] for r in s)/g:.2f} | "
              f"{sum(r['e']['score_acc'] for r in s)/g:.2f} | {sum(r['e']['score_post'] for r in s)/g:.2f} | "
              f"{100*sum(r['e']['score_acc'] for r in s)/g/15:.1f}% | {100*g/sum(r['e']['n_notes'] for r in s):.3f}% |")
    print("\n## Wall-break estimate (combo loss beyond observed miss/bad/bomb breaks)")
    wl = sorted(rows, key=lambda r: -r["cf"]["combo_wall"])
    print(f"levels with >= 1 estimated wall break: {sum(1 for r in rows if r['d']['wall_breaks_est'] >= 1)}; "
          f"estimated wall breaks total {sum(r['d']['wall_breaks_est'] for r in rows):.0f}; "
          f"cf pp {sum(r['cf']['combo_wall'] for r in rows)/1e3:.0f}k; levels with walls in chart: "
          f"{sum(1 for r in rows if r['f']['n_walls'] > 0)}")
    print("| song | mapper | diff | stars | acc | pp | walls in chart | est breaks | obs breaks | cf combo_wall pp |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in wl[:15]:
        e = r["e"]
        print(f"| {e['song'][:28]} | {e['mapper'][:14]} | {e['difficulty'][:6]} | {e['stars']:.2f} | {e['accuracy']:.4f} | "
              f"{e['pp']:.0f} | {r['f']['n_walls']} | {r['d']['est_breaks']:.0f} | {e['misses']+e['bad_cuts']+e['bombs_hit']} | "
              f"{r['cf']['combo_wall']:.0f} |")

    # ---------- regression of accuracy on chart features ----------
    print("\n## Linear fit: accuracy ~ features (levels with >= 50 notes, no early termination)")
    sel = [r for r in rows if r["e"]["n_notes"] >= 50 and r["d"]["unplayed"] == 0]
    names = ["stars", "nps", "peak_nps5", "njs", "log_dur", "frac_top", "frac_mid", "frac_double",
             "frac_cross", "frac_dot", "bomb_frac", "wall_per_min"]
    X = np.array([[r["e"]["stars"], r["f"]["nps"], r["f"]["peak_nps5"], r["f"]["njs"],
                   np.log(max(r["e"]["duration_s"], 1)), r["f"]["frac_top"], r["f"]["frac_mid"],
                   r["f"]["frac_double"], r["f"]["frac_cross"], r["f"]["frac_dot"],
                   r["f"]["n_bomb"] / max(r["f"]["n_sc"], 1),
                   60 * r["f"]["n_walls"] / max(r["e"]["duration_s"], 1)] for r in sel])
    y = np.array([r["e"]["accuracy"] for r in sel])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    pred = Z @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"n = {len(sel)}, R^2 = {r2:.3f}, intercept {beta[0]:.4f}; std-coefficients (Δacc per 1 sd of feature):")
    print("| feature | mean | sd | coef (pt) |")
    print("|---|---|---|---|")
    for i, nm in enumerate(names):
        print(f"| {nm} | {mu[i]:.3f} | {sd[i]:.3f} | {100*beta[i+1]:+.3f} |")
    # univariate for the two headline axes
    for i in (0, 1, 3):
        c = np.corrcoef(X[:, i], y)[0, 1]
        print(f"univariate corr(acc, {names[i]}) = {c:+.3f}")
    resid = y - pred
    print("\n### Worst residuals (accuracy below the fit; n_notes >= 50, fully played)")
    print("| song | mapper | diff | stars | njs | nps | acc | fit | resid (pt) | pp | headroom | miss+bad | bombs |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for j in np.argsort(resid)[:20]:
        r = sel[j]; e = r["e"]
        print(f"| {e['song'][:28]} | {e['mapper'][:14]} | {e['difficulty'][:6]} | {e['stars']:.2f} | {r['f']['njs']:.0f} | "
              f"{r['f']['nps']:.1f} | {e['accuracy']:.4f} | {pred[j]:.4f} | {100*resid[j]:+.2f} | {e['pp']:.0f} | "
              f"{r['headroom']:.0f} | {e['misses']+e['bad_cuts']} | {e['bombs_hit']} |")

    # ---------- top pp-loss levels ----------
    print("\n## Levels with the largest counterfactual pp loss (sum over categories)")
    print("| song | mapper | diff | stars | acc | pp | Σcf | unplayed | missbad | combo | q_acc | q_pre | q_post |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -sum(r["cf"].values()))[:25]:
        e = r["e"]; c = r["cf"]
        print(f"| {e['song'][:28]} | {e['mapper'][:14]} | {e['difficulty'][:6]} | {e['stars']:.2f} | {e['accuracy']:.4f} | "
              f"{e['pp']:.0f} | {sum(c.values()):.0f} | {c['unplayed']:.0f} | {c['missbad']:.0f} | "
              f"{c['combo_missbad']+c['combo_bomb']:.0f} | {c['q_acc']:.0f} | {c['q_pre']:.0f} | {c['q_post']:.0f} |")

    # ---------- mappers ----------
    print("\n## Mappers by counterfactual pp loss (>= 8 levels)")
    print("| mapper | levels | mean stars | mean acc | pp | Σcf | cf q_acc | cf missbad | cf unplayed | resid mean (pt) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    bym = collections.defaultdict(list)
    for r in rows:
        bym[r["e"]["mapper"]].append(r)
    res_by_level = {id(sel[j]["e"]): resid[j] for j in range(len(sel))}
    for m, rs in sorted(bym.items(), key=lambda kv: -sum(sum(r["cf"].values()) for r in kv[1]))[:20]:
        if len(rs) < 8:
            continue
        rr = [res_by_level[id(r["e"])] for r in rs if id(r["e"]) in res_by_level]
        print(f"| {m[:18]} | {len(rs)} | {np.mean([r['e']['stars'] for r in rs]):.2f} | "
              f"{np.mean([r['e']['accuracy'] for r in rs]):.4f} | {sum(r['e']['pp'] for r in rs)/1e3:.1f}k | "
              f"{sum(sum(r['cf'].values()) for r in rs)/1e3:.1f}k | {sum(r['cf']['q_acc'] for r in rs)/1e3:.1f}k | "
              f"{sum(r['cf']['missbad'] for r in rs)/1e3:.1f}k | {sum(r['cf']['unplayed'] for r in rs)/1e3:.1f}k | "
              f"{100*np.mean(rr) if rr else float('nan'):+.2f} |")

    # ---------- marginal value of accuracy per star bucket ----------
    print("\n## What +0.5 pt accuracy on every level would be worth, by stars (curve convexity)")
    print("| stars | levels | pp now | pp at +0.5 pt | Δ | Δ per level |")
    print("|---|---|---|---|---|---|")
    for lo, hi in zip([0, 2, 4, 6, 8, 10, 12, 14], [2, 4, 6, 8, 10, 12, 14, 20]):
        s = [r for r in rows if lo <= r["e"]["stars"] < hi]
        if not s:
            continue
        now = sum(r["e"]["pp"] for r in s)
        up = sum(pp_of(r["e"]["accuracy"] + 0.005, r["e"]["max_pp"]) for r in s)
        print(f"| {lo}–{hi} | {len(s)} | {now/1e3:.0f}k | {up/1e3:.0f}k | {(up-now)/1e3:.0f}k | {(up-now)/len(s):.1f} |")
    return rows


def compare(rows_a, rows_b, tag_a, tag_b):
    print(f"\n# Paired comparison {tag_a} vs {tag_b}")
    ka = {(r["e"]["leaderboard_id"]): r for r in rows_a}
    kb = {(r["e"]["leaderboard_id"]): r for r in rows_b}
    common = [k for k in ka if k in kb]
    print("| category | Δ score pts (a−b) | Δ cf pp |")
    print("|---|---|---|")
    for c in CATS:
        print(f"| {c} | {sum(ka[k]['d'][c]-kb[k]['d'][c] for k in common)/1e6:+.1f}M | "
              f"{sum(ka[k]['cf'][c]-kb[k]['cf'][c] for k in common)/1e3:+.0f}k |")
    dpp = np.array([ka[k]["e"]["pp"] - kb[k]["e"]["pp"] for k in common])
    print(f"Δpp total {dpp.sum():+.0f}; levels better/worse {int((dpp>0).sum())}/{int((dpp<0).sum())}; "
          f"|Δpp| > 50 on {int((np.abs(dpp)>50).sum())} levels; largest swings:")
    for k in sorted(common, key=lambda k: -abs(ka[k]["e"]["pp"] - kb[k]["e"]["pp"]))[:10]:
        a, b = ka[k]["e"], kb[k]["e"]
        print(f"  {a['song'][:30]:30} {a['difficulty'][:6]:6} {a['stars']:5.2f}★  acc {a['accuracy']:.4f} vs {b['accuracy']:.4f}"
              f"  pp {a['pp']:.0f} vs {b['pp']:.0f}  (unplayed {ka[k]['d']['unplayed']/8:.0f} vs {kb[k]['d']['unplayed']/8:.0f} notes-eq)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--ref", nargs="*", default=[])
    a = ap.parse_args()
    main_rows = analyze(a.bench, "MAIN")
    for p in a.ref:
        r = analyze(p, "REF")
        compare(main_rows, r, os.path.basename(a.bench), os.path.basename(p))
