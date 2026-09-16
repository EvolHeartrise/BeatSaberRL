"""Compare two bench results files level by level.

    python bench_compare.py A.json B.json [--worst 10]

Per-level join on leaderboard_id; prints both totals, the summed absolute
pp difference, how many levels differ in ANY scored field, the largest
accuracy difference, which fields ever differ, and the worst diffs.
Written for the CUDA-sim bench validation (2026-09-16): A = candidate,
B = reference (the Rust bench)."""
import argparse
import json

FIELDS = ("score", "accuracy", "pp", "good_cuts", "bad_cuts", "misses",
          "bombs_hit", "highest_combo", "full_combo", "score_pre",
          "score_acc", "score_post", "flail_rad", "flail_ticks")


def load(path):
    d = json.load(open(path))
    lv = {e["leaderboard_id"]: e for e in d.get("levels", [])
          if "skipped" not in e}
    return d, lv


def compare(a_path, b_path, worst=10, out=print):
    da, la = load(a_path)
    db, lb = load(b_path)
    common = sorted(set(la) & set(lb))
    out(f"A: {a_path}  sim={da.get('sim', 'rust')}  levels {len(la)}  "
        f"upp {da.get('total_unweighted_pp')}  wpp {da.get('total_weighted_pp')}")
    out(f"B: {b_path}  sim={db.get('sim', 'rust')}  levels {len(lb)}  "
        f"upp {db.get('total_unweighted_pp')}  wpp {db.get('total_weighted_pp')}")
    out(f"common levels: {len(common)}  (A-only {len(set(la) - set(lb))}, "
        f"B-only {len(set(lb) - set(la))})")
    diff_levels = 0
    field_diff = {f: 0 for f in FIELDS}
    sum_abs_dpp = 0.0
    max_dacc = 0.0
    rows = []
    for k in common:
        ea, eb = la[k], lb[k]
        d = {f: (ea.get(f), eb.get(f)) for f in FIELDS
             if _ne(ea.get(f), eb.get(f))}
        dpp = float(ea["pp"]) - float(eb["pp"])
        dacc = float(ea["accuracy"]) - float(eb["accuracy"])
        sum_abs_dpp += abs(dpp)
        max_dacc = max(max_dacc, abs(dacc))
        if d:
            diff_levels += 1
            for f in d:
                field_diff[f] += 1
            rows.append((abs(dpp), k, ea, dpp, dacc, d))
    out(f"levels with any scored field differing: {diff_levels} of "
        f"{len(common)} ({100.0 * diff_levels / max(1, len(common)):.2f}%)")
    out(f"sum |dpp| {sum_abs_dpp:,.1f} | net dpp (A-B) "
        f"{sum(float(la[k]['pp']) - float(lb[k]['pp']) for k in common):+,.1f} "
        f"| max |dacc| {100 * max_dacc:.3f} pt")
    out("fields differing (levels): " + ", ".join(
        f"{f} {n}" for f, n in field_diff.items() if n))
    rows.sort(reverse=True)
    for _, k, ea, dpp, dacc, d in rows[:worst]:
        out(f"  {ea['song'][:28]!r:30s} {ea['difficulty'][:6]:6s} "
            f"{ea['stars']:5.2f}* dpp {dpp:+7.1f} dacc {100 * dacc:+.3f} pt | "
            + " ".join(f"{f}={va}/{vb}" for f, (va, vb) in d.items()))
    return {"common": len(common), "diff_levels": diff_levels,
            "sum_abs_dpp": sum_abs_dpp, "max_dacc": max_dacc,
            "fields": field_diff}


def _ne(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) > 1e-9
        except (TypeError, ValueError):
            return a != b
    return a != b


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--worst", type=int, default=10)
    a = p.parse_args()
    compare(a.a, a.b, a.worst)
