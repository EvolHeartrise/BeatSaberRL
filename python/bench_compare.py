"""Compare two bench results files level by level.

    python bench_compare.py A.json B.json [--worst 10] [--force]

Per-level join on leaderboard_id; prints both totals, the summed absolute
pp difference, how many levels differ in ANY scored field, the largest
accuracy difference, which fields ever differ, and the worst diffs.
Written for the CUDA-sim bench validation (2026-09-16): A = candidate,
B = reference (the Rust bench).

Header check (review B14, 2026-09-30): the two files must agree on the
run regime — tau_scale (the env v46 body), mirror, scoring_version and
sim_dtype — or the per-level diffs measure the regime change, not the
thing under test. A mismatch is an error; --force compares anyway (and
says so). `sim` itself is deliberately NOT checked: cuda-vs-rust is the
comparison this tool exists for.

Catalog (2026-09-30, --catalog scoresaber|beatleader|both): files of
different catalogs are REFUSED outright, --force or not — their level
identities (ScoreSaber id / BeatLeader id / hash_difficulty_mode key)
and pp scales differ, so there is no per-level join to make. Files
without the stamp are scoresaber. beatleader files join on the BL
leaderboard id and compare BL pp; `both` files join on level_key and
compare ss_pp and bl_pp per level; the BL totals print whenever both
files carry them.

Termination telemetry (review B09): when both files carry the per-level
breaks / wall_enters / fell / envelope / terminated_tick fields, their
totals over the common levels are printed and they join the per-field
diff; files that predate them are compared on the old fields only."""
import argparse
import json

FIELDS = ("score", "accuracy", "pp", "good_cuts", "bad_cuts", "misses",
          "bombs_hit", "highest_combo", "full_combo", "score_pre",
          "score_acc", "score_post", "flail_rad", "flail_ticks")
# B09 fields: compared only where BOTH entries carry them (older files
# lack them; `fell` None = an engine that could not report them)
TERM_FIELDS = ("breaks", "wall_enters", "fell", "tilted", "envelope",
               "terminated_tick")


class HeaderMismatch(SystemExit):
    """The two files were benched under different regimes (B14)."""


class CatalogMismatch(HeaderMismatch):
    """The two files bench different catalogs (never --force-able)."""


def catalog_of(d):
    return d.get("catalog", "scoresaber")


def _key(e):
    return e.get("level_key", e.get("leaderboard_id"))


def _flat(e):
    """A `both` record's per-leaderboard pp as flat ss_pp / bl_pp fields
    (None where that leaderboard does not rank the level)."""
    if "level_key" not in e:
        return e
    f = dict(e)
    f["ss_pp"] = e["ss"]["pp"] if e.get("ss") else None
    f["bl_pp"] = e["bl"]["pp"] if e.get("bl") else None
    return f


def load(path):
    d = json.load(open(path))
    lv = {_key(e): _flat(e) for e in d.get("levels", [])
          if "skipped" not in e}
    return d, lv


def check_catalogs(da, db, out=print):
    ca, cb = catalog_of(da), catalog_of(db)
    if ca != cb:
        out(f"error: catalog differs: A={ca!r} B={cb!r}")
        raise CatalogMismatch(
            f"bench_compare: the files bench different catalogs ({ca} vs "
            f"{cb}) — there is no per-level join across leaderboards "
            f"(--force does not override this)")
    return ca


def _pp(e):
    """The primary per-level pp: the file's leaderboard; `both` = SS
    where ranked, else BL."""
    if "pp" in e:
        return float(e["pp"])
    return float(e["ss_pp"] if e.get("ss_pp") is not None else e["bl_pp"])


def _header(d, has_levels):
    """The regime keys, normalized the way bench_scoresaber reads them:
    a file without tau_scale predates env v46 (nominal body, all 1.0);
    without scoring_version but WITH results it is v1; without
    sim_dtype it is f64 for rust, f32 for cuda; without mirror, False."""
    ts = d.get("tau_scale")
    if ts is None or all(abs(float(x) - 1.0) < 1e-12 for x in ts):
        ts = "nominal"
    else:
        ts = tuple(round(float(x), 9) for x in ts)
    sim = d.get("sim", "rust")
    return {"tau_scale": ts, "mirror": bool(d.get("mirror", False)),
            "scoring_version": d.get("scoring_version",
                                     1 if has_levels else None),
            "sim_dtype": d.get("sim_dtype",
                               "f64" if sim == "rust" else "f32")}


def check_headers(da, db, has_a, has_b, force=False, out=print):
    """Names of the regime keys that differ; raises HeaderMismatch on
    any unless force (then each is printed as a WARNING)."""
    ha, hb = _header(da, has_a), _header(db, has_b)
    bad = [k for k in ha if ha[k] != hb[k]]
    for k in bad:
        out(f"{'WARNING' if force else 'error'}: header {k} differs: "
            f"A={ha[k]!r} B={hb[k]!r}")
    if bad and not force:
        raise HeaderMismatch(
            f"bench_compare: {', '.join(bad)} differ between the files — "
            f"the diff would measure the regime change; pass --force to "
            f"compare anyway")
    return bad


def _has_term(e):
    return e.get("fell") is not None


def compare(a_path, b_path, worst=10, out=print, force=False):
    da, la = load(a_path)
    db, lb = load(b_path)
    cat = check_catalogs(da, db, out)
    mismatched = check_headers(da, db, bool(la), bool(lb), force, out)
    common = sorted(set(la) & set(lb))
    out(f"catalog: {cat}")
    out(f"A: {a_path}  sim={da.get('sim', 'rust')}  levels {len(la)}  "
        f"upp {da.get('total_unweighted_pp')}  wpp {da.get('total_weighted_pp')}")
    out(f"B: {b_path}  sim={db.get('sim', 'rust')}  levels {len(lb)}  "
        f"upp {db.get('total_unweighted_pp')}  wpp {db.get('total_weighted_pp')}")
    bl_tot = None
    if "bl_total_weighted_pp" in da and "bl_total_weighted_pp" in db:
        bl_tot = {k: (da[k], db[k]) for k in ("bl_total_unweighted_pp",
                                              "bl_total_weighted_pp")}
        out(f"BeatLeader totals (A / B): upp {da['bl_total_unweighted_pp']} "
            f"/ {db['bl_total_unweighted_pp']}  wpp "
            f"{da['bl_total_weighted_pp']} / {db['bl_total_weighted_pp']}")
    if cat == "both":
        out(f"ScoreSaber totals (A / B): upp {da.get('ss_total_unweighted_pp')}"
            f" / {db.get('ss_total_unweighted_pp')}  wpp "
            f"{da.get('ss_total_weighted_pp')} / "
            f"{db.get('ss_total_weighted_pp')}")
    fields_base = (tuple(f for f in FIELDS if f != "pp") + ("ss_pp", "bl_pp")
                   if cat == "both" else FIELDS)
    out(f"common levels: {len(common)}  (A-only {len(set(la) - set(lb))}, "
        f"B-only {len(set(lb) - set(la))})")
    diff_levels = 0
    field_diff = {f: 0 for f in fields_base + TERM_FIELDS}
    sum_abs_dpp = 0.0
    max_dacc = 0.0
    rows = []
    term_common = [k for k in common if _has_term(la[k]) and _has_term(lb[k])]
    term_set = set(term_common)
    for k in common:
        ea, eb = la[k], lb[k]
        fields = fields_base + (TERM_FIELDS if k in term_set else ())
        d = {f: (ea.get(f), eb.get(f)) for f in fields
             if _ne(ea.get(f), eb.get(f))}
        dpp = _pp(ea) - _pp(eb)
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
        f"{sum(_pp(la[k]) - _pp(lb[k]) for k in common):+,.1f} "
        f"| max |dacc| {100 * max_dacc:.3f} pt")
    term = None
    if term_common:
        term = {"levels": len(term_common),
                "A": term_totals([la[k] for k in term_common]),
                "B": term_totals([lb[k] for k in term_common])}
        ta, tb = term["A"], term["B"]
        out(f"termination telemetry over {len(term_common)} common levels "
            f"(A / B): " + " | ".join(f"{f} {ta[f]} / {tb[f]}" for f in ta))
    elif common:
        out("termination telemetry: not in both files (older bench JSON) "
            "- skipped")
    out("fields differing (levels): " + ", ".join(
        f"{f} {n}" for f, n in field_diff.items() if n))
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    for _, k, ea, dpp, dacc, d in rows[:worst]:
        stars = ea.get("stars")
        if stars is None:
            stars = (ea.get("ss_stars") if ea.get("ss_stars") is not None
                     else ea.get("bl_stars"))
        out(f"  {ea['song'][:28]!r:30s} {ea['difficulty'][:6]:6s} "
            f"{stars:5.2f}* dpp {dpp:+7.1f} dacc {100 * dacc:+.3f} pt | "
            + " ".join(f"{f}={va}/{vb}" for f, (va, vb) in d.items()))
    return {"common": len(common), "diff_levels": diff_levels,
            "sum_abs_dpp": sum_abs_dpp, "max_dacc": max_dacc,
            "fields": field_diff, "term": term,
            "header_mismatch": mismatched, "catalog": cat,
            "bl_totals": bl_tot}


def term_totals(levels):
    """B09 totals over entries that carry the fields."""
    return {
        "breaks": sum(int(e.get("breaks", 0)) for e in levels),
        "wall_enters": sum(int(e.get("wall_enters") or 0) for e in levels),
        "fell": sum(bool(e.get("fell")) for e in levels),
        "tilted": sum(bool(e.get("tilted")) for e in levels),
        "envelope": sum(bool(e.get("envelope")) for e in levels),
        "terminated": sum(e.get("terminated_tick") is not None
                          for e in levels),
    }


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
    p.add_argument("--force", action="store_true",
                   help="compare even if tau_scale / mirror / "
                        "scoring_version / sim_dtype differ (never "
                        "across catalogs)")
    a = p.parse_args()
    compare(a.a, a.b, a.worst, force=a.force)
