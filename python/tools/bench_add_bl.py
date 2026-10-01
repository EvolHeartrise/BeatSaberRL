"""BeatLeader view of an existing ScoreSaber-only bench (no re-play).

    python tools/bench_add_bl.py --out-dir DIR BENCH.json [BENCH.json ...]

For every played level that BeatLeader also ranks (same song hash,
difficulty and Standard characteristic in maps/beatleader_catalog.json),
the SAME play is priced with BeatLeader's rules — bench_scoresaber.
bl_result: accuracy = score / BeatLeader's maxScore, BL pp from the
pass / acc / tech ratings, 0 pp for a failed play — and written to a
sidecar DIR/<bench basename>.json; the bench file itself is untouched.

Failed play: these benches predate the B09 `fell` / `terminated_tick`
fields, so a cut-short episode is inferred from unjudged notes
(good + bad + misses < n_notes) — on the three benches that do record
termination it matches `terminated_tick is not None` exactly (98/98 cut
levels, 0 false positives, 2026-10-01). Levels BeatLeader ranks but the
bench never played (BeatLeader-only maps) are simply absent, so the BL
totals cover the shared maps only (`n_bl_played` of `n_bl_catalog`).
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_scoresaber as bs  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BL_CATALOG = os.path.join(HERE, "maps", "beatleader_catalog.json")


def _key(h, diff, char="Standard"):
    return (h.lower(), diff, char or "Standard")


def bl_sidecar(bench, catalog):
    blk = {_key(e["hash"], e["difficulty"], e.get("characteristic")): e
           for e in catalog}
    out_lv = []
    for l in bench["levels"]:
        if "score" not in l or "song_hash" not in l:
            continue                                  # skip records
        e = blk.get(_key(l["song_hash"], l["difficulty"]))
        if e is None:
            continue
        judged = l["good_cuts"] + l["bad_cuts"] + l["misses"]
        failed = (l.get("terminated_tick") is not None
                  if "terminated_tick" in l else judged < l["n_notes"])
        r = bs.bl_result(e, l["score"], l["max_score"], failed)
        out_lv.append({"ss_leaderboard_id": l["leaderboard_id"],
                       "failed_inferred": "terminated_tick" not in l,
                       "bl": r})
    pps = sorted((x["bl"]["pp"] for x in out_lv), reverse=True)
    return {
        "source": "bench_add_bl",
        "model_hash": bench["model_hash"],
        "tick_hz": bench.get("tick_hz", 30),
        "action_repeat": bench.get("action_repeat", 1),
        "obs_window": bench.get("obs_window"),
        "bl_catalog_sha256": hashlib.sha256(
            json.dumps(catalog, sort_keys=True).encode()).hexdigest()[:16],
        "n_bl_catalog": len(catalog),
        "n_bl_played": len(out_lv),
        "bl_total_weighted_pp": round(sum(p * 0.965 ** i for i, p in enumerate(pps)), 2),
        "bl_total_unweighted_pp": round(sum(pps), 2),
        "bl_failed_levels": sum(x["bl"]["failed"] for x in out_lv),
        "levels": out_lv,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--catalog", default=BL_CATALOG)
    p.add_argument("benches", nargs="+")
    a = p.parse_args(argv)
    with open(a.catalog, encoding="utf-8") as f:
        catalog = json.load(f)
    os.makedirs(a.out_dir, exist_ok=True)
    for path in a.benches:
        with open(path, encoding="utf-8") as f:
            bench = json.load(f)
        if bench.get("catalog") not in (None, "scoresaber"):
            print(f"skip {path}: catalog {bench.get('catalog')!r} already has BL")
            continue
        sc = bl_sidecar(bench, catalog)
        dst = os.path.join(a.out_dir, os.path.basename(path))
        with open(dst + ".tmp", "w", encoding="utf-8") as f:
            json.dump(sc, f)
        os.replace(dst + ".tmp", dst)
        print(f"{os.path.basename(path)}: BL {sc['bl_total_weighted_pp']:,.2f} wpp / "
              f"{sc['bl_total_unweighted_pp']:,.0f} upp over {sc['n_bl_played']} of "
              f"{sc['n_bl_catalog']} BL maps, {sc['bl_failed_levels']} failed")


if __name__ == "__main__":
    main()
