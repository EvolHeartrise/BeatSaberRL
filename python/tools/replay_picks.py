"""Which plays of a best-of composite to record as replays, and from which
SOURCE bench each must be re-rolled.

    python tools/replay_picks.py COMPOSITE.json [--ss-top 10] [--bl-top 10]
        [--gap-top 5 --gap-scores THEIRS.json [--gap-name NAME]]
        [--plan OUT.json]

A composite (tools/bench_combine.py) picks each level's ScoreSaber play
and its BeatLeader play independently, possibly from different runs
(control rate, body). A replay must be recorded in its pick's own run, so
every selected play resolves to (level_key, source file): the `ss_from`
run for an SS selection, the `bl_from` run for a BL one.

Selections (each ranked within its category):
  ss   the top N levels by the composite's SS pp (ss.pp, SS pick)
  bl   the top N by BL pp (bl.pp, BL pick; BeatLeader-only levels count)
  gap  the top N BL-pp gaps over another player (our bl.pp minus their pp
       on the same BeatLeader leaderboard, BL pick), over the levels they
       have a score on. THEIRS.json: a list of {lb, pp, ...} with lb = the
       BeatLeader leaderboard id (their best pp per lb is used).

Plays that land on the same (level, source run) are one replay. --plan
writes {"replays": [...], "by_source": {source file: [level_key, ...]}}:
by_source feeds make_top10_replays.py --bench <source> --levels ...
"""
import argparse
import json
import sys

FROM = {"ss": "ss_from", "bl": "bl_from"}


def _sel(rec, board, cat, rank, extra=None):
    sub = rec[board]
    return {"category": cat, "rank": rank, "board": board,
            "level_key": rec["level_key"], "song": rec["song"],
            "difficulty": rec["difficulty"], "source": rec[FROM[board]],
            "pp": sub["pp"], "stars": sub["stars"],
            "ss_stars": (rec.get("ss") or {}).get("stars"),
            "bl_stars": (rec.get("bl") or {}).get("stars"),
            "ss_pp": (rec.get("ss") or {}).get("pp"),
            "bl_pp": (rec.get("bl") or {}).get("pp"),
            **(extra or {})}


def top_picks(composite, board, n):
    """The top-n levels by the composite's `board` pp, each with its pick's
    source run. Ties break on level_key (deterministic)."""
    recs = [r for r in composite["levels"] if r.get(board) and "skipped" not in r]
    recs.sort(key=lambda r: (-r[board]["pp"], r["level_key"]))
    return [_sel(r, board, board, k) for k, r in enumerate(recs[:n], 1)]


def gap_picks(composite, theirs, n, name="them"):
    """The top-n BL pp gaps (our bl.pp - their best pp on that BeatLeader
    leaderboard) over the levels both have a BL score on."""
    best = {}
    for s in theirs:
        lb = str(s["lb"])
        if lb not in best or s["pp"] > best[lb]["pp"]:
            best[lb] = s
    rows = []
    for r in composite["levels"]:
        if r.get("bl") and "skipped" not in r:
            t = best.get(str(r["bl"]["leaderboard_id"]))
            if t is not None:
                rows.append((r["bl"]["pp"] - t["pp"], r, t))
    rows.sort(key=lambda x: (-x[0], x[1]["level_key"]))
    return [_sel(r, "bl", "gap", k, {"their_pp": t["pp"], "their_name": name,
                                     "gap": g})
            for k, (g, r, t) in enumerate(rows[:n], 1)]


def plan(selections):
    """One replay per (level_key, source): the selections grouped, in first-
    selection order; `by_source` = the --levels lists per source bench."""
    reps = {}
    for s in selections:
        if s["source"] is None:
            raise ValueError(f"{s['level_key']}: no {s['board']} pick")
        key = (s["level_key"], s["source"])
        reps.setdefault(key, {"level_key": s["level_key"], "source": s["source"],
                              "song": s["song"], "difficulty": s["difficulty"],
                              "selected_as": []})["selected_as"].append(
            f"{s['category']} #{s['rank']}")
    by_source = {}
    for (k, src) in reps:
        by_source.setdefault(src, []).append(k)
    return {"replays": list(reps.values()), "by_source": by_source}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("composite")
    p.add_argument("--ss-top", type=int, default=10)
    p.add_argument("--bl-top", type=int, default=10)
    p.add_argument("--gap-top", type=int, default=0)
    p.add_argument("--gap-scores", default=None)
    p.add_argument("--gap-name", default="them")
    p.add_argument("--plan", default=None, help="write the plan JSON here")
    a = p.parse_args(argv)
    comp = json.load(open(a.composite, encoding="utf-8"))
    if not comp.get("composite"):
        sys.exit(f"{a.composite}: not a composite (no `composite` header)")
    sels = top_picks(comp, "ss", a.ss_top) + top_picks(comp, "bl", a.bl_top)
    if a.gap_top:
        if not a.gap_scores:
            sys.exit("--gap-top needs --gap-scores")
        sels += gap_picks(comp, json.load(open(a.gap_scores, encoding="utf-8")),
                          a.gap_top, a.gap_name)
    for s in sels:
        st = lambda x: "-" if x is None else f"{x:.2f}"
        print(f"{s['category']:>3} #{s['rank']:<2} {s['pp']:8.2f}pp  SS★ "
              f"{st(s['ss_stars']):>5} BL★ {st(s['bl_stars']):>5}  "
              f"{s['song'][:32]:32} {s['difficulty']:10} <- {s['source']}"
              + (f"  ({s['their_name']} {s['their_pp']:.2f}pp, gap "
                 f"{s['gap']:.2f})" if s["category"] == "gap" else ""))
    out = plan(sels)
    print(f"{len(sels)} selections -> {len(out['replays'])} replays")
    if a.plan:
        with open(a.plan, "w", encoding="utf-8") as f:
            json.dump({"selections": sels, **out}, f, indent=1)


if __name__ == "__main__":
    main()
