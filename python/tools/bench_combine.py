"""Per-level best-of composite of several --catalog both bench files.

    python tools/bench_combine.py OUT.json IN1.json IN2.json [...]

All inputs must be the same weights (model_hash) and catalog `both`.
For each level, the ScoreSaber view takes the input with the highest SS
pp and the BeatLeader view the input with the highest BL pp — chosen
independently, because the two leaderboards price the same play
differently. The record's top-level play fields (score, accuracy,
misses, ...) are the SS pick's; the BL pick's play fields ride in
`bl_play`, and `ss_from` / `bl_from` name the source of each.

This is a HINDSIGHT composite (an upper bound: per-level max over runs
is always >= any single run, even over repeats of one configuration),
never a model's playable result. The header says so (`composite`), and
model_hash is replaced by a pseudo-hash of the inputs so a composite can
never be mistaken for, or displace, the weights' own bench (the site
keeps one entry per weights hash).
"""
import hashlib
import json
import os
import sys

PLAY = ("score", "max_score", "accuracy", "good_cuts", "bad_cuts", "misses",
        "bombs_hit", "highest_combo", "full_combo", "breaks", "fell",
        "terminated_tick", "ticks", "torque_nm", "torque_nms")


def _pp(rec, lb):
    sub = rec.get(lb)
    return None if not sub else float(sub["pp"])


def _weighted(pps):
    pps = sorted(pps, reverse=True)
    return sum(p * 0.965 ** i for i, p in enumerate(pps)), sum(pps)


def combine(paths):
    runs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("catalog") != "both":
            raise SystemExit(f"{p}: catalog {d.get('catalog')!r}, need 'both'")
        runs.append((os.path.basename(p), d,
                     {l["level_key"]: l for l in d["levels"] if "level_key" in l}))
    hashes = {d["model_hash"] for _, d, _ in runs}
    if len(hashes) != 1:
        raise SystemExit(f"inputs are different models: {sorted(hashes)}")
    keys = sorted(set.intersection(*(set(m) for _, _, m in runs)))
    base = runs[0][1]
    levels, src = [], {"ss": {}, "bl": {}}
    for k in keys:
        pick = {}
        for lb in ("ss", "bl"):
            cand = [(_pp(m[k], lb), i) for i, (_, _, m) in enumerate(runs)
                    if _pp(m[k], lb) is not None]
            pick[lb] = max(cand)[1] if cand else None
        i_ss = pick["ss"] if pick["ss"] is not None else pick["bl"]
        rec = dict(runs[i_ss][2][k])
        rec["ss_from"] = runs[pick["ss"]][0] if pick["ss"] is not None else None
        if pick["bl"] is not None:
            b = runs[pick["bl"]][2][k]
            rec["bl"] = b["bl"]
            rec["bl_play"] = {f: b.get(f) for f in PLAY}
            rec["bl_from"] = runs[pick["bl"]][0]
            src["bl"][runs[pick["bl"]][0]] = src["bl"].get(runs[pick["bl"]][0], 0) + 1
        else:
            rec["bl_from"] = None
        if pick["ss"] is not None:
            src["ss"][rec["ss_from"]] = src["ss"].get(rec["ss_from"], 0) + 1
        levels.append(rec)
    ssw, ssu = _weighted([_pp(l, "ss") for l in levels if l.get("ss")])
    blw, blu = _weighted([_pp(l, "bl") for l in levels if l.get("bl")])
    pseudo = "sha256:" + hashlib.sha256(
        ("composite:" + base["model_hash"] + ":" + ",".join(
            sorted(n for n, _, _ in runs))).encode()).hexdigest()
    head = {k: v for k, v in base.items() if k != "levels"}
    head.update({
        "model_hash": pseudo,
        "weights_hash": base["model_hash"],
        "composite": {
            "rule": "per-level best pp, chosen independently per leaderboard",
            "hindsight": True,
            "inputs": [{"file": n, "tick_hz": d.get("tick_hz", 30),
                        "tau_scale": d.get("tau_scale"),
                        "ss_total_weighted_pp": d.get("ss_total_weighted_pp"),
                        "bl_total_weighted_pp": d.get("bl_total_weighted_pp")}
                       for n, d, _ in runs],
            "levels_from": src},
        "n_evaluated": len(levels), "n_skipped": 0,
        "ss_total_weighted_pp": round(ssw, 2), "ss_total_unweighted_pp": round(ssu, 2),
        "bl_total_weighted_pp": round(blw, 2), "bl_total_unweighted_pp": round(blu, 2),
        "total_weighted_pp": round(ssw, 2), "total_unweighted_pp": round(ssu, 2),
    })
    for k in [k for k in head if k.startswith(("torque_", "breaks_total", "wall_enters_total",
                                                 "fell_levels", "tilted_levels",
                                                 "envelope_levels", "terminated_levels",
                                                 "term_levels", "wrist_rev", "flail_rad_per"))]:
        head.pop(k)   # per-run aggregates do not survive a per-level mix
    head["levels"] = levels
    return head


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    out = combine(argv[1:])
    tmp = argv[0] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, argv[0])
    print(f"{argv[0]}: {out['n_evaluated']} levels | SS {out['ss_total_weighted_pp']:,.2f} wpp "
          f"/ {out['ss_total_unweighted_pp']:,.0f} upp | BL {out['bl_total_weighted_pp']:,.2f} wpp "
          f"/ {out['bl_total_unweighted_pp']:,.0f} upp | from {out['composite']['levels_from']}")


if __name__ == "__main__":
    main(sys.argv[1:])
