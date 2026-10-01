"""BeatLeader ranked-map benchmarking: catalog and BeatLeader PP.

The BeatLeader counterpart of scoresaber.py (bench-only, like it; the
training gym never touches this module). Map acquisition and the combo
multiplier are SHARED with ScoreSaber: BeatLeader leaderboards point at
the same BeatSaver song hashes, so their charts land in the same
maps/scoresaber/<hash>/ dat cache and maps/chart_cache pickles, and the
multiplied max score is the same formula (BeatLeader's
ReplayUtils.MaxScoreForNote is ScoreSaber's 115*(8n-63); an agent's
accuracy is one number, max scores agree — see bench_scoresaber).

1. Ranked catalog. `GET api.beatleader.com/leaderboards?type=ranked`
   paginated (count=100) gives every ranked BeatLeader leaderboard
   (3,952 over all characteristics, 2026-09-30). BeatLeader rates each
   difficulty with THREE ratings (passRating / accRating / techRating)
   plus their combined `stars` and `predictedAcc`; these are BeatLeader's
   own numbers and are kept strictly apart from ScoreSaber's stars (the
   catalog fields are `stars`, `pass_rating`, ... in a separate file,
   maps/beatleader_catalog.json; a merged `both` bench level carries them
   under its `bl` key). Only the Standard characteristic is kept (the
   env plays Standard only — the beatsaver.py parser reads the Standard
   set; ScoreSaber's catalog is SoloStandard only too).

2. BeatLeader PP (General context, no modifiers — how the agent plays).
   Transcribed from BeatLeader-Server @ 1474312940511b2810f403649cae80a15684baac
   (master, 2026-07-19), Utils/ReplayUtils.cs:
     passPP = 15.2 * exp(passRating ** (1/2.62)) - 30   (clamped >= 0)
     accPP  = Curve2(acc) * accRating * 34               (32-point curve)
     techPP = exp(1.9 * acc) * 1.08 * techRating
     pp     = Inflate(passPP + accPP + techPP)
     Inflate(x) = 650 * x**1.3 / 650**1.3
   (GetPp / Inflate / Curve2 / PpFromScore). With no modifiers the
   modifier multiplier `mp` is exactly 1 (GetTotalMultiplier starts at 1
   and adds only for modifiers present), so rawPP == fullPP and bonusPp
   is 0; the reported pass/acc/tech parts are each scaled by
   pp / (passPP+accPP+techPP). Accuracy = baseScore / maxScore
   (ReplayUtils.ProcessReplay), maxScore = the difficulty's `maxScore`.
   The server computes in float32; this is float64 — agreement with
   the API's reported pp is pinned to < 0.01 pp by
   tests/test_beatleader.py over real ranked scores.
   Stars = Inflate(GetPp at acc 0.96) / 52 (ToStars) — reproduced too.

3. Profile weighting: Utils/PlayerUtils.cs RecalculatePPAndRankFastGeneral
   sorts the player's ranked scores (pp != 0) by pp descending and sums
   pp * 0.965**i — the SAME 0.965-per-rank decay ScoreSaber uses, applied
   the same way (best play at weight 1).
"""

from __future__ import annotations

import math
import time
import urllib.parse

from .scoresaber import _get_json

BEATLEADER_API = "https://api.beatleader.com/leaderboards"
BEATLEADER_LEADERBOARD_API = "https://api.beatleader.com/leaderboard/{}"

# BeatLeader difficulty `value` -> Info.dat difficulty name (same ints
# as ScoreSaber's)
DIFF_NAMES = {1: "Easy", 3: "Normal", 5: "Hard", 7: "Expert", 9: "ExpertPlus"}
PLAYABLE_MODES = ("Standard",)       # characteristics the env can play
STATUS_RANKED = 3                    # DifficultyStatus.ranked

# ---------------------------------------------------------------------------
# PP — BeatLeader-Server Utils/ReplayUtils.cs @ 1474312940511b2810f403649cae80a15684baac
# ---------------------------------------------------------------------------

PP_DECAY = 0.965        # Utils/PlayerUtils.cs: MathF.Pow(0.965f, i)

# ReplayUtils.pointList2 (accuracy, multiplier), descending
ACC_CURVE = [
    (1.0, 7.424), (0.999, 6.241), (0.9975, 5.158), (0.995, 4.010),
    (0.9925, 3.241), (0.99, 2.700), (0.9875, 2.303), (0.985, 2.007),
    (0.9825, 1.786), (0.98, 1.618), (0.9775, 1.490), (0.975, 1.392),
    (0.9725, 1.315), (0.97, 1.256), (0.965, 1.167), (0.96, 1.094),
    (0.955, 1.039), (0.95, 1.000), (0.94, 0.931), (0.93, 0.867),
    (0.92, 0.813), (0.91, 0.768), (0.9, 0.729), (0.875, 0.650),
    (0.85, 0.581), (0.825, 0.522), (0.8, 0.473), (0.75, 0.404),
    (0.7, 0.345), (0.65, 0.296), (0.6, 0.256), (0.0, 0.000),
]


def acc_curve(acc: float) -> float:
    """ReplayUtils.Curve2: first point with x <= acc, linear to the
    previous (higher) point; i == 0 (acc >= 1) clamps to segment 1."""
    i = 0
    while i < len(ACC_CURVE) and ACC_CURVE[i][0] > acc:
        i += 1
    if i == 0:
        i = 1
    if i == len(ACC_CURVE):       # acc < 0: unreachable (PpFromScore
        return 0.0                # returns 0 for acc <= 0 first)
    (x0, y0), (x1, y1) = ACC_CURVE[i - 1], ACC_CURVE[i]
    return y0 + (acc - x0) / (x1 - x0) * (y1 - y0)


def inflate(x: float) -> float:
    """ReplayUtils.Inflate: 650 * x^1.3 / 650^1.3."""
    return 650.0 * x ** 1.3 / 650.0 ** 1.3


def pp_parts(acc: float, pass_rating: float, acc_rating: float,
             tech_rating: float):
    """ReplayUtils.GetPp (General context): raw (passPP, accPP, techPP)."""
    try:
        pass_pp = 15.2 * math.exp(pass_rating ** (1 / 2.62)) - 30.0
    except (OverflowError, ValueError):
        pass_pp = 0.0
    if not math.isfinite(pass_pp) or pass_pp < 0:
        pass_pp = 0.0
    acc_pp = acc_curve(acc) * acc_rating * 34.0
    tech_pp = math.exp(1.9 * acc) * 1.08 * tech_rating
    return pass_pp, acc_pp, tech_pp


def pp_from_accuracy(acc: float, pass_rating: float, acc_rating: float,
                     tech_rating: float) -> dict:
    """ReplayUtils.PpFromScore, General context, NO modifiers (mp = 1).
    Returns {pp, pass_pp, acc_pp, tech_pp} — the API's score fields
    `pp`, `passPP`, `accPP`, `techPP` (bonusPp is 0 without modifiers)."""
    if acc <= 0 or acc > 1:
        return {"pp": 0.0, "pass_pp": 0.0, "acc_pp": 0.0, "tech_pp": 0.0}
    p, a, t = pp_parts(acc, pass_rating, acc_rating, tech_rating)
    s = p + a + t
    full = inflate(s) if s > 0 else 0.0
    if not math.isfinite(full):
        full = 0.0
    k = full / s if s > 0 else 0.0
    return {"pp": full, "pass_pp": p * k, "acc_pp": a * k, "tech_pp": t * k}


def pp_value(acc: float, pass_rating: float, acc_rating: float,
             tech_rating: float) -> float:
    return pp_from_accuracy(acc, pass_rating, acc_rating, tech_rating)["pp"]


def stars_from_ratings(pass_rating, acc_rating, tech_rating) -> float:
    """ReplayUtils.ToStars: Inflate(GetPp(acc 0.96)) / 52."""
    s = sum(pp_parts(0.96, pass_rating, acc_rating, tech_rating))
    return inflate(s) / 52.0 if s > 0 else 0.0


def weighted_total_pp(pps) -> float:
    """Profile total (PlayerUtils.RecalculatePPAndRankFastGeneral): pp
    descending, play i weighted 0.965^i — same rule as ScoreSaber."""
    return sum(p * PP_DECAY ** i
               for i, p in enumerate(sorted(pps, reverse=True)))


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def catalog_entry(b):
    """One API leaderboard -> catalog entry (any characteristic)."""
    d, s = b["difficulty"], b["song"]
    def _f(x):
        return None if x is None else float(x)
    return {
        "leaderboard_id": str(b["id"]),
        "hash": s["hash"].lower(),
        "song": s["name"],
        "mapper": s.get("mapper"),
        "difficulty": DIFF_NAMES.get(d["value"], d.get("difficultyName")),
        "characteristic": d["modeName"],
        "stars": _f(d.get("stars")),
        "pass_rating": _f(d.get("passRating")),
        "acc_rating": _f(d.get("accRating")),
        "tech_rating": _f(d.get("techRating")),
        "predicted_acc": _f(d.get("predictedAcc")),
        "api_max_score": int(d.get("maxScore") or 0) or None,
        "duration_s": d.get("duration") or s.get("duration"),
        "njs": _f(d.get("njs")),
        "notes": d.get("notes"), "chains": d.get("chains"),
        "bombs": d.get("bombs"), "walls": d.get("walls"),
    }


def fetch_all_ranked(log=None, per_page=100, sleep_s=0.5):
    """Every ranked BeatLeader leaderboard, ALL characteristics,
    unfiltered (catalog_entry form). ~40 pages at count=100 with a
    polite pause between pages."""
    out, seen = [], set()
    page = 1
    while True:
        q = urllib.parse.urlencode({"type": "ranked", "page": page,
                                    "count": per_page})
        d = _get_json(f"{BEATLEADER_API}?{q}")
        boards = d["data"]
        total = d["metadata"]["total"]
        if not boards:
            break
        for b in boards:
            if b["id"] in seen:
                continue
            if b["difficulty"].get("status") != STATUS_RANKED:
                continue
            seen.add(b["id"])
            out.append(catalog_entry(b))
        if log and page % 10 == 0:
            log(f"[bl-catalog] page {page}/{-(-total // per_page)} "
                f"({len(out)} leaderboards)")
        if page * per_page >= total:
            break
        page += 1
        time.sleep(sleep_s)
    return out


def playable(entries):
    """(kept, excluded-by-reason counts): Standard characteristic and a
    standard difficulty int only, with all three ratings present."""
    kept, excluded = [], {}
    for e in entries:
        if e["characteristic"] not in PLAYABLE_MODES:
            why = f"mode {e['characteristic']}"
        elif e["difficulty"] not in DIFF_NAMES.values():
            why = f"difficulty {e['difficulty']}"
        elif None in (e["pass_rating"], e["acc_rating"], e["tech_rating"]):
            why = "missing ratings"
        else:
            kept.append(e)
            continue
        excluded[why] = excluded.get(why, 0) + 1
    kept.sort(key=lambda e: (e["stars"], e["leaderboard_id"]))
    return kept, excluded


def fetch_ranked_catalog(log=None):
    """The bench catalog: every ranked, PLAYABLE (Standard) BeatLeader
    leaderboard sorted by BL stars ascending, plus the exclusion counts."""
    allb = fetch_all_ranked(log=log)
    kept, excluded = playable(allb)
    if log:
        log(f"[bl-catalog] {len(allb)} ranked leaderboards, {len(kept)} "
            f"playable (Standard); excluded {excluded or 'none'}")
    return kept, excluded


def level_key(hash_, difficulty, characteristic="Standard"):
    """Cross-leaderboard identity of a playable level: one chart =
    (song hash, difficulty, characteristic). ScoreSaber's catalog is
    SoloStandard only, so its entries are characteristic Standard."""
    return f"{hash_.lower()}_{difficulty}_{characteristic}"


def entry_key(entry):
    return level_key(entry["hash"], entry["difficulty"],
                     entry.get("characteristic", "Standard"))
