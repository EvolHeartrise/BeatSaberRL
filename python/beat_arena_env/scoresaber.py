"""ScoreSaber-ranked map acquisition (curriculum training's map source).

ScoreSaber assigns every ranked (map, difficulty) leaderboard a star
rating — the community difficulty scale the curriculum climbs. We query
`/api/leaderboards` for ranked SoloStandard leaderboards inside a star
band, pick one at random, download the map zip from BeatSaver by song
hash, and validate that our parser can actually read the target
difficulty (v4-schema maps, missing Standard sets etc. are skipped, not
fatal). stdlib-only (urllib) — no new dependencies.

Sparse bands are real: 0.9–1.1★ holds only ~5 ranked leaderboards, so
the band auto-widens in 0.1★ steps until candidates exist.
"""

from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request

from .beatsaver import load_beatmap

SCORESABER_API = "https://scoresaber.com/api/leaderboards"
BEATSAVER_API = "https://api.beatsaver.com/maps/hash/{}"
_HEADERS = {"User-Agent": "beat-arena-curriculum/0.1"}

# ScoreSaber difficulty int -> Info.dat difficulty name
DIFF_NAMES = {1: "Easy", 3: "Normal", 5: "Hard", 7: "Expert", 9: "ExpertPlus"}


def _get_json(url, timeout=30):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _leaderboard_page(min_star, max_star, page):
    q = urllib.parse.urlencode({"ranked": "true", "minStar": f"{min_star:g}",
                                "maxStar": f"{max_star:g}", "page": page})
    d = _get_json(f"{SCORESABER_API}?{q}")
    return d["leaderboards"], d["metadata"]


def ranked_candidates(min_star, max_star, rng):
    """Shuffled ranked SoloStandard candidates in [min_star, max_star].

    Fetches page 1 for the total count, then one random page. Each
    candidate: {hash, song, mapper, difficulty (Info.dat name), stars,
    leaderboard_id}.
    """
    boards, meta = _leaderboard_page(min_star, max_star, 1)
    per_page = max(1, meta.get("itemsPerPage", 14))
    n_pages = max(1, math.ceil(meta["total"] / per_page))
    page = int(rng.integers(1, n_pages + 1))
    if page != 1:
        boards, _ = _leaderboard_page(min_star, max_star, page)
    out = []
    for b in boards:
        d = b["difficulty"]
        if d["gameMode"] != "SoloStandard" or d["difficulty"] not in DIFF_NAMES:
            continue
        out.append({"hash": b["songHash"].lower(), "song": b["songName"],
                    "mapper": b["levelAuthorName"],
                    "difficulty": DIFF_NAMES[d["difficulty"]],
                    "stars": float(b["stars"]),
                    "leaderboard_id": b["id"]})
    rng.shuffle(out)
    return out


def download_zip(song_hash, dest_dir):
    """Fetch <hash>.zip from BeatSaver into dest_dir (cached). Returns path."""
    song_hash = song_hash.lower()
    path = os.path.join(dest_dir, f"{song_hash}.zip")
    if os.path.exists(path):
        return path
    info = _get_json(BEATSAVER_API.format(song_hash))
    url = None
    for v in info["versions"]:
        if v["hash"].lower() == song_hash:
            url = v["downloadURL"]
    if url is None:                       # hash superseded: take latest version
        url = info["versions"][-1]["downloadURL"]
    req = urllib.request.Request(url, headers=_HEADERS)
    tmp = path + ".part"
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    os.replace(tmp, path)
    return path


def acquire_random_map(target_star, star_band, dest_dir, rng, speed_scale=0.3,
                       max_len_s=None, exclude=(), max_tries=8):
    """Pick a random ranked map near target_star, download + validate it.

    exclude: set of (hash, difficulty) already on the ladder. Widens the
    band by 0.1★ per side while it comes up empty (sparse-band reality).
    Returns a meta dict (hash, song, mapper, difficulty, stars, zip
    basename, n_notes, duration_s, nps); raises RuntimeError if nothing
    usable is found.
    """
    os.makedirs(dest_dir, exist_ok=True)
    exclude = set(exclude)
    band, tried = star_band, 0
    while band <= star_band + 1.0:
        lo, hi = max(0.0, target_star - band), target_star + band
        for cand in ranked_candidates(lo, hi, rng):
            if (cand["hash"], cand["difficulty"]) in exclude:
                continue
            if tried >= max_tries:
                break
            tried += 1
            try:
                path = download_zip(cand["hash"], dest_dir)
                notes, meta = load_beatmap(path, cand["difficulty"],
                                           speed_scale=speed_scale,
                                           first_note_s=2.0)
            except Exception as e:      # v4 schema, no Standard set, net hiccup
                print(f"  [curriculum] skip {cand['song']!r} "
                      f"[{cand['difficulty']}]: {type(e).__name__}: {e}",
                      flush=True)
                continue
            if not notes:
                continue
            if max_len_s is not None and notes[-1].hit_time > max_len_s:
                print(f"  [curriculum] skip {cand['song']!r}: "
                      f"{notes[-1].hit_time:.0f}s > --max-song-s", flush=True)
                continue
            cand.update(zip=os.path.basename(path), n_notes=meta["n_notes"],
                        duration_s=round(meta["duration_s"], 1),
                        nps=round(meta["nps"], 2))
            return cand
        if tried >= max_tries:
            break
        band += 0.1
    raise RuntimeError(
        f"no usable ranked map near {target_star:.1f}★ "
        f"(band widened to ±{band - 0.1:.1f}, {tried} candidates tried)")
