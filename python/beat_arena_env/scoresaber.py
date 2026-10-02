"""ScoreSaber ranked-map benchmarking: catalog, maps, and real scoring.

Three independent pieces, all bench-only (the training gym never touches
them — the combo multiplier would add overhead for no training benefit):

1. Ranked catalog + map acquisition. `/api/leaderboards?ranked=true`
   paginated to completion gives every ranked SoloStandard leaderboard
   (~6.1k difficulties over ~2.5k songs, 2026-07); per-leaderboard star
   rating and maxPP come straight from the API. Map data is downloaded
   from BeatSaver by song hash and cached as EXTRACTED chart files only
   (.dat/.json allowlist: Info.dat + difficulty .dats — audio, art, and
   any other zip payload discarded; typically a few MB per song, chart
   .dats dominating). stdlib urllib only.

2. The real combo score multiplier (1x/2x/4x/8x). Progression decoded
   from ScoreSaber's own max-score formula, 115·(8n−63) for n ≥ 14
   notes: 1 unit scored at 1x, 4 at 2x, 8 at 4x, 8x thereafter —
   i.e. the progress bar needs 2/4/8 hits and the promoted multiplier
   applies to the hit that fills the bar. A combo break (bad cut, miss,
   bomb hit) halves the multiplier and resets the bar (BSMG wiki).
   The whole of a note's points (cut + follow-through) commit at its
   cut-time multiplier. Scoring units are notes (115), chain heads
   (85), chain links (20); bombs score nothing.

3. ScoreSaber PP. pp = curve(accuracy) × maxPP, where maxPP =
   42.117208413 × stars (the API's maxPP field) and the curve is the
   official 37-point piecewise-linear accuracy curve fetched from
   scoresaber.com/api/v2/realms/1/pp-curve (2026-07-23; curve(95%) = 1,
   curve(100%) = 5.37). Accuracy = multiplied score / multiplied max
   score. A profile's total weights plays 0.965^(rank−1), best first
   (wiki.scoresaber.com/ranking-system).
"""

from __future__ import annotations

import json
import os
import shutil
import ssl
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

from .constants import KIND_BOMB, KIND_CHAIN_HEAD, KIND_CHAIN_LINK, KIND_NOTE

try:                # BeatSaver's cert chain fails Windows' stock trust
    import certifi  # store (expired root); certifi's bundle verifies it
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

SCORESABER_API = "https://scoresaber.com/api/leaderboards"
BEATSAVER_API = "https://api.beatsaver.com/maps/hash/{}"
_HEADERS = {"User-Agent": "beat-arena-bench/0.1"}

# ScoreSaber difficulty int -> Info.dat difficulty name
DIFF_NAMES = {1: "Easy", 3: "Normal", 5: "Hard", 7: "Expert", 9: "ExpertPlus"}

# ---------------------------------------------------------------------------
# PP
# ---------------------------------------------------------------------------

PP_PER_STAR = 42.117208413          # maxPP = stars * this (API maxPP field)
PP_DECAY = 0.965                    # profile weighting per play rank

# Official curve, scoresaber.com/api/v2/realms/1/pp-curve (fetched
# 2026-07-23): (accuracy fraction, multiplier), descending.
PP_CURVE = [
    (1.0, 5.367394282890631), (0.9995, 5.019543595874787),
    (0.999, 4.715470646416203), (0.99825, 4.325027383589547),
    (0.9975, 3.996793606763322), (0.99625, 3.5526145337555373),
    (0.995, 3.2022017597337955), (0.99375, 2.9190155639254955),
    (0.9925, 2.685667856592722), (0.99125, 2.4902905794106913),
    (0.99, 2.324506282149922), (0.9875, 2.058947159052738),
    (0.985, 1.8563887693647105), (0.9825, 1.697536248647543),
    (0.98, 1.5702410055532239), (0.9775, 1.4664726399289512),
    (0.975, 1.3807102743105126), (0.9725, 1.3090333065057616),
    (0.97, 1.2485807759957321), (0.965, 1.1552120359501035),
    (0.96, 1.0871883573850478), (0.955, 1.0388633331418984),
    (0.95, 1.0), (0.94, 0.9417362980580238),
    (0.93, 0.9039994071865736), (0.92, 0.8728710341448851),
    (0.91, 0.8488375988124467), (0.9, 0.825756123560842),
    (0.875, 0.7816934560296046), (0.85, 0.7462290664143185),
    (0.825, 0.7150465663454271), (0.8, 0.6872268862950283),
    (0.75, 0.6451808210101443), (0.7, 0.6125565959114954),
    (0.65, 0.5866010012767576), (0.6, 0.18223233667439062),
    (0.0, 0.0),
]


def pp_multiplier(acc: float) -> float:
    """Piecewise-linear interpolation of the official accuracy curve."""
    if acc >= PP_CURVE[0][0]:
        return PP_CURVE[0][1]
    for (hi_a, hi_m), (lo_a, lo_m) in zip(PP_CURVE, PP_CURVE[1:]):
        if acc >= lo_a:
            t = (acc - lo_a) / (hi_a - lo_a)
            return lo_m + (hi_m - lo_m) * t
    return 0.0


def pp_value(acc: float, max_pp: float) -> float:
    return pp_multiplier(acc) * max_pp


def weighted_total_pp(pps) -> float:
    """Profile total: best play at full weight, each next * 0.965."""
    return sum(p * PP_DECAY ** i for i, p in enumerate(sorted(pps, reverse=True)))


# ---------------------------------------------------------------------------
# Combo score multiplier
# ---------------------------------------------------------------------------

_PROGRESS_NEEDED = {1: 2, 2: 4, 4: 8}   # hits to fill the bar at each tier
MAX_UNIT_POINTS = {KIND_NOTE: 115, KIND_CHAIN_HEAD: 85, KIND_CHAIN_LINK: 20}


def max_unit_points(n) -> int:
    """A scoreable note's max cut points. A chain head that is also an
    arc head or arc tail (the game's ChainHeadArcHead / ChainHeadArcTail
    scoring types, 1.40+) gets the 30 follow-through points as a fixed
    grant: 115, not 85 (BeatLeader's max score and the game agree;
    2026-10-02 SS/BL max-score study)."""
    if n.kind == KIND_CHAIN_HEAD and (n.free_pre or n.free_post):
        return 115
    return MAX_UNIT_POINTS[n.kind]


class ComboTracker:
    """Multiplied-score walk over the sim's event stream.

    on_cut(i) on every good-cut event ("ok"): advances combo + the
    multiplier bar (promoting BEFORE the hit commits, per the max-score
    formula) and pins note i's multiplier. on_points(i, pts) for any
    points event (the cut's own points AND its later "post"
    follow-through) — both commit at note i's cut-time multiplier.
    on_break() for bad cuts / misses / bomb hits: combo to 0, multiplier
    halves, bar resets.
    """

    def __init__(self):
        self.multiplier = 1
        self._progress = 0
        self.combo = 0
        self.max_combo = 0
        self.breaks = 0
        self.score = 0.0
        self._note_mult = {}

    def on_cut(self, note_idx: int):
        need = _PROGRESS_NEEDED.get(self.multiplier)
        if need is not None:
            self._progress += 1
            if self._progress >= need:
                self.multiplier *= 2
                self._progress = 0
        self._note_mult[note_idx] = self.multiplier
        self.combo += 1
        self.max_combo = max(self.max_combo, self.combo)

    def on_points(self, note_idx: int, pts: float):
        self.score += pts * self._note_mult.get(note_idx, self.multiplier)

    def on_break(self):
        self.combo = 0
        self.breaks += 1
        self.multiplier = max(1, self.multiplier // 2)
        self._progress = 0

    @property
    def progress_fraction(self) -> float:
        """Fill of the bar toward the next tier (1.0 at the 8x cap)."""
        need = _PROGRESS_NEEDED.get(self.multiplier)
        return 1.0 if need is None else self._progress / need

    def consume(self, events):
        """Feed one step's info["events"] (reason-string form). env v17
        wall rule (spec §12b): "wall_enter" (the rising edge of head-in-
        wall) breaks combo; "wall" (per-tick drain) has NO combo effect
        — without the explicit skip the catch-all would break combo on
        every tick spent inside a wall."""
        for note_idx, reason, pts in events:
            if reason == "ok":
                self.on_cut(note_idx)
                self.on_points(note_idx, pts)
            elif reason == "post":
                self.on_points(note_idx, pts)
            elif reason == "wall":
                pass
            else:            # slow/angle/wrong_hand/miss/bomb/wall_enter
                self.on_break()


def max_multiplied_score(chart) -> int:
    """Full-combo multiplied score for a chart (bombs excluded).

    For a pure-note chart this reproduces ScoreSaber's own formula
    (115·(8n−63) for n ≥ 14) — pinned by a test.
    """
    walk = ComboTracker()
    units = [n for n in sorted(chart, key=lambda n: n.hit_time)
             if n.kind != KIND_BOMB]
    for i, n in enumerate(units):
        walk.on_cut(i)
        walk.on_points(i, max_unit_points(n))
    return round(walk.score)


# ---------------------------------------------------------------------------
# ScoreSaber catalog + BeatSaver map cache
# ---------------------------------------------------------------------------

def _get_json(url, timeout=30, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout,
                                        context=_SSL_CTX) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))


def fetch_ranked_catalog(log=None):
    """Every ranked SoloStandard leaderboard, sorted by stars ascending.

    Entries: {leaderboard_id, hash, song, mapper, difficulty, stars,
    max_pp, api_max_score}. ~440 pages; a couple of minutes.
    """
    out, seen = [], set()
    page, total = 1, None
    while True:
        q = urllib.parse.urlencode({"ranked": "true", "page": page})
        d = _get_json(f"{SCORESABER_API}?{q}")
        boards = d["leaderboards"]
        total = d["metadata"]["total"]
        per_page = max(1, d["metadata"].get("itemsPerPage", 14))
        if not boards:
            break
        for b in boards:
            diff = b["difficulty"]
            if (diff["gameMode"] != "SoloStandard"
                    or diff["difficulty"] not in DIFF_NAMES
                    or b["id"] in seen):
                continue
            seen.add(b["id"])
            out.append({
                "leaderboard_id": b["id"],
                "hash": b["songHash"].lower(),
                "song": b["songName"],
                "mapper": b["levelAuthorName"],
                "difficulty": DIFF_NAMES[diff["difficulty"]],
                "stars": float(b["stars"]),
                "max_pp": float(b["maxPP"]),
                "api_max_score": int(b.get("maxScore") or 0) or None,
            })
        if log and page % 25 == 0:
            log(f"[catalog] page {page}/{-(-total // per_page)} "
                f"({len(out)} leaderboards)")
        if page * per_page >= total:
            break
        page += 1
        time.sleep(0.05)
    out.sort(key=lambda e: e["stars"])
    return out


# chart data is ONLY .dat/.json — allowlist, not blocklist (2026-07-25):
# BeatSaver zips are untrusted user uploads and have shipped .exe, .db,
# .srt, ... alongside the audio; extract nothing but chart files
_CHART_EXT = (".dat", ".json")


def _info_named_files(info):
    """Every file Info.dat names as chart data: the difficulty beatmaps
    (v2 `_beatmapFilename`, v4 `beatmapDataFilename`) and v4's audio
    data. Mappers may give these ANY extension (VENTEN ships `ben.ben`,
    `benben.ben`, ...), which the .dat/.json allowlist alone dropped —
    5 ScoreSaber levels pruned as unparseable (2026-08-08)."""
    out = set()
    for d in info.get("difficultyBeatmaps", []) or []:          # v4
        if d.get("beatmapDataFilename"):
            out.add(d["beatmapDataFilename"])
    for st in info.get("_difficultyBeatmapSets", []) or []:     # v2/v3
        for d in st.get("_difficultyBeatmaps", []) or []:
            if d.get("_beatmapFilename"):
                out.add(d["_beatmapFilename"])
    aud = (info.get("audio") or {}).get("audioDataFilename")
    if aud:
        out.add(aud)
    return {os.path.basename(n).lower() for n in out}


def _extract_chart_files(z, dest):
    """Copy the chart files out of map zip `z` into dest: .dat/.json plus
    exactly the files Info.dat names (any extension). Nothing else —
    the zip is an untrusted upload."""
    names = z.namelist()
    info_name = next((n for n in names
                      if os.path.basename(n).lower() == "info.dat"), None)
    named = set()
    if info_name is not None:
        try:
            named = _info_named_files(json.loads(z.read(info_name)))
        except (ValueError, UnicodeDecodeError):
            named = set()
    for name in names:
        base = os.path.basename(name)
        if not base:
            continue
        if not (base.lower().endswith(_CHART_EXT) or base.lower() in named):
            continue
        with z.open(name) as src, open(os.path.join(dest, base), "wb") as f:
            shutil.copyfileobj(src, f)


def _missing_named_charts(dest):
    """Cached map dir whose Info.dat names chart files that are absent
    (cached before the named-file rule above): re-download it."""
    for cand in ("Info.dat", "info.dat"):
        path = os.path.join(dest, cand)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    named = _info_named_files(json.load(f))
            except (ValueError, UnicodeDecodeError):
                return False
            have = {n.lower() for n in os.listdir(dest)}
            # only the bug class: a named chart the extension allowlist
            # dropped (a .dat the zip never had would re-fetch forever)
            return any(n not in have and not n.endswith(_CHART_EXT)
                       for n in named)
    return False


def ensure_map_dats(song_hash, cache_dir):
    """Chart files for song_hash, downloading from BeatSaver on a cache
    miss. Returns the map directory (feed it to load_beatmap). Only
    .dat/.json chart files and the files Info.dat names are kept (the
    zip's audio/art/video and any other payload are dropped)."""
    song_hash = song_hash.lower()
    dest = os.path.join(cache_dir, song_hash)
    if os.path.isdir(dest):
        if not _missing_named_charts(dest):
            return dest
        shutil.rmtree(dest)               # pre-fix cache: charts dropped
    os.makedirs(cache_dir, exist_ok=True)
    info = _get_json(BEATSAVER_API.format(song_hash))
    url = None
    for v in info["versions"]:
        if v["hash"].lower() == song_hash:
            url = v["downloadURL"]
    if url is None:                       # hash superseded: take latest
        url = info["versions"][-1]["downloadURL"]
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as r:
        blob = r.read()
    tmp = tempfile.mkdtemp(dir=cache_dir)
    try:
        with open(os.path.join(tmp, "map.zip"), "wb") as f:
            f.write(blob)
        with zipfile.ZipFile(os.path.join(tmp, "map.zip")) as z:
            _extract_chart_files(z, tmp)
        os.remove(os.path.join(tmp, "map.zip"))
        os.replace(tmp, dest)             # atomic: cache dir appears complete
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest
