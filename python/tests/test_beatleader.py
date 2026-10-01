"""BeatLeader bench (2026-09-30): the BL pp formula against real ranked
scores, the BL catalog fetch, the `both` catalog union, the bench JSON
schema for --catalog scoresaber / beatleader / both, bench_compare's
catalog refusal, and a tiny real --sim rust bench for beatleader and
both (random-init checkpoint; skipped when the map cache is absent).

All CPU. The fixture tests/data/beatleader_pp_scores.json holds 52 real
no-modifier ranked scores over 26 leaderboards spanning BL 0.39*-15.8*
(api.beatleader.com/leaderboard/{id}, captured 2026-09-30) with the
API's reported pp / passPP / accPP / techPP."""
import json
import os
import subprocess
import sys

import pytest

from beat_arena_env import beatleader, scoresaber

import bench_scoresaber as bs
import bench_compare as bc

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(HERE, "tests", "data", "beatleader_pp_scores.json")


# ---------------------------------------------------------------------------
# PP formula
# ---------------------------------------------------------------------------

def _fixture():
    with open(FIXTURE) as f:
        return json.load(f)["scores"]


def test_pp_matches_real_scores():
    """Every captured score's pp AND its pass/acc/tech parts reproduce
    the API's numbers to < 0.01 pp (the server's float32 arithmetic vs
    float64 here: measured max error 0.0011 pp)."""
    scores = _fixture()
    assert len(scores) >= 20
    stars = sorted(s["stars"] for s in scores)
    assert stars[0] < 2.0 and stars[-1] > 12.0          # star-range spread
    worst = 0.0
    for s in scores:
        assert s["modifiers"] == ""
        r = beatleader.pp_from_accuracy(s["accuracy"], s["pass_rating"],
                                        s["acc_rating"], s["tech_rating"])
        for k_ours, k_api in (("pp", "pp"), ("pass_pp", "pass_pp"),
                              ("acc_pp", "acc_pp"), ("tech_pp", "tech_pp")):
            worst = max(worst, abs(r[k_ours] - s[k_api]))
        assert beatleader.pp_value(s["accuracy"], s["pass_rating"],
                                   s["acc_rating"], s["tech_rating"]) \
            == pytest.approx(s["pp"], abs=0.01)
    assert worst < 0.01, worst


def test_accuracy_definition_and_stars():
    """BL accuracy = baseScore / maxScore (ReplayUtils.ProcessReplay), and
    the stars field = ToStars(ratings) — both checked on the real data."""
    for s in _fixture():
        assert s["base_score"] / s["max_score"] == pytest.approx(
            s["accuracy"], abs=1e-6)
        assert beatleader.stars_from_ratings(
            s["pass_rating"], s["acc_rating"], s["tech_rating"]) \
            == pytest.approx(s["stars"], abs=1e-4)


def test_acc_curve_anchors_and_edges():
    assert beatleader.acc_curve(1.0) == pytest.approx(7.424)
    assert beatleader.acc_curve(0.95) == pytest.approx(1.0)
    assert beatleader.acc_curve(0.0) == pytest.approx(0.0)
    assert beatleader.acc_curve(0.9525) == pytest.approx((1.0 + 1.039) / 2)
    accs = [i / 400 for i in range(401)]
    vals = [beatleader.acc_curve(a) for a in accs]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    # out-of-range accuracy is worth nothing (PpFromScore's guard)
    assert beatleader.pp_value(0.0, 5, 5, 5) == 0.0
    assert beatleader.pp_value(1.01, 5, 5, 5) == 0.0
    # passPP clamps at 0 for tiny pass ratings
    assert beatleader.pp_parts(0.9, 0.0, 1.0, 1.0)[0] == 0.0


def test_bl_curve_differs_from_scoresaber():
    """Separate curves, separate star scales: 100% is 7.424x on BL's acc
    curve vs 5.367x on ScoreSaber's."""
    assert beatleader.acc_curve(1.0) != pytest.approx(
        scoresaber.pp_multiplier(1.0))


def test_weighted_total_same_decay_as_ss():
    pps = [300.0, 100.0, 200.0]
    assert beatleader.weighted_total_pp(pps) == pytest.approx(
        300 + 0.965 * 200 + 0.965 ** 2 * 100)
    assert beatleader.weighted_total_pp(pps) == pytest.approx(
        scoresaber.weighted_total_pp(pps))
    assert beatleader.weighted_total_pp([]) == 0.0


# ---------------------------------------------------------------------------
# Catalog fetch (network monkeypatched)
# ---------------------------------------------------------------------------

def _board(i, mode="Standard", value=9, status=3):
    return {
        "id": f"{i:x}91", "song": {"hash": f"AB{i:02d}", "name": f"s{i}",
                                   "mapper": "m", "duration": 100},
        "difficulty": {"value": value, "difficultyName": "ExpertPlus",
                       "modeName": mode, "status": status,
                       "stars": 10.0 - i, "passRating": 5.0,
                       "accRating": 6.0, "techRating": 2.0,
                       "predictedAcc": 0.97, "maxScore": 1000 + i,
                       "duration": 100, "njs": 18, "notes": 10,
                       "chains": 0, "bombs": 0, "walls": 0}}


def test_fetch_ranked_catalog_paginates_and_filters(monkeypatch):
    boards = [_board(i) for i in range(7)]
    boards[2] = _board(2, mode="OneSaber")
    boards[4] = _board(4, status=4)                  # not ranked: dropped
    urls = []

    def fake_get(url, **kw):
        urls.append(url)
        page = int(url.split("page=")[1].split("&")[0])
        count = int(url.split("count=")[1].split("&")[0])
        assert "type=ranked" in url
        return {"data": boards[(page - 1) * count:page * count],
                "metadata": {"total": 7, "itemsPerPage": count,
                             "page": page}}

    monkeypatch.setattr(beatleader, "_get_json", fake_get)
    monkeypatch.setattr(beatleader.time, "sleep", lambda s: None)
    allb = beatleader.fetch_all_ranked(per_page=3)
    assert len(allb) == 6 and len(urls) == 3          # 3 pages of 3
    urls.clear()
    cat, excluded = beatleader.fetch_ranked_catalog()
    assert len(cat) == 5 and excluded == {"mode OneSaber": 1}
    assert [e["stars"] for e in cat] == sorted(e["stars"] for e in cat)
    e = cat[0]
    assert e["hash"] == e["hash"].lower() and e["difficulty"] == "ExpertPlus"
    assert e["characteristic"] == "Standard"
    assert {"pass_rating", "acc_rating", "tech_rating", "predicted_acc",
            "api_max_score", "duration_s"} <= set(e)
    assert "max_pp" not in e        # BL has no maxPP; never an SS field


# ---------------------------------------------------------------------------
# --catalog both: union + dedup
# ---------------------------------------------------------------------------

def _ss(i, h, diff="Expert", stars=3.0):
    return {"leaderboard_id": 1000 + i, "hash": h, "song": f"s{h}",
            "mapper": "m", "difficulty": diff, "stars": stars,
            "max_pp": stars * scoresaber.PP_PER_STAR, "api_max_score": 1000}


def _bl(i, h, diff="Expert", stars=4.0):
    return {"leaderboard_id": f"{i:x}bl", "hash": h, "song": f"s{h}",
            "mapper": "m", "difficulty": diff, "characteristic": "Standard",
            "stars": stars, "pass_rating": 4.0, "acc_rating": 6.0,
            "tech_rating": 2.0, "predicted_acc": 0.97, "api_max_score": 1000}


def test_merge_catalogs_union_dedup():
    ss = [_ss(0, "aa"), _ss(1, "aa", "Hard", 2.0), _ss(2, "bb")]
    bl = [_bl(0, "aa"), _bl(1, "cc"), _bl(2, "bb", "ExpertPlus")]
    m = bs.merge_catalogs(ss, bl)
    keys = [e["level_key"] for e in m]
    assert len(keys) == len(set(keys)) == 5      # 1 shared of 3 + 3
    shared = [e for e in m if e["ss"] and e["bl"]]
    assert [e["level_key"] for e in shared] == ["aa_Expert_Standard"]
    assert shared[0]["ss"]["stars"] == 3.0 and shared[0]["bl"]["stars"] == 4.0
    assert bs.catalog_counts(m) == {"n_ss_ranked": 3, "n_bl_ranked": 3,
                                    "n_shared": 1, "n_ss_only": 2,
                                    "n_bl_only": 2}
    # same hash, different difficulty / characteristic = different level
    assert "bb_ExpertPlus_Standard" in keys and "bb_Expert_Standard" in keys
    # deterministic, input-order independent
    assert bs.merge_catalogs(ss[::-1], bl[::-1]) == m
    # star filter on each leaderboard's OWN stars
    lone = [e for e in m if bs.in_star_range(e, (3.9, 4.1), "both")]
    assert {e["level_key"] for e in lone} == {
        "aa_Expert_Standard", "cc_Expert_Standard",
        "bb_ExpertPlus_Standard"}
    with pytest.raises(ValueError, match="duplicate"):
        bs.merge_catalogs(ss + [_ss(9, "aa")], bl)


def test_merge_hash_case_insensitive():
    m = bs.merge_catalogs([_ss(0, "aa")], [_bl(0, "AA")])
    assert len(m) == 1 and m[0]["ss"] and m[0]["bl"]


# ---------------------------------------------------------------------------
# finish_slot / save_results schema per catalog
# ---------------------------------------------------------------------------

_INFO = {"good_cuts": 8, "bad_cuts": 1, "misses": 1, "n_notes": 10,
         "bombs_hit": 0, "score_pre": 600.0, "score_acc": 100.0,
         "score_post": 250.0, "flail_rad": 0.5, "flail_ticks": 10,
         "wrist_rev_l": 3, "wrist_rev_r": 4, "ticks": 900,
         "wall_enters": 0, "term_flags": 0}


def _slot(entry, max_score=1000):
    return {"entry": entry, "max_score": max_score, "duration_s": 60.0}


def test_finish_slot_scoresaber_unchanged():
    """The default catalog's record keeps the historical keys, in order."""
    e = dict(_ss(0, "aa"))
    r = bs.finish_slot(_slot(e), _INFO, (960.0, 8, 1))
    assert list(r)[:12] == ["leaderboard_id", "song_hash", "song", "mapper",
                            "difficulty", "stars", "max_pp", "duration_s",
                            "n_notes", "score", "max_score", "api_max_score"]
    assert r["accuracy"] == 0.96
    assert r["pp"] == round(scoresaber.pp_value(0.96, e["max_pp"]), 2)
    assert "ss" not in r and "bl" not in r and "level_key" not in r


def test_finish_slot_beatleader():
    e = dict(_bl(0, "aa"), api_max_score=1200)     # BL max > computed max
    r = bs.finish_slot(_slot(e), _INFO, (960.0, 8, 1))
    assert r["leaderboard_id"] == "0bl" and r["stars"] == 4.0
    assert r["accuracy"] == 0.96                    # the agent's accuracy
    assert r["bl_accuracy"] == 0.8                  # BL's: score / its max
    want = beatleader.pp_from_accuracy(0.8, 4.0, 6.0, 2.0)
    assert r["pp"] == round(want["pp"], 2)
    assert r["acc_pp"] == round(want["acc_pp"], 2)
    assert "max_pp" not in r
    json.dumps(r)


def test_finish_slot_both():
    m = bs.merge_catalogs([_ss(0, "aa"), _ss(1, "bb")],
                          [_bl(0, "aa"), _bl(1, "cc")])
    by = {e["level_key"]: e for e in m}
    shared = bs.finish_slot(_slot(by["aa_Expert_Standard"]), _INFO,
                            (960.0, 8, 1))
    assert shared["level_key"] == "aa_Expert_Standard"
    assert shared["ss"]["pp"] == round(
        scoresaber.pp_value(0.96, 3.0 * scoresaber.PP_PER_STAR), 2)
    assert shared["bl"]["pp"] == round(beatleader.pp_value(
        0.96, 4.0, 6.0, 2.0), 2)
    assert shared["ss_stars"] == 3.0 and shared["bl_stars"] == 4.0
    assert "pp" not in shared and "stars" not in shared
    ss_only = bs.finish_slot(_slot(by["bb_Expert_Standard"]), _INFO,
                             (960.0, 8, 1))
    assert ss_only["bl"] is None and ss_only["bl_stars"] is None
    bl_only = bs.finish_slot(_slot(by["cc_Expert_Standard"]), _INFO,
                             (960.0, 8, 1))
    assert bl_only["ss"] is None and bl_only["bl"]["pp"] > 0
    json.dumps([shared, ss_only, bl_only])


def test_save_results_totals_per_catalog(tmp_path):
    m = bs.merge_catalogs([_ss(0, "aa"), _ss(1, "bb")],
                          [_bl(0, "aa"), _bl(1, "cc"), _bl(2, "dd")])
    recs = {}
    for e in m:
        r = bs.finish_slot(_slot(e), _INFO, (960.0, 8, 1))
        recs[bs.task_key(r)] = r
    skip = [e for e in m if e["level_key"] == "dd_Expert_Standard"][0]
    recs[skip["level_key"]] = bs.skip_record(skip, "no notes parsed")
    out = str(tmp_path / "both.json")
    data = {"model_hash": "sha256:aa", "catalog": "both"}
    bs.save_results(out, data, recs)
    d = json.load(open(out))
    assert d["n_evaluated"] == 3 and d["n_skipped"] == 1
    ss_pp = [r["ss"]["pp"] for r in recs.values()
             if "skipped" not in r and r["ss"]]
    bl_pp = [r["bl"]["pp"] for r in recs.values()
             if "skipped" not in r and r["bl"]]
    assert len(ss_pp) == 2 and len(bl_pp) == 2
    assert d["ss_total_weighted_pp"] == round(
        scoresaber.weighted_total_pp(ss_pp), 2)
    assert d["bl_total_weighted_pp"] == round(
        beatleader.weighted_total_pp(bl_pp), 2)
    assert d["bl_total_unweighted_pp"] == round(sum(bl_pp), 2)
    assert d["total_weighted_pp"] == d["ss_total_weighted_pp"]
    assert (d["n_shared"], d["n_ss_only"], d["n_bl_only"]) == (1, 1, 1)
    # resume: the reloaded file keys back onto the same task keys
    lv = {bs.task_key(e): e for e in d["levels"]}
    assert set(lv) == {e["level_key"] for e in m}

    bl_recs = {}
    for e in [_bl(0, "aa"), _bl(1, "cc")]:
        r = bs.finish_slot(_slot(e), _INFO, (960.0, 8, 1))
        bl_recs[bs.task_key(r)] = r
    out2 = str(tmp_path / "bl.json")
    bs.save_results(out2, {"model_hash": "x", "catalog": "beatleader"},
                    bl_recs)
    d2 = json.load(open(out2))
    assert d2["total_weighted_pp"] == d2["bl_total_weighted_pp"] == round(
        beatleader.weighted_total_pp([r["pp"] for r in bl_recs.values()]), 2)
    assert "ss_total_weighted_pp" not in d2

    out3 = str(tmp_path / "ss.json")      # scoresaber: no new pp keys
    r = bs.finish_slot(_slot(_ss(0, "aa")), _INFO, (960.0, 8, 1))
    bs.save_results(out3, {"model_hash": "x"}, {r["leaderboard_id"]: r})
    d3 = json.load(open(out3))
    assert not any(k.startswith(("bl_", "ss_total")) for k in d3)
    assert d3["total_weighted_pp"] == r["pp"]


def test_parse_args_catalog():
    a = bs.parse_args(["--ckpt", "x.pt"])
    assert a.catalog == "scoresaber"
    assert a.ss_catalog_path == "maps/scoresaber_catalog.json"
    a = bs.parse_args(["--ckpt", "x.pt", "--catalog", "both"])
    assert a.catalog == "both"
    a = bs.parse_args(["--ckpt", "x.pt", "--catalog", "my/cat.json"])
    assert a.catalog == "scoresaber" and a.ss_catalog_path == "my/cat.json"
    with pytest.raises(SystemExit):
        bs.parse_args(["--ckpt", "x.pt", "--catalog", "osu"])


# ---------------------------------------------------------------------------
# bench_compare
# ---------------------------------------------------------------------------

def _write(tmp_path, name, levels, **hdr):
    p = str(tmp_path / name)
    d = {"tau_scale": None, "mirror": False, "scoring_version": 3,
         "sim": "rust", "sim_dtype": "f32", **hdr}
    bs.save_results(p, d, {bs.task_key(e): e for e in levels})
    return p


def test_bench_compare_refuses_cross_catalog(tmp_path):
    ss = [bs.finish_slot(_slot(_ss(0, "aa")), _INFO, (960.0, 8, 1))]
    bl = [bs.finish_slot(_slot(_bl(0, "aa")), _INFO, (960.0, 8, 1))]
    pa = _write(tmp_path, "a.json", ss)               # unstamped = scoresaber
    pb = _write(tmp_path, "b.json", bl, catalog="beatleader")
    for force in (False, True):                       # --force cannot
        with pytest.raises(bc.CatalogMismatch):
            bc.compare(pa, pb, out=lambda s: None, force=force)
    pc = _write(tmp_path, "c.json", ss, catalog="scoresaber")
    assert bc.compare(pa, pc, out=lambda s: None)["catalog"] == "scoresaber"


def test_bench_compare_both_and_bl_totals(tmp_path):
    m = bs.merge_catalogs([_ss(0, "aa")], [_bl(0, "aa"), _bl(1, "cc")])
    la = [bs.finish_slot(_slot(e), _INFO, (960.0, 8, 1)) for e in m]
    lb = [bs.finish_slot(_slot(e), _INFO, (900.0, 8, 1)) for e in m]
    pa = _write(tmp_path, "a.json", la, catalog="both")
    pb = _write(tmp_path, "b.json", lb, catalog="both")
    out = []
    r = bc.compare(pa, pb, out=out.append)
    assert r["common"] == 2 and r["diff_levels"] == 2
    assert r["fields"]["ss_pp"] == 1 and r["fields"]["bl_pp"] == 2
    assert r["bl_totals"] is not None
    assert any(s.startswith("BeatLeader totals") for s in out)
    assert r["sum_abs_dpp"] > 0


# ---------------------------------------------------------------------------
# a tiny real bench (--sim rust --limit 3)
# ---------------------------------------------------------------------------

MAPS = os.path.join(HERE, "maps")


def _real_levels():
    """(ss_catalog_subset, bl_catalog_subset) with one shared, one
    SS-only and one BL-only level whose charts are in the dat cache."""
    try:
        with open(os.path.join(MAPS, "scoresaber_catalog.json")) as f:
            ss = json.load(f)
        with open(os.path.join(MAPS, "beatleader_catalog.json")) as f:
            bl = json.load(f)
    except OSError:
        return None
    cache = os.path.join(MAPS, "scoresaber")

    def have(e):
        return os.path.isfile(os.path.join(cache, e["hash"], "Info.dat"))
    bk = {beatleader.entry_key(e): e for e in bl if have(e)}
    sk = {beatleader.entry_key(e): e for e in ss if have(e)}
    shared = sorted(set(bk) & set(sk))
    ss_only = sorted(set(sk) - set(bk))
    bl_only = sorted(set(bk) - set(sk))
    if not (shared and ss_only and bl_only):
        return None
    return ([sk[shared[0]], sk[ss_only[0]]], [bk[shared[0]], bk[bl_only[0]]])


def _ckpt(tmp_path):
    import torch
    from policy import NotePolicy
    torch.manual_seed(0)
    p = NotePolicy()
    path = str(tmp_path / "run" / "rand.pt")
    os.makedirs(os.path.dirname(path))
    torch.save({"arch": dict(p.arch), "state_dict": p.state_dict(),
                "global_step": 0}, path)
    return path


def _bench(tmp_path, ckpt, catalog, ss_path, bl_path, out):
    cmd = [sys.executable, os.path.join(HERE, "bench_scoresaber.py"),
           "--ckpt", ckpt, "--catalog", catalog, "--sim", "rust",
           "--limit", "3", "--num-envs", "4", "--groups", "1",
           "--device", "cpu", "--quiet",
           "--ss-catalog-path", ss_path, "--bl-catalog-path", bl_path,
           "--map-cache", os.path.join(MAPS, "scoresaber"),
           "--chart-cache", str(tmp_path / "cc"), "--out", out]
    env = dict(os.environ)
    env.pop("BEAT_Z_ASINH", None)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.mark.skipif(_real_levels() is None,
                    reason="needs maps/ catalogs + cached BeatSaver dats")
def test_real_bench_beatleader_and_both(tmp_path):
    ss_sub, bl_sub = _real_levels()
    ss_path, bl_path = str(tmp_path / "ss.json"), str(tmp_path / "bl.json")
    json.dump(ss_sub, open(ss_path, "w"))
    json.dump(bl_sub, open(bl_path, "w"))
    ckpt = _ckpt(tmp_path)

    out_bl = str(tmp_path / "out_bl.json")
    _bench(tmp_path, ckpt, "beatleader", ss_path, bl_path, out_bl)
    d = json.load(open(out_bl))
    assert d["catalog"] == "beatleader" and len(d["levels"]) == 2
    assert d["n_evaluated"] + d["n_skipped"] == 2
    for e in d["levels"]:
        assert isinstance(e["leaderboard_id"], str)
        if "skipped" not in e:
            assert e["pp"] >= 0 and 0 <= e["bl_accuracy"] <= 1
            assert "pass_rating" in e and "max_pp" not in e
    assert d["bl_total_weighted_pp"] == d["total_weighted_pp"]

    out_both = str(tmp_path / "out_both.json")
    stdout = _bench(tmp_path, ckpt, "both", ss_path, bl_path, out_both)
    assert "shared 1, SS-only 1, BL-only 1" in stdout
    d2 = json.load(open(out_both))
    assert d2["catalog"] == "both" and len(d2["levels"]) == 3
    assert d2["catalog_levels"] == {"n_levels": 3, "n_ss_ranked": 2,
                                    "n_bl_ranked": 2, "n_shared": 1,
                                    "n_ss_only": 1, "n_bl_only": 1}
    keys = [e["level_key"] for e in d2["levels"]]
    assert len(set(keys)) == 3                       # each level once
    played = [e for e in d2["levels"] if "skipped" not in e]
    for e in played:
        assert (e["ss"] is None) != (e["bl"] is None) or (e["ss"] and e["bl"])
    shared = [e for e in played if e["ss"] and e["bl"]]
    if shared:      # the same play scored by both leaderboards
        s = shared[0]
        assert s["ss"]["accuracy"] == s["accuracy"]
    # the same level benched under both modes plays identically
    bl_lv = {e["leaderboard_id"]: e for e in d["levels"]
             if "skipped" not in e}
    for e in played:
        if e["bl"] and e["bl"]["leaderboard_id"] in bl_lv:
            o = bl_lv[e["bl"]["leaderboard_id"]]
            assert (e["score"], e["bl"]["pp"]) == (o["score"], o["pp"])

    # resume: nothing left to play, same levels, no duplicates
    stdout = _bench(tmp_path, ckpt, "both", ss_path, bl_path, out_both)
    assert "0 to play" in stdout
    assert len(json.load(open(out_both))["levels"]) == 3
    # a different catalog into the same file is refused
    cmd_bad = subprocess.run(
        [sys.executable, os.path.join(HERE, "bench_scoresaber.py"),
         "--ckpt", ckpt, "--catalog", "beatleader", "--sim", "rust",
         "--device", "cpu", "--bl-catalog-path", bl_path,
         "--chart-cache", str(tmp_path / "cc"), "--out", out_both],
        capture_output=True, text=True, timeout=600)
    assert cmd_bad.returncode != 0 and "catalog" in cmd_bad.stderr
    # bench_compare: the two modes' files never compare
    with pytest.raises(bc.CatalogMismatch):
        bc.compare(out_both, out_bl, out=lambda s: None)


def test_failed_level_earns_no_bl_pp():
    """A cut-short level (fall / tilt / envelope / max-steps) is a failed
    play: 0 BL pp even though BL's pass term is accuracy-free; the raw
    curve value stays in pp_formula (user decision, 2026-09-30)."""
    import bench_scoresaber as bs
    e = {"leaderboard_id": "x", "stars": 7.0, "pass_rating": 6.0,
         "acc_rating": 7.0, "tech_rating": 5.0, "api_max_score": 100000}
    ok = bs.bl_result(e, 1000, 100000)
    bad = bs.bl_result(e, 1000, 100000, failed=True)
    assert ok["pp"] > 0 and ok["pp"] == ok["pp_formula"] and not ok["failed"]
    assert bad["pp"] == 0.0 and bad["failed"]
    assert bad["pp_formula"] == ok["pp"] and bad["pass_pp"] == ok["pass_pp"] > 0
