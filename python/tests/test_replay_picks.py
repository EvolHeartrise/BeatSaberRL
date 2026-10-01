"""Replay selection for best-of composites (tools/replay_picks.py) and
make_top10_replays' --catalog both record handling (board views, explicit
--levels selection, composite refusal, header level identity)."""
import pytest

import make_top10_replays as mt
from tools import replay_picks as rp


def _rec(key, ss_pp, bl_pp, score, ss_from="a.json", bl_from="a.json",
         bl_id=None):
    return {"level_key": key, "song": key.upper(), "song_hash": key,
            "difficulty": "Expert", "characteristic": "Standard",
            "score": score, "accuracy": score / 1000, "highest_combo": 10,
            "good_cuts": 10, "bad_cuts": 0, "misses": 0, "bombs_hit": 0,
            "full_combo": True,
            "ss": None if ss_pp is None else {
                "pp": ss_pp, "stars": 5.0, "leaderboard_id": sum(map(ord, key)),
                "accuracy": score / 1000},
            "bl": None if bl_pp is None else {
                "pp": bl_pp, "stars": 6.0, "leaderboard_id": bl_id or key + "x91",
                "accuracy": score / 1001},
            "ss_from": ss_from if ss_pp is not None else None,
            "bl_from": bl_from if bl_pp is not None else None}


COMP = {"composite": {"inputs": []}, "levels": [
    _rec("k1", 100, 300, 900),
    _rec("k2", 120, None, 800, ss_from="b.json"),
    _rec("k3", None, 400, 700, bl_from="b.json"),          # BL-only
    _rec("k4", 90, 350, 950, ss_from="a.json", bl_from="b.json"),
]}


# --- tools/replay_picks -----------------------------------------------------

def test_top_picks_per_board_use_that_boards_source():
    ss = rp.top_picks(COMP, "ss", 2)
    assert [(s["level_key"], s["source"], s["rank"]) for s in ss] == [
        ("k2", "b.json", 1), ("k1", "a.json", 2)]
    bl = rp.top_picks(COMP, "bl", 3)
    # BL-only k3 included; k4's BL pick is run b even though its SS pick is a
    assert [(s["level_key"], s["source"]) for s in bl] == [
        ("k3", "b.json"), ("k4", "b.json"), ("k1", "a.json")]
    assert bl[0]["ss_stars"] is None and bl[0]["bl_stars"] == 6.0


def test_gap_picks_only_levels_they_played_best_score_wins():
    theirs = [{"lb": "k1x91", "pp": 250.0}, {"lb": "k1x91", "pp": 280.0},
              {"lb": "k3x91", "pp": 100.0}, {"lb": "nope", "pp": 1.0}]
    g = rp.gap_picks(COMP, theirs, 5, "CR")
    assert [(s["level_key"], round(s["gap"], 2), s["their_pp"]) for s in g] == [
        ("k3", 300.0, 100.0), ("k1", 20.0, 280.0)]
    assert all(s["board"] == "bl" and s["category"] == "gap" for s in g)


def test_plan_dedupes_same_level_and_source_only():
    sels = (rp.top_picks(COMP, "ss", 4) + rp.top_picks(COMP, "bl", 4))
    out = rp.plan(sels)
    keys = [(r["level_key"], r["source"]) for r in out["replays"]]
    # k1 (a/a) once; k4 twice (SS from a, BL from b)
    assert keys.count(("k1", "a.json")) == 1
    assert ("k4", "a.json") in keys and ("k4", "b.json") in keys
    assert len(keys) == len(set(keys)) == 5
    k1 = next(r for r in out["replays"] if r["level_key"] == "k1")
    assert k1["selected_as"] == ["ss #2", "bl #3"]
    assert out["by_source"] == {"b.json": ["k2", "k3", "k4"],
                                "a.json": ["k1", "k4"]}


# --- make_top10_replays: --catalog both records ------------------------------

SRC = {"catalog": "both", "model_hash": "sha256:aa", "levels": [
    _rec("k1", 100, 300, 900), _rec("k3", None, 400, 700),
    {"level_key": "k9", "skipped": "no notes parsed"}]}


def test_board_views():
    ss = mt.bench_pool(SRC, "ss")
    assert [e["level_key"] for e in ss] == ["k1"]
    assert ss[0]["pp"] == 100 and ss[0]["stars"] == 5.0
    assert ss[0]["leaderboard_id"] == SRC["levels"][0]["ss"]["leaderboard_id"]
    bl = mt.bench_pool(SRC, "bl")
    assert [e["level_key"] for e in bl] == ["k1", "k3"]
    assert bl[1]["pp"] == 400 and bl[1]["leaderboard_id"] == "k3x91"
    assert bl[1]["accuracy"] == pytest.approx(700 / 1001)
    # the play itself is the record's, whatever the board
    assert bl[0]["score"] == ss[0]["score"] == 900


def test_select_levels_order_and_errors():
    bl = mt.bench_pool(SRC, "bl")
    assert [e["level_key"] for e in mt.select_levels(bl, ["k3", "k1"])] == ["k3", "k1"]
    with pytest.raises(SystemExit):
        mt.select_levels(bl, ["k9"])                     # skipped level
    with pytest.raises(SystemExit):
        mt.select_levels(mt.bench_pool(SRC, "ss"), ["k3"])  # not SS-ranked
    # explicit --levels: a level the board does not rank is seen from the
    # other board (an SS selection list may hold BL-only levels and v.v.)
    both = mt.select_levels(mt.bench_pool(SRC, "ss", fallback=True), ["k3", "k1"])
    assert [(e["level_key"], e["pp"]) for e in both] == [("k3", 400), ("k1", 100)]


def test_single_catalog_keys_and_board_guard():
    ss_bench = {"catalog": "scoresaber", "levels": [
        {"leaderboard_id": 42, "pp": 1.0, "max_pp": 2.0}]}
    pool = mt.bench_pool(ss_bench, "ss")
    assert mt.select_levels(pool, ["42"])[0]["leaderboard_id"] == 42
    with pytest.raises(SystemExit):
        mt.bench_pool(ss_bench, "bl")


def test_composites_are_refused():
    with pytest.raises(SystemExit, match="SOURCE"):
        mt.bench_pool(COMP, "ss")


def test_level_identity():
    k1, k3 = SRC["levels"][0], SRC["levels"][1]
    assert mt.level_identity(k1) == {
        "level_key": "k1", "characteristic": "Standard",
        "ss_leaderboard_id": k1["ss"]["leaderboard_id"],
        "bl_leaderboard_id": "k1x91"}
    assert mt.level_identity(k3)["ss_leaderboard_id"] is None
    assert mt.level_identity({"leaderboard_id": 7, "max_pp": 1.0}) == {
        "level_key": None, "characteristic": "Standard",
        "ss_leaderboard_id": 7, "bl_leaderboard_id": None}
