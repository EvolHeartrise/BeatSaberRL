"""viz/replays-site/scripts/publish_replays.py on a best-of composite: a
replay is matched to the composite pick(s) it reproduces — the picked input
run's rate and body, and that pick's play counters."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "publish_replays",
    os.path.join(HERE, "..", "..", "viz", "replays-site", "scripts",
                 "publish_replays.py"))
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

LEAF = [0.9] * 16
BENCH = {"composite": {"inputs": [
    {"file": "a30.json", "tick_hz": 30, "tau_scale": None},
    {"file": "b60.json", "tick_hz": 60, "tau_scale": LEAF}]}}


def _play(score):
    return {"score": score, "highest_combo": 10, "good_cuts": 10,
            "bad_cuts": 0, "misses": 0, "bombs_hit": 0}


def _got(score):
    return {"score": score, "combo": 10, "good": 10, "bad": 0, "miss": 0,
            "bombs": 0}


REC = {"level_key": "k", **_play(900), "ss": {"pp": 1.0}, "bl": {"pp": 2.0},
       "ss_from": "a30.json", "bl_from": "b60.json", "bl_play": _play(950)}


def test_matches_the_pick_whose_run_and_play_it_reproduces():
    h30 = {"tick_hz": 30, "tau_scale": None}
    h60 = {"tick_hz": 60, "tau_scale": [v + 1e-8 for v in LEAF]}
    assert pr.composite_matches(BENCH, REC, h30, _got(900)) == ["ss"]
    assert pr.composite_matches(BENCH, REC, h60, _got(950)) == ["bl"]
    # right run, wrong play / right play, wrong run or body
    assert pr.composite_matches(BENCH, REC, h30, _got(950)) == []
    assert pr.composite_matches(BENCH, REC, {"tick_hz": 60}, _got(900)) == []
    assert pr.composite_matches(BENCH, REC, {"tick_hz": 60, "tau_scale": None},
                                _got(950)) == []


def test_one_replay_can_be_both_picks():
    rec = {**REC, "bl_from": "a30.json", "bl_play": _play(900)}
    assert pr.composite_matches(BENCH, rec, {}, _got(900)) == ["ss", "bl"]
    bl_only = {**rec, "ss": None, "ss_from": None}
    assert pr.composite_matches(BENCH, bl_only, {}, _got(900)) == ["bl"]


def test_file_ids_differ_per_body_and_keep_the_single_run_scheme():
    a = pr.file_id("sha256:w", 30, 1, False, "k|" + pr.body_key(None))
    b = pr.file_id("sha256:w", 30, 1, False, "k|" + pr.body_key(LEAF))
    assert a != b and len(a) == 16
    # the single-run naming is unchanged (published v60-60hz ids stay put)
    import hashlib
    assert pr.file_id("sha256:w", 60, 1, False, 198039) == hashlib.sha256(
        b"sha256:w|60|1|0|198039").hexdigest()[:16]
