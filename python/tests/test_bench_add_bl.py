"""tools/bench_add_bl.py: BeatLeader view of a ScoreSaber-only bench —
the same play priced with bench_scoresaber.bl_result, shared levels only,
failed plays inferred from unjudged notes when terminated_tick is absent."""
from tools import bench_add_bl as t
import bench_scoresaber as bs

CAT = [{"hash": "AAA", "difficulty": "Expert", "characteristic": "Standard",
        "leaderboard_id": "bl1", "stars": 7.0, "pass_rating": 6.0,
        "acc_rating": 7.0, "tech_rating": 5.0, "api_max_score": 100000},
       {"hash": "CCC", "difficulty": "Hard", "characteristic": "Standard",
        "leaderboard_id": "bl3", "stars": 5.0, "pass_rating": 4.0,
        "acc_rating": 5.0, "tech_rating": 3.0, "api_max_score": 50000}]


def _lv(lb, h, diff, score, judged, n):
    return {"leaderboard_id": lb, "song_hash": h, "difficulty": diff,
            "score": score, "max_score": 100000, "good_cuts": judged,
            "bad_cuts": 0, "misses": 0, "n_notes": n}


def test_shared_levels_priced_and_failures_inferred():
    bench = {"model_hash": "sha256:x", "levels": [
        _lv(1, "aaa", "Expert", 98000, 100, 100),     # shared, completed
        _lv(2, "bbb", "Expert", 97000, 100, 100),     # SS-only: absent
        _lv(3, "ccc", "Hard", 20000, 40, 100)]}       # shared, cut short
    sc = t.bl_sidecar(bench, CAT)
    assert sc["n_bl_played"] == 2 and sc["n_bl_catalog"] == 2
    by = {x["ss_leaderboard_id"]: x["bl"] for x in sc["levels"]}
    assert by[1] == bs.bl_result(CAT[0], 98000, 100000, False)
    assert by[3]["failed"] and by[3]["pp"] == 0.0 and by[3]["pp_formula"] > 0
    assert sc["bl_failed_levels"] == 1
    assert sc["bl_total_weighted_pp"] == round(by[1]["pp"], 2)


def test_recorded_termination_wins_over_inference():
    lv = _lv(1, "aaa", "Expert", 98000, 100, 100)
    lv["terminated_tick"] = 400                        # recorded cut
    sc = t.bl_sidecar({"model_hash": "h", "levels": [lv]}, CAT)
    assert sc["levels"][0]["bl"]["failed"] and not sc["levels"][0]["failed_inferred"]
