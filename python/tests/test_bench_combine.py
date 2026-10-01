"""tools/bench_combine.py: per-level best-of composite of --catalog both
benches — SS and BL picks independent, provenance recorded, a pseudo
model_hash, totals re-weighted 0.965^i."""
import json

import pytest

from tools import bench_combine as bc


def _rec(key, ss_pp, bl_pp, score):
    return {"level_key": key, "score": score, "accuracy": score / 1000,
            "misses": 0, "full_combo": True,
            "ss": None if ss_pp is None else {"pp": ss_pp, "stars": 5.0},
            "bl": None if bl_pp is None else {"pp": bl_pp, "stars": 6.0}}


def _bench(tmp_path, name, recs, mh="sha256:aa", tick=30):
    p = tmp_path / name
    p.write_text(json.dumps({"catalog": "both", "model_hash": mh,
                             "tick_hz": tick, "levels": recs}))
    return str(p)


def test_independent_picks_and_totals(tmp_path):
    a = _bench(tmp_path, "a.json", [_rec("k1", 100, 50, 900), _rec("k2", 80, None, 800),
                                    _rec("k3", None, 70, 700)])
    b = _bench(tmp_path, "b.json", [_rec("k1", 90, 60, 950), _rec("k2", 85, None, 810),
                                    _rec("k3", None, 65, 690)], tick=60)
    out = bc.combine([a, b])
    lv = {l["level_key"]: l for l in out["levels"]}
    assert lv["k1"]["ss"]["pp"] == 100 and lv["k1"]["ss_from"] == "a.json"
    assert lv["k1"]["bl"]["pp"] == 60 and lv["k1"]["bl_from"] == "b.json"
    assert lv["k1"]["score"] == 900 and lv["k1"]["bl_play"]["score"] == 950
    assert lv["k2"]["ss_from"] == "b.json" and lv["k2"]["bl_from"] is None
    assert lv["k3"]["bl_from"] == "a.json"
    assert out["ss_total_unweighted_pp"] == 185
    assert out["ss_total_weighted_pp"] == round(100 + 85 * 0.965, 2)
    assert out["bl_total_weighted_pp"] == round(70 + 60 * 0.965, 2)
    assert out["model_hash"] != "sha256:aa" and out["weights_hash"] == "sha256:aa"
    assert out["composite"]["hindsight"] is True
    assert [i["tick_hz"] for i in out["composite"]["inputs"]] == [30, 60]


def test_refuses_mixed_models_and_catalogs(tmp_path):
    a = _bench(tmp_path, "a.json", [_rec("k1", 1, 1, 1)])
    b = _bench(tmp_path, "b.json", [_rec("k1", 1, 1, 1)], mh="sha256:bb")
    with pytest.raises(SystemExit):
        bc.combine([a, b])
    c = tmp_path / "c.json"
    c.write_text(json.dumps({"catalog": "scoresaber", "model_hash": "sha256:aa", "levels": []}))
    with pytest.raises(SystemExit):
        bc.combine([a, str(c)])
