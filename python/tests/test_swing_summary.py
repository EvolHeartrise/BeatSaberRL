"""2026-10-04 swing telemetry, CPU side: the trainer's ep_stats v2 rows
(from a fake env's pop_episodes), the reader on both layouts, the
levels.json level table, and swing_summary's per-level windows, mirror
merging, completed-only filtering, best-possible PP totals and
incremental reads."""
import json
import os

import numpy as np
import pytest

import ep_stats as es
import swing_summary as sw
from beat_arena_env import beatleader, scoresaber

EP_REC = 40


class FakeEnv:
    """pop_episodes() like CudaBeatVecEnv's: (M, 40) rows, one batch per
    call (the trainer pops once per update)."""

    def __init__(self, batches):
        self.batches = list(batches)

    def pop_episodes(self):
        return self.batches.pop(0) if self.batches else np.zeros((0, EP_REC), np.float32)


def env_row(chart, cw, completed=1, ticks=900, parts=None, good=10,
            ret=1.5, fell=0):
    r = np.zeros(EP_REC, np.float32)
    r[:16] = 1.0
    r[16], r[17], r[18], r[19] = ticks, ret, fell, good
    r[20] = cw / 2.0          # (unmultiplied score: not used here)
    r[21] = chart
    r[22], r[23], r[24], r[25] = cw, 7, 1, completed
    r[26], r[27] = 2, 3
    if parts is not None:
        r[28:40] = parts
    return r


def parts(preL, postL, precL, preR, postR, precR, nL=10, nR=10):
    """12 swing cols from per-hand points over nL / nR note cuts."""
    return [preL, 70 * nL, postL, 30 * nL, precL, 15 * nL,
            preR, 70 * nR, postR, 30 * nR, precR, 15 * nR]


# 3 levels x (original, mirror) = 6 charts: A shared SS+BL, B SS-only,
# C BL-only. chart 2k = original, 2k+1 = mirror.
LEVELS = {
    "format": "beat-arena-levels", "version": 1, "catalog": "both",
    "source": "trainer", "n_charts": 6,
    "chart_level": [0, 0, 1, 1, 2, 2], "chart_mirrored": [0, 1] * 3,
    "levels": [
        {"key": "a_ExpertPlus_Standard", "song": "Alpha", "difficulty": "ExpertPlus",
         "n_notes": 100, "duration_s": 50.0, "nps": 2.0, "max_score": 10000.0,
         "ss": {"id": 11, "stars": 9.0, "max_pp": 379.05, "max_score": 9800},
         "bl": {"id": "aa1", "stars": 10.0, "pass": 5.0, "acc": 9.0, "tech": 4.0,
                "max_score": 10000}},
        {"key": "b_Hard_Standard", "song": "Beta", "difficulty": "Hard",
         "n_notes": 80, "duration_s": 40.0, "nps": 2.0, "max_score": 8000.0,
         "ss": {"id": 22, "stars": 4.0, "max_pp": 168.47, "max_score": None},
         "bl": None},
        {"key": "c_Expert_Standard", "song": "Gamma", "difficulty": "Expert",
         "n_notes": 60, "duration_s": 30.0, "nps": 2.0, "max_score": 6000.0,
         "ss": None,
         "bl": {"id": "cc3", "stars": 6.0, "pass": 3.0, "acc": 7.0, "tech": 2.0,
                "max_score": None}},
    ],
}
META = {"level_lb": np.array([11, 11, 22, 22, -1, -1]),
        "level_max_pp": np.array([379.05, 379.05, 168.47, 168.47, 0, 0], np.float32),
        "level_max_score": np.array([10000, 10000, 8000, 8000, 6000, 6000], np.float32)}


def batches():
    """Update 1..4 pops. Level A: plays on both copies; the best
    completed score is 9700 (update 3, mirror); a 9990 play FELL (not
    completed). B: one completed play. C: completed twice."""
    P1 = parts(560, 240, 120, 630, 270, 135)           # 80/80/80 %, 90/90/90 %
    P2 = parts(700, 300, 150, 700, 300, 150)           # 100 %
    P3 = parts(350, 150, 75, 350, 150, 75)             # 50 %
    return [
        np.stack([env_row(0, 9000, parts=P1), env_row(4, 5000, parts=P3)]),
        np.stack([env_row(1, 9990, completed=0, fell=1, parts=P2),
                  env_row(2, 7000, parts=P1)]),
        np.stack([env_row(1, 9700, parts=P2), env_row(5, 5500, parts=P2),
                  env_row(3, 7500, completed=0, ticks=30, parts=P2)]),
        np.stack([env_row(0, 9100, parts=P3)]),
    ]


def write_run(tmp_path, upto=None, levels=True):
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    if levels:
        (run / "levels.json").write_text(json.dumps(LEVELS))
    env = FakeEnv(batches()[:upto])
    w = es.EpStatsWriter(str(run), log=lambda *a, **k: None)
    for update in range(len(batches()[:upto])):
        rows = env.pop_episodes()            # the trainer's per-update block
        w.append(es.build_rows(rows, META, update + 1))
    return run


# ------------------------------------------------------------- trainer rows
def test_build_rows_layout_from_a_fake_env(tmp_path):
    run = write_run(tmp_path)
    R = es.load(str(run))
    assert R.shape == (8, es.EP_STATS_COLS) == (8, 29)
    assert es.file_layout(str(run / "ep_stats.f32"))[:2] == (2, 29)
    r = R[0]
    assert r[0] == 0 and r[1] == 11 and r[3] == 10000      # chart, SS id, max
    assert r[4] == 900 and r[6] == 1 and r[7] == 9000
    assert r[8] == pytest.approx(0.9)                      # acc = cw / max
    assert r[9] == pytest.approx(scoresaber.pp_multiplier(0.9) * 379.05, rel=1e-6)
    assert (r[10], r[11], r[12], r[13], r[14]) == (7, 1, 10, 2, 3)
    assert r[15] == pytest.approx(1.5)
    assert r[es.COL_UPDATE] == 1 and R[-1, es.COL_UPDATE] == 4
    np.testing.assert_array_equal(r[es.COL_PART:], parts(560, 240, 120, 630, 270, 135))
    assert R[2, 9] == 0.0                                  # unfinished: no pp


def test_reader_v1_and_partial_rows(tmp_path):
    v1 = np.arange(3 * 16, dtype=np.float32).reshape(3, 16)
    p = tmp_path / "ep_stats.f32"
    p.write_bytes(v1.tobytes() + b"\x00" * 20)            # + a torn row
    assert es.file_layout(str(p))[:2] == (1, 16)
    np.testing.assert_array_equal(es.load(str(tmp_path)), v1)
    rows, off, ver, ncol = es.read_rows(str(p), 0)
    assert (len(rows), off, ver, ncol) == (3, 3 * 64, 1, 16)
    rows2, off2, _, _ = es.read_rows(str(p), off)
    assert len(rows2) == 0 and off2 == off                # torn row waits


def test_writer_moves_a_v1_file_aside(tmp_path):
    v1 = np.ones((2, 16), np.float32)
    (tmp_path / "ep_stats.f32").write_bytes(v1.tobytes())
    w = es.EpStatsWriter(str(tmp_path), log=lambda *a, **k: None)
    assert (tmp_path / "ep_stats.v1.f32").read_bytes() == v1.tobytes()
    assert es.file_layout(w.path)[:2] == (2, 29) and len(es.load(str(tmp_path))) == 0
    # a v2 file is appended to, not rotated
    w.append(np.zeros((1, 29), np.float32))
    es.EpStatsWriter(str(tmp_path), log=lambda *a, **k: None)
    assert len(es.load(str(tmp_path))) == 1


def test_tau_fit_reads_v2(tmp_path):
    import tau_fit
    run = write_run(tmp_path)
    R = es.load(str(run))
    T = np.zeros((len(R), 22), np.float32)
    T[:, 17] = R[:, 15]
    T[:, 16] = R[:, 4]
    T.tofile(run / "tau_draws.f32")
    E, T2 = tau_fit.load(str(run))
    assert E.shape[1] == 16 and len(E) == len(T2) == 7    # ticks >= 60


# --------------------------------------------------------------- aggregator
def _summary(run):
    sw.update_run(str(run))
    return json.loads((run / "swing.json").read_text())


def _row(d, key):
    return d["levels"]["key"].index(key)


def test_totals_completed_only_and_mirror_merge(tmp_path):
    d = _summary(write_run(tmp_path))
    L = d["levels"]
    a, b, c = (_row(d, k) for k in ("a_ExpertPlus_Standard", "b_Hard_Standard",
                                    "c_Expert_Standard"))
    assert d["n_levels"] == 3 and d["swing"] and d["ep_layout"] == 2
    # A: 4 attempts over both copies, 3 completed; best = 9700 (the
    # 9990 play fell)
    assert L["att"][a] == 4 and L["done"][a] == 3
    assert L["best_upd"][a] == 3
    ss_acc = 9700 / 9800                               # SS API maxScore
    bl_acc = 9700 / 10000                              # BL maxScore
    assert L["ss_acc"][a] == pytest.approx(100 * ss_acc, abs=0.01)
    ss_a = round(scoresaber.pp_value(ss_acc, 379.05), 2)
    bl_a = round(beatleader.pp_from_accuracy(bl_acc, 5.0, 9.0, 4.0)["pp"], 2)
    assert L["ss_pp"][a] == ss_a and L["bl_pp"][a] == bl_a
    # B: SS only, its SS maxScore unknown -> the computed max; the
    # ticks-30 incomplete play is neither an attempt nor a play
    assert L["att"][b] == 1 and L["done"][b] == 1 and L["bl_pp"][b] is None
    ss_b = round(scoresaber.pp_value(7000 / 8000, 168.47), 2)
    assert L["ss_pp"][b] == ss_b
    # C: BL only, best of 5000 / 5500
    assert L["ss_pp"][c] is None
    bl_c = round(beatleader.pp_from_accuracy(5500 / 6000, 3.0, 7.0, 2.0)["pp"], 2)
    assert L["bl_pp"][c] == bl_c
    t = d["totals"]
    assert t["ss_levels"] == 2 and t["bl_levels"] == 2
    hi, lo = sorted([ss_a, ss_b], reverse=True)
    assert t["ss_wpp"] == pytest.approx(hi + 0.965 * lo, abs=0.01)
    assert t["ss_upp"] == pytest.approx(ss_a + ss_b, abs=0.01)
    hi, lo = sorted([bl_a, bl_c], reverse=True)
    assert t["bl_wpp"] == pytest.approx(hi + 0.965 * lo, abs=0.01)
    assert t["bl_upp"] == pytest.approx(bl_a + bl_c, abs=0.01)


def test_windows(tmp_path):
    d = _summary(write_run(tmp_path))
    a = _row(d, "a_ExpertPlus_Standard")
    W = d["win"]
    # A's completed plays in order: P1 (upd 1), P2 (upd 3), P3 (upd 4)
    assert W["all"]["plays"][a] == 3
    assert W["n1"]["plays"][a] == 1 and W["n1"]["pre"][a] == 50.0
    assert W["n5"]["plays"][a] == 3
    # whole run: L pre = (560 + 700 + 350) / (3 * 700); R = (630+700+350)/2100
    assert W["all"]["preL"][a] == round(100 * 1610 / 2100, 1)
    assert W["all"]["preR"][a] == round(100 * 1680 / 2100, 1)
    assert W["all"]["pre"][a] == round(100 * 3290 / 4200, 1)
    assert W["all"]["post"][a] == round(100 * (240 + 270 + 600 + 300) / 1800, 1)
    assert W["all"]["prec"][a] == round(100 * (120 + 135 + 300 + 150) / 900, 1)
    # mean accuracy (score / the computed max) over the window's plays
    assert W["all"]["acc"][a] == round(100 * (0.9 + 0.97 + 0.91) / 3, 1)
    assert W["n1"]["acc"][a] == 91.0
    # last-K updates: last_update = 4, so K = 100 covers everything
    assert d["last_update"] == 4
    assert W["k100"]["plays"][a] == 3
    keys = [w["key"] for w in d["windows"]]
    assert keys == ["n1", "n5", "n20", "k100", "k500", "all"]


def test_last_k_updates_window(tmp_path):
    """Plays older than last_update - K drop out of the k window only."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "levels.json").write_text(json.dumps(LEVELS))
    w = es.EpStatsWriter(str(run), log=lambda *a, **k: None)
    P = parts(700, 300, 150, 700, 300, 150)
    Q = parts(0, 0, 0, 0, 0, 0)
    w.append(es.build_rows(np.stack([env_row(0, 9000, parts=Q)]), META, 1))
    w.append(es.build_rows(np.stack([env_row(1, 9000, parts=P)]), META, 150))
    d = _summary(run)
    a = _row(d, "a_ExpertPlus_Standard")
    assert d["win"]["k100"]["plays"][a] == 1 and d["win"]["k100"]["pre"][a] == 100.0
    assert d["win"]["k500"]["plays"][a] == 2 and d["win"]["k500"]["pre"][a] == 50.0


def test_incremental_equals_full(tmp_path):
    run = write_run(tmp_path, upto=2)
    sw.update_run(str(run))
    meta = sw.State.load_meta(str(run))
    off1 = meta["offset"]
    # unchanged file: no re-read, swing.json untouched
    mt = os.stat(run / "swing.json").st_mtime_ns
    sw.update_run(str(run))
    assert os.stat(run / "swing.json").st_mtime_ns == mt
    # the remaining updates arrive (+ a torn row mid-write)
    w = es.EpStatsWriter(str(run), log=lambda *a, **k: None)
    for update, rows in enumerate(batches()[2:], start=2):
        w.append(es.build_rows(rows, META, update + 1))
    with open(run / "ep_stats.f32", "ab") as f:
        f.write(b"\x01" * 40)
    sw.update_run(str(run))
    meta2 = sw.State.load_meta(str(run))
    assert meta2["offset"] > off1 and (os.path.getsize(run / "ep_stats.f32")
                                       - meta2["offset"]) == 40
    inc = json.loads((run / "swing.json").read_text())
    full_dir = write_run(tmp_path / "full")
    full = _summary(full_dir)
    for k in ("generated",):
        inc.pop(k), full.pop(k)
    assert inc == full


def test_levels_change_rebuilds(tmp_path):
    run = write_run(tmp_path)
    d0 = _summary(run)
    lv = json.loads((run / "levels.json").read_text())
    lv["levels"][0]["song"] = "Alpha (renamed)"
    (run / "levels.json").write_text(json.dumps(lv))
    d1 = _summary(run)
    assert "Alpha (renamed)" in d1["levels"]["song"]
    assert d1["totals"] == d0["totals"] and d1["rows"] == d0["rows"]


def test_v1_run_without_swing_cols(tmp_path):
    """An old 16-col file: plays / acc / totals, swing columns null."""
    run = tmp_path / "old"
    run.mkdir()
    (run / "levels.json").write_text(json.dumps(LEVELS))
    rows = np.concatenate(batches())
    er = es.build_rows(rows, META, 1)[:, :16]
    er.tofile(run / "ep_stats.f32")
    d = _summary(run)
    a = _row(d, "a_ExpertPlus_Standard")
    assert d["ep_layout"] == 1 and not d["swing"]
    assert d["win"]["all"]["pre"] is None and d["win"]["all"]["plays"][a] == 3
    # no metrics.jsonl: the update windows are unavailable
    av = {w["key"]: w["available"] for w in d["windows"]}
    assert not av["k100"] and av["n1"] and av["all"]
    v2 = _summary(write_run(tmp_path / "new"))
    assert d["totals"] == v2["totals"]


def test_v1_update_index_from_metrics(tmp_path):
    run = tmp_path / "old"
    run.mkdir()
    rows = np.concatenate(batches())                   # 8 rows
    es.build_rows(rows, META, 1)[:, :16].tofile(run / "ep_stats.f32")
    (run / "levels.json").write_text(json.dumps(LEVELS))
    recs = [{"upd": 101, "ep_stats_episodes": 2}, {"upd": 102, "ep_stats_episodes": 4},
            {"upd": 103, "ep_stats_episodes": 7}, {"upd": 104, "ep_stats_episodes": 8}]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    np.testing.assert_array_equal(sw.v1_updates(str(run), 0, 8),
                                  [101, 101, 102, 102, 103, 103, 103, 104])
    d = _summary(run)
    assert d["last_update"] == 104
    assert d["levels"]["best_upd"][_row(d, "a_ExpertPlus_Standard")] == 103


def test_no_levels_json_falls_back_to_ss_ids(tmp_path):
    run = write_run(tmp_path, levels=False)
    d = _summary(run)
    assert d["levels_source"] is None
    # A's two copies merge on their SS id; C (BL-only, id -1) by chart
    assert sorted(d["levels"]["key"]) == ["chart:4", "chart:5", "ss:11", "ss:22"]
    assert d["totals"]["ss_levels"] == 0 and d["totals"]["bl_levels"] == 0


def test_index_entry(tmp_path):
    run = write_run(tmp_path)
    s = sw.update_run(str(run))
    e = sw.index_entry(s, str(run), rel="camp/runs/run")
    assert e["path"] == "camp/runs/run/swing.json" and e["swing"]
    assert set(e["totals"]) >= {"ss_wpp", "ss_upp", "bl_wpp", "bl_upp"}


# -------------------------------------------------------------- level table
@pytest.fixture
def catalogs(tmp_path, monkeypatch):
    import bench_scoresaber as bs
    from beat_arena_env import ranked_pool as rp

    def payload(n, max_score):
        tup = [(2.0 + 0.5 * i, 0.3, 1.4, i % 2, 4.71, 8.0, False, 0, False, False)
               for i in range(n)]
        return {"tuples": tup, "walls": [], "max_score": max_score,
                "duration_s": 2.0 + 0.5 * n, "last_hit_s": tup[-1][0]}
    pay = {"aaa": payload(40, 4000), "bbb": payload(20, 2000), "ccc": payload(2, 0),
           "ddd": payload(30, 3000)}
    monkeypatch.setattr(bs, "load_or_parse", lambda e, *a, **k: pay[e["hash"]])
    monkeypatch.setattr(rp, "_BASE", str(tmp_path))     # pool_cache under tmp
    ss = [{"leaderboard_id": 1, "hash": "aaa", "song": "A", "mapper": "m",
           "difficulty": "Expert", "stars": 5.0, "max_pp": 210.0, "api_max_score": 3990},
          {"leaderboard_id": 2, "hash": "bbb", "song": "B", "mapper": "m",
           "difficulty": "Hard", "stars": 2.0, "max_pp": 84.0, "api_max_score": 2000},
          {"leaderboard_id": 3, "hash": "ccc", "song": "C", "mapper": "m",
           "difficulty": "Easy", "stars": 1.0, "max_pp": 42.0, "api_max_score": 0}]
    bl = [{"leaderboard_id": "x1", "hash": "AAA", "song": "A", "mapper": "m",
           "difficulty": "Expert", "characteristic": "Standard", "stars": 6.0,
           "pass_rating": 3.0, "acc_rating": 7.0, "tech_rating": 2.0,
           "api_max_score": 4000},
          {"leaderboard_id": "x4", "hash": "ddd", "song": "D", "mapper": "m",
           "difficulty": "Expert", "characteristic": "Standard", "stars": 4.0,
           "pass_rating": 2.0, "acc_rating": 6.0, "tech_rating": 1.0,
           "api_max_score": 3000}]
    tasks = bs.merge_catalogs(ss, bl)
    train = [{"leaderboard_id": m["ss"]["leaderboard_id"] if m["ss"] else -1,
              "hash": m["hash"], "song": m["song"], "mapper": m["mapper"],
              "difficulty": m["difficulty"],
              "stars": m["ss"]["stars"] if m["ss"] else 0.0,
              "max_pp": m["ss"]["max_pp"] if m["ss"] else 0.0,
              "level_key": m["level_key"]} for m in tasks]
    cpath = tmp_path / "train_both.json"
    cpath.write_text(json.dumps(train))
    return str(cpath), ss, bl


def test_level_table_both(catalogs):
    from beat_arena_env import ranked_pool as rp
    cpath, ss, bl = catalogs
    T = rp.level_table("both", lead_s=0.5, mirror=True, catalog_path=cpath,
                       ss_catalog=ss, bl_catalog=bl, log=lambda *a, **k: None)
    # ccc is sparse (dropped); 3 levels x 2 copies
    assert T["n_charts"] == 6 and len(T["levels"]) == 3
    assert T["chart_mirrored"] == [0, 1, 0, 1, 0, 1]
    assert T["chart_level"] == [0, 0, 1, 1, 2, 2]
    by = {v["song"]: v for v in T["levels"]}
    assert by["A"]["ss"]["id"] == 1 and by["A"]["ss"]["max_score"] == 3990
    assert by["A"]["bl"]["id"] == "x1" and by["A"]["bl"]["pass"] == 3.0
    assert by["B"]["bl"] is None and by["D"]["ss"] is None
    assert by["A"]["max_score"] == 4000 and by["A"]["n_notes"] == 40
    assert by["A"]["nps"] == pytest.approx(40 / 22.0, abs=1e-3)
    # agrees with the pool meta the trainer would load
    meta = rp._meta_arrays(rp._level_meta(cpath, 0.5, True))
    assert rp.check_levels(T, meta["level_lb"], meta["level_max_score"])[1:] == (0, 0)
    # a subsample indexes the same charts load_ranked_pool keeps
    Ts = rp.level_table("both", lead_s=0.5, mirror=True, catalog_path=cpath,
                        ss_catalog=ss, bl_catalog=bl, sample=4, seed=3,
                        log=lambda *a, **k: None)
    idx = rp._sample_idx(6, 4, 3)
    assert Ts["chart_level"] == [T["chart_level"][i] for i in idx]
    sub = rp._subset_ragged({**{k: v for k, v in meta.items()},
                             "note_off": np.arange(7), "wall_off": np.zeros(7, int),
                             "n_notes": np.ones(6, int), "n_walls": np.zeros(6, int),
                             "hit": np.arange(6.0), "wall_t": np.zeros(0)}, idx)
    assert rp.check_levels(Ts, sub["level_lb"], sub["level_max_score"])[1:] == (0, 0)


def test_check_levels_flags_a_wrong_table(catalogs):
    from beat_arena_env import ranked_pool as rp
    cpath, ss, bl = catalogs
    T = rp.level_table("both", lead_s=0.5, mirror=True, catalog_path=cpath,
                       ss_catalog=ss, bl_catalog=bl, log=lambda *a, **k: None)
    # merge_catalogs order: SS stars ascending (B 2*, A 5*), then BL-only D
    lb = np.array([2, 2, 1, 1, -1, -1])
    assert rp.check_levels(T, lb)[1] == 0
    assert rp.check_levels(T, lb[::-1])[1] > 0
    ms = np.array([2000, 2000, 4000, 4000, 3000, 9999.0])
    assert rp.check_levels(T, None, ms)[2] == 1


def test_backfill_swing_record(tmp_path):
    import metrics_backfill as mb
    camp = tmp_path / "camp" / "runs"
    camp.mkdir(parents=True)
    run = write_run(camp)                       # camp/runs/run
    bf = mb.Backfill(str(tmp_path), verbose=False)
    rec = bf.swing_record({"run_dir": "camp/runs/run"})
    assert rec["path"] == "camp/runs/run/swing.json"
    assert rec["totals"]["ss_levels"] == 2 and rec["swing"]
    assert (run / "swing.json").exists()
    assert bf.swing_record({"run_dir": None}) is None
    (tmp_path / "camp" / "runs" / "empty").mkdir()
    assert bf.swing_record({"run_dir": "camp/runs/empty"}) is None


def test_aggregator_imports_without_torch(tmp_path):
    """metrics_backfill runs it under the system python3 (numpy, no
    torch / gymnasium / Rust env): the PP modules load without the
    package __init__."""
    import subprocess
    import sys
    run = write_run(tmp_path)
    code = ("import sys; sys.modules['torch'] = None; "
            "sys.modules['gymnasium'] = None; sys.path.insert(0, '.'); "
            "import swing_summary as sw; s = sw.update_run(sys.argv[1]); "
            "assert 'beat_arena_env' not in sys.modules; "
            "print(s['totals']['ss_upp'])")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code, str(run)], cwd=here,
                         capture_output=True, text=True, check=True)
    assert float(out.stdout) > 0


def test_resumed_run_counts_its_v1_rows_for_best_plays_only(tmp_path):
    """A run resumed onto the v2 writer: the pre-resume rows (moved to
    ep_stats.v1.f32) feed the best-possible totals — a run's best plays
    are over the WHOLE run — but not plays / accuracy / swing averages
    (they have no swing columns)."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "levels.json").write_text(json.dumps(LEVELS))
    # before the resume: level B's best ever play (7900), no swing data
    old = np.stack([env_row(2, 7900), env_row(0, 9000)])
    es.build_rows(old, META, 1)[:, :16].tofile(run / "ep_stats.v1.f32")
    recs = [{"upd": 50, "ep_stats_episodes": 2},
            {"upd": 51, "ep_stats_episodes": 1}]          # counter resets
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert list(sw.v1_updates(str(run), 0, 2)) == [50, 50]
    w = es.EpStatsWriter(str(run), log=lambda *a, **k: None)
    for update, rows in enumerate(batches(), start=51):
        w.append(es.build_rows(rows, META, update))
    d = _summary(run)
    b = _row(d, "b_Hard_Standard")
    L = d["levels"]
    assert L["ss_pp"][b] == round(scoresaber.pp_value(7900 / 8000, 168.47), 2)
    assert L["best_upd"][b] == 50
    assert L["done"][b] == 1                    # the v2 play only
    # P1 = 80 % L / 90 % R: undiluted by the swing-less v1 play
    assert d["win"]["all"]["pre"][b] == pytest.approx(85, abs=0.1)
    assert d["last_update"] == 54
    # a v1 file appearing later invalidates the cached state
    (run / "ep_stats.v1.f32").unlink()
    d2 = _summary(run)
    assert d2["levels"]["ss_pp"][b] == round(scoresaber.pp_value(7000 / 8000, 168.47), 2)


def test_level_with_only_pre_resume_plays_counts_in_totals(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "levels.json").write_text(json.dumps(LEVELS))
    es.build_rows(np.stack([env_row(2, 7900)]), META, 1)[:, :16].tofile(
        run / "ep_stats.v1.f32")                       # level B, pre-resume
    w = es.EpStatsWriter(str(run), log=lambda *a, **k: None)
    P1 = parts(560, 240, 120, 630, 270, 135)
    w.append(es.build_rows(np.stack([env_row(0, 9000, parts=P1)]), META, 2))
    d = _summary(run)
    t = d["totals"]
    ss_a = round(scoresaber.pp_value(9000 / 9800, 379.05), 2)
    ss_b = round(scoresaber.pp_value(7900 / 8000, 168.47), 2)
    assert t["ss_levels"] == 2
    assert t["ss_upp"] == pytest.approx(ss_a + ss_b, abs=0.01)
    b = _row(d, "b_Hard_Standard")
    assert d["levels"]["done"][b] == 0                 # no swing-era plays
    assert d["win"]["all"]["pre"][b] is None
