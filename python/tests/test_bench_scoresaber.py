"""ScoreSaber bench: PP curve, combo multiplier walk, catalog, merge.

All offline — network functions are monkeypatched. The multiplier rules
are pinned against ScoreSaber's own max-score formula 115*(8n-63).
"""

import io
import json
import os
import zipfile

import numpy as np
import pytest

from beat_arena_env import scoresaber
from beat_arena_env.chart import ChartConfig, Note, generate_chart
from beat_arena_env.constants import (KIND_BOMB, KIND_CHAIN_HEAD,
                                      KIND_CHAIN_LINK)

import bench_scoresaber


# ---------------------------------------------------------------------------
# PP curve + weighting
# ---------------------------------------------------------------------------

def test_pp_curve_anchors():
    assert scoresaber.pp_multiplier(0.95) == pytest.approx(1.0)
    assert scoresaber.pp_multiplier(1.0) == pytest.approx(5.367394282890631)
    assert scoresaber.pp_multiplier(1.5) == pytest.approx(5.367394282890631)
    assert scoresaber.pp_multiplier(0.0) == 0.0
    # linear midpoint of the (0.95, 1.0)-(0.955, 1.0388...) segment
    mid = (1.0 + 1.0388633331418984) / 2
    assert scoresaber.pp_multiplier(0.9525) == pytest.approx(mid)
    # monotone over the whole range
    accs = np.linspace(0, 1, 400)
    mults = [scoresaber.pp_multiplier(a) for a in accs]
    assert all(b >= a for a, b in zip(mults, mults[1:]))


def test_pp_value_uses_max_pp():
    # 95% on a 3.66-star map: exactly its maxPP
    max_pp = 3.66 * scoresaber.PP_PER_STAR
    assert scoresaber.pp_value(0.95, max_pp) == pytest.approx(max_pp)


def test_weighted_total_pp():
    total = scoresaber.weighted_total_pp([100.0, 100.0, 100.0])
    assert total == pytest.approx(100.0 * (1 + 0.965 + 0.965 ** 2))
    # order-independent, best play weighted first
    assert scoresaber.weighted_total_pp([50.0, 200.0]) == pytest.approx(
        200.0 + 0.965 * 50.0)
    assert scoresaber.weighted_total_pp([]) == 0.0


# ---------------------------------------------------------------------------
# Combo multiplier
# ---------------------------------------------------------------------------

def _full_combo_score(n_notes, pts=115):
    walk = scoresaber.ComboTracker()
    for i in range(n_notes):
        walk.on_cut(i)
        walk.on_points(i, pts)
    return walk


def test_multiplier_matches_scoresaber_formula():
    # ScoreSaber's own max score for n plain notes: 115*(8n-63), n >= 14
    for n in (14, 20, 100, 563):
        assert _full_combo_score(n).score == 115 * (8 * n - 63)
    # small-n pieces of the same formula
    assert _full_combo_score(1).score == 115           # 1 unit at 1x
    assert _full_combo_score(5).score == 115 * (2 * 5 - 1)   # 1x1 + 4x2
    assert _full_combo_score(13).score == 115 * (4 * 13 - 11)


def test_break_halves_and_resets_progress():
    walk = scoresaber.ComboTracker()
    for i in range(5):                   # units at 1,2,2,2,2 -> mult 2, bar 3/4
        walk.on_cut(i)
        walk.on_points(i, 115)
    assert walk.multiplier == 2 and walk.combo == 5
    walk.on_break()                      # halve 2 -> 1, bar resets
    assert walk.multiplier == 1 and walk.combo == 0 and walk.breaks == 1
    walk.on_cut(9)                       # bar 1/2: still 1x
    walk.on_points(9, 115)
    assert walk.score == 115 * (1 + 2 + 2 + 2 + 2 + 1)
    assert walk.max_combo == 5


def test_post_points_use_cut_time_multiplier():
    walk = scoresaber.ComboTracker()
    walk.on_cut(0)                       # note 0 commits at 1x
    walk.on_points(0, 85)
    for i in range(1, 3):                # notes 1-2 at 2x
        walk.on_cut(i)
        walk.on_points(i, 85)
    walk.on_break()                      # 2x -> 1x
    walk.on_points(0, 30)                # note 0's follow-through: STILL 1x
    walk.on_points(1, 30)                # note 1's: still 2x
    assert walk.score == 85 * (1 + 2 + 2) + 30 * 1 + 30 * 2


def test_consume_event_stream():
    walk = scoresaber.ComboTracker()
    walk.consume([(0, "ok", 85.0), (0, "post", 30.0)])
    walk.consume([(1, "wrong_hand", 0.0), (2, "miss", 0.0),
                  (3, "bomb", 0.0)])
    walk.consume([(4, "ok", 85.0)])
    assert walk.breaks == 3 and walk.combo == 1 and walk.max_combo == 1
    assert walk.score == 115 + 85


def test_max_multiplied_score_kinds():
    def note(t, kind):
        return Note(hit_time=t, col=1, row=1, color=1, angle=0.0,
                    speed=4.0, kind=kind)
    chart = [note(0.5, KIND_BOMB)]                     # bombs excluded
    chart += [note(1.0 + 0.1 * i, 0) for i in range(3)]
    chart += [note(2.0, KIND_CHAIN_HEAD), note(2.1, KIND_CHAIN_LINK),
              note(2.2, KIND_CHAIN_LINK)]
    # walk: 115@1x, 115@2x, 115@2x, 85@2x, 20@2x (bar fills -> 4x on next),
    # 20@4x
    expected = 115 + 230 + 230 + 170 + 40 + 80
    assert scoresaber.max_multiplied_score(chart) == expected
    # order independence: same chart shuffled sorts by hit_time
    assert scoresaber.max_multiplied_score(chart[::-1]) == expected


# ---------------------------------------------------------------------------
# Catalog + map cache (network monkeypatched)
# ---------------------------------------------------------------------------

def _fake_pages(per_page=3):
    boards = []
    for i in range(7):
        boards.append({
            "id": 100 + i, "songHash": f"AB{i:02d}",
            "songName": f"song{i}", "levelAuthorName": "mapper",
            "difficulty": {"gameMode": "SoloStandard",
                           "difficulty": [1, 3, 5, 7, 9][i % 5]},
            "stars": 5.0 - 0.5 * i, "maxPP": (5.0 - 0.5 * i) * 42.117,
            "maxScore": 1000 + i})
    # one non-Standard board that must be filtered out
    boards[3] = dict(boards[3], difficulty={"gameMode": "SoloOneSaber",
                                            "difficulty": 7})
    pages = [boards[i:i + per_page] for i in range(0, len(boards), per_page)]
    return pages, len(boards)


def test_fetch_ranked_catalog_paginates(monkeypatch):
    pages, total = _fake_pages()

    def fake_get(url, **kw):
        page = int(url.split("page=")[1].split("&")[0])
        return {"leaderboards": pages[page - 1],
                "metadata": {"total": total, "itemsPerPage": 3}}

    monkeypatch.setattr(scoresaber, "_get_json", fake_get)
    monkeypatch.setattr(scoresaber.time, "sleep", lambda s: None)
    cat = scoresaber.fetch_ranked_catalog()
    assert len(cat) == 6                       # 7 minus the non-Standard one
    assert [e["stars"] for e in cat] == sorted(e["stars"] for e in cat)
    assert cat[0]["difficulty"] in ("Easy", "Normal", "Hard", "Expert",
                                    "ExpertPlus")
    assert cat[0]["hash"] == cat[0]["hash"].lower()
    assert all(e["max_pp"] > 0 and e["api_max_score"] for e in cat)


def test_ensure_map_dats_strips_media(monkeypatch, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Info.dat", "{}")
        z.writestr("Easy.dat", "{}")
        z.writestr("song.egg", b"\x00" * 512)
        z.writestr("cover.jpg", b"\x00" * 64)
    blob = buf.getvalue()

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(
        scoresaber, "_get_json",
        lambda url, **kw: {"versions": [{"hash": "abcd",
                                         "downloadURL": "http://x/zip"}]})
    monkeypatch.setattr(scoresaber.urllib.request, "urlopen",
                        lambda req, timeout=0, context=None: FakeResp(blob))
    d = scoresaber.ensure_map_dats("ABCD", str(tmp_path))
    assert sorted(os.listdir(d)) == ["Easy.dat", "Info.dat"]
    # cache hit: no network the second time
    monkeypatch.setattr(scoresaber, "_get_json",
                        lambda *a, **kw: pytest.fail("network on cache hit"))
    assert scoresaber.ensure_map_dats("abcd", str(tmp_path)) == d


# ---------------------------------------------------------------------------
# Bench command pieces
# ---------------------------------------------------------------------------

def test_model_hash_stable_and_discriminating():
    import torch
    sd1 = {"a": torch.zeros(2, 3), "b": torch.ones(4)}
    sd2 = {"a": torch.zeros(2, 3), "b": torch.ones(4)}
    assert bench_scoresaber.model_hash(sd1) == bench_scoresaber.model_hash(sd2)
    sd2["b"][0] = 2.0
    assert bench_scoresaber.model_hash(sd1) != bench_scoresaber.model_hash(sd2)


def test_save_and_merge_results(tmp_path):
    out = str(tmp_path / "r.json")
    data = {"checkpoint": "x.pt", "checkpoint_name": "run/x.pt",
            "model_hash": "sha256:aa"}
    levels = {
        1: {"leaderboard_id": 1, "stars": 2.0, "pp": 100.0},
        2: {"leaderboard_id": 2, "stars": 1.0, "pp": 50.0},
        3: {"leaderboard_id": 3, "stars": 3.0, "song_hash": "ff",
            "song": "s", "difficulty": "Expert", "skipped": "no notes"},
    }
    bench_scoresaber.save_results(out, data, levels)
    loaded = json.load(open(out))
    assert loaded["n_evaluated"] == 2 and loaded["n_skipped"] == 1
    assert loaded["total_weighted_pp"] == pytest.approx(
        100.0 + 0.965 * 50.0, abs=0.01)
    assert [e["stars"] for e in loaded["levels"]] == [1.0, 2.0, 3.0]

    # same hash: loads for merging
    again = bench_scoresaber.load_results(out, "sha256:aa")
    assert {e["leaderboard_id"] for e in again["levels"]} == {1, 2, 3}
    # different weights, same file: refuse
    with pytest.raises(SystemExit):
        bench_scoresaber.load_results(out, "sha256:bb")


@pytest.mark.skip(reason="pending P4 torch mirror: the policy is still "
                  "v9-shaped (obs 290 / 14 heads) and cannot drive the "
                  "v10 env (obs 340 / 29 heads); P4 restores this")
def test_greedy_rollout_smoke():
    """Full finish_slot path on a tiny procedural chart, offline, random
    policy: result well-formed and internally consistent."""
    import torch
    from beat_arena_env.rust_env import RustBeatEnv
    from policy import NotePolicy

    cfg = ChartConfig(chart_len_s=6.0, bomb_prob=0.25)
    notes = generate_chart(cfg, np.random.default_rng(3))
    env = RustBeatEnv(external_chart=notes, seed=0)
    obs, _ = env.reset()
    policy = NotePolicy()
    slot = {"entry": {"leaderboard_id": 9, "hash": "aa", "song": "s",
                      "mapper": "m", "difficulty": "Easy", "stars": 1.0,
                      "max_pp": 42.117, "api_max_score": None},
            "walk": scoresaber.ComboTracker(),
            "max_score": scoresaber.max_multiplied_score(notes),
            "duration_s": 6.0}
    done = False
    while not done:
        with torch.no_grad():
            logits, _ = policy(torch.as_tensor(obs).unsqueeze(0))
        act = [int(l.argmax()) for l in logits]
        obs, _, done, _, info = env.step(act)
        slot["walk"].consume(info["events"])
    w = slot["walk"]
    r = bench_scoresaber.finish_slot(slot, info,
                                     (w.score, w.max_combo, w.breaks))
    assert r["max_score"] > 0 and 0 <= r["score"] <= r["max_score"]
    assert 0.0 <= r["accuracy"] <= 1.0 and r["pp"] >= 0.0
    assert r["good_cuts"] + r["bad_cuts"] + r["misses"] == r["n_notes"]
    assert r["highest_combo"] <= r["n_notes"]
    assert isinstance(r["full_combo"], bool)


def test_vectorized_events_match_serial():
    """The vec bench path (VecBeatSim.step's per-env ordered events)
    reproduces the serial RustBeatEnv rollout exactly: same events in
    the same order, same combo-walk score, same counters — the property
    the ScoreSaber multiplier walk depends on."""
    import beat_arena_rs
    from beat_arena_env.chart import note_tuple
    from beat_arena_env.constants import EVENT_REASONS
    from beat_arena_env.rust_env import RustBeatEnv

    charts = [generate_chart(ChartConfig(chart_len_s=6.0, colors="both",
                                         bomb_prob=0.25),
                             np.random.default_rng(s)) for s in (3, 4)]
    acts = np.random.default_rng(7).integers(0, 3, size=(400, 2, 29))

    serial = []
    for ci, chart in enumerate(charts):
        env = RustBeatEnv(external_chart=chart, seed=0)
        env.reset()
        walk, evs, done, t = scoresaber.ComboTracker(), [], False, 0
        while not done:
            _, _, done, _, info = env.step(acts[t, ci])
            walk.consume(info["events"])
            evs.extend(info["events"])
            t += 1
        serial.append((evs, walk, info))

    vec = beat_arena_rs.VecBeatSim(2, max_steps=100_000_000)
    for i, chart in enumerate(charts):
        vec.set_chart(i, [note_tuple(n) for n in chart])
    walks = [scoresaber.ComboTracker(), scoresaber.ComboTracker()]
    vec_evs = [[], []]
    counters, t = [None, None], 0
    while not all(c is not None for c in counters):
        _, _, dones, events = vec.step(
            np.ascontiguousarray(acts[t], dtype=np.int64))
        for i, note, code, pts, _pre, _acc in events:
            if counters[i] is None:
                ev = (int(note), EVENT_REASONS[code], float(pts))
                walks[i].consume([ev])
                vec_evs[i].append(ev)
        for i in range(2):
            if bool(dones[i]) and counters[i] is None:
                counters[i] = vec.counters(i)
                vec.set_chart(i, [])        # idle, as the bench does
        t += 1

    for i, (evs, walk, info) in enumerate(serial):
        assert vec_evs[i] == evs
        assert walks[i].score == walk.score
        assert walks[i].max_combo == walk.max_combo
        g, b, m, n, _, bombs = counters[i]
        assert (g, b, m, n, bombs) == (info["good_cuts"], info["bad_cuts"],
                                       info["misses"], info["n_notes"],
                                       info["bombs_hit"])


def test_rust_combo_walk_matches_python():
    """The Rust-side combo walk (VecBeatSim combo_walk=True, 2026-07-25)
    reproduces the Python ComboTracker exactly over the same episodes:
    multiplied score, max combo, and break count."""
    import beat_arena_rs
    from beat_arena_env.chart import note_tuple
    from beat_arena_env.constants import EVENT_REASONS

    try:
        vec = beat_arena_rs.VecBeatSim(2, max_steps=100_000_000,
                                       combo_walk=True)
    except TypeError:
        pytest.skip("extension predates combo_walk (rebuild pending)")
    charts = [generate_chart(ChartConfig(chart_len_s=6.0, colors="both",
                                         bomb_prob=0.25),
                             np.random.default_rng(s)) for s in (3, 4)]
    for i, chart in enumerate(charts):
        vec.set_chart(i, [note_tuple(n) for n in chart])
    acts = np.random.default_rng(7).integers(0, 3, size=(400, 2, 29))
    py = [scoresaber.ComboTracker(), scoresaber.ComboTracker()]
    rust_result, t = [None, None], 0
    while not all(r is not None for r in rust_result):
        _, _, dones, events = vec.step(
            np.ascontiguousarray(acts[t], dtype=np.int64))
        for i, note, code, pts, _pre, _acc in events:
            if rust_result[i] is None:
                py[i].consume([(note, EVENT_REASONS[code], pts)])
        for i in range(2):
            if bool(dones[i]) and rust_result[i] is None:
                rust_result[i] = vec.combo(i)   # before set_chart resets
                vec.set_chart(i, [])
        t += 1
    for i in range(2):
        score, max_combo, breaks = rust_result[i]
        assert score == pytest.approx(py[i].score, abs=1e-9)
        assert max_combo == py[i].max_combo
        assert breaks == py[i].breaks
        assert py[i].score > 0                  # episodes actually scored
