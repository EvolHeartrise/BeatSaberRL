"""Fast replay generation (2026-10-02): the VecBeatSim recorder captures
exactly what RustBeatEnv's recorder captures (the two format-2 files are
byte-identical), and make_replays_batched.py (batched sim + one
fixed-width forward per step, CPU eager here) plays exactly what the
single-env path plays at the same forward width — then --write-back makes
the source benches and the composite agree with it. CPU only, synthetic
maps, a tiny random policy."""
import json
import os

import numpy as np
import pytest
import torch

import beat_arena_rs
from beat_arena_env.beatsaver import load_beatmap
from beat_arena_env.chart import note_tuple, wall_tuple
from beat_arena_env.constants import OBS_DIM
from beat_arena_env.replay_format import read_raw, read_replay, write_replay
from beat_arena_env.rust_env import RustBeatEnv, record_header
from test_convert_scoresaber import BOMB, NJS, RAW_NOTES

FIRST = 0.35
MAP = {"hash": "ab" * 20, "difficulty": "Expert", "song": "fixture",
       "audio_offset_s": -0.5}
LEVEL = {"level_key": "k", "characteristic": "Standard",
         "ss_leaderboard_id": 7, "bl_leaderboard_id": None}


def _write_map(d, shift_beats=0.0):
    d.mkdir(parents=True, exist_ok=True)
    info = {"_beatsPerMinute": 120.0, "_songName": "fixture",
            "_difficultyBeatmapSets": [{
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [
                    {"_difficulty": "Expert", "_beatmapFilename": "Expert.dat",
                     "_noteJumpMovementSpeed": NJS}]}]}
    (d / "Info.dat").write_text(json.dumps(info))
    v2 = {"_version": "2.0.0", "_notes":
          [{"_time": t + shift_beats, "_lineIndex": x, "_lineLayer": y,
            "_type": c, "_cutDirection": dd} for t, x, y, c, dd in RAW_NOTES]
          + [{"_time": BOMB[0] + shift_beats, "_lineIndex": BOMB[1],
              "_lineLayer": BOMB[2], "_type": 3, "_cutDirection": 0}]}
    (d / "Expert.dat").write_text(json.dumps(v2))


def _actions(k, rng):
    """A whip on the right arm, then seeded random gears."""
    a = np.zeros(29, dtype=np.int64)
    if k < 6:
        a[10] = 1
    elif k < 15:
        a[10] = 2
    else:
        a = rng.integers(0, 3, size=29)
    return a


def test_recorder_writer_matches_rust_env_recording(tmp_path):
    """60 Hz, action repeat 2, on a map with cuts: env 0 of a two-env
    VecBeatSim (env 1 at 30 Hz / repeat 1 alongside) records what
    RustBeatEnv records for the same actions — every sim tick: the two
    format-2 files are byte-identical (and so deterministic), with the
    full header written up front."""
    _write_map(tmp_path / "map")
    notes, meta = load_beatmap(str(tmp_path / "map"), "Expert",
                               first_note_s=FIRST)
    env = RustBeatEnv(external_chart=notes, external_walls=meta["walls"],
                      seed=0, tick_hz=60, action_repeat=2)
    ref = str(tmp_path / "ref.rpl.gz")
    env.set_record(ref, map=MAP, level=LEVEL, model_hash="sha256:ff")
    env.reset()
    max_steps = int((notes[-1].hit_time + 0.5) * 30) + 90
    sim = beat_arena_rs.VecBeatSim(2, max_steps=max_steps, combo_walk=True,
                                   return_events=False,
                                   tick_hz_env=[60, 30],
                                   action_repeat_env=[2, 1])
    assert (sim.env_tick_hz(0), sim.env_tick_hz(1)) == (60, 30)
    tup = [note_tuple(n) for n in notes]
    walls = [wall_tuple(w) for w in meta["walls"]]
    for i in range(2):
        sim.set_chart(i, tup, walls)
    sim.set_record_env(0, True)
    obs = np.zeros((2, OBS_DIM), dtype=np.float32)
    rng, k, done = np.random.default_rng(3), 0, False
    while not done:
        a = _actions(k, rng)
        _, _, done, _, info = env.step(a)
        _, dones, _ = sim.step_into(np.stack([a, a]), obs)
        assert bool(dones[0]) == done
        k += 1
    assert sim.counters(0)[:3] == (info["good_cuts"], info["bad_cuts"],
                                   info["misses"])
    rec = sim.take_record_env(0)
    head = record_header(tup, walls, tick_hz=60, action_repeat=2,
                         tau_scale=[1.0] * 16, gears=3, map=MAP,
                         level=LEVEL, model_hash="sha256:ff")
    out = str(tmp_path / "new.rpl.gz")
    stats = {}
    n = write_replay(out, head, rec, stats=stats)
    assert open(ref, "rb").read() == open(out, "rb").read()
    h, ticks = read_replay(out)
    assert n == len(ticks) > 2 * (k - 1)
    assert [t["t"] for t in ticks] == list(range(n))
    assert any(t["events"] for t in ticks), "fixture produced no events"
    assert (h["map"], h["level"], h["model_hash"]) == (MAP, LEVEL,
                                                       "sha256:ff")
    assert h["format"] == 2 and h["tick_hz"] == 60
    assert h["action_repeat"] == 2
    # every derived field reproduced within tolerance: nothing stored
    assert stats["stored"] == [], stats
    assert stats["activation"] < 1e-9 and stats["torque"] < 1e-9
    write_replay(str(tmp_path / "b.rpl.gz"), head, rec)
    assert read_raw(out) == read_raw(str(tmp_path / "b.rpl.gz"))
    # the recorder keeps running empty after a take; off = an error
    assert len(sim.take_record_env(0)["tick"]) == 0
    sim.set_record_env(0, False)
    with pytest.raises(RuntimeError):
        sim.take_record_env(0)


def test_vec_beat_sim_rate_args_validated():
    with pytest.raises(ValueError):
        beat_arena_rs.VecBeatSim(2, tick_hz_env=[30])
    with pytest.raises(ValueError):
        beat_arena_rs.VecBeatSim(2, tick_hz_env=[30, 45])
    with pytest.raises(ValueError):
        beat_arena_rs.VecBeatSim(2, action_repeat_env=[1, 0])


# --- make_replays_batched end to end ---------------------------------------

W = 8                    # forward width of both paths


def _tiny_ckpt(path):
    import policy as P
    torch.manual_seed(0)
    p = P.NotePolicy(embed=32, trunk_dim=64, n_blocks=1, n_heads=2,
                     n_gears=5, ffn_mult=2, note_geo=True, value_bins=101,
                     value_vmax=6.0, value_sigma=0.75, z_asinh=2.0)
    torch.save({"arch": dict(p.arch), "state_dict": p.state_dict()}, path)
    return p


def _row(key, h, lb, score, ss_pp, bl_pp):
    play = {"score": score, "max_score": 1000, "accuracy": score / 1000,
            "good_cuts": 1, "bad_cuts": 0, "misses": 3, "bombs_hit": 0,
            "highest_combo": 1, "full_combo": False, "breaks": 1,
            "fell": True, "terminated_tick": 20, "ticks": 20}
    return {"level_key": key, "song_hash": h, "song": f"song {lb}",
            "mapper": "m", "difficulty": "Expert",
            "characteristic": "Standard", "ss_stars": 5.0, "bl_stars": 6.0,
            "duration_s": 3.0, "n_notes": 4, **play,
            "ss": {"leaderboard_id": lb, "stars": 5.0, "max_pp": 300.0,
                   "api_max_score": None, "accuracy": score / 1000,
                   "pp": ss_pp},
            "bl": {"leaderboard_id": f"b{lb}", "stars": 6.0,
                   "pass_rating": 5.0, "acc_rating": 7.0,
                   "tech_rating": 4.0, "predicted_acc": 0.95,
                   "api_max_score": None, "accuracy": score / 1000,
                   "pp": bl_pp}}


def _single_env_play(payload, tau, tick_hz, repeat, policy):
    """make_top10_replays' loop on the single-env sim (the bench's
    max_steps): the play's obs in row 0 of a zero-padded width-W batch,
    eager CPU forward, the action held `repeat` sim ticks."""
    from beat_arena_rs32 import RawBeatSim
    sim = RawBeatSim(max_steps=100_000_000, gears=policy.n_gears,
                     tick_hz=tick_hz)
    sim.set_tau_scale(tau)
    obs = sim.set_chart(payload["tuples"], payload["walls"])
    buf = torch.zeros(W, OBS_DIM)
    acts, done = [], False
    while not done:
        buf[0].copy_(torch.as_tensor(obs, dtype=torch.float32))
        with torch.no_grad():
            a = policy.forward_stacked(buf)[0].argmax(-1)[0].tolist()
        for _ in range(repeat):
            obs, _, done, _ = sim.step([int(x) for x in a])
            acts.append([int(x) for x in a])
            if done:
                break
    return acts, (sim.good_cuts, sim.bad_cuts, sim.misses)


def test_batched_tool_matches_single_env_and_writes_back(tmp_path,
                                                         monkeypatch):
    import make_replays_batched as mrb
    import bench_scoresaber as bs
    from tools import bench_combine
    monkeypatch.setenv("BEAT_Z_ASINH", repr(2.0))
    ck = str(tmp_path / "tiny.pt")
    policy = _tiny_ckpt(ck).eval()
    mhash = bs.model_hash(torch.load(ck, weights_only=True)["state_dict"])
    mc, cc = tmp_path / "maps", tmp_path / "charts"
    hashes = ["cd" * 20, "ef" * 20]
    for j, h in enumerate(hashes):
        _write_map(mc / h, shift_beats=0.5 * j)
    keys = [f"{h}_Expert_Standard" for h in hashes]
    tau = [0.9] * 16
    srcd = tmp_path / "src"
    srcd.mkdir()
    for name, hz, sc in (("a30.json", 30, (500, 400)),
                         ("b60.json", 60, (450, 600))):
        rows = [_row(keys[0], hashes[0], 11, sc[0], sc[0] / 10, sc[0] / 5),
                _row(keys[1], hashes[1], 12, sc[1], sc[1] / 10, sc[1] / 5)]
        (srcd / name).write_text(json.dumps({
            "model_hash": mhash, "catalog": "both", "sim": "rust",
            "sim_dtype": "f32", "forward_width": 1024, "tick_hz": hz,
            "tau_scale": tau, "levels": rows}))
    comp = bench_combine.combine([str(srcd / "a30.json"),
                                  str(srcd / "b60.json")])
    cpath = str(tmp_path / "best.json")
    with open(cpath, "w") as f:
        json.dump(comp, f)
    argv = ["--ckpt", ck, "--composite", cpath, "--sources", str(srcd),
            "--tag", "t", "--all", "--forward-width", str(W),
            "--device", "cpu", "--writers", "0",
            "--out-dir", str(tmp_path / "out"),
            "--map-cache", str(mc), "--chart-cache", str(cc)]
    res = mrb.main(argv + ["--write-back"])
    assert all(r["status"] in ("match", "mismatch") for r in res), res
    # every pick of the composite got exactly one job (deduplicated)
    picks = sorted(p for r in res for p in r["job"]["picks"])
    want = sorted((r["level_key"], b) for r in comp["levels"]
                  for b in ("ss", "bl") if r.get(b))
    assert picks == want
    for r in res:
        src, job = r["job"]["src"], r["job"]
        h, ticks = read_replay(r["write"]["out"])
        assert h["tick_hz"] == src.tick_hz and h["tau_scale"] == tau
        assert h["map"]["hash"] == job["row"]["song_hash"]
        assert h["level"]["level_key"] == job["row"]["level_key"]
        assert h["model_hash"] == mhash
        # batched play == single-env play at the same forward width
        acts, counts = _single_env_play(job["payload"], tau, src.tick_hz,
                                        src.repeat, policy)
        assert [t["action"] for t in ticks] == acts
        assert counts == (r["row"]["good_cuts"], r["row"]["bad_cuts"],
                          r["row"]["misses"])
    # write-back: sources and composite now hold the replays' plays
    comp2 = json.load(open(cpath))
    assert comp2["replay_overrides"]
    by_key = {r["level_key"]: r for r in comp2["levels"]}
    for r in res:
        src = json.load(open(r["job"]["src"].path))
        row = {x["level_key"]: x for x in src["levels"]}[
            r["job"]["row"]["level_key"]]
        assert mrb.play_of(row) == mrb.play_of(r["row"])
        if r["status"] == "mismatch":
            assert row["replay_override"]["tool"] == "make_replays_batched.py"
            assert src["replay_overrides"]
        for key, board in r["job"]["picks"]:
            pick = by_key[key] if board == "ss" else by_key[key]["bl_play"]
            assert mrb.play_of(pick) == mrb.play_of(r["row"])
            assert by_key[key][board] == r["row"][board]
    total = comp2["ss_total_weighted_pp"]
    assert total == bench_combine.recompute_totals(
        json.loads(json.dumps(comp2)))["ss_total_weighted_pp"]
    # a second run reproduces the written-back rows exactly
    res2 = mrb.main(argv + ["--tag", "t2"])
    assert [r["status"] for r in res2] == ["match"] * len(res)


def test_fast_mode_switches_to_exact_for_large_jobs():
    """--fast only applies below FAST_MAX_PLAYS plays; bigger jobs run the
    bench-width forward, which is faster there and bench-exact."""
    import make_replays_batched as mrb
    n = mrb.FAST_MAX_PLAYS
    assert mrb.fast_mode(True, 20) and mrb.fast_mode(True, n - 1)
    assert not mrb.fast_mode(True, n) and not mrb.fast_mode(True, 7455)
    assert not mrb.fast_mode(False, 20)
