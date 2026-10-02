"""viz/replays-site/scripts/publish_replays.py on format-2 replays
(2026-10-02): the score walk reads the event blocks alone, the published
file keeps the column blocks verbatim under a sanitized header (named
r/<id>.rpl.gz), each replay is written as it passes (a failing one is
skipped, exit 1), and --allow-mismatch publishes a replay whose play
differs from the bench."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

from beat_arena_env import replay_format as rf
from beat_arena_env.beatsaver import load_beatmap
from beat_arena_env.rust_env import RustBeatEnv
from test_replay_batched import FIRST, LEVEL, MAP, _actions, _write_map

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "publish_replays",
    os.path.join(HERE, "..", "..", "viz", "replays-site", "scripts",
                 "publish_replays.py"))
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

MH = "sha256:" + "ab" * 32


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pub")
    _write_map(tmp / "map")
    notes, meta = load_beatmap(str(tmp / "map"), "Expert", first_note_s=FIRST)
    env = RustBeatEnv(external_chart=notes, seed=0)
    path = str(tmp / "rep.rpl.gz")
    # a single-catalog bench's level identity (no level_key)
    env.set_record(path, map=MAP, level={**LEVEL, "level_key": None},
                   model_hash=MH)
    env.reset()
    rng, k, done = np.random.default_rng(3), 0, False
    while not done:
        _, _, done, _, _ = env.step(_actions(k, rng))
        k += 1
    _, ticks = rf.read_replay(path)
    return {"tmp": tmp, "path": path, "ticks": ticks}


def test_events_only_is_the_full_decode(recorded):
    meta, data = rf.unpack(rf.read_raw(recorded["path"]))
    evs = rf.events_only(meta, data)
    assert evs == [t["events"] for t in recorded["ticks"]] and any(evs)


def _bench(tmp, got, score_delta=0):
    b = {"model_hash": MH, "levels": [{
        "leaderboard_id": 7, "song_hash": MAP["hash"], "song": "fixture",
        "difficulty": "Expert", "pp": 100.0,
        "score": got["score"] + score_delta, "highest_combo": got["combo"],
        "good_cuts": got["good"], "bad_cuts": got["bad"],
        "misses": got["miss"], "bombs_hit": got["bombs"]}]}
    p = str(tmp / f"bench{score_delta}.json")
    with open(p, "w") as f:
        json.dump(b, f)
    return p


def _publish(monkeypatch, store, bench, files, *extra):
    monkeypatch.setattr(pr, "weights_hash", lambda ck: MH)
    monkeypatch.setattr(sys, "argv", ["publish", "--model", "m", "--bench",
                                      bench, "--ckpt", "x", "--store",
                                      str(store), "--jobs", "1", *extra,
                                      *files])
    pr.main()


def _got(recorded):
    return pr.walk_events(t["events"] for t in recorded["ticks"])


def test_publish_sanitizes_header_keeps_blocks(recorded, monkeypatch,
                                               tmp_path):
    bench = _bench(recorded["tmp"], _got(recorded))
    store = tmp_path / "s"
    _publish(monkeypatch, store, bench, [recorded["path"]])
    man = json.load(open(store / "manifest.json"))
    rel = man["models"]["m"]["replays"]["7"]["file"]
    assert rel.startswith("r/") and rel.endswith(".rpl.gz")
    assert not [f for f in os.listdir(store / "r") if ".tmp" in f]
    src_meta, src_data = rf.unpack(rf.read_raw(recorded["path"]))
    meta, data = rf.unpack(rf.read_raw(str(store / rel)))
    h = meta["header"]
    assert h["model"] == "m" and h["model_hash"] == "ab" * 6
    assert h["format"] == 2 and h["body"] == src_meta["header"]["body"]
    assert meta["blocks"] == src_meta["blocks"] and data == src_data
    # the published file decodes to the recorded ticks
    assert rf.to_ticks(rf.decode(rf.read_raw(str(store / rel)))) == \
        recorded["ticks"]
    # byte-identical re-runs
    blob = open(store / rel, "rb").read()
    _publish(monkeypatch, tmp_path / "t", bench, [recorded["path"]])
    assert open(tmp_path / "t" / rel, "rb").read() == blob


def test_unknown_blocks_and_header_keys_dropped(recorded, monkeypatch,
                                                tmp_path):
    meta, data = rf.unpack(rf.read_raw(recorded["path"]))
    meta["header"]["run_dir"] = "/workspace/secret"
    meta["blocks"].append({"name": "debug", "dtype": "u8", "rows": 1,
                           "len": 3})
    path = str(tmp_path / "x.rpl.gz")
    with open(path, "wb") as f:
        f.write(rf.gz(rf.pack(meta, data + [b"abc"])))
    bench = _bench(recorded["tmp"], _got(recorded))
    _publish(monkeypatch, tmp_path / "s", bench, [path])
    man = json.load(open(tmp_path / "s" / "manifest.json"))
    rel = man["models"]["m"]["replays"]["7"]["file"]
    m2, _ = rf.unpack(rf.read_raw(str(tmp_path / "s" / rel)))
    assert "run_dir" not in m2["header"]
    assert "debug" not in [b["name"] for b in m2["blocks"]]


def test_mismatch_refused_unless_allowed(recorded, monkeypatch, tmp_path):
    got = _got(recorded)
    bench = _bench(recorded["tmp"], got, score_delta=5)
    with pytest.raises(SystemExit):
        _publish(monkeypatch, tmp_path / "s", bench, [recorded["path"]])
    man = json.load(open(tmp_path / "s" / "manifest.json"))
    assert man["models"]["m"]["replays"] == {}
    _publish(monkeypatch, tmp_path / "s", bench, [recorded["path"]],
             "--allow-mismatch")
    man = json.load(open(tmp_path / "s" / "manifest.json"))
    assert man["models"]["m"]["replays"]["7"]["score"] == got["score"]
