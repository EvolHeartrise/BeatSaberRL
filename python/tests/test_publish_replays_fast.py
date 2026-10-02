"""viz/replays-site/scripts/publish_replays.py (2026-10-02): compact
.jsonl.gz replays keep their tick lines verbatim (only event lines are
parsed), the store gets the same bytes as from the plain recording, each
replay is written as it passes (a failing one is skipped, exit 1), and
--allow-mismatch publishes a replay whose play differs from the bench."""
import gzip
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

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
    plain = str(tmp / "rep.jsonl")
    # a single-catalog bench's level identity (no level_key)
    env.set_record(plain, map=MAP, level={**LEVEL, "level_key": None},
                   model_hash=MH)
    env.reset()
    rng, k, done = np.random.default_rng(3), 0, False
    while not done:
        _, _, done, _, _ = env.step(_actions(k, rng))
        k += 1
    lines = open(plain).read().split("\n")
    compact = str(tmp / "rep.jsonl.gz")
    with gzip.open(compact, "wt") as f:
        for l in lines:
            if l.strip():
                f.write(json.dumps(json.loads(l), separators=(",", ":"))
                        + "\n")
    return {"tmp": tmp, "plain": plain, "compact": compact}


def test_compact_lines_kept_verbatim(recorded, monkeypatch):
    plain = pr.read_lines(recorded["plain"])[1:]
    compact = pr.read_lines(recorded["compact"])[1:]
    parsed = []
    real_loads = json.loads
    monkeypatch.setattr(pr.json, "loads",
                        lambda s: parsed.append(s) or real_loads(s))
    a, ev_a = pr.clean_ticks(plain, set())
    n_plain = len(parsed)
    parsed.clear()
    b, ev_b = pr.clean_ticks(compact, set())
    assert a == b == compact and ev_a == ev_b and ev_a
    assert n_plain == len(plain)
    # compact: the first event-free line (proof) + the event lines only
    assert len(parsed) == 1 + sum(1 for l in compact
                                  if not l.endswith(',"events":[]}'))
    assert pr.walk_events(ev_b) == pr.walk_events(
        [json.loads(l) for l in plain])


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


def test_publish_plain_and_compact_same_bytes(recorded, monkeypatch,
                                              tmp_path):
    got = pr.walk_events([json.loads(l)
                          for l in pr.read_lines(recorded["plain"])[1:]])
    bench = _bench(recorded["tmp"], got)
    blobs = []
    for src in ("plain", "compact"):
        store = tmp_path / src
        _publish(monkeypatch, store, bench, [recorded[src]])
        man = json.load(open(store / "manifest.json"))
        rel = man["models"]["m"]["replays"]["7"]["file"]
        blobs.append(open(store / rel, "rb").read())
        assert not [f for f in os.listdir(store / "r") if ".tmp" in f]
    assert blobs[0] == blobs[1]
    head = json.loads(gzip.decompress(blobs[0]).split(b"\n")[0])["header"]
    assert head["model"] == "m" and head["model_hash"] == "ab" * 6


def test_mismatch_refused_unless_allowed(recorded, monkeypatch, tmp_path):
    got = pr.walk_events([json.loads(l)
                          for l in pr.read_lines(recorded["plain"])[1:]])
    bench = _bench(recorded["tmp"], got, score_delta=5)
    with pytest.raises(SystemExit):
        _publish(monkeypatch, tmp_path / "s", bench, [recorded["compact"]])
    man = json.load(open(tmp_path / "s" / "manifest.json"))
    assert man["models"]["m"]["replays"] == {}
    _publish(monkeypatch, tmp_path / "s", bench, [recorded["compact"]],
             "--allow-mismatch")
    man = json.load(open(tmp_path / "s" / "manifest.json"))
    assert man["models"]["m"]["replays"]["7"]["score"] == got["score"]
