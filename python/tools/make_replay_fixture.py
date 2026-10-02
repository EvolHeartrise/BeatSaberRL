"""Format-2 replay fixtures for the viewer's decoder test (viz/app/test/
replay-v2.test.js), derived from the committed scripted episode
viz/app/test/fixtures/v10_scripted_ep00_seed0.rpl.gz:

  - v10_scripted_ep00_seed0_stored.rpl.gz: its first STORED_TICKS ticks
    re-encoded with every derivable field STORED as a block (sabers, head,
    activation, torque) — the writer's fallback path when a derivation
    fails its write-time check;
  - replay_v2_ref.json: what python/beat_arena_env/replay_format.py decodes
    from both files (sampled ticks in full, every tick's events / contacts
    / action) — the JS decoder must agree with it.

python/tests/test_replay_format.py checks the committed files are current.

Run: python tools/make_replay_fixture.py [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beat_arena_env import replay_format as rf

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "viz", "app", "test", "fixtures")
MAIN = os.path.join(FIX, "v10_scripted_ep00_seed0.rpl.gz")
STORED = os.path.join(FIX, "v10_scripted_ep00_seed0_stored.rpl.gz")
REF = os.path.join(FIX, "replay_v2_ref.json")
STORED_TICKS = 120
SAMPLE = (0, 1, 2, 57, 119, 250, 401, 595)


def rec_of(d, T=None):
    """A recorder capture (REC_LANES frame) from a decode() result."""
    T = d["T"] if T is None else min(T, d["T"])
    fr = np.zeros((T, rf.REC_LANES))
    for name, lane, w in (("q", rf.L_Q, 36), ("twist", rf.L_TWIST, 6),
                          ("omega", rf.L_OMEGA, 29),
                          ("sabers", rf.L_SABERS, 12), ("head", rf.L_HEAD, 7),
                          ("feet", rf.L_FEET, 6), ("activation", rf.L_ACT, 4),
                          ("torque", rf.L_TORQUE, 4)):
        fr[:, lane:lane + w] = d[name][:T]
    fr[:, rf.L_REWARD] = d["reward"][:T]
    nc, ne = int(d["contact_off"][T]), int(d["event_off"][T])
    return {"lanes": rf.REC_LANES, "tick": np.arange(T),
            "frame": fr.ravel(), "action": d["action"][:T].ravel(),
            "contact_off": d["contact_off"][:T + 1],
            "contact_pair": d["contact_pair"][:nc],
            "contact_val": np.concatenate([d["contact_p"][:nc],
                                           d["contact_j"][:nc, None]],
                                          1).ravel(),
            "event_off": d["event_off"][:T + 1],
            "event_note": d["event_note"][:ne],
            "event_code": d["event_code"][:ne],
            "event_val": d["event_val"][:ne].ravel()}


def stored_payload():
    """The STORED fixture's payload: every derivable field stored."""
    d = rf.decode(rf.read_raw(MAIN))
    tol = (rf.TOL_POS, rf.TOL_ANG, rf.TOL_ACT, rf.TOL_TORQUE)
    rf.TOL_POS = rf.TOL_ANG = rf.TOL_ACT = rf.TOL_TORQUE = -1.0
    try:
        st = {}
        raw = rf.encode({"header": d["header"]}, rec_of(d, STORED_TICKS),
                        stats=st)
    finally:
        rf.TOL_POS, rf.TOL_ANG, rf.TOL_ACT, rf.TOL_TORQUE = tol
    assert st["stored"] == ["sabers", "head", "activation", "torque"]
    return raw


def reference(stored_raw):
    out = {}
    for name, raw in (("main", rf.read_raw(MAIN)), ("stored", stored_raw)):
        d = rf.decode(raw)
        ticks = rf.to_ticks(d)
        out[name] = {
            "T": d["T"],
            "sample": {str(k): ticks[k] for k in SAMPLE if k < d["T"]},
            "events": [t["events"] for t in ticks],
            "contacts": [t["contacts"] for t in ticks],
            "actions": [t["action"] for t in ticks]}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true",
                   help="verify the committed files instead of writing")
    a = p.parse_args()
    raw = stored_payload()
    ref = reference(raw)
    if a.check:
        assert rf.read_raw(STORED) == raw, f"{STORED} is stale"
        assert json.load(open(REF)) == json.loads(json.dumps(ref)), \
            f"{REF} is stale"
        print("replay fixtures current")
        return
    with open(STORED, "wb") as f:
        f.write(rf.gz(raw, 9))
    with open(REF, "w") as f:
        json.dump(ref, f, separators=(",", ":"))
    print(f"wrote {STORED} ({os.path.getsize(STORED)} B), "
          f"{REF} ({os.path.getsize(REF)} B)")


if __name__ == "__main__":
    main()
