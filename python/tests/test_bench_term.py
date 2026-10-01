"""Review B09 / B14 (2026-09-30): the bench records combo breaks, wall
entries and how each level ended (fell / tilted / envelope /
terminated_tick), from both engines — lib.rs VecBeatSim::term on the rust
path, bench_rec cols 17/18 on the CUDA path — and bench_compare refuses
to diff files benched under different regimes (tau_scale, mirror,
scoring_version, sim_dtype) unless --force.

CPU: the JSON assembly, the save totals, bench_compare on old + new
files and its header assert, and the Rust getter against the RawBeatSim
twin on scripted wall / tilt / completed episodes. CUDA (pod): the
kernel's new columns against the Rust counts on the same episodes under
per-tick state resync."""
import json

import numpy as np
import pytest

from beat_arena_env.chart import Note, Wall, note_tuple, wall_tuple
from beat_arena_env.constants import N_JOINTS
from tests._helpers import cuda

TERM_FELL, TERM_TILT, TERM_ENVELOPE, TERM_TICKCAP = 1, 2, 4, 8

_ENTRY = {"leaderboard_id": 1, "hash": "h", "song": "s", "mapper": "m",
          "difficulty": "Expert", "stars": 5.0, "max_pp": 300.0,
          "api_max_score": 1000}
_SLOT = {"entry": _ENTRY, "max_score": 1000, "duration_s": 60.0}
_INFO = {"good_cuts": 8, "bad_cuts": 1, "misses": 1, "n_notes": 10,
         "bombs_hit": 0, "score_pre": 600.0, "score_acc": 100.0,
         "score_post": 250.0, "flail_rad": 0.5, "flail_ticks": 10,
         "wrist_rev_l": 3, "wrist_rev_r": 4, "ticks": 900}


# --- JSON assembly (bench_scoresaber.finish_slot) ----------------------------

def test_term_constants_match_cuda_env():
    import bench_scoresaber as bs
    src = open(bs.__file__.replace("bench_scoresaber.py",
                                   "beat_arena_env/cuda_env.py")).read()
    # cuda_env imports torch + the extension lazily; read the constants
    # from source so this stays a CPU test
    assert ("TERM_FELL, TERM_TILT, TERM_ENVELOPE, TERM_TICKCAP = 1, 2, 4, 8"
            in src)
    assert (bs.TERM_FELL, bs.TERM_TILT, bs.TERM_ENVELOPE,
            bs.TERM_TICKCAP) == (1, 2, 4, 8)


def test_finish_slot_completed_level():
    import bench_scoresaber as bs
    r = bs.finish_slot(_SLOT, dict(_INFO, wall_enters=2, term_flags=0),
                       (700.0, 6, 4))
    assert r["breaks"] == 4 and r["wall_enters"] == 2
    assert r["fell"] is False and r["tilted"] is False
    assert r["envelope"] is False and r["terminated_tick"] is None
    assert not r["full_combo"]
    json.dumps(r)                              # JSON-serializable


@pytest.mark.parametrize("flags,fell,tilted,envelope", [
    (TERM_FELL, True, False, False),
    (TERM_TILT, True, True, False),
    (TERM_ENVELOPE, False, False, True),
    (TERM_FELL | TERM_ENVELOPE, True, False, True),
    (TERM_TICKCAP, False, False, False),
])
def test_finish_slot_terminated_level(flags, fell, tilted, envelope):
    import bench_scoresaber as bs
    r = bs.finish_slot(_SLOT, dict(_INFO, wall_enters=0, term_flags=flags),
                       (700.0, 6, 2))
    assert (r["fell"], r["tilted"], r["envelope"]) == (fell, tilted, envelope)
    assert r["terminated_tick"] == 900         # = the level's ticks
    assert type(r["breaks"]) is int and type(r["wall_enters"]) is int


def test_finish_slot_engine_without_term_reports_none():
    """An info dict without the B09 keys (e.g. RustBeatEnv) must not
    fabricate zeros."""
    import bench_scoresaber as bs
    r = bs.finish_slot(_SLOT, _INFO, (700.0, 6, 3))
    assert r["breaks"] == 3 and r["wall_enters"] is None
    assert r["fell"] is None and r["terminated_tick"] is None


def test_cuda_row_to_result_mapping():
    """run_cuda's row -> info mapping reads the widened bench row: fake
    a popped row and push it through finish_slot the way run_cuda does."""
    import inspect
    import bench_scoresaber as bs
    src = inspect.getsource(bs.run_cuda)
    assert '"wall_enters": int(r[17])' in src
    assert '"term_flags": int(r[18])' in src
    r = np.zeros(19)
    r[:3] = (8, 1, 1)
    r[10], r[11], r[12] = 700.0, 6, 5          # cw score / maxc / breaks
    r[16], r[17], r[18] = 450, 3, TERM_TILT
    res = bs.finish_slot(_SLOT, dict(_INFO, ticks=int(r[16]),
                                     wall_enters=int(r[17]),
                                     term_flags=int(r[18])),
                         (float(r[10]), int(r[11]), int(r[12])))
    assert res["breaks"] == 5 and res["wall_enters"] == 3
    assert res["fell"] and res["tilted"] and res["terminated_tick"] == 450


def test_save_results_term_totals(tmp_path):
    import bench_scoresaber as bs
    lv = {}
    for i, (tf, we, br) in enumerate([(0, 2, 3), (TERM_FELL, 1, 1),
                                      (TERM_TILT | TERM_ENVELOPE, 0, 0)]):
        e = bs.finish_slot(dict(_SLOT, entry=dict(_ENTRY, leaderboard_id=i)),
                           dict(_INFO, wall_enters=we, term_flags=tf),
                           (700.0, 6, br))
        lv[i] = e
    # an old-format level (no B09 fields) is excluded from the totals
    old = dict(lv[0], leaderboard_id=9)
    for k in ("breaks", "wall_enters", "fell", "tilted", "envelope",
              "terminated_tick"):
        old.pop(k)
    lv[9] = old
    out = tmp_path / "r.json"
    data = {}
    bs.save_results(str(out), data, lv)
    d = json.load(open(out))
    assert d["term_levels"] == 3
    assert d["breaks_total"] == 4 and d["wall_enters_total"] == 3
    assert d["fell_levels"] == 2 and d["tilted_levels"] == 1
    assert d["envelope_levels"] == 1 and d["terminated_levels"] == 2
    assert "breaks" in bs._term_summary(d)


# --- bench_compare -----------------------------------------------------------

def _level(i, **kw):
    e = {"leaderboard_id": i, "song": f"s{i}", "difficulty": "E",
         "stars": 1.0 + i, "score": 100 * i, "accuracy": 0.9, "pp": 10.0 * i,
         "good_cuts": 5, "bad_cuts": 0, "misses": 0, "bombs_hit": 0,
         "highest_combo": 5, "full_combo": True, "score_pre": 1.0,
         "score_acc": 1.0, "score_post": 1.0, "flail_rad": 0.0,
         "flail_ticks": 0}
    e.update(kw)
    return e


_TERM = dict(breaks=1, wall_enters=1, fell=False, tilted=False,
             envelope=False, terminated_tick=None)
_HDR = {"sim": "rust", "sim_dtype": "f32", "scoring_version": 3,
        "mirror": False, "tau_scale": [1.25] + [1.0] * 15}


def _write(tmp_path, name, levels, **hdr):
    d = dict(_HDR, levels=levels, total_unweighted_pp=1.0,
             total_weighted_pp=1.0)
    d.update(hdr)
    p = tmp_path / name
    p.write_text(json.dumps(d))
    return str(p)


def test_bench_compare_new_files_print_term_totals(tmp_path):
    import bench_compare as bc
    a = [_level(i, **_TERM) for i in range(3)]
    b = [_level(i, **_TERM) for i in range(3)]
    b[1].update(fell=True, terminated_tick=300, wall_enters=3, breaks=4)
    out = []
    r = bc.compare(_write(tmp_path, "a.json", a, sim="cuda"),
                   _write(tmp_path, "b.json", b), out=out.append)
    assert r["term"]["levels"] == 3
    assert r["term"]["A"] == {"breaks": 3, "wall_enters": 3, "fell": 0,
                              "tilted": 0, "envelope": 0, "terminated": 0}
    assert r["term"]["B"]["breaks"] == 6 and r["term"]["B"]["fell"] == 1
    assert r["term"]["B"]["wall_enters"] == 5
    assert r["term"]["B"]["terminated"] == 1
    assert r["diff_levels"] == 1
    for f in ("breaks", "wall_enters", "fell", "terminated_tick"):
        assert r["fields"][f] == 1
    assert any("termination telemetry over 3" in s for s in out)


def test_bench_compare_old_vs_new_files(tmp_path):
    """An old JSON (no B09 fields, no sim_dtype / scoring_version /
    tau_scale / mirror keys — a pre-v46 f64 rust file) against a new one
    of the same regime: no crash, no phantom term diffs, telemetry
    skipped."""
    import bench_compare as bc
    old = [_level(i) for i in range(3)]
    new = [_level(i, **_TERM) for i in range(3)]
    po = tmp_path / "old.json"
    po.write_text(json.dumps({"levels": old, "total_unweighted_pp": 1.0,
                              "total_weighted_pp": 1.0}))
    pn = _write(tmp_path, "new.json", new, sim_dtype="f64",
                scoring_version=1, tau_scale=None)
    out = []
    r = bc.compare(str(po), pn, out=out.append)
    assert r["term"] is None and r["diff_levels"] == 0
    assert r["header_mismatch"] == []
    assert any("not in both files" in s for s in out)


@pytest.mark.parametrize("key,val", [
    ("tau_scale", [1.0] * 16), ("mirror", True), ("scoring_version", 2),
    ("sim_dtype", "f64"),
])
def test_bench_compare_header_mismatch_asserts(tmp_path, key, val):
    import bench_compare as bc
    lv = [_level(i, **_TERM) for i in range(2)]
    pa = _write(tmp_path, "a.json", lv)
    pb = _write(tmp_path, "b.json", lv, **{key: val})
    with pytest.raises(bc.HeaderMismatch, match=key):
        bc.compare(pa, pb, out=lambda s: None)
    out = []
    r = bc.compare(pa, pb, out=out.append, force=True)   # escape hatch
    assert r["header_mismatch"] == [key] and r["common"] == 2
    assert any(s.startswith("WARNING: header " + key) for s in out)


def test_bench_compare_header_defaults():
    """Missing keys normalize as bench_scoresaber reads them: nominal
    tau == no tau_scale, rust without sim_dtype = f64, cuda = f32."""
    import bench_compare as bc
    h = bc._header({"tau_scale": [1.0] * 16, "sim": "rust"}, True)
    assert h == bc._header({}, True)
    assert h["sim_dtype"] == "f64" and h["scoring_version"] == 1
    assert bc._header({"sim": "cuda"}, False)["sim_dtype"] == "f32"


# --- engines: scripted wall / tilt / completed episodes ----------------------

_BIG = dict(cx=0.0, cy=0.0, width=20.0, height=40.0, speed=8.0)
_FAR = [Note(hit_time=100.0, col=2, row=1, x=2.5, y=1.4, color=0,
             angle=4.712, speed=4.0)]
_SHORT = [Note(hit_time=0.1, col=2, row=1, x=2.5, y=1.4, color=0,
               angle=4.712, speed=4.0)]
# coast actions: the unactuated body tips over (TILT) at ~tick 23; the
# walls cross the head plane inside that window (test_v17_walls._sim)
# (chart, walls, expected (wall_enters, flags)) — the expectations were
# measured on the f64 build and document the scenario coverage
SCENARIOS = [
    (_FAR, [], (0, TERM_TILT)),                                  # no walls
    (_FAR, [Wall(start_time=0.2, length=1.05, **_BIG)], (1, TERM_TILT)),
    (_FAR, [Wall(start_time=0.2, length=0.4, **_BIG),            # 2 edges
            Wall(start_time=0.35, length=0.4, **_BIG)], (2, TERM_TILT)),
    (_FAR, [Wall(start_time=0.2, cx=2.5, cy=1.5, width=0.6, height=3.0,
                 length=1.0, speed=8.0)], (0, TERM_TILT)),       # dodged
    (_SHORT, [Wall(start_time=0.1, length=0.2, **_BIG)], (1, 0)),  # ends
]
MAX_STEPS = 600


def _raw_flags(sim):
    return ((sim.falls > 0) * TERM_FELL | (sim.tilts > 0) * TERM_TILT
            | bool(sim.escaped) * TERM_ENVELOPE
            | (sim.tick >= MAX_STEPS) * TERM_TICKCAP)


def _raw_sims():
    import beat_arena_rs
    sims = []
    for ch, ws, _ in SCENARIOS:
        s = beat_arena_rs.RawBeatSim(max_steps=MAX_STEPS, gears=5)
        s.set_chart([note_tuple(n) for n in ch], [wall_tuple(w) for w in ws])
        sims.append(s)
    return sims


@pytest.mark.parametrize("mod_name", ["beat_arena_rs", "beat_arena_rs32"])
def test_vec_term_matches_raw_sim(mod_name):
    """VecBeatSim.term (the rust bench path) == the RawBeatSim getters
    (wall_enters, falls, tilts, escaped, tick) at done, and the combo
    walk's breaks count the wall entries."""
    mod = pytest.importorskip(mod_name)
    import beat_arena_rs
    if not hasattr(mod.VecBeatSim, "term"):
        pytest.skip(f"{mod_name} build predates VecBeatSim.term — rebuild")
    n = len(SCENARIOS)
    vec = mod.VecBeatSim(n, max_steps=MAX_STEPS, combo_walk=True,
                         return_events=False, gears=5)
    for i, (ch, ws, _) in enumerate(SCENARIOS):
        vec.set_chart(i, [note_tuple(x) for x in ch],
                      [wall_tuple(w) for w in ws])
    raws = _raw_sims()
    obs = np.zeros((n, vec.obs().shape[0] // n), np.float32)
    acts = np.zeros((n, N_JOINTS), np.int64)
    done_v = np.zeros(n, bool)
    done_r = [False] * n
    for _ in range(200):
        _, d, _ = vec.step_into(acts, obs)
        done_v |= np.asarray(d)
        for i, s in enumerate(raws):
            if not done_r[i]:
                done_r[i] = s.step([0] * N_JOINTS)[2]
        if done_v.all() and all(done_r):
            break
    assert done_v.all() and all(done_r)
    for i, (_, _, expect) in enumerate(SCENARIOS):
        we, flags = vec.term(i)
        _, _, breaks = vec.combo(i)
        if mod is beat_arena_rs:              # same f64 engine: exact
            assert (we, flags) == (raws[i].wall_enters, _raw_flags(raws[i]))
            assert vec.wrist_rev(i)[2] == raws[i].tick
        assert (we, flags) == expect, (i, we, flags)
        assert breaks >= we                   # every entry breaks the combo
    # coverage: completed + tilted levels, single and double wall entries
    assert {f for _, _, (_, f) in SCENARIOS} == {0, TERM_TILT}


@cuda
def test_cuda_bench_rec_term_columns_match_rust():
    """Pod gate: the kernel's bench_rec cols 16/17/18 (ticks, wall
    entries, term flags) and col 12 (combo breaks) equal the f64 Rust
    reference's counts on the scripted episodes, CUDA state resynced to
    Rust every tick (test_bench_cuda pattern), one level per slot
    (bench_refill with an exhausted queue: the slot freezes)."""
    import torch
    from beat_arena_env.cuda_env import BENCH_COLS, CudaBeatVecEnv
    from tests._helpers import resync_from_sims
    from tests.test_bench_cuda import RefWalk
    n = len(SCENARIOS)
    env = CudaBeatVecEnv(
        n, charts=[ch for ch, _, _ in SCENARIOS],
        walls=[[wall_tuple(w) for w in ws] for _, ws, _ in SCENARIOS],
        max_steps=MAX_STEPS, gears=5, device="cuda", dtype=torch.float64,
        auto_reset=False, want_contacts=False, combo_walk=True,
        bench_refill=True, seed=0)
    env.set_next_chart(torch.arange(n, device="cuda"))
    env.reset()
    env.set_next_chart(torch.full((n,), -1, dtype=torch.int64,
                                  device="cuda"))
    sims = _raw_sims()
    resync_from_sims(env, sims)
    walks = [RefWalk(len(ch)) for ch, _, _ in SCENARIOS]
    acts = torch.zeros((n, N_JOINTS), dtype=torch.int64, device="cuda")
    done_r = [False] * n
    rows = {}
    for t in range(200):
        env.step(acts)
        for i, s in enumerate(sims):
            if not done_r[i]:
                _o, _r, d, ev = s.step([0] * N_JOINTS)
                for row in ev:
                    walks[i].consume(int(row[0]), int(row[1]), float(row[2]))
                done_r[i] = d
        ids, rr = env.pop_bench_rows()
        assert rr.shape[1] == BENCH_COLS == 23
        for e, r in zip(ids.tolist(), rr):
            assert e not in rows, "a frozen slot re-reported"
            assert done_r[e], (t, e, "CUDA finished before Rust")
            rows[e] = r.copy()
        resync_from_sims(env, sims)
        if all(done_r) and len(rows) == n:
            break
    assert sorted(rows) == list(range(n))
    for i, s in enumerate(sims):
        r = rows[i]
        got = (int(r[16]), int(r[17]), int(r[18]), int(r[12]))
        want = (s.tick, s.wall_enters, _raw_flags(s), walks[i].breaks)
        assert got == want, (i, got, want)
        assert (int(r[17]), int(r[18])) == SCENARIOS[i][2]
