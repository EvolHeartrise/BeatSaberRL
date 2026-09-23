"""CUDA-sim bench path (2026-09-16): the kernel-side ScoreSaber combo walk
(CudaBeatVecEnv(combo_walk=True)) against a reference walk over the Rust
sim's event stream under per-tick state resync, the finished-level
freeze, the bench result schema, and bench_compare."""
import json

import numpy as np
import torch

from beat_arena_env.chart import ChartConfig, generate_chart, note_tuple
from beat_arena_env.constants import N_JOINTS
from tests._helpers import cuda, resync, rust_state


class RefWalk:
    """lib.rs ComboWalk::consume, verbatim (codes: 0 cut, 5 post, 7 wall
    tick = nothing, anything else = break)."""

    def __init__(self, n):
        self.mult, self.prog, self.combo, self.maxc, self.breaks = 1, 0, 0, 0, 0
        self.score = 0.0
        self.note_mult = [0] * n

    def consume(self, note, code, pts):
        if code == 0:
            need = {1: 2, 2: 4, 4: 8}.get(self.mult)
            if need is not None:
                self.prog += 1
                if self.prog >= need:
                    self.mult *= 2
                    self.prog = 0
            self.note_mult[note] = self.mult
            self.combo += 1
            self.maxc = max(self.maxc, self.combo)
            self.score += float(np.float32(pts)) * self.mult
        elif code == 5:
            m = self.note_mult[note] or self.mult
            self.score += float(np.float32(pts)) * m
        elif code == 7:
            pass
        else:
            self.combo = 0
            self.breaks += 1
            self.mult = max(1, self.mult // 2)
            self.prog = 0


def test_finish_slot_schema_and_full_combo():
    import bench_scoresaber as bs
    slot = {"entry": {"leaderboard_id": 1, "hash": "h", "song": "s",
                      "mapper": "m", "difficulty": "Expert", "stars": 5.0,
                      "max_pp": 300.0, "api_max_score": 1000},
            "max_score": 1000, "duration_s": 60.0}
    info = {"good_cuts": 10, "bad_cuts": 0, "misses": 0, "n_notes": 10,
            "bombs_hit": 0, "score_pre": 700.0, "score_acc": 120.0,
            "score_post": 300.0, "flail_rad": 1.5, "flail_ticks": 30}
    r = bs.finish_slot(slot, info, (900.0, 10, 0))
    assert set(r) == {"leaderboard_id", "song_hash", "song", "mapper",
                      "difficulty", "stars", "max_pp", "duration_s",
                      "n_notes", "score", "max_score", "api_max_score",
                      "accuracy", "pp", "good_cuts", "bad_cuts", "misses",
                      "bombs_hit", "highest_combo", "full_combo",
                      "score_pre", "score_acc", "score_post", "flail_rad",
                      "flail_ticks", "wrist_rev_l", "wrist_rev_r", "ticks",
                      "wrist_rev_per_s_l", "wrist_rev_per_s_r"}
    assert r["score"] == 900 and r["accuracy"] == 0.9 and r["full_combo"]
    r2 = bs.finish_slot(slot, dict(info, good_cuts=9, misses=1),
                        (600.0, 6, 1))
    assert not r2["full_combo"] and r2["highest_combo"] == 6


def test_bench_compare(tmp_path):
    import bench_compare as bc
    lv = [{"leaderboard_id": i, "song": f"s{i}", "difficulty": "E",
           "stars": 1.0 + i, "score": 100 * i, "accuracy": 0.9, "pp": 10.0 * i,
           "good_cuts": 5, "bad_cuts": 0, "misses": 0, "bombs_hit": 0,
           "highest_combo": 5, "full_combo": True, "score_pre": 1.0,
           "score_acc": 1.0, "score_post": 1.0, "flail_rad": 0.0,
           "flail_ticks": 0} for i in range(4)]
    a = {"levels": lv, "total_unweighted_pp": 60.0, "total_weighted_pp": 50.0,
         "sim": "cuda"}
    b = json.loads(json.dumps(a))
    b["sim"] = "rust"
    b["levels"][2]["pp"] = 25.0
    b["levels"][2]["misses"] = 1
    b["levels"][3]["accuracy"] = 0.95
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    out = []
    r = bc.compare(str(pa), str(pb), out=out.append)
    assert r["common"] == 4 and r["diff_levels"] == 2
    assert abs(r["sum_abs_dpp"] - 5.0) < 1e-9
    assert abs(r["max_dacc"] - 0.05) < 1e-9
    assert r["fields"]["pp"] == 1 and r["fields"]["misses"] == 1
    assert r["fields"]["accuracy"] == 1 and r["fields"]["score"] == 0


@cuda
def test_cuda_combo_walk_matches_reference_walk_under_resync():
    """Random charts, random actions, CUDA dynamics resynced to Rust every
    tick (test_v10_parity_cuda pattern): the Rust event stream fed to
    RefWalk must equal the kernel's walk exactly on every tick, and a
    finished level must freeze."""
    import beat_arena_rs
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    cfg = dict(colors="both", chart_len_s=2.5, bomb_prob=0.25,
               extra_prob=0.3, dot_prob=0.15)
    charts = [generate_chart(ChartConfig(**cfg), np.random.default_rng(s))
              for s in range(900, 906)]
    n = len(charts)
    gears = 5
    max_steps = max(int((ch[-1].hit_time + 0.5) * 30) + 90 for ch in charts)
    env = CudaBeatVecEnv(n, charts=charts, max_steps=max_steps + 60,
                         gears=gears, device="cuda",
                         dtype=torch.float64, auto_reset=False,
                         want_contacts=False, combo_walk=True)
    env.reset()
    sims = []
    for ch in charts:
        sim = beat_arena_rs.RawBeatSim(max_steps=max_steps,
                                       gears=gears)
        sim.set_chart([note_tuple(x) for x in ch])
        sims.append(sim)
    walks = [RefWalk(len(ch)) for ch in charts]
    rng = np.random.default_rng(7)
    done_r = [False] * n
    frozen = {}
    for t in range(max_steps + 60):
        acts = rng.integers(0, gears, size=(n, N_JOINTS))
        _tobs, _trew, tdone, _ = env.step(torch.as_tensor(acts))
        cw_s, cw_c, cw_b = [x.cpu().numpy() for x in env.combo()]
        q36s, v35s = [], []
        for e, sim in enumerate(sims):
            if not done_r[e]:
                _o, _r, d_r, ev = sim.step([int(x) for x in acts[e]])
                for row in ev:
                    note, code, pts = int(row[0]), int(row[1]), float(row[2])
                    walks[e].consume(note, code, pts)
                done_r[e] = d_r
                w = walks[e]
                assert abs(float(cw_s[e]) - w.score) < 1e-6, (t, e, cw_s[e], w.score)
                assert int(cw_c[e]) == w.maxc and int(cw_b[e]) == w.breaks, (t, e)
                if d_r:
                    frozen[e] = (float(cw_s[e]), int(cw_c[e]), int(cw_b[e]),
                                 int(env.good[e]), int(env.miss[e]))
            else:
                # finished level: the kernel must not touch it again
                assert (float(cw_s[e]), int(cw_c[e]), int(cw_b[e]),
                        int(env.good[e]), int(env.miss[e])) == frozen[e], (t, e)
                assert bool(tdone[e])
            q, v = rust_state(sim)
            q36s.append(q)
            v35s.append(v)
        resync(env, q36s, v35s)
        if all(done_r) and t > max(0, max_steps - 1):
            break
    assert all(done_r)
    assert sum(w.combo > 0 or w.breaks > 0 for w in walks) > 0
