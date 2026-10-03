"""Per-level applied torque in the bench (2026-10-01, user): Σ_j|τ_j|
(N·m, the env v43 energy base) summed over each level's ticks, in total
and per group — lib.rs VecBeatSim.pop_torque_env on the rust path, the
CUDA bench_rec cols 19..22 on the cuda path — plus the bench JSON
fields (torque_nm*, torque_nms = N·m·s, torque_per_tick) and totals."""
import numpy as np
import pytest

from beat_arena_env.chart import note_tuple
from beat_arena_env.constants import N_JOINTS
from tests._helpers import cuda
from tests.test_bench_term import MAX_STEPS, SCENARIOS

import bench_scoresaber as bs


def _rand_acts(rng, n):
    return rng.integers(0, 5, size=(n, N_JOINTS)).astype(np.int64)


@pytest.mark.parametrize("mod_name", ["beat_arena_rs", "beat_arena_rs32"])
def test_pop_torque_env_is_per_env_and_resets(mod_name):
    mod = pytest.importorskip(mod_name)
    if not hasattr(mod.VecBeatSim, "pop_torque_env"):
        pytest.skip(f"{mod_name} build predates pop_torque_env — rebuild")
    n = 3
    vec = mod.VecBeatSim(n, max_steps=MAX_STEPS, gears=5,
                         return_events=False)
    for i in range(n):
        vec.set_chart(i, [note_tuple(x) for x in SCENARIOS[0][0]], [])
    vec.pop_torque()                                  # clear from creation
    obs = np.zeros((n, vec.obs().shape[0] // n), np.float32)
    rng = np.random.default_rng(0)
    acts = _rand_acts(rng, n)
    acts[1] = 0                                       # env 1 coasts: 0 N·m
    for _ in range(5):
        vec.step_into(acts, obs)
    per = [vec.pop_torque_env(i) for i in range(n)]
    assert per[0][0] > 0 and per[2][0] > 0 and per[1][0] == 0.0
    for p in per:
        assert 1 <= p[4] <= 5          # one add per tick (an env may end early)
        assert abs(p[0] - (p[1] + p[2] + p[3])) <= 1e-3 * max(1.0, p[0])
    assert all(vec.pop_torque_env(i)[0] == 0.0 for i in range(n))  # reset
    # the per-env pops partition the all-env accumulator — over the SAME
    # window: a twin sim (deterministic, same charts + actions) pops the
    # total while this one pops per env. (Comparing two consecutive
    # windows of one sim only held while every env was alive in both or
    # dead in both; with constant random gears the envs tilt-terminate
    # within ~10 ticks, so which side of that a window falls on is
    # body-dependent.)
    twin = mod.VecBeatSim(n, max_steps=MAX_STEPS, gears=5,
                          return_events=False)
    for i in range(n):
        twin.set_chart(i, [note_tuple(x) for x in SCENARIOS[0][0]], [])
    twin.pop_torque()
    obs2 = np.zeros_like(obs)
    for _ in range(5):
        twin.step_into(acts, obs2)
    twin.pop_torque()
    for _ in range(3):
        vec.step_into(acts, obs)
        twin.step_into(acts, obs2)
    per = [vec.pop_torque_env(i) for i in range(n)]
    tot = twin.pop_torque()
    assert sum(p[4] for p in per) == tot[4]
    assert abs(tot[0] - sum(p[0] for p in per)) <= 1e-3 * max(1.0, tot[0])


def test_torque_fields_and_totals():
    f = bs._torque_fields({"torque": (3000.0, 1000.0, 500.0, 1500.0),
                           "ticks": 300}, tick_hz=30)
    assert f == {"torque_nm": 3000.0, "torque_nm_spine": 1000.0,
                 "torque_nm_arms": 500.0, "torque_nm_legs": 1500.0,
                 "torque_nms": 100.0, "torque_per_tick": 10.0}
    # the same motion at 60 Hz: twice the ticks and the per-tick sum,
    # the same N·m·s
    f60 = bs._torque_fields({"torque": (6000.0, 2000.0, 1000.0, 3000.0),
                             "ticks": 600}, tick_hz=60)
    assert f60["torque_nms"] == f["torque_nms"]
    assert all(v is None for v in bs._torque_fields({}, 30).values())
    t = bs.torque_totals([f, f60, {"torque_nm": None}])
    assert t["torque_levels"] == 2 and t["torque_nms_total"] == 200.0
    assert t["torque_nm_total"] == 9000.0


@cuda
def test_cuda_bench_torque_columns_equal_level_tau_stat():
    """Pod gate: with random (torque-spending) actions, each finished
    level's bench_rec cols 19..22 equal that env's tau_stat accumulated
    over exactly that level (same dtype, same adds: bitwise)."""
    import torch
    from beat_arena_env.cuda_env import BENCH_COLS, CudaBeatVecEnv
    from beat_arena_env.chart import wall_tuple
    n = len(SCENARIOS)
    env = CudaBeatVecEnv(
        n, charts=[ch for ch, _, _ in SCENARIOS],
        walls=[[wall_tuple(w) for w in ws] for _, ws, _ in SCENARIOS],
        max_steps=MAX_STEPS, gears=5, device="cuda", dtype=torch.float64,
        auto_reset=False, want_contacts=False, combo_walk=True,
        bench_refill=True, seed=0)
    env.set_next_chart(torch.arange(n, device="cuda"))
    env.reset()
    env.set_next_chart(torch.full((n,), -1, dtype=torch.int64, device="cuda"))
    env.tau_stat.zero_()
    rng = np.random.default_rng(1)
    rows = {}
    for _ in range(MAX_STEPS + 5):
        acts = torch.as_tensor(_rand_acts(rng, n), device="cuda")
        env.step(acts)
        lvl = env.tau_stat[:, :4].double().cpu().numpy().copy()
        ids, rr = env.pop_bench_rows()
        assert rr.shape[1] == BENCH_COLS == 23
        for e, r in zip(ids.tolist(), rr):
            rows[e] = (r[19:23].copy(), lvl[e])
        if len(rows) == n:
            break
    assert sorted(rows) == list(range(n))
    for e, (got, want) in rows.items():
        assert got[0] > 0, e
        np.testing.assert_array_equal(got, want, err_msg=f"env {e}")
