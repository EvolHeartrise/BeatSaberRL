"""CUDA bench slot refill (2026-09-16): the LPT scheduler (CPU) and the
kernel-side refill (CUDA twins): every level plays exactly once, the
result of a level does not depend on which slot or wave played it, and
two runs are bit-identical."""
import numpy as np
import torch

from beat_arena_env.chart import ChartConfig, generate_chart
from beat_arena_env.constants import N_JOINTS
from tests._helpers import cuda


def test_plan_slots_every_level_once_and_lpt_bound():
    from bench_scoresaber import plan_slots
    rng = np.random.default_rng(0)
    ticks = list(rng.integers(600, 18000, size=137))
    for n in (1, 4, 16, 137, 500):
        queues, makespan, ideal = plan_slots(ticks, n)
        assert len(queues) == min(n, len(ticks))
        flat = sorted(i for q in queues for i in q)
        assert flat == list(range(len(ticks)))            # each exactly once
        assert makespan >= ideal - 1e-9
        assert makespan <= ideal * 4 / 3 + max(ticks)      # LPT bound
        # longest level goes first
        assert queues[0][0] == int(np.argmax(ticks))
    assert plan_slots(ticks, 16) == plan_slots(ticks, 16)  # deterministic


def _charts(seeds, chart_len_s):
    cfg = dict(colors="both", chart_len_s=chart_len_s, bomb_prob=0.25,
               extra_prob=0.3, dot_prob=0.15)
    return [generate_chart(ChartConfig(**cfg), np.random.default_rng(s))
            for s in seeds]


def _play(charts, n_slots, device="cuda"):
    """Drive a bench_refill env with coast actions until every level has
    a popped row; returns {chart_idx: row} and the tick count."""
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    from bench_scoresaber import plan_slots
    ticks = [int((ch[-1].hit_time + 1.0) * 30) + 60 for ch in charts]
    max_steps = max(ticks) + 300
    queues, _, _ = plan_slots(ticks, n_slots)
    n = len(queues)
    env = CudaBeatVecEnv(n, charts=charts, max_steps=max_steps, gears=5,
                         device=device,
                         dtype=torch.float32, auto_reset=False,
                         want_contacts=False, foot_contact=True,
                         combo_walk=True, bench_refill=True, seed=0)
    current = [q.pop(0) for q in queues]
    cand = np.array([(q[0] if q else -1) for q in queues], dtype=np.int64)
    # reset() starts each slot on the candidate chart: first the FIRST
    # level, then the candidate array holds the NEXT level
    env.set_next_chart(torch.as_tensor(np.array(current), device=device))
    env.reset()
    env.set_next_chart(torch.as_tensor(cand, device=device))
    acts = torch.zeros((n, N_JOINTS), dtype=torch.int64, device=device)
    rows_by_chart = {}
    tick = 0
    while len(rows_by_chart) < len(charts) and tick < 4 * max_steps:
        env.step(acts)
        tick += 1
        ids, rows = env.pop_bench_rows()
        for e, r in zip(ids.tolist(), rows):
            assert int(r[13]) == current[e], "row chart != slot's level"
            assert current[e] not in rows_by_chart, "level finished twice"
            rows_by_chart[current[e]] = r.copy()
            if cand[e] >= 0:
                current[e] = int(cand[e])
                queues[e].pop(0)
                cand[e] = queues[e][0] if queues[e] else -1
            else:
                current[e] = -1
        env.set_next_chart(torch.as_tensor(cand, device=device))
    return rows_by_chart, tick


@cuda
def test_cuda_refill_plays_every_level_once():
    charts = _charts(range(900, 912), 6.0)
    rows, tick = _play(charts, 4)
    assert sorted(rows) == list(range(12))
    for r in rows.values():
        assert np.all(np.isfinite(r))
        assert r[0] + r[1] + r[2] <= len(charts[0]) + 200


@cuda
def test_cuda_refill_matches_one_wave_and_is_deterministic():
    charts = _charts(range(920, 930), 6.0)
    one_wave, _ = _play(charts, len(charts))    # one level per slot
    refill_a, _ = _play(charts, 3)
    refill_b, _ = _play(charts, 3)
    for k in range(len(charts)):
        np.testing.assert_array_equal(refill_a[k], refill_b[k])   # determinism
        # a level's result does not depend on the slot that played it
        np.testing.assert_array_equal(refill_a[k], one_wave[k])
