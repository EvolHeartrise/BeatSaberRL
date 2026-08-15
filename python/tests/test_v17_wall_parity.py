"""env v17 walls — CUDA mirror vs Rust reference (spec §5 + §12).

The §5 lockstep tiers gate the mirror on procedural charts, which have
NO walls, so this module is the wall path's own gate: weld-mode f64
lockstep-resync (the v10 protocol — CUDA re-takes Rust's dynamics state
every tick, so what is measured is one tick of the wall block at a
time) across wall configurations that exercise the z window, the x/y
box (via the unactuated body's slump through the crossing), overlap
de-duplication, and the entity-window wall tokens.
"""
import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CudaBeatVecEnv needs CUDA", allow_module_level=True)

from beat_arena_env.chart import Note, Wall, note_tuple, wall_tuple
from beat_arena_env.constants import N_JOINTS, OBS_DIM
from beat_arena_env.cuda_env import CudaBeatVecEnv
import beat_arena_rs

FAR_NOTE = Note(hit_time=200.0, col=2, row=1, x=0.3, y=1.4, color=1,
                angle=4.712, speed=4.0)
MAX_STEPS = 6900

# four regimes: full crossing, dodged side wall, hanging crouch wall,
# two overlapping walls (must charge once per tick)
WALLS = [
    [Wall(1.0, 0.0, 0.0, 20.0, 40.0, 2.0, 4.0)],
    [Wall(1.0, 2.5, 1.5, 0.6, 3.0, 2.0, 4.0)],
    [Wall(0.5, 0.0, 2.1, 2.4, 1.8, 4.0, 4.0)],
    [Wall(1.0, 0.0, 0.0, 20.0, 40.0, 2.0, 4.0),
     Wall(1.2, 0.0, 0.0, 20.0, 40.0, 1.0, 4.0)],
]
N = len(WALLS)


def _rust_sims():
    sims = []
    for ws in WALLS:
        sim = beat_arena_rs.RawBeatSim(max_steps=MAX_STEPS,
                                       effort_coef=0.0, wide_acc=False,
                                       gears=3, foot_contact=False)
        sim.set_chart([note_tuple(FAR_NOTE)],
                      [wall_tuple(w) for w in ws])
        sims.append(sim)
    return sims


def _resync(env, sims):
    q36s, v35s = [], []
    for sim in sims:
        q36s.append(list(sim.qpos()))
        v35s.append(list(np.concatenate([sim.twist(), sim.omega()])))
    wv = env.wrist_vel.clone()
    env.set_state(torch.tensor(q36s, dtype=torch.float64),
                  torch.tensor(v35s, dtype=torch.float64))
    env.wrist_vel.copy_(wv)


def test_wall_lockstep_parity():
    sims = _rust_sims()
    env = CudaBeatVecEnv(N, charts=[[FAR_NOTE]] * N,
                         walls=[[wall_tuple(w) for w in ws]
                                for ws in WALLS],
                         max_steps=MAX_STEPS, gears=3, wide_acc=False,
                         effort_coef=0.0, device="cuda",
                         dtype=torch.float64, auto_reset=False,
                         foot_contact=False, seed=0)
    env.reset()   # charts= test mode assigns env i -> chart i HERE
    assert env.chart_idx.cpu().tolist() == list(range(N))
    total_in = [0] * N
    # zero actions: the unactuated weld body SLUMPS deterministically —
    # the head sweeps a real trajectory through/past the boxes without
    # ever tripping the root-escape backstop (random gears do, ~t 60)
    for tick in range(80):                      # spans every crossing
        acts = np.zeros((N, N_JOINTS), dtype=np.int64)
        _resync(env, sims)
        r_rust, ev_codes = [], []
        for i, sim in enumerate(sims):
            _obs, r, done, ev = sim.step([int(a) for a in acts[i]])
            assert not done
            r_rust.append(r)
            ev_codes.append({e[1] for e in ev})
        obs_c, r_cuda, done_c, _ev = env.step(
            torch.as_tensor(acts, device="cuda"))
        assert not done_c.any()
        for i in range(N):
            # identical state + identical f64 ops -> identical reward
            assert float(r_cuda[i]) == pytest.approx(r_rust[i], abs=1e-12), \
                (tick, i)
            assert (7 in ev_codes[i]) == (env.walls_in[i].item()
                                          > total_in[i]), (tick, i)
            total_in[i] = env.walls_in[i].item()
        # entity window (lanes 100..556: 24x18 tokens + mask). The token
        # z lanes carry z_off = head_z − PLANE_Z0, and the engines' FK
        # heads differ by ≤1 f64 ulp per §21 (transcendentals + FMA
        # contraction) which one violent-collapse tick amplifies to
        # ~1e-5 m (measured: 3.9e-6 on the z lane at tick 34) — so the
        # continuous lanes get atol 1e-4 (any selection/indexing bug is
        # O(0.1)+), while the slot mask stays EXACT. Lanes ≥556
        # (ground/body FK) belong to the §5 dynamics tiers, not here.
        obs_c2 = obs_c.cpu().numpy().reshape(N, OBS_DIM)
        obs_r = np.stack([np.asarray(s.obs()) for s in sims])
        np.testing.assert_allclose(obs_c2[:, 100:532], obs_r[:, 100:532],
                                   rtol=0.0, atol=1e-4,
                                   err_msg=f"tick {tick}")
        np.testing.assert_array_equal(obs_c2[:, 532:556], obs_r[:, 532:556],
                                      err_msg=f"tick {tick} (mask)")
    for i, sim in enumerate(sims):
        assert sim.walls_in == env.walls_in[i].item(), i
    # the crossing walls actually charged, the dodged wall did not
    assert sims[0].walls_in > 0
    assert sims[1].walls_in == 0
    assert sims[3].walls_in > 0
