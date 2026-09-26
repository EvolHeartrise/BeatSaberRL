"""Flux game mode — CUDA mirror parity vs the Rust reference (SPEC.md
§Flux; rust/src/flux.rs <-> csrc/beat_arena_cuda.cu FX instantiations).

Same tiers as the standard game's gates (test_v10_parity_cuda.py /
test_production_lockstep.py), on Flux charts:

 - reset obs at f64 rounding (the Flux hand + entity lanes included);
 - scripted placements (tests/test_flux.py's circle / wrong-hand /
   miss / bar / sweet-spot cases) played free-running in weld mode:
   the exact points, counters, statuses and rewards of Rust;
 - LOCKSTEP-RESYNC whole episodes (weld and contact) on generated
   Flux charts with a scripted prefix that guarantees circle and bar
   hits: the CUDA dynamics re-take Rust's state every tick (the palm
   and wrist finite-difference velocities carried across, as
   tests/_helpers.resync carries the wrist's), the env layer evolves
   freely on each side; statuses / prev record / counters / score
   decomposition EXACT every tick, obs and reward inside the
   knife-edge band, the in-kernel combo walk vs scoresaber.ComboTracker
   fed Rust's events at each done tick;
 - an f32 well-formedness tier.

Skipped without CUDA (the kernel has no CPU path)."""
import math

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CudaBeatVecEnv needs CUDA", allow_module_level=True)

import beat_arena_rs
from beat_arena_env.constants import (EVENT_REASONS, KIND_BAR, KIND_CIRCLE,
                                      N_JOINTS, OBS_DIM, OFF_GROUND_IMP,
                                      OFF_NOTE_MASK, OFF_PREV_L, OFF_PREV_R,
                                      PLANE_Z0, PREV_FEAT, ROW_Y)
from beat_arena_env.cuda_env import CudaBeatVecEnv
from beat_arena_env.flux_chart import (FluxConfig, FluxObj, flux_tuple,
                                       generate_flux_chart)
from beat_arena_env.scoresaber import ComboTracker
from tests._helpers import rust_state

DT = 1.0 / 30.0
SPEED = 12.0
GEARS = 5
ZERO = [0] * N_JOINTS
_PREV_VALID = [OFF_PREV_L + PREV_FEAT - 1, OFF_PREV_R + PREV_FEAT - 1]
POSTURE_J = [0, 1, 2] + list(range(17, 29))
HOLD_EVERY = 3


def _build(charts, max_steps, foot_contact, dtype=torch.float64,
           combo_walk=False, energy_coef=7.5e-6):
    tups = [[flux_tuple(o) for o in ch] for ch in charts]
    env = CudaBeatVecEnv(len(charts), game="flux", flux_charts=tups,
                         max_steps=max_steps, gears=GEARS,
                         foot_contact=foot_contact, energy_coef=energy_coef,
                         combo_walk=combo_walk, device="cuda", dtype=dtype,
                         auto_reset=False, want_contacts=True)
    sims = []
    for ch in tups:
        s = beat_arena_rs.RawBeatSim(max_steps=max_steps, gears=GEARS,
                                     foot_contact=foot_contact,
                                     energy_coef=energy_coef, flux=True)
        s.set_flux_chart(ch)
        sims.append(s)
    return env, sims


def _resync(env, sims):
    """Lockstep resync: CUDA re-takes each Rust state. set_state zeroes
    the finite-diff wrist AND palm velocities (Rust set_joints
    semantics); both are env-layer state Rust keeps, so CUDA's own are
    carried across (tests/_helpers.resync pattern)."""
    wv = env.wrist_vel.clone()
    pv = env.palm_vel.clone()
    q36s, v35s = zip(*(rust_state(s) for s in sims))
    env.set_state(torch.tensor(q36s, dtype=torch.float64),
                  torch.tensor(v35s, dtype=torch.float64))
    env.wrist_vel.copy_(wv)
    env.palm_vel.copy_(pv)


# -- scripted placements (tests/test_flux.py geometry) -------------------------

def _reset_rust():
    s = beat_arena_rs.RawBeatSim(max_steps=400, foot_contact=False,
                                 flux=True)
    s.set_flux_chart([])
    return s


def _arrive_now(sim, palm_z, lead=0.2, t_tick=1):
    z_off = sim.head_pose()[2] - PLANE_Z0
    return t_tick * DT + (palm_z + lead - z_off) / SPEED


def _bar_across(sim, palms, length, sweet, dy=0.0, t_tick=1, dev_deg=0.0):
    (lx, ly, lz), (rx, ry, rz) = palms
    fx, fy = rx - lx, ry - ly
    n = math.hypot(fx, fy)
    ux, uy = -fy / n, fx / n
    rot = math.atan2(-ux, uy) + math.radians(dev_deg)   # off square
    ux, uy = -math.sin(rot), math.cos(rot)
    mx, my = 0.5 * (lx + rx), 0.5 * (ly + ry)
    return FluxObj(hit_time=_arrive_now(sim, 0.5 * (lz + rz), t_tick=t_tick),
                   x=mx + dy * ux, y=my + dy * uy, kind=KIND_BAR,
                   speed=SPEED, rot=rot, length=length, sweet=sweet)


def _scripted_charts():
    sim = _reset_rust()
    palms = sim.palms()
    (rx, ry, rz) = palms[1]
    hit = _arrive_now(sim, rz)
    circ = lambda x, color: FluxObj(hit_time=hit, x=x, y=ry,
                                    kind=KIND_CIRCLE, speed=SPEED,
                                    color=color)
    charts = [
        [circ(rx, 1)],                                    # 60
        [circ(rx + 0.15, 1)],                             # 55
        [circ(rx, 0)],                                    # wrong hand
        [FluxObj(hit_time=0.3, x=0.9, y=ROW_Y[2], kind=KIND_CIRCLE,
                 speed=SPEED, color=1)],                  # miss
        [_bar_across(sim, palms, 0.4, 0.0)],              # 115
        [_bar_across(sim, palms, 0.4, 1.0)],              # 105
        [_bar_across(sim, palms, 0.4, -1.0)],             # 105
        [_bar_across(sim, palms, 0.4, 0.5)],              # 110
        [_bar_across(sim, palms, 0.4, 0.0, dy=0.5)],      # miss
        [_bar_across(sim, palms, 0.4, 0.0, dev_deg=50.0)],  # 115 (in tol)
        [_bar_across(sim, palms, 0.4, 0.0, dev_deg=70.0)],  # angle: bad
    ]
    expect = [60.0, 55.0, 0.0, 0.0, 115.0, 105.0, 105.0, 110.0, 0.0,
              115.0, 0.0]
    return charts, expect


def test_flux_reset_obs():
    charts = [generate_flux_chart(FluxConfig(chart_len_s=4.0),
                                  np.random.default_rng(s))
              for s in (10, 11, 12)]
    env, sims = _build(charts, 400, foot_contact=False)
    obs = env.reset().cpu().numpy()
    for i, s in enumerate(sims):
        o_r = np.asarray(s.obs())
        np.testing.assert_allclose(o_r, obs[i], rtol=0, atol=2e-15,
                                   err_msg=f"env{i} reset obs")
        np.testing.assert_array_equal(
            obs[i, OFF_NOTE_MASK:OFF_GROUND_IMP],
            o_r[OFF_NOTE_MASK:OFF_GROUND_IMP])
    np.testing.assert_allclose(env.palms().cpu().numpy(),
                               np.array([s.palms() for s in sims]),
                               atol=1e-12)


def test_flux_scripted_points_match_rust():
    charts, expect = _scripted_charts()
    env, sims = _build(charts, 400, foot_contact=False)
    env.reset()
    done = [False] * len(sims)
    for t in range(40):
        tobs, trew, tdone, _ = env.step(torch.zeros(len(sims), N_JOINTS,
                                                    dtype=torch.long))
        obs_c = tobs.cpu().numpy()
        for e, s in enumerate(sims):
            if done[e]:
                continue
            o_r, r_r, d_r, _ev = s.step(ZERO)
            tag = f"t{t} env{e}"
            assert d_r == bool(tdone[e]), f"{tag} done"
            assert abs(r_r - float(trew[e])) < 1e-9, f"{tag} reward"
            np.testing.assert_allclose(obs_c[e], np.asarray(o_r), atol=1e-6,
                                       err_msg=f"{tag} obs")
            st = np.array(s.status(), dtype=np.uint8)
            np.testing.assert_array_equal(
                env.status[e, :len(st)].cpu().numpy(), st, err_msg=tag)
            done[e] = d_r
        if all(done):
            break
    assert all(done)
    for e, s in enumerate(sims):
        assert s.score == expect[e], f"env{e} rust score {s.score}"
        assert float(env.score[e]) == s.score, f"env{e} cuda score"
        assert (float(env.score_pre[e]), float(env.score_acc[e])) == \
            s.score_components()[:2], f"env{e} components"
        assert [int(env.good[e]), int(env.bad[e]), int(env.miss[e])] == \
            [s.good_cuts, s.bad_cuts, s.misses], f"env{e} counters"
    assert [s.bad_cuts for s in sims][2] == 1       # wrong hand
    assert sims[3].misses == 1 and sims[8].misses == 1
    assert sims[10].bad_cuts == 1 and sims[10].misses == 0   # > 60 deg


# -- lockstep whole episodes ----------------------------------------------------

def _lockstep_charts(seeds, chart_s):
    """Scripted prefix (a right circle and a bar at the reset palms, then
    a left circle at the left palm) + a dense generated tail."""
    sim = _reset_rust()
    palms = sim.palms()
    cfg = FluxConfig(chart_len_s=chart_s, first_note_s=1.0,
                     circle_prob=0.6, bar_prob=0.4)
    charts = []
    for s in seeds:
        (lx, ly, lz), (rx, ry, rz) = palms
        pre = [
            FluxObj(hit_time=_arrive_now(sim, rz, t_tick=2), x=rx, y=ry,
                    kind=KIND_CIRCLE, speed=SPEED, color=1),
            _bar_across(sim, palms, 0.5, 0.3, t_tick=3),
            FluxObj(hit_time=_arrive_now(sim, lz, t_tick=4), x=lx, y=ly,
                    kind=KIND_CIRCLE, speed=SPEED, color=0),
        ]
        tail = generate_flux_chart(cfg, np.random.default_rng(s))
        charts.append(sorted(pre + tail, key=lambda o: o.hit_time))
    return charts


@pytest.mark.parametrize("foot_contact", [False, True])
def test_flux_lockstep_parity(foot_contact):
    N = 6
    held = range(N) if foot_contact else ()
    charts = _lockstep_charts(range(900, 900 + N), chart_s=3.0)
    max_steps = max(int((ch[-1].hit_time + 0.5) * 30) + 90 for ch in charts)
    env, sims = _build(charts, max_steps, foot_contact, combo_walk=True)
    env.reset()
    stance_q, _ = rust_state(sims[0])
    stance_v = [0.0] * 35
    _resync(env, sims)
    rng = np.random.default_rng(77)
    done = [False] * N
    walks = [ComboTracker() for _ in range(N)]
    flips = macro = ticks = 0
    for t in range(max_steps + 1):
        acts = rng.integers(0, GEARS, size=(N, N_JOINTS))
        for e in held:
            acts[e, POSTURE_J] = 0
        for e in range(N):
            if done[e]:
                acts[e] = 0
        tobs, trew, tdone, _ = env.step(torch.as_tensor(acts))
        obs_c = tobs.cpu().numpy()
        rew_c = trew.cpu().numpy()
        done_c = tdone.cpu().numpy()
        q_c = env.qpos().cpu().numpy()
        status_c = env.status.cpu().numpy()
        prev_c = np.stack([env.prev_idx.cpu().numpy(),
                           env.prev_valid.cpu().numpy()], -1)
        cnt_c = torch.stack([env.good, env.bad, env.miss], 1).cpu().numpy()
        sc_c = torch.stack([env.score, env.score_pre, env.score_acc,
                            env.score_post], 1).cpu().numpy()
        for e, s in enumerate(sims):
            if done[e]:
                continue
            o_r, r_r, d_r, ev = s.step([int(x) for x in acts[e]])
            tag = f"t{t} env{e}"
            assert d_r == bool(done_c[e]), f"{tag} done"
            o_r = np.asarray(o_r, dtype=np.float64)
            od = np.abs(o_r - obs_c[e])
            rew_d = abs(r_r - float(rew_c[e]))
            q_d = float(np.abs(np.array(s.qpos()) - q_c[e]).max())
            ticks += 1
            np.testing.assert_array_equal(
                obs_c[e, OFF_NOTE_MASK:OFF_GROUND_IMP],
                o_r[OFF_NOTE_MASK:OFF_GROUND_IMP], err_msg=f"{tag} mask")
            np.testing.assert_array_equal(obs_c[e, _PREV_VALID],
                                          o_r[_PREV_VALID],
                                          err_msg=f"{tag} prev_valid")
            if not (rew_d < 1e-9 and od.max() < 1e-5 and q_d < 1e-9):
                flips += 1
                macro += int(od.max() > 1e-3)
                assert rew_d < 0.05 and q_d < 0.5, \
                    f"{tag} beyond the knife-edge band: rew {rew_d} q {q_d}"
            st_r = np.array(s.status(), dtype=np.uint8)
            np.testing.assert_array_equal(status_c[e, :len(st_r)], st_r,
                                          err_msg=f"{tag} status")
            assert [tuple(x) for x in prev_c[e].tolist()] == \
                [(int(i), int(v)) for i, v in s.prev_note()], \
                f"{tag} prev record"
            assert cnt_c[e].tolist() == [s.good_cuts, s.bad_cuts,
                                         s.misses], f"{tag} counters"
            assert sc_c[e].tolist() == [s.score, *s.score_components()], \
                f"{tag} score decomposition"
            assert {c for _i, c, *_ in ev} <= {0, 3, 4}, f"{tag} codes"
            walks[e].consume([(int(i), EVENT_REASONS[c], p)
                              for i, c, p, *_ in ev])
            if d_r:
                done[e] = True
                cw = env.combo()
                assert (float(cw[0][e]), int(cw[1][e]), int(cw[2][e])) == \
                    (walks[e].score, walks[e].max_combo, walks[e].breaks), \
                    f"env{e} combo walk"
        if all(done):
            break
        hold = (t + 1) % HOLD_EVERY == 0
        zero_v = []
        for e, s in enumerate(sims):
            if done[e]:
                continue
            if hold and e in held:
                q, v = rust_state(s)
                for i in range(7):
                    q[i] = stance_q[i]
                for i in range(6):
                    v[i] = 0.0
                for j in POSTURE_J:
                    q[7 + j] = stance_q[7 + j]
                    v[6 + j] = 0.0
                s.set_joints(q, v)   # zeroes Rust's wrist + palm vel
                zero_v.append(e)
        # ended envs: Rust and the combo-walk-frozen CUDA env both hold
        # their final state, so resyncing every env is harmless
        _resync(env, sims)
        if zero_v:
            env.wrist_vel[zero_v] = 0.0
            env.palm_vel[zero_v] = 0.0
    assert all(done), "episodes did not finish inside max_steps"
    print(f"flux lockstep (contact={foot_contact}): {ticks} env-ticks, "
          f"{flips} flips ({macro} > 1e-3)")
    assert flips <= 0.2 * ticks and macro <= 0.08 * ticks
    # coverage: circles hit by the right hand, bars crossed, misses
    goods = sum(s.good_cuts for s in sims)
    assert goods >= N and sum(s.misses for s in sims) > 0
    bar_goods = sum(
        1 for ch, s in zip(charts, sims)
        for o, st in zip(ch, s.status()) if o.kind == KIND_BAR and st == 1)
    assert bar_goods >= N


def test_flux_parity_f32_bounded():
    """f32 well-formedness + bounded agreement (the standard f32 tier)."""
    charts = _lockstep_charts(range(950, 954), chart_s=3.0)
    max_steps = max(int((ch[-1].hit_time + 0.5) * 30) + 90 for ch in charts)
    env, sims = _build(charts, max_steps, foot_contact=False,
                       dtype=torch.float32)
    env.reset()
    rng = np.random.default_rng(4)
    r_rew = np.zeros(4)
    c_rew = np.zeros(4)
    for t in range(60):
        acts = rng.integers(0, GEARS, size=(4, N_JOINTS))
        tobs, trew, _td, _ = env.step(torch.as_tensor(acts))
        assert torch.isfinite(tobs).all(), f"t{t}: non-finite obs"
        assert torch.isfinite(env.palm_vel).all()
        for e, s in enumerate(sims):
            o_r, r_r, _d, _ev = s.step([int(x) for x in acts[e]])
            d = np.abs(tobs[e].cpu().numpy() - np.asarray(o_r))
            assert (d > 5e-3).mean() <= 0.75, f"t{t} env{e} lane ceiling"
            r_rew[e] += r_r
            c_rew[e] += float(trew[e])
    np.testing.assert_allclose(c_rew, r_rew, atol=3.0)
    for e, s in enumerate(sims):
        assert abs(s.good_cuts - int(env.good[e])) <= 3
        assert abs(s.score - float(env.score[e])) <= 300.0


def test_flux_procedural_pool_and_auto_reset():
    env = CudaBeatVecEnv(4, game="flux", seed=3, chart_pool=8,
                         chart_len_s=2.0, dtype=torch.float32,
                         foot_contact=True)
    assert env.flux and bool((env.pool["len"] > 0).any())
    obs = env.reset()
    assert obs.shape == (4, OBS_DIM)
    rng = np.random.default_rng(0)
    dones = 0
    for _ in range(env.max_steps):
        obs, _r, done, _ = env.step(
            torch.as_tensor(rng.integers(0, 3, size=(4, N_JOINTS))))
        dones += int(done.sum())
    assert torch.isfinite(obs).all() and dones >= 4
    assert int(env.bombs.sum()) == 0 and int(env.walls_in.sum()) == 0
