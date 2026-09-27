"""CUDA-vs-Rust lockstep in the PRODUCTION configuration (test audit
item 17).

The other cross-engine lockstep gates run a reduced env: weld mode
(test_v10_parity_cuda) or contact mode with a far-away note and zero
shaping (test_v11). This one (it replaced the weld whole-episode
lockstep tiers of test_v10_parity_cuda, test_v45_prev_note and
test_v17_wall_parity, 2026-09-22) builds both engines the way the
trainer does: the kwargs of the ranked-pool CudaBeatVecEnv call in
train_batch.py, values read from the trainer's own argparse defaults
(= the v58 recipe):

  contact mode, gears, fall_const, foot z/x rent + hinge + shaping,
  energy, limit (all-body), wall despawn, self-contact coef, ep_stats
  (in-kernel ScoreSaber combo walk), TAU_BODY_DEFAULT body, asinh z
  (C = 8 m), env_order=True

with the training-only randomness that has no Rust path zeroed (pushes,
start kick, start-pose jitter, crouch starts, tau draws — the body is
then pinned at the recipe's tau CENTER on both engines). The one
production departure is the dtype: f64 on CUDA vs the f64 Rust engine,
the only tier with a tight tolerance.

Charts: procedural (both colours, dots, bombs, extras), ~4 s, plus
hand-placed walls per env (crossing, dodged side wall, hanging crouch
wall, two overlapping crossings).

Protocol = test_v11's contact lockstep: the CUDA dynamics state
re-takes Rust's after EVERY tick (drift never accumulates; the env
layer — swing chain, trackers, status, prev-note record, walls, combo,
telemetry — evolves freely on each side). Two harness devices keep
whole episodes in frame without touching engine semantics:
  * HELD envs (0..5): arms random, spine + legs coast, and every
    HOLD_EVERY ticks the root / spine / legs are written back to the
    reset stance at rest on BOTH engines (set_joints / set_state, which
    zero the finite-diff wrist velocity on both). The raw body has no
    postural controller, so without this no random-action episode
    survives to its chart end; with it they play the whole chart.
  * FREE envs (6, 7): random gears on every joint — they tilt or fall,
    which is the termination-reason half of the gate.
  * A CUDA env that ended is PARKED (re-takes the reset stance each
    tick, zero actions): auto_reset is off but the kernel still steps
    it, and parking keeps its ticks out of the global pop counters.

Asserted per tick: done timing; obs (tight band 1e-5 on all 624 lanes,
knife-edge flips counted and bounded as in test_v11; mask, tau and
prev-valid lanes EXACT); reward (1e-9 tight, flips < 0.1); qpos;
per-note status, prev-note record, counters, score decomposition,
walls_in EXACT; global pop counters (fall, tilt, foot lifts, envelope,
pd floor) tick by tick. At each env's done tick: the ep_stats row
(ticks, return, fell, good, score, combo walk vs scoresaber.ComboTracker
fed Rust's events, completed, bad, misses, body) and the per-env
telemetry (activation, torque, limit rest, self contact, wrist
reversals).
"""
import sys

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CudaBeatVecEnv needs CUDA", allow_module_level=True)

import beat_arena_rs
from beat_arena_env.chart import (ChartConfig, Wall, generate_chart,
                                  note_tuple, wall_tuple)
from beat_arena_env.constants import (EVENT_REASONS, KIND_BOMB, N_JOINTS,
                                      OBS_DIM, OFF_CONTACT, OFF_GROUND_IMP,
                                      OFF_NOTE_MASK, OFF_NOTES, OFF_PREV_L,
                                      OFF_PREV_R, OFF_TAU, PREV_FEAT,
                                      TAU_BODY_DEFAULT, parse_tau_scale)
from beat_arena_env.cuda_env import CudaBeatVecEnv
from beat_arena_env.scoresaber import ComboTracker
from tests._helpers import Z_ASINH_C, resync, rust_state

HOLD_EVERY = 3
ACT_SEED = 58
# joints the hold pins (spine 0..2, legs 17..28); the arms (3..16) play
POSTURE_J = [0, 1, 2] + list(range(17, 29))

# walls per env, (start_time, cx, cy, width, height, length, speed):
# 0 crossing, 1 dodged side wall, 2 hanging crouch wall, 3 two
# overlapping crossings, 4/5 later crossings, 6/7 (free) early
# crossings (charged before those bodies go down)
_CROSS = lambda t: Wall(t, 0.0, 0.0, 20.0, 40.0, 2.0, 4.0)
WALLS = [
    [_CROSS(1.5)],
    [Wall(1.2, 2.5, 1.5, 0.6, 3.0, 2.0, 4.0)],
    [Wall(1.4, 0.0, 2.1, 2.4, 1.8, 4.0, 4.0)],
    [_CROSS(1.6), Wall(1.8, 0.0, 0.0, 20.0, 40.0, 1.0, 4.0)],
    [_CROSS(2.0)],
    [_CROSS(2.4)],
    [_CROSS(0.2)],
    [_CROSS(0.3)],
]
# (n_envs, held envs, chart seconds, chart seeds, walls) per arm
PRODUCTION = (8, range(0, 6), 3.0, range(500, 508), WALLS)
PERSISTENT = (4, range(0, 4), 2.2, range(600, 604), WALLS[:4])

# obs lane groups for the flip census
_GROUPS = {
    "body": (0, OFF_CONTACT),
    "contact": (OFF_CONTACT, OFF_NOTES),
    "notes": (OFF_NOTES, OFF_NOTE_MASK),
    "mask": (OFF_NOTE_MASK, OFF_GROUND_IMP),
    "ground_bodypts": (OFF_GROUND_IMP, OFF_PREV_L),
    "prev": (OFF_PREV_L, OFF_TAU),
    "tau": (OFF_TAU, OBS_DIM),
}
_PREV_VALID = [OFF_PREV_L + PREV_FEAT - 1, OFF_PREV_R + PREV_FEAT - 1]


def _recipe():
    """The trainer's argparse defaults (= the production recipe)."""
    from train_batch import parse_args
    old = sys.argv
    try:
        sys.argv = ["train_batch.py"]
        return parse_args()
    finally:
        sys.argv = old


def _charts(chart_s, seeds):
    cfg = ChartConfig(chart_len_s=chart_s, colors="both", bomb_prob=0.25,
                      extra_prob=0.3, dot_prob=0.15)
    return [generate_chart(cfg, np.random.default_rng(s)) for s in seeds]


def _build(a, charts, walls, max_steps, tau):
    N = len(charts)
    wt = [[wall_tuple(w) for w in ws] for ws in walls]
    # the ranked-pool VecCls(...) call in train_batch.py, kwarg for
    # kwarg; zeroed = training-only randomness the Rust twin lacks
    env = CudaBeatVecEnv(
        N, charts=charts, walls=wt, max_steps=max_steps,
        gears=a.gears, fall_const=a.fall_pen,
        push_prob=0.0, push_vmax=a.push_vmax, start_jitter=0.0,
        foot_z_coef=a.foot_z_coef, foot_x_coef=a.foot_x_coef,
        foot_hinge_r=a.foot_hinge_r, foot_shape_coef=a.foot_shape_coef,
        energy_coef=a.energy_coef, limit_coef=a.limit_coef,
        limit_arms=a.limit_arms, wall_despawn=a.wall_despawn,
        start_pose_jitter=0.0, crouch_start_prob=0.0,
        z_asinh=Z_ASINH_C, self_contact_coef=a.self_contact_coef,
        tau_rand=0.0, ep_stats=a.ep_stats, tau_center=tau,
        env_order=True, seed=a.seed, device="cuda",
        dtype=torch.float64, auto_reset=False)
    sims = []
    for ch, ws in zip(charts, wt):
        sim = beat_arena_rs.RawBeatSim(
            max_steps=max_steps, gears=a.gears,
            fall_const=a.fall_pen, foot_z_coef=a.foot_z_coef,
            foot_x_coef=a.foot_x_coef, foot_hinge_r=a.foot_hinge_r,
            foot_shape_coef=a.foot_shape_coef, energy_coef=a.energy_coef,
            limit_coef=a.limit_coef, limit_arms=a.limit_arms,
            wall_despawn=a.wall_despawn,
            self_contact_coef=a.self_contact_coef)
        sim.set_tau_scale(list(tau))   # persists across set_chart resets
        sim.set_chart([note_tuple(n) for n in ch], ws)
        sims.append(sim)
    return env, sims


def _rust_pops(sim):
    return np.array([sim.pop_fall_hits(), sim.pop_tilt_hits(),
                     *sim.pop_foot_lifts(), sim.pop_envelope_hits(),
                     sim.pop_pd_hits()], dtype=np.int64)


def _cuda_pops(env):
    return np.array([env.pop_fall_hits(), env.pop_tilt_hits(),
                     *env.pop_foot_lifts(), env.pop_envelope_hits(),
                     env.pop_pd_hits()], dtype=np.int64)


def test_production_lockstep_parity():
    """The trainer's configuration, recipe wall_despawn ON."""
    a = _recipe()
    assert a.wall_despawn
    ends, sims = _lockstep(a, *PRODUCTION)
    # coverage: whole charts played AND the fall/tilt path fired
    reasons = [r for r, _ in ends.values()]
    assert "completed" in reasons, ends
    assert "fall" in reasons or "tilt" in reasons, ends
    assert sims[0].walls_in > 0 and sims[3].walls_in > 0
    assert sims[6].walls_in > 0, "free env went down before its wall"


def test_persistent_wall_lockstep_parity():
    """Same lockstep with wall_despawn OFF — the bench / replay wall
    mode (bench_scoresaber runs the CUDA env at the default False):
    persistent walls, the per-tick event-7 stream, walls_in counting
    in-wall ticks once per tick however many walls overlap the head,
    and the entry edge (event 8) as the combo break. The v17 wall
    gate's regimes (crossing, dodged, crouch, overlapping) in contact
    mode."""
    a = _recipe()
    a.wall_despawn = False
    ends, sims = _lockstep(a, *PERSISTENT)
    assert all(r == "completed" for r, _ in ends.values()), ends
    # a crossing wall is 0.5 s deep: many in-wall ticks per entry
    assert sims[0].walls_in > 5 * sims[0].wall_enters >= 5
    # overlap de-duplication: the second wall lies inside the first's
    # span, so one entry edge and at most one in-wall count per tick
    # (the per-tick check); the crouch wall charges the standing head
    assert sims[0].wall_enters == 1 and sims[3].wall_enters == 1
    assert sims[2].walls_in > 0


def _lockstep(a, N, HELD, chart_s, seeds, walls):
    """Run the lockstep over whole episodes; returns (ends, sims) where
    ends[e] = (termination reason, ticks)."""
    # the kwargs mirror the ranked-pool (production) branch
    assert a.ranked_maps and a.foot_contact and a.ep_stats
    assert a.z_asinh is None and Z_ASINH_C == 8.0, \
        "recipe z encoding is asinh C = 8 m (unset BEAT_Z_ASINH)"
    tau = parse_tau_scale(a.tau_center)
    assert tuple(tau) == TAU_BODY_DEFAULT
    charts = _charts(chart_s, seeds)
    kinds = {(n.kind == KIND_BOMB, n.color) for ch in charts for n in ch}
    assert {(False, 0), (False, 1), (True, 0)} <= kinds, kinds
    max_steps = max(int((max(n.hit_time for n in ch) + 0.5) * 30) + 90
                    for ch in charts)
    env, sims = _build(a, charts, walls, max_steps, tau)
    env.reset()
    assert env.chart_idx.cpu().tolist() == list(range(N))
    np.testing.assert_array_equal(env.tau_scale.cpu().numpy(),
                                  np.tile(tau, (N, 1)))
    for sim in sims:
        assert list(sim.tau_scale()) == list(tau)
    stance_q, _ = rust_state(sims[0])
    stance_v = [0.0] * 35
    q36s, v35s = zip(*(rust_state(s) for s in sims))
    resync(env, list(q36s), list(v35s))

    rng = np.random.default_rng(ACT_SEED)
    done = [False] * N
    combo = [ComboTracker() for _ in range(N)]
    ret_r = np.zeros(N)
    ret_c = np.zeros(N)
    walls_in_prev = [[0] * N, [0] * N]   # walls_in, wall_enters
    ends = {}
    flips = macro = ticks = 0
    grp_flips = dict.fromkeys(_GROUPS, 0)
    worst = {"rew": 0.0, "obs": 0.0, "q": 0.0}
    max_flip_rew = 0.0
    for t in range(max_steps + 1):
        acts = rng.integers(0, a.gears, size=(N, N_JOINTS))
        for e in HELD:
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
        cnt_c = torch.stack([env.good, env.bad, env.miss, env.bombs,
                             env.walls_in], 1).cpu().numpy()
        sc_c = torch.stack([env.score, env.score_pre, env.score_acc,
                            env.score_post], 1).cpu().numpy()
        pops_c = _cuda_pops(env)
        pops_r = np.zeros_like(pops_c)
        for e, sim in enumerate(sims):
            if done[e]:
                continue
            obs_r, rew_r, d_r, ev = sim.step([int(x) for x in acts[e]])
            pops_r += _rust_pops(sim)
            tag = f"t{t} env{e}"
            assert d_r == bool(done_c[e]), \
                f"{tag} done: rust {d_r} cuda {bool(done_c[e])}"
            obs_r = np.asarray(obs_r, dtype=np.float64)
            od = np.abs(obs_r - obs_c[e])
            rew_d = abs(rew_r - float(rew_c[e]))
            q_d = float(np.abs(np.array(sim.qpos()) - q_c[e]).max())
            ticks += 1
            ret_r[e] += rew_r
            ret_c[e] += float(rew_c[e])
            # discrete lanes: EXACT every tick
            for g in ("mask", "tau"):
                lo, hi = _GROUPS[g]
                np.testing.assert_array_equal(obs_c[e, lo:hi], obs_r[lo:hi],
                                              err_msg=f"{tag} {g} lanes")
            np.testing.assert_array_equal(obs_c[e, _PREV_VALID],
                                          obs_r[_PREV_VALID],
                                          err_msg=f"{tag} prev_valid")
            # identical end state => the env-layer lanes (note / wall
            # tokens, prev-note blocks) are functions of it and the
            # chart; hold them at the v17 wall gate's 1e-4 even when a
            # velocity lane flipped
            if q_d < 1e-9:
                for g in ("notes", "prev"):
                    lo, hi = _GROUPS[g]
                    assert od[lo:hi].max() < 1e-4, \
                        f"{tag} {g} lanes {od[lo:hi].max():.3e} off " \
                        f"with the state in the seed band"
            if rew_d < 1e-9 and od.max() < 1e-5 and q_d < 1e-9:
                worst["rew"] = max(worst["rew"], rew_d)
                worst["obs"] = max(worst["obs"], float(od.max()))
                worst["q"] = max(worst["q"], q_d)
            else:
                flips += 1
                macro += int(od.max() > 1e-3)
                max_flip_rew = max(max_flip_rew, rew_d)
                for g, (lo, hi) in _GROUPS.items():
                    grp_flips[g] += int(od[lo:hi].max() >= 1e-5)
                # reward: the weld tier's 0.05 (score-rounding flips
                # <= 2/115 plus shaping; measured 1.3e-3); state: the
                # contact tier's knife-edge sanity bound
                assert rew_d < 0.05 and q_d < 0.5, \
                    f"{tag} beyond the knife-edge band: rew {rew_d} q {q_d}"
            # env layer: EXACT every tick
            st_r = np.array(sim.status(), dtype=np.uint8)
            np.testing.assert_array_equal(status_c[e, :len(st_r)], st_r,
                                          err_msg=f"{tag} status")
            pr = sim.prev_note()
            assert [tuple(x) for x in prev_c[e].tolist()] == \
                [(int(i), int(v)) for i, v in pr], f"{tag} prev record"
            cnt_r = [sim.good_cuts, sim.bad_cuts, sim.misses,
                     sim.bombs_hit, sim.walls_in]
            assert cnt_c[e].tolist() == cnt_r, \
                f"{tag} counters cuda {cnt_c[e].tolist()} rust {cnt_r}"
            assert sc_c[e].tolist() == [sim.score,
                                        *sim.score_components()], \
                f"{tag} score decomposition"
            # wall event contract (lib.rs / .cu wall block). walls_in is
            # compared CUDA vs Rust above; the combo break event 8
            # carries is checked through the combo walk at done.
            codes = [c for _i, c, *_ in ev]
            d_in = sim.walls_in - walls_in_prev[0][e]
            d_ent = sim.wall_enters - walls_in_prev[1][e]
            if a.wall_despawn:
                # env v37: each charged wall emits ONE entry event 8,
                # adds 1 to walls_in (entries) and despawns; no 7s
                assert 7 not in codes, f"{tag} event 7 under despawn"
                assert codes.count(8) == d_in == d_ent, \
                    f"{tag} wall entry events vs walls_in"
            else:
                # persistent: one 7 per in-wall tick (overlaps charge
                # once), one 8 on the entry edge
                assert codes.count(7) == d_in <= 1, f"{tag} event 7"
                assert codes.count(8) == d_ent <= d_in, f"{tag} event 8"
            walls_in_prev[0][e] = sim.walls_in
            walls_in_prev[1][e] = sim.wall_enters
            combo[e].consume([(int(i), EVENT_REASONS[c], p)
                              for i, c, p, *_ in ev])
            if d_r:
                done[e] = True
                ends[e] = _check_episode_end(env, sim, e, combo[e],
                                             ret_r[e], ret_c[e], tau,
                                             max_steps)
        np.testing.assert_array_equal(
            pops_c, pops_r,
            err_msg=f"t{t} [fall, tilt, lift L, lift R, envelope, pd]")
        if all(done):
            break
        # resync: live envs re-take Rust's state; ended envs park at the
        # reset stance; held envs are re-posed every HOLD_EVERY ticks
        hold = (t + 1) % HOLD_EVERY == 0
        q36s, v35s, zero_wv = [], [], []
        for e, sim in enumerate(sims):
            if done[e]:
                q36s.append(list(stance_q))
                v35s.append(list(stance_v))
                zero_wv.append(e)
                continue
            if hold and e in HELD:
                q, v = rust_state(sim)
                for i in range(7):
                    q[i] = stance_q[i]
                for i in range(6):
                    v[i] = 0.0
                for j in POSTURE_J:
                    q[7 + j] = stance_q[7 + j]
                    v[6 + j] = 0.0
                sim.set_joints(q, v)   # zeroes Rust's wrist velocity
                zero_wv.append(e)
            q, v = rust_state(sim)
            q36s.append(q)
            v35s.append(v)
        resync(env, q36s, v35s)
        if zero_wv:
            env.wrist_vel[zero_wv] = 0.0
    assert all(done), "episodes did not finish inside max_steps"

    print(f"lockstep (wall_despawn={a.wall_despawn}): {ticks} env-ticks, {flips} tight-band "
          f"flips ({macro} > 1e-3), per group {grp_flips}, max flip "
          f"reward diff {max_flip_rew:.3e}; tight max diffs rew "
          f"{worst['rew']:.3e} obs {worst['obs']:.3e} q {worst['q']:.3e}; "
          f"ends {ends}")
    # the contact tier's ceiling (test_v11: measured 8.3 %, gate 20 %),
    # and the weld tiers' macroscopic-flip ceiling (8 %)
    assert flips <= 0.2 * ticks, f"{flips}/{ticks} flip env-ticks"
    assert macro <= 0.08 * ticks, f"{macro}/{ticks} flips beyond 1e-3"
    # coverage: notes resolved; the dodged side wall never charged
    assert sum(s.good_cuts + s.bad_cuts + s.misses for s in sims) > 0
    assert sims[1].walls_in == 0 and sims[1].wall_enters == 0
    return ends, sims


def _check_episode_end(env, sim, e, walk, ret_r, ret_c, tau, max_steps):
    """At env e's done tick: the ep_stats row and the per-env telemetry
    against the Rust sim. Returns (termination reason, ticks)."""
    tag = f"env{e} end"
    rec = env.ep_rec[e].cpu().numpy()
    assert int(env.ep_flag[e]) == 1, f"{tag}: no ep_stats row"
    fell_r = sim.falls + sim.tilts > 0
    if sim.falls:
        reason = "fall"
    elif sim.tilts:
        reason = "tilt"
    elif sim.escaped:
        reason = "escape"
    elif sim.tick >= max_steps:
        reason = "cap"
    else:
        reason = "completed"
    np.testing.assert_array_equal(rec[:16], np.asarray(tau),
                                  err_msg=f"{tag} body")
    assert int(rec[16]) == sim.tick, f"{tag} ticks"
    # the kernel's return is its own reward sum; the engines' returns
    # differ by at most the per-tick flip band summed
    assert abs(rec[17] - ret_c) < 1e-9, f"{tag} cuda ep_return"
    assert abs(ret_c - ret_r) < 0.5, f"{tag} return {ret_c} vs {ret_r}"
    assert bool(rec[18]) == fell_r, f"{tag} fell/tilted"
    assert int(rec[19]) == sim.good_cuts and rec[20] == sim.score, tag
    assert int(rec[21]) == e, f"{tag} chart index"
    assert (rec[22], int(rec[23]), int(rec[24])) == \
        (walk.score, walk.max_combo, walk.breaks), \
        f"{tag} combo walk cuda {rec[22:25]} rust " \
        f"{(walk.score, walk.max_combo, walk.breaks)}"
    assert bool(rec[25]) == (reason == "completed"), f"{tag} completed"
    assert (int(rec[26]), int(rec[27])) == (sim.bad_cuts, sim.misses), tag
    # per-env telemetry accumulated over the episode (the kernel keeps
    # stepping an ended env, so read it on the done tick)
    act_r = sim.pop_activation()
    act_c = env.act_stat[e].tolist()
    np.testing.assert_allclose(act_c[:4], act_r[:4], rtol=0, atol=1e-9,
                               err_msg=f"{tag} activation")
    assert int(act_c[4]) == act_r[4] == sim.tick, f"{tag} activation ticks"
    tq_r = sim.pop_torque()
    tq_c = env.tau_stat[e].tolist()
    np.testing.assert_allclose(tq_c[:4], tq_r[:4], rtol=1e-12, atol=1e-9,
                               err_msg=f"{tag} torque")
    assert int(tq_c[4]) == tq_r[4], f"{tag} torque ticks"
    # solver-output telemetry: a knife-edge flip tick moves it, so a
    # band rather than the seed tolerance
    lr_r = sim.pop_limit_rest()
    np.testing.assert_allclose(env.limit_rest[e].tolist(), lr_r[:3],
                               rtol=0.05, atol=1e-6,
                               err_msg=f"{tag} limit rest")
    assert int(env.limit_rest_ticks[e]) == lr_r[3], f"{tag} limit ticks"
    sc_r = sim.pop_self_contact()
    sc_c = env.sc_stat[e].tolist()
    np.testing.assert_allclose(sc_c[0], sc_r[0], rtol=0.05, atol=1e-6,
                               err_msg=f"{tag} self contact")
    assert int(sc_c[2]) == sc_r[2], f"{tag} self-contact ticks"
    wl, wr, wt = sim.wrist_rev()
    assert env.wrev[e].tolist() == [wl, wr] and wt == sim.tick, \
        f"{tag} wrist reversals cuda {env.wrev[e].tolist()} rust {(wl, wr)}"
    return reason, sim.tick
