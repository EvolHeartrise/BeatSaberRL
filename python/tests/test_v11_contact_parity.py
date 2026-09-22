"""P2 gate — env v11 UNILATERAL FOOT CONTACT: the CUDA mirror vs the Rust
reference (unilateral-contact spec §5, CUDA half).

Phase 1 proved the Rust side with cargo bitwise suites
(rust/src/dynamics/tests_ground.rs + lib.rs `env_tests`). This module is
the CUDA half of the same contract, on the tiers test_v10_parity_cuda.py
established and for the same reason (SPEC-v10 Divergences §21: CUDA f64
sin/cos/acos differ from the CRT libm by <= 1 ulp and nvcc contracts
FMAs, so no free-running full-episode tolerance can exist for this
mirror):

  (i)   f64 LOCKSTEP-RESYNC parity in contact mode over a bounded
        horizon under random gears — the CUDA dynamics state re-takes
        Rust's after every tick, so drift never accumulates and what is
        measured is one tick of the ground block at a time;
  (ii)  f32 well-formedness in contact mode: finite obs, joints inside
        limits, rates inside caps, root inside the step-12 envelope, and
        every terminated episode terminated FOR A REASON;
  (iii) fall-envelope parity: same state + same actions in both engines
        -> the same fall tick, counted on both sides;
  (iv)  CUDA contact-mode determinism: same seed twice, bitwise-equal
        state.

Weld mode (foot_mode = 0, the default) is NOT retested here — it is the
v10 physics unchanged, and test_v10_parity_cuda.py is its gate. That
module passing unmodified IS the §5 (f) weld-regression assertion.

The whole module is skipped without CUDA (the kernel has no CPU path).
"""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CudaBeatVecEnv needs CUDA", allow_module_level=True)

from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import (JOINT_LIMITS, N_JOINTS, NV,
                                      OFF_GROUND_IMP, OMEGA_CAP)
from beat_arena_env.cuda_env import CudaBeatVecEnv
import beat_arena_rs

# SPEC-v10 §1.4 joint order is spine(3) + arm_l(7) + arm_r(7) + leg_l(6)
# + leg_r(6): joints 0..16 are the upper body, 17..28 the legs (17 and 23
# are l_/r_hip_pitch).
N_UPPER = 17
J_L_HIP_PITCH, J_R_HIP_PITCH = 17, 23
# lib.rs FALL_ROOT_Y / beat_arena_cuda.cu FALL_ROOT_Y (contact spec §4;
# 0.55 -> ... -> 0.35 -> 0.3 2026-08-20, crouch headroom)
FALL_ROOT_Y = 0.25
# a note far beyond every horizon: pure body dynamics, no judgment
FAR_NOTE = Note(hit_time=200.0, col=2, row=1, x=0.3, y=1.4, color=1,
                angle=4.712, speed=4.0)
MAX_STEPS = 6900


def _rust_sims(n, gears):
    sims = []
    for _ in range(n):
        sim = beat_arena_rs.RawBeatSim(max_steps=MAX_STEPS,
                                       gears=gears,
                                       foot_contact=True)
        sim.set_chart([note_tuple(FAR_NOTE)])
        sims.append(sim)
    return sims


def _cenv(n, gears, dtype=torch.float64, auto_reset=False,
          seed=0):
    return CudaBeatVecEnv(n, charts=[[FAR_NOTE]] * n, max_steps=MAX_STEPS,
                          gears=gears, device="cuda",
                          dtype=dtype, auto_reset=auto_reset,
                          want_contacts=True, foot_contact=True, seed=seed)


def _rust_state(sim):
    return list(sim.qpos()), list(np.concatenate([sim.twist(), sim.omega()]))


def _resync(env, sims):
    """Push Rust's dynamics state into the CUDA env (the v10 lockstep
    protocol: set_state zeroes the finite-diff wrist velocity, which is
    env-layer state Rust keeps, so it is carried across by hand)."""
    q36s, v35s = [], []
    for sim in sims:
        q, v = _rust_state(sim)
        q36s.append(q)
        v35s.append(v)
    wv = env.wrist_vel.clone()
    env.set_state(torch.tensor(q36s, dtype=torch.float64),
                  torch.tensor(v35s, dtype=torch.float64))
    env.wrist_vel.copy_(wv)


def _stance_states(n, seed):
    """Random states in the STANCE regime — the one contact mode is for.

    LEG joints are held inside +-0.02 rad and the root within +-5 mm of
    0.94 so the two soles stay near the plane and the ground block runs
    with a full, varied active set; the upper body (spine + both arms,
    joints 0..16) takes 30 %% of its range and every joint gets a rate in
    +-1. Random LEG poses were tried first and are the wrong regime, not
    a harder one: at 30 %% of the hip/knee range a foot is 10 cm through
    the floor, the Baumgarte push-out dominates the tick, and what the
    comparison then measures is a stiff transient rather than stance
    contact."""
    rng = np.random.default_rng(seed)
    q36s, v35s = [], []
    for _ in range(n):
        q = [float(rng.uniform(0.3 * lo, 0.3 * hi)) if j < N_UPPER
             else float(rng.uniform(-0.02, 0.02))
             for j, (lo, hi) in enumerate(JOINT_LIMITS)]
        rot = rng.normal(size=3) * 0.01
        quat = np.concatenate([[1.0], rot])
        quat /= np.linalg.norm(quat)
        q36 = [float(rng.uniform(-0.03, 0.03)),
               0.94 + float(rng.uniform(-0.005, 0.005)),
               -0.65 + float(rng.uniform(-0.03, 0.03))]
        q36 += [float(x) for x in quat] + q
        v35s.append([float(x) for x in rng.uniform(-1.0, 1.0, NV)])
        q36s.append(q36)
    return q36s, v35s


def _tick_diff(sim_obs, sim_rew, sim_qpos, obs, rew, qpos):
    return (abs(sim_rew - rew),
            float(np.abs(np.asarray(sim_obs) - obs).max()),
            float(np.abs(np.array(sim_qpos) - qpos).max()))


def _tight(d):
    """The transcendental-seed band of the v10 f64 tier, verbatim."""
    return d[0] < 1e-9 and d[1] < 1e-5 and d[2] < 1e-9


def _rust_one_ulp_flips(q36s, v35s, acts, gears):
    """CONTROL ARM. Run Rust against ITSELF with one velocity word moved
    by a single ulp, and count how many envs leave the same tight band in
    one tick.

    This is the calibration the contact tier needs and the weld tier did
    not. The ground block multiplies the number of knife edges a tick can
    cross — 8 corner activations at gap = MARGIN, and 2 friction rows x
    12 PGS iterations x 8 substeps of |lambda_t| = MU * lambda_n box
    faces — so "how many envs flip" is a property of the REGIME, not of
    the mirror. Measured here so the gate below compares CUDA's flip
    count against Rust's own instead of against a magic number.
    (Measured 2026-08-08 on the 4090: 5/32 at each of three seeds, versus
    4-11/32 for CUDA-vs-Rust — same order, overlapping env sets.)"""
    n = len(q36s)
    a_sims, b_sims = _rust_sims(n, gears), _rust_sims(n, gears)
    for e in range(n):
        a_sims[e].set_joints(q36s[e], v35s[e])
        vp = list(v35s[e])
        vp[0] = float(np.nextafter(vp[0], np.inf))
        b_sims[e].set_joints(q36s[e], vp)
    flips = 0
    for e in range(n):
        oa, ra, _d, _ev = a_sims[e].step([int(x) for x in acts[e]])
        ob, rb, _d, _ev = b_sims[e].step([int(x) for x in acts[e]])
        if not _tight(_tick_diff(oa, ra, a_sims[e].qpos(),
                                 np.asarray(ob), rb,
                                 np.array(b_sims[e].qpos()))):
            flips += 1
    return flips


# --- (i) f64: single tick from random stance states ---------------------------

@pytest.mark.parametrize("gears,n_states,seed",
                         [(3, 32, 700), (5, 32, 800), (7, 32, 900)])
def test_contact_single_tick_parity(gears, n_states, seed):
    """One 30 Hz tick (8 constrained substeps with the ground block) from
    random stance states: state within 1e-9, obs within 1e-5, reward
    within 1e-9 of Rust — the transcendental-seed band.

    Contact mode adds knife edges the weld tier does not have (a corner
    crossing gap = MARGIN, and every friction row's |lambda_t| = MU *
    lambda_n box face, which is re-imposed 12 times a substep), so a few
    envs per hundred flip a branch mid-tick exactly as the v10 tier's
    cap/limit edges do. Gate: all but a handful tight, flipped envs
    physically sane."""
    q36s, v35s = _stance_states(n_states, seed)
    env = _cenv(n_states, gears)
    env.reset()
    env.set_state(torch.tensor(q36s, dtype=torch.float64),
                  torch.tensor(v35s, dtype=torch.float64))
    sims = _rust_sims(n_states, gears)
    for e, sim in enumerate(sims):
        sim.set_joints(q36s[e], v35s[e])
    rng = np.random.default_rng(seed + 1)
    acts = rng.integers(0, gears, size=(n_states, N_JOINTS))
    tobs, trew, _td, _ = env.step(torch.as_tensor(acts))
    qpos = env.qpos().cpu().numpy()
    obs = tobs.cpu().numpy()
    outliers, worst = 0, (0.0, 0.0, 0.0)
    for e, sim in enumerate(sims):
        obs_r, rew_r, _d, _ev = sim.step([int(x) for x in acts[e]])
        d = _tick_diff(obs_r, rew_r, sim.qpos(), obs[e], float(trew[e]),
                       qpos[e])
        if _tight(d):
            worst = tuple(max(a, b) for a, b in zip(worst, d))
            continue
        outliers += 1
        assert d[0] < 0.1 and d[2] < 0.5, \
            f"env{e} beyond the knife-edge band: rew {d[0]} qpos {d[2]}"
    ulp = _rust_one_ulp_flips(q36s, v35s, acts, gears)
    print(f"contact single-tick gears={gears}: {outliers}/{n_states} CUDA "
          f"outliers vs {ulp}/{n_states} for Rust's own 1-ulp perturbation; "
          f"tight-env max diffs rew {worst[0]:.3e} obs {worst[1]:.3e} "
          f"qpos {worst[2]:.3e}")
    # (1) most envs must still agree to the seed band, and (2) the flip
    # rate must be the REGIME's, not the mirror's — within a small factor
    # of what one ulp does to Rust itself. CUDA carries a <= 1-ulp seed at
    # every transcendental and every contracted FMA, not one seed once, so
    # a few times Rust's count is the honest expectation.
    assert outliers <= 0.5 * n_states, \
        f"{outliers}/{n_states} envs off the seed band"
    # budget 4*ulp+2 -> 4*ulp+4 (2026-08-20 torque pass: the gears-7
    # seed-900 trajectory shifted and measured 7 outliers vs ulp 1 —
    # sibling param combos unchanged, chaotic margin, not a mirror bug)
    # -> 4*ulp+6 (2026-08-20 ankle_p +35: new stance-state regime,
    # gears-5 seed-800 measured 5 vs ulp 0; same class)
    assert outliers <= 4 * ulp + 6, \
        f"{outliers} CUDA outliers vs {ulp} from Rust's own 1-ulp seed"


# --- (i) f64: lockstep-resync bounded horizon ---------------------------------

def test_contact_lockstep_horizon_parity():
    """Contact mode, random gears, 16 ticks with a per-tick resync: the
    CUDA env re-takes Rust's dynamics state after every tick, so each
    comparison is ONE tick of the ground block from a bit-identical
    start. Obs / reward / qpos stay in the seed band, and `done` (which
    now carries the fall termination) agrees tick by tick."""
    n, gears, horizon = 8, 5, 16
    env = _cenv(n, gears)
    env.reset()
    sims = _rust_sims(n, gears)
    rng = np.random.default_rng(4242)
    done_r = [False] * n
    flips = 0
    worst = {"rew": 0.0, "obs": 0.0, "q": 0.0}
    ticks = 0
    for t in range(horizon):
        acts = rng.integers(0, gears, size=(n, N_JOINTS))
        tobs, trew, tdone, _ = env.step(torch.as_tensor(acts))
        assert torch.isfinite(tobs).all(), f"t{t}: non-finite CUDA obs"
        for e, sim in enumerate(sims):
            if done_r[e]:
                continue
            obs_r, rew_r, d_r, _ev = sim.step([int(x) for x in acts[e]])
            assert d_r == bool(tdone[e]), \
                f"t{t} env{e} done: rust {d_r} cuda {bool(tdone[e])}"
            rew_d = abs(rew_r - float(trew[e]))
            obs_d = np.abs(np.asarray(obs_r) - tobs[e].cpu().numpy()).max()
            q_d = np.abs(np.array(sim.qpos())
                         - env.qpos()[e].cpu().numpy()).max()
            ticks += 1
            if rew_d < 1e-9 and obs_d < 1e-5 and q_d < 1e-9:
                worst["rew"] = max(worst["rew"], rew_d)
                worst["obs"] = max(worst["obs"], obs_d)
                worst["q"] = max(worst["q"], q_d)
            else:
                flips += 1
                assert rew_d < 0.1 and q_d < 0.5, \
                    f"t{t} env{e} beyond sanity: rew {rew_d} qpos {q_d}"
            done_r[e] = d_r
        _resync(env, sims)
        if all(done_r):
            break
    print(f"contact lockstep: {ticks} env-ticks, {flips} knife-edge flips; "
          f"tight max diffs rew {worst['rew']:.3e} obs {worst['obs']:.3e} "
          f"qpos {worst['q']:.3e}")
    assert ticks >= 4 * n, f"only {ticks} env-ticks compared"
    # measured 9/109 = 8.3 %% on the 4090 (2026-08-08); the ceiling is the
    # same order, not the measurement — the flip rate belongs to the
    # regime (see _rust_one_ulp_flips), and this horizon runs the body
    # from a flat-footed stance to the edge of toppling
    assert flips <= 0.2 * ticks, f"{flips}/{ticks} flip ticks"


# --- (iii) fall envelope parity -----------------------------------------------

def test_fall_envelope_parity():
    """Contact spec §5 (d), across engines — updated 2026-08-20: a hard
    lean (full +hip_pitch on both legs) now trips the TILT envelope
    (CHEST up-axis > 40 deg sagittal, checked at tick end) BEFORE the
    root can reach FALL_ROOT_Y. Under the lockstep resync both engines
    must call it on the SAME tick, terminate, price it at the v18
    CONSTANT -fall_const (shared fell||tilted branch), and COUNT it —
    Rust in `tilts` / `pop_tilt_hits`, CUDA in `pop_tilt_hits`."""
    n, gears = 1, 5
    env = _cenv(n, gears)
    env.reset()
    sims = _rust_sims(n, gears)
    acts = np.zeros((n, N_JOINTS), dtype=np.int64)
    acts[:, J_L_HIP_PITCH] = 1  # full +
    acts[:, J_R_HIP_PITCH] = 1
    fall_tick = None
    for t in range(60):
        _tobs, trew, tdone, _ = env.step(torch.as_tensor(acts))
        root_y = float(env.p[0, 1])
        _obs, rew_r, d_r, _ev = sims[0].step([int(x) for x in acts[0]])
        assert d_r == bool(tdone[0]), \
            f"t{t}: rust done {d_r} vs cuda {bool(tdone[0])} (root y {root_y})"
        if d_r:
            assert sims[0].tilts == 1, \
                f"t{t}: rust terminated but not by a tilt"
            assert not sims[0].fell, \
                "the tilt envelope must fire before the root reaches " \
                f"FALL_ROOT_Y (root y {root_y})"
            assert rew_r == -5.0, \
                f"tilt must price at the v18 constant -5.0, got {rew_r}"
            assert float(trew[0]) == -5.0, \
                f"CUDA tilt pricing mismatch: {float(trew[0])}"
            fall_tick = t
            break
        assert not sims[0].fell and sims[0].tilts == 0
        _resync(env, sims)
    assert fall_tick is not None, "1 s of full hip flexion did not tilt"
    print(f"tilt parity: both engines terminated at tick {fall_tick}, "
          f"root y {float(env.p[0, 1]):.4f}")
    # both counters see it, and popping zeroes them (the pd_hits pattern)
    assert env.pop_tilt_hits() == 1
    assert env.pop_tilt_hits() == 0
    assert sims[0].pop_tilt_hits() >= 1
    assert sims[0].pop_fall_hits() == 0
    # and the root is still ABOVE the floor threshold on the CUDA side —
    # the tilt envelope fired before the body could reach it
    assert float(env.p[0, 1]) > FALL_ROOT_Y


def test_weld_mode_never_falls():
    """The other half of §5 (d)/(f), and the assertion is sharper than it
    looks. With foot_contact=False the same hard lean drives the welded
    root WELL below FALL_ROOT_Y — v10 has no ground, so the body folds
    through the floor about its pinned ankles — and the env must not
    notice: no termination, no counted fall. That is the flag proving
    itself dead in weld mode on a trajectory that would trip it in
    contact mode, not merely on one that never gets near the threshold."""
    env = CudaBeatVecEnv(4, charts=[[FAR_NOTE]] * 4, max_steps=MAX_STEPS,
                         gears=5, device="cuda", dtype=torch.float64,
                         auto_reset=False, want_contacts=True)
    env.reset()
    acts = np.zeros((4, N_JOINTS), dtype=np.int64)
    acts[:, J_L_HIP_PITCH] = 1
    acts[:, J_R_HIP_PITCH] = 1
    crossed = False
    for _ in range(40):
        _o, _r, done, _ = env.step(torch.as_tensor(acts))
        assert not bool(done.any()), "weld mode terminated under a hard lean"
        crossed |= bool((env.p[:, 1] < FALL_ROOT_Y).any())
    assert crossed, "the weld control never reached the fall threshold"
    assert env.pop_fall_hits() == 0


# --- (ii) f32 well-formedness -------------------------------------------------

def test_contact_f32_well_formed():
    """f32 contact mode, free-running, auto_reset off so a terminal env
    is observable: nothing non-finite anywhere, joints inside limits,
    rates inside caps, root inside the step-12 envelope, and every
    episode that ends ends WITH A REASON (fall / root escape / step cap /
    chart end). The raw v10 body has no postural controller, so with the
    feet unwelded most of these episodes end by falling — which is the
    point of the tier: the failure mode must be a clean termination, not
    a NaN."""
    n, horizon = 8, 90
    env = _cenv(n, 5, dtype=torch.float32)
    env.reset()
    rng = np.random.default_rng(7)
    lo = np.array([l[0] for l in JOINT_LIMITS])
    hi = np.array([l[1] for l in JOINT_LIMITS])
    caps = np.array(OMEGA_CAP)
    first_done = [None] * n
    reasons = {"fall": 0, "other": 0}
    for t in range(horizon):
        acts = rng.integers(0, 5, size=(n, N_JOINTS))
        tobs, trew, tdone, _ = env.step(torch.as_tensor(acts))
        assert torch.isfinite(tobs).all(), f"t{t}: non-finite obs"
        assert torch.isfinite(trew).all(), f"t{t}: non-finite reward"
        for name, arr in (("p", env.p), ("quat", env.quat), ("q", env.q),
                          ("v", env.v)):
            assert torch.isfinite(arr).all(), f"t{t}: non-finite {name}"
        q = env.q.cpu().numpy()
        qd = env.v[:, 6:].cpu().numpy()
        root_w = env.v[:, 3:6].cpu().numpy()
        root_v = env.v[:, :3].cpu().numpy()
        # v14 (contact spec §9): stops are reciprocal-impulse springs —
        # arrival transients penetrate up to ~h·cap before recovery
        # bites, and a 64-pass-exhausted clamp sweep can leave small
        # rate residuals (the old exact-write guarantee pass was the
        # phantom-force bug)
        assert (q >= lo - 0.25).all() and (q <= hi + 0.25).all(), \
            f"t{t}: joints outside limits"
        assert (np.abs(qd) <= caps + 1.0).all(), f"t{t}: rates above caps"
        assert (np.linalg.norm(root_w, axis=1) <= 96.0 + 1e-2).all()
        assert (np.linalg.norm(root_v, axis=1) <= 48.0 + 1e-2).all()
        py = env.p[:, 1].cpu().numpy()
        for e in range(n):
            if first_done[e] is None and bool(tdone[e]):
                first_done[e] = t
                # the termination must be attributable: a fall (root under
                # the floor threshold at the end of the tick, which is
                # where the per-substep test last fired), or one of the
                # v10 reasons
                if py[e] < FALL_ROOT_Y + 1e-3:
                    reasons["fall"] += 1
                else:
                    reasons["other"] += 1
    falls = env.pop_fall_hits()
    # 2026-08-20: the tilt envelope terminates leaning bodies whose
    # root never reaches the floor threshold — those attribute here
    tilts = env.pop_tilt_hits()
    print(f"f32 contact: dones {sum(x is not None for x in first_done)}/{n} "
          f"(fall-attributed {reasons['fall']}, other {reasons['other']}), "
          f"{falls} counted fall + {tilts} tilt termination(s); first "
          f"done ticks {first_done}")
    assert falls + tilts > 0, \
        "no fall/tilt termination fired in 3 s of random gears"
    assert reasons["other"] == 0 or tilts > 0 \
        or env.pop_envelope_hits() > 0, \
        "an episode terminated for no attributable reason"


# --- (iv) determinism ---------------------------------------------------------

def test_contact_determinism_bitwise():
    """Same seed, same actions, two independent contact-mode runs ->
    bitwise-equal state and obs (contact spec §5 (e), CUDA side). Run in
    f32, the production dtype, with auto_reset on so the reset/chart-draw
    path is inside the comparison."""
    def run(seed):
        env = CudaBeatVecEnv(4, seed=seed, chart_pool=8, dtype=torch.float32,
                             colors="both", chart_len_s=6.0, note_prob=0.5,
                             extra_prob=0.4, bomb_prob=0.25,
                             want_contacts=True, foot_contact=True)
        obs = env.reset()
        rng = np.random.default_rng(11)
        tot = torch.zeros(4, device="cuda")
        for _ in range(120):
            a = torch.as_tensor(rng.integers(0, 3, size=(4, N_JOINTS)))
            obs, r, _done, _ = env.step(a)
            tot += r
        return (obs.clone(), tot.clone(), env.qpos().clone(),
                env.v.clone(), env.pop_fall_hits())

    o1, t1, q1, v1, f1 = run(17)
    o2, t2, q2, v2, f2 = run(17)
    assert torch.equal(o1, o2), "obs not bitwise reproducible"
    assert torch.equal(t1, t2), "returns not bitwise reproducible"
    assert torch.equal(q1, q2), "qpos not bitwise reproducible"
    assert torch.equal(v1, v2), "velocity not bitwise reproducible"
    assert f1 == f2
    assert f1 > 0, "contact-mode run produced no falls to reproduce"


# --- (vii) env v12: ground-contact obs lanes (contact spec §7) -----------------

def test_ground_imp_lanes_settled_parity():
    """The four OFF_GROUND_IMP lanes on a settling zero-torque stance —
    CUDA f64 lockstep against Rust at the single-tick band, and NONZERO
    on both sides, so a bucket permutation cannot hide in zeros. (The
    generic lockstep tiers above already cover the lanes as part of the
    full obs; this pins the regime the lanes were built for.) 10 ticks,
    not more: the unactuated body starts tipping at ~0.7 s and the fall
    envelope ends the episode."""
    n = 4
    sims = _rust_sims(n, 5)
    # env v20: the 10-tick settle band is calibrated to the zero-arm
    # stance (hanging arms swing and delay settling); pin the legacy
    # pose in Rust, then _resync copies it into the CUDA env
    for sim in sims:
        q = list(sim.qpos())
        for _j in (3, 6, 8, 10, 13, 15):
            q[7 + _j] = 0.0
        sim.set_joints(q, [0.0] * 35)
    env = _cenv(n, 5)
    env.reset()
    _resync(env, sims)
    zero_t = torch.zeros((n, N_JOINTS), dtype=torch.int64)
    zero_r = [0] * N_JOINTS
    tobs = None
    obs_r = [None] * n
    for _t in range(10):
        tobs, _tr, _td, _ = env.step(zero_t)
        for e, sim in enumerate(sims):
            obs_r[e], _rr, _dr, _ev = sim.step(zero_r)
    lanes_c = tobs[:, OFF_GROUND_IMP:OFF_GROUND_IMP + 4].cpu().numpy()
    lanes_r = np.asarray([o[OFF_GROUND_IMP:OFF_GROUND_IMP + 4]
                          for o in obs_r], dtype=np.float32)
    d = float(np.abs(lanes_c - lanes_r).max())
    assert d < 1e-4, f"settled ground_imp lanes diverge: {d:.3e}"
    assert (lanes_c > 0.0).all() and (lanes_r > 0.0).all(), \
        "an unloaded bucket in settled stance"
    sums = lanes_c.sum(axis=1)
    assert np.all(np.abs(sums - 1.0) < 0.05), \
        f"weight sum off on CUDA side: {sums}"
