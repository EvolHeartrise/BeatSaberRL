"""Wrist-reset telemetry (2026-10-04, user spec), end to end on the Rust
sim with REAL cuts, plus the CUDA twin (GPU).

Definition (lib.rs WristReset, the kernel mirror): per hand, every good
note cut (notes, dots, chain heads; chain links are not hits) is
classified by the sign of that hand's wrist-BEND angular velocity (joint
8 / 15) at the cut, or the detector's committed direction when |qd| <
1 rad/s. A reversal of the bend motion is a 30 deg turn-back from the
running extreme (angle-travel hysteresis, its own WRESET_HYST; tremor
below it is no reversal). A hit with the previous swing's sign and a
reversal since the last hit = one RESET (and a new swing); the opposite
sign = alternation, a new swing; the same sign without a reversal = the
same swing (stacks / sliders cut in one motion). Bad cuts and misses
neither break nor continue a swing. The scripted state-machine
sequences (alternation, wind-back, stack / slider, tremor, a long
break with many reversals, per-hand independence, near-zero velocity)
are Rust unit tests (lib.rs env_tests::wrist_reset_*); here the sim
drives real swings through real notes.

Driver: each tick the right arm is posed (set_joints) straight forward
with the bend at a scripted angle and velocity, coasting (a full gear
covers ~1.5 rad a tick — through every note on the arc); on non-cut
ticks the arm is yawed out so the blade sweeps nowhere near a note. A dry run on a twin sim (same poses
and actions, a far note) gives each cut tick's blade sweep; the real
run places a dot note at the sweep's middle at that tick's time.
"""
import numpy as np
import pytest

import beat_arena_rs
from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import N_JOINTS, PLANE_Z0
from tests._helpers import FAR_NOTE, cuda, rust_state

R_ROLL, R_BEND, R_YAW = 14, 15, 11
SPEED = 2.0
DT = 1.0 / 30.0


def _pose(stance, phi, omega, cut):
    q, v = list(stance), [0.0] * 35
    for j in (10, 11, 12, 13, 16):          # right arm straight forward
        q[7 + j] = 0.0
    q[7 + R_ROLL] = -0.05                   # palm down, off the roll stop
    q[7 + R_YAW] = 0.0 if cut else 1.4      # yawed out: blade far away
    q[7 + R_BEND] = phi
    v[6 + R_BEND] = omega
    return q, v


def _play(plan, notes):
    """Run `plan` = [(phi, omega, cut)] on a sim with `notes`; returns
    (sim, per-tick [(sabers before, sabers after, head z, time)])."""
    sim = beat_arena_rs.RawBeatSim(max_steps=2000, gears=5)
    sim.set_chart([note_tuple(n) for n in notes] or [note_tuple(FAR_NOTE)])
    stance, _ = rust_state(sim)
    rec = []
    for k, (phi, omega, cut) in enumerate(plan):
        q, v = _pose(stance, phi, omega, cut)
        sim.set_joints(q, v)
        before = np.asarray(sim.sabers(), dtype=np.float64)
        sim.step([0] * N_JOINTS)            # coast: the set velocity carries
        rec.append((before, np.asarray(sim.sabers(), dtype=np.float64),
                    sim.head_pose()[2], (k + 1) * DT))
    return sim, rec


def _notes_for(plan, radii=(0.6,)):
    """Dry run, then one right-hand dot per cut tick (and radius) at the
    middle of that tick's blade sweep."""
    _, rec = _play(plan, [])
    notes = []
    for (phi, omega, cut), (b, a, head_z, t) in zip(plan, rec):
        if not cut:
            continue
        for r in radii:
            p0 = b[1, :3] + r * b[1, 3:]
            p1 = a[1, :3] + r * a[1, 3:]
            c = 0.5 * (p0 + p1)
            z_off = head_z - PLANE_Z0
            notes.append(Note(hit_time=t + (c[2] - z_off) / SPEED, col=0,
                              row=0, color=1, angle=0.0, speed=SPEED,
                              any_dir=True, x=float(c[0]), y=float(c[1])))
    return sorted(notes, key=lambda n: n.hit_time)


DOWN, UP = -20.0, 20.0
HOLD = [(0.9, 0.0, False)] * 3              # arm the detector at +0.9


def park(phi, n=30):
    """n still ticks at bend phi, arm yawed out — ~1 s, so the next
    cut's note is ~2 m away (out of reach) while the earlier cut sweeps
    (a dot's good box is 0.8 m tall). Holding still moves no detector."""
    return [(phi, 0.0, False)] * n


def _run(plan, radii=(0.6,)):
    notes = _notes_for(plan, radii)
    sim, _ = _play(plan, notes)
    return sim, len(notes)


def test_alternation_is_no_reset():
    # down cut (+0.9 -> ~+0.2), park low, up cut (-0.9 -> ~-0.2)
    plan = HOLD + [(0.9, DOWN, True)] + park(-0.9) + [(-0.9, UP, True)]
    sim, n = _run(plan)
    assert sim.good_cuts == n == 2
    assert sim.wrist_resets() == (0, 0, 0, 2)


def test_windback_past_30_is_a_reset():
    # down cut (ends ~+0.16), wind back to +0.9 (~42 deg) without a hit,
    # down cut again
    plan = (HOLD + [(0.9, DOWN, True)]
            + [(0.25, 10.0, False), (0.57, 10.0, False)]
            + park(0.9) + [(0.9, DOWN, True)])
    sim, n = _run(plan)
    assert sim.good_cuts == n == 2
    assert sim.wrist_resets() == (0, 1, 0, 2)    # left hand untouched


def test_windback_under_30_is_the_same_swing():
    # a wind-back under the hysteresis (+0.16 -> +0.6, ~25 deg): no
    # reversal, one swing
    plan = (HOLD + [(0.9, DOWN, True)]
            + [(0.25, 5.0, False), (0.42, 5.0, False)]
            + park(0.6) + [(0.6, DOWN, True)])
    sim, n = _run(plan)
    assert sim.good_cuts == n == 2
    assert sim.wrist_resets() == (0, 0, 0, 1)


def test_stack_in_one_motion_is_one_swing():
    # two dots stacked along the blade, cut by the same down sweep
    plan = HOLD + [(0.9, DOWN, True)]
    sim, n = _run(plan, radii=(0.4, 0.9))
    assert sim.good_cuts == n == 2
    assert sim.wrist_resets() == (0, 0, 0, 1)


def test_vec_sim_getter_matches():
    sim = beat_arena_rs.VecBeatSim(2, max_steps=300, gears=5,
                                   return_events=False)
    assert sim.wrist_resets(0) == (0, 0, 0, 0)


def test_random_play_invariants():
    """Free random play on generated charts: per hand resets <= swings
    <= good note cuts, and any good note cut makes a swing (coverage of
    real swings: test_production_lockstep)."""
    from beat_arena_env.chart import ChartConfig, generate_chart
    rng = np.random.default_rng(4)
    tot = np.zeros(4)
    for seed in range(6):
        ch = generate_chart(ChartConfig(chart_len_s=4.0, colors="both",
                                        dot_prob=0.3),
                            np.random.default_rng(seed))
        sim = beat_arena_rs.RawBeatSim(max_steps=300, gears=5)
        sim.set_chart([note_tuple(n) for n in ch])
        for _ in range(200):
            _, _, done, _ = sim.step([int(a) for a in
                                      rng.integers(0, 5, N_JOINTS)])
            if done:
                break
        rl, rr, sl, sr = sim.wrist_resets()
        assert rl <= sl and rr <= sr
        assert sl + sr <= sim.good_cuts
        assert (sl + sr > 0) == (sim.good_cuts > 0)
        tot += (rl, rr, sl, sr)


@cuda
def test_cuda_pop_and_episode_record():
    """The kernel's per-update accumulator equals its per-episode counts
    (no reset in between), and pop_wrist_resets zeroes it. Rust parity
    is pinned tick by tick in test_production_lockstep."""
    import torch
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    env = CudaBeatVecEnv(256, chart_len_s=6.0, seed=5, device="cuda",
                         dtype=torch.float64, gears=5, auto_reset=False)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(150):
        env.step(torch.as_tensor(rng.integers(0, 5, (256, N_JOINTS)),
                                 device="cuda"))
    ep = env.wrs[:, :, 3:5].sum(0).cpu().numpy()   # [[rL, sL], [rR, sR]]
    rl, rr, sl, sr = env.pop_wrist_resets()
    assert (rl, rr, sl, sr) == (ep[0, 0], ep[1, 0], ep[0, 1], ep[1, 1])
    # exp/bend7: random play on this body may cut nothing in 150 ticks
    # (real-swing coverage: the production lockstep asserts swings > 0)
    assert (sl + sr > 0) == bool(env.good.sum() > 0)
    assert rl <= sl and rr <= sr
    assert env.pop_wrist_resets() == (0, 0, 0, 0)
