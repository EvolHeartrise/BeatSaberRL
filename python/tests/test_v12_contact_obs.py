"""env v12 (unilateral-contact spec §7): the ground-contact obs lanes and
the survival gradient, on the Rust reference. No CUDA needed here — the
CUDA mirror of both is gated by test_v11_contact_parity.py (the lockstep
tiers compare full 344-lane obs and rewards, and a targeted settled-lane
test runs there as well)."""
import numpy as np

from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import (ALIVE_BONUS, GROUND_IMP_NORM,
                                      N_JOINTS, OBS_DIM, OFF_GROUND_IMP)
import beat_arena_rs

# a note far beyond every horizon: pure body dynamics, no judgment
FAR_NOTE = Note(hit_time=200.0, col=2, row=1, x=0.3, y=1.4, color=1,
                angle=4.712, speed=4.0)
ZERO_ACT = [0] * N_JOINTS  # gear 0 = zero torque on every joint


def _sim(foot_contact):
    sim = beat_arena_rs.RawBeatSim(max_steps=6900, effort_coef=4e-4,
                                   wide_acc=False, gears=5,
                                   foot_contact=foot_contact)
    sim.set_chart([note_tuple(FAR_NOTE)])
    return sim


def test_settled_stance_lanes_carry_body_weight():
    """10 zero-torque ticks from reset settle the two-foot stance; the
    four OFF_GROUND_IMP lanes then sum to ≈ one tick of body weight over
    GROUND_IMP_NORM (≈ 1.0), with every front/back corner pair carrying
    load. NOT more ticks: an unactuated body starts tipping at ~0.7 s
    (heels unload, then the fall envelope ends the episode) — the very
    behavior the v12 lanes exist to let a policy prevent."""
    sim = _sim(True)
    obs = None
    for _ in range(10):
        obs, _r, _d, _ev = sim.step(ZERO_ACT)
    lanes = np.asarray(obs[OFF_GROUND_IMP:OFF_GROUND_IMP + 4],
                       dtype=np.float64)
    # 736 N · (1/30) / 24.5 ≈ 1.001; the §5 (b) gate allows 2% on the
    # settled force, so 5% here is slack, not precision
    assert abs(lanes.sum() - 1.0) < 0.05, f"weight sum off: {lanes}"
    assert (lanes > 0.0).all(), f"unloaded bucket in settled stance: {lanes}"


def test_weld_mode_lanes_stay_zero():
    """foot_contact=False: the ground block does not exist, so the four
    v12 lanes are exactly 0 and lanes 0..340 remain the v10 obs. (The
    v13 body-point lanes after them are NOT zero — positions exist in
    every mode.)"""
    sim = _sim(False)
    obs = None
    for _ in range(5):
        obs, _r, _d, _ev = sim.step(ZERO_ACT)
    assert obs.shape == (OBS_DIM,)
    assert not np.asarray(obs[OFF_GROUND_IMP:OFF_GROUND_IMP + 4]).any()


def test_alive_bonus_priced_per_tick():
    """Zero torque (no effort tax), FAR_NOTE (no judgments): a contact
    tick's reward is exactly ALIVE_BONUS, and the same weld tick is
    exactly 0 — the bonus is contact-mode only."""
    _obs, rew, _d, _ev = _sim(True).step(ZERO_ACT)
    assert rew == ALIVE_BONUS
    _obs, rew, _d, _ev = _sim(False).step(ZERO_ACT)
    assert rew == 0.0


def test_ground_imp_norm_is_the_spec_literal():
    # both engines hardcode the §7 literal; constants.py is the mirror
    # the tools read — pin it so a drift cannot hide
    assert GROUND_IMP_NORM == 24.5
    assert ALIVE_BONUS == 0.002
