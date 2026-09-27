"""env v12 (unilateral-contact spec §7): the ground-contact obs lanes and
the survival gradient, on the Rust reference. No CUDA needed here — the
CUDA mirror of both is gated by test_v11_contact_parity.py (the lockstep
tiers compare full 344-lane obs and rewards, and a targeted settled-lane
test runs there as well)."""
import numpy as np

from beat_arena_env.chart import note_tuple
from beat_arena_env.constants import (ALIVE_BONUS, GROUND_IMP_NORM,
                                      OBS_DIM, OFF_GROUND_IMP)
import beat_arena_rs
from tests._helpers import FAR_NOTE, ZERO_ACT


def _sim():
    sim = beat_arena_rs.RawBeatSim(max_steps=6900,
                                   gears=5)
    sim.set_chart([note_tuple(FAR_NOTE)])
    # env v20: this script is calibrated to the pre-v20 zero-arm pose;
    # pin it (the ready stance only moves the episode START pose, not
    # the mechanics under test)
    q = list(sim.qpos())
    for _j in (3, 6, 8, 10, 13, 15):
        q[7 + _j] = 0.0
    sim.set_joints(q, [0.0] * 35)
    return sim


def test_settled_stance_lanes_carry_body_weight():
    """10 zero-torque ticks from reset settle the two-foot stance; the
    four OFF_GROUND_IMP lanes then sum to ≈ one tick of body weight over
    GROUND_IMP_NORM (≈ 1.0), with every front/back corner pair carrying
    load. NOT more ticks: an unactuated body starts tipping at ~0.7 s
    (heels unload, then the fall envelope ends the episode) — the very
    behavior the v12 lanes exist to let a policy prevent."""
    sim = _sim()
    obs = None
    for _ in range(10):
        obs, _r, _d, _ev = sim.step(ZERO_ACT)
    lanes = np.asarray(obs[OFF_GROUND_IMP:OFF_GROUND_IMP + 4],
                       dtype=np.float64)
    # 736 N · (1/30) / 24.5 ≈ 1.001; the §5 (b) gate allows 2% on the
    # settled force, so 5% here is slack, not precision
    assert abs(lanes.sum() - 1.0) < 0.05, f"weight sum off: {lanes}"
    assert (lanes > 0.0).all(), f"unloaded bucket in settled stance: {lanes}"


def test_alive_bonus_priced_per_tick():
    """Zero torque (no effort tax), FAR_NOTE (no judgments): a tick's
    reward is exactly ALIVE_BONUS."""
    _obs, rew, _d, _ev = _sim().step(ZERO_ACT)
    assert rew == ALIVE_BONUS


def test_ground_imp_norm_is_the_spec_literal():
    # both engines hardcode the §7 literal; constants.py is the mirror
    # the tools read — pin it so a drift cannot hide
    assert GROUND_IMP_NORM == 24.5
    assert ALIVE_BONUS == 0.002
