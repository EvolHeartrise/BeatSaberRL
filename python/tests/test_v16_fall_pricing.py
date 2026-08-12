"""env v16 (unilateral-contact spec §11): gamma-discounted fall pricing,
on the Rust reference. A fall charges each pending scoreable at
fall_gamma^(ticks until its v15 miss boundary) — return-equivalent to
standing-and-missing under the agent's own discount, so the v11.1
lump sum's manufactured risk premium on reaching is gone. fall_gamma
defaults to 1.0 = the v11.1 count, bitwise (gated by the pre-existing
pricing test in rust tests_dyn). The CUDA mirror is gated by the §5
lockstep tiers (reward parity)."""
import math

from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import DT, N_JOINTS
import beat_arena_rs

ZERO_ACT = [0] * N_JOINTS
PLANE_Z0 = -0.65

# far off to the side (x = 2.5): unhittable, unjudged before the fall
NOTES = [Note(hit_time=5.0, col=2, row=1, x=2.5, y=1.4, color=1,
              angle=4.712, speed=4.0),
         Note(hit_time=100.0, col=2, row=1, x=2.5, y=1.4, color=0,
              angle=4.712, speed=4.0)]


def _fall(fall_gamma):
    """Unactuated contact body tips and falls ~1 s in; returns
    (fall_tick, fall_reward)."""
    sim = beat_arena_rs.RawBeatSim(max_steps=6900, effort_coef=4e-4,
                                   wide_acc=False, gears=5,
                                   foot_contact=True,
                                   fall_gamma=fall_gamma)
    sim.set_chart([note_tuple(n) for n in NOTES])
    for k in range(1, 120):
        _obs, r, done, _ev = sim.step(ZERO_ACT)
        if done:
            # max_steps is 6900 and the chart runs to 100 s, so the only
            # terminator inside 120 ticks is the fall envelope
            return k, r
    raise AssertionError("no fall within 4 s")


def _expected(fall_gamma, fall_tick):
    """The spec §11 formula with the engines' exact float ops."""
    t_now = fall_tick * DT
    charge = 0.0
    for n in NOTES:
        t_miss = n.hit_time + (-PLANE_Z0) / n.speed
        dt = max(t_miss - t_now, 0.0)
        k = min(math.ceil(dt / DT), 600)
        w = 1.0
        for _ in range(k):
            w *= fall_gamma
        charge += w
    return -0.5 * charge


def test_discounted_fall_charge_matches_formula():
    tick, r = _fall(0.965)
    want = _expected(0.965, tick)
    assert abs(r - want) < 1e-12, (tick, r, want)
    # sanity on magnitudes: the 5 s note is ~4 s of discount away
    # (0.965^~120 ~ 0.014), the 100 s note is capped dust — so the
    # charge is a small fraction of one miss, not the v11.1 -1.0
    assert -0.05 < r < 0.0, r


def test_gamma_one_is_the_v11_lump_sum():
    tick, r = _fall(1.0)
    assert r == -0.5 * len(NOTES), (tick, r)
