"""env v15 (unilateral-contact spec §10): the head-anchored note track,
on the Rust reference. Walking toward the notes must shift them away by
exactly the walked amount — approach distance and miss timing are
invariant to head z. The CUDA mirror is gated by the lockstep parity
tiers (note lanes are part of the full obs; judgment feeds reward)."""
import numpy as np

from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import (DT, N_JOINTS, NOTE_FEAT, OFF_HEAD,
                                      OFF_NOTES, PLANE_Z0, X_NORM, Z_CLIP,
                                      Z_SCALE)
import beat_arena_rs

ZERO_ACT = [0] * N_JOINTS


def _sim(chart, **kw):
    sim = beat_arena_rs.RawBeatSim(max_steps=6900, effort_coef=4e-4,
                                   wide_acc=False, gears=5,
                                   foot_contact=True, **kw)
    sim.set_chart([note_tuple(n) for n in chart])
    return sim


def test_note_obs_z_carries_live_head_offset():
    """Each tick, the note obs z lane must equal the chart's base z
    (hit_time - t)·speed PLUS (head_z - PLANE_Z0), read from the same
    obs vector's head lane. An unactuated contact-mode body starts
    tipping ~0.7 s in, so the offset is exercised at growing magnitude,
    not just at the reference pose where it is ~0."""
    note = Note(hit_time=6.0, col=2, row=1, x=0.3, y=1.4, color=1,
                angle=4.712, speed=4.0)
    sim = _sim([note])
    max_off = 0.0
    for k in range(1, 26):
        obs, _r, done, _ev = sim.step(ZERO_ACT)
        if done:
            break
        t = k * DT
        head_z = float(obs[OFF_HEAD + 2]) * X_NORM + PLANE_Z0
        z_off = head_z - PLANE_Z0
        max_off = max(max_off, abs(z_off))
        base = (note.hit_time - t) * note.speed
        want = np.clip(base + z_off, *Z_CLIP)
        got = float(obs[OFF_NOTES + 2]) * Z_SCALE
        # f32 lanes; Z_SCALE=8 keeps the quantum ~1e-6·8
        assert abs(got - want) < 1e-4, (k, got, want)
    # the invariant must have been tested away from the trivial pose
    assert max_off > 2e-3, max_off


def test_miss_tick_invariant_to_body_motion():
    """The miss plane is z < head_z, which under v15 reduces to
    base z < PLANE_Z0 — a pure function of chart time. A note the body
    cannot touch (x = 2.5) must miss at the same analytic tick whether
    the body stands still or lunges: t > hit_time + 0.65/speed. The
    note is fast and early (speed 8, hit 0.1 s) because honest physics
    is unforgiving — a leaning body terminates within ticks (pre-
    2026-08-20 by falling at tick ~9; now the TILT envelope catches
    the spine gear's pelvis reaction torque even sooner), so the miss
    must land before the lean can kill the episode. The lunge is
    full-gear ankle pitch: the whole body pivots slowly about the
    ankles (head z moves, pelvis stays inside the 30-deg cone for far
    longer than 6 ticks)."""
    note = Note(hit_time=0.1, col=2, row=1, x=2.5, y=1.4, color=1,
                angle=4.712, speed=8.0)
    # miss when (0.1 - t)*8 < -0.65  =>  t > 0.18125  =>  step 6
    want_tick = 6

    lunge = list(ZERO_ACT)
    lunge[21] = 1  # l_ankle_pitch full +
    lunge[27] = 1  # r_ankle_pitch full +
    for acts in (ZERO_ACT, lunge):
        sim = _sim([note])
        miss_tick = None
        for k in range(1, 20):
            _obs, r, done, _ev = sim.step(acts)
            if r < -0.3:  # REWARD_MISS dominates effort/alive at this k
                miss_tick = k
                break
            if done:
                break
        assert miss_tick == want_tick, (acts is lunge, miss_tick)
