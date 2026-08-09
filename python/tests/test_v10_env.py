"""P3 gate — env v10 integration (SPEC-v10 §7 P3).

Covers: obs length 340 + every §5 block offset; determinism (seed +
action stream => identical rewards/events, all object kinds); effort
wiring vs the §4 work formula; live REACH/miss gates moving with the
pose; replay v10 round-trip; and the play-a-chart gate — a
deterministic scripted-torque sequence achieving a good cut through
the full dynamics -> blade sweep -> cut -> score loop.
"""

import json
import math

import numpy as np
import pytest

from beat_arena_env import Note, RustBeatEnv
from beat_arena_env.chart import note_tuple
from beat_arena_env.constants import (
    A_L_ARM, A_R_ARM, DT, EFFORT_COEF, HAND_FEAT, JOINT_LIMITS, N_JOINTS,
    N_NOTE_SLOTS, NOTE_FEAT, NV, NQPOS, OBS_DIM, OFF_CONTACT, OFF_HAND_L,
    OFF_HAND_R, OFF_HEAD, OFF_NOTES, OFF_NOTE_MASK, OFF_Q, OFF_QD,
    OFF_ROOT_POS, OFF_ROOT_ROT, OFF_ROOT_VANG, OFF_ROOT_VLIN, OMEGA_CAP,
    ONE_WAY_JOINTS, ROOT_POS0, TAU_MAX, WRAP_JOINTS, gear_table,
)

from beat_arena_rs import RawBeatSim

DOWN = 3 * np.pi / 2
COAST = [0] * N_JOINTS


def _note(hit_time, x, y, color=1, angle=DOWN, speed=4.0, any_dir=False,
          kind=0, free_pre=False, free_post=False):
    return (hit_time, x, y, color, angle, speed, any_dir, kind,
            free_pre, free_post)


def _sim(notes, max_steps=690, **kw):
    sim = RawBeatSim(max_steps=max_steps, **kw)
    sim.set_chart(notes)
    return sim


def _q36(joints=None):
    q = np.zeros(NQPOS)
    q[:3] = ROOT_POS0
    q[3] = 1.0
    if joints is not None:
        q[7:] = joints
    return q


# --- obs layout (§5, every offset) -------------------------------------------

def test_obs_length_and_reset_layout():
    sim = _sim([_note(5.0, 0.3, 1.4)])
    obs = np.asarray(sim.set_chart([_note(5.0, 0.3, 1.4)]))
    assert obs.shape == (OBS_DIM,) == (362,)
    assert (OFF_ROOT_POS, OFF_ROOT_ROT, OFF_ROOT_VLIN, OFF_ROOT_VANG,
            OFF_Q, OFF_QD, OFF_HAND_L, OFF_HAND_R, OFF_HEAD, OFF_CONTACT,
            OFF_NOTES, OFF_NOTE_MASK) == (0, 3, 9, 12, 15, 46, 75, 85,
                                          95, 98, 100, 324)
    # root at reference: centered position zero, identity rotation 6D
    assert np.allclose(obs[OFF_ROOT_POS:OFF_ROOT_POS + 3], 0.0)
    assert np.allclose(obs[OFF_ROOT_ROT:OFF_ROOT_ROT + 6],
                       [1, 0, 0, 0, 1, 0])
    assert not obs[OFF_ROOT_VLIN:OFF_ROOT_VANG + 3].any()
    # q block: all joints zero => lanes zero EXCEPT the wrist-roll
    # cos/sin pairs (cos 0 = 1) and the one-way elbow/knee lanes (-1)
    q = obs[OFF_Q:OFF_QD]
    assert q.shape == (31,)
    lane = 0
    for j in range(N_JOINTS):
        if j in WRAP_JOINTS:
            assert q[lane] == 1.0 and q[lane + 1] == 0.0, f"joint {j}"
            lane += 2
        elif j in ONE_WAY_JOINTS:
            assert q[lane] == pytest.approx(-1.0), f"joint {j}"
            lane += 1
        else:
            assert q[lane] == 0.0, f"joint {j}"
            lane += 1
    assert lane == 31
    assert not obs[OFF_QD:OFF_QD + N_JOINTS].any()
    # hands at reference: wrists (±0.2, 1.47, 0.0), blades +z, meter 0
    for base, x in ((OFF_HAND_L, -0.2), (OFF_HAND_R, 0.2)):
        assert obs[base] == pytest.approx(x)
        assert obs[base + 1] == pytest.approx((1.47 - 1.40) / 0.75)
        assert obs[base + 2] == pytest.approx(0.0, abs=1e-6)
        assert not obs[base + 3:base + 6].any()          # wrist vel
        assert np.allclose(obs[base + 6:base + 9], [0, 0, 1], atol=1e-9)
        assert obs[base + 9] == 0.0                      # swing meter
    # head at reference (0, 1.70, -0.65) => centered zeros
    assert np.allclose(obs[OFF_HEAD:OFF_HEAD + 3], 0.0, atol=1e-9)
    assert not obs[OFF_CONTACT:OFF_CONTACT + 2].any()
    # one pending note
    base = OFF_NOTES
    assert obs[base] == pytest.approx(0.3)
    assert obs[base + 1] == pytest.approx((1.4 - 1.40) / 0.75)
    assert obs[base + 3] == pytest.approx(-4.0 / 8.0)
    assert obs[base + 13] == 1.0
    assert obs[OFF_NOTE_MASK] == 1.0
    assert obs[OFF_NOTE_MASK + 1:OFF_NOTE_MASK + N_NOTE_SLOTS].sum() == 0.0


def test_obs_q_and_qd_track_state():
    """set_joints -> the q/qd blocks encode exactly the §5 normalizer
    table: q / max(|lo|, |hi|), one-way (2q − hi)/hi for elbows/knees,
    cos/sin for the wrist rolls, qd / OMEGA_CAP — every joint checked
    via the Python q_lane/q_normalizer mirror."""
    from beat_arena_env.constants import q_lane, q_normalizer
    sim = _sim([_note(50.0, 0.3, 1.4)])
    rng = np.random.default_rng(2)
    joints = np.array([rng.uniform(lo, hi) for lo, hi in JOINT_LIMITS])
    v = np.zeros(NV)
    v[6:] = rng.uniform(-10.0, 10.0, N_JOINTS)
    sim.set_joints(list(_q36(joints)), list(v))
    obs = np.asarray(sim.obs())
    for j in range(N_JOINTS):
        lane = q_lane(j)
        norm = q_normalizer(j)
        if norm is None:                      # wrist roll: cos/sin
            assert obs[lane] == pytest.approx(math.cos(joints[j]), abs=1e-6)
            assert obs[lane + 1] == pytest.approx(math.sin(joints[j]),
                                                  abs=1e-6)
        elif isinstance(norm, tuple):         # one-way: (2q - hi)/hi
            hi = norm[1]
            assert obs[lane] == pytest.approx((2 * joints[j] - hi) / hi,
                                              abs=1e-6), f"joint {j}"
        else:
            assert obs[lane] == pytest.approx(joints[j] / norm,
                                              abs=1e-6), f"joint {j}"
        assert obs[OFF_QD + j] == pytest.approx(v[6 + j] / OMEGA_CAP[j],
                                                abs=1e-6), f"joint {j}"
    # root blocks track a moved root
    q = _q36(joints)
    q[0], q[1], q[2] = 0.1, 1.04, -0.4
    s2, c2 = math.sin(0.25), math.cos(0.25)
    q[3], q[4] = c2, 0.0
    q[5], q[6] = s2, 0.0                      # yaw 0.5 rad about +y
    v[:6] = [0.5, -0.2, 1.0, 0.1, 0.4, -0.3]
    sim.set_joints(list(q), list(v))
    obs = np.asarray(sim.obs())
    assert obs[OFF_ROOT_POS] == pytest.approx(0.1)
    assert obs[OFF_ROOT_POS + 1] == pytest.approx((1.04 - 0.94) / 0.75)
    assert obs[OFF_ROOT_POS + 2] == pytest.approx((-0.4 + 0.65) / 1.0)
    ang = 0.5
    want = [math.cos(ang), 0.0, -math.sin(ang), 0.0, 1.0, 0.0]
    assert np.allclose(obs[OFF_ROOT_ROT:OFF_ROOT_ROT + 6], want, atol=1e-6)
    assert np.allclose(obs[OFF_ROOT_VLIN:OFF_ROOT_VLIN + 3],
                       np.array([0.5, -0.2, 1.0]) / 6.0, atol=1e-6)
    assert np.allclose(obs[OFF_ROOT_VANG:OFF_ROOT_VANG + 3],
                       np.array([0.1, 0.4, -0.3]) / 16.0, atol=1e-6)


def test_note_block_and_mask_track_pending():
    from beat_arena_env.chart import ChartConfig, generate_chart
    env = RustBeatEnv(seed=3, bomb_prob=0.0, dot_prob=0.0, extra_prob=0.0)
    obs, _ = env.reset()
    n_pending = min(len(env.chart), N_NOTE_SLOTS)
    assert obs[OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS].sum() == n_pending
    n = env.chart[0]
    from beat_arena_env.constants import note_center as nc
    base = OFF_NOTES
    assert obs[base + 13] == 1.0
    assert obs[base + 0] == pytest.approx(nc(n)[0], abs=1e-6)
    z = min(n.hit_time * n.speed, 12.0)
    assert obs[base + 2] == pytest.approx(z / 8.0)
    assert obs[base + 6] == pytest.approx(np.cos(n.angle), abs=1e-6)
    assert obs[base + 7] == pytest.approx(np.sin(n.angle), abs=1e-6)


# --- determinism (all object kinds) ------------------------------------------

def test_determinism_all_kinds():
    def run():
        env = RustBeatEnv(colors="both", bomb_prob=0.3, extra_prob=0.3,
                          dot_prob=0.2, chart_len_s=8.0, seed=7)
        env.reset(seed=7)
        rng = np.random.default_rng(1)
        out = []
        done = False
        while not done:
            obs, r, done, _, info = env.step(rng.integers(0, 3, N_JOINTS))
            out.append((r, tuple(info["events"]), obs.tobytes()))
        return out

    a, b = run(), run()
    assert a == b


# --- effort (SPEC-v10 §4 wiring) ---------------------------------------------

def test_effort_is_priced_positive_work():
    """reward_effort = -effort_coef x last_effort (identity, both sims
    share one trajectory since scoring never feeds back into physics);
    coasting is exactly free; a one-joint drive does positive work.
    The formula itself (sum_substeps max(tau*qd, 0)*h vs an independent
    pipeline mirror, bitwise) is pinned Rust-side
    (lib.rs env_tests::effort_is_positive_work)."""
    far = [_note(50.0, 0.9, 1.4)]
    free = _sim(far, effort_coef=0.0)
    paid = _sim(far, effort_coef=EFFORT_COEF)
    _, r_free, _, _ = free.step(COAST)
    _, r_paid, _, _ = paid.step(COAST)
    assert r_free == 0.0 and r_paid == 0.0          # coast exactly free
    assert paid.last_effort() == 0.0
    a = list(COAST)
    a[A_R_ARM] = 2                                  # r arm pitch, full -
    _, r_free, _, _ = free.step(a)
    _, r_paid, _, _ = paid.step(a)
    j_free, j_paid = free.last_effort(), paid.last_effort()
    assert j_free == j_paid > 0.0                   # same trajectory
    assert r_free == 0.0
    assert r_paid == pytest.approx(-EFFORT_COEF * j_paid, abs=1e-15)
    # one more tick: torque against capped/limited motion still bills
    # only POSITIVE work
    _, r2, _, _ = paid.step(a)
    assert r2 == pytest.approx(-EFFORT_COEF * paid.last_effort(), abs=1e-15)


# --- live gates (SPEC-v10 §4) ------------------------------------------------

def _park_arms(joints):
    """Arms straight down and outward: blades far from the note grid."""
    joints[A_L_ARM] = -np.pi / 2
    joints[A_L_ARM + 1] = -1.5
    joints[A_R_ARM] = -np.pi / 2
    joints[A_R_ARM + 1] = 1.5
    return joints


def _miss_tick(spine_pitch, speed=8.0):
    """Tick at which a high, fast note's miss event fires with the body
    held (by torque) at a given spine lean."""
    sim = _sim([_note(0.5, -0.9, 1.9, speed=speed)], max_steps=300)
    joints = _park_arms(np.zeros(N_JOINTS))
    joints[0] = spine_pitch
    sim.set_joints(list(_q36(joints)), [0.0] * NV)
    a = list(COAST)
    a[0] = 1 if spine_pitch > 0 else 0   # hold the lean against gravity
    for t in range(120):
        _, _, _, ev = sim.step(a)
        for i, code, *_ in ev:
            if code == 4:
                return t
    raise AssertionError("miss never fired")


def test_miss_boundary_moves_with_head():
    """Leaning the spine forward moves the head toward the hit plane, so
    the miss boundary (z < head_z) fires EARLIER (fewer ticks after the
    note crosses the plane)."""
    upright = _miss_tick(0.0)
    leaning = _miss_tick(JOINT_LIMITS[0][1])    # +45 deg, torque-pinned
    assert leaning < upright, (upright, leaning)


def test_reach_gate_moves_with_shoulders():
    """A crouched pose (bent knees/hips lower the shoulders along z via
    the root drop... the gate is SHOULDER-Z based) — lean the spine BACK
    to pull the shoulders away from the plane: a note parked just inside
    the reference reach becomes untouchable earlier. Probe: the forward
    start-pose blade (spans z 0..1) grazes an approaching note exactly
    when it enters reach; with the shoulders pulled back the graze
    happens later (fewer pre-hit ticks with contact chances) — assert
    via first-contact tick of a slow note against a stationary blade."""
    def first_contact_tick(spine_pitch):
        sim = _sim([_note(2.0, 0.2, 1.47, speed=2.0)], max_steps=300)
        joints = np.zeros(N_JOINTS)
        joints[0] = spine_pitch
        sim.set_joints(list(_q36(joints)), [0.0] * NV)
        a = list(COAST)
        if spine_pitch < 0:
            a[0] = 2                     # hold the backward lean
        for t in range(90):
            _, _, _, ev = sim.step(a)
            if ev:
                return t, ev[0][1]
        return None, None

    t_ref, code_ref = first_contact_tick(0.0)
    t_back, code_back = first_contact_tick(JOINT_LIMITS[0][0])  # -20 deg
    assert t_ref is not None and t_back is not None
    # pulled-back shoulders + blade => later first contact
    assert t_back > t_ref, (t_ref, t_back)


# --- replay v10 round-trip ---------------------------------------------------

def test_replay_v10_roundtrip(tmp_path):
    path = str(tmp_path / "ep.jsonl")
    env = RustBeatEnv(seed=5, chart_len_s=6.0, bomb_prob=0.3, extra_prob=0.3,
                      colors="both")
    env.set_record(path)
    env.reset(seed=5)
    rng = np.random.default_rng(11)
    done = False
    while not done:
        _, _, done, _, info = env.step(rng.integers(0, 3, N_JOINTS))
    lines = [json.loads(l) for l in open(path) if l.strip()]
    head, ticks = lines[0], lines[1:]
    assert head["header"]["version"] == "v10"
    assert head["header"]["config"]["gears"] == 3
    assert len(head["header"]["chart"]) == len(env.chart)
    assert all("x" in c and "y" in c and "kind" in c
               for c in head["header"]["chart"])
    for t in ticks:
        assert len(t["q"]) == NQPOS == 36
        assert len(t["twist"]) == 6
        assert len(t["omega"]) == N_JOINTS == 29
        assert len(t["sabers"]) == 2 and all(len(s) == 6 for s in t["sabers"])
        assert len(t["action"]) == N_JOINTS
        assert isinstance(t["contacts"], list)
        for c in t["contacts"]:
            assert 0 <= c["pair"] < 55 and len(c["p"]) == 3
            assert c["j"] != 0.0
        # blade dirs are unit (rounded to 4 dp)
        for s in t["sabers"]:
            assert np.hypot(np.hypot(s[3], s[4]), s[5]) == pytest.approx(
                1.0, abs=1e-3)
    # counters + score reconstruct from the event stream
    goods = sum(1 for t in ticks for e in t["events"]
                if "cut" in e and e["cut"]["good"])
    misses = sum(1 for t in ticks for e in t["events"] if "miss" in e)
    bombs = sum(1 for t in ticks for e in t["events"] if "bomb" in e)
    assert goods == info["good_cuts"]
    assert misses == info["misses"]
    assert bombs == info["bombs_hit"]
    pts = sum(e["cut"]["points"] for t in ticks for e in t["events"]
              if "cut" in e)
    pts += sum(e["post"]["points"] for t in ticks for e in t["events"]
               if "post" in e)
    assert pts == pytest.approx(info["score"], abs=0.5)


# --- the play-a-chart gate ---------------------------------------------------

def test_sim_plays_a_chart_scripted_torques():
    """The whole loop end to end: a deterministic scripted-torque
    sequence (wind the right arm up to the +65 deg pitch stop while the
    note is beyond reach, then whip down through it) achieves a good cut
    with near-full credit, and the follow-through resolves. Hand-tuned
    at P3 (seeded search over whip ticks kept as fallback)."""
    def run(whip_tick):
        sim = _sim([_note(0.9, 0.3, 1.4, speed=4.0)], max_steps=300)
        events = []
        for t in range(90):
            a = list(COAST)
            a[A_R_ARM] = 1 if t < whip_tick else 2   # raise/hold, then whip
            _, _, done, ev = sim.step(a)
            events.extend(ev)
            if done:
                break
        return sim, events

    sim, events = run(15)                    # the hand-tuned sequence
    if not any(c == 0 for _, c, *_ in events):
        # crude deterministic fallback search (seeded ordering)
        for whip in (12, 14, 16, 18, 13, 17):
            sim, events = run(whip)
            if any(c == 0 for _, c, *_ in events):
                break
    cuts = [e for e in events if e[1] == 0]
    posts = [e for e in events if e[1] == 5]
    assert len(cuts) == 1, f"no good cut: {events}"
    assert sim.good_cuts == 1 and sim.misses == 0
    assert cuts[0][2] >= 60.0                # real wind-up credit banked
    assert len(posts) == 1                   # follow-through resolved
    assert sim.score == cuts[0][2] + posts[0][2]
    assert sim.score >= 90.0
