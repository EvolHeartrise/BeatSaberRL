"""P3 gate — env v10 integration (SPEC-v10 §7 P3).

Covers: obs length 340 + every §5 block offset; determinism (seed +
action stream => identical rewards/events, all object kinds); (the §4
effort-wiring case retired with the work reading 2026-09-13; activation
telemetry is covered in test_v10_bot.py); live REACH/miss gates moving with the
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
    A_L_ARM, A_R_ARM, DT, HAND_FEAT, JOINT_LIMITS, N_JOINTS,
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

def _reset_drop():
    """The reset's root drop (env v21): the wide leg stance lifts the
    ankles, and the root is lowered until the lowest sole corner sits
    exactly on the ground — fk_body mirror of the same corner arithmetic."""
    from beat_arena_env.constants import q_start
    from beat_arena_env.kinematics import F_L_FOOT, F_R_FOOT, fk_body
    corners = ((-0.045, -0.070, -0.075), (-0.045, -0.070, 0.175),
               (0.045, -0.070, -0.075), (0.045, -0.070, 0.175))
    rots, pos = fk_body([0.0, 0.94, -0.65, 1.0, 0.0, 0.0, 0.0] + q_start())
    return min(pos[fr][1] + sum(rots[fr][1][k] * c[k] for k in range(3))
               for fr in (F_L_FOOT, F_R_FOOT) for c in corners)


def test_obs_length_and_reset_layout():
    sim = _sim([_note(5.0, 0.3, 1.4)])
    obs = np.asarray(sim.set_chart([_note(5.0, 0.3, 1.4)]))
    drop = _reset_drop()
    assert 0.0 < drop < 0.02         # ~8.5 mm
    assert obs.shape == (OBS_DIM,) == (630,)   # env v49: blade-tip velocity lanes
    assert (OFF_ROOT_POS, OFF_ROOT_ROT, OFF_ROOT_VLIN, OFF_ROOT_VANG,
            OFF_Q, OFF_QD, OFF_HAND_L, OFF_HAND_R, OFF_HEAD, OFF_CONTACT,
            OFF_NOTES, OFF_NOTE_MASK) == (0, 3, 9, 12, 15, 46, 75, 86,
                                          97, 100, 102, 534)
    # root at reference less the stance drop, identity rotation 6D
    assert np.allclose(obs[OFF_ROOT_POS:OFF_ROOT_POS + 3],
                       [0.0, -drop / 0.75, 0.0], atol=1e-6)
    assert np.allclose(obs[OFF_ROOT_ROT:OFF_ROOT_ROT + 6],
                       [1, 0, 0, 0, 1, 0])
    assert not obs[OFF_ROOT_VLIN:OFF_ROOT_VANG + 3].any()
    # q block (env v20): every joint at its q_start() value — the arm
    # ready stance is nonzero (pitch/elbow/bend), everything else zero.
    # Expectations derive from the §5 normalizer table, not literals.
    import math as _m
    from beat_arena_env.constants import JOINT_LIMITS, q_start
    q0 = q_start()
    q = obs[OFF_Q:OFF_QD]
    assert q.shape == (31,)
    lane = 0
    for j in range(N_JOINTS):
        if j in WRAP_JOINTS:
            assert q[lane] == pytest.approx(_m.cos(q0[j])), f"joint {j}"
            assert q[lane + 1] == pytest.approx(_m.sin(q0[j])), f"joint {j}"
            lane += 2
        elif j in ONE_WAY_JOINTS:
            hi = JOINT_LIMITS[j][1]
            assert q[lane] == pytest.approx((2 * q0[j] - hi) / hi), \
                f"joint {j}"
            lane += 1
        else:
            lo, hi = JOINT_LIMITS[j]
            assert q[lane] == pytest.approx(
                q0[j] / max(abs(lo), abs(hi))), f"joint {j}"
            lane += 1
    assert lane == 31
    assert not obs[OFF_QD:OFF_QD + N_JOINTS].any()
    # hands at the v20 ready stance: wrist positions and blade dirs
    # from the fk_body python mirror of the same start pose
    from beat_arena_env.kinematics import (F_L_HAND, F_L_WRIST, F_R_HAND,
                                           F_R_WRIST, fk_body)
    rots, pos = fk_body([0.0, 0.94 - drop, -0.65, 1.0, 0.0, 0.0, 0.0] + q0)
    for base, fw, fh in ((OFF_HAND_L, F_L_WRIST, F_L_HAND),
                         (OFF_HAND_R, F_R_WRIST, F_R_HAND)):
        w, r = pos[fw], rots[fh]
        # env v38: wrist x/y head-relative (live head (0, 1.70 - drop))
        assert obs[base] == pytest.approx(w[0] - 0.0, abs=1e-6)
        assert obs[base + 1] == pytest.approx((w[1] - (1.70 - drop)) / 0.75,
                                              abs=1e-6)
        assert obs[base + 2] == pytest.approx(w[2], abs=1e-6)
        assert not obs[base + 3:base + 6].any()          # wrist vel
        assert np.allclose(obs[base + 6:base + 9],
                           [r[0][2], r[1][2], r[2][2]], atol=1e-6)
        assert obs[base + 9] == 0.0                      # swing meter
    # head at reference (0, 1.70, -0.65) less the drop
    assert np.allclose(obs[OFF_HEAD:OFF_HEAD + 3], [0.0, -drop / 0.75, 0.0],
                       atol=1e-6)
    assert not obs[OFF_CONTACT:OFF_CONTACT + 2].any()
    # one pending note
    base = OFF_NOTES
    # env v38: entity x/y are head-relative (live head (0, 1.70 - drop))
    assert obs[base] == pytest.approx(0.3 - 0.0)
    assert obs[base + 1] == pytest.approx((1.4 - (1.70 - drop)) / 0.75)
    assert obs[base + 3] == pytest.approx(-4.0 / 8.0)
    assert obs[base + 13] == 0.0        # ex-valid lane, retired 2026-08-22
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
    # env v17: slots are DISTANCE-sorted (nearest first), not
    # hit_time-sorted, so the soonest note is somewhere in the window
    # (procedural charts have no walls to displace it) — find it by its
    # x/z signature and pin the lane contents there.
    n = env.chart[0]
    from beat_arena_env.constants import note_center as nc
    from tests._helpers import z_lane
    z = n.hit_time * n.speed
    hit_slots = [s for s in range(N_NOTE_SLOTS)
                 if obs[OFF_NOTE_MASK + s] == 1.0
                 and abs(obs[OFF_NOTES + s * NOTE_FEAT]
                         - (nc(n)[0] - obs[OFF_HEAD])) < 1e-6
                 and abs(obs[OFF_NOTES + s * NOTE_FEAT + 2] - z_lane(z)) < 1e-6]
    assert hit_slots, "soonest note missing from the entity window"
    base = OFF_NOTES + hit_slots[0] * NOTE_FEAT
    assert obs[base + 13] == 0.0        # ex-valid lane, retired 2026-08-22
    assert obs[base + 14] == 0.0                    # not a wall
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
    started at a given spine lean (reference stance otherwise).

    2026-09-27 (weld harness removed): the body stands on ground contact
    and coasts — a torque-held lean now throws it over the chest-tilt
    envelope within a few ticks — so the note arrives early (0.2 s) and
    the lean is 0.3 rad (~17 deg), which the coasting body still holds
    well inside the tilt bound when the miss fires."""
    sim = _sim([_note(0.2, -0.9, 1.9, speed=speed)], max_steps=300)
    joints = _park_arms(np.zeros(N_JOINTS))
    joints[0] = spine_pitch
    sim.set_joints(list(_q36(joints)), [0.0] * NV)
    for t in range(120):
        _, _, done, ev = sim.step(list(COAST))
        for i, code, *_ in ev:
            if code == 4:
                return t
        assert not done, f"episode ended at tick {t} before the miss"
    raise AssertionError("miss never fired")


def test_miss_boundary_is_head_invariant():
    """env v15 (contact spec §10): the note track is head-anchored, so
    leaning the spine forward shifts the track away by the head's own
    displacement — the miss boundary z < head_z reduces to base z <
    PLANE_Z0, a pure function of chart time, and fires at the SAME tick
    at any lean. (Pre-v15 this asserted leaning < upright: the boundary
    used to move with the head while the track stood still.)"""
    upright = _miss_tick(0.0)
    leaning = _miss_tick(0.3)                   # ~17 deg forward
    assert leaning == upright, (upright, leaning)


def test_reach_gate_moves_with_shoulders():
    """A crouched pose (bent knees/hips lower the shoulders along z via
    the root drop... the gate is SHOULDER-Z based) — lean the spine BACK
    to pull the shoulders away from the plane: a note parked just inside
    the reference reach becomes untouchable earlier. Probe: the forward
    start-pose blade (spans z 0..1) grazes an approaching note exactly
    when it enters reach; with the shoulders pulled back the graze
    happens later (fewer pre-hit ticks with contact chances) — assert
    via first-contact tick of a slow note against a stationary blade.

    2026-09-27 (weld harness removed): the body stands on ground contact
    and coasts (a torque-held -20 deg lean now trips the 20 deg backward
    chest-tilt envelope at once), so the lean is -0.2 rad (~11 deg) and
    the note enters reach ~0.15 s in, before the coasting body tips."""
    def first_contact_tick(spine_pitch):
        sim = _sim([_note(1.1, 0.2, 1.47, speed=2.0)], max_steps=300)
        joints = np.zeros(N_JOINTS)
        joints[0] = spine_pitch
        sim.set_joints(list(_q36(joints)), [0.0] * NV)
        for t in range(90):
            _, _, done, ev = sim.step(list(COAST))
            if ev:
                return t, ev[0][1]
            if done:
                break
        return None, None

    t_ref, code_ref = first_contact_tick(0.0)
    t_back, code_back = first_contact_tick(-0.2)   # ~11 deg back
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
    # additive per-tick schema revision (§6): 2 = "head" on every tick,
    # 3 = "activation" replaced the P6 "effort" field (2026-09-13),
    # 4 = "torque" [total, spine, arms, legs] applied |tau| (env v43),
    # 5 = header "tau_scale" (env v46 per-axis torque-limit scales),
    # 6 = header "tick_hz" / "action_repeat" (tick-rate)
    assert head["header"]["record_rev"] == 6
    assert head["header"]["tick_hz"] == 30
    assert head["header"]["action_repeat"] == 1
    assert "obs_window" not in head["header"]
    assert head["header"]["tau_scale"] == [1.0] * 16
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
        # record_rev 2: FK head pose [p(3), quat wxyz(4)], unit
        # quaternion (rounded to 6 dp)
        assert len(t["head"]) == 7
        assert np.linalg.norm(t["head"][3:7]) == pytest.approx(1.0, abs=1e-5)
    # the head starts at the §1.1 neutral anchor and is NOT a constant:
    # exporters (convert_scoresaber.py) put this rotation on the
    # ScoreSaber viewer's first-person camera
    assert ticks[0]["head"][:3] == pytest.approx([0.0, 1.70, -0.65], abs=0.1)  # tick 0 records AFTER one (randomly actuated) step — the reset pose itself is exact
    assert ticks[0]["head"][3:7] == pytest.approx([1.0, 0.0, 0.0, 0.0],
                                                 abs=0.1)
    assert any(t["head"][:3] != ticks[0]["head"][:3] for t in ticks)
    assert any(t["head"][3:7] != ticks[0]["head"][3:7] for t in ticks)
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
        # hit 0.9 -> 0.8 (env v15, contact spec §10): the head-anchored
        # track moves with the whip's trunk reaction. 0.8 -> 0.6
        # (2026-09-27, ground contact replaced the weld harness): the
        # whip tips the free-standing body within ~17 ticks, so the cut
        # comes earlier (swept: whip 8..12 all land 100-109 at 0.6)
        sim = _sim([_note(0.6, 0.3, 1.4, speed=4.0)], max_steps=300)
        # env v20: this script is calibrated to the pre-v20 zero-arm pose;
        # pin it (the ready stance only moves the episode START pose, not
        # the mechanics under test)
        q = list(sim.qpos())
        for _j in (3, 6, 8, 10, 13, 15):
            q[7 + _j] = 0.0
        sim.set_joints(q, [0.0] * 35)
        events = []
        for t in range(90):
            a = list(COAST)
            a[A_R_ARM] = 1 if t < whip_tick else 2   # raise/hold, then whip
            _, _, done, ev = sim.step(a)
            events.extend(ev)
            if done:
                break
        return sim, events

    # 15 -> 17 (2026-08-18 fist-only hand — see the test_env whip note)
    # 17 -> 14 (env v42 hilt grip: the blade base moved 0.16 m out along
    # the blade, so the same whip meets the note 3 ticks earlier in the
    # swing; whip 14 scores the full 115 on v42, 17 misses the box)
    # 14 -> 10 (2026-09-27, ground contact: see the hit note above)
    sim, events = run(10)
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
