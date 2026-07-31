"""P2b gate 4: contact wiring of the MuJoCo oracle vs the SPEC-v10
collision spec (section 2).

Checks:
(1) GATE  the MJCF <contact> roster is EXACTLY the spec's 55 live pairs,
          in normative pair-id order (0-54), with the 11 excluded pairs
          absent (the exclusion list is rebuilt here from the spec table,
          not read from the MJCF).
(2) GATE  pose A (wrists driven across the midline, arms crossed:
          l/r_arm_yaw = +/-22 deg, l/r_arm_pitch = +/-4 deg, in-limit):
          MuJoCo reports contact between EXACTLY the pairs an independent
          capsule-distance computation on the SPEC-v10 reference FK
          predicts (dist < margin 0.01) - here farm_l-farm_r.
(3) GATE  pose B (forearm folded across the torso: l_arm_yaw 38 deg,
          l_arm_pitch -25 deg, l_arm_twist -30 deg, l_elbow 120 deg,
          in-limit): same exact-set check - here torso-farm_l.
(4) GATE  dynamic press: muscle-tone springs (as gen_reference.py) hold
          the contact pose, per-substep gravity compensation on the joint
          rows removes droop, and a constant EXTRA torque drives the
          wrists further across the midline (pose A: +/-5 N*m on both
          shoulder yaws; pose B: +5 N*m yaw and +3 N*m elbow) for 1 s.
          The contact must balance the push: max penetration
          < 2 x slop = 0.01 m at every substep, the required pair must be
          force-active (dist < margin - gap = 0.005) at some substep, and
          every contact ever seen must be one of the 55 live pairs.

Run:  python check_contacts.py     (exit 0 = PASS)
"""
import os
import sys

import numpy as np
import mujoco

import check_fk as fk

HERE = os.path.dirname(os.path.abspath(__file__))
MJCF = os.path.join(HERE, "..", "..", "mjcf", "body_v10.xml")

RAD = np.pi / 180.0
MARGIN = 0.01
SLOP = 0.005
GAP = 0.005          # MJCF pair gap: force-active when dist < margin - gap

# spec section 2: collider roster (id order) and geom names in the MJCF
IDS = ["torso", "head", "uarm_l", "uarm_r", "farm_l", "farm_r",
       "fist_l", "fist_r", "thigh_l", "thigh_r", "shank_l", "shank_r"]
GEOM = {"torso": "g_torso", "head": "g_head",
        "uarm_l": "g_uarm_l", "uarm_r": "g_uarm_r",
        "farm_l": "g_farm_l", "farm_r": "g_farm_r",
        "fist_l": "g_fist_l", "fist_r": "g_fist_r",
        "thigh_l": "g_thigh_l", "thigh_r": "g_thigh_r",
        "shank_l": "g_shank_l", "shank_r": "g_shank_r"}
# spec's 11 excluded pairs
EXCL = {(0, 1), (0, 2), (0, 3), (0, 8), (0, 9),
        (2, 4), (3, 5), (4, 6), (5, 7), (8, 10), (9, 11)}
LIVE = [(i, j) for i in range(12) for j in range(i + 1, 12)
        if (i, j) not in EXCL]                       # lexicographic = pair id
assert len(LIVE) == 55


def seg_seg(p1, q1, p2, q2):
    """Closest distance between segments (Ericson 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    EPS = 1e-12
    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(r))
    if a <= EPS:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= EPS:
            s, t = np.clip(-c / a, 0, 1), 0.0
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > EPS else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm(p1 + s * d1 - (p2 + t * d2)))


def collider_segments(q):
    """World (endpoint, endpoint, radius) per collider from the reference
    FK (spec geometry, independent of MuJoCo)."""
    ref = fk.fk_reference(fk.ROOT_POS, np.eye(3), q)

    def seg(body, a, b, r):
        p, R = ref[body]
        return (p + R @ np.array(a), p + R @ np.array(b), r)

    c = {"torso": seg("chest", (0, 0, 0), (0, 0.43, 0), 0.15)}
    p, _ = ref["head"]
    c["head"] = (p, p, 0.09)
    for s in ("l", "r"):
        c[f"uarm_{s}"] = seg(f"uarm_{s}", (0, 0, 0), (0, 0, 0.35), 0.045)
        c[f"farm_{s}"] = seg(f"farm_{s}", (0, 0, 0), (0, 0, 0.30), 0.04)
        fist = ref[f"fist_{s}"]
        c[f"fist_{s}"] = (fist, fist, 0.05)
        c[f"thigh_{s}"] = seg(f"thigh_{s}", (0, 0, 0), (0, -0.44, 0), 0.06)
        c[f"shank_{s}"] = seg(f"shank_{s}", (0, 0, 0), (0, -0.43, 0), 0.045)
    return c


def expected_contacts(q):
    """{(id_i, id_j): gap} for live pairs with gap < MARGIN, from the
    independent spec geometry."""
    c = collider_segments(q)
    out = {}
    for i, j in LIVE:
        a1, b1, r1 = c[IDS[i]]
        a2, b2, r2 = c[IDS[j]]
        g = seg_seg(a1, b1, a2, b2) - r1 - r2
        if g < MARGIN:
            out[(i, j)] = g
    return out


def set_pose(model, data, q):
    mujoco.mj_resetData(model, data)
    for name, _, _ in fk.ACTUATED:
        data.qpos[model.joint(name).qposadr[0]] = q.get(name, 0.0)
    mujoco.mj_forward(model, data)


def mj_contact_pairs(model, data, max_dist=MARGIN):
    """{(id_i, id_j): dist} seen by MuJoCo (geom names -> spec ids)."""
    g2id = {GEOM[k]: i for i, k in enumerate(IDS)}
    out = {}
    for k in range(data.ncon):
        con = data.contact[k]
        if con.dist >= max_dist:
            continue
        n1 = model.geom(con.geom1).name
        n2 = model.geom(con.geom2).name
        i, j = sorted((g2id[n1], g2id[n2]))
        out[(i, j)] = min(out.get((i, j), np.inf), float(con.dist))
    return out


def fmt(pairs):
    return "{" + ", ".join(f"{IDS[i]}-{IDS[j]}:{d:+.4f}"
                           for (i, j), d in sorted(pairs.items())) + "}"


# All pose angles are inside the SPEC-v10 1.4 joint limits (checked at
# import below); note l_arm_yaw hi = +45 deg caps how far the left wrist
# can cross by yaw alone - pose B folds the forearm with the elbow.
POSE_A = {"l_arm_yaw": 22 * RAD, "r_arm_yaw": -22 * RAD,
          "l_arm_pitch": 4 * RAD, "r_arm_pitch": -4 * RAD}
POSE_B = {"l_arm_yaw": 38 * RAD, "l_arm_pitch": -25 * RAD,
          "l_arm_twist": -30 * RAD, "l_elbow": 120 * RAD}
# constant extra torques (N*m, joint name -> value) pressing the contact
PUSH_A = {"l_arm_yaw": +5.0, "r_arm_yaw": -5.0}
PUSH_B = {"l_arm_yaw": +5.0, "l_elbow": +3.0}

_LIM = {n: (lo, hi) for n, lo, hi in fk.ACTUATED}
for _pose in (POSE_A, POSE_B):
    for _n, _v in _pose.items():
        assert _LIM[_n][0] <= _v <= _LIM[_n][1], f"{_n}={_v} out of limits"


def check_roster(model):
    """Compiled pair SET must equal the spec's 55 live pairs, and the MJCF
    <pair> DECLARATION order must be the normative pair-id order (MuJoCo
    re-sorts pairs internally by geom ids, so the XML text is the order
    authority)."""
    g2id = {GEOM[k]: i for i, k in enumerate(IDS)}
    compiled = set()
    for p in range(model.npair):
        n1 = model.geom(model.pair_geom1[p]).name
        n2 = model.geom(model.pair_geom2[p]).name
        compiled.add(tuple(sorted((g2id[n1], g2id[n2]))))
    set_ok = compiled == set(LIVE)

    import re
    xml = open(MJCF).read()
    declared = [tuple(sorted((g2id[a], g2id[b])))
                for a, b in re.findall(
                    r'<pair\s+geom1="(\w+)"\s+geom2="(\w+)"', xml)]
    order_ok = declared == LIVE
    ok = set_ok and order_ok
    print(f"(1) MJCF pair roster: compiled set {len(compiled)}/55 match ="
          f" {set_ok}, XML declaration order match = {order_ok} -> "
          + ("PASS" if ok else "FAIL"))
    if not order_ok:
        for k, (a, b) in enumerate(zip(declared, LIVE)):
            if a != b:
                print(f"    first order mismatch at pair id {k}: "
                      f"xml {a} spec {b}")
                break
    return ok


def check_static(model, data, tag, pose, required):
    exp = expected_contacts(pose_full(pose))
    set_pose(model, data, pose)
    got = mj_contact_pairs(model, data)
    ok = set(exp) == set(got) and required in got
    # gaps must also agree pairwise (same narrow phase) to 1e-6
    gap_err = max((abs(exp[p] - got[p]) for p in exp if p in got),
                  default=0.0)
    ok = ok and gap_err < 1e-6
    print(f"({tag}) static pose: expected {fmt(exp)}")
    print(f"      mujoco   {fmt(got)}  max|gap err| {gap_err:.2e} -> "
          + ("PASS" if ok else "FAIL"))
    return ok


def pose_full(pose):
    q = {name: 0.0 for name, _, _ in fk.ACTUATED}
    q.update(pose)
    return q


def check_dynamic(tag, pose, push, required):
    model = mujoco.MjModel.from_xml_path(MJCF)
    d0 = mujoco.MjData(model)
    mujoco.mj_forward(model, d0)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, d0, M)
    push_vec = np.zeros(model.nv)
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        dof = model.jnt_dofadr[j]
        kp = model.actuator_gear[a, 0] / 0.2
        model.jnt_stiffness[j] = kp
        model.dof_damping[dof] = max(model.dof_damping[dof],
                                     2.0 * np.sqrt(kp * M[dof, dof]))
        name = model.joint(j).name
        model.qpos_spring[model.jnt_qposadr[j]] = pose.get(name, 0.0)
        push_vec[dof] = push.get(name, 0.0)
    joint_dofs = np.array([int(model.jnt_dofadr[model.actuator_trnid[a, 0]])
                           for a in range(model.nu)])
    data = mujoco.MjData(model)
    set_pose(model, data, pose)   # start AT the contact pose, v = 0
    max_pen = 0.0
    req_min = np.inf
    seen = set()
    for _ in range(240):
        # per-substep gravity compensation on the joint rows (safe: the
        # muscle-tone springs keep the body feedback-stable) removes the
        # droop; push_vec presses the contact on top of the spring hold
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[joint_dofs] = data.qfrc_bias[joint_dofs]
        data.qfrc_applied += push_vec
        mujoco.mj_step(model, data)
        got = mj_contact_pairs(model, data)
        seen |= set(got)
        pen = max((-d for d in got.values() if d < 0), default=0.0)
        max_pen = max(max_pen, pen)
        if required in got:
            req_min = min(req_min, got[required])
    rogue = seen - set(LIVE)
    req_active = req_min < MARGIN - GAP
    ok = (max_pen < 2 * SLOP) and req_active and not rogue
    print(f"({tag}) dynamic press: pairs seen "
          + "{" + ", ".join(f"{IDS[i]}-{IDS[j]}" for i, j in sorted(seen)) + "}")
    print(f"      max penetration {max_pen:.4f} m (gate < {2 * SLOP}), "
          f"required {IDS[required[0]]}-{IDS[required[1]]} min dist "
          f"{req_min:+.4f} (force-active = {req_active}), rogue pairs = "
          f"{sorted(rogue)} -> " + ("PASS" if ok else "FAIL"))
    return ok


def main():
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    farm_ff = (IDS.index("farm_l"), IDS.index("farm_r"))
    torso_fl = (IDS.index("torso"), IDS.index("farm_l"))
    ok = check_roster(model)
    ok &= check_static(model, data, "2", POSE_A, farm_ff)
    ok &= check_static(model, data, "3", POSE_B, torso_fl)
    ok &= check_dynamic("4a", POSE_A, PUSH_A, farm_ff)
    ok &= check_dynamic("4b", POSE_B, PUSH_B, torso_fl)
    print("PASS: check_contacts" if ok else "FAIL: check_contacts")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
