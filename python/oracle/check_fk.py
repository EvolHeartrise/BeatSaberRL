"""P2b gate 1 (MANDATORY FIRST): FK cross-check of mjcf/body_v10.xml
against a reference forward kinematics implemented DIRECTLY from the
SPEC-v10.md kinematic tree table (sections 1.2 / 1.3).

Validates that the MJCF's stacked-joint declaration order actually
reproduces the spec's stacking order (Divergence #7/#10 risk:
shoulder yaw->pitch->twist, wrist roll->dev->bend, MuJoCo composing
same-body hinges in declaration order).

Poses checked:
  (a) neutral (root at reference, all 29 joints 0),
  (b) each actuated joint individually at lo/2 and hi/2,
  (c) 100 seeded random in-limit poses with random root pose.

For every pose, every body's world position AND orientation matrix,
plus the fist centers and blade tips, must agree to <= 1e-8.

Run:  python check_fk.py        (exit 0 = PASS)
"""
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MJCF = os.path.join(HERE, "..", "..", "mjcf", "body_v10.xml")

TOL = 1e-8

# ---------------------------------------------------------------- spec data
# Actuated order (SPEC-v10 1.4) with limits (spec literals).
ACTUATED = [
    ("spine_pitch",   -0.3490658503988659, 0.7853981633974483),
    ("spine_yaw",     -0.6981317007977318, 0.6981317007977318),
    ("spine_roll",    -0.5235987755982988, 0.5235987755982988),
    ("l_arm_pitch",   -1.5707963267948966, 1.1344640137963142),
    ("l_arm_yaw",     -1.7453292519943295, 0.7853981633974483),
    ("l_arm_twist",   -1.5707963267948966, 1.2217304763960306),
    ("l_elbow",        0.0,                2.6179938779914944),
    ("l_wrist_roll",   0.0,                3.141592653589793),
    ("l_wrist_bend",  -1.6580627893946132, 1.6580627893946132),
    ("l_wrist_dev",   -0.5235987755982988, 0.2617993877991494),
    ("r_arm_pitch",   -1.5707963267948966, 1.1344640137963142),
    ("r_arm_yaw",     -0.7853981633974483, 1.7453292519943295),
    ("r_arm_twist",   -1.2217304763960306, 1.5707963267948966),
    ("r_elbow",        0.0,                2.6179938779914944),
    ("r_wrist_roll",  -3.141592653589793,  0.0),
    ("r_wrist_bend",  -1.6580627893946132, 1.6580627893946132),
    ("r_wrist_dev",   -0.2617993877991494, 0.5235987755982988),
    ("l_hip_pitch",   -0.3490658503988659, 2.0943951023931953),
    ("l_hip_yaw",     -0.7853981633974483, 0.6981317007977318),
    ("l_hip_roll",    -0.7853981633974483, 0.5235987755982988),
    ("l_knee",         0.0,                2.6179938779914944),
    ("l_ankle_pitch", -0.8726646259971648, 0.3490658503988659),
    ("l_ankle_roll",  -0.3490658503988659, 0.3490658503988659),
    ("r_hip_pitch",   -0.3490658503988659, 2.0943951023931953),
    ("r_hip_yaw",     -0.6981317007977318, 0.7853981633974483),
    ("r_hip_roll",    -0.5235987755982988, 0.7853981633974483),
    ("r_knee",         0.0,                2.6179938779914944),
    ("r_ankle_pitch", -0.8726646259971648, 0.3490658503988659),
    ("r_ankle_roll",  -0.3490658503988659, 0.3490658503988659),
]

ROOT_POS = np.array([0.0, 0.94, -0.65])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_to_mat(q):
    """wxyz quaternion -> rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def fk_reference(root_p, root_R, q):
    """Reference FK straight from the SPEC-v10 tree table.

    q: dict joint-name -> angle. Returns dict body-name -> (p, R) world,
    plus 'fist_l/r' and 'blade_tip_l/r' world points.
    All anchors are pure translations in the parent frame (1.2);
    stacking orders are the normative products of 1.3.
    """
    out = {}
    out["pelvis"] = (root_p, root_R)

    # spine: R_x(pitch) R_y(yaw) R_z(roll), anchor (0, 0.10, 0)
    p = root_p + root_R @ np.array([0.0, 0.10, 0.0])
    R = root_R @ Rx(q["spine_pitch"]) @ Ry(q["spine_yaw"]) @ Rz(q["spine_roll"])
    out["chest"] = (p, R)

    # head: WELD, anchor (0, 0.66, 0)
    out["head"] = (p + R @ np.array([0.0, 0.66, 0.0]), R)

    for side, sx in (("l", -1.0), ("r", +1.0)):
        # shoulder: stack yaw -> pitch -> twist = R_y(yaw) R_x(-pitch) R_z(twist)
        pu = p + R @ np.array([sx * 0.20, 0.43, 0.0])
        Ru = (R @ Ry(q[f"{side}_arm_yaw"]) @ Rx(-q[f"{side}_arm_pitch"])
              @ Rz(q[f"{side}_arm_twist"]))
        out[f"uarm_{side}"] = (pu, Ru)
        # elbow: axis (-1,0,0) => R_x(-elbow), anchor (0,0,0.35)
        pf = pu + Ru @ np.array([0.0, 0.0, 0.35])
        Rf = Ru @ Rx(-q[f"{side}_elbow"])
        out[f"farm_{side}"] = (pf, Rf)
        # wrist: stack roll -> dev -> bend = R_z(roll) R_y(dev) R_x(-bend),
        # anchor (0,0,0.30)
        ph = pf + Rf @ np.array([0.0, 0.0, 0.30])
        Rh = (Rf @ Rz(q[f"{side}_wrist_roll"]) @ Ry(q[f"{side}_wrist_dev"])
              @ Rx(-q[f"{side}_wrist_bend"]))
        out[f"hand_{side}"] = (ph, Rh)
        out[f"fist_{side}"] = ph + Rh @ np.array([0.0, 0.0, 0.06])
        out[f"blade_tip_{side}"] = ph + Rh @ np.array([0.0, 0.0, 1.0])

        # hip: stack pitch -> yaw -> roll = R_x(-pitch) R_y(yaw) R_z(roll)
        pt = root_p + root_R @ np.array([sx * 0.10, 0.0, 0.0])
        Rt = (root_R @ Rx(-q[f"{side}_hip_pitch"]) @ Ry(q[f"{side}_hip_yaw"])
              @ Rz(q[f"{side}_hip_roll"]))
        out[f"thigh_{side}"] = (pt, Rt)
        # knee: axis (+1,0,0) => R_x(+knee), anchor (0,-0.44,0)
        ps = pt + Rt @ np.array([0.0, -0.44, 0.0])
        Rs = Rt @ Rx(q[f"{side}_knee"])
        out[f"shank_{side}"] = (ps, Rs)
        # ankle: stack pitch -> roll = R_x(-pitch) R_z(roll), anchor (0,-0.43,0)
        pfo = ps + Rs @ np.array([0.0, -0.43, 0.0])
        Rfo = Rs @ Rx(-q[f"{side}_ankle_pitch"]) @ Rz(q[f"{side}_ankle_roll"])
        out[f"foot_{side}"] = (pfo, Rfo)
    return out


BODIES = ["pelvis", "chest", "head",
          "uarm_l", "farm_l", "hand_l", "uarm_r", "farm_r", "hand_r",
          "thigh_l", "shank_l", "foot_l", "thigh_r", "shank_r", "foot_r"]


def mj_pose_error(model, data, root_p, root_quat, q):
    """Set MuJoCo state, run mj_kinematics, return max |ref - mj|."""
    data.qpos[:] = 0.0
    data.qpos[0:3] = root_p
    data.qpos[3:7] = root_quat
    for name, _, _ in ACTUATED:
        adr = model.joint(name).qposadr[0]
        data.qpos[adr] = q[name]
    data.qvel[:] = 0.0
    mujoco.mj_kinematics(model, data)

    ref = fk_reference(np.asarray(root_p, float),
                       quat_to_mat(np.asarray(root_quat, float)), q)
    err = 0.0
    worst = ""
    for b in BODIES:
        p_ref, R_ref = ref[b]
        e_p = np.max(np.abs(data.body(b).xpos - p_ref))
        e_R = np.max(np.abs(data.body(b).xmat.reshape(3, 3) - R_ref))
        for e, tag in ((e_p, f"{b}.pos"), (e_R, f"{b}.rot")):
            if e > err:
                err, worst = e, tag
    for side in ("l", "r"):
        e = np.max(np.abs(data.geom(f"g_fist_{side}").xpos - ref[f"fist_{side}"]))
        if e > err:
            err, worst = e, f"fist_{side}"
        # blade capsule endpoints (geom frame z sign is MuJoCo's choice for
        # fromto geoms): the tip is the endpoint away from the wrist
        g = data.geom(f"g_blade_{side}")
        half = g.xmat.reshape(3, 3) @ np.array([0.0, 0.0, 0.5])
        e1, e2 = g.xpos + half, g.xpos - half
        wrist = data.body(f"hand_{side}").xpos
        tip = e1 if np.linalg.norm(e1 - wrist) > np.linalg.norm(e2 - wrist) else e2
        e = np.max(np.abs(tip - ref[f"blade_tip_{side}"]))
        if e > err:
            err, worst = e, f"blade_tip_{side}"
    return err, worst


def main():
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    neutral = {name: 0.0 for name, _, _ in ACTUATED}

    max_err, max_tag, max_pose = 0.0, "", ""

    def check(tag, root_p, root_q, q):
        nonlocal max_err, max_tag, max_pose
        e, worst = mj_pose_error(model, data, root_p, root_q, q)
        if e > max_err:
            max_err, max_tag, max_pose = e, worst, tag

    # (a) neutral
    check("neutral", ROOT_POS, ident, neutral)
    # sanity anchors of the continuity theorem (1.3) at neutral
    mujoco.mj_kinematics(model, data)
    assert np.max(np.abs(data.body("head").xpos - [0, 1.70, -0.65])) < TOL
    assert np.max(np.abs(data.body("uarm_l").xpos - [-0.2, 1.47, -0.65])) < TOL
    assert np.max(np.abs(data.body("uarm_r").xpos - [+0.2, 1.47, -0.65])) < TOL
    assert np.max(np.abs(data.body("foot_l").xpos - [-0.1, 0.07, -0.65])) < TOL
    assert np.max(np.abs(data.body("foot_r").xpos - [+0.1, 0.07, -0.65])) < TOL

    # (b) each joint at +/- half-limit
    for name, lo, hi in ACTUATED:
        for val in (0.5 * lo, 0.5 * hi):
            q = dict(neutral)
            q[name] = val
            check(f"half-limit {name}={val:+.4f}", ROOT_POS, ident, q)

    # (c) 100 seeded random in-limit poses, random root pose
    rng = np.random.default_rng(20260731)
    for i in range(100):
        q = {name: rng.uniform(lo, hi) for name, lo, hi in ACTUATED}
        root_p = ROOT_POS + rng.uniform(-0.5, 0.5, 3)
        v = rng.normal(size=4)
        root_q = v / np.linalg.norm(v)
        check(f"random[{i}]", root_p, root_q, q)

    print(f"poses checked: 1 + {2 * len(ACTUATED)} + 100")
    print(f"max |ref - mujoco| = {max_err:.3e}  ({max_tag} @ {max_pose})")
    if max_err <= TOL:
        print(f"PASS: FK cross-check (tol {TOL:g})")
        return 0
    print(f"FAIL: exceeds tol {TOL:g}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
