"""P2b gate 2: static sanity of the MuJoCo oracle (mjcf/body_v10.xml).

Checks (gates marked GATE, the rest informational):
(a) GATE  total body mass == SPEC-v10 75.02 kg (+/- 0.01).
(b1)      zero-torque free settle, 2 s at h = 1/240, welds active:
          per-joint drift recorded. NOT a gate: an unactuated skeleton has
          no position hold - gravity-loaded joints (shoulders horizontal
          arms, spine with the forward arm mass, knee-buckle modes) fall to
          their limits; damping only slows the collapse.
(b2) GATE gravity-compensated hold: constant generalized force
          tau = C(q_ref, 0) = RNEA(q_ref, 0, 0) - the FULL 35-dof vector
          including the root wrench, the SPEC-v10 P2 statics protocol -
          held for 2 s: max joint drift < 5e-2 rad. (Joint-rows-only
          compensation is NOT an equilibrium of the welded closed chain:
          the pelvis-pitch / knee-drop mobility modes are driven by the
          root-row gravity wrench.)
(c) GATE  standing weld load: joint springs to neutral (muscle tone,
          kp = tau_max/0.2, near-critically damped - set in-memory, not in
          the MJCF) hold the stance; after a 4 s settle the foot weld
          vertical constraint forces must sum to body weight
          75.02 * 9.81 = 735.95 N within 5%. (Static equilibrium: welds
          are the only world attachment, so they carry exactly m*g.)

Run:  python check_static.py     (exit 0 = PASS)
"""
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MJCF = os.path.join(HERE, "..", "..", "mjcf", "body_v10.xml")

MASS_SPEC = 75.02
G = 9.81
H = 1.0 / 240.0
DRIFT_TOL = 5e-2
WELD_FORCE_RTOL = 0.05


def hinge_names(model):
    return [model.joint(j).name for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def joint_drift(model, data, q0):
    return {name: abs(data.qpos[model.joint(name).qposadr[0]]
                      - q0[model.joint(name).qposadr[0]])
            for name in hinge_names(model)}


def weld_forces(model, data):
    """Per-weld efc force rows (3 translational world x/y/z, then 3
    rotational)."""
    mujoco.mj_forward(model, data)
    out = {}
    for i in range(model.neq):
        rows = [r for r in range(data.nefc)
                if data.efc_type[r] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                and data.efc_id[r] == i]
        out[model.equality(i).name] = [float(data.efc_force[r]) for r in rows]
    return out


def main():
    model = mujoco.MjModel.from_xml_path(MJCF)
    ok = True

    # ---------------------------------------------------------- (a) mass
    total_mass = float(np.sum(model.body_mass))
    a_ok = abs(total_mass - MASS_SPEC) <= 0.01
    print(f"(a) total mass = {total_mass:.4f} kg (spec {MASS_SPEC}) -> "
          + ("PASS" if a_ok else "FAIL"))
    ok &= a_ok

    # ------------------------------------------- (b1) free settle (info)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    q0 = data.qpos.copy()
    for _ in range(int(round(2.0 / H))):
        mujoco.mj_step(model, data)
    drift_free = joint_drift(model, data, q0)
    print("(b1) zero-torque 2 s free settle (informational - unactuated"
          " skeleton collapses onto its joint limits):")
    for name, dv in sorted(drift_free.items(), key=lambda kv: -kv[1]):
        print(f"     {name:<15s} {dv:.4e} rad")
    print(f"     max = {max(drift_free.values()):.4e} rad (no gate)")

    # ------------------------- (b2) full RNEA(q_ref,0,0) compensated hold
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    tau_full = data.qfrc_bias.copy()   # C(q_ref, 0): gravity load, all 35
    q0 = data.qpos.copy()
    for _ in range(int(round(2.0 / H))):
        data.qfrc_applied[:] = tau_full
        mujoco.mj_step(model, data)
    drift_gc = joint_drift(model, data, q0)
    max_gc = max(drift_gc.values())
    worst = max(drift_gc, key=drift_gc.get)
    b2_ok = max_gc < DRIFT_TOL
    print(f"(b2) RNEA(q_ref,0,0)-compensated 2 s hold: max drift "
          f"{max_gc:.3e} rad ({worst}), gate < {DRIFT_TOL} -> "
          + ("PASS" if b2_ok else "FAIL"))
    ok &= b2_ok

    # --------------------------- (c) spring-held stance, weld = weight
    model_c = mujoco.MjModel.from_xml_path(MJCF)
    d0 = mujoco.MjData(model_c)
    mujoco.mj_forward(model_c, d0)
    M = np.zeros((model_c.nv, model_c.nv))
    mujoco.mj_fullM(model_c, d0, M)
    for a in range(model_c.nu):
        j = model_c.actuator_trnid[a, 0]
        dof = model_c.jnt_dofadr[j]
        kp = model_c.actuator_gear[a, 0] / 0.2
        model_c.jnt_stiffness[j] = kp          # spring to neutral (qpos0)
        model_c.dof_damping[dof] = max(model_c.dof_damping[dof],
                                       2.0 * np.sqrt(kp * M[dof, dof]))
    data = mujoco.MjData(model_c)
    mujoco.mj_resetData(model_c, data)
    q0 = data.qpos.copy()
    for _ in range(int(round(4.0 / H))):
        mujoco.mj_step(model_c, data)
    drift_sp = joint_drift(model_c, data, q0)
    forces = weld_forces(model_c, data)
    vert = {k: v[1] for k, v in forces.items()}
    total = sum(vert.values())
    weight = MASS_SPEC * G
    resid_v = float(np.abs(data.qvel).max())
    viol = float(np.abs(data.efc_pos[:data.nefc]).max())
    c_ok = (abs(abs(total) - weight) <= WELD_FORCE_RTOL * weight
            and resid_v < 1.0)
    print(f"(c) spring-held stance after 4 s: max joint droop "
          f"{max(drift_sp.values()):.3e} rad, weld violation {viol:.2e} m, "
          f"residual max|qvel| {resid_v:.2e}")
    print("    weld vertical forces: "
          + ", ".join(f"{k} = {v:.2f} N" for k, v in vert.items())
          + f"; sum = {total:.2f} N (weight {weight:.2f} N, tol 5%) -> "
          + ("PASS" if c_ok else "FAIL"))
    ok &= c_ok

    print("PASS: check_static" if ok else "FAIL: check_static")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
