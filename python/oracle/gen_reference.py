"""P2b deliverable 3: MuJoCo reference trajectories for the v10 engine's
tolerance-band comparisons (SPEC-v10 P2b: different constraint solvers -
bands, not parity).

5 seeded scenarios, each 1 s at h = 1/240 (240 substeps), on the
FREE-FLOATING body: the MJCF's two foot welds are deactivated at load
(`eq_active0 = 0`) and the floor plane never collides (every geom is
contype 0), so the body falls freely under gravity with only the joint
limits, the 55 self-collision pairs and the stabilizer acting. This is the
engine's `ConstraintSet::new(m, contacts, ground = false)` regime - the
v10 foot-weld mode these references used to exercise was removed from the
engines 2026-09-27 (issue #2), and MuJoCo's soft ground contact is too
far from our PGS ground block for tolerance bands. Per-substep qpos (36) /
qvel (35) at full precision to refdata/<name>.json:

  1 neutral_hold       stabilized stance, no pattern torque
  2 const_l_arm_pitch  constant +60 N*m on l_arm_pitch
  3 sin_both_arms      sinusoidal torques on both arms (R phase +pi/2)
  4 leg_spine          sinusoidal spine + leg torque pattern
  5 random_all29       seeded random bounded torque on all 29 joints,
                       piecewise-constant per 8-substep control tick

STABILIZER (normative for consumers). The raw v10 body has near-zero
wrist damping and relies on the engine's hard omega caps; open-loop it is
chaotically unintegrable at h = 1/240 (MuJoCo NaNs within ~0.5 s under
any sustained torque, with limits on it rattles the wrists at hundreds of
rad/s - measured during P2b bring-up). All references therefore run on
the "muscle tone" stabilized body:

    total joint torque_j = tau29_j(t) + kp_j * (0 - q_j) - kd_j * qdot_j

with kp_j = tau_max_j / 0.2 and kd_j = max(b_j, 2*sqrt(kp_j * M_jj(q0)))
(near-critical; REPLACES the spec damping b_j - it is the total dof
damping). kp/kd/setpoint are exported per joint in each JSON; an engine
consumer must implement the same spring-damper law (trivial per-substep
joint torque), then band-compare qpos/qvel. Resulting trajectories are
tame: max |qvel| < 10 rad/s, joint excursions < 0.16 rad - inside all
joint limits and omega caps, so limit/clamp model differences are
irrelevant to the comparison.

Joint limits stay ENABLED (never touched); integrator Euler; contacts
enabled (none activate at these excursions).

File schema:
  {"scenario", "seed", "h", "n_substeps", "integrator",
   "stabilizer": {"type", "kp": [29], "kd": [29], "setpoint": [29]},
   "joint_order": [29 names, SPEC-v10 1.4 actuated order],
   "qpos_joint_names": [29 names, MJCF tree/qpos order],
   "tau29": [n_substeps][29] pattern torque (actuated order),
   "qpos": [n_substeps+1][36], "qvel": [n_substeps+1][35]}
qpos rows: root pos(3) + root quat wxyz(4) + 29 joints in MJCF tree order
(shoulders yaw,pitch,twist / wrists roll,dev,bend in qpos - map before
comparing against the actuated-order engine state).

Run:  python gen_reference.py     (writes refdata/*.json)
"""
import json
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MJCF = os.path.join(HERE, "..", "..", "mjcf", "body_v10.xml")
OUT = os.path.join(HERE, "refdata")

H = 1.0 / 240.0
N_SUB = 240          # 1 s
TICK = 8             # control tick = 8 substeps
TWO_PI = 2.0 * np.pi
QVEL_SANE = 30.0


def build_stabilized():
    """Model with muscle-tone joint springs + near-critical damping.
    Returns (model, names29, dofs29, taumax29, kp29, kd29)."""
    model = mujoco.MjModel.from_xml_path(MJCF)
    # free-floating references: the MJCF welds stay in the file for the
    # check_static.py statics gates, but are OFF here (see module doc)
    model.eq_active0[:] = 0
    d0 = mujoco.MjData(model)
    mujoco.mj_forward(model, d0)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, d0, M)
    names, dofs, taumax, kp29, kd29 = [], [], [], [], []
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        dof = int(model.jnt_dofadr[j])
        kp = float(model.actuator_gear[a, 0]) / 0.2
        model.jnt_stiffness[j] = kp
        kd = max(float(model.dof_damping[dof]), 2.0 * np.sqrt(kp * M[dof, dof]))
        model.dof_damping[dof] = kd
        names.append(model.joint(j).name)
        dofs.append(dof)
        taumax.append(float(model.actuator_gear[a, 0]))
        kp29.append(kp)
        kd29.append(kd)
    return (model, names, np.array(dofs), np.array(taumax),
            np.array(kp29), np.array(kd29))


def simulate(model, dofs, tau_fn):
    """Run N_SUB substeps applying tau_fn(k) (pattern torque, actuated
    order) on top of the model's built-in spring/damper. Asserts the
    trajectory stays tame and MuJoCo never fired an instability warning
    (warnings silently reset the state and would corrupt the data)."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    qpos_h = [data.qpos.copy()]
    qvel_h = [data.qvel.copy()]
    tau_h = []
    for k in range(N_SUB):
        tau29 = np.asarray(tau_fn(k), dtype=float)
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dofs] = tau29
        mujoco.mj_step(model, data)
        tau_h.append(tau29.copy())
        qpos_h.append(data.qpos.copy())
        qvel_h.append(data.qvel.copy())
    assert int(np.sum(data.warning.number)) == 0, "MuJoCo warning fired"
    vmax = float(np.max(np.abs(np.array(qvel_h))))
    assert np.isfinite(np.array(qpos_h)).all()
    assert vmax < QVEL_SANE, f"trajectory too hot: max|qvel| = {vmax:.1f}"
    return qpos_h, qvel_h, tau_h, vmax


def qpos_joint_names(model):
    return [model.joint(j).name for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def main():
    os.makedirs(OUT, exist_ok=True)
    model, names, dofs, taumax, kp29, kd29 = build_stabilized()
    idx = {n: i for i, n in enumerate(names)}
    stab = {"type": "joint_spring_damper_to_neutral",
            "law": "tau_total_j = tau29_j + kp_j*(setpoint_j - q_j) - kd_j*qdot_j",
            "kp": list(kp29), "kd": list(kd29),
            "setpoint": [0.0] * 29,
            "note": "kd is the TOTAL joint damping (replaces spec b_j)"}

    def write(name, seed, qp, qv, th, vmax):
        obj = {"scenario": name, "seed": seed, "h": H, "n_substeps": N_SUB,
               "integrator": "Euler", "stabilizer": stab,
               "joint_order": names,
               "qpos_joint_names": qpos_joint_names(model),
               "tau29": [list(t) for t in th],
               "qpos": [list(q) for q in qp],
               "qvel": [list(v) for v in qv]}
        path = os.path.join(OUT, f"{name}.json")
        with open(path, "w") as f:
            json.dump(obj, f)
        print(f"wrote {path}  ({os.path.getsize(path) / 1024:.0f} KiB, "
              f"max|qvel| {vmax:.2f})")

    # 1 neutral hold (stabilized stance, no pattern)
    qp, qv, th, vm = simulate(model, dofs, lambda k: np.zeros(29))
    write("neutral_hold", 0, qp, qv, th, vm)

    # 2 constant torque on left shoulder pitch
    tau2 = np.zeros(29)
    tau2[idx["l_arm_pitch"]] = 60.0
    qp, qv, th, vm = simulate(model, dofs, lambda k: tau2)
    write("const_l_arm_pitch", 0, qp, qv, th, vm)

    # 3 sinusoidal torques on both arms (right side phase +pi/2)
    def tau_arms(k):
        t = k * H  # torque held over (t, t+H]
        tau = np.zeros(29)
        for side, ph in (("l", 0.0), ("r", 0.5 * np.pi)):
            tau[idx[f"{side}_arm_pitch"]] = 100.0 * np.sin(TWO_PI * 1.5 * t + ph)
            tau[idx[f"{side}_arm_yaw"]] = 75.0 * np.sin(TWO_PI * 1.0 * t + ph)
            tau[idx[f"{side}_elbow"]] = 18.0 * np.sin(TWO_PI * 2.0 * t + ph)
            tau[idx[f"{side}_wrist_bend"]] = 1.0 * np.sin(TWO_PI * 3.0 * t + ph)
        return tau
    qp, qv, th, vm = simulate(model, dofs, tau_arms)
    write("sin_both_arms", 0, qp, qv, th, vm)

    # 4 leg + spine torque pattern (hip pitch L/R antiphase)
    def tau_legs(k):
        t = k * H
        tau = np.zeros(29)
        tau[idx["spine_pitch"]] = 250.0 * np.sin(TWO_PI * 1.0 * t)
        tau[idx["spine_yaw"]] = 150.0 * np.sin(TWO_PI * 0.7 * t)
        tau[idx["spine_roll"]] = 200.0 * np.sin(TWO_PI * 0.9 * t)
        tau[idx["l_hip_pitch"]] = 200.0 * np.sin(TWO_PI * 1.2 * t)
        tau[idx["r_hip_pitch"]] = -200.0 * np.sin(TWO_PI * 1.2 * t)
        tau[idx["l_knee"]] = 60.0 * np.sin(TWO_PI * 1.5 * t)
        tau[idx["r_knee"]] = 60.0 * np.sin(TWO_PI * 1.5 * t)
        tau[idx["l_ankle_pitch"]] = 25.0 * np.sin(TWO_PI * 2.0 * t)
        tau[idx["r_ankle_pitch"]] = 25.0 * np.sin(TWO_PI * 2.0 * t)
        return tau
    qp, qv, th, vm = simulate(model, dofs, tau_legs)
    write("leg_spine", 0, qp, qv, th, vm)

    # 5 random bounded torque on all 29, piecewise-constant per tick
    # (wrist tier scaled down - tiny inertia)
    seed = 20260731
    rng = np.random.default_rng(seed)
    scale = np.where(taumax < 2.0, 0.15, 0.45) * taumax
    ticks = [rng.uniform(-1.0, 1.0, 29) * scale
             for _ in range(N_SUB // TICK)]
    qp, qv, th, vm = simulate(model, dofs, lambda k: ticks[k // TICK])
    write("random_all29", seed, qp, qv, th, vm)

    print("done: 5 scenarios")
    return 0


if __name__ == "__main__":
    sys.exit(main())
