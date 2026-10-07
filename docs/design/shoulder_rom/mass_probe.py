"""Shoulder effective inertias from the model's own M(q).

Poses -> the Rust probe (srom_probe.rs, a temporary test that dumps CRBA M
with armature) -> per pose the 3x3 shoulder block of M^-1 (yaw, pitch,
twist free; every other joint and the root free too: the inverse, not the
block of M), and the axial effective inertia 1 / (J M^-1 J^T) for the
axial coordinate tau = yaw - twist.

usage: python mass_probe.py gen   -> writes poses.txt
       copy srom_probe.rs.txt to rust/src/srom_probe.rs, register it in
       lib.rs next to tendon_tests (`#[cfg(all(test, not(feature = "f32")))]
       mod srom_probe;`), then
       SROM_POSES=.../poses.txt SROM_OUT=.../mass.txt cargo test --release srom_dump
       python mass_probe.py read  -> prints the table
(the probe is a design-time tool, not part of the test suite)
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.environ.get("SROM_SCRATCH", "/tmp/claude-1000/-home-evolheartrise-git-projects-"
                     "BeatSaberRL/dcb1a0a2-73f1-4c8b-9ec9-f6f9911b2ca5/scratchpad/srom")
Q_START = {3: -1.3962634015954636, 6: 1.7453292519943295, 8: 1.2217304763960306,
           10: -1.3962634015954636, 13: 1.7453292519943295, 15: 1.2217304763960306,
           19: -0.13962634015954636, 22: 0.13962634015954636,
           25: 0.13962634015954636, 28: -0.13962634015954636}
R = np.radians


def grid():
    out = []
    for pitch, yaw, tw, elb, roll in itertools.product(
            [-89, -60, -30, 0, 30, 64], [-99, -45, 0, 44], [-89, 0, 69],
            [0, 15, 90], [0, 90, 179]):
        q = np.zeros(29)
        for j, v in Q_START.items():
            q[j] = v
        q[3], q[4], q[5], q[6], q[7] = R(pitch), R(yaw), R(tw), R(elb), R(roll)
        q[8] = 0.0  # wrist straight (bend=0 is the stiffest hand lever case)
        out.append(((pitch, yaw, tw, elb, roll), q))
    return out


def minv_blocks():
    g = grid()
    Ms = np.loadtxt(os.path.join(SCR, "mass.txt"))
    nv = 35
    res = []
    for (key, q), m in zip(g, Ms):
        M = m.reshape(nv, nv)
        Mi = np.linalg.inv(M)
        idx = [6 + 4, 6 + 3, 6 + 5]  # yaw, pitch, twist (left)
        B = Mi[np.ix_(idx, idx)]
        res.append((key, q, B, Mi))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        os.makedirs(SCR, exist_ok=True)
        with open(os.path.join(SCR, "poses.txt"), "w") as f:
            for _, q in grid():
                f.write(" ".join(f"{x:.17g}" for x in q) + "\n")
        print("wrote", len(grid()))
    else:
        res = minv_blocks()
        J = np.array([1.0, 0.0, -1.0])
        print("pitch yaw twist elbow roll | I_eff yaw pitch twist (1/Minv_jj) | I_ax=1/(J Minv J^T)")
        iax = []
        for key, q, B, _ in res:
            ia = 1.0 / (J @ B @ J)
            iax.append((ia, key))
            if key[4] == 90 and key[2] == 0:
                print(key, " ".join(f"{1/B[i,i]:.4f}" for i in range(3)), f"| {ia:.4f}")
        iax.sort()
        print("smallest axial I_eff:", iax[:4])
        print("largest axial I_eff:", iax[-3:])
