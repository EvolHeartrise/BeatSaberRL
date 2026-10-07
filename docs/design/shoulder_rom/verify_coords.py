"""Check: our shoulder Euler stack R_y(yaw)·R_{-x}(pitch)·R_z(twist) (arm
along local +z) is EXACTLY the ISB Y-X-Y globe about the hanging-arm
reference, and the swing-twist split from hanging has twist yaw - twist."""
import numpy as np


def R(axis, a):
    x, y, z = np.asarray(axis, float)
    s, c = np.sin(a), np.cos(a)
    k = 1 - c
    return np.array([[c + k*x*x, k*x*y - s*z, k*x*z + s*y],
                     [k*x*y + s*z, c + k*y*y, k*y*z - s*x],
                     [k*x*z - s*y, k*y*z + s*x, c + k*z*z]])


rng = np.random.default_rng(0)
B0 = R([1, 0, 0], np.pi / 2)  # hanging reference: local z -> world -y
err = 0.0
for _ in range(2000):
    yaw, p, tw = rng.uniform(-3, 3, 3)
    M = R([0, 1, 0], yaw) @ R([-1, 0, 0], p) @ R([0, 0, 1], tw)
    el = p + np.pi / 2
    M2 = R([0, 1, 0], yaw) @ R([1, 0, 0], -el) @ R([0, 1, 0], -tw) @ B0
    M3 = (R([0, 1, 0], yaw) @ R([1, 0, 0], -el) @ R([0, 1, 0], -yaw)
          @ R([0, 1, 0], yaw - tw) @ B0)
    d = np.array([np.cos(p)*np.sin(yaw), np.sin(p), np.cos(p)*np.cos(yaw)])
    err = max(err, abs(M - M2).max(), abs(M - M3).max(), abs(M[:, 2] - d).max())
print("max decomposition error", err)

poses = {"side, neutral": (0, -np.pi/2, 0),
         "abd 90, twist 0": (-np.pi/2, 0, 0),
         "abd 90, twist -90": (-np.pi/2, 0, -np.pi/2),
         "flex 90, twist 0": (0, 0, 0),
         "flex 90, twist 90": (0, 0, np.pi/2)}
for name, (yaw, p, tw) in poses.items():
    M = R([0, 1, 0], yaw) @ R([-1, 0, 0], p) @ R([0, 0, 1], tw)
    print(f"{name:18s} humerus {np.round(M[:, 2], 3)}  forearm@elbow90 "
          f"{np.round(M[:, 1], 3)}  tau=yaw-twist={np.degrees(yaw - tw):.0f} deg")
