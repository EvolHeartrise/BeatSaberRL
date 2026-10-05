import numpy as np
from mirror_ops import *
np.set_printoptions(precision=3, suppress=True, linewidth=200)
z = np.load("s.npz"); a = np.load("s_run_orig_base.npy"); b = np.load("s_run_mirror_base.npy")
mv = np.array([np.concatenate([m_qpos(r[:36]), m_qvel(r[36:])]) for r in a])
d = np.abs(b - mv)[:, 36:]
names = ["vx","vy","vz","wx","wy","wz"] + ["sp_p","sp_y","sp_r"] + [f"L_{n}" for n in "pitch yaw twist elbow roll bend dev".split()] + [f"R_{n}" for n in "pitch yaw twist elbow roll bend dev".split()] + [f"L_{n}" for n in "hp hy hr knee ap ar".split()] + [f"R_{n}" for n in "hp hy hr knee ap ar".split()]
rms = np.sqrt((d**2).mean(0))
print("per-coordinate RMS defect:"); print(" ".join(f"{n}:{r:.2f}" for n, r in zip(names, rms)))
# how often is the defect > 0.1 rad/s
print("fraction of samples with max defect > 0.01, 0.1, 1:", [(d.max(1) > t).mean().round(3) for t in (0.01, 0.1, 1)])
i = int(np.argsort(d.max(1))[len(d)//2])
s = z["S"][i]; print("median case idx", i, "q:", s[7:36], "\n v:", s[36:], "\n a:", z["A"][i])
print(" defect:", (b - mv)[i, 36:])
