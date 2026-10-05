import sys, numpy as np
from mirror_ops import *
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
mv = np.array([np.concatenate([m_qpos(r[:36]), m_qvel(r[36:])]) for r in a])
d = np.abs(b - mv)[:, 36:]
print(f"{sys.argv[2]}: n={len(d)} median max|dv|={np.median(d.max(1)):.2e}  p90={np.percentile(d.max(1),90):.2e}  max={d.max():.2e}  RMS={np.sqrt((d**2).mean()):.2e}")
