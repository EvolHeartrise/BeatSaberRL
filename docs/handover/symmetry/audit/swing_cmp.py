import sys, numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
d = np.linalg.norm(a[..., :3] - b[..., :3], axis=-1).max(1)
print(f"{sys.argv[2]}: swinging-tip path mismatch median {np.median(d)*1000:.3f} mm, p90 {np.percentile(d,90)*1000:.2f} mm, max {d.max()*1000:.1f} mm")
