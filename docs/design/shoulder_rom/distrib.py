"""Where do the policies' shoulders live, in anatomical coordinates?"""
import numpy as np

from load_replays import load

D = np.degrees
z = load()
ver = np.where(z["ver"] == "v61all", "v61", z["ver"])
q, w = z["q"], z["w"]
pitch, yaw, tw, elb, roll = q[:, 0], q[:, 1], q[:, 2], q[:, 3], q[:, 4]
theta = D(pitch) + 90          # elevation from hanging
phi = D(yaw) + 90              # plane of elevation: 0 abduction, 90 flexion
tau = D(yaw - tw)              # swing-twist axial rot from hanging, +IR
psi = -D(tw)                   # globe axial rotation, +IR
wtau = w[:, 1] - w[:, 2]

for v in ("v61", "v64", "v65"):
    m = ver == v
    print(f"=== {v}  ({m.sum()} arm-ticks)")
    for name, x in (("elev", theta), ("plane", phi), ("tau(IR+)", tau), ("psi(IR+)", psi),
                    ("elbow", D(elb)), ("roll", D(roll))):
        p = np.percentile(x[m], [1, 5, 25, 50, 75, 95, 99])
        print(f"  {name:9s} p1/5/25/50/75/95/99: " + " ".join(f"{a:7.1f}" for a in p))
    print(f"  |w_twist|>=15.9: {np.mean(abs(w[m,2])>=15.9)*100:.1f}%   elbow<15: "
          f"{np.mean(D(elb[m])<15)*100:.1f}%   |w_yaw|>=15.9: {np.mean(abs(w[m,1])>=15.9)*100:.1f}%"
          f"   |d tau/dt|>16: {np.mean(abs(wtau[m])>16)*100:.1f}%")
    print(f"  twist at stop (within 1deg): lo {np.mean(D(tw[m])<-89)*100:.1f}%  "
          f"hi {np.mean(D(tw[m])>69)*100:.1f}%;  yaw lo {np.mean(D(yaw[m])<-99)*100:.1f}% "
          f"hi {np.mean(D(yaw[m])>44)*100:.1f}%;  pitch hi {np.mean(D(pitch[m])>64)*100:.1f}% "
          f"lo {np.mean(D(pitch[m])<-89)*100:.1f}%")
    # 2-D occupancy: elevation bins x plane bins
    eb = [0, 30, 60, 90, 120, 156]
    pb = [-11, 20, 50, 80, 110, 136]
    H, _, _ = np.histogram2d(theta[m], phi[m], [eb, pb])
    print("  occupancy % (rows elev 0-30..120-155, cols plane -10-20,20-50,50-80,80-110,110-135)")
    for i in range(len(eb) - 1):
        print("   ", " ".join(f"{x:6.1f}" for x in H[i] / m.sum() * 100))
