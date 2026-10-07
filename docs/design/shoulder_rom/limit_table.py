"""Print the fitted human axial limits over the reachable (elevation, plane)
box, in tau (swing-twist from hanging; IR +) and ISB globe psi (IR +), next
to today's twist-box window, plus the fitted coefficients."""
import numpy as np

from srom_model import ANCHORS, ShoulderROM, DEG

m = ShoulderROM()
print("coefficients (deg): a0 a1 a2 B   [f = a0 + a1 c + a2 c^2 + B (d_l - d_f) + (1 - c) phi]")
for k, v in m.cf.items():
    print(f"  {k}: " + " ".join(f"{x / DEG:8.3f}" for x in v))
print("anchor reproduction:")
for n, (th, ph, lo, hi) in ANCHORS.items():
    L = m.limits(np.array((ph - 90) * DEG), np.array((th - 90) * DEG))
    print(f"  {n:7s} target ER {lo:6.1f} IR {hi:6.1f}   fit {L['lo'][0] / DEG:6.1f} {L['hi'][0] / DEG:6.1f}")
print("\nlimits: tau [ER..IR] (arc) | ISB psi [ER..IR] | today's box tau window")
for th in (0, 30, 60, 90, 120, 155):
    for ph in (-10, 30, 60, 90, 135):
        y, p = (ph - 90) * DEG, (th - 90) * DEG
        L = m.limits(np.array(y), np.array(p))
        lo, hi = L["lo"][0] / DEG, L["hi"][0] / DEG
        blo, bhi = (ph - 90) - 70, (ph - 90) + 90
        if th == 0:
            blo, bhi = -170, 135
        print(f"  elev {th:3d} plane {ph:4d}: tau [{lo:7.1f} .. {hi:6.1f}] ({hi - lo:5.1f})  "
              f"psi [{lo - ph:7.1f} .. {hi - ph:6.1f}]   box [{blo:5.0f} .. {bhi:4.0f}]")
