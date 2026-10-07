"""What axial-rotation window does today's fixed twist box allow at each arm
direction? (left-canonical; tau = yaw - twist, IR +; twist box -90..+70)"""
import numpy as np

TW = (-90.0, 70.0)
YAW = (-100.0, 45.0)


def window(yaw):
    return yaw - TW[1], yaw - TW[0]   # tau range (ER limit, IR limit)


print("direction                 yaw   tau window (ER..IR)   width")
for name, yaw in (("90 abduction (plane 0)", -90), ("scapular plane (30)", -60),
                  ("plane 45", -45), ("90 flexion (plane 90)", 0),
                  ("cross-body (plane 135)", 45)):
    lo, hi = window(yaw)
    print(f"{name:24s} {yaw:5.0f}   ER {-lo:5.0f} .. IR {hi:5.0f}       {hi - lo:4.0f}")
lo = YAW[0] - TW[1]
hi = YAW[1] - TW[0]
print(f"arm hanging (pole): yaw is axial too -> tau in [{lo:.0f}, {hi:.0f}] = ER {-lo:.0f} .. IR {hi:.0f}"
      f" (width {hi - lo:.0f} deg)")
