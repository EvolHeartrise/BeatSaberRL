"""Balance forensics over the c2h replays (2026-08-08 user hypothesis:
body leans back + rides the right leg's extension stops because the
lower body cannot hold itself upright actively).

Per tick, from the JSONL: root pitch (lean), leg-joint limit-riding
(within 0.5 deg of a stop), gear usage per leg joint, per-foot vertical
reaction force. Statics computed separately from model.rs numbers.
"""

import glob
import json
import math
import sys

# model.rs leg rows (q index = 7 + joint index; action index = joint index)
LEG = {
    17: ("l_hip_pitch", -0.3490658504, 2.0943951024, 484.68),
    18: ("l_hip_yaw",   -0.7853981634, 0.6981317008, 100.00),
    19: ("l_hip_roll",  -0.7853981634, 0.5235987756, 300.00),
    20: ("l_knee",       0.0,          2.6179938780, 250.00),
    21: ("l_ankle_pitch", -0.8726646260, 0.3490658504, 180.00),
    22: ("l_ankle_roll", -0.3490658504, 0.3490658504,  50.00),
    23: ("r_hip_pitch", -0.3490658504, 2.0943951024, 484.68),
    24: ("r_hip_yaw",   -0.6981317008, 0.7853981634, 100.00),
    25: ("r_hip_roll",  -0.5235987756, 0.7853981634, 300.00),
    26: ("r_knee",       0.0,          2.6179938780, 250.00),
    27: ("r_ankle_pitch", -0.8726646260, 0.3490658504, 180.00),
    28: ("r_ankle_roll", -0.3490658504, 0.3490658504,  50.00),
}
GEAR_FRAC = {0: 0.0, 1: 1.0, 2: -1.0, 3: 0.25, 4: -0.25}   # K=5
EPS = math.radians(0.5)

def root_pitch(qw, qx, qy, qz):
    # x-axis (pitch) Euler component of the root quaternion
    return math.atan2(2 * (qw * qx + qy * qz),
                      1 - 2 * (qx * qx + qy * qy))

files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                         else "replays/c2h_*.jsonl"))
ticks = 0
pitch_sum = pitch_min = pitch_max = 0.0
pitches = []
at_lo = {j: 0 for j in LEG}
at_hi = {j: 0 for j in LEG}
gear_use = {j: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0} for j in LEG}
fy_l = fy_r = 0.0
fy_l_only = 0        # ticks where |fy_r| carries >80% of vertical load

for path in files:
    with open(path) as f:
        next(f)                              # header
        for line in f:
            d = json.loads(line)
            q = d["q"]
            ticks += 1
            p = root_pitch(q[3], q[4], q[5], q[6])
            pitches.append(p)
            for j, (nm, lo, hi, _tm) in LEG.items():
                qq = q[7 + j]
                if qq - lo < EPS:
                    at_lo[j] += 1
                elif hi - qq < EPS:
                    at_hi[j] += 1
                gear_use[j][d["action"][j]] += 1
            fl, fr = d["feet"][0][1], d["feet"][1][1]
            fy_l += fl
            fy_r += fr
            tot = abs(fl) + abs(fr)
            if tot > 1e-9 and abs(fr) / tot > 0.8:
                fy_l_only += 1

pitches.sort()
n = len(pitches)
deg = math.degrees
print(f"files {len(files)}  ticks {ticks}")
print(f"root pitch deg: mean {deg(sum(pitches)/n):+.1f}  "
      f"p5 {deg(pitches[n//20]):+.1f}  median {deg(pitches[n//2]):+.1f}  "
      f"p95 {deg(pitches[19*n//20]):+.1f}")
print(f"vertical foot load share: L {fy_l/(fy_l+fy_r)*100:.0f}%  "
      f"R {fy_r/(fy_l+fy_r)*100:.0f}%   "
      f"ticks with R>80% of |load|: {fy_l_only/ticks*100:.0f}%")
print(f"\n{'joint':<14}{'@lo%':>6}{'@hi%':>6}   "
      f"{'coast%':>7}{'+1':>6}{'-1':>6}{'+.25':>6}{'-.25':>6}   menu (Nm)")
for j, (nm, lo, hi, tm) in LEG.items():
    g = gear_use[j]
    print(f"{nm:<14}{at_lo[j]/ticks*100:>6.1f}{at_hi[j]/ticks*100:>6.1f}   "
          f"{g[0]/ticks*100:>7.1f}{g[1]/ticks*100:>6.1f}{g[2]/ticks*100:>6.1f}"
          f"{g[3]/ticks*100:>6.1f}{g[4]/ticks*100:>6.1f}   "
          f"0/±{0.25*tm:.0f}/±{tm:.0f}")
