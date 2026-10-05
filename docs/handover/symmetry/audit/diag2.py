import numpy as np
from test_mirror import *
np.set_printoptions(precision=4, linewidth=220, suppress=True)
A = B.RawBeatSim(max_steps=10000, gears=5); A.set_chart([], [])
q0 = np.asarray(A.qpos())
print("reset legs L", q0[7+17:7+23], "\n reset legs R", q0[7+23:7+29], "root", q0[:7])
z = [0]*NQ
for t in range(6):
    o, r, d, e = A.step(z)
    q = np.asarray(A.qpos()); mq = m_qpos(q)
    print(f"t{t} legL", q[24:30], " legR(mirrored to L)", mq[24:30], " root x", q[0], "quat", q[3:7])
    print("    groundimp [Lh Lt Rh Rt]", np.asarray(o)[558:562], " GRF", np.asarray(A.ground_reaction()))
