"""Mirror operators for obs (630) and actions (29 heads), + sim check."""
import numpy as np, sys
sys.path.insert(0, "/home/evolheartrise/git projects/BeatSaberRL/python")
from beat_arena_env.constants import *

def obs_mirror_maps():
    perm = np.arange(OBS_DIM); sign = np.ones(OBS_DIM)
    def flip(i): sign[i] = -1
    def swap(a, b, n):
        for k in range(n): perm[a+k], perm[b+k] = b+k, a+k
    flip(0)                                    # root x
    for i, s in zip(range(3, 9), [1,-1,-1,-1,1,1]): sign[i] = s   # M R M
    flip(9); flip(13); flip(14)                # vlin x; vang y,z (axial)
    # q: spine 15..17
    flip(16); flip(17)
    armq = [1,-1,-1,1,1,-1,1,-1]               # pitch yaw twist elbow cos sin bend dev
    legf = [1,-1,-1,1,1,-1]
    armqd = [1,-1,-1,1,-1,1,-1]
    for base, pat in [(18,armq),(26,armq),(34,legf),(40,legf),
                      (49,armqd),(56,armqd),(63,legf),(69,legf)]:
        for k,s in enumerate(pat): sign[base+k] = s
    flip(47); flip(48)
    swap(18,26,8); swap(34,40,6); swap(49,56,7); swap(63,69,6)
    for b in (75, 86): flip(b); flip(b+3); flip(b+6)   # wrist pos/vel/blade x
    swap(75, 86, 11)
    flip(97); swap(100, 101, 1)
    for s_ in range(N_NOTE_SLOTS):
        b = OFF_NOTES + s_*NOTE_FEAT
        flip(b); flip(b+6); swap(b+4, b+5, 1)
    swap(OFF_GROUND_IMP, OFF_GROUND_IMP+2, 2)
    for off in (OFF_BODY_PTS, OFF_BODY_PT_VEL):
        for p in range(6): flip(off+3*p)
        for p in (0, 2, 4): swap(off+3*p, off+3*p+3, 3)
    for b in (OFF_PREV_L, OFF_PREV_R): flip(b); flip(b+2)
    swap(OFF_PREV_L, OFF_PREV_R, 5)
    for b in (OFF_TIP_VEL, OFF_TIP_VEL+3): flip(b)
    swap(OFF_TIP_VEL, OFF_TIP_VEL+3, 3)
    return perm, sign     # mirrored = obs[perm] * sign

FLIP_JOINTS = (1,2, 4,5,7,9, 11,12,14,16, 18,19,22, 24,25,28)
def act_mirror_maps(k_max):
    hp = list(range(3)) + list(range(10,17)) + list(range(3,10)) \
         + list(range(23,29)) + list(range(17,23))
    gid = np.tile(np.arange(k_max), (N_JOINTS, 1))
    sw = np.arange(k_max); sw[1:] = [i+1 if i % 2 else i-1 for i in range(1, k_max)]
    for j in FLIP_JOINTS: gid[j] = sw
    return np.array(hp), gid  # mirrored action: a'[j] = gid[j][a[hp[j]]]

if __name__ == "__main__":
    from beat_arena_rs import RawBeatSim
    from beat_arena_env.chart import ChartConfig, generate_chart, note_tuple
    from beat_arena_env.mirror import mirror_tuples
    perm, sign = obs_mirror_maps()
    assert (perm[perm] == np.arange(OBS_DIM)).all()
    K = 5; hp, gid = act_mirror_maps(K)
    rng = np.random.default_rng(0)
    chart = [note_tuple(n) for n in generate_chart(ChartConfig(chart_len_s=20.0), rng)]
    A = RawBeatSim(max_steps=700, gears=K, fall_const=5.0, tick_hz=30)
    B = RawBeatSim(max_steps=700, gears=K, fall_const=5.0, tick_hz=30)
    oa = np.asarray(A.set_chart(chart, [])); ob = np.asarray(B.set_chart(mirror_tuples(chart), []))
    print("t=0 max|M oa - ob|", np.abs(oa[perm]*sign - ob).max())
    for t in range(400):
        a = rng.integers(0, K, N_JOINTS) if t % 3 else np.zeros(N_JOINTS, int)
        a[17:] = 0  # keep legs coasting
        am = np.array([gid[j][a[hp[j]]] for j in range(N_JOINTS)])
        oa, ra, da, _ = A.step([int(x) for x in a]); ob, rb, db, _ = B.step([int(x) for x in am])
        d = np.abs(np.asarray(oa)[perm]*sign - np.asarray(ob))
        if t in (0, 1, 5, 30, 100, 200, 399) or da or db:
            i = int(d.argmax()); print(t, f"max {d.max():.3e} at lane {i}", f"dr {abs(ra-rb):.2e}", da, db)
        if da or db: break
