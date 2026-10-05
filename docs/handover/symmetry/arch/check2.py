import numpy as np, sys
sys.path.insert(0, "/tmp/claude-1000/-home-evolheartrise-git-projects-BeatSaberRL/dcb1a0a2-73f1-4c8b-9ec9-f6f9911b2ca5/scratchpad/sym_arch")
from mirror_ops import *
from beat_arena_rs import RawBeatSim
from beat_arena_env.chart import ChartConfig, generate_chart, note_tuple
from beat_arena_env.mirror import mirror_tuples
perm, sign = obs_mirror_maps()
chart = [note_tuple(n) for n in generate_chart(ChartConfig(chart_len_s=20.0), np.random.default_rng(0))]
for mod in ("beat_arena_rs",):
    A = RawBeatSim(max_steps=700, gears=5, fall_const=5.0, tick_hz=30)
    B = RawBeatSim(max_steps=700, gears=5, fall_const=5.0, tick_hz=30)
    A.set_chart(chart, []); B.set_chart(mirror_tuples(chart), [])
    # also self-mirror test: same chart both, symmetric idle -> state should be self-symmetric
    C = RawBeatSim(max_steps=700, gears=5, fall_const=5.0, tick_hz=30); C.set_chart([], [])
    for t in range(60):
        oa = np.asarray(A.step([0]*29)[0]); ob = np.asarray(B.step([0]*29)[0]); oc = np.asarray(C.step([0]*29)[0])
        d = oa[perm]*sign - ob; dc = oc[perm]*sign - oc
        dc[OFF_NOTES:OFF_NOTE_MASK+24] = 0
        if t in (0, 1, 2, 5, 10, 30, 59):
            idx = np.argsort(-np.abs(d))[:4]
            print(t, "pair", [(int(i), round(float(oa[perm][i]*sign[i]),4), round(float(ob[i]),4)) for i in idx],
                  "| self-sym empty chart max", f"{np.abs(dc).max():.2e} lane {int(np.abs(dc).argmax())}")
