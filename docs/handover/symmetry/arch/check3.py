import numpy as np, sys, torch
sys.path.insert(0, "/tmp/claude-1000/-home-evolheartrise-git-projects-BeatSaberRL/dcb1a0a2-73f1-4c8b-9ec9-f6f9911b2ca5/scratchpad/sym_arch")
from mirror_ops import *
from beat_arena_env.chart import ChartConfig, generate_chart, note_tuple
from beat_arena_env.mirror import mirror_tuples
perm, sign = obs_mirror_maps()
chart = [note_tuple(n) for n in generate_chart(ChartConfig(chart_len_s=20.0), np.random.default_rng(0))]
import beat_arena_rs, beat_arena_rs32
for mod in (beat_arena_rs, beat_arena_rs32):
    A = mod.RawBeatSim(max_steps=700, gears=5, fall_const=5.0, tick_hz=30)
    B = mod.RawBeatSim(max_steps=700, gears=5, fall_const=5.0, tick_hz=30)
    A.set_chart(chart, []); B.set_chart(mirror_tuples(chart), [])
    oa = np.asarray(A.step([0]*29)[0]); ob = np.asarray(B.step([0]*29)[0])
    d = np.abs(oa[perm]*sign - ob); print(mod.__name__, "1 coast tick: max", d.max(), "lanes>1e-4:", np.nonzero(d > 1e-4)[0].tolist())
# policy: symmetrized wrapper equivariance check on a random net
from policy import NotePolicy
torch.manual_seed(0)
pol = NotePolicy(embed=64, trunk_dim=64, n_gears=5, note_geo=True).eval()
K = pol.k_max; hp, gid = act_mirror_maps(K)
P = torch.tensor(np.stack([hp[j]*K + gid[j] for j in range(29)]).reshape(-1))  # flat gather
permt, signt = torch.tensor(perm), torch.tensor(sign, dtype=torch.float32)
def Mo(o): return o[:, permt] * signt
def Pl(l): return l.reshape(l.shape[0], -1)[:, P].view_as(l)
def sym(o, lam=0.0):
    l, v = pol.forward_stacked(torch.cat([o, Mo(o)]))
    B = o.shape[0]
    return 0.5*(1+lam)*l[:B] + 0.5*(1-lam)*Pl(l[B:]), 0.5*(v[:B]+v[B:])
oa = torch.tensor(np.stack([oa, ob]), dtype=torch.float32)
with torch.no_grad():
    l, v = pol.forward_stacked(oa); lm, vm = pol.forward_stacked(Mo(oa))
    print("raw net: |l(Ms) - P l(s)| max", (lm - Pl(l)).abs().max().item(), " |v(Ms)-v(s)|", (vm-v).abs().max().item())
    l, v = sym(oa); lm, vm = sym(Mo(oa))
    print("sym net: |l(Ms) - P l(s)| max", (lm - Pl(l)).abs().max().item(), " |v(Ms)-v(s)|", (vm-v).abs().max().item())
