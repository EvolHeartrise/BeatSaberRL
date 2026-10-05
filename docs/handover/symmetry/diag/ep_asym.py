import sys,json,numpy as np
sys.path.insert(0,'/home/evolheartrise/git projects/BeatSaberRL/python')
import ep_stats as E
run=sys.argv[1]
R=E.load(run)
lv=json.load(open(run+'/levels.json'))
mir=np.array(lv['chart_mirrored']); lvl=np.array(lv['chart_level'])
ci=R[:,0].astype(int); m=mir[ci]; L=lvl[ci]
upd=R[:,16]
C={c:i for i,c in enumerate(E.COLUMNS)}
ncol=R.shape[1]
print(run, R.shape, 'upd range', np.nanmin(upd), np.nanmax(upd), 'frac mirrored', m.mean())
def parts(sel):
    out={}
    for h in 'lr':
        for f in ('pre','post','prec'):
            out[f+'_'+h]=np.nansum(R[sel,C[f+'_'+h]])/np.nansum(R[sel,C[f+'_max_'+h]])
        out['acc_'+h]=sum(np.nansum(R[sel,C[f+'_'+h]]) for f in ('pre','post','prec'))/sum(np.nansum(R[sel,C[f+'_max_'+h]]) for f in ('pre','post','prec'))
        if ncol>=33:
            out['wrs_'+h]=100*np.nansum(R[sel,C['wreset_'+h]])/np.nansum(R[sel,C['wswing_'+h]])
    return out
umax=np.nanmax(upd)
edges=[0,25,50,100,200,400,800,1200,1600,2000,2400,2800,3200]
edges=[e for e in edges if e<umax]+[umax+1]
print('bin | (R-L)pp: acc pre post prec | orig acc dR-L | mirr acc dR-L | wreset/100sw L R')
for a,b in zip(edges[:-1],edges[1:]):
    s=(upd>=a)&(upd<b)
    if s.sum()<100: continue
    p=parts(s); po=parts(s&(m==0)); pm=parts(s&(m==1))
    d=lambda k,pp=p:100*(pp[k+'_r']-pp[k+'_l'])
    w=f" | {p['wrs_l']:.2f} {p['wrs_r']:.2f}" if 'wrs_l' in p and p['wrs_l']==p['wrs_l'] else ''
    print(f"{int(a):5d}-{int(b):5d} n={s.sum():7d} | {d('acc'):+.3f} {d('pre'):+.3f} {d('post'):+.3f} {d('prec'):+.3f} | orig {d('acc',po):+.3f} | mirr {d('acc',pm):+.3f}{w}")
# last 20%: misses per hand not available; per-level paired orig vs mirror acc
s=upd>=0.8*umax
acc=R[:,8]
import collections
o=collections.defaultdict(list); mm=collections.defaultdict(list)
for i in np.where(s)[0]:
    (mm if m[i] else o)[L[i]].append(acc[i])
common=[k for k in o if k in mm]
dlt=np.array([np.mean(o[k])-np.mean(mm[k]) for k in common])
print('last20%: levels with both', len(common), 'mean acc(orig-mirr) pp', 100*dlt.mean(), '+-', 100*dlt.std()/np.sqrt(len(dlt)), 'median', 100*np.median(dlt))
p=parts(s)
print('last20% parts', {k:round(v,5) for k,v in p.items()})
# split by map's own hand: on orig charts, L hand = red notes; on mirrored, L hand = blue(originally) 
po=parts(s&(m==0)); pm=parts(s&(m==1))
print('orig  ', {k:round(v,5) for k,v in po.items()})
print('mirr  ', {k:round(v,5) for k,v in pm.items()})
# physical hand effect = avg over orig/mirr of (R-L); map (color) effect = avg of (red-blue): orig red=L, mirr red=R
phys=0.5*((po['acc_r']-po['acc_l'])+(pm['acc_r']-pm['acc_l']))
color=0.5*((po['acc_l']-po['acc_r'])+(pm['acc_r']-pm['acc_l']))
print(f'physical(R-L) {100*phys:+.3f}pp   map-colour(red-blue) {100*color:+.3f}pp')
