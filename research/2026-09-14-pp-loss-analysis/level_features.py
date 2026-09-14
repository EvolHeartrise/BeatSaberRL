"""Per-level chart features from the bench chart_cache (kinds, NJS, NPS,
walls, geometry) -> level_features.json keyed by "<hash>_<difficulty>".
Run from anywhere; paths are relative to the beat-arena project dir."""
import glob, json, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(PROJ, "python", "maps", "chart_cache")
OUT = os.path.join(HERE, "level_features.json")

KIND_NOTE, KIND_BOMB, KIND_HEAD, KIND_LINK = 0, 1, 2, 3
MAXPTS = {KIND_NOTE: 115.0, KIND_HEAD: 85.0, KIND_LINK: 20.0}


def features(pkl):
    d = pickle.load(open(pkl, "rb"))
    tup = sorted(d["tuples"], key=lambda t: t[0])
    if not tup:
        return None
    t = np.array([x[0] for x in tup]); x = np.array([v[1] for v in tup])
    y = np.array([v[2] for v in tup]); col = np.array([v[3] for v in tup])
    ang = np.array([v[4] for v in tup]); spd = np.array([v[5] for v in tup])
    anyd = np.array([v[6] for v in tup]); kind = np.array([v[7] for v in tup])
    sc = kind != KIND_BOMB
    n_sc = int(sc.sum())
    ts = t[sc]
    dur = float(ts[-1] - ts[0]) if n_sc > 1 else 1.0
    # peak density: max scoreable notes in any 5 s window
    peak5 = 0
    if n_sc:
        j = 0
        for i in range(n_sc):
            while ts[i] - ts[j] > 5.0:
                j += 1
            peak5 = max(peak5, i - j + 1)
    # doubles / stacks / crossovers among plain notes+heads
    same_t = np.zeros(n_sc, bool)
    if n_sc > 1:
        dt = np.diff(ts)
        eq = dt < 1e-3
        same_t[1:] |= eq; same_t[:-1] |= eq
    xs, cs = x[sc], col[sc]
    cross = ((cs == 0) & (xs > 0.31)) | ((cs == 1) & (xs < -0.31))
    ys = y[sc]
    rows = [int(((ys > 1.1) & (ys < 1.65)).sum()), int((ys >= 1.65).sum())]
    max_unmult = float(sum(MAXPTS.get(int(k), 0.0) for k in kind))
    return dict(
        n_obj=len(tup), n_sc=n_sc, n_bomb=int((~sc).sum()),
        n_head=int((kind == KIND_HEAD).sum()), n_link=int((kind == KIND_LINK).sum()),
        n_walls=len(d.get("walls", [])),
        njs=float(np.median(spd[sc])) if n_sc else 0.0,
        nps=n_sc / max(dur, 1.0), peak_nps5=peak5 / 5.0,
        span_s=dur, dur_s=float(d.get("duration_s", 0.0)),
        frac_double=float(same_t.mean()) if n_sc else 0.0,
        frac_cross=float(cross.mean()) if n_sc else 0.0,
        frac_dot=float(anyd[sc].mean()) if n_sc else 0.0,
        frac_mid=rows[0] / max(n_sc, 1), frac_top=rows[1] / max(n_sc, 1),
        max_unmult=max_unmult, max_score=d.get("max_score"),
        max_pre=70.0 * (int((kind == KIND_NOTE).sum()) + int((kind == KIND_HEAD).sum())),
        max_acc=15.0 * (int((kind == KIND_NOTE).sum()) + int((kind == KIND_HEAD).sum())),
        max_post=30.0 * int((kind == KIND_NOTE).sum()),
    )


def main():
    out = {}
    files = sorted(glob.glob(os.path.join(CACHE, "*.pkl")))
    for i, f in enumerate(files):
        base = os.path.basename(f)[:-4]
        h, diff, _ver = base.split("_")
        r = features(f)
        if r:
            out[f"{h}_{diff}"] = r
        if (i + 1) % 1000 == 0:
            print(i + 1, "/", len(files), file=sys.stderr)
    json.dump(out, open(OUT, "w"))
    print("wrote", OUT, len(out))


if __name__ == "__main__":
    main()
