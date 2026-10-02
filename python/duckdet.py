"""Wall/duck analysis over record_rev-2 replays, using the KERNEL's own
collision test (beat_arena_cuda.cu ~L5650, env v36 interval sweep):

  dzf   = (wall.t - time) * speed - PLANE_Z0 - WALL_FRONT_OFF
  z_hit = dzf <= 0 and dzf + len + speed*DT >= 0
  hit   = z_hit and |hx - cx| <= w/2 and |hy - cy| <= h/2

The head is a POINT in the env (no radius). A wall is DUCKABLE when its
bottom edge (cy - h/2) is above 0.9 m, i.e. crouching clears it.
"""
import sys, glob, os, statistics as st

from beat_arena_env.replay_format import read_replay

PLANE_Z0, WALL_FRONT_OFF, DT = -0.65, 0.25, 1.0 / 30.0

def analyze(path):
    head, ticks = read_replay(path)
    walls = head.get("walls") or []
    if not walls:
        return None
    hy_all = [t["head"][1] for t in ticks]
    base = st.median(hy_all)
    rows = []
    for w in walls:
        bottom = w["cy"] - w["h"] / 2.0
        duckable = bottom >= 0.9
        zt = xin = yin = hit = 0
        dips, xs = [], []
        for t in ticks:
            hx, hy = t["head"][0], t["head"][1]
            dzf = (w["t"] - t["time"]) * w["speed"] - PLANE_Z0 - WALL_FRONT_OFF
            if dzf <= 0 and dzf + w["l"] + w["speed"] * DT >= 0:
                zt += 1
                x_ok = abs(hx - w["cx"]) <= w["w"] / 2.0
                y_ok = abs(hy - w["cy"]) <= w["h"] / 2.0
                xin += x_ok; yin += y_ok
                dips.append(hy); xs.append(hx)
                if x_ok and y_ok:
                    hit += 1
        rows.append(dict(bottom=round(bottom, 2), duckable=duckable, z_ticks=zt,
                         x_in=xin, y_in=yin, hit=hit,
                         min_y=round(min(dips), 3) if dips else None,
                         dip=round(base - min(dips), 3) if dips else None,
                         lat=round(min(abs(x - w["cx"]) for x in xs) - w["w"] / 2, 3) if xs else None))
    return dict(file=os.path.basename(path), base=base, rows=rows,
                min_y=min(hy_all), walls_in=sum(t.get("events", {}).get("walls_in", 0)
                                                if isinstance(t.get("events"), dict) else 0
                                                for t in ticks))

if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "*.rpl.gz"
    T = dict(walls=0, duckable=0, zt=0, hit=0, duck_hit=0, ducked=0, cleared_lat=0)
    for f in sorted(glob.glob(pat)):
        r = analyze(f)
        if not r: continue
        for e in r["rows"]:
            T["walls"] += 1; T["duckable"] += e["duckable"]
            if e["z_ticks"]:
                T["zt"] += 1
                if e["hit"]: T["hit"] += 1
                if e["duckable"]:
                    T["duck_hit"] += 1
                    if e["dip"] is not None and e["dip"] > 0.15: T["ducked"] += 1
                    if e["hit"] == 0 and e["lat"] is not None and e["lat"] > 0: T["cleared_lat"] += 1
        print(f"{r['file'][:44]:44} base {r['base']:.3f} min_y {r['min_y']:.3f} | " +
              " ".join(f"[bot{e['bottom']} d{int(e['duckable'])} z{e['z_ticks']} x{e['x_in']} y{e['y_in']} HIT{e['hit']} dip{e['dip']} lat{e['lat']}]"
                       for e in r["rows"][:6]))
    print(f"\nwalls {T['walls']}  reaching-head {T['zt']}  duckable-reaching {T['duck_hit']}  "
          f"HIT {T['hit']}  ducked(>0.15m) {T['ducked']}  cleared-laterally {T['cleared_lat']}")
