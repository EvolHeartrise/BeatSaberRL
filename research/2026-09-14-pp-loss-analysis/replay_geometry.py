"""Is the 15-pt accuracy component lost to BIAS or VARIANCE?

For every good cut in a replay: reconstruct the note centre at the cut
tick (z = PLANE_Z0 + (t_note - time) * speed, x/y from the chart), take the
cutting saber's line (base + dir from the `sabers` record, 1 m blade) and
measure the perpendicular offset of the note centre from that line
(the cut-plane distance the 15-pt component is paid on, up to the
plane-vs-line approximation).  Report signed x/y offsets (bias), their
spread (variance), reach (shoulder->saber base distance) and blade tip
z at the cut, split by row / column / crossover / hand.

usage: python3 replay_geometry.py <replay.jsonl> ...
"""
import glob, json, os, sys, collections
import numpy as np

DT = 1.0 / 30.0
PLANE_Z0 = -0.65
SHOULDER = {0: np.array([-0.2, 1.47, -0.65]), 1: np.array([0.2, 1.47, -0.65])}
LANE_X = [-0.9, -0.3, 0.3, 0.9]; ROW_Y = [0.85, 1.40, 1.90]


def nearest(v, opts):
    return int(np.argmin([abs(v - o) for o in opts]))


def run(files):
    rows = []
    for f in files:
        with open(f) as fh:
            head = json.loads(fh.readline())["header"]
            chart = head["chart"]
            for line in fh:
                r = json.loads(line)
                for ev in r["events"]:
                    if "cut" not in ev or not ev["cut"]["good"]:
                        continue
                    c = ev["cut"]; nt = chart[c["note"]]
                    if nt["kind"] != 0:
                        continue
                    hand = nt["color"]
                    s = np.array(r["sabers"][hand][:3]); d = np.array(r["sabers"][hand][3:6])
                    d = d / (np.linalg.norm(d) + 1e-9)
                    z = PLANE_Z0 + (nt["t"] - r["time"]) * nt["speed"]
                    cen = np.array([nt["x"], nt["y"], z])
                    v = cen - s
                    along = float(v @ d)
                    perp = v - along * d
                    reach = float(np.linalg.norm(s - SHOULDER[hand]))
                    rows.append(dict(acc=c["acc"], pre=c["pre"], dist=float(np.linalg.norm(perp)),
                                     px=float(perp[0]), py=float(perp[1]), pz=float(perp[2]), along=along,
                                     reach=reach, tip_z=float(s[2] + d[2]), note_z=z,
                                     row=nearest(nt["y"], ROW_Y), col=nearest(nt["x"], LANE_X), hand=hand,
                                     cross=(hand == 0 and nt["x"] > 0.31) or (hand == 1 and nt["x"] < -0.31),
                                     early=(r["time"] - nt["t"]) / DT, speed=nt["speed"],
                                     head_x=r["head"][0], head_y=r["head"][1]))
    A = np.array([r["acc"] for r in rows]); D = np.array([r["dist"] for r in rows])
    print(f"good plain cuts {len(rows)}; corr(acc pts, line-offset dist) = {np.corrcoef(A, D)[0,1]:+.3f} "
          f"(expected strongly negative: acc = 15*(1 - dist/0.3) up to the plane-vs-line approximation)")
    implied = 0.3 * (1 - A / 15.0)
    print(f"implied cut-plane distance from acc pts: mean {100*implied.mean():.1f} cm, median {100*np.median(implied):.1f} cm, "
          f"p90 {100*np.percentile(implied,90):.1f} cm; measured line offset mean {100*D.mean():.1f} cm")
    lost = 15 - A
    tail = lost[A <= 8].sum() / lost.sum()
    print(f"share of acc-points lost on cuts with acc <= 8 (>= 14 cm off): {100*tail:.0f}% "
          f"({100*(A<=8).mean():.1f}% of cuts); on cuts with acc >= 12 (<= 6 cm): {100*lost[A>=12].sum()/lost.sum():.0f}% "
          f"({100*(A>=12).mean():.1f}% of cuts)")
    print(f"reach (shoulder->saber base) mean {np.mean([r['reach'] for r in rows]):.3f} m, p10 "
          f"{np.percentile([r['reach'] for r in rows],10):.3f}, p90 {np.percentile([r['reach'] for r in rows],90):.3f}; "
          f"note z at cut mean {np.mean([r['note_z'] for r in rows]):+.2f} m (hit plane {PLANE_Z0}); blade tip z mean "
          f"{np.mean([r['tip_z'] for r in rows]):+.2f} m")

    def grp(name, keyf):
        g = collections.defaultdict(list)
        for r in rows:
            g[keyf(r)].append(r)
        print(f"\n| {name} | cuts | acc pts | mean signed x off (cm) | mean signed y off (cm) | sd x | sd y | |bias|/rms | reach m | early ticks |")
        print("|" + "---|" * 10)
        for k in sorted(g):
            s = g[k]
            px = np.array([r["px"] for r in s]); py = np.array([r["py"] for r in s])
            rms = np.sqrt((px**2 + py**2).mean()); bias = np.hypot(px.mean(), py.mean())
            print(f"| {k} | {len(s)} | {np.mean([r['acc'] for r in s]):.2f} | {100*px.mean():+.1f} | {100*py.mean():+.1f} | "
                  f"{100*px.std():.1f} | {100*py.std():.1f} | {bias/rms:.2f} | {np.mean([r['reach'] for r in s]):.3f} | "
                  f"{np.mean([r['early'] for r in s]):+.2f} |")

    grp("row", lambda r: r["row"]); grp("col", lambda r: r["col"]); grp("hand", lambda r: r["hand"])
    grp("crossover", lambda r: "cross" if r["cross"] else "no")
    grp("reach bin (m)", lambda r: f"{np.floor(r['reach']*10)/10:.1f}")
    grp("acc bin", lambda r: "0-5" if r["acc"] < 6 else ("6-10" if r["acc"] < 11 else ("11-13" if r["acc"] < 14 else "14-15")))
    # acc vs reach correlation and vs earliness
    R = np.array([r["reach"] for r in rows]); E = np.array([r["early"] for r in rows])
    print(f"\ncorr(acc, reach) = {np.corrcoef(A, R)[0,1]:+.3f}; corr(acc, earliness) = {np.corrcoef(A, E)[0,1]:+.3f}; "
          f"corr(acc, |x off|) = {np.corrcoef(A, np.abs([r['px'] for r in rows]))[0,1]:+.3f}; "
          f"corr(acc, |y off|) = {np.corrcoef(A, np.abs([r['py'] for r in rows]))[0,1]:+.3f}")
    # autocorrelation of acc along the sequence (is a bad cut followed by bad cuts?)
    if len(A) > 10:
        a = A - A.mean()
        print(f"lag-1 autocorr of acc pts along the play: {np.corrcoef(a[:-1], a[1:])[0,1]:+.3f}")


if __name__ == "__main__":
    files = []
    for a in sys.argv[1:]:
        files += sorted(glob.glob(a))
    run(files)
