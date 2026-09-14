"""Per-note cut-quality analysis over v10 replays (record_rev 2/3 JSONL).

For every scoreable chart note: was it cut (good/bad), missed, or never
reached (early termination)?  For good cuts: pre / acc / post points, cut
earliness (cut tick vs chart hit time), and chart context (row, column,
color/hand, angle, dot, double, stack, crossover, in-flight density).
Also: bombs hit, wall entries, termination point.

usage: python3 replay_cuts.py <replay.jsonl> ...   (glob patterns ok)
"""
import collections, glob, json, math, os, sys
import numpy as np

DT = 1.0 / 30.0
LANE_X = [-0.9, -0.3, 0.3, 0.9]
ROW_Y = [0.85, 1.40, 1.90]


def nearest(v, opts):
    return int(np.argmin([abs(v - o) for o in opts]))


def load(path):
    with open(path) as fh:
        head = json.loads(fh.readline())["header"]
        ticks = [json.loads(l) for l in fh]
    return head, ticks


def analyze(path):
    head, ticks = load(path)
    chart = head["chart"]; walls = head.get("walls") or []
    n = len(chart)
    notes = []
    for i, c in enumerate(chart):
        notes.append(dict(i=i, t=c["t"], x=c["x"], y=c["y"], color=c["color"], angle=c["angle"],
                          speed=c["speed"], any=c["any"], kind=c["kind"],
                          col=nearest(c["x"], LANE_X), row=nearest(c["y"], ROW_Y),
                          status="unreached", pre=None, acc=None, post=None, pts=None,
                          cut_tick=None, early=None, reason=None))
    # doubles / stacks / crossovers
    by_t = collections.defaultdict(list)
    for nt in notes:
        by_t[round(nt["t"], 3)].append(nt)
    for grp in by_t.values():
        sc = [g for g in grp if g["kind"] in (0, 2)]
        for g in sc:
            g["double"] = len(sc) > 1
            g["stack"] = any(o is not g and o["col"] == g["col"] for o in sc)
    for nt in notes:
        nt.setdefault("double", False); nt.setdefault("stack", False)
        nt["cross"] = (nt["color"] == 0 and nt["x"] > 0.31) or (nt["color"] == 1 and nt["x"] < -0.31)
    # local density: scoreable notes within ±0.5 s
    ts = np.array([nt["t"] for nt in notes if nt["kind"] != 1])
    for nt in notes:
        nt["dens1s"] = int(((ts > nt["t"] - 0.5) & (ts < nt["t"] + 0.5)).sum())
    wall_enter = 0; bombs = 0; bad = 0; miss = 0; falls = 0
    for r in ticks:
        for ev in r["events"]:
            if "cut" in ev:
                c = ev["cut"]; nt = notes[c["note"]]
                nt["cut_tick"] = r["t"]; nt["reason"] = c.get("reason")
                if c["good"]:
                    nt["status"] = "good"; nt["pre"] = c["pre"]; nt["acc"] = c["acc"]; nt["pts"] = c["points"]
                    nt["early"] = (r["time"] - nt["t"]) / DT
                else:
                    nt["status"] = "bad"; bad += 1
            elif "post" in ev:
                p = ev["post"]; nt = notes[p["note"]]
                nt["post"] = p["points"]
            elif "miss" in ev:
                notes[ev["miss"]["note"]]["status"] = "miss"; miss += 1
            elif "bomb" in ev:
                bombs += 1
            elif "wall_enter" in ev or "wall" in ev:
                wall_enter += 1
            elif "fall" in ev or "fell" in ev:
                falls += 1
    last_t = ticks[-1]["time"]
    last_note = max(nt["t"] for nt in notes)
    for nt in notes:
        if nt["status"] == "good" and nt["post"] is None and nt["kind"] == 0:
            nt["post"] = 0.0   # follow-through never resolved (termination / no arc)
    return head, notes, dict(ticks=len(ticks), last_t=last_t, last_note=last_note,
                             terminated=last_t < last_note - 0.5, wall_enter=wall_enter,
                             bombs=bombs, bad=bad, miss=miss, falls=falls)


def summarize(files):
    allnotes = []; per = []
    for f in files:
        head, notes, info = analyze(f)
        sc = [nt for nt in notes if nt["kind"] != 1]
        good = [nt for nt in sc if nt["status"] == "good"]
        per.append((os.path.basename(f)[:48], head["map"]["difficulty"], info, len(sc), len(good),
                    sum(1 for nt in sc if nt["status"] == "unreached"),
                    np.mean([nt["pre"] for nt in good]) if good else 0,
                    np.mean([nt["acc"] for nt in good]) if good else 0,
                    np.mean([nt["post"] for nt in good if nt["kind"] == 0]) if good else 0,
                    np.mean([nt["early"] for nt in good]) if good else 0))
        for nt in sc:
            nt["file"] = os.path.basename(f)
        allnotes += sc
    print("| replay | diff | ticks | ended at (s) | last note (s) | term? | scoreable | good | bad | miss | unreached | bombs | walls | pre | acc | post | early (ticks) |")
    print("|" + "---|" * 17)
    for (nm, dif, info, nsc, ngood, nunr, pre, acc, post, early) in per:
        print(f"| {nm} | {dif[:6]} | {info['ticks']} | {info['last_t']:.1f} | {info['last_note']:.1f} | "
              f"{'YES' if info['terminated'] else 'no'} | {nsc} | {ngood} | {info['bad']} | {info['miss']} | {nunr} | "
              f"{info['bombs']} | {info['wall_enter']} | {pre:.2f} | {acc:.2f} | {post:.2f} | {early:+.2f} |")
    good = [nt for nt in allnotes if nt["status"] == "good"]
    plain = [nt for nt in good if nt["kind"] == 0]
    print(f"\nscoreable notes {len(allnotes)}: good {len(good)} bad {sum(nt['status']=='bad' for nt in allnotes)} "
          f"miss {sum(nt['status']=='miss' for nt in allnotes)} unreached {sum(nt['status']=='unreached' for nt in allnotes)}")
    if not plain:
        return
    tot_loss = sum((70 - nt["pre"]) + (15 - nt["acc"]) + (30 - nt["post"]) for nt in plain)
    print(f"plain-note good cuts {len(plain)}: mean pre {np.mean([nt['pre'] for nt in plain]):.2f} / acc "
          f"{np.mean([nt['acc'] for nt in plain]):.2f} / post {np.mean([nt['post'] for nt in plain]):.2f}; "
          f"points lost per cut: pre {np.mean([70-nt['pre'] for nt in plain]):.2f} acc "
          f"{np.mean([15-nt['acc'] for nt in plain]):.2f} post {np.mean([30-nt['post'] for nt in plain]):.2f} "
          f"(shares {100*sum(70-nt['pre'] for nt in plain)/tot_loss:.0f}% / "
          f"{100*sum(15-nt['acc'] for nt in plain)/tot_loss:.0f}% / {100*sum(30-nt['post'] for nt in plain)/tot_loss:.0f}%)")
    accs = np.array([nt["acc"] for nt in plain])
    print("acc-component histogram (15 = perfect centre cut, 0.3 m normalizer):")
    hist = collections.Counter(int(round(a)) for a in accs)
    print("  " + "  ".join(f"{k}:{100*hist[k]/len(plain):.1f}%" for k in sorted(hist)))
    early = np.array([nt["early"] for nt in good])
    print(f"cut earliness (ticks, negative = before chart hit time): mean {early.mean():+.2f}, p10 {np.percentile(early,10):+.1f}, "
          f"p50 {np.percentile(early,50):+.1f}, p90 {np.percentile(early,90):+.1f}; share early {100*(early<-0.5).mean():.0f}%")
    # acc vs earliness
    print("| earliness bin (ticks) | cuts | mean acc pts | mean pre | mean post |")
    print("|---|---|---|---|---|")
    for lo, hi in [(-9, -5), (-5, -4), (-4, -3), (-3, -2), (-2, -1), (-1, 0), (0, 3)]:
        s = [nt for nt in plain if lo <= nt["early"] < hi]
        if s:
            print(f"| [{lo},{hi}) | {len(s)} | {np.mean([nt['acc'] for nt in s]):.2f} | {np.mean([nt['pre'] for nt in s]):.2f} | {np.mean([nt['post'] for nt in s]):.2f} |")

    def bucket(name, keyf, keys=None):
        print(f"\n| {name} | notes | good% | miss% | bad% | unreached% | acc pts | pre | post | pts lost/cut | early |")
        print("|" + "---|" * 11)
        groups = collections.defaultdict(list)
        for nt in allnotes:
            groups[keyf(nt)].append(nt)
        for k in (keys or sorted(groups)):
            g = groups.get(k, [])
            if not g:
                continue
            gg = [nt for nt in g if nt["status"] == "good" and nt["kind"] == 0]
            print(f"| {k} | {len(g)} | {100*sum(nt['status']=='good' for nt in g)/len(g):.1f} | "
                  f"{100*sum(nt['status']=='miss' for nt in g)/len(g):.2f} | {100*sum(nt['status']=='bad' for nt in g)/len(g):.2f} | "
                  f"{100*sum(nt['status']=='unreached' for nt in g)/len(g):.1f} | "
                  + (f"{np.mean([nt['acc'] for nt in gg]):.2f} | {np.mean([nt['pre'] for nt in gg]):.2f} | "
                     f"{np.mean([nt['post'] for nt in gg]):.2f} | {np.mean([115-nt['pts']-nt['post'] for nt in gg]):.2f} | "
                     f"{np.mean([nt['early'] for nt in gg]):+.2f} |" if gg else "- | - | - | - | - |"))

    bucket("row (0 bottom,1 mid,2 top)", lambda nt: nt["row"])
    bucket("column (0 far-left .. 3 far-right)", lambda nt: nt["col"])
    bucket("row,col", lambda nt: f"r{nt['row']}c{nt['col']}")
    bucket("hand (color 0=left/red, 1=right/blue)", lambda nt: nt["color"])
    bucket("direction (deg, 0=down-cut? chart angle bins)", lambda nt: "dot" if nt["any"] else f"{int(round(math.degrees(nt['angle'])/45)*45)%360:03d}")
    bucket("double note", lambda nt: "double" if nt["double"] else "single")
    bucket("stack", lambda nt: "stack" if nt["stack"] else "no")
    bucket("crossover", lambda nt: "cross" if nt["cross"] else "no")
    bucket("notes within ±0.5 s", lambda nt: min(nt["dens1s"], 9))
    bucket("note speed (NJS m/s)", lambda nt: int(nt["speed"]))
    bucket("kind (0 note, 2 chain head, 3 link)", lambda nt: nt["kind"])
    # the 15-pt acc component vs x offset from lane centre / y
    print("\nacc-component by lateral hand side: left-hand notes on the right side (cross) etc. above; "
          "worst (row,col,hand) cells by points lost per cut (>= 30 cuts):")
    cells = collections.defaultdict(list)
    for nt in plain:
        cells[(nt["row"], nt["col"], nt["color"])].append(115 - nt["pts"] - nt["post"])
    for k, v in sorted(cells.items(), key=lambda kv: -np.mean(kv[1]))[:8]:
        if len(v) >= 30:
            print(f"  row{k[0]} col{k[1]} hand{k[2]}: {len(v)} cuts, {np.mean(v):.2f} pts lost/cut")


if __name__ == "__main__":
    files = []
    for a in sys.argv[1:]:
        files += sorted(glob.glob(a))
    summarize(files)
