"""Replay impact: how often v61/v64/v65 sit outside the proposed human axial
range, and what passive torque they would have felt (open-loop: the
recorded poses re-scored, no re-simulation)."""
import numpy as np

from load_replays import load
from srom_model import ShoulderROM, anat

z = load()
ver = np.where(z["ver"] == "v61all", "v61", z["ver"])
q, w = z["q"], z["w"]
pitch, yaw, tw = q[:, 0], q[:, 1], q[:, 2]
elb = np.degrees(q[:, 3])


def report(m, label):
    E, (ty, tp, tt), u, zz = m.eval(yaw, pitch, tw)
    b = m.damping(u)
    wtau = w[:, 1] - w[:, 2]
    theta, phi, tau, psi = anat(yaw, pitch, tw)
    print(f"##### {label}: T_peak {m.T:.3f} N.m  s {m.s}  kb {m.kb} wb {m.wb}  b_pk {m.bpk:.3f}")
    rows = []
    for v in ("v61", "v64", "v65"):
        k = ver == v
        cap = np.abs(w[k, 2]) >= 15.9
        st = elb[k] < 15
        r = dict(
            v=v,
            zone=np.mean(zz[k] > 0) * 100,
            beyond=np.mean(np.abs(u[k]) > 1) * 100,
            er=np.mean(u[k] < -1) * 100,
            ir=np.mean(u[k] > 1) * 100,
            far=np.mean(np.abs(u[k]) > 1.25) * 100,
            tt_mean=np.mean(np.abs(tt[k])),
            tt_p95=np.percentile(np.abs(tt[k]), 95),
            tt_p99=np.percentile(np.abs(tt[k]), 99),
            tp_p99=np.percentile(np.abs(tp[k]), 99),
            ty_p99=np.percentile(np.abs(ty[k]), 99),
            beyond_cap=np.mean(np.abs(u[k][cap]) > 1) * 100,
            beyond_straight=np.mean(np.abs(u[k][st]) > 1) * 100,
            pdiss=np.mean(b[k] * wtau[k] ** 2),
            umax=np.percentile(np.abs(u[k]), 99.9),
        )
        rows.append(r)
        print(f"  {v}: in zone {r['zone']:5.1f}%  beyond human limit {r['beyond']:5.2f}% "
              f"(ER {r['er']:.2f}% / IR {r['ir']:.2f}%), >25% past {r['far']:.2f}%  "
              f"|u| p99.9 {r['umax']:.2f}")
        print(f"       twist-row passive |tau|: mean {r['tt_mean']:.3f}  p95 {r['tt_p95']:.2f}  "
              f"p99 {r['tt_p99']:.2f} N.m;  pitch p99 {r['tp_p99']:.2f}  yaw p99 {r['ty_p99']:.2f}")
        print(f"       beyond while twist at cap {r['beyond_cap']:.2f}%, while elbow<15 "
              f"{r['beyond_straight']:.2f}%;  edge-damping power (open loop) {r['pdiss']:.3f} W")
        L = m.limits(yaw[k], pitch[k])
        w_half = 0.5 * (L["hi"][0] - L["lo"][0])
        past = np.maximum(np.abs(u[k]) - 1, 0) * np.degrees(w_half)
        b_ = past > 0
        print(f"       degrees past the limit when beyond: median {np.median(past[b_]):.1f}  "
              f"p90 {np.percentile(past[b_], 90):.1f}  max {past.max():.1f};  "
              f"|tau_t| > twist taumax (16.98): {np.mean(np.abs(tt[k]) > 16.98) * 100:.2f}% of ticks")
        th = theta[k]
        line = []
        for lo_, hi_ in ((0, 45), (45, 60), (60, 75), (75, 90), (90, 160)):
            sel = (th >= lo_) & (th < hi_)
            if sel.sum() > 100:
                line.append(f"{lo_}-{hi_}: {np.mean(np.abs(u[k][sel]) > 1) * 100:4.1f}% "
                            f"(of {sel.mean() * 100:4.1f}% ticks)")
        print("       beyond by elevation: " + "; ".join(line))
        per = [np.mean(np.abs(u[k][z["rid"][k] == r]) > 1) * 100 for r in sorted(set(z["rid"][k]))]
        print(f"       per-replay beyond%: median {np.median(per):.1f}  min {min(per):.1f}  max {max(per):.1f};"
              f"  L {np.mean(np.abs(u[k][z['side'][k] == 0]) > 1) * 100:.1f}%  "
              f"R {np.mean(np.abs(u[k][z['side'][k] == 1]) > 1) * 100:.1f}%")
        # today's box vs the human window: where the policy's tau sits
        print(f"       tau (IR+) p5/p50/p95: {np.percentile(tau[k], [5, 50, 95]).round(1)};  "
              f"human ER limit at those poses p50: {np.degrees(np.median(L['lo'][0])):.1f}")
    return rows


if __name__ == "__main__":
    report(ShoulderROM(), "default")
