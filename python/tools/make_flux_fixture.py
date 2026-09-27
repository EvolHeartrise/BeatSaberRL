"""Scripted Flux replay for the viewer's fixtures (SPEC.md §Flux).

The body plays a deterministic square-wave arm pattern with its trunk
PINNED: after every tick the root, spine and legs are written back to the
reset stance at rest (`pin_trunk`, via set_joints), so the body never
falls while the arms move freely. (The v10 weld harness used to provide
this platform; it was removed from the engines 2026-09-27.) A first pass
records the palm path; the chart
is then built ON that path — circles at a palm, bars across the Flux
with random sweet spots (every other one turned 75 deg off square, a
bad "angle" cut), a wrong-hand circle and a few deliberate misses — and a second pass records the episode. Objects never touch the
body, so the second pass follows the same trajectory and the placed
objects really are touched.

Run: python tools/make_flux_fixture.py OUT.jsonl [--ticks 300] [--seed 0]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beat_arena_env.constants import KIND_BAR, KIND_CIRCLE, N_JOINTS, PLANE_Z0
from beat_arena_env.flux_chart import FluxObj
from beat_arena_env.flux_env import RustFluxEnv

DT = 1.0 / 30.0
SPEED = 12.0
LEAD = 0.2  # object centre this far beyond the palm at its tick (m)


def arm_actions(t):
    """Square-wave gears on shoulder pitch/yaw + elbow, right arm phase-
    shifted, everything else coasting."""
    a = [0] * N_JOINTS
    for base, phase in ((3, 0), (10, 7)):
        a[base] = 1 if ((t + phase) // 14) % 2 == 0 else 2       # pitch
        a[base + 1] = 1 if ((t + phase) // 19) % 2 == 0 else 2   # yaw
        a[base + 3] = 1 if ((t + phase) // 11) % 2 == 0 else 2   # elbow
    return a


POSTURE_J = [0, 1, 2] + list(range(17, 29))


def pin_trunk(sim, stance_q):
    """Write root + spine + legs back to the stance at rest; the arms
    keep their state (q and rates)."""
    q = list(sim.qpos())
    v = list(sim.twist()) + list(sim.omega())
    q[:7] = stance_q[:7]
    v[:6] = [0.0] * 6
    for j in POSTURE_J:
        q[7 + j] = stance_q[7 + j]
        v[6 + j] = 0.0
    sim.set_joints(q, v)


def rollout(env, ticks):
    """Palms + head z per step (index k = after step k+1)."""
    env.reset()
    stance_q = list(env.sim.qpos())
    palms, head_z = [], []
    for t in range(ticks):
        _o, _r, done, _t, _i = env.step(arm_actions(t))
        pin_trunk(env.sim, stance_q)
        palms.append(env.sim.palms())
        head_z.append(env.sim.head_pose()[2])
        if done:
            break
    return palms, head_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # pass 1: the palm path (an empty-ish chart far in the future keeps
    # the episode alive for the whole window)
    far = [FluxObj(hit_time=args.ticks * DT + 5.0, x=0.0, y=1.4,
                   kind=KIND_CIRCLE, speed=SPEED)]
    env = RustFluxEnv(external_chart=far, seed=args.seed)
    palms, head_z = rollout(env, args.ticks)

    def hit_time(k, pz):
        # step k+1 ends at time (k+1)*DT; z_off from that tick's head
        z_off = head_z[k] - PLANE_Z0
        return (k + 1) * DT + (pz + LEAD - z_off) / SPEED

    chart = []
    for j, k in enumerate(range(20, len(palms) - 30, 9)):
        (l, r) = palms[k]
        kind = j % 7
        if kind in (0, 3):      # left circle at the left palm
            chart.append(FluxObj(hit_time(k, l[2]), l[0], l[1], KIND_CIRCLE,
                                 SPEED, color=0))
        elif kind in (1, 4):    # right circle at the right palm
            chart.append(FluxObj(hit_time(k, r[2]), r[0], r[1], KIND_CIRCLE,
                                 SPEED, color=1))
        elif kind == 2:         # bar across the Flux, random sweet spot
            fx, fy = r[0] - l[0], r[1] - l[1]
            n = math.hypot(fx, fy) or 1.0
            ux, uy = -fy / n, fx / n
            # every other bar is turned 75 deg off square: the Flux
            # crosses it too obliquely (> 60 deg) -> bad cut "angle"
            off = math.radians(75.0) if (j // 7) % 2 else float(
                rng.uniform(-0.5, 0.5))
            rot = math.atan2(-ux, uy) + off
            length = float(rng.uniform(0.4, 0.9))
            sweet = float(rng.uniform(-1.0, 1.0))
            f = float(rng.uniform(0.3, 0.7))
            px, py = l[0] + f * fx, l[1] + f * fy
            s = sweet * 0.5 * length
            bx, by = -math.sin(rot), math.cos(rot)
            chart.append(FluxObj(hit_time(k, 0.5 * (l[2] + r[2])),
                                 px - s * bx, py - s * by, KIND_BAR, SPEED,
                                 rot=rot, length=length, sweet=sweet))
        elif kind == 5:         # wrong hand: a LEFT circle at the right palm
            chart.append(FluxObj(hit_time(k, r[2]), r[0], r[1], KIND_CIRCLE,
                                 SPEED, color=0))
        else:                   # deliberate misses: a circle and a bar
            chart.append(FluxObj(hit_time(k, 0.0), 0.9, 1.9, KIND_CIRCLE,
                                 SPEED, color=1))
            chart.append(FluxObj(hit_time(k, 0.0) + 0.15, -0.9, 2.0,
                                 KIND_BAR, SPEED, rot=1.2, length=0.3,
                                 sweet=0.5))
    chart.sort(key=lambda o: o.hit_time)

    env = RustFluxEnv(external_chart=chart, seed=args.seed)
    env.set_record(args.out)
    env.reset()
    stance_q = list(env.sim.qpos())
    t, done = 0, False
    while not done:
        _o, _r, done, _tr, info = env.step(arm_actions(t))
        pin_trunk(env.sim, stance_q)
        t += 1
    print(f"{args.out}: {len(chart)} objects, {t} ticks — good "
          f"{info['good']}, bad {info['bad']}, miss {info['misses']}, "
          f"score {info['score']:.0f}")


if __name__ == "__main__":
    main()
