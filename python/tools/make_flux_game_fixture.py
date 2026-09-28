"""Env ↔ WebXR-game parity fixture (SPEC.md §Flux WebXR game).

Plays scripted Flux episodes on the REAL env (RawBeatSim, flux=True) and
records the installed chart's rotation record (sin_a, cos_a per object)
and, for every tick, the complete input of that tick's Flux judgment
— last tick's palms, this tick's palms, the head, the forward-most
shoulder z — at full float precision, plus the events the env emitted,
its done flag and its final tallies. `viz/app/test/flux-parity.test.js`
feeds the same inputs to the browser game's wasm judge
(rust/flux-wasm) and requires the identical event stream, tick for
tick, and the identical ending.

Episodes: the viewer fixture's scripted chart (make_flux_fixture.py:
circles, sweet-spot bars, 75°-off "angle" bars, a wrong-hand circle,
deliberate misses) at two seeds, and a short generator-f1 chart (the
training distribution) under the same scripted arms.

Run: python tools/make_flux_game_fixture.py OUT.json
     python tools/make_flux_game_fixture.py --check viz/app/test/fixtures/flux_game_parity.json
(--check regenerates in memory and fails if the file differs.)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beat_arena_env.constants import KIND_BAR  # noqa: E402
from beat_arena_env.flux_chart import (FluxConfig, FluxObj,  # noqa: E402
                                       flux_tuple, generate_flux_chart)
from beat_arena_env.flux_env import RustFluxEnv  # noqa: E402
from make_flux_fixture import arm_actions, build_chart, pin_trunk  # noqa: E402

EPISODES = (
    ("scripted seed 0", "scripted", 0),
    ("scripted seed 1", "scripted", 1),
    ("generator f1, 8 s, seed 5", "generated", 5),
)


def _chart(kind, seed):
    if kind == "scripted":
        return build_chart(seed, 300)
    return generate_flux_chart(FluxConfig(chart_len_s=8.0),
                               np.random.default_rng(seed))


def record_episode(name, kind, seed):
    chart = _chart(kind, seed)
    env = RustFluxEnv(external_chart=chart, seed=seed)
    env.reset()
    sim = env.sim
    trig = [[float(a), float(b)] for a, b in sim.chart_trig()]
    stance_q = list(sim.qpos())
    ticks, t, done = [], 0, False
    while not done:
        p0 = sim.palms()
        _obs, _r, done, events = sim.step(arm_actions(t))
        p1 = sim.palms()
        head = sim.head_pose()[:3]
        sh = max(sim.shoulder_z())
        ticks.append({
            "in": [*p0[0], *p0[1], *p1[0], *p1[1], *head, sh],
            "events": [[int(i), int(c), float(p), float(pre), float(acc)]
                       for i, c, p, pre, acc in events],
            "done": bool(done),
        })
        # the trunk pin moves the body AFTER the tick's judgment; the next
        # tick's p0 is read after it, exactly what the env starts from
        pin_trunk(sim, stance_q)
        t += 1
    base, acc, _post = sim.score_components()
    assert not (sim.fell or sim.tilts), f"{name}: body fell"
    return {
        "name": name,
        "chart": [list(flux_tuple(o)) for o in chart],
        "max_steps": int((chart[-1].hit_time + 0.5) * 30) + 90,
        "trig": trig,
        "ticks": ticks,
        "final": {"good": sim.good_cuts, "bad": sim.bad_cuts,
                  "misses": sim.misses, "score": sim.score,
                  "score_base": base, "score_acc": acc},
    }


# Trig probe: bar rotations installed on the env, whose (sin_a, cos_a)
# the wasm install must reproduce bit for bit. The platform libms (glibc
# natively, musl's port on wasm32) disagree in the last bit for ~6% of
# angles, so a few hundred uniform draws are sure to hit such angles —
# the episodes' handful of bars need not.
TRIG_PROBE_N = 400


def trig_probe():
    rng = np.random.default_rng(12345)
    rots = [float(r) for r in rng.uniform(0.0, 2.0 * math.pi, TRIG_PROBE_N)]
    env = RustFluxEnv(external_chart=[FluxObj(hit_time=1.0 + 0.01 * i, x=0.0,
                                              y=1.4, kind=KIND_BAR, speed=10.0,
                                              rot=r, length=0.5, sweet=0.0)
                                      for i, r in enumerate(rots)])
    env.reset()
    trig = [[float(a), float(b)] for a, b in env.sim.chart_trig()]
    disagree = sum((math.sin(r), math.cos(r)) != tuple(t)
                   for r, t in zip(rots, trig))
    print(f"trig probe: {len(rots)} bar rotations, {disagree} where the "
          "platform libm differs from the env's deterministic trig",
          file=sys.stderr)
    return {"rot": rots, "trig": trig}


def build():
    eps = [record_episode(*e) for e in EPISODES]
    for e in eps:
        f = e["final"]
        assert all(math.isfinite(v) for tk in e["ticks"] for v in tk["in"])
        print(f"{e['name']}: {len(e['chart'])} objects, {len(e['ticks'])} "
              f"ticks — good {f['good']}, bad {f['bad']}, miss "
              f"{f['misses']}, score {f['score']:.0f}", file=sys.stderr)
    return {"about": "env <-> WebXR game Flux judge parity; generated by "
                     "python/tools/make_flux_game_fixture.py — do not edit",
            "episodes": eps, "trig_probe": trig_probe()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--check", action="store_true",
                    help="compare against OUT instead of writing it")
    args = ap.parse_args()
    data = build()
    text = json.dumps(data, separators=(",", ":")) + "\n"
    if args.check:
        with open(args.out) as f:
            if f.read() != text:
                sys.exit(f"{args.out} is stale: regenerate it with "
                         f"python tools/make_flux_game_fixture.py {args.out}")
        print(f"{args.out}: up to date", file=sys.stderr)
        return
    with open(args.out, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
