"""exp/tick-rate regression probe: play the same real charts with a
checkpoint's greedy policy on a Rust sim build and dump a digest of every
step (obs bytes, rewards, dones, events) plus the final per-level
counters, so two builds can be compared bitwise (the 30 Hz gate: the
runtime-rate sim must reproduce the fixed-rate one exactly).

    python tools/tick_rate_regress.py --ckpt X.pt --charts DIR --out a.json \
        [--module beat_arena_rs32] [--tick-hz 30] [--action-repeat 1] [-n 16]

Run it once per build (PYTHONPATH selects the module) and diff the JSON.
With --tick-hz / --action-repeat it doubles as a quick smoke run of the
other arms (the per-step digest then differs by construction; compare the
per-level outcome table instead).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import inspect
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beat_arena_env.constants import N_JOINTS, OBS_DIM, parse_tau_scale  # noqa: E402
from policy import load_policy  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--charts", required=True, help="dir of bench chart_cache pickles")
    p.add_argument("--out", required=True)
    p.add_argument("--module", default="beat_arena_rs32")
    p.add_argument("-n", type=int, default=16)
    p.add_argument("--tick-hz", type=int, default=None,
                   help="omit to call the constructor without the kwarg "
                        "(works on pre-tick-rate builds)")
    p.add_argument("--action-repeat", type=int, default=1)
    p.add_argument("--max-ticks", type=int, default=0,
                   help="stop after this many decision steps (0 = play out)")
    a = p.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    z = float(ckpt.get("arch", {}).get("z_asinh", 8.0))
    os.environ["BEAT_Z_ASINH"] = repr(z)
    mod = importlib.import_module(a.module)
    policy = load_policy(ckpt).eval()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    files = sorted(glob.glob(os.path.join(a.charts, "*.pkl")))
    payloads = []
    for f in files:
        with open(f, "rb") as fh:
            pl = pickle.load(fh)
        if pl["tuples"] and pl["last_hit_s"] < 150:
            payloads.append((os.path.basename(f), pl))
        if len(payloads) == a.n:
            break
    n = len(payloads)
    kw = {} if a.tick_hz is None else {"tick_hz": a.tick_hz}
    sim = mod.VecBeatSim(n, max_steps=100_000_000, combo_walk=True,
                         return_events=True, gears=policy.n_gears, **kw)
    sim.set_tau_scale([float(x) for x in parse_tau_scale("default")])
    for i, (_, pl) in enumerate(payloads):
        sim.set_chart(i, pl["tuples"], pl.get("walls", []))
    obs = np.zeros((n, OBS_DIM), dtype=np.float32)
    for i in range(n):
        sim.obs_one_into(i, obs)
    done = np.zeros(n, dtype=bool)
    h = hashlib.sha256()
    steps = 0
    results = [None] * n
    step_kw = {} if a.action_repeat == 1 else {"repeat": a.action_repeat}
    while not done.all():
        with torch.no_grad():
            logits, _ = policy.forward_stacked(torch.from_numpy(obs))
        act = logits.argmax(-1).numpy().astype(np.int64)
        act[done] = 0
        rew, dn, ev = sim.step_into(act, obs, **step_kw)
        h.update(obs.tobytes())
        h.update(np.asarray(rew).tobytes())
        h.update(np.asarray(dn).tobytes())
        h.update(repr(ev).encode())
        for i in np.flatnonzero(np.asarray(dn) & ~done):
            g, b, m, ns, sc, bombs = sim.counters(int(i))
            results[i] = {"chart": payloads[i][0], "good": g, "bad": b,
                          "miss": m, "n": ns, "score": sc, "bombs": bombs,
                          "combo": list(sim.combo(int(i))),
                          "components": list(sim.components(int(i))),
                          "flail": list(sim.flail(int(i))),
                          "wrist_rev": list(sim.wrist_rev(int(i)))}
        done |= np.asarray(dn)
        steps += 1
        if a.max_ticks and steps >= a.max_ticks:
            break
    out = {"module": a.module, "file": inspect.getfile(mod),
           "tick_hz": a.tick_hz, "action_repeat": a.action_repeat,
           "steps": steps, "digest": h.hexdigest(), "levels": results}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    tot = sum(r["combo"][0] for r in results if r)
    print(f"{a.module} tick_hz={a.tick_hz} rep={a.action_repeat}: {n} levels, "
          f"{steps} steps, digest {h.hexdigest()[:16]}, sum combo score {tot:.0f}")


if __name__ == "__main__":
    main()
