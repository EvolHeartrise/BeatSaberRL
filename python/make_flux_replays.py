"""Record watchable Flux replays of a checkpoint (SPEC.md §Flux).

    python make_flux_replays.py --ckpt runs/X/leaf_from_upd00350.pt \
        --tag flux_v1_cd300 [--n 10] [--seed0 1] [--workers 10]

Greedy (argmax) plays — the bench protocol — on N fresh procedural
Flux charts (seeds seed0..seed0+N-1) drawn with the checkpoint's own
generator settings (chart length, speeds, circle/bar odds), on the
contact-physics Rust sim (f32 by default: the regime training runs).
Writes replays/<tag>_bot<k>_seed<s>_<pct>pct.jsonl (the viewer format;
"_bot" lets the dashboard link them to the run) and
replays/<tag>_summary.json, and prints one line per play: raw score /
max (circle 60, bar 115), multiplied score (the 1/2/4/8x combo walk),
touches, bad hits by reason, misses by kind, falls / tilts.
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from beat_arena_env import scoresaber
from beat_arena_env.constants import KIND_BAR, KIND_CIRCLE
from beat_arena_env.flux_chart import FluxConfig, generate_flux_chart
from beat_arena_env.flux_env import RustFluxEnv
from policy import load_policy

_W = {}


def _worker_init(ckpt_path, device, sim_dtype):
    torch.set_num_threads(1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    z_asinh = float(ckpt["arch"]["z_asinh"])
    os.environ["BEAT_Z_ASINH"] = repr(z_asinh)   # before any sim exists
    a = ckpt.get("args", {})
    _W["cfg"] = FluxConfig(
        chart_len_s=a.get("chart_len", 20.0), cols=tuple(a.get("cols", (0, 1, 2, 3))),
        rows=tuple(a.get("rows", (0, 1, 2))),
        note_speed_min=a.get("note_speed_min", 10.0),
        note_speed_max=a.get("note_speed_max", 18.0),
        circle_prob=a.get("circle_prob", 0.3), bar_prob=a.get("bar_prob", 0.15))
    _W["policy"] = load_policy(ckpt, game="flux").to(device).eval()
    _W["dev"] = torch.device(device)
    _W["sim_dtype"] = sim_dtype


def _play(job):
    k, seed, out_dir, tag = job
    policy, dev = _W["policy"], _W["dev"]
    chart = generate_flux_chart(_W["cfg"], np.random.default_rng(seed))
    env = RustFluxEnv(external_chart=chart, gears=policy.n_gears,
                      sim_dtype=_W["sim_dtype"])
    tmp = os.path.join(out_dir, f".{tag}_{k}.jsonl")
    env.set_record(tmp)
    obs, _ = env.reset()
    walk = scoresaber.ComboTracker()
    reasons = Counter()
    done = False
    while not done:
        with torch.no_grad():
            logits, _ = policy(torch.as_tensor(obs, dtype=torch.float32,
                                               device=dev).unsqueeze(0))
        obs, _, done, _, info = env.step([int(l.argmax()) for l in logits])
        walk.consume(info["events"])
        for i, reason, _p in info["events"]:
            kind = "circle" if chart[i].kind == KIND_CIRCLE else "bar"
            reasons[f"{kind}_{reason}"] += 1
    max_raw = sum(60 if o.kind == KIND_CIRCLE else 115 for o in chart)
    pct = 100.0 * info["score"] / max_raw if max_raw else 0.0
    name = f"{tag}_bot{k:02d}_seed{seed}_{pct:.0f}pct.jsonl"
    os.replace(tmp, os.path.join(out_dir, name))
    return {
        "file": name, "seed": seed, "ticks": env.sim.tick,
        "n_circles": sum(o.kind == KIND_CIRCLE for o in chart),
        "n_bars": sum(o.kind == KIND_BAR for o in chart),
        "score": info["score"], "max_raw": max_raw, "pct": round(pct, 2),
        "multiplied": walk.score, "max_combo": walk.max_combo,
        "good": info["good"], "bad": info["bad"], "misses": info["misses"],
        "falls": info["falls"], "tilts": info["tilts"], "by_reason": dict(sorted(reasons.items())),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sim-dtype", choices=["f32", "f64"], default="f32")
    ap.add_argument("--out", default="replays")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(k + 1, a.seed0 + k, a.out, a.tag) for k in range(a.n)]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(a.workers, mp_context=ctx,
                             initializer=_worker_init,
                             initargs=(a.ckpt, a.device, a.sim_dtype)) as ex:
        rows = list(ex.map(_play, jobs))
    for r in rows:
        print(f"{r['file']}: {r['score']:.0f}/{r['max_raw']} ({r['pct']:.1f}%) "
              f"x-mult {r['multiplied']:.0f} combo {r['max_combo']} | good "
              f"{r['good']} bad {r['bad']} miss {r['misses']} fell "
              f"{r['falls']} tilt {r['tilts']} | {r['by_reason']}")
    tot = {k: sum(r[k] for r in rows) for k in
           ("score", "max_raw", "good", "bad", "misses", "falls",
            "tilts")}
    print(f"TOTAL {tot['score']:.0f}/{tot['max_raw']} "
          f"({100 * tot['score'] / max(1, tot['max_raw']):.1f}%) good "
          f"{tot['good']} bad {tot['bad']} miss {tot['misses']} falls "
          f"{tot['falls']} tilts {tot['tilts']} over {len(rows)} plays")
    with open(os.path.join(a.out, f"{a.tag}_summary.json"), "w") as f:
        json.dump({"ckpt": a.ckpt, "sim_dtype": a.sim_dtype, "plays": rows,
                   "total": tot}, f, indent=1)


if __name__ == "__main__":
    main()
