"""Record/replay harness for CPU-sim (VecBeatSim.step_into) optimization work.

The laptop has no GPU and random actions make the bodies fall within ~30
ticks (finished envs early-out, so timing would be meaningless). Instead:

  record  run the leaf policy on CPU torch (greedy argmax, exactly like
          bench_scoresaber.py) on a fixed seeded set of real ranked levels
          with the sim configured like the bench (combo_walk, no events,
          policy gears, foot contact, tau_scale all 1.0), and save the
          per-tick actions, per-tick hashes of (obs, rewards, dones), dones,
          periodic full obs snapshots and the final per-env counters.
  replay  feed the recorded actions open-loop into a (freshly built) sim
          module with no policy in the loop: times step_into only (repeat 0
          is warm-up, reported separately) and checks every tick's hashes
          against the recording -> bitwise-identity check + realistic
          timing workload in one run. On a mismatch (result-changing
          opt-in builds) it reports the divergence: first differing tick,
          obs max-abs diff at the snapshot ticks, done-flag disagreements,
          final counter disagreements.

Open-loop actions: the trace is only an exact reference for bitwise-
identical builds, and the recording is per sim dtype (an f64 replay of an
f32 trace is timing-only).

  python tools/cpu_trace.py record --out TRACE_DIR [--envs 64 --ticks 3000]
      [--ckpt runs/v47_taurand_cd300/leaf_from_upd01068.pt]  (pod path)
  RAYON_NUM_THREADS=1 python tools/cpu_trace.py replay --trace TRACE_DIR \
      [--module-dir DIR] [--sim-dtype f32|f64] [--repeats 3] [--threads 8]

--threads sets RAYON_NUM_THREADS before the module is imported (rayon
sizes its pool once). --module-dir puts DIR first on sys.path so a saved
copy of a build (a directory containing the beat_arena_rs32/ package) can
be A/B'd against the installed one without reinstalling.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PYDIR = os.path.dirname(HERE)
DEFAULT_CKPT = os.path.expanduser(
    "~/beat-arena-volume-backup/beat-arena-v46/runs/v47_taurand_cd300/"
    "leaf_from_upd01068.pt")
SNAP_EVERY = 10          # full obs snapshot period (ticks)
TICK_HZ = 30.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--ckpt", default=DEFAULT_CKPT)
    r.add_argument("--envs", type=int, default=64)
    r.add_argument("--ticks", type=int, default=3000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--sim-dtype", choices=["f32", "f64"], default="f32")
    r.add_argument("--module-dir", default=None)
    r.add_argument("--threads", type=int, default=None)
    r.add_argument("--torch-threads", type=int, default=4)
    y = sub.add_parser("replay")
    y.add_argument("--trace", required=True)
    y.add_argument("--sim-dtype", choices=["f32", "f64"], default=None,
                   help="default: the trace's dtype")
    y.add_argument("--module-dir", default=None)
    y.add_argument("--threads", type=int, default=None)
    y.add_argument("--repeats", type=int, default=3,
                   help="timed repeats (after one warm-up pass)")
    y.add_argument("--ticks", type=int, default=0, help="0 = whole trace")
    y.add_argument("--json", default=None, help="append a result line here")
    return p.parse_args()


def import_sim(dtype, module_dir, threads):
    if threads is not None:
        os.environ["RAYON_NUM_THREADS"] = str(threads)
    if module_dir:
        sys.path.insert(0, os.path.abspath(module_dir))
    if dtype == "f32":
        import beat_arena_rs32 as mod
    else:
        import beat_arena_rs as mod
    return mod


def set_z_asinh(c):
    """BEAT_Z_ASINH for the rust sim (read once per process, at its first
    step) from the checkpoint / trace; a disagreeing preset is refused."""
    if os.environ.get("BEAT_Z_ASINH") not in (None, repr(c)):
        sys.exit(f"BEAT_Z_ASINH={os.environ['BEAT_Z_ASINH']} disagrees with "
                 f"the checkpoint's --z-asinh {c}; unset it")
    os.environ["BEAT_Z_ASINH"] = repr(c)
    print(f"obs z lane: asinh(z / {c:g} m) (from the checkpoint)", flush=True)


def so_sha(mod):
    d = os.path.dirname(mod.__file__)
    for f in sorted(os.listdir(d)):
        if f.endswith(".so") or f.endswith(".pyd"):
            with open(os.path.join(d, f), "rb") as fh:
                return os.path.join(d, f), hashlib.sha256(fh.read()).hexdigest()
    return mod.__file__, None


def make_sim(mod, levels, gears):
    """Configured exactly like bench_scoresaber._Group + its refill."""
    from beat_arena_env.constants import parse_tau_scale
    n = len(levels)
    sim = mod.VecBeatSim(n, max_steps=100_000_000, combo_walk=True,
                         return_events=False, gears=gears,
                         foot_contact=True)
    sim.set_tau_scale([float(x) for x in parse_tau_scale(None)])
    for i in range(n):
        sim.set_chart(i, [])
    return sim


def load_levels(meta_levels):
    out = []
    for lv in meta_levels:
        with open(os.path.join(PYDIR, lv["pkl"]), "rb") as f:
            out.append(pickle.load(f))
    return out


def fill(sim, payloads, obs_buf):
    for i, p in enumerate(payloads):
        sim.set_chart(i, p["tuples"], p.get("walls", []))
        sim.obs_one_into(i, obs_buf)


def tick_hash(obs_buf, rewards, dones):
    h = hashlib.blake2b(digest_size=16)
    h.update(memoryview(obs_buf).cast("B"))
    h.update(np.ascontiguousarray(rewards).tobytes())  # native dtype
    h.update(np.ascontiguousarray(dones, dtype=np.bool_).tobytes())
    return h.digest()


def final_state(sim, n):
    rows = []
    for i in range(n):
        rows.append({"counters": list(sim.counters(i)),
                     "combo": list(sim.combo(i)),
                     "components": list(sim.components(i)),
                     "flail": list(sim.flail(i))})
    return rows


def pick_levels(n, ticks, seed):
    """Seeded random ranked levels long enough to outlast the trace."""
    sys.path.insert(0, PYDIR)
    from bench_scoresaber import parser_version
    ver = parser_version()
    with open(os.path.join(PYDIR, "maps", "scoresaber_catalog.json")) as f:
        catalog = sorted(json.load(f), key=lambda e: e["leaderboard_id"])
    cache = os.path.join(PYDIR, "maps", "chart_cache")
    rng = np.random.default_rng(seed)
    need_s = ticks / TICK_HZ + 5.0
    picked, payloads = [], []
    for k in rng.permutation(len(catalog)):
        e = catalog[int(k)]
        # current parser version first; else any cached version (the
        # trace only needs SOME fixed real chart, and names its pkl)
        stem = f"{e['hash']}_{e['difficulty']}_"
        rel = os.path.join("maps", "chart_cache", f"{stem}{ver}.pkl")
        path = os.path.join(PYDIR, rel)
        if not os.path.exists(path):
            alts = sorted(glob.glob(os.path.join(cache, stem + "*.pkl")))
            if not alts:
                continue
            path = alts[0]
            rel = os.path.relpath(path, PYDIR)
        with open(path, "rb") as f:
            p = pickle.load(f)
        if not p["tuples"] or p["duration_s"] < need_s:
            continue
        picked.append({"leaderboard_id": e["leaderboard_id"], "pkl": rel,
                       "song": e["song"], "difficulty": e["difficulty"],
                       "stars": e["stars"], "duration_s": p["duration_s"]})
        payloads.append(p)
        if len(picked) == n:
            break
    if len(picked) < n:
        sys.exit(f"only {len(picked)} levels >= {need_s:.0f}s available")
    return picked, payloads, ver


def record(args):
    import torch
    torch.set_num_threads(args.torch_threads)
    mod = import_sim(args.sim_dtype, args.module_dir, args.threads)
    sys.path.insert(0, PYDIR)
    from beat_arena_env.constants import N_JOINTS, OBS_DIM
    from policy import load_policy
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    # env v48: the obs z encoding is the CHECKPOINT's asinh knee C (as in
    # bench_scoresaber.py). The rust sim reads BEAT_Z_ASINH once per
    # process (lazily, at the first step), so set it before any sim exists.
    z_asinh = float(ckpt.get("arch", {}).get(
        "z_asinh", ckpt.get("args", {}).get("z_asinh", 0.0)))
    if not z_asinh > 0:
        sys.exit(f"checkpoint has z_asinh {z_asinh:g}: it predates the asinh "
                 f"z encoding (legacy clamp, removed 2026-09-22)")
    set_z_asinh(z_asinh)
    policy = load_policy(ckpt).eval()
    levels, payloads, ver = pick_levels(args.envs, args.ticks, args.seed)
    n, T = args.envs, args.ticks
    so_path, so_hash = so_sha(mod)
    print(f"[record] {n} levels, {T} ticks, module {so_path}", flush=True)

    obs_t = torch.zeros((n, OBS_DIM), dtype=torch.float32)
    obs_buf = obs_t.numpy()
    sim = make_sim(mod, levels, policy.n_gears)
    fill(sim, payloads, obs_buf)
    actions = np.zeros((T, n, N_JOINTS), dtype=np.uint8)
    hashes = np.zeros((T + 1, 16), dtype=np.uint8)
    dones_all = np.zeros((T, n), dtype=np.bool_)
    snaps = np.zeros((T // SNAP_EVERY + 1, n, OBS_DIM), dtype=np.float32)
    hashes[0] = np.frombuffer(tick_hash(obs_buf, np.zeros(0), np.zeros(0)),
                              dtype=np.uint8)
    snaps[0] = obs_buf
    act_np = np.zeros((n, N_JOINTS), dtype=np.int64)
    t0 = time.perf_counter()
    sim_s = 0.0
    for t in range(T):
        with torch.no_grad():
            logits, _ = policy.forward_stacked(obs_t)
        act = logits.argmax(-1)
        act_np[:] = act.numpy()
        assert act_np.min() >= 0 and act_np.max() < 256
        actions[t] = act_np
        a = pc()
        rew, dones, _ = sim.step_into(act_np, obs_buf)
        sim_s += pc() - a
        dones_all[t] = np.asarray(dones)
        hashes[t + 1] = np.frombuffer(tick_hash(obs_buf, rew, dones),
                                      dtype=np.uint8)
        if (t + 1) % SNAP_EVERY == 0:
            snaps[(t + 1) // SNAP_EVERY] = obs_buf
        if (t + 1) % 250 == 0:
            print(f"  tick {t + 1}/{T}  done {int(dones_all[t].sum())}/{n}  "
                  f"wall {time.perf_counter() - t0:.0f}s  sim {sim_s:.1f}s",
                  flush=True)
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "actions.npy"), actions)
    np.save(os.path.join(args.out, "hashes.npy"), hashes)
    np.save(os.path.join(args.out, "dones.npy"), dones_all)
    np.save(os.path.join(args.out, "snaps.npy"), snaps)
    meta = {"envs": n, "ticks": T, "seed": args.seed, "ckpt": args.ckpt,
            "z_asinh": z_asinh,
            "sim_dtype": args.sim_dtype, "module": so_path,
            "module_sha256": so_hash, "parser_version": ver,
            "gears": policy.n_gears, "snap_every": SNAP_EVERY,
            "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
            "record_sim_s": round(sim_s, 3),
            "pd_hits": sim.pop_pd_hits(),
            "envelope_hits": sim.pop_envelope_hits(),
            "done_at_end": int(dones_all[-1].sum()),
            "levels": levels, "final": final_state(sim, n)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[record] saved {args.out}: done at end {meta['done_at_end']}/{n}, "
          f"sim {sim_s:.1f}s", flush=True)


def replay(args):
    with open(os.path.join(args.trace, "meta.json")) as f:
        meta = json.load(f)
    dtype = args.sim_dtype or meta["sim_dtype"]
    # the sim must encode z as the recording did; traces recorded before
    # z_asinh was stored ran the rust default (8 m, BEAT_Z_ASINH unset)
    if "z_asinh" in meta:
        set_z_asinh(float(meta["z_asinh"]))
    mod = import_sim(dtype, args.module_dir, args.threads)
    sys.path.insert(0, PYDIR)
    from beat_arena_env.constants import OBS_DIM
    so_path, so_hash = so_sha(mod)
    exact_ref = dtype == meta["sim_dtype"]
    actions8 = np.load(os.path.join(args.trace, "actions.npy"))
    hashes = np.load(os.path.join(args.trace, "hashes.npy"))
    dones_ref = np.load(os.path.join(args.trace, "dones.npy"))
    snaps = np.load(os.path.join(args.trace, "snaps.npy"), mmap_mode="r")
    T = min(args.ticks or meta["ticks"], meta["ticks"])
    n = meta["envs"]
    actions = np.ascontiguousarray(actions8[:T], dtype=np.int64)
    del actions8
    payloads = load_levels(meta["levels"])
    # active env-ticks: envs not yet done when the tick starts
    active = n * T - int(np.concatenate(
        [np.zeros((1, n), bool), dones_ref[:T - 1]]).sum()) if T else 0
    print(f"[replay] module {so_path} sha {str(so_hash)[:12]} dtype {dtype} "
          f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')} "
          f"trace {n}x{T} (active env-ticks {active})", flush=True)

    obs_buf = np.zeros((n, OBS_DIM), dtype=np.float32)
    results = []
    div = None
    for rep in range(args.repeats + 1):
        sim = make_sim(mod, meta["levels"], meta["gears"])
        fill(sim, payloads, obs_buf)
        first_bad = None
        n_bad = 0
        if tick_hash(obs_buf, np.zeros(0), np.zeros(0)) != hashes[0].tobytes():
            first_bad, n_bad = 0, 1
        max_abs = 0.0
        done_dis = 0
        step_s = 0.0
        for t in range(T):
            a = pc()
            rew, dones, _ = sim.step_into(actions[t], obs_buf)
            step_s += pc() - a
            if tick_hash(obs_buf, rew, dones) != hashes[t + 1].tobytes():
                n_bad += 1
                if first_bad is None:
                    first_bad = t + 1
            if rep == 0:
                d = np.asarray(dones)
                done_dis += int((d != dones_ref[t]).sum())
                if (t + 1) % meta["snap_every"] == 0:
                    diff = np.abs(obs_buf.astype(np.float64)
                                  - snaps[(t + 1) // meta["snap_every"]])
                    diff = np.nan_to_num(diff, nan=np.inf)
                    max_abs = max(max_abs, float(diff.max()))
        if rep == 0:
            fin = final_state(sim, n)
            fin_dis = sum(1 for x, y in zip(fin, meta["final"]) if x != y)
            div = {"first_bad_tick": first_bad, "bad_ticks": n_bad,
                   "done_disagreements": done_dis,
                   "done_at_end": int(np.asarray(dones).sum())
                   if T else 0,
                   "done_at_end_ref": int(dones_ref[T - 1].sum()) if T else 0,
                   "obs_max_abs_diff_snapshots": max_abs,
                   "final_counter_env_disagreements":
                       fin_dis if T == meta["ticks"] else None}
            ident = n_bad == 0 and (fin_dis == 0 or T != meta["ticks"])
            tag = ("BITWISE IDENTICAL" if ident else
                   "DIFFERS" + ("" if exact_ref else " (dtype != trace dtype: "
                                "timing only)"))
            print(f"[replay] warm-up {step_s:.2f}s  {tag}  {div}", flush=True)
        else:
            results.append(step_s)
            print(f"[replay] repeat {rep}: step_into {step_s:.3f}s = "
                  f"{1e6 * step_s / active:.1f} us/active-env-tick"
                  f"{'' if n_bad == 0 else f'  ({n_bad} hash mismatches)'}",
                  flush=True)
    if results:
        med = float(np.median(results))
        print(f"[replay] step_into min {min(results):.3f}s median {med:.3f}s "
              f"max {max(results):.3f}s | median "
              f"{1e6 * med / active:.1f} us/active-env-tick", flush=True)
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps({
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "module": so_path, "module_sha256": so_hash, "dtype": dtype,
                "threads": os.environ.get("RAYON_NUM_THREADS"),
                "ticks": T, "active_env_ticks": active,
                "step_into_s": results, "divergence": div}) + "\n")
    if div and div["bad_ticks"] and exact_ref:
        sys.exit(2)


pc = time.perf_counter

if __name__ == "__main__":
    a = parse_args()
    record(a) if a.cmd == "record" else replay(a)
