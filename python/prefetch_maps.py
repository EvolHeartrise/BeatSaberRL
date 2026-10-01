"""Bootstrap the full ScoreSaber map cache on a fresh machine (no GPU).

    python prefetch_maps.py [--limit N] [--skip-pool]
    python prefetch_maps.py --catalog beatleader [--limit N]

Fills all four cache layers front-to-back: ranked catalog (fetched once
if missing), extracted chart .dats from BeatSaver, parser-versioned
chart_cache pickles, and — unless --skip-pool / --limit — the whole-level
ranked pool npz that --ranked-maps training loads. Everything
downstream (train legs, bench, pods) then runs with zero network
fetches.

Failure policy (2026-08-08): a map BeatSaver no longer serves would
otherwise crash the pool build at whatever entry it dies on
(ranked_pool._build_segments has no per-entry recovery, by design — a
partial pool must never cache). So entries that still fail after a
second pass are DROPPED FROM THE CATALOG FILE, with the original saved
as scoresaber_catalog.orig.json and the drops recorded in
prefetch_failures.json. The pruned catalog is the frozen instrument:
its bytes key the pool cache, so pruning must happen here, once, before
anything downstream reads it — not ad hoc later.

BeatLeader (--catalog beatleader, 2026-09-30): the same dat cache
(maps/scoresaber/<hash>/ — BeatLeader leaderboards are BeatSaver hashes
too, so levels ranked on both leaderboards are already there), the same
chart_cache, the same parser and the same failure policy — but its OWN
catalog (maps/beatleader_catalog.json, fetched from the BeatLeader API
if missing), its own pruned-catalog backup (beatleader_catalog.orig.json)
and its own failures file (prefetch_failures_beatleader.json). It never
reads or writes scoresaber_catalog.json and never builds the training
pool (the pool is the ScoreSaber catalog's; BeatLeader is bench-only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

MAPS = os.path.join(BASE, "maps")
CATALOG = os.path.join(MAPS, "scoresaber_catalog.json")
BL_CATALOG = os.path.join(MAPS, "beatleader_catalog.json")
MAP_CACHE = os.path.join(MAPS, "scoresaber")
CHART_CACHE = os.path.join(MAPS, "chart_cache")

# Sequential polite pause after each actual BeatSaver download (cache
# hits skip it): ~2.5k songs -> adds ~4 min total, keeps us a good
# citizen of a hobbyist API.
DOWNLOAD_SLEEP_S = 0.1


def _prefetch(catalog, ver, log=print, workers=1):
    """One pass of load_or_parse over the catalog; returns failed entries.
    `workers` > 1 (2026-09-19) runs the download+parse in a thread pool:
    the wall time is BeatSaver latency, not CPU. Entries sharing a song
    hash (one map, several difficulties) are serialized on a per-hash
    lock so a map is downloaded once — ensure_map_dats' os.replace of
    the temp dir onto an existing cache dir would fail otherwise."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import bench_scoresaber as bs
    failed = []
    t0 = time.time()
    locks = {}
    locks_guard = threading.Lock()
    done = [0]
    progress = threading.Lock()

    def one(entry):
        with locks_guard:
            lock = locks.setdefault(entry["hash"], threading.Lock())
        with lock:
            was_cached = os.path.isdir(os.path.join(MAP_CACHE, entry["hash"]))
            try:
                bs.load_or_parse(entry, MAP_CACHE, CHART_CACHE, ver)
            except Exception as e:
                with progress:
                    failed.append(entry)
                log(f"[prefetch] FAIL {entry['hash']} {entry['difficulty']} "
                    f"({entry['song']!r}): {e!r}", flush=True)
            else:
                if not was_cached:
                    time.sleep(DOWNLOAD_SLEEP_S)
        with progress:
            done[0] += 1
            n = done[0]
        if n % 100 == 0:
            log(f"[prefetch] {n}/{len(catalog)} "
                f"({time.time() - t0:.0f}s, {len(failed)} failed)",
                flush=True)

    if workers <= 1:
        for entry in catalog:
            one(entry)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, catalog))
    return failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0,
                   help="smoke mode: first N catalog entries only "
                        "(implies --skip-pool; never prunes the catalog)")
    p.add_argument("--skip-pool", action="store_true",
                   help="stop after dats + chart_cache; no pool npz")
    # both are in the pool cache key: default to the training recipe
    # (ranked_pool.RECIPE_*, resolved lazily to keep torch off the
    # --skip-pool path) or the npz is one no training run loads
    p.add_argument("--lead-s", type=float, default=None,
                   help="pool npz front gap (default: the trainer's "
                        "--lead-s recipe value)")
    p.add_argument("--mirror-augment", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="pool npz mirror doubling (default: the trainer's "
                        "--mirror-augment recipe value)")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel download+parse threads (BeatSaver "
                        "latency-bound; 8 measured fine, 1 = the old "
                        "serial loop)")
    p.add_argument("--catalog", choices=["scoresaber", "beatleader"],
                   default="scoresaber",
                   help="which ranked catalog to fetch maps for. "
                        "beatleader: maps/beatleader_catalog.json (fetched "
                        "if missing), own failures file + pruned backup, "
                        "no pool npz (implies --skip-pool)")
    args = p.parse_args()

    if args.catalog == "beatleader":
        return main_beatleader(args)

    from beat_arena_env import scoresaber
    import bench_scoresaber as bs

    if not os.path.exists(CATALOG):
        print("[catalog] fetching all ranked leaderboards from ScoreSaber "
              "(one-time, ~2 min)...", flush=True)
        os.makedirs(MAPS, exist_ok=True)
        catalog = scoresaber.fetch_ranked_catalog(log=print)
        with open(CATALOG, "w") as f:
            json.dump(catalog, f)
        print(f"[catalog] {len(catalog)} ranked leaderboards -> {CATALOG}",
              flush=True)
    with open(CATALOG, encoding="utf-8") as f:
        catalog = json.load(f)

    # load_or_parse writes pickles into chart_cache but never creates it
    # (the bench's producer thread does, bench_scoresaber.py:381)
    os.makedirs(CHART_CACHE, exist_ok=True)

    ver = bs.parser_version()
    print(f"[prefetch] parser version {ver}, {len(catalog)} leaderboards",
          flush=True)

    work = catalog[:args.limit] if args.limit > 0 else catalog
    failed = _prefetch(work, ver, workers=args.workers)
    if failed:                      # transient-network second chance
        print(f"[prefetch] retrying {len(failed)} failures once...",
              flush=True)
        failed = _prefetch(failed, ver, workers=args.workers)

    if args.limit > 0:
        print(f"[prefetch] smoke done: {args.limit - len(failed)}"
              f"/{args.limit} ok ({len(failed)} failed); catalog untouched",
              flush=True)
        return

    if failed:
        dead = {(e["hash"], e["difficulty"]) for e in failed}
        pruned = [e for e in catalog
                  if (e["hash"], e["difficulty"]) not in dead]
        orig = os.path.join(MAPS, "scoresaber_catalog.orig.json")
        os.replace(CATALOG, orig)
        with open(CATALOG, "w") as f:
            json.dump(pruned, f)
        with open(os.path.join(MAPS, "prefetch_failures.json"), "w") as f:
            json.dump(failed, f, indent=1)
        print(f"[prefetch] PRUNED {len(failed)} dead leaderboards from the "
              f"catalog ({len(catalog)} -> {len(pruned)}); original at "
              f"{os.path.basename(orig)}, drops in prefetch_failures.json",
              flush=True)
    else:
        print(f"[prefetch] all {len(catalog)} leaderboards cached clean",
              flush=True)

    if not args.skip_pool:
        from beat_arena_env.ranked_pool import (
            RECIPE_LEAD_S, RECIPE_MIRROR, load_ranked_pool)
        lead_s = RECIPE_LEAD_S if args.lead_s is None else args.lead_s
        mirror = (RECIPE_MIRROR if args.mirror_augment is None
                  else args.mirror_augment)
        load_ranked_pool(device="cpu", lead_s=lead_s, mirror=mirror)
        print("[prefetch] pool npz built; cache complete", flush=True)


def main_beatleader(args):
    """--catalog beatleader: dats + chart_cache for every playable
    BeatLeader-ranked level (see the module docstring)."""
    from beat_arena_env import beatleader
    import bench_scoresaber as bs

    if not os.path.exists(BL_CATALOG):
        print("[bl-catalog] fetching all ranked leaderboards from "
              "BeatLeader (one-time, ~30 s)...", flush=True)
        os.makedirs(MAPS, exist_ok=True)
        catalog, excluded = beatleader.fetch_ranked_catalog(log=print)
        with open(BL_CATALOG, "w") as f:
            json.dump(catalog, f)
        print(f"[bl-catalog] {len(catalog)} playable ranked leaderboards "
              f"-> {BL_CATALOG} (excluded: {excluded or 'none'})",
              flush=True)
    with open(BL_CATALOG, encoding="utf-8") as f:
        catalog = json.load(f)
    os.makedirs(CHART_CACHE, exist_ok=True)
    ver = bs.parser_version()
    hashes = {e["hash"] for e in catalog}
    cached = {h for h in hashes if os.path.isdir(os.path.join(MAP_CACHE, h))}
    print(f"[prefetch] beatleader: parser version {ver}, {len(catalog)} "
          f"leaderboards over {len(hashes)} songs ({len(cached)} already "
          f"in the dat cache, {len(hashes) - len(cached)} to download)",
          flush=True)
    work = catalog[:args.limit] if args.limit > 0 else catalog
    failed = _prefetch(work, ver, workers=args.workers)
    if failed:                      # transient-network second chance
        print(f"[prefetch] retrying {len(failed)} failures once...",
              flush=True)
        failed = _prefetch(failed, ver, workers=args.workers)
    if args.limit > 0:
        print(f"[prefetch] smoke done: {args.limit - len(failed)}"
              f"/{args.limit} ok ({len(failed)} failed); catalog untouched",
              flush=True)
        return
    if failed:
        dead = {e["leaderboard_id"] for e in failed}
        pruned = [e for e in catalog if e["leaderboard_id"] not in dead]
        os.replace(BL_CATALOG,
                   os.path.join(MAPS, "beatleader_catalog.orig.json"))
        with open(BL_CATALOG, "w") as f:
            json.dump(pruned, f)
        with open(os.path.join(MAPS, "prefetch_failures_beatleader.json"),
                  "w") as f:
            json.dump(failed, f, indent=1)
        print(f"[prefetch] PRUNED {len(failed)} dead leaderboards from the "
              f"BeatLeader catalog ({len(catalog)} -> {len(pruned)}); "
              f"original at beatleader_catalog.orig.json, drops in "
              f"prefetch_failures_beatleader.json", flush=True)
    else:
        print(f"[prefetch] all {len(catalog)} BeatLeader leaderboards "
              f"cached clean", flush=True)


if __name__ == "__main__":
    main()
