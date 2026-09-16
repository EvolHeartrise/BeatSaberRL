"""Bootstrap the full ScoreSaber map cache on a fresh machine (no GPU).

    python prefetch_maps.py [--limit N] [--skip-pool]

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
MAP_CACHE = os.path.join(MAPS, "scoresaber")
CHART_CACHE = os.path.join(MAPS, "chart_cache")

# Sequential polite pause after each actual BeatSaver download (cache
# hits skip it): ~2.5k songs -> adds ~4 min total, keeps us a good
# citizen of a hobbyist API.
DOWNLOAD_SLEEP_S = 0.1


def _prefetch(catalog, ver, log=print):
    """One pass of load_or_parse over the catalog; returns failed entries."""
    import bench_scoresaber as bs
    failed = []
    t0 = time.time()
    for i, entry in enumerate(catalog):
        was_cached = os.path.isdir(os.path.join(MAP_CACHE, entry["hash"]))
        try:
            bs.load_or_parse(entry, MAP_CACHE, CHART_CACHE, ver)
        except Exception as e:
            failed.append(entry)
            log(f"[prefetch] FAIL {entry['hash']} {entry['difficulty']} "
                f"({entry['song']!r}): {e!r}", flush=True)
        else:
            if not was_cached:
                time.sleep(DOWNLOAD_SLEEP_S)
        if (i + 1) % 100 == 0:
            log(f"[prefetch] {i + 1}/{len(catalog)} "
                f"({time.time() - t0:.0f}s, {len(failed)} failed)",
                flush=True)
    return failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0,
                   help="smoke mode: first N catalog entries only "
                        "(implies --skip-pool; never prunes the catalog)")
    p.add_argument("--skip-pool", action="store_true",
                   help="stop after dats + chart_cache; no pool npz")
    args = p.parse_args()

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
    failed = _prefetch(work, ver)
    if failed:                      # transient-network second chance
        print(f"[prefetch] retrying {len(failed)} failures once...",
              flush=True)
        failed = _prefetch(failed, ver)

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
        from beat_arena_env.ranked_pool import load_ranked_pool
        load_ranked_pool(device="cpu")
        print("[prefetch] pool npz built; cache complete", flush=True)


if __name__ == "__main__":
    main()
