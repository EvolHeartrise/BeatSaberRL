# Beat Arena training dashboard

Local, no-backend dashboard over the laptop run archive
(`~/beat-arena-volume-backup`). Three pieces:

1. **`python/metrics_log.py`** — the per-update metrics vocabulary. The
   trainer (`train_batch.py`) appends one JSON line per update to
   `<run_dir>/metrics.jsonl` at its print sync (same host values as the
   `update` / `[kl]` / event lines; no extra sync; a write failure
   disables the file, never the run). The same module parses a stdout
   log into identical objects.
2. **`python/metrics_backfill.py`** — walks the archive, parses every
   `*.log`, writes `metrics.jsonl` next to each run dir (or under
   `<campaign>/metrics/<run>/` when the run dir was never pulled) and
   one `~/beat-arena-volume-backup/index.json` registry. Idempotent;
   `--watch 30` re-scans every 30 s (cached: ~0.5 s per pass).
3. **`viz/dash/`** — this Vite + Svelte 5 + uPlot page.

## Run

```bash
cd ~/git\ projects/BeatSaberRL

# one-off backfill (first pass ~35 s over 133 runs / 310k updates)
python3 python/metrics_backfill.py

# build the page once (node 24, npm) — dist/ is gitignored
cd viz/dash && npm install && npm run build && cd ../..

# serve dist/ + the archive + the replay viewer; also runs the backfill
# in --watch mode so a live campaign's rsync loop shows up within 30 s
python3 viz/dash/serve.py            # -> http://127.0.0.1:8765/
```

Alternative with nothing but `http.server`: copy `viz/dash/dist/*` into
`~/beat-arena-volume-backup/` and run `python3 -m http.server 8765`
from there — the page finds `./index.json` first, `/archive/index.json`
second (`?archive=/some/prefix/` overrides). Development:
`npm run dev` (vite on :5173, proxies `/archive` to serve.py on :8765).

## Views

- **Live** — the most recently updated run (or any run from the
  selector). Panels: acc (+accL/accR), return, self-contact (impulse / % ticks, env v41), entropy
  (+entfull), ev, steps/s, falls+tilts per 1M samples, clipfrac
  first/last, KL first/last (log), gnorm+clipc, per-group entropies,
  lifts per 1M, flail, limit-rest, joint torque Σ|τ| (N·m / env-tick,
  total / spine / arms / legs — the env v43 energy-price base), joint activation (total / spine /
  arms / legs — the energy-conservation stat), fp16 r1/maxlr,
  checkpoint peak VRAM.
  Panels an era did not measure are marked "no data". X axis:
  samples | updates | wall-clock (hours; from `t` stamps, else
  cumulative `elapsed`). Auto-refresh every 30 s re-fetches index.json
  and the run's metrics.jsonl; the status line shows "last update N /
  total · samples · T ago" (T from the record's `t`, i.e. the log's
  mtime for backfilled runs). **Window lock** (2026-09-14): "window:
  last N" pins every panel's x axis to the most recent N units of the
  current x mode (M samples / updates / hours; 0 or "all" = whole run),
  remembered per mode in localStorage, re-applied after each refresh so
  the view follows the run. Refreshes now update the plots in place
  instead of rebuilding them, so a manual drag-zoom (with the lock off)
  also survives the 30 s refresh; a view showing the full run keeps
  extending to the new tail.
- **Compare** — tick any runs (filter box), align on samples (default)
  / updates / wall; small multiples per metric (acc, return, ent, entfull,
  ev, steps/s, falls / 1M and tilts / 1M on log axes, clipfrac, KL, gnorm,
  clipc, limit rest, flail, lifts, self-contact, activation, torque); a **difference panel**
  (run − reference at matched x, reference interpolated onto each run's
  grid, rolling mean over N points); a **matched-samples table** at
  chosen sample counts (default: the reference's checkpointed sample
  counts) with per-run acc / return / ent / ev and the
  delta vs the reference.
- **Runs** — sortable table from index.json: campaign, run, status,
  samples, updates, wall, steps/s, peak VRAM, final acc/return/ent,
  best bench upp/wpp, checkpoint count, last seen. Click a row to add
  it to Compare. Links: metrics.jsonl, first log, bench JSON (archive
  copies are links; repo-only copies are named), replay dir + the 3D
  viewer (`/replay.html`, drop a replay file in), run dir listing.

## Flux runs (SPEC.md §Flux)

Runs trained with `train_batch.py --game flux` carry `game: "flux"` in
index.json (read from args.json / the banner config; absent =
standard). The Live view badges them FLUX and relabels the accuracy
panel: the trainer's acc is points per touch / 115, so a perfect circle
reads 60/115 = 0.52 and a perfect bar 1.0 (bars count for both hands —
both palms make the Flux line); the saber-only flail panel is dropped.
The Runs table has a `game` column, and typing `flux` in either filter
box selects the Flux runs. Pod runs reach the archive the usual way —
an rsync loop pulling `runs/flux_*` (logs, metrics.jsonl, args.json)
into `~/beat-arena-volume-backup/beat-arena-flux/runs/`; replays from
`python/make_flux_replays.py` go in that campaign's `replays/` and are
linked by their `<tag>_bot…` names.

## index.json (per run)

`id` (`<campaign>/<run>`), `campaign`, `run`, `run_dir`, `logs`,
`metrics`, `metrics_source` (backfill | trainer), `game` (standard |
flux), `args` (args.json
when present, else the banner argv + `config:` dict), `invocations`,
`role` (trunk | leaf), `leaf_from`, `resume_from`, `num_envs`,
`horizon`, `batch`, `device`, `n_params`, `seed`, `status` (finished |
crashed | partial), `samples`, `updates`, `last_update`, `num_updates`,
`t_first` / `t_last` (epoch; t_last = log mtime, t_first = t_last −
wall), `wall_s` (sum of each invocation's final elapsed), `final`
{acc, accL, accR, return, ep_return, ent, ev, entfull}, `sps` (median
of the last 50 prints), `sps_total` (THROUGHPUT line), `peak_vram`,
`bench` [{path, url, ckpt, wpp, upp, n_evaluated, scoring_version,
updated}], `checkpoints` [{name, bytes}], `replays` (loose tag match —
candidates), `git_commit` (always null: no archive log records one),
`fields` (which metric keys this run has), `line_counts`.

## Metrics vocabulary (metrics.jsonl)

See `UPDATE_FIELDS` in `python/metrics_log.py`: `upd`, `n_upd`,
`steps`, `t`, `elapsed`, `sps`, `acc`, `accL`, `accR`, `return`,
`ep_return`, `ent`, `ev`, `r1`, `maxlr`, `fp16_overflow`,
`fp16_scale`, `kl_first`, `kl_last`, `clipfrac_first`,
`clipfrac_last`, `gnorm`, `clipc`, `ent_spine` … `ent_legR`,
`ent_min`, `ent_min_head`, `entfull`, `afsd`, `falls`, `tilts`,
`lifts`(+`_L`/`_R`), `flail`(+`_L`/`_R`/`_ticks`), `limitrest`
(+`_legs`/`_spine`/`_arms`), `torque` (+`_spine`/`_arms`/`_legs`: applied Σ|τ| N·m per env-tick, env v43), `activation` (+`_spine`/`_arms`/`_legs`:
mean commanded Σ|gear fraction| per env-tick, the `--iso-coef` base —
the energy-conservation stat since 2026-09-13), `pd_net`, `envelope`, `diag_*`,
`vram_*`, `ckpt`, `peak_vram`. Absent = not measured in that era
(or printed as nan). Tests: `python/tests/test_metrics_log.py`.
