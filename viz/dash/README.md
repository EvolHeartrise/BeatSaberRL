# Beat Arena training dashboard

Local, no-backend dashboard over the laptop run archive
(`~/beat-arena-volume-backup`). Three pieces:

1. **`python/metrics_log.py`** - the per-update metrics vocabulary. The
   trainer (`train_batch.py`) appends one JSON line per update to
   `<run_dir>/metrics.jsonl` at its print sync (same host values as the
   `update` / `[kl]` / event lines; no extra sync; a write failure
   disables the file, never the run). The same module parses a stdout
   log into identical objects.
2. **`python/metrics_backfill.py`** - walks the archive, parses every
   `*.log`, writes `metrics.jsonl` next to each run dir (or under
   `<campaign>/metrics/<run>/` when the run dir was never pulled) and
   one `~/beat-arena-volume-backup/index.json` registry. Idempotent;
   `--watch 30` re-scans every 30 s (cached: ~0.5 s per pass).
3. **`viz/dash/`** - this Vite + Svelte 5 + uPlot page.

## Run

```bash
cd ~/git\ projects/BeatSaberRL

# one-off backfill (first pass ~35 s over 133 runs / 310k updates)
python3 python/metrics_backfill.py

# build the page once (node 24, npm) - dist/ is gitignored
cd viz/dash && npm install && npm run build && cd ../..

# serve dist/ + the archive + the replay viewer; also runs the backfill
# in --watch mode so a live campaign's rsync loop shows up within 30 s
python3 viz/dash/serve.py            # -> http://127.0.0.1:8765/
```

Alternative with nothing but `http.server`: copy `viz/dash/dist/*` into
`~/beat-arena-volume-backup/` and run `python3 -m http.server 8765`
from there - the page finds `./index.json` first, `/archive/index.json`
second (`?archive=/some/prefix/` overrides). Development:
`npm run dev` (vite on :5173, proxies `/archive` to serve.py on :8765).
Test: `npm test` builds, runs `test/incremental.mjs` (no browser:
chunked / cut-line parsing gives the very columns and chart series of one
full parse, incl. --resume re-appends and the swing_hist merge; also over
the real archive's biggest metrics file when present) and
`test/refresh-browser.mjs` - one
headless Chrome (from `~/.cache/puppeteer/chrome`, or
`PUPPETEER_EXECUTABLE_PATH`) against a fake archive in `test/.archive`
served by serve.py: idle refreshes must not redraw or reset a zoom
(Live, window lock, Compare), appended rows must reach every chart,
hovering must not change any card's height, a cut line changes nothing
until complete, growing files are read by Range, a shrunk one whole;
hiding / restoring graphs (menu, keyboard, reload, refresh, per tab).

Chart cards (2026-10-06): the panel title floats on the plot's top-left
corner, the legend under the plot is static (names + colours, click a
name to hide/show that series; capped at three rows, scrolls past that),
and the values at the cursor show in a floating tooltip on the hovered
chart only - x in the header, series sorted by value with colour
swatches, clamped to the viewport - so hovering never reflows the page.

Hidden graphs (2026-10-06): each card's eye-off button (top-right; on
hover, always on touch screens) hides that panel; "Hidden (n)" in the
header lists the current tab's hidden graphs - click one to bring it
back, or "Show all" (Esc / outside click closes, arrow keys move). Live
and Compare keep separate sets, keyed by the panel ids in series.js
(Compare: the metric key, `diff:<key>` for the difference panels), in
localStorage `dash.hidden.<tab>`; refreshes never un-hide, unknown ids
are pruned, new panels start visible.

## Views

- **Live** - the most recently updated run (or any run from the
  selector). Panels: acc (+accL/accR), return, self-contact (impulse / % ticks, env v41), entropy
  (+entfull), ev, steps/s, falls+tilts per 1M samples, clipfrac
  first/last, KL first/last (log), gnorm+clipc, per-group entropies,
  lifts per 1M, flail, limit-rest, joint torque Σ|τ| (N·m / env-tick,
  total / spine / arms / legs - the env v43 energy-price base), joint activation (total / spine /
  arms / legs - the energy-conservation stat), fp16 r1/maxlr,
  checkpoint peak VRAM.
  Panels an era did not measure are marked "no data". X axis:
  samples | updates | wall-clock (hours; from `t` stamps, else
  cumulative `elapsed`). Auto-refresh every 15 s (`?refresh=<s>`
  overrides) re-checks index.json and the run's metrics.jsonl; the
  status line shows "last update N /
  total · samples · T ago" (T from the record's `t`, i.e. the log's
  mtime for backfilled runs). **Window lock** (2026-09-14): "window:
  last N" pins every panel's x axis to the most recent N units of the
  current x mode (M samples / updates / hours; 0 or "all" = whole run),
  remembered per mode in localStorage, re-applied when new rows arrive so
  the view follows the run. **Idle refreshes leave the charts alone**
  (2026-10-02): a refresh that brought nothing new does not redraw,
  re-zoom or re-window anything, so a drag-zoom (with or without the
  lock) survives while no run is training. Detection: serve.py sends an
  ETag (mtime + size) and the page polls with If-None-Match (304 = no
  body); a 200 is still compared with what the page holds (metrics bytes;
  index.json minus its `generated` stamp, which the backfill bumps every
  pass), and the charts compare points before touching uPlot. When rows
  do arrive the plots update in place: a zoomed view keeps its window
  and redraws, a full-run view extends to the new tail. Switching run
  rebuilds the plots (no inherited zoom).
  **Refreshes never block the page** (2026-10-06): fetching, change
  detection and parsing run in a Web Worker (`src/lib/fetch.worker.js`);
  a growing metrics.jsonl is re-read from where the last read stopped
  (HTTP Range, which serve.py answers with 206; a plain `http.server`
  sends the whole file and the worker checks the consumed prefix by hash)
  and only the new lines are parsed (`src/lib/metricsCore.js`; a line cut
  mid-write waits for its end). The page gets columns (Float64Arrays,
  transferred), held outside Svelte's deep proxies; series are memoized
  per run, so an unchanged run costs nothing. Chart work is queued and
  run a few charts per ~8 ms slice (`src/lib/sched.js`); charts scrolled
  out of view update when they come back. Hover on a joined grid
  (Compare) finds each run's point by binary search.
- **Swing** (2026-10-04) - per-level swing quality over the ranked
  catalogue and the run's best-possible PP, from `<run>/swing.json`
  (`python/swing_summary.py`; runs with an `ep_stats.f32`). Four tiles:
  the best-possible **SS WPP / SS UPP / BL WPP / BL UPP** - each level's
  best completed play of the whole run, per leaderboard, with the bench's
  accuracy definitions (scoring v4: SS acc against the SS API maxScore,
  BL against its own, both falling back to the env's computed max;
  WPP = descending × 0.965^i, UPP = sum; shared levels on both boards,
  a mirrored training copy merged with its original), next to the run's
  best bench. A sortable table (default: lowest pre-swing first; missing
  values sort last), 200 rows at a time: song, difficulty, SS / BL
  stars, NPS (scoreable notes / song length), completed plays in the
  window, completed / attempted over the run, mean accuracy, and pre-swing
  (/70), post-swing (/30), precision (/15) as % of max, overall and L / R,
  plus each level's best SS / BL pp and the update it came from, and
  (ep_stats v3, 2026-10-04) wrist resets per 100 swings, overall and
  L / R (resets / swings summed over the window's plays, or the best
  play's own; higher = redder). Window:
  the last 1 / 5 / 20 completed plays per level, the completed plays of
  the last 100 / 500 updates, or the whole run. Only COMPLETED plays (the
  level's end reached: no fall, tilt, envelope or tick cap) count. Old
  runs (ep_stats v1, before 2026-10-04) show plays, accuracy and the
  totals with the swing columns marked unavailable; their update windows
  come from the trainer's metrics.jsonl `ep_stats_episodes` counter when
  it is consistent. Same refresh rules as Live: an unchanged swing.json
  (ETag 304 / same bytes) touches nothing - sort, filter, window and the
  scroll position stay - and new data updates the keyed rows in place
  (scroll anchoring is off so the page does not follow a moved row).
- **Compare** - tick any runs (filter box), align on samples (default)
  / updates / wall; small multiples per metric (acc, return, ent, entfull,
  ev, steps/s, falls / 1M and tilts / 1M on log axes, clipfrac, KL, gnorm,
  clipc, limit rest, flail, lifts, self-contact, activation, torque); a **difference panel**
  (run − reference at matched x, reference interpolated onto each run's
  grid, rolling mean over N points); a **matched-samples table** at
  chosen sample counts (default: the reference's checkpointed sample
  counts) with per-run acc / return / ent / ev and the
  delta vs the reference.
- **Runs** - sortable table from index.json: campaign, run, status,
  samples, updates, wall, steps/s, peak VRAM, final acc/return/ent,
  best bench upp/wpp, the training best-possible SS / BL upp (Swing
  totals), checkpoint count, last seen. Click a row to add
  it to Compare. Links: metrics.jsonl, first log, bench JSON (archive
  copies are links; repo-only copies are named), replay dir + the 3D
  viewer (`/replay.html`, drop a replay file in), run dir listing.

## Flux runs (SPEC.md §Flux)

Runs trained with `train_batch.py --game flux` carry `game: "flux"` in
index.json (read from args.json / the banner config; absent =
standard). The Live view badges them FLUX and relabels the accuracy
panel: the trainer's acc is points per touch / 115, so a perfect circle
reads 60/115 = 0.52 and a perfect bar 1.0 (bars count for both hands -
both palms make the Flux line); the saber-only flail panel is dropped.
The Runs table has a `game` column, and typing `flux` in either filter
box selects the Flux runs. Pod runs reach the archive the usual way -
an rsync loop pulling `runs/flux_*` (logs, metrics.jsonl, args.json)
into `~/beat-arena-volume-backup/beat-arena-flux/runs/`; replays from
`python/make_flux_replays.py` go in that campaign's `replays/` and are
linked by their `<tag>_bot…` names.

## Swing data (swing.json, levels.json)

Inputs, synced from the pod with the run dir: `ep_stats.f32` (v2: header
row + 29 float32 cols, see `python/ep_stats.py`) and `levels.json` (the
trainer writes it at startup under `--ep-stats`: `chart_level[c]` /
`chart_mirrored[c]` per pool chart, `levels[]` with key, song, mapper,
difficulty, n_notes, duration_s, nps, max_score (computed), `ss` {id,
stars, max_pp, max_score} and `bl` {id, stars, pass, acc, tech,
max_score}, null where that leaderboard does not rank the level). **A
campaign's sync script must not exclude `*.f32` or `levels.json`** (the
wb3 / arch scripts pull everything but `*.pt`, which is right; the older
wb / wb2 / up16-ab ones excluded `*.f32`). For a run that predates the
sidecar, `python/.venv/bin/python python/swing_summary.py --reconstruct
<run_dir>` (or `--archive ~/beat-arena-volume-backup`) rebuilds it from
args.json + the laptop's maps and refuses a table whose SS ids do not
match the run's rows (done for every archive run on 2026-10-04).

`metrics_backfill.py` calls `swing_summary.update_run` for every run dir
with an `ep_stats.f32` on each pass: incremental (a `.swing_state.json` /
`.swing_state.npz` pair next to the run keeps the byte offset, the
whole-run sums, the last 20 completed plays per level and the plays of
the last 500 updates; an unchanged file costs two stats), and swing.json
is rewritten only when rows arrived. swing.json is columnar: `levels`
{key, song, mapper, diff, ss_stars, bl_stars, ss_id, bl_id, nps, att,
done, best_acc, ss_acc, ss_pp, bl_acc, bl_pp, best_upd}, `win` {n1, n5,
n20, k100, k500, all: {plays, acc, pre, preL, preR, post, postL, postR,
prec, precL, precR} - % values, null = no data / no swing columns},
`windows` (key, label, available), `totals` {ss_wpp, ss_upp, bl_wpp,
bl_upp, ss_levels, bl_levels}, plus rows / completed / last_update /
ep_layout / swing / levels_source.

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
updated, bl_wpp, bl_upp}], `checkpoints` [{name, bytes}], `replays` (loose tag match -
candidates), `swing` (null without an ep_stats.f32: {path, ep_layout,
swing, levels_source, rows, completed, last_update, n_levels, totals}), `git_commit` (always null: no archive log records one),
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
mean commanded Σ|gear fraction| per env-tick, the `--iso-coef` base -
the energy-conservation stat since 2026-09-13), `pd_net`, `envelope`, `diag_*`,
`vram_*`, `ckpt`, `peak_vram`. Absent = not measured in that era
(or printed as nan). Tests: `python/tests/test_metrics_log.py`.
