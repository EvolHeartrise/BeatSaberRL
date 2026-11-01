# Beat Arena records site

Public leaderboard of ScoreSaber and BeatLeader bench results
(`python/bench_scoresaber.py`) per model and per map, ScoreSaber and
BeatLeader side by side (SS WPP / UPP / PP and BL WPP / UPP / PP columns). SvelteKit (adapter-static, client-rendered) with
Tailwind v4 + shadcn-svelte / bits-ui components (`src/lib/components/ui/` is
generated: add more with `npx shadcn-svelte@latest add <name>`), served as
static assets by a Cloudflare Worker at beatsaber.flee.systems and
beatsaber.alexandrorodriguez.com.

```bash
npm install
npm run data      # rebuild static/data/ from the bench JSONs
npm run dev       # local dev
npm run deploy    # vite build + wrangler deploy
```

`npm run data` scans `python/bench/` and `~/beat-arena-volume-backup/*/bench/`
(add more with `-- --src DIR`) and publishes only current-era benches:
`scoring_version` 3, asinh z encoding (`arch.z_asinh` > 0), Rust sim, full
catalog. A checkpoint benched under several body presets (same weights hash)
shows only its best bench by weighted PP. Only a run's version prefix is
published (`v56` for `v56_6blk_g965_nstuned_cd400`), with `-N` appended when
several benches share one (numbered in file-name order); bench file names, full
run names and checkpoint paths stay private. After a new
bench lands: `npm run data && npm run deploy`.

BeatLeader: `levels.json` is one unified catalog, one row per level (song
hash + difficulty + characteristic) across both leaderboards: every
ScoreSaber-ranked level plus the BeatLeader-only ones, with `ssId` / `blId`
(null = not ranked there) and each board's own stars and ratings, never mixed.
`--catalog both` benches also publish a `bl` block in models.json and
`results/<hash>b.json` (BL columns, aligned to the same rows as the SS
`results/<hash>.json`). ScoreSaber-only benches get their BL view from a
sidecar `~/beat-arena-volume-backup/bl-derived/<bench basename>.json`
(`python/tools/bench_add_bl.py`: the bench's own plays priced by BeatLeader's
rules on the maps both boards rank; skipped with a warning unless its
model_hash / tick_hz / action_repeat / obs_window match the bench). Their BL
totals cover the shared maps only: models.json `bl.coverage` = {played,
catalog}, shown as "N / M maps". Without a sidecar a
ScoreSaber-only bench shows " - " in BL columns.
Map pages live at `/maps/<ScoreSaber id>` for every level ScoreSaber ranks and
`/maps/bl/<BeatLeader id>` for BeatLeader-only levels; a `/maps/bl/` id of a
level ScoreSaber also ranks redirects to its `/maps/<ScoreSaber id>` page. A
composite bench (header `composite`, a per-level best of several runs of one
checkpoint) is published under its public id like any other model (v61); its
input runs and per-level source runs are used at build time only (replay
matching) and appear neither in the UI nor in the published data. Model
colors are assigned by `npm run data` (models.json `color`).

"Watch" links: plays with a replay on replays.flee.systems (the hosted 3D
viewer, `viz/replays-site`) link to it from the model and map pages.
`npm run data` reads the replay store's manifest
(`~/beat-arena-volume-backup/replays-site/manifest.json`, or
`-- --replays PATH`) and links a replay only when it matches the published
bench: weights hash, tick_hz, action_repeat, obs_window, level and score. For a
composite (v61) each replay is matched against the level's PICKED source run
(weights hash, that run's tick_hz and body, level, score); a row links the
ScoreSaber pick's replay, else the BeatLeader pick's. After
publishing replays (`viz/replays-site/scripts/publish_replays.py`), redeploy
the replays site, then `npm run data && npm run deploy` here.
