<script>
  // Swing tab (2026-10-04, user): per-level swing quality over the ranked
  // catalogue — are the body's swing speeds enough to full-swing every map?
  // — plus the run's best-possible PP totals. Data: <run>/swing.json from
  // python/swing_summary.py (the backfill keeps it current). Completed
  // plays only. Sort, filter, window and scroll survive refreshes: an
  // unchanged swing.json does not bump the store version, so nothing here
  // re-derives; new data updates the rows in place (keyed by level).
  import { store, loadIndex, loadSwing, swingOf, runById, REFRESH_MS } from "./data.svelte.js";
  import { fmtSamples } from "./series.js";

  let { runId = $bindable(null) } = $props();

  const WIN_KEY = "dash.swing.window";
  let winKey = $state((() => { try { return localStorage.getItem(WIN_KEY) || "n5"; } catch { return "n5"; } })());
  // per-level stats as the window's mean, or its best play (highest
  // accuracy: that play's own acc and swing %) — swing.json win / win_best
  const STAT_KEY = "dash.swing.stat";
  let stat = $state((() => { try { return localStorage.getItem(STAT_KEY) === "best" ? "best" : "mean"; } catch { return "mean"; } })());
  let sortKey = $state(null); // null = the default (pre-swing, or acc when the run has no swing columns)
  let sortDir = $state(1);
  let filter = $state("");
  let board = $state("all");
  let limit = $state(200);
  let refreshing = $state(false);

  const runs = $derived((store.index?.runs ?? []).filter((r) => r.swing));
  const newest = $derived(runs.reduce((b, r) => (!b || (r.t_last ?? 0) > (b.t_last ?? 0) ? r : b), null));
  const run = $derived(runId ? runById(runId) : newest);
  const ver = $derived(run ? store.swingVer[run.id] : undefined);
  const data = $derived(ver != null ? swingOf(run.id) : null);
  const W = $derived((stat === "best" && data?.win_best ? data.win_best : data?.win)?.[winKey] ?? null);
  const bench = $derived.by(() => {
    let best = null;
    for (const b of run?.bench ?? []) if (!best || (b.upp ?? -1) > (best.upp ?? -1)) best = b;
    return best;
  });

  const pct = (v) => (v == null ? "—" : v.toFixed(1));
  const num0 = (v) => (v == null ? "—" : Math.round(v).toLocaleString());
  const COLS = [
    { key: "song", label: "song", src: "lv", text: true },
    { key: "diff", label: "diff", src: "lv", text: true },
    { key: "ss_stars", label: "SS ★", src: "lv", f: (v) => (v == null ? "—" : v.toFixed(2)) },
    { key: "bl_stars", label: "BL ★", src: "lv", f: (v) => (v == null ? "—" : v.toFixed(2)) },
    { key: "nps", label: "NPS", src: "lv", f: (v) => (v == null ? "—" : v.toFixed(2)), title: "scoreable notes / song length" },
    { key: "plays", label: "plays", src: "win", f: (v) => v ?? 0, title: "completed plays in the window" },
    { key: "done", label: "done/att", src: "lv", title: "whole run: completed / attempted (≥ 60 ticks)" },
    { key: "acc", label: "acc %", src: "win", f: pct, heat: true, title: "accuracy (score / the env's max)" },
    { key: "pre", label: "pre %", src: "win", f: pct, heat: true, swing: true, title: "pre-swing points / 70 per note" },
    { key: "preL", label: "L", src: "win", f: pct, heat: true, swing: true },
    { key: "preR", label: "R", src: "win", f: pct, heat: true, swing: true },
    { key: "post", label: "post %", src: "win", f: pct, heat: true, swing: true, title: "post-swing (follow-through) points / 30 per note" },
    { key: "postL", label: "L", src: "win", f: pct, heat: true, swing: true },
    { key: "postR", label: "R", src: "win", f: pct, heat: true, swing: true },
    { key: "prec", label: "prec %", src: "win", f: pct, heat: true, swing: true, title: "precision (cut centre) points / 15 per note" },
    { key: "precL", label: "L", src: "win", f: pct, heat: true, swing: true },
    { key: "precR", label: "R", src: "win", f: pct, heat: true, swing: true },
    { key: "wrs", label: "resets/100", src: "win", f: pct, heatUp: true, wrs: true, title: "wrist resets per 100 swings, both hands: a hit whose bend-velocity sign equals the previous swing's, with a ≥ 30° reversal of the bend in between (good note cuts; hits of one continuous bend motion are one swing)" },
    { key: "wrsL", label: "L", src: "win", f: pct, heatUp: true, wrs: true, title: "wrist resets per 100 swings, left hand" },
    { key: "wrsR", label: "R", src: "win", f: pct, heatUp: true, wrs: true, title: "wrist resets per 100 swings, right hand" },
    { key: "ss_pp", label: "best SS pp", src: "lv", f: (v) => (v == null ? "—" : v.toFixed(1)), title: "best completed play of the run, ScoreSaber" },
    { key: "bl_pp", label: "best BL pp", src: "lv", f: (v) => (v == null ? "—" : v.toFixed(1)), title: "best completed play of the run, BeatLeader" },
    { key: "best_upd", label: "best @upd", src: "lv", f: (v) => v ?? "—" },
  ];
  const activeSort = $derived(sortKey ?? (data && !data.swing ? "acc" : "pre"));

  function val(i, c) {
    if (c.key === "done") return data.levels.done[i];
    if (c.src === "lv") return data.levels[c.key]?.[i] ?? null;
    return W?.[c.key]?.[i] ?? null;
  }

  const order = $derived.by(() => {
    if (!data) return [];
    const L = data.levels;
    const f = filter.trim().toLowerCase();
    const idx = [];
    for (let i = 0; i < L.key.length; i++) {
      if (board === "ss" && L.ss_stars[i] == null) continue;
      if (board === "bl" && L.bl_stars[i] == null) continue;
      if (f && !(`${L.song[i] ?? ""} ${L.mapper[i] ?? ""} ${L.diff[i] ?? ""} ${L.key[i]}`.toLowerCase().includes(f))) continue;
      idx.push(i);
    }
    const c = COLS.find((x) => x.key === activeSort) ?? COLS[0];
    const v = idx.map((i) => val(i, c));
    const pos = new Map(idx.map((i, j) => [i, j]));
    idx.sort((a, b) => {
      const va = v[pos.get(a)], vb = v[pos.get(b)];
      if (va == null && vb == null) return a - b;
      if (va == null) return 1; // missing values last in both directions
      if (vb == null) return -1;
      const d = c.text ? String(va).localeCompare(String(vb)) : va - vb;
      return d * sortDir || a - b;
    });
    return idx;
  });

  function sortBy(key) {
    if (activeSort === key) sortDir = -sortDir;
    else { sortKey = key; sortDir = 1; }
  }
  function setStat(k) {
    stat = k;
    try { localStorage.setItem(STAT_KEY, k); } catch {}
  }
  function setWin(k) {
    winKey = k;
    try { localStorage.setItem(WIN_KEY, k); } catch {}
  }
  function heat(v) {
    if (v == null) return "";
    const a = Math.max(0, Math.min(60, (100 - v) * 2.5));
    return `background: color-mix(in srgb, var(--bad) ${a.toFixed(0)}%, transparent)`;
  }
  // higher = worse (wrist resets per 100 swings): 24 / 100 is full tint
  function heatUp(v) {
    if (v == null) return "";
    const a = Math.max(0, Math.min(60, v * 2.5));
    return `background: color-mix(in srgb, var(--bad) ${a.toFixed(0)}%, transparent)`;
  }

  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      await loadIndex();
      if (run) await loadSwing(run, { force: true });
    } finally {
      refreshing = false;
    }
  }
  $effect(() => {
    if (run && ver == null) loadSwing(run);
  });
  $effect(() => {
    const h = setInterval(refresh, REFRESH_MS);
    return () => clearInterval(h);
  });
</script>

<div class="bar">
  <label>run
    <select value={runId ?? ""} onchange={(e) => { runId = e.target.value || null; limit = 200; }}>
      <option value="">most recent ({newest?.id ?? "—"})</option>
      {#each runs as r (r.id)}
        <option value={r.id}>{r.id} · {fmtSamples(r.samples)} · {r.status}{r.swing.swing ? "" : " · no swing cols"}</option>
      {/each}
    </select>
  </label>
  <span class="wins">
    window:
    {#each data?.windows ?? [] as w (w.key)}
      <button class:on={winKey === w.key} disabled={!w.available} title={w.available ? "" : "no update index in this run's rows"} onclick={() => setWin(w.key)}>{w.label}</button>
    {/each}
  </span>
  <span class="wins" title="per-level stats: the mean over the window's plays, or the window's best play (highest accuracy) — that play's own accuracy and swing %">
    stats:
    <button class:on={stat === "mean"} onclick={() => setStat("mean")}>mean</button>
    <button class:on={stat === "best"} disabled={!data?.win_best} onclick={() => setStat("best")}>best</button>
  </span>
  <select bind:value={board} title="leaderboard">
    <option value="all">all levels</option>
    <option value="ss">SS-ranked</option>
    <option value="bl">BL-ranked</option>
  </select>
  <input placeholder="filter song / mapper / diff…" bind:value={filter} />
  <span class="auto">{refreshing ? "refreshing…" : `auto ${REFRESH_MS / 1000} s`}</span>
</div>

{#if !runs.length}
  <p class="muted">no run has a swing summary yet — runs with an ep_stats.f32 get one from the backfill (python/swing_summary.py)</p>
{:else if !run}
  <p class="muted">run not found</p>
{:else if !data}
  <p class="muted">loading {run.swing.path}…</p>
{:else}
  <div class="totals">
    {#each [["SS WPP", "ss_wpp", "bench_wpp"], ["SS UPP", "ss_upp", "bench_upp"], ["BL WPP", "bl_wpp", "bench_bl_wpp"], ["BL UPP", "bl_upp", "bench_bl_upp"]] as [lab, k, bk]}
      {@const bv = bench ? { bench_wpp: bench.wpp, bench_upp: bench.upp, bench_bl_wpp: bench.bl_wpp, bench_bl_upp: bench.bl_upp }[bk] : null}
      <div class="tile">
        <div class="lab">{lab}</div>
        <div class="big">{num0(data.totals[k])}</div>
        {#if bv != null}<div class="sub" title={bench.name}>bench {num0(bv)} ({((100 * data.totals[k]) / bv).toFixed(1)}%)</div>{/if}
      </div>
    {/each}
    <div class="note">
      best-possible: each level's best <b>completed</b> play of the whole run, per leaderboard (SS acc vs its API max score, BL vs its own; scoring v4) ·
      WPP = descending × 0.965<sup>i</sup>, UPP = plain sum · {data.totals.ss_levels.toLocaleString()} SS / {data.totals.bl_levels.toLocaleString()} BL levels played
    </div>
  </div>
  <div class="status">
    <b>{run.id}</b> · {data.rows.toLocaleString()} episodes, {data.completed.toLocaleString()} completed
    · {data.n_levels.toLocaleString()} levels
    {#if data.last_update != null}· last update {data.last_update.toLocaleString()}{/if}
    · ep_stats v{data.ep_layout} · levels {data.levels_source ?? "unresolved (SS ids only)"}
    {#if !data.swing}<span class="warn">swing columns unavailable: this run's ep_stats rows predate the per-hand swing log (2026-10-04)</span>
    {:else if !data.wreset}<span class="warn">wrist-reset columns unavailable: this run's ep_stats rows predate layout v3 (2026-10-04)</span>{/if}
  </div>

  <div class="tablewrap">
    <table>
      <thead>
        <tr>
          <th class="num">#</th>
          {#each COLS as c (c.key)}
            <th class:num={!c.text} class:dim={(c.swing && !data.swing) || (c.wrs && !data.wreset)} title={c.src === "win" && c.key !== "plays" && stat === "best" ? `${c.title ?? c.label} — of the window's best play (highest accuracy)` : c.title} onclick={() => sortBy(c.key)}>{c.label}{activeSort === c.key ? (sortDir > 0 ? " ▲" : " ▼") : ""}</th>
          {/each}
        </tr>
      </thead>
      <tbody>
        {#each order.slice(0, limit) as i, n (data.levels.key[i])}
          <tr>
            <td class="num muted">{n + 1}</td>
            {#each COLS as c (c.key)}
              {@const v = val(i, c)}
              {#if c.key === "song"}
                <td class="song" title={`${data.levels.mapper[i] ?? ""} · ${data.levels.key[i]}`}>{v ?? data.levels.key[i]}</td>
              {:else if c.key === "done"}
                <td class="num">{data.levels.done[i]}/{data.levels.att[i]}</td>
              {:else}
                <td class:num={!c.text} style={c.heat ? heat(v) : c.heatUp ? heatUp(v) : ""}>{c.f ? c.f(v) : (v ?? "—")}</td>
              {/if}
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <div class="more">
    showing {Math.min(limit, order.length).toLocaleString()} of {order.length.toLocaleString()} levels
    {#if limit < order.length}<button onclick={() => (limit += 500)}>show 500 more</button> <button onclick={() => (limit = order.length)}>all</button>{/if}
  </div>
{/if}

<style>
  .bar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 8px 0; }
  .wins button.on { background: var(--accent); color: #fff; }
  .auto { color: var(--muted); font-size: 12px; }
  select { max-width: 46vw; }
  .totals { display: flex; flex-wrap: wrap; gap: 10px; align-items: stretch; margin: 6px 0 8px; }
  .tile { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; min-width: 120px; }
  .tile .lab { color: var(--muted); font-size: 12px; }
  .tile .big { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .tile .sub { color: var(--muted); font-size: 11px; }
  .note { color: var(--muted); font-size: 12px; flex: 1 1 280px; align-self: center; }
  .status { font-size: 13px; color: var(--muted); margin: 4px 0 8px; }
  .status b { color: var(--fg); }
  .warn { display: inline-block; margin-left: 6px; background: color-mix(in srgb, var(--bad) 25%, transparent); color: var(--fg); border-radius: 4px; padding: 0 6px; }
  /* no scroll anchoring: new data reorders the keyed rows, and an anchored
     row would drag the page along; the scroll position stays put instead */
  .tablewrap { overflow-x: auto; overflow-anchor: none; }
  table { border-collapse: collapse; font-size: 12px; width: 100%; --bad: #e0534a; }
  .totals, .warn { --bad: #e0534a; }
  th, td { padding: 2px 6px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }
  th { cursor: pointer; position: sticky; top: 0; background: var(--bg); user-select: none; }
  th.dim { color: var(--muted); }
  th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.song { max-width: 280px; overflow: hidden; text-overflow: ellipsis; }
  tbody tr:hover { background: var(--panel); }
  .more { margin: 8px 0; color: var(--muted); display: flex; gap: 8px; align-items: center; }
</style>
