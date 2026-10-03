<script>
  import { store, loadIndex, loadMetrics, mostRecentRun, runById, REFRESH_MS } from "./data.svelte.js";
  import { X_MODES, xValues, seriesValues, hasAny, label, color, fmt,
    fmtDuration, fmtSamples, gameOf, panelsFor } from "./series.js";
  import Chart from "./Chart.svelte";

  let { xMode = $bindable("samples"), runId = $bindable(null) } = $props();

  let now = $state(Date.now());
  let refreshing = $state(false);

  const runs = $derived(store.index?.runs ?? []);
  const run = $derived(runId ? runById(runId) : mostRecentRun());
  const recs = $derived(run ? (store.metrics[run.id]?.recs ?? null) : null);
  const last = $derived(recs && recs.length ? recs[recs.length - 1] : null);
  const x = $derived(recs ? xValues(recs, xMode) : []);
  const xLabel = $derived(X_MODES.find((m) => m.key === xMode)?.label ?? xMode);

  // Window lock (2026-09-14, user request): pin every chart's x range to
  // the most recent W units of the current x mode so a refresh never
  // resets the zoom. W is entered in the mode's display unit (M samples,
  // updates, hours) and remembered per mode in localStorage; 0 = all.
  const WIN_UNIT = { samples: { label: "M samples", mult: 1e6 }, updates: { label: "updates", mult: 1 }, wall: { label: "h", mult: 1 } };
  const WIN_KEY = "dash.live.window";
  function loadWin() {
    try { const v = JSON.parse(localStorage.getItem(WIN_KEY) || "{}"); return { samples: 0, updates: 0, wall: 0, ...v }; }
    catch { return { samples: 0, updates: 0, wall: 0 }; }
  }
  let win = $state(loadWin());
  function setWin(v) {
    const n = Math.max(0, Number(v) || 0);
    win = { ...win, [xMode]: n };
    try { localStorage.setItem(WIN_KEY, JSON.stringify(win)); } catch {}
  }
  const xRange = $derived.by(() => {
    const W = (win[xMode] || 0) * WIN_UNIT[xMode].mult;
    if (!W || !x.length) return null;
    const hi = x[x.length - 1];
    return [Math.max(x[0], hi - W), hi];
  });

  // "T seconds ago": the record's own epoch stamp when it has one, else
  // the run's t_last from the index (log mtime)
  const lastT = $derived(last?.t ?? run?.t_last ?? null);
  const ago = $derived(lastT ? Math.max(0, now / 1000 - lastT) : null);

  const panels = $derived.by(() => {
    if (!recs || !recs.length) return [];
    return panelsFor(gameOf(run)).map((p) => {
      const keys = p.series.filter((k) => hasAny(recs, k));
      return {
        ...p,
        datasets: keys.map((k, i) => ({ label: label(k), x, y: seriesValues(recs, k), color: color(i) })),
      };
    });
  });

  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      await loadIndex();
      const r = runId ? runById(runId) : mostRecentRun();
      if (r) await loadMetrics(r, { force: true });
    } finally {
      refreshing = false;
      now = Date.now();
    }
  }

  $effect(() => {
    if (run && !store.metrics[run.id]) loadMetrics(run);
  });

  $effect(() => {
    const tick = setInterval(() => (now = Date.now()), 1000);
    return () => clearInterval(tick);
  });

  $effect(() => {
    const h = setInterval(refresh, REFRESH_MS);  // always on
    return () => clearInterval(h);
  });
</script>

<div class="bar">
  <label>run
    <select value={run?.id ?? ""} onchange={(e) => { runId = e.target.value || null; }}>
      <option value="">most recent ({mostRecentRun()?.id ?? "—"})</option>
      {#each runs as r (r.id)}
        <option value={r.id}>{r.id}{gameOf(r) === "flux" ? " · FLUX" : ""} · {fmtSamples(r.samples)} · {r.status}</option>
      {/each}
    </select>
  </label>
  <span class="xmode">
    x:
    {#each X_MODES as m}
      <button class:on={xMode === m.key} onclick={() => (xMode = m.key)}>{m.label}</button>
    {/each}
  </span>
  <label class="win" title="pin the x axis to the most recent window; 0 = whole run. Remembered per x mode. A manual drag-zoom also survives refreshes now.">
    window: last
    <input type="number" min="0" step="any" value={win[xMode]} onchange={(e) => setWin(e.target.value)} />
    {WIN_UNIT[xMode].label}
    <button class:on={!win[xMode]} onclick={() => setWin(0)}>all</button>
  </label>
  <span class="auto" title="auto-refreshes every 15 s">{refreshing ? "refreshing…" : "auto 15 s"}</span>
</div>

{#if run}
  <div class="status">
    <b>{run.id}</b>
    {#if gameOf(run) === "flux"}<span class="game">FLUX</span>{/if}
    {#if last}
      · last update <b>{last.upd.toLocaleString()}</b>{#if last.n_upd} / {last.n_upd.toLocaleString()}{/if}
      · {fmtSamples(last.steps)} samples
      · <b>{ago == null ? "—" : fmtDuration(ago)} ago</b>
      · {gameOf(run) === "flux" ? "pts/touch÷115" : "acc"} {fmt(last.acc)} · return {fmt(last.return, 3)} · ent {fmt(last.ent, 3)} · ev {fmt(last.ev, 3)}
      · {fmt(last.sps / 1000, 3)}k steps/s
    {:else if store.loading[run.id]}
      · loading metrics…
    {:else}
      · no records
    {/if}
    · status {run.status} · {run.campaign} · N={run.num_envs} T={run.horizon}
    {#if run.peak_vram}· peak VRAM {run.peak_vram} GiB{/if}
  </div>

  <div class="grid">
    {#each panels as p (p.id)}
      <div class="cell">
        <Chart title={p.title} datasets={p.datasets} {xLabel} log={p.log} syncKey={"live-" + xMode} {xRange} resetKey={run.id} />
      </div>
    {/each}
  </div>
{:else}
  <p class="muted">no runs in index.json</p>
{/if}

<style>
  .bar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 8px 0; }
  .status { font-size: 13px; color: var(--muted); margin: 4px 0 10px; }
  .status b { color: var(--fg); }
  .game { background: #8b5cf6; color: #fff; border-radius: 4px; padding: 0 5px; margin-left: 4px; font-size: 11px; font-weight: 700; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 10px; }
  .cell { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 6px; }
  .xmode button.on, .win button.on { background: var(--accent); color: #fff; }
  .win input { width: 6em; }
  select { max-width: 52vw; }
</style>
