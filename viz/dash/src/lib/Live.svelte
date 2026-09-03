<script>
  import { store, loadIndex, loadMetrics, mostRecentRun, runById } from "./data.svelte.js";
  import { PANELS, X_MODES, xValues, seriesValues, hasAny, label, color, fmt,
    fmtDuration, fmtSamples } from "./series.js";
  import Chart from "./Chart.svelte";

  let { xMode = $bindable("samples"), runId = $bindable(null) } = $props();

  let auto = $state(true);
  let now = $state(Date.now());
  let refreshing = $state(false);

  const runs = $derived(store.index?.runs ?? []);
  const run = $derived(runId ? runById(runId) : mostRecentRun());
  const recs = $derived(run ? (store.metrics[run.id]?.recs ?? null) : null);
  const last = $derived(recs && recs.length ? recs[recs.length - 1] : null);
  const x = $derived(recs ? xValues(recs, xMode) : []);
  const xLabel = $derived(X_MODES.find((m) => m.key === xMode)?.label ?? xMode);

  // "T seconds ago": the record's own epoch stamp when it has one, else
  // the run's t_last from the index (log mtime)
  const lastT = $derived(last?.t ?? run?.t_last ?? null);
  const ago = $derived(lastT ? Math.max(0, now / 1000 - lastT) : null);

  const panels = $derived.by(() => {
    if (!recs || !recs.length) return [];
    return PANELS.map((p) => {
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
    if (!auto) return;
    const h = setInterval(refresh, 30000);
    return () => clearInterval(h);
  });
</script>

<div class="bar">
  <label>run
    <select value={run?.id ?? ""} onchange={(e) => { runId = e.target.value || null; }}>
      <option value="">most recent ({mostRecentRun()?.id ?? "—"})</option>
      {#each runs as r (r.id)}
        <option value={r.id}>{r.id} · {fmtSamples(r.samples)} · {r.status}</option>
      {/each}
    </select>
  </label>
  <span class="xmode">
    x:
    {#each X_MODES as m}
      <button class:on={xMode === m.key} onclick={() => (xMode = m.key)}>{m.label}</button>
    {/each}
  </span>
  <label class="auto"><input type="checkbox" bind:checked={auto} /> auto-refresh 30 s</label>
  <button onclick={refresh} disabled={refreshing}>{refreshing ? "…" : "refresh now"}</button>
</div>

{#if run}
  <div class="status">
    <b>{run.id}</b>
    {#if last}
      · last update <b>{last.upd.toLocaleString()}</b>{#if last.n_upd} / {last.n_upd.toLocaleString()}{/if}
      · {fmtSamples(last.steps)} samples
      · <b>{ago == null ? "—" : fmtDuration(ago)} ago</b>
      · acc {fmt(last.acc)} · return {fmt(last.return, 3)} · ent {fmt(last.ent, 3)} · ev {fmt(last.ev, 3)}
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
        <Chart title={p.title} datasets={p.datasets} {xLabel} log={p.log} syncKey={"live-" + xMode} />
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
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 10px; }
  .cell { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 6px; }
  .xmode button.on { background: var(--accent); color: #fff; }
  select { max-width: 52vw; }
</style>
