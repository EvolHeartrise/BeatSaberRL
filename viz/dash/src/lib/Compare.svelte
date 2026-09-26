<script>
  import { store, loadIndex, loadMetrics, runById } from "./data.svelte.js";
  import { COMPARE_KEYS, DIFF_KEYS, X_MODES, xValues, seriesValues, hasAny,
    label, color, interp, rolling, atSamples, fmt, fmtSamples, parseSampleList } from "./series.js";
  import Chart from "./Chart.svelte";

  let { selected = $bindable([]), xMode = $bindable("samples") } = $props();

  let filter = $state("");
  let refId = $state(null);
  let window_ = $state(25);
  let sampleText = $state("");

  const runs = $derived(store.index?.runs ?? []);
  const shown = $derived.by(() => {
    const f = filter.trim().toLowerCase();
    return runs.filter((r) => !f || r.id.toLowerCase().includes(f)
      || (r.game ?? "standard") === f);
  });
  const sel = $derived(selected.map((id) => runById(id)).filter(Boolean));
  const loaded = $derived(sel.filter((r) => store.metrics[r.id]?.recs?.length));
  const ref = $derived((refId && loaded.find((r) => r.id === refId)) || loaded[0] || null);
  const xLabel = $derived(X_MODES.find((m) => m.key === xMode)?.label ?? xMode);

  // one x array per loaded run, per mode
  const xs = $derived.by(() => {
    const m = {};
    for (const r of loaded) m[r.id] = xValues(store.metrics[r.id].recs, xMode);
    return m;
  });

  const smalls = $derived.by(() =>
    COMPARE_KEYS.map((k) => ({
      key: k,
      datasets: loaded
        .filter((r) => hasAny(store.metrics[r.id].recs, k))
        .map((r, i) => ({
          label: r.run, x: xs[r.id], y: seriesValues(store.metrics[r.id].recs, k),
          color: color(loaded.indexOf(r)),
        })),
    })).filter((p) => p.datasets.length));

  // difference vs the reference at matched x: interpolate the reference
  // onto each run's own grid, subtract, rolling-mean over `window_` points
  const diffs = $derived.by(() => {
    if (!ref) return [];
    const rx = xs[ref.id];
    const rrecs = store.metrics[ref.id].recs;
    return DIFF_KEYS.map((k) => {
      const ry = seriesValues(rrecs, k);
      const datasets = [];
      for (const r of loaded) {
        if (r.id === ref.id) continue;
        const recs = store.metrics[r.id].recs;
        if (!hasAny(recs, k) || !hasAny(rrecs, k)) continue;
        const x = xs[r.id];
        const y = seriesValues(recs, k);
        const d = new Array(x.length);
        for (let i = 0; i < x.length; i++) {
          const rv = interp(rx, ry, x[i]);
          d[i] = rv == null || y[i] == null ? null : y[i] - rv;
        }
        datasets.push({ label: `${r.run} − ${ref.run}`, x, y: rolling(d, window_), color: color(loaded.indexOf(r)) });
      }
      return { key: k, datasets };
    }).filter((p) => p.datasets.length);
  });

  // matched-checkpoint table at chosen sample counts (default: the
  // reference's checkpointed sample counts, or 5 even splits)
  const samplePoints = $derived.by(() => {
    const typed = parseSampleList(sampleText);
    if (typed.length) return typed;
    if (!ref) return [];
    const recs = store.metrics[ref.id].recs;
    const ck = recs.filter((r) => r.ckpt).map((r) => r.steps);
    if (ck.length >= 2 && ck.length <= 40) return ck;
    const last = recs[recs.length - 1].steps;
    return [1, 2, 3, 4, 5].map((i) => Math.round((last * i) / 5));
  });
  const TABLE_KEYS = ["acc", "return", "ent", "ev"];

  function toggle(id) {
    selected = selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id];
  }

  $effect(() => {
    for (const r of sel) if (!store.metrics[r.id] && !store.loading[r.id]) loadMetrics(r);
  });

  // auto-refresh every 15 s: re-read the index and re-fetch the selected runs
  // that are still being written (finished/crashed runs never change).
  let refreshing = $state(false);
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      await loadIndex();
      const live = selected.map((id) => runById(id)).filter((r) => r && r.status !== "finished" && r.status !== "crashed");
      await Promise.all(live.map((r) => loadMetrics(r, { force: true })));
    } finally {
      refreshing = false;
    }
  }
  $effect(() => {
    const h = setInterval(refresh, 15000);
    return () => clearInterval(h);
  });
</script>

<div class="layout">
  <aside>
    <input placeholder="filter runs…" bind:value={filter} />
    <div class="list">
      {#each shown as r (r.id)}
        <label class:sel={selected.includes(r.id)}>
          <input type="checkbox" checked={selected.includes(r.id)} onchange={() => toggle(r.id)} />
          <span class="dot" style:background={selected.includes(r.id) ? color(selected.indexOf(r.id)) : "transparent"}></span>
          <span class="name">{r.id}{#if r.game === "flux"} <b class="flux">FLUX</b>{/if}</span>
          <span class="meta">{fmtSamples(r.samples)}</span>
        </label>
      {/each}
    </div>
    {#if selected.length}
      <button onclick={() => (selected = [])}>clear ({selected.length})</button>
    {/if}
  </aside>

  <main>
    <div class="bar">
      <span class="xmode">align on:
        {#each X_MODES as m}
          <button class:on={xMode === m.key} onclick={() => (xMode = m.key)}>{m.label}</button>
        {/each}
      </span>
      <label>reference
        <select value={ref?.id ?? ""} onchange={(e) => (refId = e.target.value || null)}>
          {#each loaded as r (r.id)}<option value={r.id}>{r.id}</option>{/each}
        </select>
      </label>
      <label>rolling window <input type="number" min="1" max="1000" bind:value={window_} style="width:5em" /> pts</label>
    </div>

    {#if !sel.length}
      <p class="muted">pick runs on the left (or click a row in Runs). Series align on {xLabel}; the difference panel subtracts the reference at matched x.</p>
    {:else if !loaded.length}
      <p class="muted">loading…</p>
    {:else}
      <h3>small multiples — {loaded.length} run{loaded.length === 1 ? "" : "s"}</h3>
      <div class="grid">
        {#each smalls as p (p.key)}
          <div class="cell"><Chart title={label(p.key)} datasets={p.datasets} {xLabel} log={p.key === "kl_last_pos" || p.key === "falls_per_1m_pos" || p.key === "tilts_per_1m_pos"} height={200} syncKey={"cmp-" + xMode} /></div>
        {/each}
      </div>

      {#if ref && loaded.length > 1}
        <h3>difference vs {ref.run} (rolling mean, {window_} pts)</h3>
        <div class="grid">
          {#each diffs as p (p.key)}
            <div class="cell"><Chart title={"Δ " + label(p.key)} datasets={p.datasets} {xLabel} height={200} syncKey={"cmp-" + xMode} /></div>
          {/each}
        </div>
      {/if}

      <h3>matched samples</h3>
      <div class="bar">
        <label>sample counts <input placeholder="e.g. 250M, 500M, 1B (default: reference checkpoints)" bind:value={sampleText} style="width:28em" /></label>
      </div>
      <div class="tablewrap">
        <table>
          <thead>
            <tr><th>samples</th><th>run</th>{#each TABLE_KEYS as k}<th>{label(k)}</th>{/each}<th>upd</th><th>ckpt</th></tr>
          </thead>
          <tbody>
            {#each samplePoints as s}
              {#each loaded as r, i (r.id)}
                {@const rec = atSamples(store.metrics[r.id].recs, s)}
                <tr class:first={i === 0}>
                  <td>{i === 0 ? fmtSamples(s) : ""}</td>
                  <td><span class="dot" style:background={color(i)}></span>{r.run}</td>
                  {#each TABLE_KEYS as k}
                    <td class="num">{rec ? fmt(rec[k]) : "—"}{#if rec && ref && r.id !== ref.id}
                      {@const rr = atSamples(store.metrics[ref.id].recs, s)}
                      {#if rr && rec[k] != null && rr[k] != null}<small class="delta">{rec[k] - rr[k] >= 0 ? "+" : ""}{fmt(rec[k] - rr[k])}</small>{/if}
                    {/if}</td>
                  {/each}
                  <td class="num">{rec ? rec.upd : "—"}</td>
                  <td>{rec?.ckpt ?? ""}</td>
                </tr>
              {/each}
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </main>
</div>

<style>
  .flux { color: #8b5cf6; font-size: 10px; }
  .layout { display: grid; grid-template-columns: 300px 1fr; gap: 14px; }
  aside { border-right: 1px solid var(--border); padding-right: 10px; }
  aside > input { width: 100%; box-sizing: border-box; }
  .list { max-height: 75vh; overflow: auto; margin: 6px 0; font-size: 12px; }
  .list label { display: flex; gap: 6px; align-items: center; padding: 2px 2px; cursor: pointer; }
  .list label.sel { background: var(--panel); }
  .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .meta { color: var(--muted); }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 5px; margin-right: 6px; border: 1px solid var(--border); }
  .bar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 6px 0; }
  .xmode button.on { background: var(--accent); color: #fff; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 10px; }
  .cell { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 6px; }
  h3 { margin: 14px 0 6px; font-size: 14px; }
  .tablewrap { overflow-x: auto; }
  table { border-collapse: collapse; font-size: 12px; }
  th, td { padding: 3px 8px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  tr.first td { border-top: 2px solid var(--border); }
  .delta { color: var(--muted); margin-left: 4px; }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } aside { border-right: 0; } }
</style>
