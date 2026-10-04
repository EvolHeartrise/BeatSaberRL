<script>
  import { store, archiveUrl } from "./data.svelte.js";
  import { fmt, fmtDuration, fmtSamples, gameOf } from "./series.js";

  let { onpick = () => {}, selected = [] } = $props();

  let sortKey = $state("t_last");
  let sortDir = $state(-1);
  let filter = $state("");

  const COLS = [
    { key: "campaign", label: "campaign" },
    { key: "run", label: "run" },
    { key: "game", label: "game" },
    { key: "status", label: "status" },
    { key: "samples", label: "samples", num: true, fmt: fmtSamples },
    { key: "updates", label: "updates", num: true, fmt: (v) => v?.toLocaleString() },
    { key: "wall_s", label: "wall", num: true, fmt: fmtDuration },
    { key: "sps", label: "steps/s", num: true, fmt: (v) => (v == null ? "—" : Math.round(v).toLocaleString()) },
    { key: "peak_vram", label: "VRAM GiB", num: true, fmt: (v) => (v == null ? "—" : v.toFixed(2)) },
    { key: "final.acc", label: "acc", num: true, fmt: (v) => fmt(v) },
    { key: "final.return", label: "return", num: true, fmt: (v) => fmt(v, 3) },
    { key: "final.ent", label: "ent", num: true, fmt: (v) => fmt(v, 3) },
    { key: "bench.upp", label: "bench upp", num: true, fmt: (v) => (v == null ? "—" : Math.round(v).toLocaleString()) },
    { key: "bench.wpp", label: "bench wpp", num: true, fmt: (v) => (v == null ? "—" : v.toFixed(0)) },
    { key: "swing.ss_upp", label: "train best SS upp", num: true, fmt: (v) => (v == null ? "—" : Math.round(v).toLocaleString()) },
    { key: "swing.bl_upp", label: "train best BL upp", num: true, fmt: (v) => (v == null ? "—" : Math.round(v).toLocaleString()) },
    { key: "bench.wrev_l", label: "wrist rev L /s", num: true, fmt: (v) => (v == null ? "—" : v.toFixed(2)) },
    { key: "bench.wrev_r", label: "wrist rev R /s", num: true, fmt: (v) => (v == null ? "—" : v.toFixed(2)) },
    { key: "bench.wrev_total", label: "wrist rev total", num: true, fmt: (v) => (v == null ? "—" : Math.round(v).toLocaleString()) },
    { key: "ckpts", label: "ckpts", num: true },
    { key: "t_last", label: "last seen", num: true, fmt: (v) => (v ? new Date(v * 1000).toLocaleString() : "—") },
  ];

  function get(r, key) {
    if (key === "ckpts") return r.checkpoints?.length ?? 0;
    if (key === "game") return gameOf(r);
    if (key.startsWith("final.")) return r.final?.[key.slice(6)] ?? null;
    if (key.startsWith("swing.")) return r.swing?.totals?.[key.slice(6)] ?? null;
    if (key.startsWith("bench.")) {
      const b = bestBench(r);
      return b ? b[key.slice(6)] : null;
    }
    return r[key];
  }
  // the bench with the highest unweighted PP (several checkpoints may
  // have been benched)
  function bestBench(r) {
    let best = null;
    for (const b of r.bench ?? []) if (!best || (b.upp ?? -1) > (best.upp ?? -1)) best = b;
    return best;
  }

  const rows = $derived.by(() => {
    const f = filter.trim().toLowerCase();
    const rs = (store.index?.runs ?? []).filter((r) => !f || r.id.toLowerCase().includes(f)
      || gameOf(r) === f);
    const col = COLS.find((c) => c.key === sortKey);
    return rs.slice().sort((a, b) => {
      const va = get(a, sortKey), vb = get(b, sortKey);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      const c = col?.num ? va - vb : String(va).localeCompare(String(vb));
      return c * sortDir;
    });
  });

  function sortBy(key) {
    if (sortKey === key) sortDir = -sortDir;
    else { sortKey = key; sortDir = COLS.find((c) => c.key === key)?.num ? -1 : 1; }
  }

  function replayDir(r) {
    const p = r.replays?.[0];
    if (!p) return null;
    return archiveUrl(p.slice(0, p.lastIndexOf("/") + 1));
  }
</script>

<div class="bar">
  <input placeholder="filter… (run id, or 'flux')" bind:value={filter} />
  <span class="muted">{rows.length} runs · click a row to add it to Compare · generated {store.index ? new Date(store.index.generated * 1000).toLocaleString() : "—"}</span>
</div>
<div class="tablewrap">
  <table>
    <thead>
      <tr>
        {#each COLS as c}
          <th class:num={c.num} onclick={() => sortBy(c.key)}>{c.label}{sortKey === c.key ? (sortDir > 0 ? " ▲" : " ▼") : ""}</th>
        {/each}
        <th>links</th>
      </tr>
    </thead>
    <tbody>
      {#each rows as r (r.id)}
        {@const bb = bestBench(r)}
        <tr class:sel={selected.includes(r.id)} onclick={() => onpick(r.id)}>
          {#each COLS as c}
            {@const v = get(r, c.key)}
            <td class:num={c.num} title={c.key === "run" ? r.id : undefined}>{c.fmt ? c.fmt(v) : (v ?? "—")}</td>
          {/each}
          <td class="links" onclick={(e) => e.stopPropagation()}>
            <a href={archiveUrl(r.metrics)} target="_blank">metrics</a>
            {#if r.logs?.length}· <a href={archiveUrl(r.logs[0])} target="_blank">log</a>{/if}
            {#if bb}
              · {#if bb.url}<a href={archiveUrl(bb.url)} target="_blank">bench {bb.ckpt ?? ""}</a>{:else}<span title={bb.path}>bench {bb.ckpt ?? ""} (repo)</span>{/if}
              {#if r.bench.length > 1}<small>(+{r.bench.length - 1})</small>{/if}
            {/if}
            {#if r.replays?.length}
              · <a href={replayDir(r)} target="_blank">{r.replays.length} replays</a>
              · <a href="/replay.html" target="_blank">viewer</a>
            {/if}
            {#if r.run_dir}· <a href={archiveUrl(r.run_dir + "/")} target="_blank">dir</a>{/if}
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .bar { display: flex; gap: 12px; align-items: center; margin: 8px 0; }
  .tablewrap { overflow-x: auto; }
  table { border-collapse: collapse; font-size: 12px; width: 100%; }
  th, td { padding: 3px 7px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }
  th { cursor: pointer; position: sticky; top: 0; background: var(--bg); user-select: none; }
  th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
  tbody tr { cursor: pointer; }
  tbody tr:hover { background: var(--panel); }
  tbody tr.sel { background: color-mix(in srgb, var(--accent) 18%, transparent); }
  .links a { color: var(--accent); }
</style>
