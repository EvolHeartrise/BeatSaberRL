<script>
  // uPlot wrapper. `datasets`: [{label, x: number[], y: (number|null)[],
  // color, dash?}] — each on its own x grid; uPlot.join aligns them on
  // the union with nulls, spanGaps draws through. Charts sharing a
  // `syncKey` share the cursor.
  //
  // Data refreshes UPDATE the existing plot instead of rebuilding it
  // (2026-09-14, user request): a manual drag-zoom survives the live
  // tab's 30 s refresh, and `xRange` ([min, max] or null) pins the x
  // scale to a window, e.g. "the last 250 updates", re-applied after
  // every data update so the view follows the run.
  import uPlot from "uplot";

  let {
    title = "",
    datasets = [],
    xLabel = "",
    log = false,
    height = 230,
    syncKey = "dash",
    legendValues = true,
    xRange = null,
  } = $props();

  let el;
  let u = null;
  let sig = "";           // series structure the live plot was built for
  let xExtent = [0, 0];   // full data extent at the last (re)build/update

  function css(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function validOf(ds) {
    return ds.filter((d) => d && d.x && d.x.length);
  }
  function signature(valid) {
    return JSON.stringify([title, xLabel, log, height, syncKey, valid.map((d) => [d.label, d.color, d.dash ?? null])]);
  }
  function extentOf(data) {
    const xs = data[0];
    return xs.length ? [xs[0], xs[xs.length - 1]] : [0, 0];
  }

  function build(valid) {
    if (u) { u.destroy(); u = null; }
    if (!el || !valid.length) return;
    const data = uPlot.join(valid.map((d) => [d.x, d.y]));
    const fg = css("--fg", "#222");
    const grid = css("--grid", "rgba(128,128,128,0.25)");
    const series = [
      { label: xLabel || "x" },
      ...valid.map((d) => ({
        label: d.label,
        stroke: d.color,
        width: d.width ?? 1.2,
        dash: d.dash,
        spanGaps: true,
        points: { show: false },
        value: (self, v) => (v == null ? "—" : fmtVal(v)),
      })),
    ];
    const opts = {
      title,
      width: el.clientWidth || 600,
      height,
      cursor: { sync: { key: syncKey, setSeries: false }, drag: { x: true, y: false } },
      legend: { show: true, live: legendValues },
      scales: { x: { time: false }, y: log ? { distr: 3 } : {} },
      axes: [
        { stroke: fg, grid: { stroke: grid }, ticks: { stroke: grid },
          values: (self, ticks) => ticks.map(fmtTick) },
        { stroke: fg, grid: { stroke: grid }, ticks: { stroke: grid }, size: 58,
          values: (self, ticks) => ticks.map(fmtTick) },
      ],
      series,
    };
    u = new uPlot(opts, data, el);
    xExtent = extentOf(data);
  }

  // Same series structure, new points: keep the plot. If the user had
  // the full extent in view, let uPlot re-range so the new tail shows;
  // if they had zoomed, keep their window (resetScales = false).
  function update(valid) {
    const data = uPlot.join(valid.map((d) => [d.x, d.y]));
    const eps = 1e-9;
    const sx = u.scales.x;
    const wasFull = sx.min == null || sx.max == null
      || (sx.min <= xExtent[0] + eps && sx.max >= xExtent[1] - eps);
    u.setData(data, wasFull);
    xExtent = extentOf(data);
  }

  function applyRange() {
    if (!u || !xRange) return;
    const [lo, hi] = xRange;
    if (lo == null || hi == null || !(hi > lo)) return;
    u.setScale("x", { min: lo, max: hi });
  }

  function fmtTick(v) {
    if (v == null) return "";
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(1) + "B";
    if (a >= 1e6) return (v / 1e6).toFixed(0) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(0) + "k";
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return +v.toFixed(2) + "";
    if (a === 0) return "0";
    if (a < 1e-3) return v.toExponential(0);
    return +v.toFixed(4) + "";
  }
  function fmtVal(v) {
    const a = Math.abs(v);
    if (a >= 1e6) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (a >= 100) return v.toFixed(1);
    if (a < 1e-3 && a > 0) return v.toExponential(2);
    return v.toFixed(4);
  }

  // data / structure / window changes: update in place when only the
  // points changed, rebuild when the series set or the options did
  $effect(() => {
    void xRange;
    const valid = validOf(datasets);
    const s = signature(valid);
    if (!el) return;
    if (u && s === sig) update(valid);
    else { build(valid); sig = s; }
    applyRange();
  });

  // lifetime: destroy on unmount only (not on every data change)
  $effect(() => {
    void el;
    return () => { if (u) { u.destroy(); u = null; sig = ""; } };
  });

  $effect(() => {
    if (!el) return;
    const ro = new ResizeObserver(() => {
      if (u && el.clientWidth) u.setSize({ width: el.clientWidth, height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  });
</script>

<div class="chart" bind:this={el}>
  {#if !datasets.some((d) => d && d.x && d.x.length)}
    <div class="empty">{title}<br /><small>no data for this metric</small></div>
  {/if}
</div>

<style>
  .chart { width: 100%; min-height: 60px; }
  .empty { color: var(--muted); font-size: 12px; padding: 12px 8px; }
  :global(.uplot .u-title) { font-size: 13px; font-weight: 600; color: var(--fg); }
  :global(.uplot .u-legend) { font-size: 11px; color: var(--fg); }
  :global(.uplot .u-legend .u-value) { min-width: 0; }
</style>
