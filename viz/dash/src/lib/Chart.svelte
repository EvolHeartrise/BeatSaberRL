<script>
  // uPlot wrapper. `datasets`: [{label, x: number[], y: (number|null)[],
  // color, dash?}] — each on its own x grid; uPlot.join aligns them on
  // the union with nulls, spanGaps draws through. Charts sharing a
  // `syncKey` share the cursor.
  import uPlot from "uplot";

  let {
    title = "",
    datasets = [],
    xLabel = "",
    log = false,
    height = 230,
    syncKey = "dash",
    legendValues = true,
  } = $props();

  let el;
  let u = null;

  function css(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function build() {
    if (!el) return;
    if (u) { u.destroy(); u = null; }
    const valid = datasets.filter((d) => d && d.x && d.x.length);
    if (!valid.length) return;
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

  $effect(() => {
    // re-read every reactive input so the chart rebuilds on change
    void datasets; void title; void log; void height; void xLabel;
    build();
    return () => { if (u) { u.destroy(); u = null; } };
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
