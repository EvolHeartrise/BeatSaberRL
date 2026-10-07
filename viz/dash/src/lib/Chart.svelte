<script module>
  // a drag-zoom in progress: off-screen charts must follow its synced
  // cursor moves too (the zoom is applied from them)
  let dragging = false;
  if (typeof window !== "undefined") {
    addEventListener("mousedown", () => { dragging = true; }, true);
    addEventListener("mouseup", () => { dragging = false; }, true);
  }
</script>

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
  //
  // An auto-refresh that brought nothing new does not touch the plot at
  // all (2026-10-02): the points are compared with the ones plotted
  // (data.svelte.js already skips unchanged files, this is the backstop
  // for re-derived but identical arrays). `resetKey` (Live: the run id)
  // rebuilds the plot, so a different run never inherits a zoom.
  //
  // Compact card layout (2026-10-06, user request): the title floats on
  // the plot's top-left corner (moved into uPlot's .u-over, so it tracks
  // the plot area through resizes; pointer-events: none), the x axis is
  // tight and the y axis is sized to its widest tick label. Hover values
  // never reflow the card: the legend below is static (names + colours,
  // click toggles a series) and the values at the cursor go to a floating
  // tooltip (position: fixed, clamped to the viewport, so it never
  // changes layout or scroll height), series sorted by value with
  // swatches; only the hovered chart shows it, the synced ones just
  // draw the cursor line.
  import uPlot from "uplot";
  import { schedule, cancel } from "./sched.js";

  let {
    title = "",
    datasets = [],
    xLabel = "",
    log = false,
    height = 230,
    syncKey = "dash",
    legendValues = true,
    xRange = null,
    resetKey = "",
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
    return JSON.stringify([resetKey, title, xLabel, log, height, syncKey, valid.map((d) => [d.label, d.color, d.dash ?? null])]);
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
    const tip = legendValues ? tooltip(valid) : null;
    const opts = {
      title,
      width: el.clientWidth || 600,
      height,
      cursor: {
        sync: { key: syncKey, setSeries: false, filters: { pub: () => true, sub: (type) => visible || type !== "mousemove" || dragging } },
        drag: { x: true, y: false }, dataIdx,
      },
      // static legend: hover values live in the tooltip, never here
      legend: { show: true, live: false },
      scales: { x: { time: false }, y: log ? { distr: 3 } : {} },
      axes: [
        { stroke: fg, grid: { stroke: grid }, ticks: { stroke: grid, size: 4 }, gap: 3, size: 24,
          values: (self, ticks) => ticks.map(fmtTick) },
        { stroke: fg, grid: { stroke: grid }, ticks: { stroke: grid, size: 4 }, gap: 4, size: yAxisSize,
          values: (self, ticks) => ticks.map(fmtTick) },
      ],
      series,
      hooks: tip ? { setCursor: [tip.update], setSeries: [tip.update], setSize: [tip.dropRect], destroy: [tip.detach] } : {},
    };
    holdX(valid);
    u = new uPlot(opts, data, el);
    // the title floats on the plot area (CSS: .u-over > .u-title)
    const t = u.root.querySelector(".u-title");
    if (t) u.over.appendChild(t);
    if (tip) tip.attach(u);
    el.__uplot = u; // test hook (test/refresh-browser.mjs)
    xExtent = extentOf(data);
  }

  // Hover index per series: on a joined grid (Compare: every run on its
  // own x) a series is undefined at most cursor positions, and uPlot's
  // default scans the joined arrays for its nearest point — per series,
  // per synced chart, per mouse move: 60-120 ms a move with 16 runs on
  // 40 charts. Two binary searches instead: the nearest x of the series'
  // own grid, then that x's slot on the joined grid.
  let ownX = [null];
  let sharedX = true;
  function holdX(valid) {
    ownX = [null, ...valid.map((d) => d.x)];
    sharedX = valid.every((d) => d.x === valid[0].x);
  }
  function dataIdx(self, si, ci) {
    if (si === 0 || sharedX || ci == null) return ci;
    const ys = self.data[si];
    if (ys[ci] !== undefined) return ci;
    const xs = ownX[si], X = self.data[0];
    if (!xs || !xs.length) return null;
    const xv = X[ci];
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= xv) lo = mid;
      else hi = mid;
    }
    const xj = Math.abs(xs[hi] - xv) < Math.abs(xv - xs[lo]) ? xs[hi] : xs[lo];
    let a = 0, b = X.length - 1;
    while (a < b) {
      const mid = (a + b) >> 1;
      if (X[mid] < xj) a = mid + 1;
      else b = mid;
    }
    return X[a] === xj ? a : null;
  }

  // y axis just wide enough for its widest tick label (was a fixed 58 px).
  // Within one size-convergence pass it only grows: a later cycle can
  // bring more ticks (e.g. 0.25 steps once the plot got taller), and
  // shrinking back would clip them.
  function yAxisSize(self, values, axisIdx, cycleNum) {
    const axis = self.axes[axisIdx];
    let w = 0;
    if (values && values.length) {
      self.ctx.font = axis.font[0];
      for (const v of values) if (v != null && v !== "") w = Math.max(w, self.ctx.measureText(String(v)).width);
      w /= uPlot.pxRatio;
    }
    const size = Math.max(28, Math.ceil((axis.ticks?.size ?? 4) + (axis.gap ?? 4) + 5 + w));
    return cycleNum > 1 ? Math.max(size, axis._size ?? 0) : size;
  }

  // floating hover readout: one row per series (swatch, name, value),
  // sorted by value (top line first), x at the cursor in the header.
  // position: fixed + viewport clamp = no effect on layout or scroll size.
  function tooltip(valid) {
    const box = document.createElement("div");
    box.className = "dash-tip";
    const head = document.createElement("div");
    head.className = "dash-tip-head";
    box.appendChild(head);
    const list = document.createElement("div");
    list.className = "dash-tip-rows";
    box.appendChild(list);
    const rows = valid.map((d, k) => {
      const row = document.createElement("div");
      row.className = "dash-tip-row";
      const sw = document.createElement("span");
      sw.className = "dash-tip-sw";
      sw.style.borderColor = d.color;
      if (d.dash) sw.style.borderTopStyle = "dashed";
      const name = document.createElement("span");
      name.className = "dash-tip-name";
      name.textContent = d.label;
      const val = document.createElement("span");
      val.className = "dash-tip-val";
      row.append(sw, name, val);
      return { row, val, si: k + 1, v: null };
    });
    let hovering = false;
    let order = "";
    let plot = null;
    function hide() { if (box.style.display !== "none") box.style.display = "none"; at = null; }
    function update(self) {
      const c = self.cursor;
      if (!hovering || c.idx == null || c.left == null || c.left < 0) return hide();
      // no layout reads per move (each one forced a synchronous style +
      // layout pass after uPlot's cursor writes): the plot's rect is
      // cached (dropped on scroll / resize / re-enter), the box size
      // comes from a ResizeObserver, unchanged text is not rewritten
      if (!rect) rect = self.over.getBoundingClientRect();
      for (const r of rows) {
        const j = c.idxs?.[r.si] ?? c.idx;
        const v = j == null ? null : self.data[r.si][j];
        r.v = v == null || !Number.isFinite(v) ? null : v;
        const t = r.v == null ? "—" : fmtVal(r.v);
        if (r.text !== t) r.val.textContent = r.text = t;
        const d = self.series[r.si].show ? "" : "none";
        if (r.row.style.display !== d) r.row.style.display = d;
      }
      const sorted = [...rows].sort((a, b) => (b.v ?? -Infinity) - (a.v ?? -Infinity) || a.si - b.si);
      const key = sorted.map((r) => r.si).join(",");
      if (key !== order) { for (const r of sorted) list.appendChild(r.row); order = key; }
      const ht = `${xLabel || "x"} ${fmtX(self.data[0][c.idx])}`;
      if (head.textContent !== ht) head.textContent = ht;
      if (box.style.display === "none") box.style.display = "block";
      at = [rect.left + c.left, rect.top + c.top];
      place();
    }
    // right-below the cursor; flip left / up at the viewport edge
    let at = null, size = [0, 0];
    function place() {
      if (!at) return;
      const [w, h] = size, [cx, cy] = at;
      const vw = window.innerWidth, vh = window.innerHeight, m = 4, gap = 14;
      let x = cx + gap;
      if (x + w > vw - m) x = cx - gap - w;
      x = Math.max(m, Math.min(x, vw - m - w));
      let y = cy + gap;
      if (y + h > vh - m) y = cy - gap - h;
      y = Math.max(m, Math.min(y, vh - m - h));
      box.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    }
    const ro = new ResizeObserver((es) => {
      const b = es[es.length - 1].borderBoxSize?.[0];
      size = b ? [b.inlineSize, b.blockSize] : [box.offsetWidth, box.offsetHeight];
      place();
    });
    ro.observe(box);
    let rect = null;
    const dropRect = () => { rect = null; };
    function attach(self) {
      plot = self;
      self.root.appendChild(box);
      self.over.addEventListener("mouseenter", () => { hovering = true; rect = null; });
      self.over.addEventListener("mouseleave", () => { hovering = false; hide(); });
      addEventListener("scroll", dropRect, { capture: true, passive: true });
      addEventListener("resize", dropRect, { passive: true });
      hide();
    }
    function detach() {
      ro.disconnect();
      removeEventListener("scroll", dropRect, { capture: true });
      removeEventListener("resize", dropRect);
    }
    return { update: (self) => { if (self === plot) update(self); }, attach, detach, dropRect };
  }

  // Same series structure, new points: keep the plot. If the user had
  // the full extent in view, let uPlot re-range so the new tail shows;
  // if they had zoomed, keep their window (resetScales = false).
  // (setData(data, false) alone stores the points but never redraws, so
  // the zoomed case re-sets the same x window: that redraws and re-ranges
  // y over the new points.)
  function update(valid) {
    const data = uPlot.join(valid.map((d) => [d.x, d.y]));
    holdX(valid);
    const sx = u.scales.x;
    if (isFull(sx)) u.setData(data);
    else {
      const { min, max } = sx;
      u.setData(data, false);
      u.setScale("x", { min, max });
    }
    xExtent = extentOf(data);
  }
  function isFull(sx) {
    const eps = 1e-9;
    return sx.min == null || sx.max == null
      || (sx.min <= xExtent[0] + eps && sx.max >= xExtent[1] - eps);
  }

  // the points this plot holds, as the caller's own arrays: a refresh
  // that re-derived identical arrays must not touch the plot at all
  // (2026-10-02: no redraw, no zoom reset, no flicker when idle)
  let held = null;
  function sameArr(a, b) {
    if (a === b) return true;
    if (!a || !b || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }
  function samePoints(valid) {
    if (!held || held.length !== valid.length) return false;
    return valid.every((d, i) => sameArr(d.x, held[i][0]) && sameArr(d.y, held[i][1]));
  }

  // window lock: re-applied when the window itself moves (new data, or
  // the user changed it), never on an idle refresh; turning it off
  // (null) goes back to the whole run
  let heldRange = null;
  function windowOf() {
    return xRange && xRange[0] != null && xRange[1] != null && xRange[1] > xRange[0] ? xRange : null;
  }
  function rangeHeld() {
    const r = windowOf();
    return r && heldRange ? r[0] === heldRange[0] && r[1] === heldRange[1] : r === heldRange;
  }
  function applyRange(force) {
    if (!u) return;
    const r = xRange && xRange[0] != null && xRange[1] != null && xRange[1] > xRange[0] ? xRange : null;
    const same = r && heldRange ? r[0] === heldRange[0] && r[1] === heldRange[1] : r === heldRange;
    if (same && !force) return;
    if (r) u.setScale("x", { min: r[0], max: r[1] });
    else if (heldRange) u.setScale("x", { min: xExtent[0], max: xExtent[1] });
    heldRange = r ? [r[0], r[1]] : null;
  }

  function fmtTick(v) {
    if (v == null) return "";
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(1) + "B";
    if (a >= 1e6) return +(v / 1e6).toFixed(1) + "M"; // 1.5M, not "2M" twice
    if (a >= 1e4) return (v / 1e3).toFixed(0) + "k";
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return +v.toFixed(2) + "";
    if (a === 0) return "0";
    if (a < 1e-3) return v.toExponential(0);
    return +v.toFixed(4) + "";
  }
  function fmtX(v) {
    if (v == null) return "—";
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(3) + "B";
    if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (Number.isInteger(v)) return v.toLocaleString();
    return +v.toFixed(3) + "";
  }
  function fmtVal(v) {
    const a = Math.abs(v);
    if (a >= 1e6) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (a >= 100) return v.toFixed(1);
    if (a < 1e-3 && a > 0) return v.toExponential(2);
    return v.toFixed(4);
  }

  // data / structure / window changes: nothing when the points are the
  // ones already plotted (only a moved window is applied), update in
  // place when only the points changed, rebuild when the series set, the
  // options or `resetKey` (e.g. the run shown) did. The work itself is
  // queued (sched.js): many charts changing at once are drawn a few per
  // slice, never in one long task that blocks input (2026-10-06).
  // Off-screen charts (2026-10-06): a refresh updates only the charts in
  // (or near) the viewport; one scrolled away keeps its plot and catches
  // up when it comes back into view. Every chart is still built up front
  // (so a drag-zoom syncs to all of them), and off-screen ones skip the
  // synced cursor moves (40 charts x 16 runs of hover points per move).
  const owner = {};
  let alive = true;
  let visible = true;
  let deferred = false;
  function queue() {
    schedule(owner, () => {
      if (!alive || !el) return;
      const valid = validOf(datasets);
      sync(valid, signature(valid));
    });
  }
  $effect(() => {
    void xRange;
    const valid = validOf(datasets);
    const s = signature(valid);
    if (!el) return;
    if (u && s === sig && samePoints(valid) && rangeHeld()) return;
    if (u && !visible) { deferred = true; return; }
    queue();
  });
  $effect(() => {
    if (!el) return;
    const io = new IntersectionObserver((es) => {
      visible = es[es.length - 1].isIntersecting;
      if (visible && deferred) { deferred = false; queue(); }
    }, { rootMargin: "200px 0px" });
    io.observe(el);
    return () => io.disconnect();
  });
  function sync(valid, s) {
    if (u && s === sig && samePoints(valid)) { u.batch(() => applyRange(false)); return; }
    if (u && s === sig) {
      u.batch(() => { update(valid); held = valid.map((d) => [d.x, d.y]); applyRange(true); });
      return;
    }
    build(valid); sig = s; heldRange = null;
    held = valid.map((d) => [d.x, d.y]);
    applyRange(true); // coalesces with the new plot's first draw
  }

  // lifetime: destroy on unmount only (not on every data change)
  $effect(() => {
    void el;
    return () => {
      alive = false;
      cancel(owner);
      if (u) { u.destroy(); u = null; sig = ""; held = null; heldRange = null; }
    };
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
    <div class="empty" style:min-height={height + 16 + "px"}>{title}<br /><small>no data for this metric</small></div>
  {/if}
</div>

<style>
  .chart { width: 100%; min-height: 60px; }
  .empty { color: var(--muted); font-size: 12px; padding: 12px 8px; }
  /* title floats on the plot's top-left (moved into .u-over in build());
     translucent plate so it reads over lines and grid; clicks and drags
     pass through to the plot */
  :global(.uplot .u-over > .u-title) {
    position: absolute; top: 2px; left: 4px; z-index: 3; pointer-events: none;
    max-width: calc(100% - 8px); padding: 0 6px; border-radius: 4px;
    font-size: 12px; font-weight: 600; line-height: 18px; text-align: left; color: var(--fg);
    background: color-mix(in srgb, var(--panel) 62%, transparent);
    text-shadow: 0 0 2px var(--panel), 0 0 3px var(--panel), 0 0 5px var(--panel);
  }
  /* static legend (no hover values: those are in the tooltip), compact,
     capped at three rows and scrolls past that (many runs in Compare) */
  :global(.uplot .u-legend) {
    font-size: 11px; line-height: 14px; color: var(--fg); margin-top: 1px;
    max-height: 48px; overflow-y: auto; overscroll-behavior: contain;
  }
  :global(.uplot .u-legend.u-inline tr) { margin-right: 8px; height: 16px; }
  :global(.uplot .u-legend .u-series > *) { padding: 1px 2px; }
  :global(.uplot .u-legend .u-marker) { width: 0.9em; height: 0.9em; margin-right: 3px; }
  /* (a non-live legend has no x-series entry: every row is a metric) */
  :global(.dash-tip) {
    position: fixed; left: 0; top: 0; z-index: 50; pointer-events: none;
    max-width: min(340px, calc(100vw - 8px));
    background: color-mix(in srgb, var(--bg) 94%, transparent); color: var(--fg);
    border: 1px solid var(--border); border-radius: 5px;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.3);
    padding: 3px 7px 4px; font-size: 11px; line-height: 15px;
  }
  :global(.dash-tip-head) { color: var(--muted); font-variant-numeric: tabular-nums; margin-bottom: 1px; white-space: nowrap; }
  :global(.dash-tip-row) { display: flex; align-items: center; gap: 5px; white-space: nowrap; }
  :global(.dash-tip-sw) { flex: none; width: 12px; height: 0; border-top: 3px solid; }
  :global(.dash-tip-name) { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  :global(.dash-tip-val) { flex: none; margin-left: 10px; font-variant-numeric: tabular-nums; font-weight: 600; }
</style>
