<script>
  // The timeline: a seek bar that is also the replay's effort profile — the
  // applied torque (Σ|τ|, N·m) of every tick, as a filled area across the
  // whole track (lib/torquegraph.js). Click or drag anywhere to seek (that
  // pauses, as the timeline always has); arrows step one tick, Home / End
  // jump to the ends; hovering names the time and that tick's torque.
  //
  // Drawing: the area is painted ONCE per replay and per track size into two
  // canvases — the unplayed copy dim, the played copy in the accent colour —
  // and playback only moves a CSS variable (--p): it clips the played copy
  // and places the playhead, the track fill and the thumb. No canvas work
  // per frame.
  //
  // A replay without torque data (a person's Flux run: no body) gets the
  // plain track.
  import { lastTick, pb, seekTo, setPlaying } from "./state.svelte.js";
  import { DT } from "./replay.js";
  import { downsample, fracToTick, mistakeTicks, robustMax, tickToFrac,
    torqueSeries } from "./torquegraph.js";

  let { compact = false } = $props();

  // the timeline's position
  const tickIdx = $derived(
    pb.rep ? Math.min(Math.floor(pb.cur), lastTick()) : 0);
  const last = $derived(lastTick());
  const frac = $derived(tickToFrac(tickIdx, last));
  const mmss = (s) => {
    const t = Math.max(0, Math.floor(s));
    return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
  };
  const seekText = $derived(`${mmss(tickIdx * DT)} of ${mmss(last * DT)}`);

  // per replay: the series, its full scale and the mistake marks
  const series = $derived(torqueSeries(pb.rep));
  const top = $derived(series ? robustMax(series) : 1);
  const marks = $derived(series ? mistakeTicks(pb.rep) : []);

  let root = $state(null), graph = $state(null);
  let lit = $state(null), dim = $state(null);
  let gw = $state(0), gh = $state(0);
  let draws = 0; // canvas repaints, on data-draws (the browser test pins it)

  $effect(() => {
    if (!graph) return;
    const ro = new ResizeObserver(([e]) => {
      gw = e.contentRect.width;
      gh = e.contentRect.height;
    });
    ro.observe(graph);
    return () => ro.disconnect();
  });

  // colours: the bar's accent for what has played, zinc for the rest
  // (the max envelope faint, the mean solid: the level reads first, and a
  // spike still shows as a light whisker above it)
  const LIT = { env: "rgba(122,162,247,0.30)", mean: "rgba(122,162,247,0.95)" };
  const DIM = { env: "rgba(161,161,170,0.14)", mean: "rgba(161,161,170,0.55)" };
  const CLIP = "rgba(240,163,94,0.95)"; // a column above full scale: amber cap
  const MARK = "rgba(255,99,99,0.9)";   // a miss / bad cut / bomb

  function paint(cv, col, cols, rows, dpr, ds) {
    cv.width = cols;
    cv.height = rows;
    const g = cv.getContext("2d");
    g.clearRect(0, 0, cols, rows);
    const h = (v) => Math.round(Math.min(1, v / top) * rows);
    g.fillStyle = col.env;
    for (let i = 0; i < cols; i++) { const y = h(ds.max[i]); g.fillRect(i, rows - y, 1, y); }
    g.fillStyle = col.mean;
    for (let i = 0; i < cols; i++) { const y = h(ds.mean[i]); g.fillRect(i, rows - y, 1, y); }
    g.fillStyle = CLIP;
    const cap = Math.max(1, Math.round(1.5 * dpr));
    for (let i = 0; i < cols; i++) if (ds.max[i] > top) g.fillRect(i, 0, 1, cap);
    g.fillStyle = MARK;
    const mw = Math.max(1, Math.round(dpr)), mh = Math.round(4 * dpr);
    for (const k of marks) {
      const x = Math.round(tickToFrac(k, last) * (cols - 1));
      g.fillRect(Math.max(0, x - (mw >> 1)), rows - mh, mw, mh);
    }
  }

  $effect(() => {
    const s = series, W = gw, H = gh;
    if (!s || !W || !H || !lit || !dim) return;
    const dpr = window.devicePixelRatio || 1;
    const cols = Math.max(1, Math.round(W * dpr)), rows = Math.max(1, Math.round(H * dpr));
    const ds = downsample(s, cols);
    paint(dim, DIM, cols, rows, dpr, ds);
    paint(lit, LIT, cols, rows, dpr, ds);
    graph.dataset.cols = String(cols);
    root.dataset.draws = String(++draws);
  });

  // ---- seeking ----------------------------------------------------------------
  function seek(k) {
    setPlaying(false);
    seekTo(Math.min(Math.max(k, 0), lastTick()));
  }
  const tickAt = (clientX) => {
    const r = root.getBoundingClientRect();
    return fracToTick((clientX - r.left) / r.width, lastTick());
  };

  let dragging = false;
  let hover = $state(null); // { x (px in the bar), k (tick) }
  function hoverAt(clientX) {
    const r = root.getBoundingClientRect();
    hover = { x: Math.min(Math.max(clientX - r.left, 0), r.width), w: r.width, k: tickAt(clientX) };
  }
  function onpointerdown(e) {
    if (e.button !== 0) return;
    e.preventDefault(); // no text selection / native drag; focus explicitly
    root.focus({ preventScroll: true });
    root.setPointerCapture?.(e.pointerId);
    dragging = true;
    hoverAt(e.clientX);
    seek(tickAt(e.clientX));
  }
  function onpointermove(e) {
    if (dragging || e.pointerType === "mouse") hoverAt(e.clientX);
    if (dragging) {
      const k = tickAt(e.clientX);
      if (k !== tickIdx) seek(k);
    }
  }
  function onpointerup(e) {
    dragging = false;
    if (root.hasPointerCapture?.(e.pointerId)) root.releasePointerCapture(e.pointerId);
    if (e.pointerType !== "mouse") hover = null;
  }
  function onkeydown(e) {
    let k;
    switch (e.key) {
      case "ArrowRight": case "ArrowUp": k = tickIdx + 1; break;
      case "ArrowLeft": case "ArrowDown": k = tickIdx - 1; break;
      case "Home": k = 0; break;
      case "End": k = lastTick(); break;
      default: return;
    }
    e.preventDefault();
    seek(k);
  }

  // the hover readout
  const tip = $derived.by(() => {
    if (!hover || !pb.rep) return null;
    const k = Math.min(hover.k, lastTick());
    const t = pb.rep.ticks[k]?.torque;
    return {
      left: Math.min(Math.max(hover.x, 84), Math.max(84, hover.w - 84)),
      time: mmss(k * DT),
      total: series ? `${series[k].toFixed(0)} N·m` : "",
      split: series && t ? `spine ${t[1].toFixed(0)} · arms ${t[2].toFixed(0)} · legs ${t[3].toFixed(0)}` : "",
    };
  });
</script>

<div bind:this={root} id="seekbar" role="slider" tabindex="0"
  aria-label="seek" aria-orientation="horizontal"
  aria-valuemin={0} aria-valuemax={last} aria-valuenow={tickIdx} aria-valuetext={seekText}
  data-graph={series ? "torque" : "none"} data-top={top.toFixed(1)}
  class="group/seek relative w-full cursor-pointer touch-none rounded-sm outline-none select-none
    focus-visible:ring-2 focus-visible:ring-ring/60
    {series ? (compact ? 'h-[30px]' : 'h-[38px]') : 'h-4'}"
  style:--p="{frac * 100}%" style:--f={frac}
  {onpointerdown} {onpointermove} {onpointerup} onpointercancel={onpointerup}
  onpointerleave={() => { if (!dragging) hover = null; }}
  {onkeydown}>
  {#if series}
    <div bind:this={graph} class="pointer-events-none absolute inset-x-0 top-0
      {compact ? 'h-[20px]' : 'h-[28px]'}">
      <canvas bind:this={dim} class="absolute inset-0 size-full"></canvas>
      <canvas bind:this={lit} class="absolute inset-0 size-full
        [clip-path:inset(0_calc(100%-var(--p))_0_0)]"></canvas>
      <div class="absolute inset-y-0 left-(--p) w-px -translate-x-1/2 bg-white/85"></div>
      {#if hover}
        <div class="absolute inset-y-0 w-px -translate-x-1/2 bg-white/40" style:left="{hover.x}px"></div>
      {/if}
    </div>
  {/if}
  <!-- the track: played part in the accent colour (a scaleX, not a width:
       no layout per frame), and the thumb at the playhead -->
  <div class="pointer-events-none absolute inset-x-0 bottom-[5px] h-1 overflow-hidden rounded-full bg-muted">
    <div class="h-full w-full origin-left scale-x-(--f) bg-on"></div>
  </div>
  <div class="pointer-events-none absolute bottom-[1px] left-(--p) size-3 -translate-x-1/2 rounded-full
    border border-on bg-white shadow-sm transition-[box-shadow] group-hover/seek:ring-4
    group-hover/seek:ring-ring/50 group-focus-visible/seek:ring-4 group-focus-visible/seek:ring-ring/60"></div>
  {#if tip}
    <div id="seektip" role="presentation" class="pointer-events-none absolute bottom-[calc(100%+6px)] z-10
      -translate-x-1/2 rounded-md bg-foreground px-2 py-1 text-center text-xs whitespace-nowrap
      text-background tabular-nums shadow-md" style:left="{tip.left}px">
      <span class="font-medium">{tip.time}</span>{#if tip.total}&nbsp;· {tip.total}{/if}
      {#if tip.split}<span class="block text-[11px] opacity-70">{tip.split}</span>{/if}
    </div>
  {/if}
</div>
