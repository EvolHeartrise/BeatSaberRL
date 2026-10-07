<script>
  // One graph card: the chart plus its hide button (2026-10-06). The
  // button sits in the card's top-right corner, clear of the floating
  // title (top-left, kept short of the corner) and of the tooltip
  // (position: fixed, pointer-events: none); it shows on card hover or
  // keyboard focus, always on touch screens.
  import Chart from "./Chart.svelte";
  import { hidePanel } from "./hidden.svelte.js";

  let { tab, id, title, ...chart } = $props();
</script>

<div class="cell" data-panel={id}>
  <button class="hide" type="button" aria-label={`Hide graph: ${title}`} title="Hide this graph (restore it from “Hidden” at the top)"
    onclick={() => hidePanel(tab, id)}>
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M10.7 5.1A10.4 10.4 0 0 1 12 5c7 0 10 7 10 7a13.2 13.2 0 0 1-1.7 2.7" />
      <path d="M6.6 6.6A13.5 13.5 0 0 0 2 12s3 7 10 7a9.7 9.7 0 0 0 5.4-1.6" />
      <path d="M9.9 9.9a3 3 0 0 0 4.2 4.2" />
      <path d="m2 2 20 20" />
    </svg>
  </button>
  <Chart {title} {...chart} />
</div>

<style>
  .cell {
    position: relative; min-width: 0;
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 3px 4px 2px;
  }
  .hide {
    position: absolute; top: 4px; right: 5px; z-index: 12;
    display: grid; place-items: center; width: 22px; height: 22px; padding: 0;
    border-radius: 4px; border: 1px solid var(--border); color: var(--muted);
    background: color-mix(in srgb, var(--panel) 85%, transparent);
    opacity: 0; transition: opacity 0.12s;
  }
  .cell:hover .hide, .hide:focus-visible { opacity: 1; }
  .hide:hover { color: var(--fg); border-color: var(--accent); }
  .hide:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
  @media (hover: none) { .hide { opacity: 1; } }
  /* the floating title stops short of the button */
  .cell :global(.uplot .u-over > .u-title) { max-width: calc(100% - 34px); }
</style>
