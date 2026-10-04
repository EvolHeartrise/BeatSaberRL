<script>
  import { store, loadIndex } from "./lib/data.svelte.js";
  import Live from "./lib/Live.svelte";
  import Compare from "./lib/Compare.svelte";
  import Runs from "./lib/Runs.svelte";
  import Swing from "./lib/Swing.svelte";

  const TABS = ["live", "swing", "compare", "runs"];
  let tab = $state(TABS.includes(location.hash.replace("#", "")) ? location.hash.slice(1) : "live");
  let xMode = $state("samples");
  let liveRun = $state(null);
  let swingRun = $state(null);
  let selected = $state([]);

  $effect(() => { location.hash = tab; });
  $effect(() => { loadIndex(); });

  function pick(id) {
    if (!selected.includes(id)) selected = [...selected, id];
    tab = "compare";
  }
</script>

<header>
  <h1>Beat Arena — training</h1>
  <nav>
    {#each TABS as t}
      <button class:on={tab === t} onclick={() => (tab = t)}>{t}</button>
    {/each}
  </nav>
  <span class="src">{store.base ? `archive @ ${store.base}` : "locating archive…"}</span>
</header>

{#if store.error}
  <div class="err">{store.error}</div>
{/if}

<section>
  {#if !store.index}
    <p class="muted">loading index.json…</p>
  {:else if tab === "live"}
    <Live bind:xMode bind:runId={liveRun} />
  {:else if tab === "swing"}
    <Swing bind:runId={swingRun} />
  {:else if tab === "compare"}
    <Compare bind:selected bind:xMode />
  {:else}
    <Runs onpick={pick} {selected} />
  {/if}
</section>

<style>
  :global(:root) {
    --bg: #fafafa; --fg: #1e1e1e; --muted: #6b6b6b; --panel: #f0f0f2;
    --border: #d8d8dc; --accent: #2f6fdb; --grid: rgba(0, 0, 0, 0.12);
    color-scheme: light dark;
  }
  @media (prefers-color-scheme: dark) {
    :global(:root) {
      --bg: #17181b; --fg: #e6e6e6; --muted: #9a9aa2; --panel: #202227;
      --border: #34363d; --accent: #6ea0ff; --grid: rgba(255, 255, 255, 0.12);
    }
  }
  :global(body) {
    margin: 0; padding: 10px 14px; background: var(--bg); color: var(--fg);
    font: 13px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  :global(button) {
    background: var(--panel); color: var(--fg); border: 1px solid var(--border);
    border-radius: 4px; padding: 3px 9px; cursor: pointer; font: inherit;
  }
  :global(button:disabled) { opacity: 0.5; cursor: default; }
  :global(select), :global(input) {
    background: var(--panel); color: var(--fg); border: 1px solid var(--border);
    border-radius: 4px; padding: 3px 6px; font: inherit;
  }
  :global(.muted) { color: var(--muted); }
  :global(a) { color: var(--accent); }
  header { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }
  h1 { font-size: 16px; margin: 0 6px 0 0; }
  nav button { text-transform: capitalize; }
  nav button.on { background: var(--accent); color: #fff; border-color: var(--accent); }
  .src { color: var(--muted); font-size: 12px; margin-left: auto; }
  .err { background: #b3261e; color: #fff; padding: 6px 10px; border-radius: 4px; margin: 8px 0; }
</style>
