<script>
  // Header menu of the current tab's hidden graphs (2026-10-06): "Hidden
  // (n)" opens a list; an entry brings its graph back, "Show all" brings
  // back every one. Not shown while nothing on the tab is hidden. Closes
  // on outside click or Esc (focus back to the button); arrow keys,
  // Home / End move between the entries.
  import { hidden, showPanel, showAll, PANEL_TITLES } from "./hidden.svelte.js";

  let { tab } = $props();

  let open = $state(false);
  let root;
  let trigger;
  let menu;

  const items = $derived((hidden[tab] ?? []).map((id) => ({ id, title: PANEL_TITLES[tab]?.get(id) ?? id })));

  // nothing left to restore: the button goes away, so does the menu
  $effect(() => {
    if (!items.length && open) open = false;
  });
  // a tab switch closes it
  $effect(() => {
    void tab;
    open = false;
  });

  function entries() {
    return menu ? [...menu.querySelectorAll("[role=menuitem]")] : [];
  }
  function focusAt(i) {
    const es = entries();
    if (es.length) es[(i + es.length) % es.length].focus();
  }
  async function toggle(e) {
    open = !open;
    if (open) {
      await Promise.resolve();
      // keyboard opening lands on the first entry; a click keeps focus put
      if (e.detail === 0) focusAt(0);
    }
  }
  function close(refocus) {
    open = false;
    if (refocus) trigger?.focus();
  }
  function restore(id) {
    const es = entries();
    const at = es.findIndex((el) => el.dataset.id === id);
    showPanel(tab, id);
    // keep the keyboard in the list: focus the next entry (or the button)
    queueMicrotask(() => {
      if (!items.length) trigger?.focus();
      else focusAt(Math.min(at, items.length - 1));
    });
  }
  function onkey(e) {
    const es = entries();
    const i = es.indexOf(document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); focusAt(i + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); focusAt(i < 0 ? -1 : i - 1); }
    else if (e.key === "Home") { e.preventDefault(); focusAt(0); }
    else if (e.key === "End") { e.preventDefault(); focusAt(-1); }
    else if (e.key === "Tab") close(false);
  }
  $effect(() => {
    if (!open) return;
    const down = (e) => { if (root && !root.contains(e.target)) close(false); };
    const esc = (e) => { if (e.key === "Escape") close(true); };
    document.addEventListener("pointerdown", down, true);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", down, true);
      document.removeEventListener("keydown", esc);
    };
  });
</script>

{#if items.length}
  <div class="hidden-menu" bind:this={root}>
    <button bind:this={trigger} class="trigger" type="button" aria-haspopup="menu" aria-expanded={open}
      aria-controls="hidden-menu-list" aria-label={`Hidden graphs on this tab: ${items.length}`} onclick={toggle}
      onkeydown={async (e) => { if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); if (!open) open = true; await Promise.resolve(); focusAt(e.key === "ArrowDown" ? 0 : -1); } }}>
      Hidden <span class="badge">{items.length}</span>
    </button>
    {#if open}
      <div id="hidden-menu-list" class="pop" role="menu" aria-label="Hidden graphs" tabindex="-1" bind:this={menu} onkeydown={onkey}>
        {#each items as it (it.id)}
          <button type="button" role="menuitem" data-id={it.id} title="Show this graph again" onclick={() => restore(it.id)}>{it.title}</button>
        {/each}
        <div class="sep" role="separator"></div>
        <button type="button" role="menuitem" class="all" data-id="*" onclick={() => { showAll(tab); close(true); }}>Show all ({items.length})</button>
      </div>
    {/if}
  </div>
{/if}

<style>
  .hidden-menu { position: relative; }
  .trigger { display: inline-flex; align-items: center; gap: 6px; }
  .badge {
    min-width: 18px; padding: 0 5px; border-radius: 9px; box-sizing: border-box;
    background: var(--accent); color: #fff; font-size: 11px; line-height: 16px; text-align: center;
    font-variant-numeric: tabular-nums;
  }
  .pop {
    position: absolute; top: calc(100% + 4px); right: 0; z-index: 60;
    min-width: 220px; max-width: min(420px, calc(100vw - 24px)); max-height: 60vh; overflow-y: auto;
    padding: 4px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
  }
  .pop button {
    display: block; width: 100%; text-align: left; border: 0; background: transparent;
    padding: 5px 8px; border-radius: 4px; white-space: normal;
  }
  .pop button:hover, .pop button:focus-visible { background: var(--panel); outline: none; }
  .pop button:focus-visible { box-shadow: inset 0 0 0 2px var(--accent); }
  .sep { height: 1px; margin: 4px 2px; background: var(--border); }
  .all { color: var(--accent); }
</style>
