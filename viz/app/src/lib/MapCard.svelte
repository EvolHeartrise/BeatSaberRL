<script>
  // Map metadata card, top-left over the scene (2026-09-14): cover art,
  // song + sub name, song author, mapper, the difficulty badge and a link
  // to the map on BeatSaver. Pure DOM overlay, like PosePanel — the scene
  // is a Threlte canvas, so a card drawn in it would cost an in-scene HUD
  // and an unclickable link for nothing.
  //
  // Only the hash, difficulty and song name are in the replay header; the
  // rest comes from BeatSaver by hash (lib/mapmeta.js). Offline, the card
  // falls back to exactly what the header knows.
  import { pb } from "./state.svelte.js";
  import { cachedMeta, mapMeta } from "./mapmeta.js";

  const m = $derived(pb.rep?.map ?? null);

  let meta = $state(null);   // BeatSaver's answer, once it lands
  let err = $state("");      // why it did not
  let coverOk = $state(true);

  $effect(() => {
    const hash = m?.hash ?? "";
    meta = null;
    err = "";
    coverOk = true;
    if (!hash) return;
    const hit = cachedMeta(hash); // same map twice: no request at all
    if (hit) {
      meta = hit;
      return;
    }
    // Debounced, and stale-guarded: clicking through replays must not fire
    // a request per file, and an answer that arrives after the next replay
    // was loaded is dropped rather than shown against the wrong map.
    let stale = false;
    const t = setTimeout(() => {
      mapMeta(hash)
        .then((x) => { if (!stale) meta = x; })
        .catch((e) => { if (!stale) err = e?.message || "lookup failed"; });
    }, 150);
    return () => { stale = true; clearTimeout(t); };
  });

  // "ExpertPlus" is how the header (and Beat Saber) spells it; everyone
  // else writes Expert+.
  const diff = $derived(
    (m?.difficulty ?? "").replace(/Plus$/, "+"));
  const song = $derived(meta?.song || m?.song || "unknown song");
  const byline = $derived([
    meta?.author ? "by " + meta.author : "",
    meta?.mapper ? "mapped by " + meta.mapper : "",
  ].filter(Boolean).join(" · "));
</script>

{#if m}
  <div id="mapcard">
    <div class="mc">
      <div class="art" class:blank={!meta?.cover || !coverOk}>
        {#if meta?.cover && coverOk}
          <!-- crossOrigin: BeatSaver's CDN sends access-control-allow-origin
               *, and an anonymous request keeps the image usable if it is
               ever drawn into a canvas -->
          <img src={meta.cover} alt="" crossorigin="anonymous"
            onerror={() => (coverOk = false)} />
        {:else}
          <span>♪</span>
        {/if}
      </div>
      <div class="txt">
        <div class="song" title={song}>{song}</div>
        {#if meta?.sub}
          <div class="sub" title={meta.sub}>{meta.sub}</div>
        {/if}
        {#if byline}
          <div class="by" title={byline}>{byline}</div>
        {/if}
        <div class="foot">
          {#if diff}<span class="badge">{diff}</span>{/if}
          {#if meta?.url}
            <a href={meta.url} target="_blank" rel="noreferrer noopener"
              title="open this map on BeatSaver">↗ {meta.id}</a>
          {:else}
            <span class="hash">{(m.hash ?? "").slice(0, 8)}</span>
          {/if}
        </div>
        {#if err}
          <div class="err" title={err}>metadata unavailable</div>
        {:else if !meta}
          <div class="err">looking up BeatSaver…</div>
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  /* container passes pointer events through; the card itself is live */
  #mapcard {
    position: absolute; top: 12px; left: 12px; z-index: 3;
    pointer-events: none;
  }
  .mc {
    pointer-events: auto;
    display: flex; gap: 10px; align-items: flex-start;
    background: rgba(17, 19, 25, 0.72);
    border: 1px solid rgba(58, 64, 80, 0.6);
    border-radius: 10px;
    backdrop-filter: blur(6px);
    padding: 10px; width: 300px; box-sizing: border-box;
    color: #9aa0ae; font-size: 12px;
  }
  .art {
    width: 96px; height: 96px; flex: none; border-radius: 6px;
    overflow: hidden; background: #171a21;
    border: 1px solid rgba(58, 64, 80, 0.6);
  }
  .art img { width: 100%; height: 100%; display: block; object-fit: cover; }
  .art.blank {
    display: flex; align-items: center; justify-content: center;
    color: #3a4050; font-size: 32px;
  }
  .txt { min-width: 0; display: flex; flex-direction: column; gap: 2px; }
  .song, .sub {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  /* the byline is the one line worth wrapping — "by X · mapped by Y" is
     routinely longer than the card is wide */
  .by { overflow-wrap: anywhere; }
  .song { color: #e8eaf0; font-weight: 600; font-size: 13px; }
  .sub { color: #6b7180; }
  .by { color: #9aa0ae; }
  .foot { display: flex; gap: 6px; align-items: center; margin-top: 4px; }
  .badge {
    border: 1px solid #3a4050; border-radius: 5px; padding: 1px 6px;
    background: #232733; color: #cfd3dc; font-size: 11px;
  }
  .hash {
    color: #6b7180; font-size: 11px;
    font-variant-numeric: tabular-nums;
  }
  a { color: #7aa2f7; text-decoration: none; font-size: 11px; }
  a:hover { text-decoration: underline; }
  .err { color: #6b7180; font-size: 11px; margin-top: 2px; }
</style>
