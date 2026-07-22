<script>
  import { Canvas } from "@threlte/core";
  import Scene from "./lib/Scene.svelte";
  import { pb, lastTick, loadText, setPlaying, setSpeed, step }
    from "./lib/state.svelte.js";
  import { DT, configSummary, counts } from "./lib/replay.js";

  let wrap; // canvas container, for video export
  let dropHover = $state(false);
  let fileInput;

  const tickIdx = $derived(
    pb.rep ? Math.min(Math.floor(pb.cur), lastTick()) : 0);
  const time = $derived(pb.cur * DT);
  const c = $derived(pb.rep ? counts(pb.rep, tickIdx) : null);

  function load(text, name) {
    try {
      loadText(text, name);
    } catch (err) {
      alert("could not load replay: " + err.message);
    }
  }

  function onFiles(files) {
    const f = files && files[0];
    if (f) f.text().then((t) => load(t, f.name));
  }

  function onKey(e) {
    if (e.key === " ") { e.preventDefault(); setPlaying(!pb.playing); }
    else if (e.key === "ArrowRight") step(1);
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "+" || e.key === "=") setSpeed(pb.speed * 2);
    else if (e.key === "-") setSpeed(pb.speed / 2);
    else if (e.key === "r") { pb.cur = 0; setPlaying(true); }
    else if (e.key === "g") pb.gauges = !pb.gauges;
  }

  const speedLabel = $derived(
    pb.speed.toFixed(2).replace(/0+$/, "").replace(/\.$/, ".0") + "×");

  // ---- video export (canvas.captureStream + MediaRecorder) -----------------
  // Records one real-time playback pass at the current speed, downloads a
  // .webm. Score popups/labels are in-scene sprites, so they're captured.
  function exportVideo() {
    if (!pb.rep || pb.exporting) return;
    const canvas = wrap && wrap.querySelector("canvas");
    if (!canvas || !("MediaRecorder" in window)) return;
    pb.exporting = true;
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9" : "video/webm";
    const rec = new MediaRecorder(canvas.captureStream(30),
      { mimeType: mime, videoBitsPerSecond: 3_000_000 });
    const chunks = [];
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = () => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
      a.download = (pb.name || "replay") + ".webm";
      a.click();
      URL.revokeObjectURL(a.href);
      pb.exporting = false;
    };
    pb.cur = 0;
    setPlaying(true);
    rec.start();
    const poll = setInterval(() => { // playback auto-pauses at the end
      if (!pb.playing) { clearInterval(poll); rec.stop(); }
    }, 200);
  }
</script>

<svelte:window
  onkeydown={onKey}
  ondragover={(e) => { e.preventDefault(); dropHover = true; }}
  ondragenter={(e) => { e.preventDefault(); dropHover = true; }}
  ondragleave={(e) => { e.preventDefault(); dropHover = false; }}
  ondrop={(e) => {
    e.preventDefault();
    dropHover = false;
    onFiles(e.dataTransfer.files);
  }}
/>

<main>
  <h1>Beat Arena — 3D replay viewer</h1>
  <button id="drop" class:hover={dropHover} onclick={() => fileInput.click()}>
    {#if pb.rep}
      {pb.name} — {configSummary(pb.rep)}
    {:else}
      drop a replay <code>.jsonl</code> here, or click to choose
    {/if}
  </button>
  <input
    type="file"
    accept=".jsonl,.json"
    hidden
    bind:this={fileInput}
    onchange={(e) => onFiles(e.target.files)}
  />

  <div id="stage" bind:this={wrap}>
    <Canvas>
      <Scene />
    </Canvas>
  </div>

  <div id="bar">
    <button onclick={() => setPlaying(!pb.playing)}>
      {pb.playing ? "⏸ pause" : "▶ play"}
    </button>
    <button onclick={() => { pb.cur = 0; setPlaying(true); }}>↺ restart</button>
    <button onclick={() => setSpeed(pb.speed / 2)}>−</button>
    <span>{speedLabel}</span>
    <button onclick={() => setSpeed(pb.speed * 2)}>+</button>
    <button onclick={exportVideo} disabled={pb.exporting || !pb.rep}>
      {pb.exporting ? "⏺ recording…" : "⏺ export video"}
    </button>
    <input
      type="range"
      min="0"
      max={lastTick()}
      step="1"
      value={tickIdx}
      oninput={(e) => { setPlaying(false); pb.cur = +e.target.value; }}
    />
  </div>

  <div id="hud">
    {#if pb.rep && c}
      t <b>{time.toFixed(2)}s</b> · tick <b>{tickIdx}</b>/{lastTick()} ·
      return <b>{c.reward.toFixed(2)}</b> ·
      score <b>{c.score.toFixed(0)}</b>/{115 * pb.rep.chart.length} ·
      <span class="good">good {c.good}</span> /
      <span class="bad">bad {c.bad}</span> /
      <span class="miss">miss {c.miss}</span> of {pb.rep.chart.length}
      {#if tickIdx >= lastTick()}· <b>END</b>{/if}
    {:else}
      no replay loaded
    {/if}
  </div>
  <div id="help">
    drag = orbit · wheel = zoom · right-drag = pan · space = play/pause ·
    ←/→ = step one tick · +/− = speed · r = restart · g = wrist gauges
    (roll dial shows the allowed 280° arc; bend 0° = at the notes,
    ±90° = flat, ±95° = limit; orange = near a joint limit) ·
    40 cm blocks fly in; each hand is a wrist on the plane holding a 1 m
    blade · white blade core = contact-capable (tip ≥ 1 m/s) · white block
    outline = within blade reach · +N popups = points (max 115/note:
    70 pre-swing, 30 follow-through, 15 accuracy)
  </div>
</main>

<style>
  :global(:root) { color-scheme: dark; }
  :global(body) {
    margin: 0; background: #14161c; color: #cfd3dc;
    font: 14px/1.4 system-ui, sans-serif;
  }
  main {
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh;
  }
  h1 { font-size: 16px; font-weight: 600; margin: 14px 0 6px; color: #e8eaf0; }
  #drop {
    border: 1.5px dashed #4a4f5c; border-radius: 10px; padding: 10px 18px;
    margin: 6px 0; color: #9aa0ae; cursor: pointer; background: none;
    font: inherit;
  }
  #drop.hover { border-color: #7aa2f7; color: #cfd3dc; }
  #stage {
    width: 960px; height: 700px; max-width: 96vw;
    background: #0b0d11; border-radius: 10px; margin-top: 10px;
    overflow: hidden;
  }
  #bar {
    display: flex; gap: 10px; align-items: center; margin: 10px 0 4px;
    flex-wrap: wrap; justify-content: center;
  }
  button {
    background: #232733; color: #cfd3dc; border: 1px solid #3a4050;
    border-radius: 6px; padding: 4px 12px; cursor: pointer; font: inherit;
  }
  button:hover { background: #2d3342; }
  button:disabled { opacity: 0.5; cursor: default; }
  input[type="range"] { width: 320px; }
  #hud { font-variant-numeric: tabular-nums; color: #9aa0ae; margin-bottom: 16px; }
  #hud b { color: #e8eaf0; }
  .good { color: #7dd3a0; }
  .bad { color: #f0a35e; }
  .miss { color: #e06c75; }
  #help {
    font-size: 12px; color: #6b7180; margin-bottom: 14px; max-width: 900px;
    text-align: center;
  }
</style>
