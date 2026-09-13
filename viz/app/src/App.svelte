<script>
  import { Canvas } from "@threlte/core";
  import { NoToneMapping } from "three";
  import Scene from "./lib/Scene.svelte";
  import PosePanel from "./lib/PosePanel.svelte";
  import { audio, gfx, pb, pose, lastTick, loadText, setPlaying, setSpeed,
    step, togglePose } from "./lib/state.svelte.js";
  import { songBlob } from "./lib/audio.js";
  import { CAN_EXPORT_EXACT, decodeSong, exportExact } from "./lib/record.js";
  import { flushSync } from "svelte";
  import { DT, configSummary, counts, maxScore } from "./lib/replay.js";

  let wrap; // canvas container, for video export
  let dropHover = $state(false);
  let fileInput;

  const tickIdx = $derived(
    pb.rep ? Math.min(Math.floor(pb.cur), lastTick()) : 0);
  const time = $derived(pb.cur * DT);
  const c = $derived(pb.rep ? counts(pb.rep, tickIdx) : null);
  // bombs are not scoreable — they resolve silently unless hit
  const scoreable = $derived(
    pb.rep ? pb.rep.chart.filter((n) => n.kind !== 1).length : 0);
  const nBombs = $derived(pb.rep ? pb.rep.chart.length - scoreable : 0);
  // cached: the HUD re-renders every tick and maxScore is O(chart)
  const maxPts = $derived(pb.rep ? maxScore(pb.rep.chart) : 0);

  // ---- song audio ---------------------------------------------------------
  // Fetch the map's audio whenever a replay that names one is loaded. The
  // blob URL is revoked on replacement; the flag guards a replay swapped out
  // mid-download from clobbering the next one's state.
  // $state, not a plain let: bind:this on a plain variable is not a tracked
  // write, so the sync effect below would never learn the element arrived.
  let audioEl = $state(null);
  let blobUrl = null;
  let songData = null;   // the fetched Blob, for the frame-by-frame export
  let songBuf = null;    // decoded once, reused across exports
  $effect(() => {
    const m = pb.rep?.map;
    if (!m) return;
    let stale = false;
    songBlob(m.hash, (stage) => { if (!stale) audio.msg = stage; })
      .then((blob) => {
        if (stale) return;
        if (!audioEl) return;
        songData = blob;
        songBuf = null;
        if (blobUrl) URL.revokeObjectURL(blobUrl);
        blobUrl = URL.createObjectURL(blob);
        audioEl.src = blobUrl;
        audio.status = "ready";
        audio.msg = "";
      })
      .catch((e) => {
        if (stale) return;
        audio.status = "error";
        audio.msg = e.message;
      });
    return () => { stale = true; };
  });

  // The audio follows the replay clock; the clock NEVER follows the audio.
  // The tick counter is authoritative — scrubbing, tick-stepping and the
  // video export all drive it directly — so the element is nudged back
  // whenever it drifts past SYNC_TOL rather than being trusted to keep time.
  // This effect reads pb.cur, so it reconciles once per frame while playing.
  const SYNC_TOL = 0.08;
  $effect(() => {
    // Every reactive read happens up front, before any early return: `||`
    // short-circuits, so testing the element first would leave this effect
    // subscribed to nothing on the run where the element is still null, and
    // it would never run again.
    const el = audioEl;
    const ready = audio.status === "ready";
    const want = audio.offset + pb.cur * DT; // song time for this frame
    const rate = pb.speed, vol = audio.muted ? 0 : audio.vol;
    const run = pb.playing && !pose.on;
    if (!el || !ready) return;
    el.playbackRate = rate; // setSpeed already clamps to [0.25, 4]
    if (gain) gain.gain.value = vol; else el.volume = vol;
    if (want < 0) {
      // the recording starts before the song does (the chart is re-based so
      // the first note lands at 2 s, which can predate t=0 in song time)
      if (!el.paused) el.pause();
      el.currentTime = 0;
      return;
    }
    if (Math.abs(el.currentTime - want) > SYNC_TOL) el.currentTime = want;
    if (run) {
      // may reject until the first user gesture; the next frame retries
      if (el.paused) el.play().catch(() => {});
    } else if (!el.paused) el.pause();
  });

  // The recording's audio: tap the <audio> element through WebAudio and fork
  // it — through a gain node to the speakers (what you hear) and straight to
  // a MediaStreamDestination (what gets recorded). Two things to know:
  //   - createMediaElementSource REROUTES the element permanently, so the
  //     graph is built once and the speaker branch must stay connected or
  //     the viewer goes silent;
  //   - the element's own volume is upstream of the tap, so leaving the
  //     mute/volume control on it would bake a muted viewer into a silent
  //     export. The level moves into `gain`, which only feeds the speakers.
  // Built lazily from the export click, which is the user gesture an
  // AudioContext needs to start unsuspended.
  let actx = null, tap = null, gain = null;
  function audioTap() {
    if (!audioEl || audio.status !== "ready") return null;
    if (!actx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      actx = new AC();
      const src = actx.createMediaElementSource(audioEl);
      gain = actx.createGain();
      tap = actx.createMediaStreamDestination();
      src.connect(gain).connect(actx.destination);
      src.connect(tap);
      audioEl.volume = 1; // the level lives in `gain` from here on
    }
    actx.resume?.();
    return tap;
  }

  const audioLabel = $derived(
    audio.status === "ready" ? (audio.song || "song")
    : audio.status === "loading" ? (audio.msg || "song…")
    : audio.status === "error" ? "no audio"
    : audio.status === "none" ? "no song"
    : "");

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
    // keys aimed at the pose panel's own controls stay there (slider
    // arrow-key nudges must not step the replay)
    if (e.target && e.target.closest && e.target.closest("#pose, #fp")) return;
    if (e.key === "p") { togglePose(); return; }
    if (e.key === " ") { e.preventDefault(); setPlaying(!pb.playing); }
    else if (e.key === "ArrowRight") step(1);
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "+" || e.key === "=") setSpeed(pb.speed * 2);
    else if (e.key === "-") setSpeed(pb.speed / 2);
    else if (e.key === "r") { pb.cur = 0; setPlaying(true); }
    else if (e.key === "g") pb.gauges = !pb.gauges;
    else if (e.key === "h") pb.hitboxes = !pb.hitboxes;
    else if (e.key === "f") pb.firstPerson = !pb.firstPerson;
  }

  // exported length so far, as m:ss — the export is scrubbed from t = 0,
  // so this is how much of the replay the file will contain if stopped now
  const mmss = (s) =>
    `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const exportLabel = $derived(
    `⏺ ${Math.round(pb.exportPct * 100)}% · ${mmss(pb.exportPct * lastTick() * DT)}`);

  const speedLabel = $derived(
    pb.speed.toFixed(2).replace(/0+$/, "").replace(/\.$/, ".0") + "×");

  // ---- video export -------------------------------------------------------
  // Two paths. The frame-by-frame one is the default: it renders exactly one
  // frame per replay tick and stamps each with its own timestamp, so the file
  // is 30 fps no matter how long a frame took to draw, and the song is copied
  // out of the decoded audio by array slice rather than recorded off the
  // speakers. On a machine that cannot render 30 fps live, the old real-time
  // pass could only produce a stretched video with choppy audio — that is
  // what MediaRecorder does with late frames and a stalling audio graph, and
  // no amount of tuning changes it. The real-time pass stays as the fallback
  // for browsers without WebCodecs.
  // Score popups/labels are in-scene sprites, so both paths capture them.
  function download(blob) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (pb.name || "replay") + ".webm";
    a.click();
    // give the click a turn to be handled before the URL goes away
    setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
  }

  async function exportVideo() {
    if (!pb.rep || pb.exporting) return;
    const canvas = wrap && wrap.querySelector("canvas");
    if (!canvas) return;
    if (!CAN_EXPORT_EXACT) return exportRealtime(canvas);

    pb.exporting = true;
    pb.exportPct = 0;
    pb.exportStop = false;
    setPlaying(false);
    const wasCur = pb.cur;
    const mode = gfx.renderMode?.current;
    try {
      if (songData && !songBuf) songBuf = await decodeSong(songData);
      // Output frames are decoupled from the replay's 30 ticks/s: frame i is
      // at i/fps seconds, i.e. tick i*TPS/fps, which the viewer interpolates
      // between logged ticks. So 60 fps genuinely smooths the motion instead
      // of duplicating frames, and 24 subsamples it.
      const fps = pb.fps;
      const TPS = 1 / DT; // replay ticks per second
      const frames = Math.floor((lastTick() / TPS) * fps) + 1;
      const t0 = performance.now(); // synthetic frame clock, monotonic
      // stop Threlte's own loop: from here the exporter owns the frame clock
      gfx.renderMode?.set("manual");
      const blob = await exportExact({
        canvas,
        frames,
        fps,
        song: songBuf,
        songStart: audio.offset,
        onProgress: (p) => { pb.exportPct = p; },
        // 'stop & save': the exporter ends at the next frame and still
        // finalizes a complete file of everything rendered so far
        shouldStop: () => pb.exportStop,
        renderFrame: (i) => {
          pb.cur = Math.min(lastTick(), (i * TPS) / fps);
          flushSync(); // derived pose + the $effects that refill GPU buffers
          // advance() only sets Threlte's "this frame wants a render" flag —
          // the render itself happens inside scheduler.run(), which is what
          // its requestAnimationFrame loop calls. Calling only advance() here
          // marked frames and drew none, so every captured frame was an
          // unrendered (black) canvas. run() takes a timestamp; feeding it
          // synthetic ones spaced 1/fps apart keeps task deltas deterministic
          // instead of tied to how long the export actually takes.
          gfx.advance();
          gfx.scheduler.run(t0 + (i * 1000) / fps);
        },
      });
      download(blob);
    } catch (e) {
      alert("export failed: " + (e?.message ?? e));
    } finally {
      if (mode) gfx.renderMode?.set(mode);
      pb.cur = wasCur;
      pb.exporting = false;
      pb.exportPct = 0;
      pb.exportStop = false;
    }
  }

  // Fallback: one real-time playback pass through MediaRecorder.
  function exportRealtime(canvas) {
    if (!("MediaRecorder" in window)) return;
    pb.exporting = true;
    pb.exportStop = false;
    // one stream: the canvas video plus the song, when the replay has one
    const track = audioTap()?.stream.getAudioTracks()[0] ?? null;
    const stream = new MediaStream([
      ...canvas.captureStream(30).getVideoTracks(),
      ...(track ? [track] : []),
    ]);
    const mime = [
      track ? "video/webm;codecs=vp9,opus" : "video/webm;codecs=vp9",
      track ? "video/webm;codecs=vp8,opus" : "video/webm",
      "video/webm",
    ].find((m) => MediaRecorder.isTypeSupported(m)) ?? "video/webm";
    const rec = new MediaRecorder(stream, {
      mimeType: mime,
      videoBitsPerSecond: 3_000_000,
      audioBitsPerSecond: 128_000,
    });
    const chunks = [];
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = () => {
      download(new Blob(chunks, { type: "video/webm" }));
      pb.exporting = false;
      pb.exportStop = false;
    };
    pb.cur = 0;
    setPlaying(true);
    rec.start();
    const poll = setInterval(() => { // playback auto-pauses at the end
      if (!pb.playing || pb.exportStop) { clearInterval(poll); rec.stop(); }
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
  <div id="stage" bind:this={wrap}>
    <!-- Threlte defaults to AgX tone mapping, which rolls the highlights
         off and desaturates: it turns the toon ramp's three flat bands
         into a soft gradient and mutes the saber red/blue. A cel look
         wants the shaded values written straight through. -->
    <Canvas toneMapping={NoToneMapping}>
      <Scene />
    </Canvas>
  </div>

  <div id="top" class="overlay">
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
  </div>

  {#if pose.on}
    <PosePanel />
  {/if}

  <div id="bottom" class="overlay">
    <div id="hud" class="panel">
      {#if pb.rep && c}
        t <b>{time.toFixed(2)}s</b> · tick <b>{tickIdx}</b>/{lastTick()} ·
        return <b>{c.reward.toFixed(2)}</b> ·
        score <b>{c.score.toFixed(0)}</b>/{maxPts} ·
        <span class="good">good {c.good}</span> /
        <span class="bad">bad {c.bad}</span> /
        <span class="miss">miss {c.miss}</span> of {scoreable}
        {#if nBombs}· <span class="miss">bombs {c.bombs}</span>/{nBombs}{/if}
        {#if tickIdx >= lastTick()}· <b>END</b>{/if}
      {:else}
        no replay loaded
      {/if}
    </div>
    <div id="bar" class="panel">
      <button onclick={() => setPlaying(!pb.playing)}>
        {pb.playing ? "⏸ pause" : "▶ play"}
      </button>
      <button onclick={() => { pb.cur = 0; setPlaying(true); }}>↺ restart</button>
      <button onclick={() => setSpeed(pb.speed / 2)}>−</button>
      <span>{speedLabel}</span>
      <button onclick={() => setSpeed(pb.speed * 2)}>+</button>
      <select
        bind:value={pb.fps}
        disabled={pb.exporting}
        title="export frame rate — the replay itself is always 30 ticks/s, so
               above that the viewer interpolates between them"
      >
        {#each [24, 30, 60, 120] as f (f)}
          <option value={f}>{f} fps</option>
        {/each}
      </select>
      <button onclick={exportVideo} disabled={pb.exporting || !pb.rep}
        title={CAN_EXPORT_EXACT
          ? "renders frame by frame at 30 fps — not real time, so a slow "
            + "machine still gets a clean file"
          : "no WebCodecs here: falls back to a real-time capture"}>
        {pb.exporting ? exportLabel : "⏺ export video"}
      </button>
      {#if pb.exporting}
        <button class="stop" onclick={() => (pb.exportStop = true)}
          disabled={pb.exportStop}
          title="end the export here and save a complete file of everything
                 rendered so far (the song is cut to match)">
          {pb.exportStop ? "■ saving…" : "■ stop & save"}
        </button>
      {/if}
      <button class:on={pose.on} onclick={togglePose}>
        {pose.on ? "🧍 posing" : "🧍 pose"}
      </button>
      <button class:on={pb.gauges} onclick={() => (pb.gauges = !pb.gauges)}
        title="per-joint dials, torque bars, foot forces and joint activation (g)">
        📊 gauges
      </button>
      <button class:on={pb.firstPerson}
        onclick={() => (pb.firstPerson = !pb.firstPerson)}
        title="camera on the body's head frame — what the headset sees (f)">
        👁 first-person
      </button>
      {#if pb.firstPerson}
        <span id="fp">
          <label title="first-person vertical field of view">
            fov <input type="range" min="40" max="110" step="1"
              value={pb.fpFov} oninput={(e) => (pb.fpFov = +e.target.value)} />
            <span>{pb.fpFov}°</span>
          </label>
          <label title="head-rotation smoothing time constant — 0 = the exact
                 head frame; a per-tick filter over the recorded head track, so
                 scrubbing and the export match playback">
            smooth <input type="range" min="0" max="1" step="0.05"
              value={pb.fpSmooth} oninput={(e) => (pb.fpSmooth = +e.target.value)} />
            <span>{pb.fpSmooth.toFixed(2)}s</span>
          </label>
        </span>
      {/if}
      {#if pb.rep}
        <!-- preservesPitch off: a speed change should sound like the game's
             practice mode (pitch follows the rate), not a time-stretch -->
        <audio
          bind:this={audioEl}
          preload="auto"
          onloadedmetadata={(e) => { e.currentTarget.preservesPitch = false; }}
        ></audio>
        <span id="snd" title={audio.msg || audio.song}>
          <button
            class:on={audio.status === "ready" && !audio.muted}
            disabled={audio.status !== "ready"}
            onclick={() => (audio.muted = !audio.muted)}
          >{audio.status === "ready" ? (audio.muted ? "🔇" : "🔊") : "🔈"}</button>
          {#if audio.status === "ready"}
            <input
              type="range"
              min="0"
              max="1"
              step="0.02"
              value={audio.vol}
              oninput={(e) => (audio.vol = +e.target.value)}
            />
          {:else}
            <span class="dim">{audioLabel}</span>
          {/if}
        </span>
      {/if}
      <input
        type="range"
        min="0"
        max={lastTick()}
        step="1"
        value={tickIdx}
        oninput={(e) => { setPlaying(false); pb.cur = +e.target.value; }}
      />
    </div>
  </div>
</main>

<style>
  :global(:root) { color-scheme: dark; }
  :global(body) {
    margin: 0; background: #0b0d11; color: #cfd3dc;
    font: 14px/1.4 system-ui, sans-serif; overflow: hidden;
  }
  main { position: fixed; inset: 0; }
  /* the viewport IS the page; everything else floats on top of it */
  #stage { position: absolute; inset: 0; background: #0b0d11; }
  .overlay {
    position: absolute; left: 0; right: 0; z-index: 2;
    display: flex; flex-direction: column; align-items: center;
    pointer-events: none; /* let orbit/drag reach the canvas… */
  }
  .overlay :is(button, input, .panel) { pointer-events: auto; } /* …except UI */
  #top { top: 0; padding: 10px 12px 0; gap: 6px; }
  #bottom { bottom: 0; padding: 0 12px 8px; gap: 8px; }
  .panel {
    background: rgba(17, 19, 25, 0.72);
    border: 1px solid rgba(58, 64, 80, 0.6);
    border-radius: 10px;
    backdrop-filter: blur(6px);
  }
  #drop {
    border: 1.5px dashed #4a4f5c; border-radius: 10px; padding: 8px 18px;
    color: #9aa0ae; cursor: pointer; font: inherit;
    background: rgba(17, 19, 25, 0.72); backdrop-filter: blur(6px);
  }
  #drop.hover { border-color: #7aa2f7; color: #cfd3dc; }
  #bar {
    display: flex; gap: 10px; align-items: center; padding: 8px 14px;
    flex-wrap: wrap; justify-content: center; max-width: 94vw;
  }
  button, select {
    background: #232733; color: #cfd3dc; border: 1px solid #3a4050;
    border-radius: 6px; padding: 4px 12px; cursor: pointer; font: inherit;
  }
  select { padding: 4px 6px; }
  button:hover { background: #2d3342; }
  button:disabled { opacity: 0.5; cursor: default; }
  button.on { border-color: #7aa2f7; color: #e8eaf0; }
  button.stop { border-color: #e06c75; color: #e8eaf0; }
  input[type="range"] { width: min(40vw, 480px); }
  #snd { display: flex; gap: 6px; align-items: center; }
  #fp { display: flex; gap: 10px; align-items: center; color: #9aa0ae;
    font-size: 12px; font-variant-numeric: tabular-nums; }
  #fp label { display: flex; gap: 4px; align-items: center; }
  #fp label > span { min-width: 4ch; color: #cfd3dc; }
  #fp input[type="range"] { width: 80px; }
  #snd input[type="range"] { width: 70px; }
  #snd .dim { color: #6b7180; font-size: 12px; max-width: 22ch; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
  #hud {
    font-variant-numeric: tabular-nums; color: #9aa0ae; padding: 5px 14px;
  }
  #hud b { color: #e8eaf0; }
  .good { color: #7dd3a0; }
  .bad { color: #f0a35e; }
  .miss { color: #e06c75; }
</style>
