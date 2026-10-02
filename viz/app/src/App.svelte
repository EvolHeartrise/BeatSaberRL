<script>
  import { Canvas } from "@threlte/core";
  import { NoToneMapping } from "three";
  import Scene from "./lib/Scene.svelte";
  import PosePanel from "./lib/PosePanel.svelte";
  import MapCard from "./lib/MapCard.svelte";
  import { audio, pb, play, pose, songClock, lastTick, loadReplay,
    restart, seekTo, setPlaying, setSpeed, step, togglePose } from "./lib/state.svelte.js";
  import { toggleXRSession } from "@threlte/xr";
  import { chartEntries, enterPlay, exitPlay, exportRun, startRun,
    watchRun } from "./lib/playctl.js";
  import { sfx } from "./lib/sfx.js";
  import { songBlob } from "./lib/audio.js";
  import { NO_NUDGE, syncAction } from "./lib/sync.js";
  import { onMount } from "svelte";
  import { fetchReplay, fileToReplay, nameFor, replayPathFrom,
    replayUrl } from "./lib/urlload.js";
  import { DT } from "./lib/replay.js";
  // site build (replays.flee.systems): no Flux game, no pose mode, no
  // file-picker button — see lib/build.js
  import { FLUX, POSE, PICKER } from "./lib/build.js";

  let dropHover = $state(false);
  let fileInput = $state(null);

  // the timeline slider's position
  const tickIdx = $derived(
    pb.rep ? Math.min(Math.floor(pb.cur), lastTick()) : 0);
  // the map card shows whenever the replay names a map (not in play mode)
  const cardShown = $derived(!!pb.rep?.map && !play.on);

  // ---- song audio ---------------------------------------------------------
  // Fetch the map's audio whenever a replay that names one is loaded. The
  // blob URL is revoked on replacement; the flag guards a replay swapped out
  // mid-download from clobbering the next one's state.
  // $state, not a plain let: bind:this on a plain variable is not a tracked
  // write, so the sync effect below would never learn the element arrived.
  let audioEl = $state(null);
  let blobUrl = null;
  $effect(() => {
    const m = pb.rep?.map;
    if (!m) return;
    let stale = false;
    songBlob(m.hash, (stage) => { if (!stale) audio.msg = stage; })
      .then((blob) => {
        if (stale) return;
        if (!audioEl) return;
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

  // ---- live A/V sync ------------------------------------------------------
  // The SONG is the master clock: once playing it runs free, Scene.svelte's
  // frame loop eases pb.cur onto it (songClock below), and the only thing
  // ever done to a sounding element is a ±2% playbackRate nudge. It is
  // seeked only while it is SILENT (paused) or on a deliberate jump —
  // scrub, tick-step, restart, a new replay, all of which bump pb.seek —
  // plus one last-resort seek if the clocks end up more than 0.5 s apart.
  //
  // It used to be the other way round: pb.cur was authoritative and this
  // effect wrote `el.currentTime = want` on any frame where the two were
  // more than 80 ms apart. That is what chopped the sound, and it re-armed
  // itself every frame — see the header of lib/sync.js.
  //
  // This effect reads pb.cur, so it still reconciles once per frame while
  // playing; the difference is that nearly every one of those frames is now
  // a no-op.
  let nudge = NO_NUDGE;   // active playbackRate window, {rate, until}
  let seenSeek = -1;      // last pb.seek acted on
  const setRate = (el, r) => {
    // never write the same rate twice: a playbackRate write resets the
    // element's resampler, so a per-frame write is its own little glitch
    if (Math.abs(el.playbackRate - r) > 1e-6) el.playbackRate = r;
  };
  $effect(() => {
    // Every reactive read happens up front, before any early return: `||`
    // short-circuits, so testing the element first would leave this effect
    // subscribed to nothing on the run where the element is still null, and
    // it would never run again.
    const el = audioEl;
    const ready = audio.status === "ready";
    const want = audio.offset + pb.cur * DT; // song time for this frame
    const rate = pb.speed, vol = audio.muted ? 0 : audio.vol;
    const seek = pb.seek;
    const run = pb.playing && !pose.on;
    if (!el || !ready) return;
    el.volume = vol;

    const jumped = seek !== seenSeek;
    seenSeek = seek;

    if (want < 0) {
      // the recording starts before the song does (the chart is re-based so
      // the first note lands at 2 s, which can predate t=0 in song time)
      if (!el.paused) el.pause();
      if (el.currentTime !== 0) el.currentTime = 0;
      setRate(el, rate); // setSpeed already clamps to [0.25, 4]
      nudge = NO_NUDGE;
      return;
    }

    if (el.duration && want >= el.duration) {
      // the song ran out before the replay did. Leave the element alone: a
      // play() on an ended element seeks back to 0 and starts the song over,
      // and seeking it to a time past its duration only re-arms that.
      if (!el.paused) el.pause();
      setRate(el, rate);
      nudge = NO_NUDGE;
      return;
    }

    if (!run) {
      if (!el.paused) el.pause();
      setRate(el, rate);
      // a silent element can be seeked for free, so park it exactly on the
      // frame being shown: pausing, scrubbing and stepping all land here,
      // and playback must resume from what is on screen
      if (Math.abs(el.currentTime - want) > 0.01) el.currentTime = want;
      nudge = NO_NUDGE;
      return;
    }

    if (jumped) {
      // the user moved the clock: one deliberate seek, then hands off
      el.currentTime = want;
      nudge = NO_NUDGE;
      setRate(el, rate);
    } else if (!el.seeking && el.readyState >= 2) {
      // HAVE_CURRENT_DATA or better, and not mid-seek: currentTime means
      // something, so the drift reading is real
      const act = syncAction({ sim: want, audio: el.currentTime,
                               now: performance.now(), active: nudge });
      if (act.action === "seek") el.currentTime = want; // hopelessly apart
      nudge = act.action === "nudge"
        ? { rate: act.rate, until: act.until } : NO_NUDGE;
      setRate(el, rate * nudge.rate);
    }
    // may reject until the first user gesture; the next frame retries
    if (el.paused && !el.ended) el.play().catch(() => {});
  });

  // The song's clock, for Scene.svelte's frame loop. Null whenever the song
  // is not actually sounding — then the replay clock free-runs on the frame
  // delta, which is the no-song behaviour the viewer always had.
  $effect(() => {
    const el = audioEl;
    songClock.tick = () => {
      if (!el || audio.status !== "ready") return null;
      // Mid-seek the element is silent but its official playback position is
      // already AT the target, so the replay holds there for the fraction of
      // a second the seek takes. Letting it free-run instead would leave it
      // ahead of the song by however long the seek took, and the follow
      // filter would then yank it backwards once the sound returned.
      if (!el.seeking && (el.paused || el.readyState < 2)) return null;
      return (el.currentTime - audio.offset) / DT;
    };
    return () => { songClock.tick = null; };
  });

  const audioLabel = $derived(
    audio.status === "ready" ? (audio.song || "song")
    : audio.status === "loading" ? (audio.msg || "song…")
    : audio.status === "error" ? "no audio"
    : audio.status === "none" ? "no song"
    : "");

  function load(data, name) {
    try {
      loadReplay(data, name);
      urlLoad.error = "";   // a dropped file replaces a link's replay/error
      urlLoad.gate = false;
    } catch (err) {
      alert("could not load replay: " + err.message);
    }
  }

  function onFiles(files) {
    const f = files && files[0];
    if (f && FLUX && play.on) exitPlay();
    // "x.rpl.gz" -> "x.rpl" -> loadReplay drops .rpl (a Flux run: x.jsonl)
    if (f) fileToReplay(f).then((t) => load(t, f.name.replace(/\.gz$/i, "")),
                              (e) => alert("could not read file: " + e.message));
  }

  // ---- replay named by the page URL (?replay=/r/<file>) ---------------------
  // The hosted viewer is opened from the records site's "watch" links. The
  // replay opens PAUSED behind a play button: nothing on the page has had a
  // user gesture yet, so the song could not start with it, and the replay
  // would run ahead of its own music.
  const urlLoad = $state({ stage: "", loaded: 0, total: 0, error: "",
                           gate: false });
  const pct = $derived(urlLoad.total
    ? Math.min(100, Math.round((100 * urlLoad.loaded) / urlLoad.total)) : null);
  const mb = (n) => (n / 1048576).toFixed(1);
  onMount(async () => {
    let path;
    try {
      path = replayPathFrom(location.search);
    } catch (e) {
      urlLoad.error = e.message;
      return;
    }
    if (!path) return;
    urlLoad.stage = "downloading";
    try {
      const data = await fetchReplay(replayUrl(path, location.origin), (p) => {
        urlLoad.stage = p.stage;
        urlLoad.loaded = p.loaded;
        urlLoad.total = p.total;
      });
      urlLoad.stage = "opening";
      await new Promise((r) => setTimeout(r, 30)); // let the label paint
      if (pb.rep) return; // a file dropped meanwhile wins
      loadReplay(data, path, { title: (rep) => nameFor(rep, path), play: false });
      urlLoad.gate = true;
    } catch (e) {
      urlLoad.error = "could not load this replay: " + e.message;
    } finally {
      urlLoad.stage = "";
    }
  });
  function startWatching() {
    urlLoad.gate = false;
    setPlaying(true);
  }

  function onKey(e) {
    // keys aimed at the pose panel's own controls stay there (slider
    // arrow-key nudges must not step the replay)
    if (e.target && e.target.closest && e.target.closest("#pose, #fp, #play")) return;
    if (FLUX && play.on) {
      // the game owns the keyboard: Esc leaves, Enter (re)starts
      if (e.key === "Escape") exitPlay();
      else if (e.key === "Enter") startRun();
      return;
    }
    if (urlLoad.gate && (e.key === " " || e.key === "Enter")) {
      e.preventDefault(); startWatching(); return;
    }
    if (POSE && e.key === "p") { togglePose(); return; }
    if (e.key === " ") { e.preventDefault(); setPlaying(!pb.playing); }
    else if (e.key === "ArrowRight") step(1);
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "+" || e.key === "=") setSpeed(pb.speed * 2);
    else if (e.key === "-") setSpeed(pb.speed / 2);
    else if (e.key === "r") restart();
    else if (e.key === "g") pb.gauges = !pb.gauges;
    else if (e.key === "h") pb.hitboxes = !pb.hitboxes;
    else if (e.key === "f") pb.firstPerson = !pb.firstPerson;
  }

  const speedLabel = $derived(
    pb.speed.toFixed(2).replace(/0+$/, "").replace(/\.$/, ".0") + "×");

  // ---- the Flux WebXR game (play mode) ------------------------------------
  const entries = $derived(FLUX && play.on ? chartEntries() : []);
  const entry = $derived(entries.find((c) => c.id === play.chartId) ?? null);
  // the chart the finished run belongs to (the select may have moved on)
  const runEntry = $derived(entries.find((c) => c.id === play.runId) ?? null);
  const running = $derived(play.phase === "countdown" || play.phase === "playing");

  function enterVR() {
    sfx.unlock(); // the click is the user gesture WebAudio needs
    play.error = "";
    toggleXRSession("immersive-vr",
      { optionalFeatures: ["local-floor", "bounded-floor", "hand-tracking"] },
      "enter")
      .catch((e) => { play.error = "could not enter VR: " + (e?.message ?? e); });
  }
  function exitVR() {
    toggleXRSession("immersive-vr", undefined, "exit").catch(() => {});
  }
  function leavePlay() {
    if (play.presenting) exitVR();
    exitPlay();
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

<!-- the pose panel shares the top-left corner with the map card, so it
     starts below whichever of the two is showing -->
<main style="--panel-top: {cardShown ? '150px' : '64px'}">
  <div id="stage">
    <!-- Threlte defaults to AgX tone mapping, which rolls the highlights
         off and desaturates: it turns the toon ramp's three flat bands
         into a soft gradient and mutes the saber red/blue. A cel look
         wants the shaded values written straight through. -->
    <Canvas toneMapping={NoToneMapping}>
      <Scene />
    </Canvas>
  </div>

  <div id="top" class="overlay">
    {#if FLUX && play.on}
    <div id="playtitle" class="panel">
      <b>Flux</b> — play it yourself in VR · judged by the env's own Rust
      (wasm), the rules the agent trains on
    </div>
    {:else if PICKER}
    <button id="drop" class:hover={dropHover} onclick={() => fileInput.click()}
      title="open a replay .rpl.gz (or a Flux run .jsonl) — or drop one anywhere">
      {#if pb.rep}
        📂 open replay…
      {:else}
        drop a replay <code>.rpl.gz</code> (or a Flux run <code>.jsonl</code>) here, or click to choose
      {/if}
    </button>
    <input
      type="file"
      accept=".gz,.jsonl,.json"
      hidden
      bind:this={fileInput}
      onchange={(e) => onFiles(e.target.files)}
    />
    {/if}
    <!-- site build: nothing up here — the map card (top-left) names the song -->
  </div>

  {#if !play.on && (urlLoad.stage || urlLoad.error || urlLoad.gate)}
    <div id="urlload" class="overlay">
      {#if urlLoad.stage}
        <div class="panel box" role="status" aria-live="polite">
          <div>{urlLoad.stage === "downloading" ? "downloading replay"
                : urlLoad.stage === "unpacking" ? "unpacking replay"
                : "opening replay"}…</div>
          <div class="meter" class:indet={urlLoad.stage === "downloading" && pct == null}>
            <div style="width: {urlLoad.stage === 'downloading' ? (pct ?? 100) : 100}%"></div>
          </div>
          {#if urlLoad.stage === "downloading"}
            <div class="dim">{mb(urlLoad.loaded)} MB{urlLoad.total ? ` of ${mb(urlLoad.total)} MB · ${pct}%` : ""}</div>
          {/if}
        </div>
      {:else if urlLoad.error}
        <div id="urlerror" class="panel box err" role="alert">
          <div>{urlLoad.error}</div>
          <div class="dim">you can still drop a replay file anywhere on this page</div>
          <button onclick={() => (urlLoad.error = "")}>dismiss</button>
        </div>
      {:else if urlLoad.gate && pb.rep}
        <button id="watch" class="panel box" onclick={startWatching}>
          <span class="big">▶ watch</span>
          <span>{pb.name}</span>
          <span class="dim">{audio.status === "ready" ? "with the song"
            : audio.status === "loading" ? "song loading from BeatSaver…"
            : audio.status === "error" ? "no audio: " + audio.msg : ""}</span>
        </button>
      {/if}
    </div>
  {/if}

  {#if cardShown}
    <MapCard />
  {/if}

  {#if POSE && pose.on}
    <PosePanel />
  {/if}

  <div id="bottom" class="overlay">
    {#if FLUX && play.on}
    <div id="play" class="panel">
      <div class="row">
        <label>chart
          <select bind:value={play.chartId} disabled={running}>
            {#each entries as c (c.id)}
              <option value={c.id}>{c.name}</option>
            {/each}
          </select>
        </label>
        {#if entry}<span class="dim">{entry.chart.length} objects · {entry.blurb}</span>{/if}
      </div>
      <div class="row">
        {#if play.presenting}
          <button onclick={exitVR}>exit VR</button>
        {:else if play.xrSupported}
          <button class="on" onclick={enterVR}
            title="then pull the trigger (or pinch) to start; grip changes the chart">
            🥽 enter VR
          </button>
        {:else if play.xrSupported === false}
          <span class="warn" title="WebXR needs a secure context: serve with
            `npm run xr` and open https://<this-machine>:5173 on the headset">
            no WebXR here — open this page on the headset (npm run xr)</span>
        {/if}
        <button onclick={startRun} disabled={running || !play.ready}
          title="start the chart without a headset (objects fly, nothing is
                 tracked) — Enter">
          {play.phase === "done" ? "↺ play again" : "▶ start (no headset)"}
        </button>
        <button onclick={leavePlay} title="back to the replay viewer (Esc)">
          ✕ replays
        </button>
      </div>
      <!-- no score readout here: the running numbers are the in-game HUD's
           (Hud3D.svelte), the final ones the in-world results text's
           (FluxPlay.svelte) -->
      {#if play.phase === "done" && play.stats}
        <div class="row result">
          {#if runEntry && runEntry.id !== play.chartId}
            <span class="dim">last run: {runEntry.name}</span>
          {/if}
          <button onclick={exportRun} title="save this run as a replay .jsonl">
            💾 save run</button>
          <button onclick={watchRun} title="open this run in the replay viewer">
            ▶ watch run</button>
        </div>
      {/if}
      {#if play.error}<div class="row warn">{play.error}</div>{/if}
    </div>
    {:else}
    <div id="bar" class="panel">
      <button onclick={() => setPlaying(!pb.playing)}>
        {pb.playing ? "⏸ pause" : "▶ play"}
      </button>
      <button onclick={restart}>↺ restart</button>
      <button onclick={() => setSpeed(pb.speed / 2)}>−</button>
      <span>{speedLabel}</span>
      <button onclick={() => setSpeed(pb.speed * 2)}>+</button>
      {#if FLUX}
        <button onclick={enterPlay}
          title="play a Flux chart yourself in VR — the same judge the env uses">
          🥽 play Flux
        </button>
      {/if}
      {#if POSE}
        <button class:on={pose.on} onclick={togglePose}>
          {pose.on ? "🧍 posing" : "🧍 pose"}
        </button>
      {/if}
      <button class:on={pb.gauges} onclick={() => (pb.gauges = !pb.gauges)}
        title="per-joint dials, torque bars, foot forces and joint activation (g)">
        📊 gauges
      </button>
      <button class:on={pb.firstPerson}
        onclick={() => (pb.firstPerson = !pb.firstPerson)}
        title="camera at the player's head, looking straight down the track (f)">
        👁 first-person
      </button>
      {#if pb.firstPerson}
        <span id="fp">
          <label title="first-person vertical field of view">
            fov <input type="range" min="40" max="110" step="1"
              value={pb.fpFov} oninput={(e) => (pb.fpFov = +e.target.value)} />
            <span>{pb.fpFov}°</span>
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
        oninput={(e) => { setPlaying(false); seekTo(+e.target.value); }}
      />
    </div>
    {/if}
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
  #playtitle { padding: 6px 14px; color: #9aa0ae; }
  #playtitle b { color: #e8eaf0; }
  #play {
    display: flex; flex-direction: column; gap: 6px; padding: 8px 14px;
    max-width: 94vw; align-items: center;
  }
  #play .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
    justify-content: center; }
  #play label { display: flex; gap: 6px; align-items: center; color: #9aa0ae; }
  #play .dim { color: #6b7180; font-size: 12px; }
  #urlload { top: 0; bottom: 0; justify-content: center; z-index: 3; }
  #urlload .box {
    display: flex; flex-direction: column; gap: 8px; align-items: center;
    padding: 16px 22px; max-width: min(92vw, 460px); text-align: center;
  }
  #urlload .dim { color: #6b7180; font-size: 12px; }
  #urlload .err { border-color: #e06c75; color: #e8eaf0; }
  #urlload .meter {
    width: min(70vw, 320px); height: 6px; border-radius: 3px;
    background: #232733; overflow: hidden;
  }
  #urlload .meter > div { height: 100%; background: #7aa2f7;
    transition: width 0.15s; }
  #urlload .meter.indet > div { animation: indet 1.2s ease-in-out infinite;
    width: 30% !important; }
  @keyframes indet { from { transform: translateX(-100%); }
                     to { transform: translateX(340%); } }
  #watch { font: inherit; color: #cfd3dc; cursor: pointer; }
  #watch .big { font-size: 22px; color: #e8eaf0; }
  #watch:hover { border-color: #7aa2f7; }
  .warn { color: #f0a35e; }
</style>
