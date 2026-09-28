<script>
  // The Flux WebXR game's in-canvas half (SPEC.md §Flux WebXR game): the
  // <XR> session, the per-frame pose read (controller grips or tracked
  // hands + the headset) that feeds the game clock (playctl.advance), the
  // in-headset controls (trigger / grip), haptics, and the in-world HUD.
  // Objects, palms and the Flux beam are drawn by the viewer's own Scene /
  // Notes / Flux components from the live recording (pb.rep).
  //
  // Frames: the player's floor origin (the local-floor reference space)
  // is placed at scene z = -PLANE_Z0 — where the env's body stands — by an
  // <XROrigin> (which carries the XR camera), and every pose read adds the
  // same offset before converting scene -> env world by the viewer's own
  // (x, y, -z). An explicit offset rather than getOffsetReferenceSpace:
  // one mechanism for the view and the reads, on every runtime.
  import { useTask, useThrelte } from "@threlte/core";
  import { Group } from "three";
  import { XR, XROrigin } from "@threlte/xr";
  import { KIND_CIRCLE, PLANE_Z0 } from "./replay.js";
  import { quatSceneToEnv, rank, sceneToEnv } from "./fluxgame.js";
  import { advance, cycleChart, entryById, startRun } from "./playctl.js";
  import { game, play } from "./state.svelte.js";
  import TextSprite from "./TextSprite.svelte";

  const { renderer } = useThrelte();
  let session = null;

  // ---- pose read ------------------------------------------------------------
  // Palm = the grip space origin (a controller's handle centre is where the
  // palm wraps it). Tracked hands without a grip space: the midpoint of
  // the wrist and the middle knuckle, i.e. the palm centre. A hand that
  // loses tracking keeps its last position for the frame.
  const HANDS = { left: 0, right: 1 };
  // The player's avatar is the palm orbs + Flux beam; controller / hand
  // meshes are never drawn. Empty model factories keep @threlte/xr from
  // fetching controller profiles off a CDN (a LAN-only headset has no
  // internet) and from racing a quick session exit.
  const NO_MODELS = {
    createControllerModel: () => new Group(),
    createHandModel: () => new Group(),
  };
  // local-floor -> scene: the origin offset
  const ORIGIN_Z = -PLANE_Z0;
  const toEnv = (p) => sceneToEnv([p.x, p.y, p.z + ORIGIN_Z]);
  let lastPalms = [[-0.2, 1.2, -0.35], [0.2, 1.2, -0.35]];
  function readXR() {
    const xr = renderer.xr;
    const frame = xr.getFrame?.();
    const ref = xr.getReferenceSpace();
    if (!frame || !ref || !session) return null;
    const viewer = frame.getViewerPose(ref);
    if (!viewer) return null;
    const palms = [lastPalms[0], lastPalms[1]];
    for (const src of session.inputSources) {
      const h = HANDS[src.handedness];
      if (h === undefined) continue;
      let p = null;
      if (src.gripSpace) {
        const pose = frame.getPose(src.gripSpace, ref);
        if (pose) p = pose.transform.position;
      }
      if (!p && src.hand && frame.getJointPose) {
        const w = frame.getJointPose(src.hand.get("wrist"), ref);
        const k = frame.getJointPose(src.hand.get("middle-finger-phalanx-proximal"), ref);
        if (w && k) {
          const a = w.transform.position, b = k.transform.position;
          p = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
        }
      }
      if (p) palms[h] = toEnv(p);
    }
    lastPalms = palms;
    const vp = viewer.transform.position, vo = viewer.transform.orientation;
    return {
      palms,
      head: toEnv(vp),
      quat: quatSceneToEnv([vo.x, vo.y, vo.z, vo.w]),
    };
  }

  useTask((delta) => {
    let get = null;
    if (play.presenting) {
      const s = readXR();
      get = () => s;
    } else if (game.script) {
      get = game.script;
    }
    advance(delta, get);
  });

  // ---- session plumbing -------------------------------------------------------
  function onSessionStart() {
    play.presenting = true;
    session = renderer.xr.getSession();
    session.addEventListener("visibilitychange", onVisibility);
    session.addEventListener("select", onSelect);
    session.addEventListener("squeeze", onSqueeze);
    game.onHaptic = haptic;
  }

  function onSessionEnd() {
    session?.removeEventListener("visibilitychange", onVisibility);
    session?.removeEventListener("select", onSelect);
    session?.removeEventListener("squeeze", onSqueeze);
    session = null;
    play.presenting = false;
    play.paused = false;
    game.onHaptic = null;
  }

  function onVisibility() {
    play.paused = session?.visibilityState !== "visible";
  }

  // trigger (or pinch): start / play again; grip: next chart (left grip:
  // previous). Ignored mid-run.
  function onSelect() {
    if (play.phase === "menu" || play.phase === "done") startRun();
  }
  function onSqueeze(e) {
    if (play.phase === "menu" || play.phase === "done") {
      cycleChart(e.inputSource?.handedness === "left" ? -1 : 1);
      play.phase = "menu";
    }
  }

  function haptic(e, obj) {
    if (!session) return;
    // a good circle buzzes its own hand; bars and faults buzz both
    const want = e.code === 0 && obj.kind === KIND_CIRCLE ? [obj.color] : [0, 1];
    for (const src of session.inputSources) {
      const h = HANDS[src.handedness];
      const act = src.gamepad?.hapticActuators?.[0];
      if (h === undefined || !act || !want.includes(h)) continue;
      try {
        act.pulse?.(e.code === 0 ? 0.5 : 0.9, e.code === 0 ? 25 : 70);
      } catch {
        /* haptics are a nicety */
      }
    }
  }

  // ---- in-world HUD (scene coords; the hit plane is scene z = 0) ------------
  const st = $derived(play.stats);
  const entry = $derived(entryById(play.chartId));
  const countdown = $derived(
    play.phase === "countdown" ? Math.max(1, Math.ceil(-play.time)) : 0);
  const HUD_Z = -0.55, HUD_Y = 1.05, HUD_X = 1.45;
  const MSG = [0, 2.35, -2.4];
  const fmt = (v) => Math.round(v ?? 0).toLocaleString("en-US");
  // accuracy so far: points over the max of every object resolved yet
  // (circles 60, bars 115); re-derived whenever the tallies change
  const accPct = $derived.by(() => {
    const rep = st && game.session?.rep;
    if (!rep) return null;
    let m = 0;
    rep.outcomes.forEach((o, i) => {
      if (o) m += rep.chart[i].kind === KIND_CIRCLE ? 60 : 115;
    });
    return m ? Math.round((100 * st.raw_score) / m) : null;
  });
</script>

<XR referenceSpace="local-floor" offerSession={false} enterGrantedSession={false}
  controllerFactory={NO_MODELS} handFactory={NO_MODELS}
  onsessionstart={onSessionStart} onsessionend={onSessionEnd} />
<XROrigin position={[0, 0, ORIGIN_Z]} />

{#if play.phase === "menu"}
  <TextSprite text={entry ? entry.name : "Flux"} size={0.2} position={MSG} />
  {#if entry?.blurb}
    <TextSprite text={entry.blurb} size={0.09} bold={false} color="#9aa0ae"
      position={[MSG[0], MSG[1] - 0.25, MSG[2]]} />
  {/if}
  {#if play.presenting}
    <TextSprite text="trigger: start  ·  grip: change chart" size={0.08}
      bold={false} color="#7aa2f7" position={[MSG[0], MSG[1] - 0.45, MSG[2]]} />
  {/if}
  {#if play.best[play.chartId] != null}
    <TextSprite text={`best ${fmt(play.best[play.chartId])}`} size={0.08}
      bold={false} color="#ffd166" position={[MSG[0], MSG[1] - 0.62, MSG[2]]} />
  {/if}
{:else if play.phase === "countdown"}
  <TextSprite text={String(countdown)} size={0.5} position={MSG} />
{:else if play.phase === "done" && st}
  <TextSprite text={`${fmt(st.score)}  ${rank(st)}`} size={0.3} position={MSG}
    color={play.newBest ? "#ffd166" : "#e8eaf0"} />
  <TextSprite
    text={`good ${st.good} · bad ${st.bad} · miss ${st.misses} · max combo ${st.max_combo}`}
    size={0.09} bold={false} position={[MSG[0], MSG[1] - 0.3, MSG[2]]} />
  <TextSprite text={play.newBest ? "new best!" : `best ${fmt(play.best[play.runId])}`}
    size={0.09} color="#ffd166" position={[MSG[0], MSG[1] - 0.47, MSG[2]]} />
  {#if play.presenting}
    <TextSprite text="trigger: play again  ·  grip: change chart" size={0.08}
      bold={false} color="#7aa2f7" position={[MSG[0], MSG[1] - 0.64, MSG[2]]} />
  {/if}
{/if}

{#if play.paused && play.phase !== "menu"}
  <TextSprite text="paused" size={0.25} position={[0, 1.6, -1.8]} />
{/if}

{#if st && (play.phase === "playing" || play.phase === "countdown")}
  <!-- left: combo + multiplier; right: score + accuracy -->
  <TextSprite text={`${st.combo}`} size={0.22}
    position={[-HUD_X, HUD_Y + 0.12, HUD_Z]} />
  <TextSprite text="combo" size={0.07} bold={false} color="#9aa0ae"
    position={[-HUD_X, HUD_Y - 0.04, HUD_Z]} />
  <TextSprite text={`×${st.multiplier}`} size={0.14} color="#7aa2f7"
    position={[-HUD_X, HUD_Y - 0.2, HUD_Z]} />
  <TextSprite text={fmt(st.score)} size={0.18}
    position={[HUD_X, HUD_Y + 0.12, HUD_Z]} />
  <TextSprite text={accPct === null ? "—" : `${accPct}%`} size={0.1}
    color="#9aa0ae" position={[HUD_X, HUD_Y - 0.06, HUD_Z]} />
{/if}

