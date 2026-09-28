// Playback state shared by the DOM shell and the 3D scene (Svelte 5 runes).
import { clampQ29, mirrorQ29, parseReplay } from "./replay.js";

// Remembered UI preferences. localStorage is not always there (node, a
// private window, file:// with site data blocked), so every access is
// guarded and falls back to the default.
const PREF = "beat-arena.viewer.";
function pref(key, dflt) {
  try {
    const v = localStorage.getItem(PREF + key);
    return v === null ? dflt : v === "1";
  } catch {
    return dflt;
  }
}
function setPref(key, on) {
  try {
    localStorage.setItem(PREF + key, on ? "1" : "0");
  } catch {
    /* a viewer that cannot remember the toggle still works */
  }
}

export const pb = $state({
  rep: null,
  name: "",
  cur: 0, // fractional tick
  seek: 0, // bumped whenever `cur` JUMPS (scrub/step/restart/load) rather
           // than advancing — the only time the song is allowed to seek
  playing: false,
  speed: 1.0,
  exporting: false,
  exportPct: 0, // frame-by-frame export progress, 0..1
  exportStop: false, // 'stop & save' pressed: finish the file with what is done
  fps: 30, // export frame rate; the replay itself is always 30 ticks/s
  gauges: true, // per-joint dial panels + force instrumentation
  hitboxes: false, // real v7 collision-volume wireframes ('h')
  firstPerson: false, // camera on the body's head frame ('f')
  fpFov: 75, // first-person vertical field of view, degrees
  fpSmooth: 0, // first-person head-rotation smoothing, seconds (0 = exact)
  // map metadata card, top-left over the scene ('m'). On by default and
  // remembered across sessions; it only draws for a replay that names a map.
  mapCard: pref("mapCard", true),
});

export function toggleMapCard() {
  pb.mapCard = !pb.mapCard;
  setPref("mapCard", pb.mapCard);
}

// Threlte's render controls, published by Scene.svelte. The frame-by-frame
// exporter drives the render loop one frame at a time, but it lives in the
// DOM shell OUTSIDE the <Canvas>, so it cannot call useThrelte itself. Plain
// object: nothing reactive depends on these.
export const gfx = { advance: null, renderMode: null, scheduler: null };

// The SONG's clock, published by App.svelte for Scene.svelte's frame loop
// (see lib/sync.js): `tick()` returns the replay tick the song is sounding
// right now, or null when the song is not the master clock (no song, paused,
// seeking, still buffering, or a video export in progress). Plain object on
// purpose — the frame loop calls it every tick and must not turn the <audio>
// element into a reactive dependency of the render.
export const songClock = { tick: null };

// Song audio (2026-09-09). `offset` is the song time at replay t = 0, from
// the replay's own header — see lib/audio.js. status: off (no replay) |
// none (replay carries no map) | loading | ready | error.
export const audio = $state({
  status: "off",
  msg: "",
  offset: 0,
  song: "",
  muted: false,
  vol: 0.8,
});

// Manual pose mode (env v10): slider-driven FULL-BODY pose — 29 joint
// angles (§1.4 actuated order) + the free root (position + yaw/pitch/
// roll). Drives the same bodyFK preview path as replay playback; no
// sim roundtrip. Works with or without a replay loaded.
export const ROOT0 = { x: 0, y: 0.94, z: -0.65, yaw: 0, pitch: 0, roll: 0 };

export const pose = $state({
  on: false,
  q: Array(29).fill(0),
  root: { ...ROOT0 },
  ref: null, // static reference note {col, row} at z = 0, or null (off)
});

export function togglePose() {
  pose.on = !pose.on;
  if (pose.on) pb.playing = false; // posing pauses playback
}

export function setPoseJoint(j, rad) {
  pose.q = clampQ29(pose.q.map((v, k) => (k === j ? rad : v)));
}

export function setPoseRoot(key, val) {
  pose.root = { ...pose.root, [key]: val };
}

export function resetPose() {
  pose.q = Array(29).fill(0);
  pose.root = { ...ROOT0 };
}

export function mirrorPoseLR() {
  pose.q = clampQ29(mirrorQ29(pose.q));
}

export function lastTick() {
  return pb.rep ? pb.rep.ticks.length - 1 : 0;
}

export function loadText(text, name) {
  const rep = parseReplay(text); // throws with a useful message on bad input
  pb.rep = rep;
  pb.name = name.replace(/\.[^.]*$/, "");
  pb.cur = 0;
  pb.seek++;
  pb.playing = true;
  audio.song = rep.map?.song ?? "";
  audio.offset = rep.map?.audio_offset_s ?? 0;
  audio.status = rep.map ? "loading" : "none";
  audio.msg = rep.map ? "" : "no map in this replay — annotate_replay_audio.py";
}

export function setPlaying(p) {
  pb.playing = p && !!pb.rep;
}

export function setSpeed(s) {
  pb.speed = Math.min(4, Math.max(0.25, s));
}

// Every deliberate JUMP of the clock goes through here, so the audio effect
// knows the difference between "the clock moved because time passed" (never
// seek the song) and "the user moved it" (seek it, exactly once).
export function seekTo(tick) {
  pb.cur = Math.min(Math.max(tick, 0), lastTick());
  pb.seek++;
}

export function step(d) {
  pb.playing = false;
  seekTo(pb.cur + d);
}

export function restart() {
  seekTo(0);
  setPlaying(true);
}

// ---- the Flux WebXR game (SPEC.md §Flux WebXR game) ------------------------
// Play mode turns the viewer into the game: the scene renders a LIVE
// recording (fluxgame.js LiveReplay, set as pb.rep) that FluxPlay.svelte
// fills from the player's tracked palms, judged by the env's own Rust
// compiled to wasm. `play` is the reactive surface for the HUD/menus; the
// session object itself lives in `game` (plain, mutated every tick).
export const play = $state({
  on: false,          // play mode (vs replay viewing)
  ready: false,       // the wasm judge is loaded
  phase: "menu",      // menu | countdown | playing | done
  presenting: false,  // an immersive-vr session is running
  xrSupported: null,  // null = still asking the browser
  chartId: null,      // selected chart (pack id, or "replay:<hash>")
  runId: null,        // the chart of the current / last run
  time: 0,            // game time (s), negative during the countdown
  paused: false,      // headset off / system menu open
  stats: null,        // the judge's tallies (fluxcore STAT_NAMES)
  palms: null,        // live palms [[x,y,z] L, R] (env world), this frame
  head: null,         // live head [x, y, z] (env world), this frame
  error: "",
  best: {},           // chartId -> best multiplied score in this browser
  newBest: false,     // the finished run set a new best
});

// Not reactive: the running FluxSession, the loaded judge module, the
// replay that was open before play mode (restored on exit), and an
// optional scripted pose source — tests / automation drive the very game
// path the headset does: (t) => {palms, head, quat} in env world coords.
export const game = { session: null, core: null, script: null, saved: null };

const BEST_KEY = "beat-arena.flux.best";
try {
  play.best = JSON.parse(localStorage.getItem(BEST_KEY) ?? "{}") ?? {};
} catch {
  play.best = {};
}
export function recordBest(id, score) {
  if (id == null || !(score > (play.best[id] ?? -1))) return false;
  play.best = { ...play.best, [id]: score };
  try {
    localStorage.setItem(BEST_KEY, JSON.stringify(play.best));
  } catch {
    /* a best score that cannot be remembered is still shown this session */
  }
  return true;
}
