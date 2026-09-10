// Playback state shared by the DOM shell and the 3D scene (Svelte 5 runes).
import { clampQ29, mirrorQ29, parseReplay } from "./replay.js";

export const pb = $state({
  rep: null,
  name: "",
  cur: 0, // fractional tick
  playing: false,
  speed: 1.0,
  exporting: false,
  exportPct: 0, // frame-by-frame export progress, 0..1
  fps: 30, // export frame rate; the replay itself is always 30 ticks/s
  gauges: true, // per-joint dial panels + force instrumentation
  hitboxes: false, // real v7 collision-volume wireframes ('h')
});

// Threlte's render controls, published by Scene.svelte. The frame-by-frame
// exporter drives the render loop one frame at a time, but it lives in the
// DOM shell OUTSIDE the <Canvas>, so it cannot call useThrelte itself. Plain
// object: nothing reactive depends on these.
export const gfx = { advance: null, renderMode: null };

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

export function step(d) {
  pb.playing = false;
  pb.cur = Math.min(Math.max(pb.cur + d, 0), lastTick());
}
