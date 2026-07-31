// Playback state shared by the DOM shell and the 3D scene (Svelte 5 runes).
import { clampQ29, mirrorQ29, parseReplay } from "./replay.js";

export const pb = $state({
  rep: null,
  name: "",
  cur: 0, // fractional tick
  playing: false,
  speed: 1.0,
  exporting: false,
  gauges: true, // per-joint dial panels + force instrumentation
  hitboxes: false, // real v7 collision-volume wireframes ('h')
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
