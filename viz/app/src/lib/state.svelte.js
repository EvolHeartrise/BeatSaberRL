// Playback state shared by the DOM shell and the 3D scene (Svelte 5 runes).
import { clampQ, mirrorQ, parseReplay } from "./replay.js";

export const pb = $state({
  rep: null,
  name: "",
  cur: 0, // fractional tick
  playing: false,
  speed: 1.0,
  exporting: false,
  gauges: true, // per-hand wrist/joint angle dials
  hitboxes: false, // real v7 collision-volume wireframes ('h')
});

// Manual pose mode: slider-driven joint angles take over the body
// (works with or without a replay loaded; omegas are zero).
export const pose = $state({
  on: false,
  q: [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], // per hand, joint order
  ref: null, // static reference note {col, row} at z = 0, or null (off)
});

export function togglePose() {
  pose.on = !pose.on;
  if (pose.on) pb.playing = false; // posing pauses playback
}

export function setPoseJoint(hand, j, rad) {
  pose.q[hand] = clampQ(hand,
    pose.q[hand].map((v, k) => (k === j ? rad : v)));
}

export function resetPoseHand(hand) {
  pose.q[hand] = [0, 0, 0, 0, 0, 0];
}

export function mirrorPoseLR() {
  pose.q[1] = clampQ(1, mirrorQ(pose.q[0]));
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
