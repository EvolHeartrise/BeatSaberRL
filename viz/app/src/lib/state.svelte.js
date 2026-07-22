// Playback state shared by the DOM shell and the 3D scene (Svelte 5 runes).
import { parseReplay } from "./replay.js";

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
