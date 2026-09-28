// Tiny synthesized game sounds for the Flux WebXR game (no audio assets):
// a bright blip per good touch (pitched by accuracy), a dull thud per bad
// touch or miss, a chord at the end. WebAudio needs a user gesture to
// start, so `unlock()` is called from the start/enter-VR click.
let ctx = null;

function ac() {
  if (ctx) return ctx;
  const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!AC) return null;
  ctx = new AC();
  return ctx;
}

function tone(freq, dur, { type = "sine", gain = 0.18, at = 0, slide = 0 } = {}) {
  const c = ac();
  if (!c || c.state !== "running") return;
  const t0 = c.currentTime + at;
  const o = c.createOscillator();
  const g = c.createGain();
  o.type = type;
  o.frequency.setValueAtTime(freq, t0);
  if (slide) o.frequency.exponentialRampToValueAtTime(freq * slide, t0 + dur);
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(gain, t0 + 0.008);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g).connect(c.destination);
  o.start(t0);
  o.stop(t0 + dur + 0.02);
}

export const sfx = {
  muted: false,
  unlock() {
    ac()?.resume?.().catch(() => {});
  },
  /** One judged event (fluxcore event, chart object). */
  event(e, obj) {
    if (this.muted) return;
    if (e.code === 0) {
      // circles 50+0..10, bars 100+0..15: pitch climbs with accuracy
      const accFrac = obj.kind === 4 ? e.acc / 10 : e.acc / 15;
      const base = obj.kind === 4 ? (obj.color === 0 ? 660 : 784) : 523;
      tone(base * (1 + 0.5 * accFrac), 0.09, { type: "triangle" });
      if (obj.kind === 5) tone(base * 2, 0.12, { gain: 0.08, at: 0.03 });
    } else {
      tone(140, 0.18, { type: "square", gain: 0.07, slide: 0.6 });
    }
  },
  finish() {
    if (this.muted) return;
    [523, 659, 784].forEach((f, i) => tone(f, 0.5, { gain: 0.1, at: i * 0.07 }));
  },
};
