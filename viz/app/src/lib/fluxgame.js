// The Flux WebXR game's session (SPEC.md §Flux WebXR game): a human plays a
// Flux chart with tracked hands. Framework-free (node tests drive it too).
//
// Everything that decides an outcome is the env's own Rust (fluxcore.js ->
// rust/flux-wasm -> rust/src/flux.rs). This file only does what the body
// does for the agent: it turns the player's tracked palms into judge-tick
// inputs. The judge runs at `hz` = 30·S (default 120): S judge ticks per
// 30 Hz env tick, on the same rules — 30 is the env's own clock, bit for
// bit. (Nothing to do with the body sim's physics substeps: there is no
// body here.) Poses arrive at the display rate (72-120 Hz); each judge
// tick's palms are the pose track linearly interpolated to the tick's END
// time (j + 1)·DT/S, and the swept Flux runs from the previous tick's palms —
// the env's p0 -> p1 on a finer clock. Everything is DRAWN at the display
// rate regardless (objects on continuous time, palms from each frame).
//
// The session also RECORDS itself in the viewer's replay format (header
// "game": "flux", "player": "human", "judge_hz"; 30 Hz frames of palms +
// head + events, each event stamped with its exact judge time "t"), so
// the existing renderers draw the live game unchanged — a hit bursts when
// it is judged, not at the frame's end — and a finished run exports as a
// .jsonl the viewer plays back.
import { DT, arcPairs } from "./replay.js";

/** Countdown before chart time 0 (s). */
export const LEAD_IN_S = 3.0;

/** Judge rates the game offers (Hz): 30 = the env's own tick. */
export const JUDGE_HZ = [120, 60, 30];
export const DEFAULT_JUDGE_HZ = 120;

// ---- scene <-> env frames -----------------------------------------------------
// The viewer draws world (x, y, z) at scene (x, y, -z). In XR the player's
// floor origin is placed at scene z = -PLANE_Z0 (the env body's plane,
// world z = PLANE_Z0; FluxPlay.svelte's XROrigin), so a pose read there
// converts with the same flip once that offset is added.
export const sceneToEnv = (p) => [p[0], p[1], -p[2]];
/** Scene quaternion [x, y, z, w] -> env frame (S R S, S = diag(1, 1, -1)). */
export const quatSceneToEnv = (q) => [-q[0], -q[1], q[2], q[3]];

const lerp3 = (a, b, u) => [
  a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
const r5 = (v) => Math.round(v * 1e5) / 1e5;

/**
 * The live recording: the same fields parseReplay() produces, so every
 * viewer helper (pendingNotes, aftermaths, counts, ...) reads it as a
 * replay. A CLASS on purpose — Svelte's $state proxies plain objects, and
 * this one is mutated from the game loop every tick; left unproxied, the
 * renderers re-read it whenever the clock (pb.cur) moves.
 */
export class LiveReplay {
  constructor(chart, config, meta = {}) {
    this.game = "flux";
    this.player = "human";
    this.config = config ?? {};
    this.chart = chart;
    this.ticks = [];
    this.outcomes = chart.map(() => null);
    this.posts = [];
    this.postsByNote = [];
    this.trackAnchor = "head";
    this.map = null;
    this.walls = [];
    this.arcs = arcPairs(chart);
    this.meta = meta;
  }
}

export class FluxSession {
  /**
   * `core` = loadFluxCore()'s module; `entry` = {id, name, chart, config}
   * (a chart-pack entry, or a loaded Flux replay's chart). `hz`: the judge
   * rate, a multiple of 30 (JUDGE_HZ). `onEvent(e, obj)` fires for every
   * judged event as it happens (sound, haptics).
   */
  constructor(core, entry, { onEvent = null, hz = DEFAULT_JUDGE_HZ } = {}) {
    const S = Math.round(hz / 30);
    if (!(S >= 1 && S * 30 === hz)) throw new Error(`judge rate ${hz} Hz: need a multiple of 30`);
    this.entry = entry;
    this.chart = entry.chart;
    this.hz = hz;
    this.S = S;
    // judge tick length: the wasm's own expression (DT / S; DT at S = 1)
    this.tickDt = DT / S;
    this.j = 0; // judge ticks done
    // throws on a chart the env refuses
    this.judge = core.newGame(this.chart, 0, hz);
    this.rep = new LiveReplay(this.chart, entry.config,
      { id: entry.id ?? null, name: entry.name ?? null });
    this.onEvent = onEvent;
    this.prevPalms = null; // last tick's palms (the swept Flux's start)
    this.last = null;      // last pose sample {t, palms, head, quat}
    this.done = false;
    this.stats = this.judge.stats();
    this.frame = { events: [], reward: 0 }; // the 30 Hz frame being filled
  }

  /** Game time of the next judge tick (its end time), as the wasm has it. */
  nextTickTime() {
    return (this.j + 1) * this.tickDt;
  }

  /**
   * One pose sample at game time `t` (s; negative during the lead-in):
   * palms [[x,y,z] L, R] and head [x,y,z] in ENV world coords, head
   * orientation `quat` (env frame, optional). Runs every tick whose end
   * time has been reached. Returns the number of ticks judged.
   */
  feed(t, palms, head, quat = [0, 0, 0, 1]) {
    const s = { t, palms, head, quat };
    if (this.done) { this.last = s; return 0; }
    if (this.prevPalms === null && t >= 0) {
      // chart time 0: the swept Flux starts where the palms are now
      // (the env starts from the body's reset pose)
      const a = this.last && this.last.t < 0 ? this.last : s;
      const u = a === s ? 0 : (0 - a.t) / (s.t - a.t);
      this.prevPalms = [lerp3(a.palms[0], s.palms[0], u),
                        lerp3(a.palms[1], s.palms[1], u)];
    }
    let n = 0;
    while (!this.done && this.prevPalms !== null && this.nextTickTime() <= t) {
      const T = this.nextTickTime();
      const a = this.last && this.last.t < T ? this.last : s;
      if (a === s || s.t === T) {
        // no earlier sample to interpolate from, or one landed on the
        // tick exactly: take it verbatim (a lerp at u = 1 is not exact)
        this.tick(s.palms, s.head, s.quat);
      } else {
        const u = (T - a.t) / (s.t - a.t);
        this.tick([lerp3(a.palms[0], s.palms[0], u), lerp3(a.palms[1], s.palms[1], u)],
          lerp3(a.head, s.head, u), u < 0.5 ? a.quat : s.quat);
      }
      n++;
    }
    this.last = s;
    return n;
  }

  /**
   * Restart the pose track at this sample, dropping the gap before it: the
   * next tick interpolates from here and its swept Flux starts at these
   * palms. For a resume after a pause or a regained track — the player
   * did not move through the gap, so it must not sweep objects.
   */
  rebase(t, { palms, head, quat = [0, 0, 0, 1] }) {
    this.last = { t, palms, head, quat };
    if (this.prevPalms !== null) this.prevPalms = palms;
  }

  /** Judge one tick with palms p1 and head `head` (env world). */
  tick(p1, head, quat) {
    const T = this.nextTickTime();
    const k = this.rep.ticks.length; // the 30 Hz frame this tick falls in
    // a standing human's shoulders sit in the head's frontal plane — the
    // env's reference pose has shoulder z = head z = PLANE_Z0 too
    const { done, events } = this.judge.step(this.prevPalms, p1, head, head[2]);
    this.prevPalms = p1;
    this.j++;
    const before = this.stats.reward;
    this.stats = this.judge.stats();
    const f = this.frame;
    f.reward += this.stats.reward - before;
    for (const e of events) {
      if (e.code === 4) {
        f.events.push({ miss: { note: e.i, t: T } });
        this.rep.outcomes[e.i] = { kind: "miss", reason: "miss", tick: k, t: T, points: 0 };
      } else {
        const good = e.code === 0;
        f.events.push({ cut: { note: e.i, good, reason: e.reason, points: e.pts,
                               pre: e.pre, acc: e.acc, t: T } });
        this.rep.outcomes[e.i] = { kind: good ? "good" : "bad",
                                   reason: e.reason, tick: k, t: T, points: e.pts };
      }
      this.onEvent?.(e, this.chart[e.i]);
    }
    if (done) this.done = true;
    // a 30 Hz frame closes on every S-th judge tick (its end is the env
    // tick's end) — or early, on the run's last judge tick
    if (this.j % this.S === 0 || done) {
      this.rep.ticks.push({
        t: k,
        time: r5(k * DT),
        palms: p1.map((p) => p.map(r5)),
        head: [...head.map(r5), ...quat.map(r5)],
        reward: r5(f.reward),
        events: f.events,
      });
      this.frame = { events: [], reward: 0 };
    }
  }

  /** The run as a viewer-loadable .jsonl (the env recorder's layout). */
  toJSONL() {
    const s = this.stats;
    const header = { header: {
      version: "v10",
      record_rev: 5,
      game: "flux",
      player: "human",
      judge_hz: this.hz,
      track_anchor: "head",
      config: this.rep.config,
      chart: this.chart,
      walls: [],
      chart_id: this.entry.id ?? null,
      chart_name: this.entry.name ?? null,
      result: { score: s.score, raw_score: s.raw_score, good: s.good,
                bad: s.bad, misses: s.misses, max_combo: s.max_combo,
                max_score: s.max_score },
    } };
    return [header, ...this.rep.ticks].map((o) => JSON.stringify(o)).join("\n") + "\n";
  }

  free() {
    this.judge.free();
  }
}

/** Rank letter for a run, from the unmultiplied accuracy (points / max). */
export function rank(stats) {
  const f = stats.max_score > 0 ? stats.raw_score / stats.max_score : 0;
  return f >= 0.9 ? "SS" : f >= 0.8 ? "S" : f >= 0.65 ? "A" : f >= 0.5 ? "B"
    : f >= 0.35 ? "C" : "D";
}

