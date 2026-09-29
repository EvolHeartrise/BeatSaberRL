// The in-game HUD's numbers (Hud3D.svelte draws them), per tick. Pure
// logic, no three/svelte imports, unit-tested in node (test/hud.test.js).
//
// Reference: ScoreSaber's replay viewer (github.com/ScoreSaber/watch @6926860,
// an ArcViewer fork, GPL-3.0). ScoreManager.InitializeMapScore walks the
// replay's scoring events once and stores the running totals on each event,
// and UpdateBeat shows the last event at or before the playhead. This module
// does the same over our replay's recorded events, as per-tick cumulative
// arrays built once per replay. hudAt(rep, k) is O(1) and a pure function
// of the tick, so scrubbing and the frame-by-frame export stay exact.
//
// The rules are this project's own scoring walk, which is also the watch's:
//  - multiplier 1/2/4/8, the bar fills in 2/4/8 hits, and the promoted
//    multiplier applies to the hit that fills it. A break (bad cut, miss,
//    bomb hit, head entering a wall) sets combo to 0, drops the multiplier
//    one tier and empties the bar. This is python/beat_arena_env/
//    scoresaber.py ComboTracker, the walk publish_replays.py uses to verify
//    a replay's multiplied score, and the Rust ComboWalk. A "wall" event (a
//    tick spent inside a wall) does NOT break (spec §12b).
//  - a note's points (the cut plus its later follow-through "post") commit
//    at its cut-time multiplier.
//  - the max score so far walks the resolved scoreable notes (good, bad,
//    miss; bombs and walls are not scoreable) with an unbroken
//    multiplier. That is the converter's ImmediateMaxPossibleScore walk, and
//    the "accuracy keeps the no-break ideal denominator" rule of spec §12b.
//
// Timing, as in the base game: combo, the multiplier and misses change at
// the cut, and a note's multiplied score and its share of the max both land
// when the note FINISHES scoring (its last follow-through post, or at once
// for a note with no post: a bad cut, a miss, a chain link). So the accuracy
// never dips while a swing's follow-through is still pending. The final
// score equals ComboTracker's to the point, which the tests pin.
//
// Energy follows the watch's PlayerPositionManager.InitializeEnergyValues
// and convert_scoresaber.py's .dat energy stream. It starts at 0.5, a good
// cut heals 0.01 (a chain link 0.002), a bad cut costs 0.1 (link 0.025), a
// miss 0.15 (link 0.03), a bomb 0.15, and each tick inside a wall drains
// 1.3 / s. The env never fails a run, so, like the converter, the value is
// floored just above zero instead of ending the level.
import { KIND_BOMB, KIND_CHAIN_LINK, KIND_MAX } from "./replay.js";

// ScoringUtils.ComboMultipliers / HitsNeededForComboIncrease
export const MULTS = [1, 2, 4, 8];
export const NEED = [2, 4, 8, Infinity];

// base-game energy constants (the watch's, and the converter's)
export const ENERGY_START = 0.5;
export const ENERGY_FLOOR = 0.001; // convert_scoresaber.py: no fail
const HEAL = 0.01, HEAL_LINK = 0.002;
const BAD = 0.1, BAD_LINK = 0.025;
const MISS = 0.15, MISS_LINK = 0.03;
const BOMB = 0.15;
const WALL_PER_S = 1.3;

const clampE = (e) => Math.min(Math.max(e, ENERGY_FLOOR), 1);

const _cache = new WeakMap(); // rep -> per-tick cumulative arrays

/** Build the per-tick HUD arrays for a (standard) replay. */
function build(rep) {
  const n = rep.ticks.length;
  const out = {
    score: new Float64Array(n), max: new Float64Array(n),
    combo: new Int32Array(n), maxCombo: new Int32Array(n),
    multIdx: new Uint8Array(n), progress: new Uint8Array(n),
    misses: new Int32Array(n), energy: new Float64Array(n),
  };
  const hz = rep.tickHz ?? 30;
  const kind = (i) => rep.chart[i]?.kind;
  const isLink = (i) => kind(i) === KIND_CHAIN_LINK;
  // posts still to land per note: a good cut's score finishes at the last
  const pendingPosts = rep.postsByNote
    ? rep.postsByNote.map((l) => (l ? l.length : 0)) : [];
  const noteMult = new Map(); // note -> multiplier at its cut
  const notePts = new Map();  // note -> points landed so far (cut + posts)
  const noteMax = new Map();  // note -> its max share, until it finishes

  let mi = 0, prog = 0, combo = 0, maxCombo = 0, misses = 0;
  let mmi = 0, mprog = 0; // the unbroken (max-score) walk
  let score = 0, max = 0, energy = ENERGY_START;

  const brk = () => { combo = 0; misses++; if (mi > 0) mi--; prog = 0; };
  const advance = () => {
    if (mi < MULTS.length - 1 && ++prog >= NEED[mi]) { mi++; prog = 0; }
  };
  // a scoreable note resolved: its max share at the unbroken multiplier
  const maxShare = (i) => {
    if (mmi < MULTS.length - 1 && ++mprog >= NEED[mmi]) { mmi++; mprog = 0; }
    return KIND_MAX[kind(i)] * MULTS[mmi];
  };
  const finish = (i) => {
    score += notePts.get(i) * noteMult.get(i);
    max += noteMax.get(i);
    notePts.delete(i); noteMax.delete(i);
  };

  for (let k = 0; k < n; k++) {
    for (const e of rep.ticks[k].events) {
      if (e.cut) {
        const i = e.cut.note;
        const share = maxShare(i);
        if (e.cut.good) {
          combo++;
          if (combo > maxCombo) maxCombo = combo;
          advance();
          noteMult.set(i, MULTS[mi]);
          notePts.set(i, e.cut.points || 0);
          noteMax.set(i, share);
          energy = clampE(energy + (isLink(i) ? HEAL_LINK : HEAL));
          if (!(pendingPosts[i] > 0)) finish(i);
        } else {
          max += share;
          brk();
          energy = clampE(energy - (isLink(i) ? BAD_LINK : BAD));
        }
      } else if (e.post) {
        const i = e.post.note;
        if (notePts.has(i)) {
          notePts.set(i, notePts.get(i) + e.post.points);
          if (--pendingPosts[i] <= 0) finish(i);
        } else {
          // a post with no open cut (never emitted by the env): commit it
          // at once at its note's multiplier, like ComboTracker.on_points
          score += e.post.points * (noteMult.get(i) ?? MULTS[mi]);
        }
      } else if (e.miss !== undefined) {
        const i = e.miss.note;
        if (kind(i) !== KIND_BOMB) max += maxShare(i);
        brk();
        energy = clampE(energy - (isLink(i) ? MISS_LINK : MISS));
      } else if (e.bomb !== undefined) {
        brk();
        energy = clampE(energy - BOMB);
      } else if (e.wall_enter !== undefined) {
        brk();
      } else if (e.wall !== undefined) {
        energy = clampE(energy - WALL_PER_S / hz);
      }
    }
    out.score[k] = score; out.max[k] = max;
    out.combo[k] = combo; out.maxCombo[k] = maxCombo;
    out.multIdx[k] = mi; out.progress[k] = prog;
    out.misses[k] = misses; out.energy[k] = energy;
  }
  return out;
}

/** HUD state after every event up to and including tick k (the DOM HUD's
 * counts() convention). k < 0 is the state before the first tick. */
export function hudAt(rep, k) {
  let a = _cache.get(rep);
  if (!a) { a = build(rep); _cache.set(rep, a); }
  const n = rep.ticks.length;
  if (n === 0 || k < 0) {
    return {
      score: 0, max: 0, pct: 100, grade: "SS", combo: 0, maxCombo: 0,
      mult: 1, progress: 0, fill: 0, misses: 0, fullCombo: true,
      energy: ENERGY_START,
    };
  }
  k = Math.min(Math.floor(k), n - 1);
  const mi = a.multIdx[k];
  const score = Math.round(a.score[k]), max = Math.round(a.max[k]);
  const pct = max === 0 ? 100 : (100 * a.score[k]) / a.max[k];
  return {
    score, max, pct, grade: gradeOf(pct),
    combo: a.combo[k], maxCombo: a.maxCombo[k],
    mult: MULTS[mi], progress: a.progress[k],
    // the ring's fill (Image.fillAmount): progress / hits needed, 0 at 8x
    fill: a.progress[k] / NEED[mi],
    misses: a.misses[k], fullCombo: a.misses[k] <= 0,
    energy: a.energy[k],
  };
}

/** ScoreManager.GradeFromPercentage (percent, 0..100). */
export function gradeOf(pct) {
  if (pct >= 90) return "SS";
  if (pct >= 80) return "S";
  if (pct >= 65) return "A";
  if (pct >= 50) return "B";
  if (pct >= 35) return "C";
  if (pct >= 20) return "D";
  return "E";
}

/** ScoreManager.FormatScore: groups of three separated by spaces. */
export function formatScore(score) {
  const s = String(Math.round(score));
  let out = "";
  for (let end = s.length; end > 0; end -= 3) {
    const chunk = s.slice(Math.max(0, end - 3), end);
    out = out ? chunk + " " + out : chunk;
  }
  return out;
}

/** ScoreManager.GetPercentageString: two decimals and a percent sign. */
export const pctString = (pct) => pct.toFixed(2) + "%";

// ---- Flux ---------------------------------------------------------------------
// The Flux judge (rust/src/combo.rs ComboWalk, compiled into the game's wasm)
// defines combo, the 1/2/4/8 multiplier with its 2/4/8-hit bar, the
// multiplied score, and "breaks" (bad cuts + misses). It has no energy and
// no rank, so the Flux HUD shows neither. Its accuracy is the game's own
// (the in-headset HUD's before this one): the unmultiplied points over the
// max of every object resolved so far (circle 60, bar 115: KIND_MAX).

const _fluxCache = new WeakMap(); // rep -> per-tick cumulative arrays

/** combo.rs's walk over a Flux replay's recorded events. */
function buildFlux(rep) {
  const n = rep.ticks.length;
  const out = {
    score: new Float64Array(n), raw: new Float64Array(n),
    max: new Float64Array(n), combo: new Int32Array(n),
    multIdx: new Uint8Array(n), progress: new Uint8Array(n),
    breaks: new Int32Array(n),
  };
  const noteMult = new Map(); // note -> multiplier at its cut
  let mi = 0, prog = 0, combo = 0, breaks = 0, score = 0, raw = 0, max = 0;
  const brk = () => { combo = 0; breaks++; if (mi > 0) mi--; prog = 0; };
  for (let k = 0; k < n; k++) {
    for (const e of rep.ticks[k].events) {
      if (e.cut) {
        const i = e.cut.note, pts = e.cut.points || 0;
        max += KIND_MAX[rep.chart[i]?.kind] ?? 0;
        raw += pts;
        if (e.cut.good) {
          // promote-before-commit: the hit that fills the bar scores at
          // the new multiplier
          if (mi < MULTS.length - 1 && ++prog >= NEED[mi]) { mi++; prog = 0; }
          noteMult.set(i, MULTS[mi]);
          combo++;
          score += pts * MULTS[mi];
        } else {
          brk();
        }
      } else if (e.post) {
        // follow-through at the note's cut-time multiplier (combo.rs code
        // 5; the Flux judge emits none today)
        raw += e.post.points;
        score += e.post.points * (noteMult.get(e.post.note) ?? MULTS[mi]);
      } else if (e.miss !== undefined) {
        max += KIND_MAX[rep.chart[e.miss.note]?.kind] ?? 0;
        brk();
      }
    }
    out.score[k] = score; out.raw[k] = raw; out.max[k] = max;
    out.combo[k] = combo; out.multIdx[k] = mi; out.progress[k] = prog;
    out.breaks[k] = breaks;
  }
  return out;
}

/** The Flux HUD's numbers from the judge's tallies (fluxcore STAT_NAMES:
 * score, combo, multiplier, progress, breaks, raw_score) and the max of
 * the objects resolved so far. */
export function fluxHud(st, resolvedMax) {
  const mi = Math.max(0, MULTS.indexOf(st.multiplier));
  return {
    score: Math.round(st.score), combo: st.combo, mult: MULTS[mi],
    progress: st.progress, fill: st.progress / NEED[mi],
    misses: st.breaks, fullCombo: st.breaks <= 0,
    pct: resolvedMax > 0 ? (100 * st.raw_score) / resolvedMax : 100,
  };
}

/** The max points of every object a Flux recording has resolved. */
export function fluxResolvedMax(rep) {
  let m = 0;
  rep.outcomes.forEach((o, i) => { if (o) m += KIND_MAX[rep.chart[i].kind] ?? 0; });
  return m;
}

/** Flux HUD state after every event up to and including tick k, walked
 * from a Flux replay's events with the judge's own rules. */
export function fluxHudAt(rep, k) {
  let a = _fluxCache.get(rep);
  if (!a) { a = buildFlux(rep); _fluxCache.set(rep, a); }
  const n = rep.ticks.length;
  if (n === 0 || k < 0) {
    return fluxHud({ score: 0, combo: 0, multiplier: 1, progress: 0,
      breaks: 0, raw_score: 0 }, 0);
  }
  k = Math.min(Math.floor(k), n - 1);
  return fluxHud({ score: a.score[k], combo: a.combo[k],
    multiplier: MULTS[a.multIdx[k]], progress: a.progress[k],
    breaks: a.breaks[k], raw_score: a.raw[k] }, a.max[k]);
}
