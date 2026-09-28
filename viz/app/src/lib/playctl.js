// Play-mode control for the Flux WebXR game: chart list, run lifecycle and
// the game clock. The DOM shell (App.svelte) and the in-canvas loop
// (FluxPlay.svelte) both drive the game through these functions; the
// scripted-pose hook for tests (window.__fluxPlay) does too.
import PACK from "./fluxcharts.data.js";
import { loadFluxCore } from "./fluxcore.js";
import { FluxSession, LEAD_IN_S } from "./fluxgame.js";
import { DT, counts, isFlux, isHuman } from "./replay.js";
import { game, loadText, pb, play, recordBest } from "./state.svelte.js";
import { sfx } from "./sfx.js";

export const DEFAULT_CHART = "train-1";

/** Every playable chart: the bundled pack, plus the chart of the Flux
 * replay that was open when play mode started (to race the agent). */
export function chartEntries() {
  const list = PACK.map((c) => ({ ...c, group: "pack" }));
  const saved = game.saved;
  if (saved?.rep && isFlux(saved.rep) && !isHuman(saved.rep)) {
    const end = counts(saved.rep, saved.rep.ticks.length - 1);
    list.push({
      id: "replay",
      name: `Agent replay: ${saved.name}`,
      blurb: `the agent scored ${end.score.toFixed(0)} (good ${end.good}, ` +
        `bad ${end.bad}, miss ${end.miss}) — unmultiplied`,
      group: "replay",
      config: saved.rep.config,
      // the replay header's chart (rounded to 0.1 mm by the recorder)
      chart: saved.rep.chart.map((o) => ({ ...o })),
      agentScore: end.score,
    });
  }
  return list;
}

export const entryById = (id) => chartEntries().find((c) => c.id === id) ?? null;

export async function enterPlay() {
  if (play.on) return;
  game.saved = { rep: pb.rep, name: pb.name, cur: pb.cur };
  pb.playing = false;
  play.on = true;
  play.phase = "menu";
  play.error = "";
  play.chartId = game.saved.rep && isFlux(game.saved.rep) && !isHuman(game.saved.rep)
    ? "replay" : (play.chartId ?? DEFAULT_CHART);
  pb.rep = null;
  try {
    game.core ??= await loadFluxCore();
    play.ready = true;
  } catch (e) {
    play.error = "could not load the Flux judge: " + (e?.message ?? e);
  }
  if (play.xrSupported === null) {
    play.xrSupported = !!(await navigator.xr?.isSessionSupported?.("immersive-vr")
      .catch(() => false));
  }
}

/** Leave play mode; restores the replay that was open before, unless
 * `keepRep` (the finished run is being watched in the viewer instead). */
export function exitPlay({ keepRep = false } = {}) {
  endRun();
  play.on = false;
  play.phase = "menu";
  if (!keepRep && game.saved) {
    pb.rep = game.saved.rep;
    pb.name = game.saved.name;
    pb.cur = game.saved.cur;
    pb.seek++;
  }
  game.saved = null;
}

export function endRun() {
  game.session?.free();
  game.session = null;
  play.stats = null;
  play.time = 0;
}

/** (Re)start the selected chart: countdown, then chart time 0. */
export function startRun() {
  const entry = entryById(play.chartId) ?? entryById(DEFAULT_CHART);
  if (!game.core || !entry) return false;
  endRun();
  try {
    game.session = new FluxSession(game.core, entry, { onEvent });
  } catch (e) {
    play.error = e.message;
    return false;
  }
  play.chartId = entry.id;
  play.error = "";
  play.phase = "countdown";
  play.paused = false;
  play.stats = game.session.stats;
  game.clock = 0;
  pb.rep = game.session.rep;
  pb.name = entry.name;
  pb.cur = 0;
  pb.seek++;
  sfx.unlock();
  return true;
}

/** Step to the next / previous chart in the list (menu or results). */
export function cycleChart(d = 1) {
  const list = chartEntries();
  const i = list.findIndex((c) => c.id === play.chartId);
  play.chartId = list[(i + d + list.length) % list.length].id;
}

function onEvent(e, obj) {
  sfx.event(e, obj);
  game.onHaptic?.(e, obj);
}

/**
 * Advance the game clock by `dt` seconds (one rendered frame) and feed the
 * session this frame's pose: `getSample(t)` returns {palms, head, quat} in
 * env world coords for game time t, or null when nothing is tracked. The
 * palms are shown in the menu too, so the player sees their hands.
 */
export function advance(dt, getSample) {
  const s = game.session;
  const live = s && (play.phase === "countdown" || play.phase === "playing");
  if (live && !play.paused) {
    // game.speed: a test-only clock scale (1 in play)
    game.clock += dt * (game.speed ?? 1);
    play.time = game.clock - LEAD_IN_S;
    if (play.phase === "countdown" && play.time >= 0) play.phase = "playing";
  }
  const sample = getSample?.(play.time) ?? null;
  if (sample) {
    play.palms = sample.palms;
    play.head = sample.head;
  }
  if (!live || play.paused) return;
  if (sample) s.feed(play.time, sample.palms, sample.head, sample.quat);
  if (s.stats !== play.stats) play.stats = s.stats;
  pb.cur = Math.max(0, play.time / DT);
  if (s.done) finishRun();
}

function finishRun() {
  play.phase = "done";
  const st = game.session.stats;
  play.newBest = recordBest(play.chartId, st.score);
  sfx.finish();
}

function download(text, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "application/jsonl" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

const runName = () =>
  `flux_${play.chartId}_${new Date().toISOString().replace(/[:.]/g, "-")}`;

/** Save the current run as a viewer replay (.jsonl). */
export function exportRun() {
  if (game.session) download(game.session.toJSONL(), runName() + ".jsonl");
}

/** Leave play mode and open the run just played in the replay viewer. */
export function watchRun() {
  if (!game.session) return;
  const text = game.session.toJSONL();
  const name = runName();
  exitPlay({ keepRep: true });
  loadText(text, name);
}

// Automation / test hook: drive the real game path without a headset.
// `setScript((t) => ({palms, head, quat}))` supplies env-world poses per
// frame exactly where the XR read would (FluxPlay.svelte).
if (typeof window !== "undefined") {
  window.__fluxPlay = {
    play, pb, enterPlay, exitPlay, startRun, chartEntries,
    setScript: (fn) => { game.script = fn; },
    setClockScale: (v) => { game.speed = v; },
    session: () => game.session,
  };
}
