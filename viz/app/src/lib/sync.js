// Live A/V sync policy for the replay viewer. Pure functions, no DOM, so the
// policy can be tested in node (test/sync.test.js); App.svelte and
// Scene.svelte only apply what these decide.
//
// THE RULE: once the song is playing it RUNS FREE and is the master clock.
// The replay clock (pb.cur) is the slave — Scene.svelte's frame loop eases it
// onto the element's currentTime with followClock() — and the only correction
// ever applied to a SOUNDING element is a small playbackRate nudge. The
// element is seeked only while it is silent (paused) or on a deliberate jump
// (scrub / step / restart / a fresh replay), plus one last-resort seek when
// the two clocks are hopelessly apart.
//
// This is the fix for the 2026-09-14 "audio keeps chopping" report. The
// viewer used to be the other way round: pb.cur was authoritative and the
// audio effect wrote `el.currentTime = want` on any frame where the two were
// more than 80 ms apart. Two things make that chop rather than converge:
//   - a seek on a playing element is audible by itself (the decoder is reset
//     and Ogg/Vorbis lands on a granule boundary, not on the requested time);
//   - it re-arms itself. Setting currentTime starts an ASYNC seek, and the
//     official playback position does not advance while the seek runs, while
//     pb.cur keeps advancing on wall-clock frame deltas. Seek latency on a
//     ~10 MB blob is comfortably more than 80 ms, so the next frame sees the
//     gap again and seeks again — the element spends its life seeking and
//     never plays a clean second.
// The export path never touched any of this: it slices the decoded song by
// array index (lib/record.js), which is why the exported video was fine.

/** Drift thresholds, in seconds unless noted. */
export const SYNC = {
  tol: 0.08,    // below this the gap is inaudible: do nothing
  hard: 0.5,    // beyond this the song is on the wrong bar: seek, once
  nudge: 0.02,  // ±2% playbackRate while closing a gap
  hold: 600,    // ms — minimum life of a nudge window (hysteresis)
  settle: 0.02, // end a nudge early once the gap is this small
};

export const NO_NUDGE = { rate: 1, until: 0 };

/** What to do with the <audio> element this frame.
 *
 * `sim`   — song time the replay clock is showing (offset + pb.cur * DT), s
 * `audio` — the element's currentTime, s
 * `now`   — performance.now(), ms
 * `active`— the nudge window in force, {rate, until} (ms), or NO_NUDGE
 *
 * Returns {action: "none" | "nudge" | "seek", rate, until, drift}. `rate` is
 * a MULTIPLIER on the user's playback speed, never the speed itself.
 */
export function syncAction({ sim, audio, now, active = NO_NUDGE, cfg = SYNC }) {
  const drift = sim - audio; // > 0: the song is BEHIND the replay
  const a = Math.abs(drift);
  if (!Number.isFinite(drift))
    return { action: "none", rate: 1, until: 0, drift: 0 };
  if (a > cfg.hard) // a stall, a fresh source, a jump nobody announced
    return { action: "seek", rate: 1, until: 0, drift };
  if (a > cfg.tol) {
    const rate = 1 + Math.sign(drift) * cfg.nudge; // behind -> speed up
    return { action: "nudge", rate, until: now + cfg.hold, drift };
  }
  // Inside tolerance. A nudge already running keeps running until it has
  // actually closed the gap: dropping it the instant the drift crosses back
  // under `tol` would leave the gap sitting right at the threshold and
  // toggle the rate on and off around it.
  if (active.until > now && a > cfg.settle &&
      Math.sign(drift) === Math.sign(active.rate - 1))
    return { action: "nudge", rate: active.rate, until: active.until, drift };
  return { action: "none", rate: 1, until: 0, drift };
}

/** Replay-clock follow filter, in TICKS (the replay runs `tps` ticks/s).
 *
 * `target` is the tick the song is currently sounding, or null when the song
 * is not the master (no song, paused, seeking, still buffering) — then the
 * clock free-runs on the frame delta exactly as it always did.
 *
 * With a target it still free-runs first and then eases onto it, rather than
 * being assigned it: currentTime is quantized to the audio callback, so
 * following it directly would render the same tick twice and then jump two.
 * A gap past `snap` ticks is not drift (it is a scrub, a load, a long stall),
 * so that one is taken in a single step.
 */
export const FOLLOW_TAU = 0.25; // s — time constant of the ease
export const FOLLOW_SNAP_S = 0.5; // s — a gap past this is a jump, not drift
export const FOLLOW_SNAP = 15;  // ticks at 30 Hz (= FOLLOW_SNAP_S * 30)

// `tps` = replay ticks per second: the replay header's tick_hz (1 / DT;
// 30 unless the recording says otherwise). The snap threshold is 0.5 s of
// replay time at any rate unless `snap` (ticks) is given.
export function followClock({ cur, target, delta, speed = 1,
                              tau = FOLLOW_TAU, tps = 30,
                              snap = FOLLOW_SNAP_S * tps }) {
  const free = cur + delta * tps * speed;
  if (target == null || !Number.isFinite(target)) return free;
  const err = target - free;
  if (Math.abs(err) > snap) return target;
  return free + err * Math.min(1, delta / tau);
}
