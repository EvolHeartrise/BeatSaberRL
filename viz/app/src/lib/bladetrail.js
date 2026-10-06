// The blades' recent history, shared by the swept RIBBON (Saber.svelte)
// and the TIP TRAIL (TipTrail.svelte): one PoseTrail (lib/tiptrail.js)
// filled from the drawn pose itself — qAt (the joint-space Hermite every
// between-tick pose comes from) + bodyFK, one FK per grid sample giving
// both hands' blade tip AND base. So the ribbon's outer edge IS the tip
// line, and both end exactly on the blade the scene draws at the clock.
import { SABER_LEN, bodyFK, fkSaber, handleLen, qAt, tickHzOf } from "./replay.js";
import { PoseTrail, TIP_TRAIL_MAX_S, TrailView } from "./tiptrail.js";

export const RIBBON_S = 0.1; // the swept ribbon's window, replay seconds
// sample layout (scene coords, z flipped): tip L, tip R, base L, base R
export const STRIDE = 12;
export const TIP = (s) => s * 3;
export const BASE = (s) => 6 + s * 3;

/** Seconds of history the ring must hold for the windows on screen. */
export const trailCover = (tipOn, tipS) => (tipOn ? Math.max(RIBBON_S, tipS) : RIBBON_S);

export function createBladeTrail() {
  const trail = new PoseTrail({ stride: STRIDE, maxS: TIP_TRAIL_MAX_S });
  const views = new Map(); // windowS -> TrailView
  const q = new Array(36); // qAt scratch
  let rep = null, hl = 0, hz = 30, cur = NaN, cover = NaN;

  // both blades at replay time t (s): tip = hand + (hl + SABER_LEN) dir,
  // base = hand + hl dir (replay.js tipOf / baseOf), into scene coords
  function sample(t, out, off) {
    const c = t * hz;
    const k = Math.min(Math.floor(c), rep.ticks.length - 1);
    const fk = bodyFK(qAt(rep, k, c - k, q));
    for (let s = 0; s < 2; s++) {
      const sab = fkSaber(fk, s);
      const T = off + TIP(s), B = off + BASE(s);
      const lt = hl + SABER_LEN;
      out[T] = sab[0] + lt * sab[3];
      out[T + 1] = sab[1] + lt * sab[4];
      out[T + 2] = -(sab[2] + lt * sab[5]);
      out[B] = sab[0] + hl * sab[3];
      out[B + 1] = sab[1] + hl * sab[4];
      out[B + 2] = -(sab[2] + hl * sab[5]);
    }
  }

  return {
    trail,
    /** Bring the history to clock `curTick` (fractional tick of `r`),
     * holding `coverS` seconds. Idempotent: every consumer calls it each
     * frame with the same arguments and only the first does any work. */
    ensure(r, curTick, coverS) {
      if (r !== rep) {
        rep = r;
        hl = handleLen(r);
        hz = tickHzOf(r);
        trail.reset();
        cur = NaN;
      }
      if (curTick === cur && coverS === cover) return;
      cur = curTick;
      cover = coverS;
      trail.update(curTick / hz, coverS, sample);
    },
    /** The last `windowS` seconds as a polyline (shared, built once per
     * clock). */
    view(windowS) {
      let v = views.get(windowS);
      if (!v) views.set(windowS, (v = new TrailView(trail)));
      v.build(windowS);
      return v;
    },
  };
}

// the scene's one instance (both Saber components and the tip trail)
export const bladeTrail = createBladeTrail();
