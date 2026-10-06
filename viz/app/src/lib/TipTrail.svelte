<script>
  // The crisp blade-TIP trail of both sabers (lib/tiptrail.js): one thin,
  // screen-space constant-width line per hand (three.js LineSegments2 /
  // LineMaterial — a THREE.Line is stuck at 1 px), a near-white tint of
  // the saber colour at the tip shading to the plain saber colour at the
  // tail, opaque all the way and on a thin dark ink rim so it stays sharp
  // over the translucent swept ribbon. Sampled from the same interpolated pose (body FK of the lerped
  // joint angles) the sabers are drawn from, on a fixed replay-time grid.
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
  import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
  import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
  import { pb, pose } from "./state.svelte.js";
  import { DT, bodyFK, fkSaber, handleLen, qAt, tipOf } from "./replay.js";
  import { NOTE_COLORS } from "./models.js";
  import { TipTrail } from "./tiptrail.js";

  const WIDTH_PX = 3; // core line width, CSS pixels
  const INK_PX = 6; // the dark ink line under it (the scene's cel outline)
  const HEAD_WHITE = 0.6; // how far the tip end is tinted toward white
  const TAIL_DIM = 0.85; // the tail end: the saber colour, slightly dimmed
  const INK = "#0a0c10"; // = the ribbon's ink (Saber.svelte)

  const trail = new TipTrail({ stride: 6 }); // both tips per sample
  const MAXSEG = trail.maxPts - 1;

  // one line per hand, its instance buffers allocated once and rewritten in
  // place every frame (only the used prefix is uploaded). Under each core a
  // wider ink line shares the SAME geometry (no second upload), so the core
  // keeps a dark rim and reads even over its own saber's ribbon.
  const hands = [0, 1].map((s) => {
    const geo = new LineSegmentsGeometry();
    geo.setPositions(new Float32Array(MAXSEG * 6));
    geo.setColors(new Float32Array(MAXSEG * 6));
    geo.instanceCount = 0;
    const mat = new LineMaterial({
      linewidth: WIDTH_PX,
      vertexColors: true,
      worldUnits: false,
    });
    const line = new LineSegments2(geo, mat);
    line.frustumCulled = false; // the bounds move every frame; never stale-culled
    line.renderOrder = 2;
    // the ink depth-tests against the scene but never writes depth: its
    // neighbouring segments' caps sit a hair nearer than the core and
    // would otherwise poke through it as dashes
    const inkMat = new LineMaterial({
      linewidth: INK_PX, color: INK, worldUnits: false, depthWrite: false,
    });
    const ink = new LineSegments2(geo, inkMat);
    ink.frustumCulled = false;
    ink.renderOrder = 1; // drawn first: the core lands on top of it
    const base = new THREE.Color(NOTE_COLORS[s]); // linear working space
    const head = base.clone().lerp(new THREE.Color(1, 1, 1), HEAD_WHITE);
    const tail = base.clone().multiplyScalar(TAIL_DIM);
    return {
      geo, mat, line, inkMat, ink,
      pos: geo.attributes.instanceStart.data,
      col: geo.attributes.instanceColorStart.data,
      head: [head.r, head.g, head.b],
      tail: [tail.r, tail.g, tail.b],
    };
  });
  $effect(() => () => {
    for (const h of hands) { h.geo.dispose(); h.mat.dispose(); h.inkMat.dispose(); }
  });

  // both blade tips at replay time t (s), in scene coords, from the same
  // interpolation the drawn body uses (Scene.svelte: qAt + bodyFK)
  let rep = null, hl = 0;
  function sample(t, out, off) {
    const c = t / DT;
    const k = Math.min(Math.floor(c), rep.ticks.length - 1);
    const fk = bodyFK(qAt(rep, k, c - k));
    for (let s = 0; s < 2; s++) {
      const tp = tipOf(fkSaber(fk, s), hl);
      out[off + s * 3] = tp[0];
      out[off + s * 3 + 1] = tp[1];
      out[off + s * 3 + 2] = -tp[2];
    }
  }

  function hide() {
    trail.reset();
    for (const h of hands) {
      h.geo.instanceCount = 0;
      h.line.visible = h.ink.visible = false;
    }
  }

  // a new replay or a deliberate jump of the clock: start over (the ring
  // would also refill on its own for any jump it cannot extend; this makes
  // it explicit, and a forward step refills too)
  $effect(() => {
    void pb.rep; void pb.seek;
    trail.reset();
  });

  $effect(() => {
    const cur = pb.cur; // the continuous clock: the line is sub-tick
    const on = pb.tipTrail && !pose.on && !!pb.rep && pb.rep.ticks.length > 0;
    if (!on) { hide(); return; }
    rep = pb.rep;
    hl = handleLen(rep);
    trail.update(cur * DT, pb.tipTrailS, sample);
    for (let s = 0; s < 2; s++) {
      const h = hands[s];
      const n = trail.writeSegments(s * 3, h.pos.array, h.col.array, h.head, h.tail);
      h.pos.clearUpdateRanges();
      h.pos.addUpdateRange(0, n * 6);
      h.pos.needsUpdate = true;
      h.col.clearUpdateRanges();
      h.col.addUpdateRange(0, n * 6);
      h.col.needsUpdate = true;
      h.geo.instanceCount = n;
      h.line.visible = h.ink.visible = n > 0;
    }
  });
</script>

<T is={hands[0].ink} />
<T is={hands[1].ink} />
<T is={hands[0].line} />
<T is={hands[1].line} />
