<script>
  // The crisp blade-TIP trail of both sabers (lib/tiptrail.js): one thin,
  // screen-space constant-width line per hand (three.js LineSegments2 /
  // LineMaterial — a THREE.Line is stuck at 1 px), a near-white tint of
  // the saber colour at the tip shading to the plain saber colour at the
  // tail, opaque all the way and on a thin dark ink rim so it stays sharp
  // over the translucent swept ribbon. The points are the shared blade
  // history (lib/bladetrail.js): the drawn pose (qAt + bodyFK) on a fixed
  // replay-time grid, the same samples the ribbon's outer edge is made of.
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
  import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
  import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
  import { pb, pose } from "./state.svelte.js";
  import { NOTE_COLORS } from "./models.js";
  import { TIP, bladeTrail, trailCover } from "./bladetrail.js";

  const WIDTH_PX = 3; // core line width, CSS pixels
  const INK_PX = 6; // the dark ink line under it (the scene's cel outline)
  const HEAD_WHITE = 0.6; // how far the tip end is tinted toward white
  const TAIL_DIM = 0.85; // the tail end: the saber colour, slightly dimmed
  const INK = "#0a0c10"; // = the ribbon's ink (Saber.svelte)

  const MAXSEG = bladeTrail.trail.maxPts - 1;

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

  function hide() {
    for (const h of hands) {
      h.geo.instanceCount = 0;
      h.line.visible = h.ink.visible = false;
    }
  }

  // (a seek needs no handling here: the shared ring refills itself on any
  // jump it cannot extend, and a new replay resets it)
  $effect(() => {
    const cur = pb.cur; // the continuous clock: the line is sub-tick
    const on = pb.tipTrail && !pose.on && !!pb.rep && pb.rep.ticks.length > 0;
    if (!on) { hide(); return; }
    bladeTrail.ensure(pb.rep, cur, trailCover(pb.tipTrail, pb.tipTrailS));
    const view = bladeTrail.view(pb.tipTrailS);
    for (let s = 0; s < 2; s++) {
      const h = hands[s];
      const n = view.writeSegments(TIP(s), h.pos.array, h.col.array, h.head, h.tail);
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
