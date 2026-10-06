<script>
  // One SABER (env v10): the arm segments belong to the articulated
  // skeleton (Body.svelte); this draws the watch viewer's Saber.fbx —
  // the 1 m blade (r 0.018) from the hand frame along the blade axis and
  // the 0.16 m handle behind the grip, exactly the placement the
  // ScoreSaber .dat export produces — with a white-hot core when the TIP
  // is contact-capable (>= 1 m/s via the finite-diff contact-velocity
  // semantics), plus the swept tip ribbon below.
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import { pb, pose } from "./state.svelte.js";
  import {
    SABER_DARK, SABER_LEN, V_MIN, baseOf, fkSaber, handleLen, tipOf, tipSpeed,
  } from "./replay.js";
  import { BASE, RIBBON_S, TIP, bladeTrail, trailCover } from "./bladetrail.js";
  import { NOTE_COLORS, modelGeo, noteColors } from "./models.js";
  import { OUTLINE_SABER, RAMP } from "./toon.js";

  let { s, fk } = $props();

  const Y = new THREE.Vector3(0, 1, 0);
  const toScene = (p) => new THREE.Vector3(p[0], p[1], -p[2]);

  const tickIdx = $derived(pb.rep
    ? Math.min(Math.floor(pb.cur), pb.rep.ticks.length - 1) : 0);
  const sab = $derived(fkSaber(fk, s)); // [w(3), blade(3)] from body FK
  // env v42 hilt grip: the blade base is `hl` in front of the hand along
  // the blade dir (0 on pre-v42 recordings, from the header config)
  const hl = $derived(handleLen(pb.rep));
  const base = $derived(baseOf(sab, hl)); // world coords
  const tip = $derived(tipOf(sab, hl)); // world coords
  const capable = $derived(
    !pose.on && !!pb.rep && tipSpeed(pb.rep, tickIdx, s) >= V_MIN);

  // y-aligned cylinder -> arbitrary segment transform
  function seg(a, b) {
    const av = toScene(a), bv = toScene(b);
    const dir = bv.clone().sub(av);
    const q = new THREE.Quaternion().setFromUnitVectors(
      Y, dir.clone().normalize());
    const mid = av.clone().add(bv).multiplyScalar(0.5);
    return { pos: [mid.x, mid.y, mid.z], quat: [q.x, q.y, q.z, q.w],
             len: dir.length() };
  }

  const blade = $derived(seg(base, tip));
  // the contact-capable core, built once: as `args` geometry it was rebuilt
  // (and uploaded) every time the blade came up to speed
  const coreGeo = new THREE.CylinderGeometry(0.008, 0.011, SABER_LEN * 1.001, 8);
  $effect(() => () => coreGeo.dispose());

  // the saber model: blade along scene -z from the grip, handle behind it.
  // Orientation = the rotation taking local -z onto the blade direction
  // (the model is rotationally symmetric about its axis, so the roll is
  // immaterial); position = the hand frame's origin (the grip).
  const bladeGeo = modelGeo("blade");
  const handleGeo = modelGeo("handle");
  const NEG_Z = new THREE.Vector3(0, 0, -1);
  const saberQuat = $derived.by(() => {
    const d = new THREE.Vector3(sab[3], sab[4], -sab[5]).normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(NEG_Z, d);
    return [q.x, q.y, q.z, q.w];
  });
  const k = noteColors(s);

  // ---- swept tip ribbon ---------------------------------------------------
  // The trail is the surface the blade swept, not a line through the tip: a
  // quad strip between the tip path and the path of the blade's base (the
  // grip — the blade runs 1 m forward from the hand frame), so the ribbon
  // is the whole plane the blade passed through. Cel-styled to match the
  // rest of the scene: flat saber colour in three HARD alpha steps rather
  // than a gradient, with a thin ink border.
  //
  // Its points are the shared blade history (lib/bladetrail.js): the DRAWN
  // pose — qAt's joint-space Hermite + bodyFK, the very pose this
  // component draws the blade from — sampled on a fixed 240 Hz replay-time
  // grid, with the newest point at pb.cur itself and the oldest exactly
  // RIBBON_S back. So the ribbon is welded to the drawn blade at every
  // instant (not only at ticks), its outer edge is exactly the tip trail's
  // line, and no smoothing is needed: the Hermite is already C1 across
  // ticks. Every quad owns its four vertices instead of sharing them with
  // its neighbours, so an alpha step is a step and not a ramp across the
  // shared edge.
  const STEPS = [0.42, 0.24, 0.1];   // head -> tail alpha, no interpolation
  const MAXQ = bladeTrail.trail.maxPts - 1; // quads (sized for any window)

  const ribGeo = new THREE.BufferGeometry();
  const ribPos = new Float32Array(MAXQ * 4 * 3);
  const ribCol = new Float32Array(MAXQ * 4 * 4); // RGBA: alpha per vertex
  const ribIdx = new Uint16Array(MAXQ * 6);
  for (let q = 0; q < MAXQ; q++) {
    const v = q * 4;
    ribIdx.set([v, v + 1, v + 2, v, v + 2, v + 3], q * 6);
  }
  ribGeo.setAttribute("position", new THREE.BufferAttribute(ribPos, 3));
  ribGeo.setAttribute("color", new THREE.BufferAttribute(ribCol, 4));
  ribGeo.setIndex(new THREE.BufferAttribute(ribIdx, 1));

  // ink: the ribbon's two long borders, one 2-vertex segment per quad
  const inkGeo = new THREE.BufferGeometry();
  const inkPos = new Float32Array(MAXQ * 4 * 3);
  const inkCol = new Float32Array(MAXQ * 4 * 4);
  inkGeo.setAttribute("position", new THREE.BufferAttribute(inkPos, 3));
  inkGeo.setAttribute("color", new THREE.BufferAttribute(inkCol, 4));
  $effect(() => () => { ribGeo.dispose(); inkGeo.dispose(); });

  const RIB_RGB = new THREE.Color(NOTE_COLORS[s]);
  const INK_RGB = new THREE.Color("#0a0c10");

  // dst vertex di <- the 3 floats at src[si ..]
  const copy3 = (dst, di, src, si) => {
    dst[di * 3] = src[si];
    dst[di * 3 + 1] = src[si + 1];
    dst[di * 3 + 2] = src[si + 2];
  };
  const paint = (dst, di, rgb, a) => {
    dst[di * 4] = rgb.r; dst[di * 4 + 1] = rgb.g;
    dst[di * 4 + 2] = rgb.b; dst[di * 4 + 3] = a;
  };

  $effect(() => {
    const cur = pb.cur; // read the continuous clock: the ribbon is sub-tick
    if (pose.on || !pb.rep || pb.rep.ticks.length === 0) { // no trail for a static pose
      ribGeo.setDrawRange(0, 0);
      inkGeo.setDrawRange(0, 0);
      return;
    }
    bladeTrail.ensure(pb.rep, cur, trailCover(pb.tipTrail, pb.tipTrailS));
    const view = bladeTrail.view(RIBBON_S);
    const P = view.pts, age = view.age, S = bladeTrail.trail.stride;
    const nq = Math.max(0, view.npts - 1);
    const T0 = TIP(s), B0 = BASE(s);
    for (let q = 0; q < nq; q++) {
      const a = q * S, b = (q + 1) * S, v = q * 4;
      copy3(ribPos, v, P, a + B0);
      copy3(ribPos, v + 1, P, a + T0);
      copy3(ribPos, v + 2, P, b + T0);
      copy3(ribPos, v + 3, P, b + B0);
      const u = 0.5 * (age[q] + age[q + 1]); // 0 at the tail, 1 at the blade
      const al = STEPS[u > 2 / 3 ? 0 : u > 1 / 3 ? 1 : 2];
      for (let j = 0; j < 4; j++) paint(ribCol, v + j, RIB_RGB, al);
      // borders: tip edge then base edge (LineSegments draws vertex pairs)
      copy3(inkPos, v, P, a + T0);
      copy3(inkPos, v + 1, P, b + T0);
      copy3(inkPos, v + 2, P, a + B0);
      copy3(inkPos, v + 3, P, b + B0);
      for (let j = 0; j < 4; j++) paint(inkCol, v + j, INK_RGB, al * 0.9);
    }
    ribGeo.setDrawRange(0, nq * 6);
    inkGeo.setDrawRange(0, nq * 4);
    for (const at of [ribGeo.attributes.position, ribGeo.attributes.color,
      inkGeo.attributes.position, inkGeo.attributes.color]) {
      at.clearUpdateRanges();
      at.addUpdateRange(0, nq * 4 * at.itemSize); // only the quads drawn
      at.needsUpdate = true;
    }
  });
</script>

<!-- the ribbon and its ink: unlit (a lit trail would band against the toon
     ramp and read as a solid object), never depth-writing, and double-sided
     because a swing shows the viewer both faces -->
<T.Mesh geometry={ribGeo} frustumCulled={false}>
  <T.MeshBasicMaterial
    vertexColors
    transparent
    depthWrite={false}
    side={THREE.DoubleSide}
  />
</T.Mesh>
<T.LineSegments geometry={inkGeo} frustumCulled={false}>
  <T.LineBasicMaterial vertexColors transparent depthWrite={false} />
</T.LineSegments>

<!-- the saber model: blade (SaberBlade.mat = saber colour, fresnel glow)
     + handle (SaberHandle.mat, dark). The model's blade starts at its
     origin with the handle behind it, so the group sits at the blade BASE:
     the grip for pre-v42 recordings, hl in front of the hand on v42 (the
     handle's bottom then lands on the hand, per the user's spec) -->
<T.Group position={[base[0], base[1], -base[2]]} quaternion={saberQuat}>
  <T.Mesh geometry={bladeGeo} material={OUTLINE_SABER} />
  <T.Mesh geometry={bladeGeo}>
    <T.MeshToonMaterial
      color={NOTE_COLORS[s]}
      gradientMap={RAMP}
      emissive={k.arrowGlow}
      emissiveIntensity={0.6}
    />
  </T.Mesh>
  <T.Mesh geometry={handleGeo} material={OUTLINE_SABER} />
  <T.Mesh geometry={handleGeo}>
    <T.MeshToonMaterial color={SABER_DARK[s]} gradientMap={RAMP} />
  </T.Mesh>
</T.Group>
{#if capable}
  <!-- contact-capable: white-hot core inside the blade (viewer cue) -->
  <T.Mesh geometry={coreGeo} position={blade.pos} quaternion={blade.quat}>
    <T.MeshBasicMaterial color="#e8eaf0" />
  </T.Mesh>
{/if}
