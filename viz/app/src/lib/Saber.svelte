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
    DT, SABER_DARK, SABER_LEN, V_MIN, fkSaber, saberAt, tipOf, tipSpeed,
  } from "./replay.js";
  import { NOTE_COLORS, modelGeo, noteColors } from "./models.js";
  import { OUTLINE_SABER, RAMP } from "./toon.js";

  let { s, fk } = $props();

  const Y = new THREE.Vector3(0, 1, 0);
  const toScene = (p) => new THREE.Vector3(p[0], p[1], -p[2]);

  const tickIdx = $derived(pb.rep
    ? Math.min(Math.floor(pb.cur), pb.rep.ticks.length - 1) : 0);
  const sab = $derived(fkSaber(fk, s)); // [w(3), blade(3)] from body FK
  const tip = $derived(tipOf(sab)); // world coords
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

  const blade = $derived(seg([sab[0], sab[1], sab[2]], tip));

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
  // Two details make the steps hard and the head solid:
  //   - every quad owns its four vertices instead of sharing them with its
  //     neighbours, so an alpha step is a step and not a ramp across the
  //     shared edge;
  //   - the newest sample is taken at pb.cur itself, not at the last whole
  //     tick, so the ribbon stays welded to the blade during playback (the
  //     old line lagged it by up to a tick).
  const TRAIL_S = 0.1;               // history window, seconds
  const TRAIL_TICKS = TRAIL_S / DT;  // 3 logged saber samples at 30 Hz
  // Resample at a fixed rate per logged tick rather than a fixed quad
  // count: the smoothing below is a filter over SAMPLES, so pinning the
  // sample density is what keeps its strength unchanged if the window
  // length moves.
  const PER_TICK = 3;
  const SEGS = Math.round(TRAIL_TICKS * PER_TICK); // quads along the ribbon
  const STEPS = [0.42, 0.24, 0.1];   // head -> tail alpha, no interpolation
  // The logged sabers are 30 Hz and saberAt is linear between them, so a
  // fast swing samples as a chain of chords no matter how finely it is
  // resampled. Two 3-tap passes over the interior round those corners; the
  // head and tail samples are left untouched, which keeps the ribbon
  // attached to the blade and its length constant.
  const SMOOTH = 2;

  const NS = SEGS + 1;
  const tipPath = new Float32Array(NS * 3); // scene-space
  const innPath = new Float32Array(NS * 3);
  const scratch = new Float32Array(NS * 3);

  const ribGeo = new THREE.BufferGeometry();
  const ribPos = new Float32Array(SEGS * 4 * 3);
  const ribCol = new Float32Array(SEGS * 4 * 4); // RGBA: alpha per vertex
  const ribIdx = new Uint16Array(SEGS * 6);
  for (let q = 0; q < SEGS; q++) {
    const v = q * 4;
    ribIdx.set([v, v + 1, v + 2, v, v + 2, v + 3], q * 6);
  }
  ribGeo.setAttribute("position", new THREE.BufferAttribute(ribPos, 3));
  ribGeo.setAttribute("color", new THREE.BufferAttribute(ribCol, 4));
  ribGeo.setIndex(new THREE.BufferAttribute(ribIdx, 1));

  // ink: the ribbon's two long borders, one 2-vertex segment per quad
  const inkGeo = new THREE.BufferGeometry();
  const inkPos = new Float32Array(SEGS * 4 * 3);
  const inkCol = new Float32Array(SEGS * 4 * 4);
  inkGeo.setAttribute("position", new THREE.BufferAttribute(inkPos, 3));
  inkGeo.setAttribute("color", new THREE.BufferAttribute(inkCol, 4));

  const RIB_RGB = new THREE.Color(NOTE_COLORS[s]);
  const INK_RGB = new THREE.Color("#0a0c10");

  const copy3 = (dst, di, src, si) => {
    dst[di * 3] = src[si * 3];
    dst[di * 3 + 1] = src[si * 3 + 1];
    dst[di * 3 + 2] = src[si * 3 + 2];
  };
  const paint = (dst, di, rgb, a) => {
    dst[di * 4] = rgb.r; dst[di * 4 + 1] = rgb.g;
    dst[di * 4 + 2] = rgb.b; dst[di * 4 + 3] = a;
  };
  function smoothPath(path) {
    for (let pass = 0; pass < SMOOTH; pass++) {
      scratch.set(path);
      for (let i = 1; i < NS - 1; i++)
        for (let c = 0; c < 3; c++)
          path[i * 3 + c] = 0.25 * scratch[(i - 1) * 3 + c]
            + 0.5 * scratch[i * 3 + c] + 0.25 * scratch[(i + 1) * 3 + c];
    }
  }

  $effect(() => {
    const cur = pb.cur; // read the continuous clock: the ribbon is sub-tick
    if (pose.on || !pb.rep) { // no trail for a static pose
      ribGeo.setDrawRange(0, 0);
      inkGeo.setDrawRange(0, 0);
      return;
    }
    const t0 = Math.max(0, cur - TRAIL_TICKS); // clamped at the replay start
    for (let i = 0; i < NS; i++) {
      const t = t0 + (cur - t0) * (i / SEGS);
      const k = Math.min(Math.floor(t), pb.rep.ticks.length - 1);
      const sa = saberAt(pb.rep, k, t - k, s);
      const tp = tipOf(sa);
      tipPath[i * 3] = tp[0];
      tipPath[i * 3 + 1] = tp[1];
      tipPath[i * 3 + 2] = -tp[2];
      innPath[i * 3] = sa[0]; // blade base = the grip / hand frame origin
      innPath[i * 3 + 1] = sa[1];
      innPath[i * 3 + 2] = -sa[2];
    }
    smoothPath(tipPath);
    smoothPath(innPath);

    for (let q = 0; q < SEGS; q++) {
      const a = q, b = q + 1, v = q * 4, w = q * 4;
      copy3(ribPos, v, innPath, a);
      copy3(ribPos, v + 1, tipPath, a);
      copy3(ribPos, v + 2, tipPath, b);
      copy3(ribPos, v + 3, innPath, b);
      const u = (q + 0.5) / SEGS; // 0 at the tail, 1 at the blade
      const al = STEPS[u > 2 / 3 ? 0 : u > 1 / 3 ? 1 : 2];
      for (let j = 0; j < 4; j++) paint(ribCol, v + j, RIB_RGB, al);
      // borders: tip edge then base edge (LineSegments draws vertex pairs)
      copy3(inkPos, w, tipPath, a);
      copy3(inkPos, w + 1, tipPath, b);
      copy3(inkPos, w + 2, innPath, a);
      copy3(inkPos, w + 3, innPath, b);
      for (let j = 0; j < 4; j++) paint(inkCol, w + j, INK_RGB, al * 0.9);
    }
    ribGeo.setDrawRange(0, SEGS * 6);
    inkGeo.setDrawRange(0, SEGS * 4);
    ribGeo.attributes.position.needsUpdate = true;
    ribGeo.attributes.color.needsUpdate = true;
    inkGeo.attributes.position.needsUpdate = true;
    inkGeo.attributes.color.needsUpdate = true;
    ribGeo.computeBoundingSphere();
    inkGeo.computeBoundingSphere();
  });
</script>

<!-- the ribbon and its ink: unlit (a lit trail would band against the toon
     ramp and read as a solid object), never depth-writing, and double-sided
     because a swing shows the viewer both faces -->
<T.Mesh geometry={ribGeo}>
  <T.MeshBasicMaterial
    vertexColors
    transparent
    depthWrite={false}
    side={THREE.DoubleSide}
  />
</T.Mesh>
<T.LineSegments geometry={inkGeo}>
  <T.LineBasicMaterial vertexColors transparent depthWrite={false} />
</T.LineSegments>

<!-- the saber model at the grip: blade (SaberBlade.mat = saber colour,
     fresnel glow) + handle (SaberHandle.mat, dark) -->
<T.Group position={[sab[0], sab[1], -sab[2]]} quaternion={saberQuat}>
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
  <T.Mesh position={blade.pos} quaternion={blade.quat}>
    <T.CylinderGeometry args={[0.008, 0.011, SABER_LEN * 1.001, 8]} />
    <T.MeshBasicMaterial color="#e8eaf0" />
  </T.Mesh>
{/if}
