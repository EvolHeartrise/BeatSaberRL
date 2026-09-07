<script>
  // One SABER (env v10): the arm segments belong to the articulated
  // skeleton (Body.svelte); this draws the watch viewer's Saber.fbx —
  // the 1 m blade (r 0.018) from the hand frame along the blade axis and
  // the 0.16 m handle behind the grip, exactly the placement the
  // ScoreSaber .dat export produces — with a white-hot core when the TIP
  // is contact-capable (>= 1 m/s via the finite-diff contact-velocity
  // semantics), plus a 10-tick tip trail from the logged sabers
  // (preallocated buffer).
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import { pb, pose } from "./state.svelte.js";
  import {
    SABER_COLORS, SABER_DARK, SABER_LEN, V_MIN, fkSaber, tipOf, tipSpeed,
  } from "./replay.js";
  import { NOTE_COLORS, modelGeo, noteColors } from "./models.js";

  let { s, fk } = $props();

  const TRAIL = 10;
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

  // tip trail: fixed-size buffer updated in place (logged sabers)
  const trailGeo = new THREE.BufferGeometry();
  const trailPos = new Float32Array((TRAIL + 1) * 3);
  trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
  $effect(() => {
    if (pose.on || !pb.rep) { // no trail for a static pose
      trailGeo.setDrawRange(0, 0);
      trailGeo.attributes.position.needsUpdate = true;
      return;
    }
    const from = Math.max(0, tickIdx - TRAIL);
    let n = 0;
    for (let k = from; k <= tickIdx; k++) {
      const t3 = tipOf(pb.rep.ticks[k].sabers[s]);
      trailPos[n * 3] = t3[0];
      trailPos[n * 3 + 1] = t3[1];
      trailPos[n * 3 + 2] = -t3[2];
      n++;
    }
    trailGeo.setDrawRange(0, n);
    trailGeo.attributes.position.needsUpdate = true;
    trailGeo.computeBoundingSphere();
  });
</script>

<T.Line geometry={trailGeo}>
  <T.LineBasicMaterial color={SABER_COLORS[s]} transparent opacity={0.35} />
</T.Line>

<!-- the saber model at the grip: blade (SaberBlade.mat = saber colour,
     fresnel glow) + handle (SaberHandle.mat, dark) -->
<T.Group position={[sab[0], sab[1], -sab[2]]} quaternion={saberQuat}>
  <T.Mesh geometry={bladeGeo}>
    <T.MeshStandardMaterial
      color={NOTE_COLORS[s]}
      emissive={k.arrowGlow}
      emissiveIntensity={0.6}
      roughness={0.4}
    />
  </T.Mesh>
  <T.Mesh geometry={handleGeo}>
    <T.MeshStandardMaterial color={SABER_DARK[s]} roughness={0.5} metalness={0.4} />
  </T.Mesh>
</T.Group>
{#if capable}
  <!-- contact-capable: white-hot core inside the blade (viewer cue) -->
  <T.Mesh position={blade.pos} quaternion={blade.quat}>
    <T.CylinderGeometry args={[0.008, 0.011, SABER_LEN * 1.001, 8]} />
    <T.MeshBasicMaterial color="#e8eaf0" />
  </T.Mesh>
{/if}
