<script>
  // One saber: blade cylinder wrist->tip (white-hot core when the TIP is
  // contact-capable, >= 1 m/s), wrist disc, wrist-velocity vector, and a
  // 10-tick tip trail (preallocated buffer — no per-frame GPU churn).
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import { pb } from "./state.svelte.js";
  import {
    SABER_COLORS, SABER_LEN, V_MIN, saberAt, tipOf, tipSpeed,
  } from "./replay.js";

  let { s } = $props();

  const TRAIL = 10;
  const Y = new THREE.Vector3(0, 1, 0);

  const tickIdx = $derived(
    Math.min(Math.floor(pb.cur), pb.rep.ticks.length - 1));
  const u = $derived(pb.cur - tickIdx);
  const sab = $derived(saberAt(pb.rep, tickIdx, u, s));
  const tip = $derived(tipOf(sab)); // world coords
  const capable = $derived(tipSpeed(sab) >= V_MIN);

  // blade transform: cylinder is y-aligned; rotate y onto the segment
  const blade = $derived.by(() => {
    const w = new THREE.Vector3(sab[0], sab[1], 0);
    const t = new THREE.Vector3(tip[0], tip[1], -tip[2]);
    const dir = t.clone().sub(w);
    const q = new THREE.Quaternion().setFromUnitVectors(
      Y, dir.clone().normalize());
    const mid = w.clone().add(t).multiplyScalar(0.5);
    return { pos: [mid.x, mid.y, mid.z], quat: [q.x, q.y, q.z, q.w] };
  });

  // tip trail + wrist velocity vector: fixed-size buffers updated in place
  const trailGeo = new THREE.BufferGeometry();
  const trailPos = new Float32Array((TRAIL + 1) * 3);
  trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
  const velGeo = new THREE.BufferGeometry();
  const velPos = new Float32Array(2 * 3);
  velGeo.setAttribute("position", new THREE.BufferAttribute(velPos, 3));
  $effect(() => {
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

    velPos.set([sab[0], sab[1], 0,
                sab[0] + sab[2] * 0.12, sab[1] + sab[3] * 0.12, 0]);
    velGeo.attributes.position.needsUpdate = true;
    velGeo.computeBoundingSphere();
  });
</script>

<T.Line geometry={trailGeo}>
  <T.LineBasicMaterial color={SABER_COLORS[s]} transparent opacity={0.35} />
</T.Line>

<T.Mesh position={blade.pos} quaternion={blade.quat}>
  <T.CylinderGeometry args={[0.02, 0.03, SABER_LEN, 12]} />
  <T.MeshBasicMaterial color={SABER_COLORS[s]} />
</T.Mesh>
{#if capable}
  <T.Mesh position={blade.pos} quaternion={blade.quat}>
    <T.CylinderGeometry args={[0.009, 0.013, SABER_LEN * 1.001, 8]} />
    <T.MeshBasicMaterial color="#e8eaf0" />
  </T.Mesh>
{/if}

<T.Mesh position={[sab[0], sab[1], 0]}>
  <T.SphereGeometry args={[0.05, 16, 12]} />
  <T.MeshStandardMaterial color={SABER_COLORS[s]} />
</T.Mesh>

<!-- wrist velocity vector (translation component only) -->
<T.Line geometry={velGeo}>
  <T.LineBasicMaterial color={SABER_COLORS[s]} />
</T.Line>
