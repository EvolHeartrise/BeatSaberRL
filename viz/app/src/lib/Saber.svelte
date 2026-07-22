<script>
  // One ARM (env v6): shoulder anchor, upper-arm and forearm segments,
  // elbow + wrist joints, and the blade cylinder wrist->tip (white-hot
  // core when the TIP is contact-capable, >= 1 m/s via the finite-diff
  // contact-velocity semantics), plus a 10-tick tip trail (preallocated
  // buffer — no per-frame GPU churn).
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import { pb, pose } from "./state.svelte.js";
  import {
    SABER_COLORS, SABER_DARK, SABER_LEN, SHOULDER, V_MIN, armFK, poseSaber,
    saberAt, tipOf, tipSpeed,
  } from "./replay.js";

  let { s } = $props();

  const TRAIL = 10;
  const Y = new THREE.Vector3(0, 1, 0);
  const toScene = (p) => new THREE.Vector3(p[0], p[1], -p[2]);

  // pose mode takes over the arm (static q, omegas 0); it may run with
  // no replay loaded at all
  const tickIdx = $derived(pb.rep
    ? Math.min(Math.floor(pb.cur), pb.rep.ticks.length - 1) : 0);
  const u = $derived(pb.cur - tickIdx);
  const sab = $derived(pose.on
    ? poseSaber(s, pose.q[s]) : saberAt(pb.rep, tickIdx, u, s));
  const fkNow = $derived(armFK(s, sab));
  const tip = $derived(tipOf(s, sab)); // world coords
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

  const upper = $derived(seg(SHOULDER[s], fkNow.elbow));
  const fore = $derived(seg(fkNow.elbow, fkNow.wrist));
  const blade = $derived(seg(fkNow.wrist, tip));

  // tip trail: fixed-size buffer updated in place
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
      const t3 = tipOf(s, pb.rep.ticks[k].sabers[s]);
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

<!-- shoulder anchor -->
<T.Mesh position={[SHOULDER[s][0], SHOULDER[s][1], -SHOULDER[s][2]]}>
  <T.SphereGeometry args={[0.055, 16, 12]} />
  <T.MeshStandardMaterial color={SABER_DARK[s]} />
</T.Mesh>

<!-- upper arm + forearm -->
<T.Mesh position={upper.pos} quaternion={upper.quat}>
  <T.CylinderGeometry args={[0.032, 0.04, upper.len, 10]} />
  <T.MeshStandardMaterial color={SABER_DARK[s]} />
</T.Mesh>
<T.Mesh position={fore.pos} quaternion={fore.quat}>
  <T.CylinderGeometry args={[0.026, 0.032, fore.len, 10]} />
  <T.MeshStandardMaterial color={SABER_DARK[s]} />
</T.Mesh>
<T.Mesh position={[fkNow.elbow[0], fkNow.elbow[1], -fkNow.elbow[2]]}>
  <T.SphereGeometry args={[0.04, 14, 10]} />
  <T.MeshStandardMaterial color={SABER_DARK[s]} />
</T.Mesh>

<!-- blade -->
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

<!-- wrist -->
<T.Mesh position={[fkNow.wrist[0], fkNow.wrist[1], -fkNow.wrist[2]]}>
  <T.SphereGeometry args={[0.05, 16, 12]} />
  <T.MeshStandardMaterial color={SABER_COLORS[s]} />
</T.Mesh>
