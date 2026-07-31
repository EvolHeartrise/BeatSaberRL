<script>
  // The REAL articulated v10 skeleton: capsules/spheres/boxes at the
  // SPEC-v10 §1.2 collider dimensions, positioned by full-body FK
  // (replay q36 or the manual pose) — replaces the v7-v9 static
  // decorative torso. World -> scene: (x, y, -z).
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import {
    F_CHEST, F_HEAD, F_L_HAND, F_L_SHANK, F_L_THIGH, F_L_FOOT,
    F_L_ELBOW, F_L_SHOULDER, F_R_ELBOW, F_R_SHOULDER,
    F_R_HAND, F_R_SHANK, F_R_THIGH, F_R_FOOT, SABER_DARK, framePoint,
  } from "./replay.js";

  let { fk } = $props();

  const Y = new THREE.Vector3(0, 1, 0);
  const toScene = (p) => new THREE.Vector3(p[0], p[1], -p[2]);

  // y-aligned cylinder -> arbitrary segment transform (world coords in)
  function seg(a, b) {
    const av = toScene(a), bv = toScene(b);
    const dir = bv.clone().sub(av);
    const q = new THREE.Quaternion().setFromUnitVectors(
      Y, dir.clone().normalize());
    const mid = av.clone().add(bv).multiplyScalar(0.5);
    return { pos: [mid.x, mid.y, mid.z], quat: [q.x, q.y, q.z, q.w],
             len: dir.length() };
  }

  // scene quaternion of a world frame rotation (conjugated by
  // S = diag(1,1,-1), same rule as frameQuatScene)
  function rotQuatScene(m) {
    const e = new THREE.Matrix4().set(
      m[0][0], m[0][1], -m[0][2], 0,
      m[1][0], m[1][1], -m[1][2], 0,
      -m[2][0], -m[2][1], m[2][2], 0,
      0, 0, 0, 1);
    const q = new THREE.Quaternion().setFromRotationMatrix(e);
    return [q.x, q.y, q.z, q.w];
  }

  const TRUNK = "#2a2f3b", LIMB = "#232733";

  // capsule segments: [frame, a_local, b_local, radius, color per side]
  const pelvis = $derived(seg(
    framePoint({ r: [fk.r0], p: [fk.p0] }, 0, [-0.10, 0, 0]),
    framePoint({ r: [fk.r0], p: [fk.p0] }, 0, [0.10, 0, 0])));
  const chest = $derived(seg(fk.p[F_CHEST],
    framePoint(fk, F_CHEST, [0, 0.43, 0])));
  const segs = $derived([
    // [seg, radius, color]
    [chest, 0.15, TRUNK],
    [seg(fk.p[F_L_SHOULDER], framePoint(fk, F_L_SHOULDER, [0, 0, 0.35])),
      0.045, SABER_DARK[0]],
    [seg(fk.p[F_L_ELBOW], framePoint(fk, F_L_ELBOW, [0, 0, 0.30])),
      0.04, SABER_DARK[0]],
    [seg(fk.p[F_R_SHOULDER], framePoint(fk, F_R_SHOULDER, [0, 0, 0.35])),
      0.045, SABER_DARK[1]],
    [seg(fk.p[F_R_ELBOW], framePoint(fk, F_R_ELBOW, [0, 0, 0.30])),
      0.04, SABER_DARK[1]],
    [seg(fk.p[F_L_THIGH], framePoint(fk, F_L_THIGH, [0, -0.44, 0])),
      0.06, LIMB],
    [seg(fk.p[F_L_SHANK], framePoint(fk, F_L_SHANK, [0, -0.43, 0])),
      0.045, LIMB],
    [seg(fk.p[F_R_THIGH], framePoint(fk, F_R_THIGH, [0, -0.44, 0])),
      0.06, LIMB],
    [seg(fk.p[F_R_SHANK], framePoint(fk, F_R_SHANK, [0, -0.43, 0])),
      0.045, LIMB],
  ]);
  const head = $derived(toScene(fk.p[F_HEAD]));
  const fists = $derived([0, 1].map((s) =>
    toScene(framePoint(fk, s === 0 ? F_L_HAND : F_R_HAND, [0, 0, 0.06]))));
  // feet: boxes at the §1.2 dims, oriented by the foot frames (at the
  // weld anchors whenever the replayed sim holds them there)
  const feet = $derived([F_L_FOOT, F_R_FOOT].map((f) => ({
    pos: toScene(framePoint(fk, f, [0, -0.035, 0.05])),
    quat: rotQuatScene(fk.r[f]),
  })));
</script>

{#each segs as s, i (i)}
  <T.Mesh position={s[0].pos} quaternion={s[0].quat}>
    <T.CylinderGeometry args={[s[1], s[1], Math.max(s[0].len, 1e-4), 12]} />
    <T.MeshStandardMaterial color={s[2]} />
  </T.Mesh>
{/each}

<!-- pelvis capsule (x-aligned in the root frame) -->
<T.Mesh position={pelvis.pos} quaternion={pelvis.quat}>
  <T.CylinderGeometry args={[0.10, 0.10, Math.max(pelvis.len, 1e-4), 12]} />
  <T.MeshStandardMaterial color={TRUNK} />
</T.Mesh>

<!-- head sphere (welded to the chest) -->
<T.Mesh position={[head.x, head.y, head.z]}>
  <T.SphereGeometry args={[0.09, 18, 14]} />
  <T.MeshStandardMaterial color={TRUNK} />
</T.Mesh>

<!-- fists -->
{#each fists as f, s (s)}
  <T.Mesh position={[f.x, f.y, f.z]}>
    <T.SphereGeometry args={[0.05, 14, 10]} />
    <T.MeshStandardMaterial color={SABER_DARK[s]} />
  </T.Mesh>
{/each}

<!-- joint balls: shoulders, elbows, hips, knees -->
{#each [fk.p[F_L_SHOULDER], fk.p[F_R_SHOULDER]] as p, i (i)}
  <T.Mesh position={[p[0], p[1], -p[2]]}>
    <T.SphereGeometry args={[0.055, 14, 10]} />
    <T.MeshStandardMaterial color={SABER_DARK[i]} />
  </T.Mesh>
{/each}
{#each [fk.p[F_L_ELBOW], fk.p[F_R_ELBOW]] as p, i (i)}
  <T.Mesh position={[p[0], p[1], -p[2]]}>
    <T.SphereGeometry args={[0.042, 12, 8]} />
    <T.MeshStandardMaterial color={SABER_DARK[i]} />
  </T.Mesh>
{/each}
{#each [fk.p[F_L_THIGH], fk.p[F_R_THIGH], fk.p[F_L_SHANK], fk.p[F_R_SHANK]] as p, i (i)}
  <T.Mesh position={[p[0], p[1], -p[2]]}>
    <T.SphereGeometry args={[0.05, 12, 8]} />
    <T.MeshStandardMaterial color={LIMB} />
  </T.Mesh>
{/each}

<!-- feet boxes on their weld anchors -->
{#each feet as f, i (i)}
  <T.Mesh position={[f.pos.x, f.pos.y, f.pos.z]} quaternion={f.quat}>
    <T.BoxGeometry args={[0.09, 0.07, 0.25]} />
    <T.MeshStandardMaterial color={LIMB} />
  </T.Mesh>
{/each}
