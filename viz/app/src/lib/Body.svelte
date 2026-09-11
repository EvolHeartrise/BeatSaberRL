<script>
  // The REAL articulated v10 skeleton: capsules/spheres/boxes at the
  // SPEC-v10 §1.2 collider dimensions, positioned by full-body FK
  // (replay q36 or the manual pose) — replaces the v7-v9 static
  // decorative torso. World -> scene: (x, y, -z).
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import {
    F_CHEST, F_HEAD, F_L_HAND, F_L_SHANK, F_L_THIGH, F_L_FOOT,
    F_L_ELBOW, F_L_SHOULDER, F_L_WRIST, F_R_ELBOW, F_R_SHOULDER,
    F_R_HAND, F_R_SHANK, F_R_THIGH, F_R_FOOT, F_R_WRIST,
    framePoint, rotQuatScene,
  } from "./replay.js";
  import { OUTLINE_BODY, RAMP } from "./toon.js";

  // showHead: off in first-person, where the camera sits at the head's
  // centre — inside the sphere and its inverted-hull outline
  let { fk, showHead = true } = $props();

  const Y = new THREE.Vector3(0, 1, 0);
  const toScene = (p) => new THREE.Vector3(p[0], p[1], -p[2]);
  const sp = (p) => [p[0], p[1], -p[2]]; // world point -> scene position
  const IDENT = [0, 0, 0, 1];

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


  // White body (2026-09-09): trunk/head/pelvis white, legs and feet a
  // half-step cooler so the limbs still read as separate volumes against
  // it. The arms carry a ~60% wash of the saber hues over white: clearly
  // red/blue at a glance, still light enough to read as part of the white
  // body. The sabers keep SABER_COLORS / SABER_DARK, so the blades stay
  // the most saturated red/blue on screen.
  const TRUNK = "#f4f6fa", LIMB = "#e3e7ef";
  const ARM = ["#eca7ac", "#a0cff5"]; // left/red-tinted, right/blue-tinted

  // Geometry built ONCE: FK is rigid, so every segment's length is the
  // model constant (chest 0.43, upper arm 0.35, forearm 0.30, thigh 0.44,
  // shank 0.43, pelvis 0.20) — feeding the per-frame measured length
  // through geometry `args` recreated (and never freed until unmount) ~10
  // CylinderGeometries every frame. The balls/boxes are hoisted for the
  // same reason and because the outline pass reuses each one.
  const cyl = (r, len) => new THREE.CylinderGeometry(r, r, len, 12);
  const chestGeo = cyl(0.15, 0.43);
  const uarmGeo = cyl(0.045, 0.35), farmGeo = cyl(0.04, 0.30);
  const thighGeo = cyl(0.06, 0.44), shankGeo = cyl(0.045, 0.43);
  const pelvisGeo = cyl(0.10, 0.20);
  const headGeo = new THREE.SphereGeometry(0.09, 18, 14);
  const fistGeo = new THREE.SphereGeometry(0.05, 14, 10);
  const shoulderGeo = new THREE.SphereGeometry(0.055, 14, 10);
  const elbowGeo = new THREE.SphereGeometry(0.042, 12, 8);
  const legJointGeo = new THREE.SphereGeometry(0.05, 12, 8);
  const footGeo = new THREE.BoxGeometry(0.09, 0.07, 0.25);

  const pelvis = $derived(seg(
    framePoint({ r: [fk.r0], p: [fk.p0] }, 0, [-0.10, 0, 0]),
    framePoint({ r: [fk.r0], p: [fk.p0] }, 0, [0.10, 0, 0])));
  const chest = $derived(seg(fk.p[F_CHEST],
    framePoint(fk, F_CHEST, [0, 0.43, 0])));

  // Every solid in one list: the cel pass draws each entry twice (ink
  // hull, then the banded surface), so the two copies cannot drift apart.
  const bone = (geo, s, color) =>
    ({ geo, pos: s.pos, quat: s.quat, color });
  const ball = (geo, p, color) =>
    ({ geo, pos: sp(p), quat: IDENT, color });

  const parts = $derived([
    bone(chestGeo, chest, TRUNK),
    // pelvis capsule (x-aligned in the root frame)
    bone(pelvisGeo, pelvis, TRUNK),
    // arm bones joint-to-joint: the shoulder/wrist clusters are stacked
    // co-located frames and F_*_SHOULDER is the FIRST (yaw) one — an
    // extrusion along its local z ignores pitch/twist and misses the
    // elbow whenever the arm is off neutral
    bone(uarmGeo, seg(fk.p[F_L_SHOULDER], fk.p[F_L_ELBOW]), ARM[0]),
    bone(farmGeo, seg(fk.p[F_L_ELBOW], fk.p[F_L_WRIST]), ARM[0]),
    bone(uarmGeo, seg(fk.p[F_R_SHOULDER], fk.p[F_R_ELBOW]), ARM[1]),
    bone(farmGeo, seg(fk.p[F_R_ELBOW], fk.p[F_R_WRIST]), ARM[1]),
    bone(thighGeo, seg(fk.p[F_L_THIGH],
      framePoint(fk, F_L_THIGH, [0, -0.44, 0])), LIMB),
    bone(shankGeo, seg(fk.p[F_L_SHANK],
      framePoint(fk, F_L_SHANK, [0, -0.43, 0])), LIMB),
    bone(thighGeo, seg(fk.p[F_R_THIGH],
      framePoint(fk, F_R_THIGH, [0, -0.44, 0])), LIMB),
    bone(shankGeo, seg(fk.p[F_R_SHANK],
      framePoint(fk, F_R_SHANK, [0, -0.43, 0])), LIMB),
    // head sphere (welded to the chest)
    ...(showHead ? [ball(headGeo, fk.p[F_HEAD], TRUNK)] : []),
    // fists
    ball(fistGeo, framePoint(fk, F_L_HAND, [0, 0, 0.06]), ARM[0]),
    ball(fistGeo, framePoint(fk, F_R_HAND, [0, 0, 0.06]), ARM[1]),
    // joint balls: shoulders, elbows, hips, knees
    ball(shoulderGeo, fk.p[F_L_SHOULDER], ARM[0]),
    ball(shoulderGeo, fk.p[F_R_SHOULDER], ARM[1]),
    ball(elbowGeo, fk.p[F_L_ELBOW], ARM[0]),
    ball(elbowGeo, fk.p[F_R_ELBOW], ARM[1]),
    ball(legJointGeo, fk.p[F_L_THIGH], LIMB),
    ball(legJointGeo, fk.p[F_R_THIGH], LIMB),
    ball(legJointGeo, fk.p[F_L_SHANK], LIMB),
    ball(legJointGeo, fk.p[F_R_SHANK], LIMB),
    // feet: boxes at the §1.2 dims, oriented by the foot frames (at the
    // weld anchors whenever the replayed sim holds them there)
    ...[F_L_FOOT, F_R_FOOT].map((f) => ({
      geo: footGeo,
      pos: sp(framePoint(fk, f, [0, -0.035, 0.05])),
      quat: rotQuatScene(fk.r[f]),
      color: LIMB,
    })),
  ]);
</script>

{#each parts as p, i (i)}
  <!-- ink hull first (back faces only — it never z-fights the lit copy),
       then the banded surface -->
  <T.Mesh geometry={p.geo} position={p.pos} quaternion={p.quat}
    material={OUTLINE_BODY} />
  <T.Mesh geometry={p.geo} position={p.pos} quaternion={p.quat}>
    <T.MeshToonMaterial color={p.color} gradientMap={RAMP} />
  </T.Mesh>
{/each}
