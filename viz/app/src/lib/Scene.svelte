<script>
  // The 3D scene: camera + orbit controls, lights, static floor/tunnel/grid/
  // reach geometry, playback clock, notes and sabers. World -> scene: (x, y, -z).
  // Env v7: floor-based world — the floor plane is at y = 0; the 4x3 cell
  // grid uses the real 0.6 m lanes and non-uniform row heights.
  import { T, useTask } from "@threlte/core";
  import { OrbitControls } from "@threlte/extras";
  import * as THREE from "three";
  import { pb, pose, lastTick, setPlaying } from "./state.svelte.js";
  import {
    BLOCK, GX0, GX1, GY0, GY1, HB, LANE_X, CELL, REACH_Z, ROW_EDGES, ROW_Y,
    SHOULDER, Z_FAR, Z_MISS, frameQuatScene, noteFrame,
  } from "./replay.js";
  import Notes from "./Notes.svelte";
  import Saber from "./Saber.svelte";
  import WristGauges from "./WristGauges.svelte";

  // ---- playback clock (30 ticks/s of replay time) --------------------------
  useTask((delta) => {
    if (!pb.playing || !pb.rep) return;
    pb.cur += delta * 30 * pb.speed;
    if (pb.cur >= lastTick()) {
      pb.cur = lastTick();
      setPlaying(false);
    }
  });

  // ---- static scenery, built once ------------------------------------------
  function segs(pairs) {
    const g = new THREE.BufferGeometry();
    g.setFromPoints(pairs.flat().map(([x, y, z]) => new THREE.Vector3(x, y, -z)));
    return g;
  }
  const rect = (z) => [
    [[GX0, GY0, z], [GX1, GY0, z]], [[GX1, GY0, z], [GX1, GY1, z]],
    [[GX1, GY1, z], [GX0, GY1, z]], [[GX0, GY1, z], [GX0, GY0, z]],
  ];

  const tunnelPairs = [];
  for (const [x, y] of [[GX0, GY0], [GX1, GY0], [GX1, GY1], [GX0, GY1]])
    tunnelPairs.push([[x, y, 0], [x, y, Z_FAR]]);
  for (let z = 2; z <= Z_FAR; z += 2) tunnelPairs.push(...rect(z));
  for (let c = 0; c <= 4; c++)
    tunnelPairs.push([[GX0 + c * CELL, GY0, 0], [GX0 + c * CELL, GY0, Z_FAR]]);
  // floor edge lines along the tunnel (the runway on the y = 0 floor)
  for (const x of [GX0, GX1])
    tunnelPairs.push([[x, 0, -Z_MISS], [x, 0, Z_FAR]]);
  const tunnelGeo = segs(tunnelPairs);

  // 4x3 cell grid at the hit plane: lane edges every 0.6 m; row edges
  // midway between the real non-uniform row centers
  const gridPairs = [];
  for (let c = 0; c <= 4; c++)
    gridPairs.push([[GX0 + c * CELL, GY0, 0], [GX0 + c * CELL, GY1, 0]]);
  for (const y of ROW_EDGES) gridPairs.push([[GX0, y, 0], [GX1, y, 0]]);
  const gridGeo = segs(gridPairs);

  // blade-reach region: miss plane (z = -0.65, the head plane since
  // env v7.3) to max reach (z = 1.75)
  const reachGeo = segs([...rect(REACH_Z), ...rect(-Z_MISS)]);

  // torso (env v7 placement): shoulders at (±0.2, 1.47, -0.65); the trunk
  // reaches down toward the floor to suggest a 1.8 m player
  const SH_Y = SHOULDER[1][1], SH_Z = -SHOULDER[1][2]; // scene z = +0.65

  // pose-mode reference note: a static cube at a chosen cell, z = 0.
  // A canonical down-swing note (roll 0); at z = 0 the v7.6 look-at
  // frame is FULLY billboarded (lam = 1) — a nice way to eyeball the
  // mechanic — so cube and hitboxes ride noteFrame(p, 0, 1). Pass
  // hitboxRot as psi if this reference ever grows an angle control.
  const refEdgesGeo = new THREE.EdgesGeometry(
    new THREE.BoxGeometry(BLOCK, BLOCK, BLOCK));
  const refGoodHbGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(
    2 * HB.goodHalf[0], 2 * HB.goodHalf[1], 2 * HB.goodHalf[2]));
  const refBadHbGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(
    2 * HB.badHalf[0], 2 * HB.badHalf[1], 2 * HB.badHalf[2]));
</script>

<T.PerspectiveCamera makeDefault position={[0, 2.1, 3.4]} fov={55}>
  <OrbitControls
    enableDamping
    dampingFactor={0.12}
    target={[0, 1.4, -1.5]}
    maxDistance={25}
  />
  {#if (pb.rep || pose.on) && pb.gauges}
    <WristGauges />
  {/if}
</T.PerspectiveCamera>

<T.AmbientLight intensity={1.1} />
<T.DirectionalLight position={[3, 6, 5]} intensity={1.4} />

<!-- the floor: y = 0 plane under the whole play area -->
<T.Mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, (Z_MISS - Z_FAR) / 2]}>
  <T.PlaneGeometry args={[2 * (GX1 + 0.8), Z_FAR + Z_MISS]} />
  <T.MeshStandardMaterial color="#12151c" />
</T.Mesh>

<T.LineSegments geometry={tunnelGeo}>
  <T.LineBasicMaterial color="#1d212b" />
</T.LineSegments>
<T.LineSegments geometry={reachGeo}>
  <T.LineBasicMaterial color="#2a3040" />
</T.LineSegments>
<T.LineSegments geometry={gridGeo}>
  <T.LineBasicMaterial color="#39404f" />
</T.LineSegments>

{#if pb.rep}
  <Notes />
{/if}

{#if pose.on && pose.ref}
  <!-- static reference note at a chosen cell, z = 0, for eyeballing reach -->
  {@const rx = LANE_X[pose.ref.col]}
  {@const ry = ROW_Y[pose.ref.row]}
  {@const refQ = frameQuatScene(noteFrame([rx, ry, 0], 0, 1))}
  <T.Group position={[rx, ry, 0]} quaternion={refQ}>
    <T.Mesh>
      <T.BoxGeometry args={[BLOCK, BLOCK, BLOCK]} />
      <T.MeshStandardMaterial color="#3a4050" transparent opacity={0.75} />
    </T.Mesh>
    <T.LineSegments geometry={refEdgesGeo}>
      <T.LineBasicMaterial color="#e8eaf0" />
    </T.LineSegments>
    {#if pb.hitboxes}
      <T.LineSegments geometry={refGoodHbGeo} position={[0, 0, -HB.goodZShift]}>
        <T.LineBasicMaterial color="#7dd3a0" transparent opacity={0.5} />
      </T.LineSegments>
      <T.LineSegments geometry={refBadHbGeo}>
        <T.LineBasicMaterial color="#f0a35e" transparent opacity={0.5} />
      </T.LineSegments>
    {/if}
  </T.Group>
{/if}

{#if pb.rep || pose.on}
  <!-- body: head + shoulder crossbar + trunk down toward the floor -->
  <T.Mesh position={[0, SH_Y + 0.24, SH_Z]}>
    <T.SphereGeometry args={[0.11, 16, 12]} />
    <T.MeshStandardMaterial color="#2a2f3b" />
  </T.Mesh>
  <T.Mesh position={[0, SH_Y, SH_Z]} rotation={[0, 0, Math.PI / 2]}>
    <T.CylinderGeometry args={[0.045, 0.045, 0.4, 10]} />
    <T.MeshStandardMaterial color="#2a2f3b" />
  </T.Mesh>
  <T.Mesh position={[0, 1.12, SH_Z]}>
    <T.BoxGeometry args={[0.34, 0.62, 0.16]} />
    <T.MeshStandardMaterial color="#20242e" />
  </T.Mesh>
  <!-- legs hint: trunk continues to the floor -->
  <T.Mesh position={[0, 0.41, SH_Z]}>
    <T.BoxGeometry args={[0.24, 0.82, 0.13]} />
    <T.MeshStandardMaterial color="#1a1e26" />
  </T.Mesh>
  <Saber s={0} />
  <Saber s={1} />
{/if}
