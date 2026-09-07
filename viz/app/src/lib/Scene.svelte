<script>
  // The 3D scene: camera + orbit controls, lights, static floor/tunnel/grid/
  // reach geometry, playback clock, notes, the ARTICULATED v10 skeleton +
  // blades, and self-collision contact markers. World -> scene: (x, y, -z).
  // Env v7 world geometry: floor plane at y = 0; the 4x3 cell grid uses the
  // real 0.6 m lanes and non-uniform row heights.
  import { T, useTask } from "@threlte/core";
  import { OrbitControls } from "@threlte/extras";
  import * as THREE from "three";
  import { pb, pose, lastTick, setPlaying } from "./state.svelte.js";
  import {
    BLOCK, DT, GX0, GX1, GY0, GY1, HB, LANE_X, CELL, REACH_Z, ROW_EDGES,
    ROW_Y, Z_FAR, Z_MISS, activeContacts, bodyFK, frameQuatScene,
    noteFrame, poseQ36, qAt, viewerTrackOffset,
  } from "./replay.js";
  import Body from "./Body.svelte";
  import Notes from "./Notes.svelte";
  import Saber from "./Saber.svelte";
  import BodyGauges from "./BodyGauges.svelte";
  import { modelGeo } from "./models.js";

  // ---- playback clock (30 ticks/s of replay time) --------------------------
  useTask((delta) => {
    if (!pb.playing || !pb.rep) return;
    pb.cur += delta * 30 * pb.speed;
    if (pb.cur >= lastTick()) {
      pb.cur = lastTick();
      setPlaying(false);
    }
  });

  // ---- the body pose this frame (replay q36 interpolated, or manual) ------
  const tickIdx = $derived(pb.rep
    ? Math.min(Math.floor(pb.cur), pb.rep.ticks.length - 1) : 0);
  const u = $derived(pb.rep ? pb.cur - tickIdx : 0);
  const q36 = $derived(pose.on
    ? poseQ36(pose)
    : (pb.rep ? qAt(pb.rep, tickIdx, u) : null));
  const fkAll = $derived(q36 ? bodyFK(q36) : null);

  // ---- env v15 head-anchored note track (contact spec §10) -----------------
  // z_off = head_z - PLANE_Z0 of the body being DRAWN this frame, so the
  // notes can never drift from the sabers. Taking it here (from the FK
  // that is already computed) also keeps the notes and the track scenery
  // in lockstep. viewerTrackOffset holds the strict header gate.
  const zOff = $derived(viewerTrackOffset(pb.rep, pose.on, fkAll));

  // self-collision contact markers (replay contact events, flash 0.4 s)
  const contacts = $derived(
    !pose.on && pb.rep ? activeContacts(pb.rep, pb.cur * DT) : []);

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

  // blade-reach region: reference-pose miss plane (z = -0.65) to
  // reference max reach (z = 1.75) — the sim's live gates move with the
  // pose; these lines are the static scenery datum
  const reachGeo = segs([...rect(REACH_Z), ...rect(-Z_MISS)]);

  // contact markers: one unit sphere scaled per marker (the radius varies
  // per contact, and radius-in-`args` geometry would be recreated — and
  // held undisposed — every frame the flash is alive)
  const contactGeo = new THREE.SphereGeometry(1, 12, 8);

  // pose-mode reference note: the real note model (neutral gray) at a
  // chosen cell, z = 0, with its bounding-box outline.
  const refNoteGeo = modelGeo("note");
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
    <BodyGauges />
  {/if}
</T.PerspectiveCamera>

<T.AmbientLight intensity={1.1} />
<T.DirectionalLight position={[3, 6, 5]} intensity={1.4} />

<!-- the floor: y = 0 plane under the whole play area -->
<T.Mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, (Z_MISS - Z_FAR) / 2]}>
  <T.PlaneGeometry args={[2 * (GX1 + 0.8), Z_FAR + Z_MISS]} />
  <T.MeshStandardMaterial color="#12151c" />
</T.Mesh>

<!-- The TRACK: tunnel, hit-plane grid and the reach/miss datum planes all
     belong to the note track, so under env v15 they ride the same
     head-anchored offset the notes do (world +zOff -> scene -zOff). That
     keeps the hit plane under the notes at their hit time, and puts the
     miss plane back exactly on the sim's boundary, world z = head_z
     (= -Z_MISS + zOff). The floor above is real ground and stays put. -->
<T.Group position={[0, 0, -zOff]}>
  <T.LineSegments geometry={tunnelGeo}>
    <T.LineBasicMaterial color="#1d212b" />
  </T.LineSegments>
  <T.LineSegments geometry={reachGeo}>
    <T.LineBasicMaterial color="#2a3040" />
  </T.LineSegments>
  <T.LineSegments geometry={gridGeo}>
    <T.LineBasicMaterial color="#39404f" />
  </T.LineSegments>
</T.Group>

{#if pb.rep}
  <Notes {zOff} />
{/if}

{#if pose.on && pose.ref}
  <!-- Reference note at a chosen cell, for eyeballing reach. It sits at
       the hit plane, which under v15 is world z = zOff: leaning in moves
       the note away by the head's own z travel, so the panel can no
       longer be read as "lean forward, gain reach". -->
  {@const rx = LANE_X[pose.ref.col]}
  {@const ry = ROW_Y[pose.ref.row]}
  {@const refQ = frameQuatScene(noteFrame([rx, ry, zOff], 0, 1))}
  <T.Group position={[rx, ry, -zOff]} quaternion={refQ}>
    <T.Mesh geometry={refNoteGeo}>
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

{#if fkAll}
  <!-- the articulated v10 body + blades -->
  <Body fk={fkAll} />
  <Saber s={0} fk={fkAll} />
  <Saber s={1} fk={fkAll} />
{/if}

<!-- self-collision contact markers: flash a sphere at the contact
     point, sized by the tick-summed impulse -->
{#each contacts as c, i (i)}
  <T.Mesh geometry={contactGeo} position={[c.p[0], c.p[1], -c.p[2]]}
    scale={c.radius}>
    <T.MeshBasicMaterial
      color="#f0d35e"
      transparent
      opacity={0.85 * c.alpha}
      depthTest={false}
    />
  </T.Mesh>
{/each}
