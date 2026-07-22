<script>
  // The 3D scene: camera + orbit controls, lights, static tunnel/grid/reach
  // geometry, playback clock, notes and sabers. World -> scene: (x, y, -z).
  import { T, useTask } from "@threlte/core";
  import { OrbitControls } from "@threlte/extras";
  import * as THREE from "three";
  import { pb, lastTick, setPlaying } from "./state.svelte.js";
  import { CELL, GX, GY, REACH_Z, Z_FAR, Z_MISS } from "./replay.js";
  import Notes from "./Notes.svelte";
  import Saber from "./Saber.svelte";

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
    [[-GX, -GY, z], [GX, -GY, z]], [[GX, -GY, z], [GX, GY, z]],
    [[GX, GY, z], [-GX, GY, z]], [[-GX, GY, z], [-GX, -GY, z]],
  ];

  const tunnelPairs = [];
  for (const [x, y] of [[-GX, -GY], [GX, -GY], [GX, GY], [-GX, GY]])
    tunnelPairs.push([[x, y, 0], [x, y, Z_FAR]]);
  for (let z = 2; z <= Z_FAR; z += 2) tunnelPairs.push(...rect(z));
  for (let c = 0; c <= 4; c++)
    tunnelPairs.push([[-GX + c * CELL, -GY, 0], [-GX + c * CELL, -GY, Z_FAR]]);
  const tunnelGeo = segs(tunnelPairs);

  const gridPairs = [];
  for (let c = 0; c <= 4; c++)
    gridPairs.push([[-GX + c * CELL, -GY, 0], [-GX + c * CELL, GY, 0]]);
  for (let r = 0; r <= 3; r++)
    gridPairs.push([[-GX, -GY + r * CELL, 0], [GX, -GY + r * CELL, 0]]);
  const gridGeo = segs(gridPairs);

  // blade-reach region: miss line (z = -0.6) to max tilted reach (z = 1.28)
  const reachGeo = segs([...rect(REACH_Z), ...rect(-Z_MISS)]);
</script>

<T.PerspectiveCamera makeDefault position={[0, 1.1, 3.2]} fov={55}>
  <OrbitControls
    enableDamping
    dampingFactor={0.12}
    target={[0, 0, -1.5]}
    maxDistance={25}
  />
</T.PerspectiveCamera>

<T.AmbientLight intensity={1.1} />
<T.DirectionalLight position={[3, 6, 5]} intensity={1.4} />

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
  <Saber s={0} />
  <Saber s={1} />
{/if}
