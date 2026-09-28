<script>
  // Flux game mode (SPEC.md §Flux): the hands carry no sabers. Each hand
  // is its PALM — the point PALM_OFF along the hand axis from the wrist,
  // the same point the sim tests circles against — drawn as a glowing
  // orb in the hand's colour, and the FLUX is the segment between the
  // two palms, drawn as a luminous beam. Positions come from the body
  // FK being drawn this frame (like the sabers), so the beam never lags
  // the skeleton — or, for a person playing the WebXR game (no body),
  // straight from `palms`: the tracked palms, live or from the recording.
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import { fkSaber, palmOf } from "./replay.js";
  import { NOTE_COLORS } from "./models.js";

  let { fk = null, palms: given = null } = $props();

  const Y = new THREE.Vector3(0, 1, 0);
  const toScene = (p) => new THREE.Vector3(p[0], p[1], -p[2]);

  const palms = $derived(given ?? [0, 1].map((h) => palmOf(fkSaber(fk, h))));

  // y-aligned unit cylinder -> the palm-to-palm segment
  const beam = $derived.by(() => {
    const a = toScene(palms[0]), b = toScene(palms[1]);
    const dir = b.clone().sub(a);
    const len = dir.length();
    const q = len > 1e-6
      ? new THREE.Quaternion().setFromUnitVectors(Y, dir.clone().normalize())
      : new THREE.Quaternion();
    const mid = a.clone().add(b).multiplyScalar(0.5);
    return { pos: [mid.x, mid.y, mid.z], quat: [q.x, q.y, q.z, q.w], len };
  });

  const coreGeo = new THREE.CylinderGeometry(0.006, 0.006, 1, 8, 1, true);
  const glowGeo = new THREE.CylinderGeometry(0.02, 0.02, 1, 12, 1, true);
  const palmGeo = new THREE.SphereGeometry(0.032, 16, 12);
  const haloGeo = new THREE.SphereGeometry(0.055, 16, 12);
  const FLUX_COL = "#d9c4ff";
</script>

{#if beam.len > 1e-4}
  <T.Mesh geometry={glowGeo} position={beam.pos} quaternion={beam.quat}
    scale={[1, beam.len, 1]}>
    <T.MeshBasicMaterial color={FLUX_COL} transparent opacity={0.22}
      depthWrite={false} />
  </T.Mesh>
  <T.Mesh geometry={coreGeo} position={beam.pos} quaternion={beam.quat}
    scale={[1, beam.len, 1]}>
    <T.MeshBasicMaterial color="#ffffff" />
  </T.Mesh>
{/if}
{#each palms as p, h (h)}
  <T.Mesh geometry={palmGeo} position={[p[0], p[1], -p[2]]}>
    <T.MeshBasicMaterial color={NOTE_COLORS[h]} />
  </T.Mesh>
  <T.Mesh geometry={haloGeo} position={[p[0], p[1], -p[2]]}>
    <T.MeshBasicMaterial color={NOTE_COLORS[h]} transparent opacity={0.3}
      depthWrite={false} />
  </T.Mesh>
{/each}
