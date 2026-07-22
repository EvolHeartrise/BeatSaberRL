<script>
  // Flying note cubes (direction arrows, dot discs, in-reach outlines),
  // resolved-note aftermath (split halves / fail outline + reason), and
  // score popups. Positions/outcomes are DERIVED from chart + events —
  // never re-simulated (single source of truth, as in the old viewer).
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import { pb } from "./state.svelte.js";
  import {
    BAD_COL, BLOCK, DT, GOOD_COL, HALF, MISS_COL, REACH_Z, SABER_COLORS,
    Z_FAR, Z_MISS, activePosts, aftermaths, noteCenter, pendingNotes,
  } from "./replay.js";
  import TextSprite from "./TextSprite.svelte";

  const time = $derived(pb.cur * DT);
  const pend = $derived(pendingNotes(pb.rep, time));
  const after = $derived(aftermaths(pb.rep, time));
  const posts = $derived(activePosts(pb.rep, time));

  const boxGeo = new THREE.BoxGeometry(BLOCK, BLOCK, BLOCK);
  const edgesGeo = new THREE.EdgesGeometry(boxGeo);
  const halfGeo = new THREE.BoxGeometry(BLOCK, BLOCK / 2, 0.03);
  const failGeo = new THREE.EdgesGeometry(
    new THREE.PlaneGeometry(BLOCK, BLOCK));

  // arrow (pointing +x, rotated to the note's angle) and dot-note disc
  const arrowShape = new THREE.Shape();
  arrowShape.moveTo(-0.13, -0.022);
  arrowShape.lineTo(0.03, -0.022);
  arrowShape.lineTo(0.03, -0.07);
  arrowShape.lineTo(0.14, 0);
  arrowShape.lineTo(0.03, 0.07);
  arrowShape.lineTo(0.03, 0.022);
  arrowShape.lineTo(-0.13, 0.022);
  arrowShape.closePath();
  const arrowGeo = new THREE.ShapeGeometry(arrowShape);
  const dotGeo = new THREE.CircleGeometry(0.07, 24);

  const inReach = (z) => z <= REACH_Z && z >= -Z_MISS;
  const fade = (z) => Math.min(1, (Z_FAR - z) / 2);
</script>

{#each pend as { n, z } (n)}
  {@const [cx, cy] = noteCenter(n)}
  {@const alpha = fade(z)}
  <T.Mesh position={[cx, cy, -z]}>
    <T.BoxGeometry args={[BLOCK, BLOCK, BLOCK]} />
    <T.MeshStandardMaterial
      color={SABER_COLORS[n.color]}
      transparent={alpha < 1}
      opacity={alpha}
    />
  </T.Mesh>
  {#if inReach(z)}
    <T.LineSegments geometry={edgesGeo} position={[cx, cy, -z]}>
      <T.LineBasicMaterial color="#e8eaf0" />
    </T.LineSegments>
  {/if}
  {#if n.any}
    <T.Mesh geometry={dotGeo} position={[cx, cy, -z + HALF + 0.005]}>
      <T.MeshBasicMaterial color="#0b0d11" transparent={alpha < 1} opacity={alpha} />
    </T.Mesh>
  {:else}
    <T.Mesh
      geometry={arrowGeo}
      position={[cx, cy, -z + HALF + 0.005]}
      rotation={[0, 0, n.angle]}
    >
      <T.MeshBasicMaterial color="#0b0d11" transparent={alpha < 1} opacity={alpha} />
    </T.Mesh>
  {/if}
{/each}

{#each after as { n, o, age, alpha } (n)}
  {@const [cx, cy] = noteCenter(n)}
  {#if o.kind === "good"}
    {@const px = -Math.sin(n.angle) * age * 1.4}
    {@const py = Math.cos(n.angle) * age * 1.4}
    {#each [[px, py], [-px, -py]] as [ox, oy]}
      <T.Mesh geometry={halfGeo} position={[cx + ox, cy + oy, 0]}>
        <T.MeshBasicMaterial color={GOOD_COL} transparent opacity={alpha} />
      </T.Mesh>
    {/each}
    <TextSprite
      text={"+" + o.points.toFixed(0)}
      color={GOOD_COL}
      size={0.16}
      position={[cx, cy + HALF * 1.2 + age * 0.5, 0.02]}
      opacity={alpha}
    />
  {:else}
    {@const col = o.kind === "bad" ? BAD_COL : MISS_COL}
    <T.LineSegments geometry={failGeo} position={[cx, cy, 0.01]}>
      <T.LineBasicMaterial color={col} transparent opacity={alpha} />
    </T.LineSegments>
    <TextSprite
      text={o.reason.replace("_", " ")}
      color={col}
      size={0.14}
      position={[cx, cy + HALF * 1.6, 0.02]}
      opacity={alpha}
    />
  {/if}
{/each}

{#each posts as p (p.note + "@" + p.tick)}
  {@const [cx, cy] = noteCenter(pb.rep.chart[p.note])}
  <TextSprite
    text={"+" + p.points.toFixed(0) + " follow"}
    color="#9ece6a"
    size={0.11}
    bold={false}
    position={[cx, cy - HALF * 1.4 - p.age * 0.4, 0.02]}
    opacity={p.alpha}
  />
{/each}
