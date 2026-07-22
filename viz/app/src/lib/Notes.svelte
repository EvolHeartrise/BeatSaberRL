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
    Z_FAR, Z_MISS, activePosts, aftermaths, noteCenter, noteRot,
    pendingNotes,
  } from "./replay.js";
  import TextSprite from "./TextSprite.svelte";

  const time = $derived(pb.cur * DT);
  const pend = $derived(pendingNotes(pb.rep, time));
  const after = $derived(aftermaths(pb.rep, time));
  const posts = $derived(activePosts(pb.rep, time));

  const boxGeo = new THREE.BoxGeometry(BLOCK, BLOCK, BLOCK);
  const edgesGeo = new THREE.EdgesGeometry(boxGeo);
  // real split halves (the note's actual cube sliced along the cut
  // plane): each covers local x in [0, 0.2] / [-0.2, 0]
  const halfGeo = new THREE.BoxGeometry(HALF, BLOCK, BLOCK);
  const failGeo = new THREE.EdgesGeometry(
    new THREE.PlaneGeometry(BLOCK, BLOCK));

  // Beat Saber-style low-poly chevron (pointing +x, rotated to the
  // note's angle): a solid 4-vertex wedge with a notched back — no
  // tail, white like the real game's glyphs
  const arrowShape = new THREE.Shape();
  arrowShape.moveTo(0.14, 0);
  arrowShape.lineTo(-0.09, 0.12);
  arrowShape.lineTo(-0.02, 0);
  arrowShape.lineTo(-0.09, -0.12);
  arrowShape.closePath();
  const arrowGeo = new THREE.ShapeGeometry(arrowShape);
  const dotGeo = new THREE.CircleGeometry(0.07, 12);   // low-poly BS dot
  const GLYPH_COL = "#f0f2f8";                          // BS-white glyphs

  const inReach = (z) => z <= REACH_Z && z >= -Z_MISS;
  const fade = (z) => Math.min(1, (Z_FAR - z) / 2);
</script>

{#each pend as { n, z } (n)}
  {@const [cx, cy] = noteCenter(n)}
  {@const alpha = fade(z)}
  {@const rot = noteRot(n)}
  <T.Mesh position={[cx, cy, -z]} rotation={[0, 0, rot]}>
    <T.BoxGeometry args={[BLOCK, BLOCK, BLOCK]} />
    <T.MeshStandardMaterial
      color={SABER_COLORS[n.color]}
      transparent={alpha < 1}
      opacity={alpha}
    />
  </T.Mesh>
  {#if inReach(z)}
    <T.LineSegments geometry={edgesGeo} position={[cx, cy, -z]}
      rotation={[0, 0, rot]}>
      <T.LineBasicMaterial color="#e8eaf0" />
    </T.LineSegments>
  {/if}
  {#if n.any}
    <T.Mesh geometry={dotGeo} position={[cx, cy, -z + HALF + 0.005]}>
      <T.MeshBasicMaterial color={GLYPH_COL} transparent={alpha < 1} opacity={alpha} />
    </T.Mesh>
  {:else}
    <T.Mesh
      geometry={arrowGeo}
      position={[cx, cy, -z + HALF + 0.005]}
      rotation={[0, 0, n.angle]}
    >
      <T.MeshBasicMaterial color={GLYPH_COL} transparent={alpha < 1} opacity={alpha} />
    </T.Mesh>
  {/if}
{/each}

{#each after as { n, o, age, alpha } (n)}
  {@const [cx, cy] = noteCenter(n)}
  {#if o.kind === "good"}
    <!-- the ACTUAL cube sliced along the cut plane: in the note's own
         rotated frame the cut direction is local -y, so the halves are
         the local left/right half-cubes, flying apart along local ±x,
         carried a little along the cut, tumbling and fading -->
    {@const zCut = Math.max((n.t - (o.tick + 1) * DT) * n.speed, -0.4)}
    {@const sep = HALF / 2 + age * 1.4}
    {@const carry = -age * 0.9}
    {#each [-1, 1] as side}
      <T.Group
        position={[cx, cy, -zCut]}
        rotation={[0, 0, noteRot(n)]}
      >
        <T.Mesh
          geometry={halfGeo}
          position={[side * sep, carry, 0]}
          rotation={[0, side * age * 2.2, 0]}
        >
          <T.MeshStandardMaterial
            color={SABER_COLORS[n.color]}
            transparent
            opacity={alpha}
          />
        </T.Mesh>
      </T.Group>
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
    <T.LineSegments geometry={failGeo} position={[cx, cy, 0.01]}
      rotation={[0, 0, noteRot(n)]}>
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
