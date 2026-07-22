<script>
  // Flying note cubes (direction arrows, dot discs, in-reach outlines),
  // resolved-note aftermath (split halves / fail outline + reason), and
  // score popups. Positions/outcomes are DERIVED from chart + events —
  // never re-simulated (single source of truth, as in the old viewer).
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import { ConvexGeometry }
    from "three/addons/geometries/ConvexGeometry.js";
  import { pb } from "./state.svelte.js";
  import {
    BAD_COL, BLOCK, DT, GOOD_COL, HALF, MISS_COL, REACH_Z, SABER_COLORS,
    Z_FAR, Z_MISS, activePosts, aftermaths, cutPlaneLocal, noteCenter,
    noteRot, pendingNotes,
  } from "./replay.js";
  import TextSprite from "./TextSprite.svelte";

  const time = $derived(pb.cur * DT);
  const pend = $derived(pendingNotes(pb.rep, time));
  const after = $derived(aftermaths(pb.rep, time));
  const posts = $derived(activePosts(pb.rep, time));

  const boxGeo = new THREE.BoxGeometry(BLOCK, BLOCK, BLOCK);
  const edgesGeo = new THREE.EdgesGeometry(boxGeo);
  const failGeo = new THREE.EdgesGeometry(
    new THREE.PlaneGeometry(BLOCK, BLOCK));

  // ---- real cut splits: slice the cube by the ACTUAL cut plane -------------
  // (reconstructed from the cutting blade's sweep — see cutPlaneLocal).
  // Note-local frame maps to the scene group as (x, y, -z).
  const toScene = ([x, y, z]) => new THREE.Vector3(x, y, -z);
  const CANONICAL = { p: [0, 0, 0], n: [1, 0, 0], v: [0, -1, 0], b: [0, 0, 1] };

  const _corners = [];
  for (const sx of [-HALF, HALF])
    for (const sy of [-HALF, HALF])
      for (const sz of [-HALF, HALF]) _corners.push([sx, sy, sz]);
  const _edges = [];
  for (let i = 0; i < 8; i++)
    for (let j = i + 1; j < 8; j++)
      if (_corners[i].filter((v, k) => v !== _corners[j][k]).length === 1)
        _edges.push([i, j]);

  function sliceCube(cp) {
    const sd = _corners.map(
      (c) => (c[0] - cp.p[0]) * cp.n[0] + (c[1] - cp.p[1]) * cp.n[1]
        + (c[2] - cp.p[2]) * cp.n[2]);
    const A = [], B = [];
    _corners.forEach((c, i) => {
      if (sd[i] <= 0) A.push(c);
      if (sd[i] >= 0) B.push(c);
    });
    for (const [i, j] of _edges) {
      if ((sd[i] < 0) !== (sd[j] < 0)) {
        const t = sd[i] / (sd[i] - sd[j]);
        const p = _corners[i].map(
          (v, k) => v + t * (_corners[j][k] - v));
        A.push(p);
        B.push(p);
      }
    }
    if (A.length < 4 || B.length < 4) return null;
    return [
      { geo: new ConvexGeometry(A.map(toScene)), sign: -1 },
      { geo: new ConvexGeometry(B.map(toScene)), sign: 1 },
    ];
  }

  // halves + frame per cut note, built once (cached per loaded replay)
  const _splitCache = new WeakMap();
  function splitFor(i) {
    let m = _splitCache.get(pb.rep);
    if (!m) _splitCache.set(pb.rep, (m = new Map()));
    if (!m.has(i)) {
      const cp = cutPlaneLocal(pb.rep, i) ?? CANONICAL;
      let sides = sliceCube(cp);
      if (!sides) sides = sliceCube({ ...cp, p: [0, 0, 0] }); // grazing plane
      m.set(i, { cp, sides: sides ?? [] });
    }
    return m.get(i);
  }

  const aaq = (axis, ang) => {
    const q = new THREE.Quaternion().setFromAxisAngle(
      toScene(axis).normalize(), ang);
    return [q.x, q.y, q.z, q.w];
  };

  // Beat Saber-style low-poly arrow (pointing +x, rotated to the
  // note's angle): a single 3-vertex triangle — no tail, white like
  // the real game's glyphs
  const arrowShape = new THREE.Shape();
  arrowShape.moveTo(0.14, 0);
  arrowShape.lineTo(-0.09, 0.12);
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

{#each after as { n, i, o, age, alpha } (n)}
  {@const [cx, cy] = noteCenter(n)}
  {#if o.kind === "good"}
    <!-- the cube sliced by the ACTUAL cut plane (blade sweep at the cut
         tick): the two pieces separate along the plane normal, get
         carried along the sweep, tumble about the blade axis, fade.
         Off-center cuts produce honestly unequal chunks. -->
    {@const zCut = Math.max((n.t - (o.tick + 1) * DT) * n.speed, -0.4)}
    {@const sp = splitFor(i)}
    {#each sp.sides as side}
      <T.Group
        position={[cx, cy, -zCut]}
        rotation={[0, 0, noteRot(n)]}
      >
        <T.Mesh
          geometry={side.geo}
          position={[
            side.sign * age * 1.4 * sp.cp.n[0] + age * 0.9 * sp.cp.v[0],
            side.sign * age * 1.4 * sp.cp.n[1] + age * 0.9 * sp.cp.v[1],
            -(side.sign * age * 1.4 * sp.cp.n[2] + age * 0.9 * sp.cp.v[2]),
          ]}
          quaternion={aaq(sp.cp.b, side.sign * age * 1.8)}
        >
          <T.MeshStandardMaterial
            color={SABER_COLORS[n.color]}
            transparent
            opacity={alpha}
            flatShading
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
