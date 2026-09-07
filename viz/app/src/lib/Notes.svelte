<script>
  // Flying objects (env v7: notes, bombs, chain heads/links — direction
  // arrows, dot discs, in-reach outlines, arc hint tubes), resolved-object
  // aftermath (split halves / link pop / bomb flash / fail outline +
  // reason), and score popups. Positions/outcomes are DERIVED from chart +
  // events — never re-simulated (single source of truth, as in the old
  // viewer).
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import { ConvexGeometry }
    from "three/addons/geometries/ConvexGeometry.js";
  import { pb } from "./state.svelte.js";
  import {
    BAD_COL, BLOCK, BOMB_RADIUS, DT, GOOD_COL, HALF, HB,
    KIND_BOMB, KIND_CHAIN_HEAD, KIND_CHAIN_LINK, LINK_D, LINK_H, LINK_W,
    MISS_COL, REACH_Z, Z_FAR, Z_MISS, activeArcs, activePosts,
    aftermaths, cutPlaneLocal, frameQuatScene, hitboxRot, noteCenter,
    noteFrame, noteLam, noteRot, pendingNotes, pendingWalls,
  } from "./replay.js";
  // the REAL object models + colours, as ScoreSaber's replay viewer draws
  // them (the canonical visual reference — see models.js)
  import {
    BOMB_COLOR, CHAIN_HEAD_H, CHAIN_HEAD_YC, NOTE_COLORS, WALL_ALPHA,
    WALL_COLOR, modelGeo, noteColors,
  } from "./models.js";
  import TextSprite from "./TextSprite.svelte";

  // env v15 head-anchored track: `zOff` = head_z - PLANE_Z0 of the body
  // being DRAWN this frame (Scene owns it, so notes and sabers can never
  // disagree). It is exactly 0 for pre-v15 replays.
  let { zOff = 0 } = $props();

  const time = $derived(pb.cur * DT);
  const pend = $derived(pendingNotes(pb.rep, time, zOff));
  const after = $derived(aftermaths(pb.rep, time));
  const posts = $derived(activePosts(pb.rep, time));
  const arcs = $derived(activeArcs(pb.rep, time, zOff));
  const walls = $derived(pendingWalls(pb.rep, time, zOff));

  // object meshes: the watch-viewer models (beveled 0.5 m cube, chain
  // head/link, spiked bomb, glyphs already placed on the player face)
  const noteGeo = modelGeo("note");
  const headGeo = modelGeo("chainHead");
  const linkGeo = modelGeo("chainLink");
  const bombGeo = modelGeo("bomb");
  const arrowGeo = modelGeo("arrow");
  const dotGeo = modelGeo("dot");
  const linkDotGeo = modelGeo("linkDot");
  // in-reach outline cue (viewer-specific): the object's bounding box
  const boxGeo = new THREE.BoxGeometry(BLOCK, BLOCK, BLOCK);
  const edgesGeo = new THREE.EdgesGeometry(boxGeo);
  const headEdgesGeo = new THREE.EdgesGeometry(
    new THREE.BoxGeometry(BLOCK, CHAIN_HEAD_H, BLOCK).translate(0, CHAIN_HEAD_YC, 0));
  const linkEdgesGeo = new THREE.EdgesGeometry(
    new THREE.BoxGeometry(LINK_W, LINK_H, LINK_D));
  // chain-head debris: the cube slices squashed/shifted onto the head's
  // real span (y in [-0.0435, 0.25] of the note frame)
  const HEAD_SQUASH = CHAIN_HEAD_H / BLOCK;
  const failGeo = new THREE.EdgesGeometry(
    new THREE.PlaneGeometry(BLOCK, BLOCK));
  // env v17 walls: one unit cube scaled per wall (w, h, l vary per wall)
  const wallGeo = new THREE.BoxGeometry(1, 1, 1);
  const wallEdgesGeo = new THREE.EdgesGeometry(wallGeo);
  // animated-size shapes as unit geometry + per-frame scale, same pattern
  // as the walls: size-in-`args` geometry would be recreated (and held
  // undisposed) every frame it is on screen
  const arcGeo = new THREE.CylinderGeometry(0.035, 0.035, 1, 8, 1, true);
  const flashGeo = new THREE.SphereGeometry(1, 20, 14);
  const WALL_COL = WALL_COLOR; // Wall.mat (1, 0.184, 0.184) @ alpha 0.5

  // real hitboxes ('h' overlay): good box shifted toward the player; the
  // small box is all the wrong-color saber can trigger. Env v7.4 rolls
  // both about the approach axis with the note (hitboxRot) — v7.0's
  // axis-aligned boxes were a BeatLeader-viewer simplification mistaken
  // for game behavior; real colliders sit on the rotated note transform.
  // Env v7.6 adds the look-at-player blend (noteFrame): boxes ride the
  // full note frame, exactly like the sim's colliders.
  const hbGeo = (h) => new THREE.EdgesGeometry(
    new THREE.BoxGeometry(2 * h[0], 2 * h[1], 2 * h[2]));
  const goodHbGeo = hbGeo(HB.goodHalf);
  const goodHbDotGeo = hbGeo(HB.goodHalfDot);
  const goodHbLinkGeo = hbGeo(HB.goodHalfLink);
  const badHbGeo = hbGeo(HB.badHalf);
  const badHbLinkGeo = hbGeo(HB.badHalfLink);
  const bombHbGeo = new THREE.EdgesGeometry(
    new THREE.SphereGeometry(HB.bombR, 10, 6));
  function hitboxFor(n) {
    if (n.kind === KIND_BOMB) return { good: bombHbGeo, bad: null };
    if (n.kind === KIND_CHAIN_LINK)
      return { good: goodHbLinkGeo, bad: badHbLinkGeo };
    return { good: n.any ? goodHbDotGeo : goodHbGeo, bad: badHbGeo };
  }

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

  // note/glyph colours per side (NoteManager.GetMaterialColors: note
  // emission 0.15; near-white arrow at value 0.75 with a half-saturation
  // glow). Materials are per instance (each mesh below declares its own)
  // so the distance fade stays per note.
  const NK = [noteColors(0), noteColors(1)];

  const inReach = (z) => z <= REACH_Z && z >= -Z_MISS;
  const fade = (z) => Math.min(1, (Z_FAR - z) / 2);

  // arc hint tube: y-aligned cylinder -> segment between two world points
  const Y_AXIS = new THREE.Vector3(0, 1, 0);
  function arcSeg(a, b) {
    const av = toScene(a), bv = toScene(b);
    const dir = bv.clone().sub(av);
    const q = new THREE.Quaternion().setFromUnitVectors(
      Y_AXIS, dir.clone().normalize());
    const mid = av.clone().add(bv).multiplyScalar(0.5);
    return { pos: [mid.x, mid.y, mid.z], quat: [q.x, q.y, q.z, q.w],
             len: dir.length() };
  }
</script>

<!-- env v17 walls (spec §12a): translucent red AABBs riding the SAME
     head-anchored track as the notes (zOff already applied by
     pendingWalls). World box x: cx ± w/2, y: cy ± h/2, z: zFront ..
     zFront + l (away from the player), so the scene-space center is
     -(zFront + l/2). Fade into the tunnel uses the chart-time front z,
     like the notes' zBase. -->
{#each walls as { w, zFront, zFrontBase } (w)}
  {@const alpha = fade(zFrontBase)}
  <T.Group
    position={[w.cx, w.cy, -(zFront + w.l / 2)]}
    scale={[w.w, w.h, w.l]}
  >
    <T.Mesh geometry={wallGeo}>
      <T.MeshBasicMaterial
        color={WALL_COL}
        transparent
        opacity={WALL_ALPHA * alpha}
        depthWrite={false}
      />
    </T.Mesh>
    <T.LineSegments geometry={wallEdgesGeo}>
      <T.LineBasicMaterial
        color={WALL_COL}
        transparent
        opacity={0.9 * alpha}
        depthWrite={false}
      />
    </T.LineSegments>
  </T.Group>
{/each}

<!-- arc grants: subtle glowing tube head -> tail (a scoring hint) -->
{#each arcs as a (a.k)}
  {@const s = arcSeg(a.a, a.b)}
  {#if s.len > 1e-4}
    <T.Mesh geometry={arcGeo} position={s.pos} quaternion={s.quat}
      scale={[1, s.len, 1]}>
      <T.MeshBasicMaterial
        color={NOTE_COLORS[a.color]}
        transparent
        opacity={0.22}
        depthWrite={false}
      />
    </T.Mesh>
  {/if}
{/each}

<!-- `z` is the RENDERED world z (chart-time z + the v15 track offset);
     `zBase` is the chart-time z. Distance-to-the-player judgements (fade
     into the tunnel, in-reach outline) use zBase — the tunnel and reach
     box ride the same offset in Scene.svelte, so a zBase test is exactly
     a test against the shifted scenery. Position/billboarding use z,
     mirroring lib.rs (lam is computed from the OFFSET z there). -->
{#each pend as { n, z, zBase } (n)}
  {@const [cx, cy] = noteCenter(n)}
  {@const alpha = fade(zBase)}
  <!-- env v7.6: plain notes (dots included) look at the player's head as
       they approach (lam blends toward full head-facing at the plane);
       heads/links keep the pure z-roll via lam = 0 -->
  {@const lam = noteLam(n, z)}
  {@const vq = frameQuatScene(noteFrame([cx, cy, z], noteRot(n), lam))}
  {#if n.kind === KIND_BOMB}
    <!-- bomb: the spiked Bomb.obj at the viewer's scale (~0.65 m over
         the spikes), Bomb.mat dark gray, glossy; rides the note frame
         like the notes do (its own rotation is a plain z-roll) -->
    <T.Mesh geometry={bombGeo} position={[cx, cy, -z]} quaternion={vq}>
      <!-- the watch scene lights its dark bomb from a bright environment;
           on our black tunnel a little self-illumination + gloss keeps the
           silhouette readable -->
      <T.MeshStandardMaterial
        color={BOMB_COLOR}
        emissive="#3a3a44"
        roughness={0.2}
        metalness={0.4}
        transparent={alpha < 1}
        opacity={alpha}
      />
    </T.Mesh>
  {:else}
    {@const geo = n.kind === KIND_CHAIN_LINK ? linkGeo
      : n.kind === KIND_CHAIN_HEAD ? headGeo : noteGeo}
    {@const edges = n.kind === KIND_CHAIN_LINK ? linkEdgesGeo
      : n.kind === KIND_CHAIN_HEAD ? headEdgesGeo : edgesGeo}
    {@const k = NK[n.color]}
    <!-- the whole visual (model + glyph) rides the v7.6 note frame. The
         glyphs are baked in place on the player-facing (+z') face, in
         the note frame where rotation 0 = a down note — the same
         convention as the watch viewer (DirectionAngles: down = 0), so
         the real Arrow mesh needs no extra roll: at rotation 0 its apex
         points -y from the far (upper) half, exactly as in the game. -->
    <T.Group position={[cx, cy, -z]} quaternion={vq}>
      <T.Mesh geometry={geo}>
        <T.MeshStandardMaterial
          color={k.base}
          emissive={k.emissive}
          roughness={0.35}
          metalness={0.1}
          transparent={alpha < 1}
          opacity={alpha}
        />
      </T.Mesh>
      {#if inReach(zBase)}
        <T.LineSegments geometry={edges}>
          <T.LineBasicMaterial color="#e8eaf0" />
        </T.LineSegments>
      {/if}
      <T.Mesh geometry={n.kind === KIND_CHAIN_LINK ? linkDotGeo
        : n.any ? dotGeo : arrowGeo}>
        <T.MeshStandardMaterial
          color={k.arrowBase}
          emissive={k.arrowGlow}
          emissiveIntensity={0.55}
          roughness={0.6}
          transparent={alpha < 1}
          opacity={alpha}
        />
      </T.Mesh>
    </T.Group>
  {/if}
  {#if n.fpre}
    <TextSprite text="free pre" color="#9ece6a" size={0.07} bold={false}
      position={[cx, cy - HALF - 0.1, -z]} opacity={alpha * 0.85} />
  {/if}
  {#if n.fpost}
    <TextSprite text="free post" color="#9ece6a" size={0.07} bold={false}
      position={[cx, cy + HALF + 0.1, -z]} opacity={alpha * 0.85} />
  {/if}
  {#if pb.hitboxes}
    {@const hb = hitboxFor(n)}
    <!-- hitboxes live on the same v7.6 frame (with the hitbox roll —
         links roll by their tangent), centered on the NOTE CENTER; the
         good box's -0.25 shift is along LOCAL z' inside the group -->
    {@const hq = frameQuatScene(noteFrame([cx, cy, z], hitboxRot(n), lam))}
    <T.Group position={[cx, cy, -z]} quaternion={hq}>
      <T.LineSegments geometry={hb.good} position={[0, 0, -HB.goodZShift]}>
        <T.LineBasicMaterial color="#7dd3a0" transparent opacity={0.5} />
      </T.LineSegments>
      {#if hb.bad}
        <T.LineSegments geometry={hb.bad}>
          <T.LineBasicMaterial color="#f0a35e" transparent opacity={0.5} />
        </T.LineSegments>
      {/if}
    </T.Group>
  {/if}
{/each}

<!-- `oz` is the v15 track offset frozen at the frame the effect spawned:
     the debris/outline sits where the object actually was, and stays put
     while later head motion moves the live track. 0 on pre-v15 replays. -->
{#each after as { n, i, o, age, alpha, zOff: oz } (n)}
  {@const [cx, cy] = noteCenter(n)}
  {#if o.kind === "bomb"}
    <!-- bomb hit: red flash + label at the bomb's position at hit time -->
    {@const zHit = (n.t - (o.tick + 1) * DT) * n.speed + oz}
    <T.Mesh geometry={flashGeo} position={[cx, cy, -zHit]}
      scale={BOMB_RADIUS + age * 1.2}>
      <T.MeshBasicMaterial color="#ff4a4a" transparent
        opacity={alpha * 0.55} depthWrite={false} />
    </T.Mesh>
    <TextSprite
      text="bomb!"
      color={MISS_COL}
      size={0.16}
      position={[cx, cy + BOMB_RADIUS + 0.15 + age * 0.4, -zHit]}
      opacity={alpha}
    />
  {:else if o.kind === "good" && n.kind === KIND_CHAIN_LINK}
    <!-- link good cut: quick shrink/pop of the slab -->
    {@const zCut = Math.max((n.t - (o.tick + 1) * DT) * n.speed, -0.4) + oz}
    {@const sc = Math.max(0.02, 1 - age * 2.4)}
    <T.Mesh geometry={linkGeo} position={[cx, cy, -zCut]}
      rotation={[0, 0, noteRot(n)]} scale={[sc, sc, sc]}>
      <T.MeshStandardMaterial color={NOTE_COLORS[n.color]} transparent
        opacity={alpha} />
    </T.Mesh>
    <TextSprite
      text={"+" + o.points.toFixed(0)}
      color={GOOD_COL}
      size={0.13}
      position={[cx, cy + LINK_H + age * 0.5, 0.02 - oz]}
      opacity={alpha}
    />
  {:else if o.kind === "good"}
    <!-- the cube sliced by the ACTUAL cut plane (blade sweep at the cut
         tick): the two pieces separate along the plane normal, get
         carried along the sweep, tumble about the blade axis, fade.
         Off-center cuts produce honestly unequal chunks. Chain heads
         reuse the split squashed and shifted onto the real head's span
         (the note minus its swing-side portion). -->
    {@const zCut = Math.max((n.t - (o.tick + 1) * DT) * n.speed, -0.4) + oz}
    {@const sp = splitFor(i)}
    {@const squash = n.kind === KIND_CHAIN_HEAD ? HEAD_SQUASH : 1}
    {@const yOff = n.kind === KIND_CHAIN_HEAD ? CHAIN_HEAD_YC : 0}
    {#each sp.sides as side}
      <T.Group
        position={[cx, cy, -zCut]}
        rotation={[0, 0, noteRot(n)]}
      >
        <T.Mesh
          geometry={side.geo}
          scale={[1, squash, 1]}
          position={[
            side.sign * age * 1.4 * sp.cp.n[0] + age * 0.9 * sp.cp.v[0],
            side.sign * age * 1.4 * sp.cp.n[1] + age * 0.9 * sp.cp.v[1] + yOff,
            -(side.sign * age * 1.4 * sp.cp.n[2] + age * 0.9 * sp.cp.v[2]),
          ]}
          quaternion={aaq(sp.cp.b, side.sign * age * 1.8)}
        >
          <T.MeshStandardMaterial
            color={NOTE_COLORS[n.color]}
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
      position={[cx, cy + HALF * 1.2 + age * 0.5, 0.02 - oz]}
      opacity={alpha}
    />
  {:else}
    <!-- bad cut / miss: outline + reason at the object's position when
         the event fired. Since env v7.3 misses register at the head
         plane (z = -0.65) so the outline reads fine there; the clamp
         only guards pre-v7.3 replays whose misses fired further back.
         Env v15: the clamp stays on the chart-time z (the sim's miss
         boundary is base z < PLANE_Z0), then the frozen offset carries
         the outline onto the live miss plane at world z = head_z. -->
    {@const col = o.kind === "bad" ? BAD_COL : MISS_COL}
    {@const zEv = Math.max((n.t - (o.tick + 1) * DT) * n.speed, -Z_MISS) + oz}
    <T.LineSegments geometry={failGeo} position={[cx, cy, -zEv]}
      rotation={[0, 0, noteRot(n)]}>
      <T.LineBasicMaterial color={col} transparent opacity={alpha} />
    </T.LineSegments>
    <TextSprite
      text={o.reason.replace("_", " ")}
      color={col}
      size={0.14}
      position={[cx, cy + HALF * 1.6, -zEv]}
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
    position={[cx, cy - HALF * 1.4 - p.age * 0.4, 0.02 - p.zOff]}
    opacity={p.alpha}
  />
{/each}
