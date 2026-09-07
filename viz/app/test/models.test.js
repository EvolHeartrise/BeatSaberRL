// Node checks for the baked ScoreSaber-viewer models (src/lib/models.js):
// every part loads as a valid indexed BufferGeometry, the reference sizes
// match the watch prefabs (the canonical visual sizes), triangle winding
// agrees with the vertex normals (outward faces after the two mirrors),
// and the arrow glyph points along the swing direction from the far half
// of the face. Run: node test/models.test.js
import assert from "node:assert/strict";
import * as THREE from "three";
import {
  BLADE_R, BOMB_R, CHAIN_HEAD_H, CHAIN_HEAD_Y, HANDLE_LEN, LINK_HALF_H,
  NOTE_COLORS, PART_NAMES, modelBBox, modelGeo, noteColors,
} from "../src/lib/models.js";

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol,
  `${msg}: ${a} vs ${b}`);

for (const part of PART_NAMES) {
  const g = modelGeo(part);
  const pos = g.attributes.position, nrm = g.attributes.normal;
  const idx = g.index.array;
  assert.ok(pos.count > 0 && pos.count === nrm.count, `${part} attrs`);
  assert.equal(idx.length % 3, 0, `${part} index`);
  for (const i of idx) assert.ok(i < pos.count, `${part} index range`);
  // winding vs normals
  let bad = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const [a, b, c] = [idx[t], idx[t + 1], idx[t + 2]];
    const A = [pos.getX(a), pos.getY(a), pos.getZ(a)];
    const B = [pos.getX(b), pos.getY(b), pos.getZ(b)];
    const C = [pos.getX(c), pos.getY(c), pos.getZ(c)];
    const e1 = B.map((v, k) => v - A[k]), e2 = C.map((v, k) => v - A[k]);
    const gn = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
      e1[0] * e2[1] - e1[1] * e2[0]];
    const nn = [0, 1, 2].map((k) =>
      nrm.getComponent(a, k) + nrm.getComponent(b, k) + nrm.getComponent(c, k));
    if (gn[0] * nn[0] + gn[1] * nn[1] + gn[2] * nn[2] < 0) bad++;
  }
  assert.ok(bad <= idx.length / 3 * 0.01, `${part}: ${bad} inverted tris`);
  // bbox in the data agrees with the geometry
  const bb = g.boundingBox, [mn, mx] = modelBBox(part);
  for (let k = 0; k < 3; k++) {
    near(bb.min.getComponent(k), mn[k], 1e-3, `${part} bbox min ${k}`);
    near(bb.max.getComponent(k), mx[k], 1e-3, `${part} bbox max ${k}`);
  }
}

// ---- reference sizes (ScoreSaber watch prefabs) ---------------------------
const [nmn, nmx] = modelBBox("note");
for (let k = 0; k < 3; k++) { near(nmn[k], -0.25, 1e-3, "note min"); near(nmx[k], 0.25, 1e-3, "note max"); }

const [amn, amx] = modelBBox("arrow");
near(amx[0] - amn[0], 0.3088, 1e-3, "arrow width");
near(amn[1], 0.0554, 1e-3, "arrow apex (past centre, swing side)");
near(amx[1], 0.1524, 1e-3, "arrow base (far edge)");
assert.ok(amn[2] >= 0.238 - 1e-4 && amx[2] <= 0.2531, "arrow sits on the +z (player) face");
// apex is the single vertex at min y, centred in x -> points -y (down at rot 0)
{
  const g = modelGeo("arrow"), p = g.attributes.position;
  const apex = [];
  for (let i = 0; i < p.count; i++)
    if (Math.abs(p.getY(i) - amn[1]) < 1e-4) apex.push(p.getX(i));
  assert.ok(apex.length > 0 && apex.every((x) => Math.abs(x) < 1e-4), "arrow apex centred");
}

const [dmn, dmx] = modelBBox("dot");
near(dmx[0] - dmn[0], 0.175, 1e-3, "dot diameter");
assert.ok(dmn[2] >= 0.215 - 1e-4, "dot on the player face");

near(CHAIN_HEAD_Y[0], -0.0435, 1e-3, "chain head flat face");
near(CHAIN_HEAD_Y[1], 0.25, 1e-3, "chain head top");
near(CHAIN_HEAD_H, 0.2935, 1e-3, "chain head height");
// the arrow lies within the head's span (it is drawn on chain heads too)
assert.ok(amn[1] > CHAIN_HEAD_Y[0] && amx[1] < CHAIN_HEAD_Y[1], "arrow on the chain head");

near(LINK_HALF_H, 0.0435, 1e-3, "link half height");
const [lmn, lmx] = modelBBox("chainLink");
near(lmx[0] - lmn[0], 0.5, 1e-3, "link width");
near(lmx[2] - lmn[2], 0.5, 1e-3, "link depth");
const [ldmn, ldmx] = modelBBox("linkDot");
near(ldmx[0] - ldmn[0], 0.0375, 1e-3, "link dot diameter");

near(BOMB_R, 0.324, 0.01, "bomb axis extent (spikes reach r 0.345 on a diagonal)");

const [bmn, bmx] = modelBBox("blade");
near(BLADE_R, 0.018, 1e-3, "blade radius");
near(bmn[2], -1.0, 1e-3, "blade tip at 1.0 m along scene -z");
near(bmx[2], 0.0, 1e-3, "blade starts at the grip");
const [hmn, hmx] = modelBBox("handle");
near(hmn[2], 0.0, 1e-3, "handle starts at the grip");
near(HANDLE_LEN, 0.16, 1e-3, "handle length");

const [hsmn, hsmx] = modelBBox("headset");
near(hsmx[0] - hsmn[0], 0.2914, 1e-3, "headset width (0.35 scene scale)");
near(hsmn[2], -0.094, 1e-3, "visor ahead of the pivot (scene -z)");
near(hsmx[2], 0.2724, 1e-3, "straps behind the pivot");

// ---- colours ---------------------------------------------------------------
assert.deepEqual(NOTE_COLORS, ["#c81414", "#288ed2"]);
for (const c of [0, 1]) {
  const m = noteColors(c);
  assert.equal(m.base, NOTE_COLORS[c]);
  // arrow base is near-white (saturation 0.05 of the base) at value 0.75
  // (read back in sRGB — the space Unity's HSV rules are written in)
  const ab = m.arrowBase.getRGB({}, THREE.SRGBColorSpace);
  const max = Math.max(ab.r, ab.g, ab.b), min = Math.min(ab.r, ab.g, ab.b);
  near(max, 0.75 * (c === 0 ? 200 / 255 : 210 / 255), 1e-2, `arrow base value ${c}`);
  assert.ok((max - min) / max < 0.06, `arrow base near-white ${c}`);
  // glow keeps the hue at half saturation, full value
  const gl = m.arrowGlow.getRGB({}, THREE.SRGBColorSpace);
  near(Math.max(gl.r, gl.g, gl.b), 1.0, 1e-3, `arrow glow value ${c}`);
  const gmin = Math.min(gl.r, gl.g, gl.b);
  near(1 - gmin, 0.5 * (c === 0 ? 0.9 : 0.8095), 1e-2, `arrow glow saturation ${c}`);
}

console.log(`models OK: ${PART_NAMES.length} parts`);

// ---- cut debris: slicing the real models ------------------------------------
import { sliceGeometry } from "../src/lib/models.js";
function volume(g) { // signed volume of a closed non-indexed triangle soup
  const p = g.attributes.position; let v = 0;
  for (let t = 0; t < p.count; t += 3) {
    const a = [p.getX(t), p.getY(t), p.getZ(t)], b = [p.getX(t + 1), p.getY(t + 1), p.getZ(t + 1)],
      c = [p.getX(t + 2), p.getY(t + 2), p.getZ(t + 2)];
    v += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
  }
  return v;
}
function indexedVolume(g) {
  const p = g.attributes.position, idx = g.index.array; let v = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const [i, j, k] = [idx[t], idx[t + 1], idx[t + 2]];
    const a = [p.getX(i), p.getY(i), p.getZ(i)], b = [p.getX(j), p.getY(j), p.getZ(j)], c = [p.getX(k), p.getY(k), p.getZ(k)];
    v += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
  }
  return v;
}
for (const part of ["note", "chainHead"]) {
  const g = modelGeo(part), vol = indexedVolume(g);
  assert.ok(vol > 0, `${part} is a closed outward-wound solid (vol ${vol})`);
  for (const [p, n] of [[[0, 0, 0], [1, 0, 0]], [[0.05, -0.02, 0.1], [0.3, 1, 0.2]], [[0, 0.1, 0], [0, 1, 0]]]) {
    const sides = sliceGeometry(g, p, n);
    assert.ok(sides && sides.length === 2, `${part} slice ${n}`);
    const [A, B] = sides;
    assert.equal(A.sign, -1); assert.equal(B.sign, 1);
    const va = volume(A.geo), vb = volume(B.geo);
    assert.ok(va > 0 && vb > 0, `${part} halves closed (${va}, ${vb})`);
    near(va + vb, vol, vol * 1e-3, `${part} halves sum to the whole`);
    // every vertex on its own side of the plane
    const nl = Math.hypot(...n);
    for (const [S, sgn] of [[A, -1], [B, 1]]) {
      const q = S.geo.attributes.position;
      for (let i = 0; i < q.count; i++) {
        const d = ((q.getX(i) - p[0]) * n[0] + (q.getY(i) - p[1]) * n[1] + (q.getZ(i) - p[2]) * n[2]) / nl;
        assert.ok(d * sgn >= -1e-4, `${part} vertex on the wrong side (${d})`);
      }
    }
  }
  // a plane that misses the mesh -> null
  assert.equal(sliceGeometry(g, [0, 5, 0], [0, 1, 0]), null, `${part} miss`);
}
console.log("slice OK");
