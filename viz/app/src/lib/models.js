// The REAL Beat Saber object models, as ScoreSaber's replay viewer
// (watch.scoresaber.com, an ArcViewer fork) renders them — the canonical
// reference for every visual size in this viewer. Geometry comes from
// models.data.js (baked from the watch repo's Assets/Models with the
// Unity import scale, prefab child transforms and scene scale applied,
// already in this viewer's scene-local frame), and the colour rules
// below mirror NoteManager.GetMaterialColors / the .mat defaults.
//
// Reference sizes (Unity metres, from the prefabs):
//   note     Cube.fbx, 0.5 m beveled cube (Note.prefab, scale 1)
//   arrow    Note.fbx "Arrow" ×0.25 import, prefab scale 2, rot z 90°,
//            z −0.238: a flat triangle in the half AWAY from the swing
//            direction, apex 0.055 m past centre pointing along the
//            swing, base at the far edge (y 0.152), proud of the face by
//            3 mm
//   dot      Note.fbx "Dot" ×0.25, prefab scale 0.35, z −0.215: 0.175 m
//            disc, 5 mm proud
//   chainHead ChainHead.fbx: the note minus its swing-side portion —
//            y ∈ [−0.0435, 0.25] in the note frame (flat face toward
//            the links), full 0.5 m width/depth
//   chainLink ChainLink.fbx: 0.5 × 0.087 × 0.5 m slab; linkDot = the Dot
//            at prefab scale (0.075, 0.075, 0.5), z −0.235
//   bomb     Bomb.obj at scale 1: ~0.53 m core, spikes to r 0.345
//   blade    Saber.fbx "Blade" ×0.5 import, prefab z −0.08: r 0.018,
//            z ∈ [0, 1.0] along the hand's +z (scene −z)
//   handle   Saber.fbx "Handle": z ∈ [−0.16, 0] (behind the grip)
// (The watch viewer's Headset.fbx was baked too and then removed on
// 2026-09-06 by user request — the skeleton's head sphere stands alone.)
import * as THREE from "three";
import data from "./models.data.js";

function build(part) {
  const d = data[part];
  if (!d) throw new Error(`models: no part '${part}'`);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(d.p, 3));
  g.setAttribute("normal", new THREE.Float32BufferAttribute(d.n, 3));
  g.setIndex(d.i);
  g.computeBoundingSphere();
  g.computeBoundingBox();
  return g;
}

export const PART_NAMES = [
  "note", "arrow", "dot", "chainHead", "chainLink", "linkDot", "bomb",
  "blade", "handle",
];

// geometries built once per page (module scope), shared by every instance
const _geo = {};
export function modelGeo(part) {
  return _geo[part] ?? (_geo[part] = build(part));
}
export const modelBBox = (part) => data[part].bbox;

// derived reference dims (metres), used for outlines / debris / labels
export const NOTE_HALF = 0.25;
export const CHAIN_HEAD_Y = [data.chainHead.bbox[0][1], data.chainHead.bbox[1][1]];
export const CHAIN_HEAD_H = CHAIN_HEAD_Y[1] - CHAIN_HEAD_Y[0]; // 0.2935
export const CHAIN_HEAD_YC = (CHAIN_HEAD_Y[0] + CHAIN_HEAD_Y[1]) / 2; // 0.103
export const LINK_HALF_H = data.chainLink.bbox[1][1]; // 0.0435
export const BOMB_R = Math.max(
  ...[0, 1, 2].map((k) => Math.max(-data.bomb.bbox[0][k], data.bomb.bbox[1][k])));
export const BLADE_R = data.blade.bbox[1][0]; // 0.018
export const HANDLE_LEN = data.handle.bbox[1][2]; // 0.16

// ---- colours -------------------------------------------------------------
// ScoreSaber's default note/saber colours (ColorManager defaults, sRGB
// bytes 200/20/20 and 40/142/210). Left = red, right = blue.
export const NOTE_COLORS = ["#c81414", "#288ed2"];
// NoteManager (Previewer.unity fields): noteEmission 0.15,
// arrowSaturation 0.05, arrowBrightness 0.75, arrowGlowSaturation 0.5,
// arrowEmission 1.5 (HDR; clamped to 1 here — no bloom pass).
const NOTE_EMISSION = 0.15;
const ARROW_SAT = 0.05, ARROW_VAL = 0.75, ARROW_GLOW_SAT = 0.5;

// Unity's Color.RGBToHSV / SetHSV run on the material's sRGB values, so
// the HSV maths is done on sRGB bytes here and only the RESULT is handed to
// three (which converts sRGB -> its linear working space).
function hexToSRGB(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((b) => b / 255);
}
function rgbToHSV([r, g, b]) {
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
  let h = 0;
  if (d > 0) {
    if (max === r) h = ((g - b) / d + 6) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h /= 6;
  }
  return [h, max === 0 ? 0 : d / max, max];
}
function hsvToColor(h, s, v) {
  s = Math.min(1, s); v = Math.min(1, v);
  const i = Math.floor(h * 6), f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  const rgb = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
  return new THREE.Color().setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
}
// Unity Color.SetHSV(hue = null, saturation, value) on an sRGB base colour
function setHSV(base, sMul, v) {
  const [h, s] = rgbToHSV(hexToSRGB(base));
  return hsvToColor(h, s * sMul, v);
}

// per-colour material parameters, computed once
const _noteMat = {};
export function noteColors(color) {
  if (_noteMat[color]) return _noteMat[color];
  const [, , vBase] = rgbToHSV(hexToSRGB(NOTE_COLORS[color]));
  return (_noteMat[color] = {
    base: NOTE_COLORS[color],
    emissive: setHSV(NOTE_COLORS[color], 1, NOTE_EMISSION * vBase),
    arrowBase: setHSV(NOTE_COLORS[color], ARROW_SAT, ARROW_VAL * vBase),
    arrowGlow: setHSV(NOTE_COLORS[color], ARROW_GLOW_SAT, 1.0),
  });
}

// Bomb.mat: base (0.1415 gray), smoothness 0.95, fresnel rim.
export const BOMB_COLOR = "#242424";
// Wall.mat: (1, 0.184, 0.184) at alpha 0.5; WallEdges emission 3.
// WALL_ALPHA dropped 0.5 -> 0.34 with the 2026-09-09 cel pass: the walls
// are unlit (MeshBasicMaterial) and no longer tone-mapped, so the old
// value burned out into a flat red slab.
export const WALL_COLOR = "#ff2f2f", WALL_ALPHA = 0.34;

// ---- cut debris: slice a model by the real cut plane ----------------------
// The note and chain-head meshes are convex, so the cross-section is a
// convex polygon: clip every triangle against the plane (each side keeps
// its 1-2 sub-triangles with interpolated normals) and cap each half with
// a fan over the angle-sorted crossing points. Returns two non-indexed
// geometries [{geo, sign: -1}, {geo, sign: +1}] (sign −1 = the side the
// plane normal points away from), or null if the plane misses the mesh.
// `p`/`n` are in the geometry's own (scene-local) frame.
// Scratch output, reused by every slice: per side, positions and normals
// as float32 (exactly what the attribute arrays hold), grown on demand.
const _sliceOut = [0, 1].map(() => ({ p: new Float32Array(0), n: new Float32Array(0), len: 0 }));

export function sliceGeometry(geo, p, n) {
  // (flat scratch buffers instead of an array per vertex, normal and
  // crossing: a cut slices a ~3k-triangle model on the frame its debris
  // appears, and the per-vertex garbage made that the costliest thing in a
  // dense replay's frame loop. Same arithmetic in the same order, so the
  // halves are the very same floats.)
  const P = geo.attributes.position.array, NR = geo.attributes.normal.array;
  const idx = geo.index ? geo.index.array : null;
  const nl = Math.hypot(n[0], n[1], n[2]);
  const N = [n[0] / nl, n[1] / nl, n[2] / nl];
  const ntri = idx ? idx.length / 3 : P.length / 9;
  // 0 = negative side, 1 = positive; a side keeps at most 2 sub-triangles
  // per triangle, plus its cap fan (at most 2 crossings per triangle)
  const out = _sliceOut;
  for (const o of out) {
    o.len = 0;
    if (o.p.length < 36 * ntri) { o.p = new Float32Array(36 * ntri); o.n = new Float32Array(36 * ntri); }
  }
  const ring = []; // crossing points, flat x y z (duplicates welded below)
  // this triangle's corners and its clipped polygon per side, 6 numbers a
  // vertex (position, normal); a plane leaves at most 4 corners a side
  const tri = new Float64Array(18), ds = new Float64Array(3), x = new Float64Array(6);
  const poly = [new Float64Array(24), new Float64Array(24)], cnt = [0, 0];
  const keep = (side, src, o) => {
    const d = poly[side], k = 6 * cnt[side]++;
    for (let c = 0; c < 6; c++) d[k + c] = src[o + c];
  };
  const emit = (side, src, o) => {
    const d = out[side], k = d.len;
    d.p[k] = src[o]; d.p[k + 1] = src[o + 1]; d.p[k + 2] = src[o + 2];
    d.n[k] = src[o + 3]; d.n[k + 1] = src[o + 4]; d.n[k + 2] = src[o + 5];
    d.len += 3;
  };
  const EPS = 1e-7;
  for (let t = 0; t < ntri; t++) {
    for (let i = 0; i < 3; i++) {
      const id = idx ? idx[3 * t + i] : 3 * t + i;
      for (let c = 0; c < 3; c++) {
        tri[6 * i + c] = P[3 * id + c];
        tri[6 * i + 3 + c] = NR[3 * id + c];
      }
      ds[i] = (tri[6 * i] - p[0]) * N[0] + (tri[6 * i + 1] - p[1]) * N[1]
        + (tri[6 * i + 2] - p[2]) * N[2];
    }
    cnt[0] = cnt[1] = 0;
    // Sutherland–Hodgman around the triangle, keeping both sides
    for (let i = 0; i < 3; i++) {
      const j = (i + 1) % 3;
      const da = ds[i], db = ds[j];
      if (da <= EPS) keep(0, tri, 6 * i);
      if (da >= -EPS) keep(1, tri, 6 * i);
      if ((da < -EPS && db > EPS) || (da > EPS && db < -EPS)) {
        const tt = da / (da - db);
        for (let c = 0; c < 6; c++) x[c] = tri[6 * i + c] + tt * (tri[6 * j + c] - tri[6 * i + c]);
        keep(0, x, 0); keep(1, x, 0); ring.push(x[0], x[1], x[2]);
      }
    }
    for (let side = 0; side < 2; side++)
      for (let k = 1; k + 1 < cnt[side]; k++) {
        emit(side, poly[side], 0); emit(side, poly[side], 6 * k);
        emit(side, poly[side], 6 * (k + 1));
      }
  }
  if (!out[0].len || !out[1].len || ring.length < 9) return null;
  const push = (side, verts) => {
    for (const [v, nn] of verts) emit(side, [...v, ...nn], 0);
  };
  // cap: weld crossing points, sort by angle in the plane, fan both ways
  const key = (v) => v.map((x) => Math.round(x * 1e4)).join(",");
  const seen = new Set(), pts = [];
  for (let r = 0; r < ring.length; r += 3) {
    const v = [ring[r], ring[r + 1], ring[r + 2]], k = key(v);
    if (!seen.has(k)) { seen.add(k); pts.push(v); }
  }
  if (pts.length >= 3) {
    const c = [0, 1, 2].map((k) => pts.reduce((s, v) => s + v[k], 0) / pts.length);
    // in-plane basis
    const ax = Math.abs(N[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const u = [N[1] * ax[2] - N[2] * ax[1], N[2] * ax[0] - N[0] * ax[2], N[0] * ax[1] - N[1] * ax[0]];
    const ul = Math.hypot(...u); u[0] /= ul; u[1] /= ul; u[2] /= ul;
    const w = [N[1] * u[2] - N[2] * u[1], N[2] * u[0] - N[0] * u[2], N[0] * u[1] - N[1] * u[0]];
    const ang = (v) => Math.atan2(
      (v[0] - c[0]) * w[0] + (v[1] - c[1]) * w[1] + (v[2] - c[2]) * w[2],
      (v[0] - c[0]) * u[0] + (v[1] - c[1]) * u[1] + (v[2] - c[2]) * u[2]);
    pts.sort((a, b) => ang(a) - ang(b));
    const negN = [-N[0], -N[1], -N[2]];
    for (let k = 0; k < pts.length; k++) {
      const a = pts[k], b = pts[(k + 1) % pts.length];
      // negative side's cap faces +N (outward from that half); winding CCW seen from +N
      push(0, [[c, N], [a, N], [b, N]]);
      push(1, [[c, negN], [b, negN], [a, negN]]);
    }
  }
  return out.map((o, i) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(o.p.slice(0, o.len), 3));
    g.setAttribute("normal", new THREE.BufferAttribute(o.n.slice(0, o.len), 3));
    g.computeBoundingSphere();
    return { geo: g, sign: i === 0 ? -1 : 1 };
  });
}
