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
export const WALL_COLOR = "#ff2f2f", WALL_ALPHA = 0.5;

// ---- cut debris: slice a model by the real cut plane ----------------------
// The note and chain-head meshes are convex, so the cross-section is a
// convex polygon: clip every triangle against the plane (each side keeps
// its 1-2 sub-triangles with interpolated normals) and cap each half with
// a fan over the angle-sorted crossing points. Returns two non-indexed
// geometries [{geo, sign: -1}, {geo, sign: +1}] (sign −1 = the side the
// plane normal points away from), or null if the plane misses the mesh.
// `p`/`n` are in the geometry's own (scene-local) frame.
export function sliceGeometry(geo, p, n) {
  const pos = geo.attributes.position, nrm = geo.attributes.normal;
  const idx = geo.index ? geo.index.array : null;
  const nl = Math.hypot(n[0], n[1], n[2]);
  const N = [n[0] / nl, n[1] / nl, n[2] / nl];
  const sdist = (v) => (v[0] - p[0]) * N[0] + (v[1] - p[1]) * N[1] + (v[2] - p[2]) * N[2];
  const V = (i) => [pos.getX(i), pos.getY(i), pos.getZ(i)];
  const Nv = (i) => [nrm.getX(i), nrm.getY(i), nrm.getZ(i)];
  const lerp = (a, b, t) => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
  const out = [{ p: [], n: [] }, { p: [], n: [] }]; // 0 = negative side, 1 = positive
  const ring = []; // crossing points (duplicates welded below)
  const push = (side, tri) => {
    for (const [v, nn] of tri) { out[side].p.push(...v); out[side].n.push(...nn); }
  };
  const EPS = 1e-7;
  const ntri = idx ? idx.length / 3 : pos.count / 3;
  for (let t = 0; t < ntri; t++) {
    const ids = idx ? [idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]] : [3 * t, 3 * t + 1, 3 * t + 2];
    const vs = ids.map(V), ns = ids.map(Nv), ds = vs.map(sdist);
    const neg = [], posv = [];
    // Sutherland–Hodgman around the triangle, keeping both sides
    for (let i = 0; i < 3; i++) {
      const j = (i + 1) % 3;
      const a = [vs[i], ns[i]], b = [vs[j], ns[j]], da = ds[i], db = ds[j];
      if (da <= EPS) neg.push(a);
      if (da >= -EPS) posv.push(a);
      if ((da < -EPS && db > EPS) || (da > EPS && db < -EPS)) {
        const tt = da / (da - db);
        const x = [lerp(a[0], b[0], tt), lerp(a[1], b[1], tt)];
        neg.push(x); posv.push(x); ring.push(x[0]);
      }
    }
    for (const [side, poly] of [[0, neg], [1, posv]])
      for (let k = 1; k + 1 < poly.length; k++) push(side, [poly[0], poly[k], poly[k + 1]]);
  }
  if (!out[0].p.length || !out[1].p.length || ring.length < 3) return null;
  // cap: weld crossing points, sort by angle in the plane, fan both ways
  const key = (v) => v.map((x) => Math.round(x * 1e4)).join(",");
  const seen = new Set(), pts = [];
  for (const v of ring) { const k = key(v); if (!seen.has(k)) { seen.add(k); pts.push(v); } }
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
    g.setAttribute("position", new THREE.Float32BufferAttribute(o.p, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(o.n, 3));
    g.computeBoundingSphere();
    return { geo: g, sign: i === 0 ? -1 : 1 };
  });
}
