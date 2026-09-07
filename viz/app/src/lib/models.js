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
//   headset  Headset.fbx at the scene's 0.35 scale: 0.29 m wide, pivot
//            at the HMD tracking point, visor 0.094 m ahead, straps
//            0.27 m behind
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
  "blade", "handle", "headset",
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
// Headset.mat base (0.53 gray), smoothness 0.
export const HEADSET_COLOR = "#878787";
// Wall.mat: (1, 0.184, 0.184) at alpha 0.5; WallEdges emission 3.
export const WALL_COLOR = "#ff2f2f", WALL_ALPHA = 0.5;
