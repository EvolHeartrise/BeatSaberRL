// Node smoke test for the viewer's pure logic (src/lib/replay.js) against a
// real recorded v10 replay + a full-precision FK fixture generated from the
// Rust sim (test/fk_fixture.json). Run: npm test [path-to-replay.jsonl]
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import {
  DT, F_HEAD, F_L_FOOT, F_R_FOOT, FRAMES, GROUPS, HEAD, JOINT_LIMITS,
  JOINT_NAMES, KIND_BOMB, KIND_CHAIN_HEAD, KIND_CHAIN_LINK, LOOK_VERT,
  N_JOINTS, OMEGA_CAP, PAIRS, PLANE_Z0, ROOT_POS0, TAU_MAX, Z_FAR, Z_MISS,
  activeArcs, activeContacts, activePosts, aftermaths, bodyFK, clampQ29,
  counts, cutPlaneLocal, effortAt, fkSaber, footForces, frameQuatScene,
  gearTable, headZAt, hitboxRot, isHeadAnchored, maxScore, mirrorQ29,
  noteFrame, noteLam, noteRot, parseReplay, pendingNotes, poseQ36, qAt,
  rotQuatScene,
  saberAt, tauOf, tipOf, tipSpeed, trackOffset, trackOffsetAt,
  trackOffsetOfFK, viewerTrackOffset,
} from "../src/lib/replay.js";

const path = process.argv[2] ??
  "../../python/replays/v10_scripted_ep00_seed0.jsonl";
const rep = parseReplay(readFileSync(path, "utf8"));
console.log(`loaded: ${rep.chart.length} objects, ${rep.ticks.length} ticks, ` +
            `${rep.posts.length} post events, ${rep.arcs.length} arc pairs`);

// ---- body model tables -------------------------------------------------------
assert.equal(N_JOINTS, 29);
assert.equal(JOINT_NAMES.length, 29);
assert.equal(JOINT_LIMITS.length, 29);
assert.equal(TAU_MAX.length, 29);
assert.equal(OMEGA_CAP.length, 29);
assert.equal(FRAMES.length, 30, "30 frames incl. the welded head");
JOINT_LIMITS.forEach(([lo, hi], j) => assert.ok(lo < hi, `limits ${j}`));
// wrist tier caps at 40, everything else 16
for (const j of [7, 8, 9, 14, 15, 16]) assert.equal(OMEGA_CAP[j], 40);
assert.equal(OMEGA_CAP[0], 16);
assert.equal(OMEGA_CAP[20], 16);
// groups partition 0..28
{
  const all = GROUPS.flatMap((g) => g.joints).sort((a, b) => a - b);
  assert.deepEqual(all, Array.from({ length: 29 }, (_, i) => i));
}
// 55 live collision pairs (SPEC-v10 §2)
assert.equal(PAIRS.length, 55);
assert.deepEqual(PAIRS[0], [0, 4]);
assert.deepEqual(PAIRS[12], [1, 8]);
assert.deepEqual(PAIRS[54], [10, 11]);
// gear tables (constants.gear_table mirror)
assert.deepEqual(gearTable(3), [0, 1, -1]);
assert.deepEqual(gearTable(5), [0, 1, -1, 0.25, -0.25]);
assert.equal(gearTable(7).length, 7);
{
  const tau = tauOf([1, 0, 2, ...Array(26).fill(0)], 5);
  assert.equal(tau[0], TAU_MAX[0]);
  assert.equal(tau[1], 0);
  assert.equal(tau[2], -TAU_MAX[2]);
}

// ---- full-body FK vs the sim (exact fixture: unrounded q + sim FK) ----------
const fx = JSON.parse(readFileSync(
  new URL("./fk_fixture.json", import.meta.url), "utf8"));
let worstFK = 0;
for (const c of fx) {
  const fk = bodyFK(c.q);
  for (const hand of [0, 1]) {
    const s = fkSaber(fk, hand);
    for (let i = 0; i < 6; i++)
      worstFK = Math.max(worstFK, Math.abs(s[i] - c.sabers[hand][i]));
  }
  for (let i = 0; i < 3; i++) {
    worstFK = Math.max(worstFK,
      Math.abs(fk.p[F_HEAD][i] - c.head[i]),
      Math.abs(fk.p[F_L_FOOT][i] - c.foot_l[i]),
      Math.abs(fk.p[F_R_FOOT][i] - c.foot_r[i]));
  }
}
assert.ok(worstFK <= 1e-9,
  `JS bodyFK diverges from the sim FK: ${worstFK}`); // P6 gate (<= 1e-6)
console.log(`bodyFK vs sim fixture: worst |Δ| = ${worstFK.toExponential(2)}`);

// neutral pose reproduces the SPEC-v10 §1.1 anchors
{
  const fk = bodyFK([...ROOT_POS0, 1, 0, 0, 0, ...Array(29).fill(0)]);
  const close = (a, b) => a.every((v, i) => Math.abs(v - b[i]) < 1e-12);
  assert.ok(close(fk.p[F_HEAD], [0, 1.7, -0.65]), "head anchor");
  assert.ok(close(fk.p[F_L_FOOT], [-0.10, 0.07, -0.65]), "l foot weld");
  assert.ok(close(fk.p[F_R_FOOT], [0.10, 0.07, -0.65]), "r foot weld");
  const s = fkSaber(fk, 1);
  assert.ok(close(s, [0.2, 1.47, 0, 0, 0, 1]), "neutral r saber");
}

// ---- replay-level FK: logged sabers == bodyFK(logged q) (rounded 5dp/4dp) --
for (let k = 0; k < rep.ticks.length; k += 7) {
  const t = rep.ticks[k];
  const fk = bodyFK(t.q);
  for (const s of [0, 1]) {
    const mine = fkSaber(fk, s);
    for (let i = 0; i < 6; i++)
      assert.ok(Math.abs(mine[i] - t.sabers[s][i]) < 5e-4,
        `replay FK diverges at tick ${k} hand ${s} lane ${i}`);
  }
}

// ---- v10 tick fields ---------------------------------------------------------
for (const t of rep.ticks) {
  assert.equal(t.q.length, 36);
  assert.equal(t.twist.length, 6);
  assert.equal(t.omega.length, 29);
  assert.equal(t.action.length, 29);
  assert.ok(Array.isArray(t.contacts));
  for (const c of t.contacts) {
    assert.ok(c.pair >= 0 && c.pair < 55 && c.p.length === 3);
    assert.ok(c.j !== 0);
  }
}
// additive P6 fields present in fresh recordings
assert.ok(rep.ticks.every((t) => t.feet && t.feet.length === 2));
assert.ok(rep.ticks.every((t) => typeof t.effort === "number"));
{
  const ff = footForces(rep, 30);
  assert.ok(ff && ff.length === 2 && ff.every(Number.isFinite));
  assert.ok(effortAt(rep, 30) >= 0);
  // some tick actually carries the body weight through the welds
  const peak = Math.max(...rep.ticks.map((_, k) =>
    footForces(rep, k).reduce((a, b) => a + b, 0)));
  assert.ok(peak > 200, `no weight through the welds? peak ${peak} N`);
  // ...and absence degrades to null, not a crash
  const stripped = { ticks: [{ ...rep.ticks[0] }] };
  delete stripped.ticks[0].feet;
  delete stripped.ticks[0].effort;
  assert.equal(footForces(stripped, 0), null);
  assert.equal(effortAt(stripped, 0), null);
}

// contact markers: the recorded episode has self-collision events and
// they surface with sane radii/alphas at their times
{
  let seen = 0;
  for (let k = 0; k < rep.ticks.length; k++) {
    if (!rep.ticks[k].contacts.length) continue;
    const time = (rep.ticks[k].t + 1) * DT + 0.01;
    const act = activeContacts(rep, time);
    seen += act.length;
    act.forEach((c) => {
      assert.ok(c.radius >= 0.02 && c.radius <= 0.08 + 1e-9);
      assert.ok(c.alpha > 0 && c.alpha <= 1);
      assert.ok(c.p.length === 3 && c.p.every(Number.isFinite));
    });
  }
  assert.ok(seen > 0, "recorded contacts never surfaced as markers");
  console.log(`contact markers surfaced across the episode: ${seen}`);
}

// ---- chart + outcomes (unchanged v7+ semantics) ------------------------------
rep.chart.forEach((n, i) => {
  assert.ok(Number.isFinite(n.x) && Number.isFinite(n.y),
    `chart entry ${i} lacks explicit x/y`);
  assert.ok(n.y > 0 && n.y < 2.5, `implausible object height at ${i}: ${n.y}`);
  assert.ok([0, 1, 2, 3].includes(n.kind), `unknown kind at ${i}`);
});

const last = rep.ticks.length - 1;
const c = counts(rep, last);
const scoreable = rep.chart.filter((n) => n.kind !== KIND_BOMB).length;
const nBombs = rep.chart.length - scoreable;
assert.equal(c.good + c.bad + c.miss, scoreable,
  "every scoreable object resolves exactly once");
assert.ok(c.bombs <= nBombs, "more bomb hits than bombs");
const evPts = rep.ticks.flatMap((t) => t.events).reduce((a, e) =>
  a + (e.cut ? e.cut.points || 0 : 0) + (e.post ? e.post.points : 0), 0);
assert.ok(Math.abs(c.score - evPts) < 1e-6, "counts.score matches events");
assert.ok(c.score <= maxScore(rep.chart), "score exceeds the per-kind max");
console.log(`counts: good ${c.good} bad ${c.bad} miss ${c.miss} ` +
            `bombs ${c.bombs}/${nBombs} score ${c.score.toFixed(1)}` +
            `/${maxScore(rep.chart)} return ${c.reward.toFixed(2)}`);

// interpolation + blade math stay finite over the whole episode
for (let k = 0; k < last; k += 7) {
  const q = qAt(rep, k, 0.5);
  assert.ok(q.every(Number.isFinite), `NaN in qAt at ${k}`);
  const qn = Math.hypot(q[3], q[4], q[5], q[6]);
  assert.ok(Math.abs(qn - 1) < 1e-9, `qAt quat not normalized at ${k}`);
  for (const s of [0, 1]) {
    const sab = saberAt(rep, k, 0.5, s);
    const tip = tipOf(sab);
    assert.ok([...sab, ...tip, tipSpeed(rep, k, s)].every(Number.isFinite),
      `NaN in saber math at tick ${k}`);
    assert.ok(Math.abs(Math.hypot(sab[3], sab[4], sab[5]) - 1) < 1e-9,
      `saberAt blade not unit at ${k}`);
  }
}

// derived render lists behave at several times
for (const frac of [0, 0.3, 0.6, 0.9, 1.0]) {
  const t = frac * last * DT;
  const p = pendingNotes(rep, t), a = aftermaths(rep, t),
    po = activePosts(rep, t), ar = activeArcs(rep, t);
  assert.ok(p.length + a.length >= 0 && po.length >= 0 && ar.length >= 0);
  // the chart-time z is what the spawn/despawn bounds are taken on
  p.forEach(({ zBase }) =>
    assert.ok(zBase >= -Z_MISS - 1e-9 && zBase <= Z_FAR + 1e-9));
}

// arc pairs are sane: same color, tail after head, within the window
rep.arcs.forEach(({ head, tail }) => {
  const h = rep.chart[head], t = rep.chart[tail];
  assert.ok(h.fpost && t.fpre, "arc pair without fpost/fpre grants");
  assert.equal(h.color, t.color, "arc pair colors differ");
  assert.ok(t.t > h.t && t.t - h.t <= 2.5, "arc pair outside the window");
});

// actual-cut-plane reconstruction: every good cut yields a sane plane
// (unit normal, point inside the cube) or a declared-degenerate null
let planes = 0;
rep.chart.forEach((n, i) => {
  const o = rep.outcomes[i];
  if (!o || o.kind !== "good") return;
  const cp = cutPlaneLocal(rep, i);
  if (!cp) return; // pure stab: canonical fallback
  planes++;
  const nl = Math.hypot(...cp.n);
  assert.ok(Math.abs(nl - 1) < 1e-9, `non-unit cut normal at note ${i}`);
  assert.ok(cp.p.every((x) => Math.abs(x) <= 0.25 * 0.95 + 1e-12 &&
    Number.isFinite(x)), `cut point outside cube at note ${i}`);
  assert.ok([...cp.v, ...cp.b].every(Number.isFinite));
});
assert.ok(planes > 0, "no cut planes reconstructed at all");
console.log(`cut planes reconstructed for ${planes} good cuts`);

// ---- manual pose mode pure logic --------------------------------------------
// clamping pins out-of-range joints to the anatomical bounds
clampQ29(Array(29).fill(99)).forEach((v, j) =>
  assert.equal(v, JOINT_LIMITS[j][1], `clamp hi joint ${j}`));
clampQ29(Array(29).fill(-99)).forEach((v, j) =>
  assert.equal(v, JOINT_LIMITS[j][0], `clamp lo joint ${j}`));
// in-range poses pass through untouched
{
  const mid = JOINT_LIMITS.map(([lo, hi]) => (lo + hi) / 2);
  assert.deepEqual(clampQ29(mid), mid);
}
// mirror: involution, legal -> legal, wrists mirror across the midplane
{
  const q = clampQ29(JOINT_LIMITS.map(
    ([lo, hi], j) => lo + (0.15 + 0.7 * ((j * 37) % 10) / 10) * (hi - lo)));
  const m = mirrorQ29(q);
  assert.deepEqual(mirrorQ29(m).map((v) => +v.toFixed(12)),
    q.map((v) => +v.toFixed(12)), "mirror is not an involution");
  assert.deepEqual(clampQ29(m), m, "mirrored pose escapes limits");
  const pose = (qq) => ({ q: qq,
    root: { x: 0, y: 0.94, z: -0.65, yaw: 0, pitch: 0, roll: 0 } });
  const fa = bodyFK(poseQ36(pose(q))), fb = bodyFK(poseQ36(pose(m)));
  const wa = fkSaber(fa, 0), wb = fkSaber(fb, 1);
  assert.ok(Math.abs(wa[0] + wb[0]) < 1e-12 &&
    Math.abs(wa[1] - wb[1]) < 1e-12 && Math.abs(wa[2] - wb[2]) < 1e-12,
    "mirrored L wrist != mirrored R wrist");
}
// poseQ36: root yaw/pitch/roll compose to a unit quaternion; neutral is
// the reference pose
{
  const q36 = poseQ36({ q: Array(29).fill(0),
    root: { x: 0.1, y: 1.0, z: -0.5, yaw: 0.3, pitch: -0.2, roll: 0.1 } });
  assert.ok(Math.abs(Math.hypot(q36[3], q36[4], q36[5], q36[6]) - 1) < 1e-12);
  const neutral = poseQ36({ q: Array(29).fill(0),
    root: { x: 0, y: 0.94, z: -0.65, yaw: 0, pitch: 0, roll: 0 } });
  assert.deepEqual(neutral.slice(0, 7), [0, 0.94, -0.65, 1, 0, 0, 0]);
}
console.log("pose-mode helpers OK");

// env v7.4 hitbox rotation (unchanged)
const deg = Math.PI / 180;
assert.equal(hitboxRot({ kind: 0, any: false, angle: 270 * deg }), 0,
  "down-swing note is axis-aligned");
assert.ok(Math.abs(hitboxRot({ kind: 0, any: false, angle: 315 * deg })
  - 45 * deg) < 1e-12, "diagonal arrow note rotates by angle - 270°");
assert.equal(hitboxRot({ kind: 0, any: true, angle: 315 * deg }), 0,
  "dot note hitbox never rotates");
assert.ok(Math.abs(hitboxRot({ kind: KIND_CHAIN_LINK, any: true,
  angle: Math.PI }) - (-Math.PI / 2)) < 1e-12,
  "chain link rotates by its tangent despite any=true");
assert.equal(hitboxRot({ kind: KIND_BOMB, any: false, angle: 1 }), 0,
  "bomb hitbox never rotates");
for (const n of [{ kind: 0, any: false, angle: 200 * deg },
                 { kind: 2, any: false, angle: 45 * deg },
                 { kind: 0, any: true, angle: 0 }])
  assert.equal(hitboxRot(n), noteRot(n), "hitboxRot/noteRot mismatch");

// env v7.6 look-at-player note frame (unchanged; head arg optional)
const close = (a, b, m) => a.forEach((v, i) =>
  assert.ok(Math.abs(v - b[i]) < 1e-12, `${m}: [${a}] vs [${b}]`));
for (const psi of [0, 45 * deg, -120 * deg, Math.PI]) {
  const f = noteFrame([0.3, 1.9, 6], psi, 0);
  close(f[0], [Math.cos(psi), Math.sin(psi), 0], "lam=0 x'");
  close(f[1], [-Math.sin(psi), Math.cos(psi), 0], "lam=0 y'");
  close(f[2], [0, 0, 1], "lam=0 z'");
  const qa = frameQuatScene(f);
  const qe = [0, 0, Math.sin(psi / 2), Math.cos(psi / 2)];
  const qd = Math.abs(qa.reduce((s, v, i) => s + v * qe[i], 0));
  assert.ok(Math.abs(qd - 1) < 1e-12, `lam=0 quat: [${qa}] vs [${qe}]`);
}
{
  const p = [0.9, 1.4, 0];
  const f = noteFrame(p, 0, 1);
  const hp = [HEAD[0], HEAD[1] + LOOK_VERT * (p[1] - HEAD[1]), HEAD[2]];
  const d = [p[0] - hp[0], p[1] - hp[1], p[2] - hp[2]];
  const dn = Math.hypot(...d);
  close(f[2], d.map((v) => v / dn), "lam=1 forward = head->note ray");
  assert.ok(f[2][0] > 0.5, "edge note yaws toward the head");
}
{
  const f = noteFrame([-0.9, 0.85, 3.1], 30 * deg, 0.61);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      assert.ok(Math.abs(dot(f[i], f[j]) - (i === j ? 1 : 0)) < 1e-12,
        "frame not orthonormal");
  const cx = [f[0][1] * f[1][2] - f[0][2] * f[1][1],
              f[0][2] * f[1][0] - f[0][0] * f[1][2],
              f[0][0] * f[1][1] - f[0][1] * f[1][0]];
  close(cx, f[2], "frame not right-handed");
}
assert.equal(noteLam({ kind: 0 }, 0), 1);
assert.equal(noteLam({ kind: 0 }, 4), 0.5);
assert.equal(noteLam({ kind: 0 }, 8), 0);
assert.equal(noteLam({ kind: 0 }, 11), 0);
assert.equal(noteLam({ kind: 0 }, -0.5), 1);
for (const kind of [KIND_BOMB, KIND_CHAIN_HEAD, KIND_CHAIN_LINK])
  assert.equal(noteLam({ kind }, 1), 0, "lam leaks past plain notes");

// ============================================================================
// env v15 (unilateral-contact spec §10): HEAD-ANCHORED NOTE TRACK.
//   note_z(t) = (hit_time - t) * speed + z_off(t),  z_off = head_z - PLANE_Z0
// The header field "track_anchor": "head" is ADDITIVE and the gate is
// STRICT: absent (every pre-v15 recording) means the old fixed track.
// ============================================================================
assert.equal(PLANE_Z0, -0.65);
assert.equal(Z_MISS, -PLANE_Z0, "Z_MISS is the reference plane's depth");

// ---- (a) LEGACY replay (no track_anchor): NO offset, ever -------------------
{
  assert.equal(rep.trackAnchor, null, "legacy replay must carry no anchor");
  assert.equal(isHeadAnchored(rep), false);
  // the legacy body's head moves plenty; none of it may reach the track
  const hz = rep.ticks.map((_, k) => headZAt(rep, k));
  const span = Math.max(...hz) - Math.min(...hz);
  assert.ok(span > 0.002, `legacy head barely moves (${span} m) — weak test`);
  for (let k = 0; k < rep.ticks.length; k += 11) {
    assert.equal(trackOffset(rep, k), 0, `legacy offset at tick ${k}`);
    assert.equal(trackOffset(rep, k, 0.5), 0);
    assert.equal(trackOffsetAt(rep, k * DT), 0);
  }
  // ...so the rendered z IS the chart-time z, for every flying object
  for (const frac of [0, 0.25, 0.5, 0.75, 1.0]) {
    const t = frac * last * DT;
    pendingNotes(rep, t, trackOffsetAt(rep, t)).forEach(({ n, z, zBase }) => {
      assert.equal(z, zBase, "legacy note picked up an offset");
      assert.equal(zBase, (n.t - t) * n.speed);
    });
    aftermaths(rep, t).forEach((a) => assert.equal(a.zOff, 0));
    activePosts(rep, t).forEach((p) => assert.equal(p.zOff, 0));
    const off = activeArcs(rep, t, trackOffsetAt(rep, t));
    activeArcs(rep, t).forEach((a, i) => {
      assert.equal(a.a[2], off[i].a[2]);
      assert.equal(a.b[2], off[i].b[2]);
    });
  }
  // the viewer-level gate agrees, in replay AND pose mode
  const fk = bodyFK(rep.ticks[10].q);
  assert.ok(Math.abs(trackOffsetOfFK(fk)) > 1e-6, "flat head — weak test");
  assert.equal(viewerTrackOffset(rep, false, fk), 0);
  assert.equal(viewerTrackOffset(rep, true, fk), 0,
    "pose mode must not resurrect the offset for a legacy replay");
  console.log(`legacy replay: fixed track held over ${rep.ticks.length} ` +
    `ticks (head z span ${span.toFixed(3)} m)`);
}

// ---- (b) HEAD-ANCHORED replay: offset applied every tick --------------------
// Prefer a real v15 recording; fall back to the legacy one with the
// ADDITIVE field spliced into its header (exactly what the writer adds).
const V15_PATH = new URL(
  "../../../python/replays/v15_probe_toppp01_delightful-introduction_1pp.jsonl",
  import.meta.url);
const legacyText = readFileSync(path, "utf8");
const anchored = (text, anchor = "head") => {
  const lines = text.split("\n");
  const h = JSON.parse(lines[0]);
  h.header.track_anchor = anchor;
  lines[0] = JSON.stringify(h);
  return lines.join("\n");
};
let v15, v15Src;
try {
  v15 = parseReplay(readFileSync(V15_PATH, "utf8"));
  v15Src = "recorded v15 probe";
} catch {
  v15 = parseReplay(anchored(legacyText));
  v15Src = "legacy replay + spliced track_anchor";
}
{
  assert.equal(v15.trackAnchor, "head");
  assert.ok(isHeadAnchored(v15));

  // z_off IS head_z - PLANE_Z0, at every tick, from the SAME full-body FK
  // the viewer draws the head with (fk.p[F_HEAD]) — no separate model.
  // Exact against the drawn pose (qAt renormalizes the logged quaternion);
  // against the raw logged q it agrees to the 5-dp quat rounding.
  let lo = Infinity, hi = -Infinity, worstHead = 0;
  for (let k = 0; k < v15.ticks.length; k++) {
    const drawn = bodyFK(qAt(v15, k, 0)).p[F_HEAD][2];
    assert.equal(trackOffset(v15, k), drawn - PLANE_Z0,
      `z_off != head_z - PLANE_Z0 at tick ${k}`);
    assert.equal(headZAt(v15, k), drawn);
    worstHead = Math.max(worstHead,
      Math.abs(drawn - bodyFK(v15.ticks[k].q).p[F_HEAD][2]));
    lo = Math.min(lo, trackOffset(v15, k));
    hi = Math.max(hi, trackOffset(v15, k));
  }
  assert.ok(worstHead < 1e-4,
    `drawn head z drifts from the logged head z: ${worstHead}`);
  // the Rust gate demands the offset actually be exercised (> 2 mm)
  assert.ok(hi - lo > 0.002,
    `head-anchored offset never moves (${hi - lo} m)`);

  // interpolated ticks come from the interpolated pose, and stay bracketed
  for (let k = 0; k + 1 < v15.ticks.length; k += 7) {
    const a = trackOffset(v15, k), b = trackOffset(v15, k + 1);
    const m = trackOffset(v15, k, 0.5);
    assert.ok(m >= Math.min(a, b) - 1e-9 && m <= Math.max(a, b) + 1e-9,
      `interpolated z_off outside its bracket at ${k}`);
    assert.equal(m, bodyFK(qAt(v15, k, 0.5)).p[F_HEAD][2] - PLANE_Z0);
  }

  // rendered z = base + z_off for EVERY flying object at several ticks
  // with visibly different offsets; the chart-time z is untouched.
  let checked = 0;
  const offsets = new Set();
  for (let k = 0; k < v15.ticks.length; k += 3) {
    const t = k * DT;
    const off = trackOffsetAt(v15, t);
    offsets.add(off.toFixed(4));
    for (const { n, z, zBase } of pendingNotes(v15, t, off)) {
      assert.equal(zBase, (n.t - t) * n.speed, `base z moved at tick ${k}`);
      assert.equal(z, zBase + off, `note z != base + z_off at tick ${k}`);
      // the sim's live miss plane is world z = head_z; nothing renders past it
      assert.ok(z >= headZAt(v15, k) - 1e-9, `note behind the head at ${k}`);
      checked++;
    }
  }
  assert.ok(checked > 20, `only ${checked} note-ticks checked`);
  assert.ok(offsets.size > 3, "offset never varied across the sampled ticks");

  // arcs ride the track too, and the frozen aftermath/post offsets are the
  // ones from the frame the effect spawns on (tick o.tick + 1)
  for (let k = 0; k < v15.ticks.length; k += 5) {
    const t = k * DT, off = trackOffsetAt(v15, t);
    const base = activeArcs(v15, t), moved = activeArcs(v15, t, off);
    base.forEach((a, i) => {
      assert.ok(Math.abs(moved[i].a[2] - (a.a[2] + off)) < 1e-12);
      assert.ok(Math.abs(moved[i].b[2] - (a.b[2] + off)) < 1e-12);
    });
    aftermaths(v15, t).forEach((a) =>
      assert.equal(a.zOff, trackOffset(v15, a.o.tick + 1),
        "aftermath offset not frozen at its spawn frame"));
    activePosts(v15, t).forEach((p) =>
      assert.equal(p.zOff, trackOffset(v15, p.tick + 1)));
  }

  // §10b: miss timing is a pure function of CHART time — every recorded
  // miss fires within one tick of the note's base z crossing PLANE_Z0,
  // whatever the body was doing. This is the property the viewer relies
  // on to keep culling on zBase.
  let misses = 0;
  v15.chart.forEach((n, i) => {
    const o = v15.outcomes[i];
    if (!o || o.kind !== "miss") return;
    misses++;
    const zb = (n.t - (o.tick + 1) * DT) * n.speed;
    assert.ok(zb < PLANE_Z0 + 1e-9 && zb >= PLANE_Z0 - n.speed * DT - 1e-9,
      `miss ${i} fired at base z ${zb}, not at the PLANE_Z0 crossing`);
  });

  // ---- A/B: the SAME ticks with the flag stripped render on the old track
  const stripped = parseReplay(anchored(
    v15Src === "recorded v15 probe"
      ? readFileSync(V15_PATH, "utf8") : legacyText, "world"));
  assert.equal(stripped.trackAnchor, "world");
  assert.equal(isHeadAnchored(stripped), false,
    "gate must accept ONLY the exact value \"head\"");
  for (let k = 0; k < stripped.ticks.length; k += 7) {
    assert.equal(trackOffset(stripped, k), 0);
    const t = k * DT;
    const a = pendingNotes(v15, t, trackOffsetAt(v15, t));
    const b = pendingNotes(stripped, t, trackOffsetAt(stripped, t));
    assert.equal(a.length, b.length);
    a.forEach((e, i) => {
      assert.equal(e.zBase, b[i].zBase, "chart-time z differs across the gate");
      assert.equal(b[i].z, b[i].zBase, "unflagged replay picked up an offset");
    });
  }
  console.log(`v15 (${v15Src}): z_off in [${lo.toFixed(3)}, ${hi.toFixed(3)}] ` +
    `m over ${v15.ticks.length} ticks, ${checked} note-ticks pinned, ` +
    `${misses} misses at the PLANE_Z0 crossing`);
}

// ---- (c) synthetic replay with an ANALYTICALLY known head trajectory --------
// Root z ramps and the spine pitches, so head z = root_z + 0.66 sin(pitch)
// (the head is welded 0.10 + 0.66 above the root through the spine chain).
{
  const chart = [
    { t: 1.0, x: 0.3, y: 1.4, color: 0, angle: 4.712389, speed: 5.0,
      any: false, kind: 0, fpre: false, fpost: false },
    { t: 1.5, x: -0.9, y: 1.9, color: 1, angle: 0.0, speed: 3.0,
      any: true, kind: 1, fpre: false, fpost: false },
  ];
  const pitches = [0, 0.2, -0.35, 0.45];
  const rootZ = [-0.65, -0.5, -0.9, -0.2];
  const ticks = pitches.map((pitch, k) => ({
    t: k,
    q: [0, 0.94, rootZ[k], 1, 0, 0, 0, pitch, ...Array(28).fill(0)],
    events: [], reward: 0,
  }));
  const build = (header) => parseReplay([JSON.stringify({ header }),
    ...ticks.map((t) => JSON.stringify(t))].join("\n"));
  const base = { version: "v10", config: { gears: 5 }, chart };
  const withFlag = build({ ...base, track_anchor: "head" });
  const noFlag = build(base);

  pitches.forEach((pitch, k) => {
    const want = rootZ[k] + 0.66 * Math.sin(pitch) - PLANE_Z0;
    assert.ok(Math.abs(trackOffset(withFlag, k) - want) < 1e-12,
      `synthetic z_off at tick ${k}: ${trackOffset(withFlag, k)} vs ${want}`);
    assert.equal(trackOffset(noFlag, k), 0);
    const t = k * DT;
    const on = pendingNotes(withFlag, t, trackOffsetAt(withFlag, t));
    const off = pendingNotes(noFlag, t, trackOffsetAt(noFlag, t));
    assert.equal(on.length, off.length);
    on.forEach((e, i) => {
      assert.ok(Math.abs(e.z - (e.zBase + want)) < 1e-12,
        `synthetic note z at tick ${k}`);
      assert.equal(off[i].z, off[i].zBase);
      // x / y are body-independent (spec §10a): lanes never move
      assert.equal(e.n.x, off[i].n.x);
      assert.equal(e.n.y, off[i].n.y);
    });
  });
  // pose mode with no replay loaded follows the drawn head; without it, 0
  const fk = bodyFK(withFlag.ticks[3].q);
  assert.ok(Math.abs(viewerTrackOffset(null, true, fk)
    - (rootZ[3] + 0.66 * Math.sin(pitches[3]) - PLANE_Z0)) < 1e-12);
  assert.equal(viewerTrackOffset(null, false, fk), 0);
  assert.equal(viewerTrackOffset(withFlag, false, null), 0);
  console.log("synthetic head-anchored track: analytic z_off matches");
}

// ---- rotQuatScene: the body-rotation twin of frameQuatScene -------------
{
  const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  assert.deepEqual(rotQuatScene(I), [0, 0, 0, 1]);
  // +90° yaw about world y (gaze +z -> +x): in the z-flipped scene that
  // is a -90° yaw, quaternion (0, -sin 45°, 0, cos 45°)
  const c = Math.SQRT1_2;
  const Ry = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]];
  const q = rotQuatScene(Ry);
  [0, -c, 0, c].forEach((v, i) =>
    assert.ok(Math.abs(q[i] - v) < 1e-12, `yaw q[${i}] = ${q[i]}`));
  // and on a real head frame it is frameQuatScene of the matrix's columns
  const m = bodyFK(rep.ticks[Math.min(100, rep.ticks.length - 1)].q).r[F_HEAD];
  assert.deepEqual(rotQuatScene(m), frameQuatScene(
    [0, 1, 2].map((j) => [m[0][j], m[1][j], m[2][j]])));
  console.log("rotQuatScene: identity, yaw, head frame OK");
}

// replays from retired parameterizations are rejected with the friendly
// "re-record" message (v10 viewers reject EVERYTHING else — house rule)
assert.throws(
  () => parseReplay('{"header": {"version": 8, "config": {}, "chart": []}}'),
  /version 8.*re-record/s);
assert.throws(
  () => parseReplay('{"header": {"version": 6, "config": {}, "chart": []}}'),
  /version 6/);
assert.throws(
  () => parseReplay('{"header": {"version": "v9", "config": {}, "chart": []}}'),
  /version v9/);

console.log("replay.js smoke test OK");
