// Node smoke test for the viewer's pure logic (src/lib/replay.js) against a
// real recorded v10 replay + a full-precision FK fixture generated from the
// Rust sim (test/fk_fixture.json). Run: npm test [path-to-replay.jsonl]
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import {
  DT, F_HEAD, F_L_FOOT, F_R_FOOT, FRAMES, GROUPS, HEAD, JOINT_LIMITS,
  JOINT_NAMES, KIND_BOMB, KIND_CHAIN_HEAD, KIND_CHAIN_LINK, LOOK_VERT,
  N_JOINTS, OMEGA_CAP, PAIRS, ROOT_POS0, TAU_MAX, Z_MISS, activeArcs,
  activeContacts, activePosts, aftermaths, bodyFK, clampQ29, counts,
  cutPlaneLocal, effortAt, fkSaber, footForces, frameQuatScene, gearTable,
  hitboxRot, maxScore, mirrorQ29, noteFrame, noteLam, noteRot, parseReplay,
  pendingNotes, poseQ36, qAt, saberAt, tauOf, tipOf, tipSpeed,
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
  p.forEach(({ z }) => assert.ok(z >= -Z_MISS - 1e-9 && z <= 12 + 1e-9));
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
