# MuJoCo oracle harness (P2b)

Validates `mjcf/body_v10.xml` (the MuJoCo oracle twin of SPEC-v10) and
emits the reference trajectories the v10 engine's tolerance-band tests
(P2/P4) compare against. MuJoCo's Newton solver is NOT our PGS-12 —
trajectory comparisons downstream use tolerance bands, never parity; this
harness only makes the oracle itself trustworthy.

## Setup / running

Uses the repo venv (`python/.venv`, README Quick start) with `mujoco`
(3.11.0, CPU) installed into it:

```
PY=../.venv/bin/python   # from this dir
$PY -m pip install mujoco     # once
$PY check_fk.py               # gate 1 - MANDATORY FIRST
$PY check_static.py           # gate 2
$PY check_contacts.py         # gate 4
$PY gen_reference.py          # writes refdata/*.json
```

Every check script exits 0 and prints `PASS: <name>` when green.
Model options: timestep 1/240, integrator **Euler**, gravity 9.81 on −y.

## check_fk.py — FK cross-check (mandatory first)

Proves the MJCF's stacked-joint declaration order reproduces the spec's
composition order (SPEC-v10 §1.3, Divergence #10: shoulder
yaw→pitch→twist, wrist roll→dev→bend, MuJoCo composing same-body hinges
in declaration order). A ~100-line reference FK written directly from the
spec tree table is compared against `mj_kinematics` on: neutral, every
actuated joint at ±half-limit (58 poses), and 100 seeded random in-limit
poses with random root pose — all 15 body positions AND orientation
matrices, plus fist centers and blade tips.

Expected: `max |ref - mujoco| ≈ 1.3e-15` (gate ≤ 1e-8). Also asserts the
§1.3 continuity anchors at neutral (head (0,1.70,−0.65), shoulders
(±0.2,1.47,−0.65), feet (±0.1,0.07,−0.65)).

## check_static.py — statics

- (a) total mass = **74.80 kg** exactly (gate ±0.01) — 0.50 kg hands
  (2026-10-03; the 0.35 kg-hand body, 74.50 kg, lives on branch
  `body/original`).
- (b1) informational: 2 s zero-torque free settle, per-joint drift
  printed. An unactuated skeleton has no position hold — gravity-loaded
  joints (horizontal arms, elbows, knee-buckle modes) fall to their
  limits; damping only slows the collapse. Expect drifts up to ~2.5 rad.
- (b2) gate: 2 s hold under constant `RNEA(q_ref, 0, 0)` — the FULL
  35-dof generalized force including the root wrench (the SPEC-v10 P2
  statics protocol). Joint-rows-only compensation is NOT an equilibrium
  of the welded closed chain (pelvis-pitch / knee-drop mobility modes are
  driven through the root rows). Expected drift **0.0** (gate < 5e-2).
- (c) gate: "muscle tone" spring-held stance (kp = τ_max/0.2 to neutral,
  near-critical damping, set in-memory only) settles in 4 s; foot weld
  vertical constraint forces must sum to body weight **733.79 N** ±5 %.
  Expected ≈ 732.8 N (366.4 per foot), weld violation ≈ 4.1 mm, droop
  ≈ 0.054 rad (shoulder pitch).

## check_contacts.py — contact wiring

- (1) the MJCF `<pair>` roster equals the spec's 55 live pairs — compiled
  set AND XML declaration order (= normative pair-id order 0–54; MuJoCo
  re-sorts pairs internally, so the XML text is checked for order). The
  11 excluded pairs are rebuilt from the spec table, not read back.
- (2)/(3) scripted static poses (in-limit): crossed wrists
  (yaw ±22°, pitch ±4°) → exactly `farm_l–farm_r`; forearm folded onto
  the chest (yaw 38°, pitch −25°, twist −30°, elbow 120°) → exactly
  `torso–farm_l`. "Exactly" = MuJoCo's contact set equals an independent
  capsule-distance computation on the reference FK (dist < margin 0.01),
  with per-pair gap agreement ≤ 1e-6 (measured ~1e-17 — same narrow
  phase).
- (4) dynamic press: springs hold each contact pose (gravity-compensated)
  while constant extra torque (±5 N·m shoulder yaw; +3 N·m elbow in B)
  drives the wrists across the midline for 1 s. Gates: max penetration
  < 2×slop = **0.01 m** (measured 0.0066 / 0.0081), required pair
  force-active (dist < margin−gap), no pair outside the live-55 ever
  appears.

## gen_reference.py — refdata/*.json

5 scenarios × 1 s × 240 substeps at h = 1/240 on the FREE-FLOATING
body (the MJCF's foot welds deactivated at load, `eq_active0 = 0`; the
floor never collides — every geom is contype 0 — so the body falls
freely), per-substep qpos(36)/qvel(35) at full JSON precision, plus the
applied pattern torque `tau29` per substep (actuated §1.4 order).
Regenerated 2026-09-27 when the engines dropped the v10 foot welds
(issue #2): `tests_oracle.rs` replays them with
`ConstraintSet::new(m, contacts = true, ground = false)`. (The welds stay
in the MJCF for the check_static.py statics gates, which are
MuJoCo-internal.)
Regenerated again 2026-10-03 for the 0.50 kg hands (the v63x2 body that
became main's; MJCF `hand_l`/`hand_r` mass 0.35 → 0.50, inertia
0.000264 → 0.000377). Regenerating the pre-change MJCF reproduces the
previous files to 1.6e-14 (qvel), so the diff is the body change alone.

| file | scenario |
|---|---|
| `neutral_hold` | stabilized stance, no pattern |
| `const_l_arm_pitch` | constant +60 N·m on l_arm_pitch |
| `sin_both_arms` | sinusoidal both-arm torques (R phase +π/2) |
| `leg_spine` | sinusoidal spine + leg pattern (hips antiphase) |
| `random_all29` | seeded random per-tick torque, all 29 joints |

**Stabilizer (read before consuming):** the raw v10 body relies on the
engine's hard ω caps; open-loop (caps absent) it is chaotically
unintegrable at h = 1/240 — measured during bring-up: MuJoCo NaNs within
~0.5 s under any sustained torque (limits off), or rattles the
near-undamped wrists at 100+ rad/s (limits on); constant full-RNEA
compensation is a knife-edge (perturbations e-fold in ~0.2 s). All
references therefore run on the muscle-tone body:
`τ_total_j = tau29_j + kp_j·(0 − q_j) − kd_j·q̇_j`, kp = τ_max/0.2,
kd near-critical (exact per-joint values in each file's `stabilizer`
block; kd REPLACES the spec damping b_j). A consumer engine must apply
the same law, then band-compare. Trajectories are tame by construction
(max |q̇| < 10 rad/s, excursions < 0.16 rad — inside all limits and ω
caps, so limit/clamp model differences cannot pollute the bands).

Note `qpos` rows are MuJoCo qpos order: root pos + root quat (wxyz) +
29 joints in MJCF TREE order (shoulders yaw,pitch,twist; wrists
roll,dev,bend) — see `qpos_joint_names`; map to actuated order before
comparing.

## MJCF fixes made by this phase (with check_static diagnosis)

1. **Weld anchors**: `<weld body1="foot_l"/>` with the default anchor
   (0,0,0) welds the world-origin-fixed point of the foot (anchor is in
   body2 = world frame) — a 0.66 m lever through softly-regularized
   rotation rows; the welds were mushy and the feet tore out under load.
   Fixed: `anchor` = each foot's world position.
2. **Weld solref/solimp**: defaults let the rotation rows slip under the
   75 kg body's collapse loads. Fixed: `solref="0.008333 1"` (= 2h) with
   high-impedance solimp.
3. **Pair solref/solimp default**: default contact softness let transient
   presses punch ~2 cm deep (> 2×slop). Fixed via `<default><pair
   solref="0.008333 1" solimp="0.95 0.99 0.001 0.5 2"/></default>`.

No kinematic, inertial, limit, damping, or actuator values were touched —
the FK cross-check passed against the original tree unchanged.
