# Beat Arena — env v10 contract (SPEC-v10)

Status: **FROZEN at P0** (2026-07-31). The contract for phases P1–P6 of the
full-body / full-dynamics / self-collision build
(`research/2026-07-31-body-v10-full-dynamics-proposal.md`, all 8 open
decisions locked to their recommended defaults — see §8). Every number in
this file is normative as written: implementations use these literals, they
do NOT re-derive them. Scoring, notes, judgment, chart generation, and the
16×14 note obs block are UNCHANGED from SPEC.md (env v7 judgment rules, v9
obs note semantics); this file specifies only what v10 changes. The MuJoCo
oracle twin is `mjcf/body_v10.xml`.

Units: meters, kilograms, seconds, radians, N·m. World: y up, floor y = 0,
hit plane z = 0, notes approach from +z. Gravity **9.81** along −y.

## 1. Kinematic tree

Floating base + 29 actuated revolute DoF = 35 velocity coordinates,
36 position coordinates (root quaternion). Reference (neutral) pose: root at
`(0, 0.94, −0.65)`, identity orientation, all 29 joints = 0, all velocities
= 0. In this pose the arm chains coincide with the v9 body EXACTLY (see
§1.3) and the feet stand flat on y = 0.

### 1.1 Geometry derivation (heights, all at neutral)

Anthropometry scaled to reproduce the v9 anchors (shoulder 1.47, head 1.70,
stature ≈ 1.79 with the 0.09 head sphere):

| level | y | from |
|---|---|---|
| sole | 0.000 | floor |
| ankle | 0.070 | foot height 0.07 |
| knee | 0.500 | shank 0.43 |
| hip | 0.940 | thigh 0.44 |
| root (pelvis frame) | 0.940 | at hip height, midline |
| waist (spine joint) | 1.040 | pelvis→waist riser 0.10 |
| shoulder | 1.470 | chest column 0.43, lateral ±0.20 |
| head center | 1.700 | head riser 0.66 above waist |

Check: 0.07 + 0.43 + 0.44 = 0.94 (hip); + 0.10 = 1.04 (waist); + 0.43 =
**1.47** (shoulder, exact); waist + 0.66 = **1.70** (head, exact). All
body-column x = 0, z = −0.65 at neutral; shoulders (±0.2, 1.47, −0.65) ✓;
fractions of stature: hip 0.53 H, knee 0.28 H, shoulder 0.83 H — standard.
Hip separation 0.20 m (hip joints at x ±0.10); **feet weld at x ±0.10**
(stance width = hip width). Foot: ankle at (±0.10, 0.07, −0.65), box sole
flat on y = 0, toes toward +z.

### 1.2 Bodies

Masses: de Leva (1996) male segment fractions × 75 kg (trunk split: pelvis =
lower trunk 11.17 %, chest = upper+mid trunk 32.29 %). COM at the capsule /
box / sphere centroid (NOT de Leva's proximal-offset COMs — see
Divergences). Inertia = homogeneous solid of the stated geometry, principal
axes aligned with the body frame. Hand body = 0.46 kg fist sphere + 0.15 kg
controller point mass at the fist center (grip): total 0.61 kg, the point
mass adds no inertia about the common center. Blade: massless, NO collider.
Body mass total **74.72 + 2×0.15 = 75.02 kg**.

Body frames: `anchor` = frame origin offset in the PARENT frame. Limb
frames at neutral are world-aligned (all joint rotations zero ⇒ every body
frame is a pure translation of world).

| idx | body | parent | anchor (in parent) | mass | COM (in body) | inertia diag (Ix,Iy,Iz) | geometry (in body) | collider |
|---|---|---|---|---|---|---|---|---|
| 0 | pelvis | — (root, free) | world (0, 0.94, −0.65) | 8.40 | (0,0,0) | 0.038640, 0.101640, 0.101640 | capsule x: (−0.10,0,0)–(0.10,0,0) r 0.10 | no |
| 1 | chest | 0 | (0, 0.10, 0) | 24.20 | (0, 0.215, 0) | 0.957507, 0.254964, 0.957507 | capsule y: (0,0,0)–(0,0.43,0) r 0.15 | yes (torso) |
| 2 | head | 1 (WELD) | (0, 0.66, 0) | 5.20 | (0,0,0) | 0.016848 ×3 | sphere r 0.09 | yes |
| 3 | uarm_l | 1 | (−0.20, 0.43, 0) | 2.00 | (0, 0, 0.175) | 0.029222, 0.029222, 0.001966 | capsule z: (0,0,0)–(0,0,0.35) r 0.045 | yes |
| 4 | farm_l | 3 | (0, 0, 0.35) | 1.20 | (0, 0, 0.15) | 0.013056, 0.013056, 0.000931 | capsule z: (0,0,0)–(0,0,0.30) r 0.04 | yes |
| 5 | hand_l | 4 | (0, 0, 0.30) | 0.61 | (0, 0, 0.06) | 0.000460 ×3 | fist sphere at (0,0,0.06) r 0.05; blade segment (0,0,0)→(0,0,1.0) massless | fist yes; blade NO |
| 6 | uarm_r | 1 | (+0.20, 0.43, 0) | 2.00 | as uarm_l | as uarm_l | as uarm_l | yes |
| 7 | farm_r | 6 | (0, 0, 0.35) | 1.20 | as farm_l | as farm_l | as farm_l | yes |
| 8 | hand_r | 7 | (0, 0, 0.30) | 0.61 | as hand_l | as hand_l | as hand_l | fist yes |
| 9 | thigh_l | 0 | (−0.10, 0, 0) | 10.60 | (0, −0.22, 0) | 0.250198, 0.018493, 0.250198 | capsule y: (0,0,0)–(0,−0.44,0) r 0.06 | yes |
| 10 | shank_l | 9 | (0, −0.44, 0) | 3.20 | (0, −0.215, 0) | 0.065964, 0.003161, 0.065964 | capsule y: (0,0,0)–(0,−0.43,0) r 0.045 | yes |
| 11 | foot_l | 10 | (0, −0.43, 0) | 1.00 | (0, −0.035, 0.05) | 0.005617, 0.005883, 0.001083 | box center (0,−0.035,0.05), half (0.045, 0.035, 0.125) | **no** (welded) |
| 12 | thigh_r | 0 | (+0.10, 0, 0) | 10.60 | as thigh_l | as thigh_l | as thigh_l | yes |
| 13 | shank_r | 12 | (0, −0.44, 0) | 3.20 | as shank_l | as shank_l | as shank_l | yes |
| 14 | foot_r | 13 | (0, −0.43, 0) | 1.00 | as foot_l | as foot_l | as foot_l | **no** (welded) |

Head is WELDED (no neck DoF): implementations may keep it a separate fixed
body or fold it into the chest composite — identical dynamics either way.

### 1.3 Joint rotation semantics — v9 continuity (HARD requirement)

Multi-DoF joints are stacked coincident revolutes; child-body orientation =
parent orientation × the products below (intrinsic composition, leftmost
applied first / outermost). **Stacking order ≠ actuated-vector order for
the shoulder and wrist clusters** — both are normative:

- **Spine** (pelvis→chest, anchor at waist):
  `R_spine = R_x(pitch) · R_y(yaw) · R_z(roll)`
  (+pitch = lean forward toward +z, +yaw = turn toward +x, +roll = lean
  toward −x). Stacking order pitch→yaw→roll = actuated order.
- **Shoulder** (chest→uarm): stacking order **yaw → pitch → twist**;
  actuated order pitch, yaw, twist.
  `M = R_y(yaw) · R_x(−pitch) · R_z(twist)` — implemented as revolute axes
  (0,1,0) angle yaw, (−1,0,0) angle pitch, (0,0,1) angle twist. Upper arm
  extends along local +z (length 0.35). This is v9's arm frame verbatim.
- **Elbow** (uarm→farm, anchor (0,0,0.35)): axis (−1,0,0), angle elbow ∈
  [0, 150°]: forearm frame `M_f = M · R_x(−elbow)`, forearm along local +z
  (length 0.30) ⇒ forearm dir = `M·(0, sin e, cos e)`. v9 verbatim.
- **Wrist** (farm→hand, anchor (0,0,0.30)): stacking order
  **roll → dev → bend**; actuated order roll, bend, dev.
  `R_wrist = R_z(roll) · R_y(dev) · R_x(−bend)` — axes (0,0,1) roll,
  (0,1,0) dev, (−1,0,0) bend. Blade = hand-frame +z unit vector
  = `M_f · R_z(ψ)·R_y(v)·R_x(−φ) · e_z` — v9 verbatim. Wrist world position
  = hand frame origin. Blade tip = wrist + 1.0·blade.
- **Hip** (pelvis→thigh, anchor (±0.10,0,0)): stacking order
  pitch→yaw→roll = actuated order.
  `R_hip = R_x(−pitch) · R_y(yaw) · R_z(roll)` — axes (−1,0,0), (0,1,0),
  (0,0,1). Thigh extends along local −y. +pitch = flexion (knee toward +z);
  +yaw = external rotation for the RIGHT leg (toes toward +x), internal for
  the left; +roll = knee toward +x (abduction right / adduction left).
- **Knee** (thigh→shank, anchor (0,−0.44,0)): axis (+1,0,0), angle ∈
  [0, 150°], + = flexion (heel toward −z). Shank along local −y.
- **Ankle** (shank→foot, anchor (0,−0.43,0)): stacking order pitch→roll =
  actuated order. Axes (−1,0,0) pitch (+ = dorsiflexion, toes up), (0,0,1)
  roll. Foot box along local +z (toes forward).

**Continuity theorem (the P1 gate)**: with the root at the reference pose
and spine = 0, chest is world-aligned, shoulder anchors sit at
(∓0.2, 1.47, −0.65) [left, right], and the 7-joint arm chains above
reproduce `rust/src/lib.rs fk()` / `kinematics.py fk()` EXACTLY — same
matrix products, same segment lengths (0.35 / 0.30 / blade 1.0), same
per-hand joint limits. Head center at neutral = (0, 1.70, −0.65) = v9
`HEAD` verbatim.

### 1.4 Actuated joint table

Actuated ORDER (indices 0–28 of q₂₉ / q̇₂₉ / action / τ; the root's 3+4
position and 6 velocity coords are separate and unactuated):
**spine[pitch,yaw,roll], arm_l[pitch,yaw,twist,elbow,roll,bend,dev],
arm_r[same], leg_l[hip_pitch,hip_yaw,hip_roll,knee,ankle_pitch,ankle_roll],
leg_r[same]**.

Arm joint limits are v9 verbatim (per-hand mirrored where noted; values in
rad). New-joint limits from anatomical range tables. τ_max derivation §1.5;
damping b in N·m·s/rad; ω cap in rad/s.

| # | joint | limits lo, hi (rad) | deg | tier (α) | τ_max | ω cap | damping b |
|---|---|---|---|---|---|---|---|
| 0 | spine_pitch | −0.3490658503988659, 0.7853981633974483 | −20, +45 | spine (160) | 800.00 | 16 | 4.3140 |
| 1 | spine_yaw | ±0.6981317007977318 | ±40 | spine (160) | 319.93 | 16 | 1.1998 |
| 2 | spine_roll | ±0.5235987755982988 | ±30 | spine (160) | 500.00 | 16 | 3.6471 |
| 3 | l_arm_pitch | −1.5707963267948966, 1.1344640137963142 | −90, +65 | arm (160) | 113.84 | 16 | 0.42689 |
| 4 | l_arm_yaw | −1.7453292519943295, 0.7853981633974483 | −100, +45 | arm (160) | 113.84 | 16 | 0.42689 |
| 5 | l_arm_twist | −1.5707963267948966, 1.2217304763960306 | −90, +70 | arm (160) | 19.45 | 16 | 0.07292 |
| 6 | l_elbow | 0, 2.6179938779914944 | 0, 150 | arm (160) | 19.13 | 16 | 0.07174 |
| 7 | l_wrist_roll | 0, 3.141592653589793 | 0, 180 | wrist (400) | 1.06 | 40 | 0.00159 |
| 8 | l_wrist_bend | ±1.6580627893946132 | ±95 | wrist (400) | 1.06 | 40 | 0.00159 |
| 9 | l_wrist_dev | −0.5235987755982988, 0.2617993877991494 | −30, +15 | wrist (400) | 1.06 | 40 | 0.00159 |
| 10 | r_arm_pitch | −1.5707963267948966, 1.1344640137963142 | −90, +65 | arm (160) | 113.84 | 16 | 0.42689 |
| 11 | r_arm_yaw | −0.7853981633974483, 1.7453292519943295 | −45, +100 | arm (160) | 113.84 | 16 | 0.42689 |
| 12 | r_arm_twist | −1.2217304763960306, 1.5707963267948966 | −70, +90 | arm (160) | 19.45 | 16 | 0.07292 |
| 13 | r_elbow | 0, 2.6179938779914944 | 0, 150 | arm (160) | 19.13 | 16 | 0.07174 |
| 14 | r_wrist_roll | −3.141592653589793, 0 | −180, 0 | wrist (400) | 1.06 | 40 | 0.00159 |
| 15 | r_wrist_bend | ±1.6580627893946132 | ±95 | wrist (400) | 1.06 | 40 | 0.00159 |
| 16 | r_wrist_dev | −0.2617993877991494, 0.5235987755982988 | −15, +30 | wrist (400) | 1.06 | 40 | 0.00159 |
| 17 | l_hip_pitch | −0.3490658503988659, 2.0943951023931953 | −20, +120 | leg (160) | 484.68 | 16 | 1.8175 |
| 18 | l_hip_yaw | −0.7853981633974483, 0.6981317007977318 | −45, +40 | leg (160) | 100.00 | 16 | 0.28120 |
| 19 | l_hip_roll | −0.7853981633974483, 0.5235987755982988 | −45, +30 | leg (160) | 300.00 | 16 | 1.8133 |
| 20 | l_knee | 0, 2.6179938779914944 | 0, 150 | leg (160) | 250.00 | 16 | 0.26294 |
| 21 | l_ankle_pitch | −0.8726646259971648, 0.3490658503988659 | −50, +20 | leg (160) | 180.00 | 16 | 0.00561 |
| 22 | l_ankle_roll | ±0.3490658503988659 | ±20 | leg (160) | 50.00 | 16 | 0.00138 |
| 23 | r_hip_pitch | −0.3490658503988659, 2.0943951023931953 | −20, +120 | leg (160) | 484.68 | 16 | 1.8175 |
| 24 | r_hip_yaw | −0.6981317007977318, 0.7853981633974483 | −40, +45 | leg (160) | 100.00 | 16 | 0.28120 |
| 25 | r_hip_roll | −0.5235987755982988, 0.7853981633974483 | −30, +45 | leg (160) | 300.00 | 16 | 1.8133 |
| 26 | r_knee | 0, 2.6179938779914944 | 0, 150 | leg (160) | 250.00 | 16 | 0.26294 |
| 27 | r_ankle_pitch | −0.8726646259971648, 0.3490658503988659 | −50, +20 | leg (160) | 180.00 | 16 | 0.00561 |
| 28 | r_ankle_roll | ±0.3490658503988659 | ±20 | leg (160) | 50.00 | 16 | 0.00138 |

Left-hand arm limits mirror right verbatim from v9 `joint_limits(hand)`
(yaw/twist/roll/dev antisymmetric; pitch/elbow/bend symmetric). Hip
yaw/roll are mirrored the same way; hip_pitch/knee/ankle are symmetric.

### 1.5 τ_max derivation (continuity) and damping

`τ_max[j] = I_ref[j] × α_tier[j]`, with `I_ref[j]` = moment of inertia of
the distal subtree about joint j's axis at the reference pose, and
α_tier = v9's per-tier accelerations (arm 160 rad/s², wrist 400; spine/leg
adopt 160). Reference poses: neutral for all joints EXCEPT the long-axis
joints, whose neutral subtree is on-axis (degenerate): **arm_twist** uses
elbow = 90°, **wrist_roll** uses bend = 90°, **hip_yaw** uses knee = 90°.
Derived values (kg·m² → N·m):

| joint | I_ref | derived τ | table max | adopted |
|---|---|---|---|---|
| spine_pitch | 7.1900 | 1150.41 | 400 | **800.00** (clamped 2×) |
| spine_yaw | 1.9996 | 319.93 | 200 | 319.93 |
| spine_roll | 6.0786 | 972.57 | 250 | **500.00** (clamped 2×) |
| arm pitch/yaw | 0.71149 | 113.84 | 120 | 113.84 |
| arm_twist | 0.12154 | 19.45 | 120 | 19.45 |
| elbow | 0.11957 | 19.13 | 70 | 19.13 |
| wrist roll/bend/dev | 0.0026556 | 1.06 | 15 | 1.06 |
| hip_pitch | 3.0292 | 484.68 | 300 | 484.68 |
| hip_yaw | 0.46866 | 74.99 | 100 | **100.00** (floored 1×) |
| hip_roll | 3.0222 | 483.55 | 150 | **300.00** (clamped 2×) |
| knee | 0.43823 | 70.12 | 250 | **250.00** (floored 1×) |
| ankle_pitch | 0.0093419 | 1.49 | 180 | **180.00** (floored 1×) |
| ankle_roll | 0.0023083 | 0.37 | 50 | **50.00** (floored 1×) |

Clamp rule: ARM tier keeps the derivation verbatim (v9 feel is the
contract; all arm values are < 2× the human table anyway). SPINE/LEG
joints (no v9 α to honor) are clamped into **[1×, 2×]** of the approximate
human isometric-maximum table (N·m): trunk flex/ext 400, rotation 200,
lateral 250; hip flex/ext 300, rotation 100, ab/adduction 150; knee 250;
ankle plantar/dorsi 180, inversion/eversion 50; (arms, for reference:
shoulder 120, elbow 70, wrist 15). The knee/ankle floors exist because the
distal-subtree derivation is ill-posed under the foot weld (the ankle's
distal subtree is the 1 kg foot) — see Divergences.

**Joint damping** (replaces v9 DRAG): viscous torque `−b_j·q̇_j` inside the
dynamics, with `b_j = (DRAG/DT)·I_ref[j] = 0.6·I_ref[j]` (table §1.4;
I_ref pre-clamp). At the reference pose this reproduces v9's 2 %-per-tick
velocity decay exactly; `b·h/I ≈ 0.0025` so explicit integration is stable.

**Armature** (ADDED at P1 — Divergences §12): per-DoF reflected rotor
inertia `armature[j] = I_ref[j]/100` (pre-clamp, same column as the
damping rule), added to M's diagonal in CRBA and as `armature·q̈_j` in
RNEA (MuJoCo semantics; the MJCF twin MUST declare it via the
`armature` attribute — P2b). Required for positive definiteness: the
shoulder stack yaw→pitch→twist gimbal-locks at pitch ±90°, and the
arm-pitch limit is exactly −90° — at the clamp the mass matrix is
exactly singular without it.

## 2. Collision spec

Colliders (12), fixed normative ids — capsule radii as §1.2:

| id | collider | geometry |
|---|---|---|
| 0 | torso | chest capsule r 0.15 |
| 1 | head | sphere r 0.09 |
| 2 | uarm_l | capsule r 0.045 |
| 3 | uarm_r | capsule r 0.045 |
| 4 | farm_l | capsule r 0.04 |
| 5 | farm_r | capsule r 0.04 |
| 6 | fist_l | sphere r 0.05 (center 0.06 past wrist along blade) |
| 7 | fist_r | sphere r 0.05 |
| 8 | thigh_l | capsule r 0.06 |
| 9 | thigh_r | capsule r 0.06 |
| 10 | shank_l | capsule r 0.045 |
| 11 | shank_r | capsule r 0.045 |

EXCLUDED from the roster entirely: **blades** (holograms, collide with
nothing), **feet** (welded; ground interaction is the weld, not a contact),
**pelvis** (inside the torso/thigh envelope). Ground plane has no contacts
in v10.

**Excluded pairs** (11), pair-by-pair with reason:

| pair | reason |
|---|---|
| (0,1) torso–head | weld parent–child |
| (0,2) (0,3) torso–uarm | parent–child (chest→uarm) |
| (0,8) (0,9) torso–thigh | adjacent via pelvis; interpenetrating at neutral (chest lower cap vs thigh upper cap) |
| (2,4) (3,5) uarm–farm same side | parent–child |
| (4,6) (5,7) farm–fist same side | parent–child (fist center 0.06 from the forearm tip < r₄+r₆) |
| (8,10) (9,11) thigh–shank same side | parent–child |

**Live pair table** (55 = C(12,2) − 11), lexicographic (i, j), i < j. The
row index in this list is the normative `pair id` (0–54) used by the PGS
constraint ordering and the replay contact events:

```
 0:(0,4)   1:(0,5)   2:(0,6)   3:(0,7)   4:(0,10)  5:(0,11)
 6:(1,2)   7:(1,3)   8:(1,4)   9:(1,5)  10:(1,6)  11:(1,7)
12:(1,8)  13:(1,9)  14:(1,10) 15:(1,11)
16:(2,3)  17:(2,5)  18:(2,6)  19:(2,7)  20:(2,8)  21:(2,9)
22:(2,10) 23:(2,11)
24:(3,4)  25:(3,6)  26:(3,7)  27:(3,8)  28:(3,9)  29:(3,10)
30:(3,11)
31:(4,5)  32:(4,7)  33:(4,8)  34:(4,9)  35:(4,10) 36:(4,11)
37:(5,6)  38:(5,8)  39:(5,9)  40:(5,10) 41:(5,11)
42:(6,7)  43:(6,8)  44:(6,9)  45:(6,10) 46:(6,11)
47:(7,8)  48:(7,9)  49:(7,10) 50:(7,11)
51:(8,9)  52:(8,11)
53:(9,10)
54:(10,11)
```

Narrow phase: capsule–capsule = segment–segment closest distance minus the
radius sum (Ericson §5.1.9; sphere = zero-length capsule). One contact
point per pair (the closest-point pair), one normal row. **Contact margin
0.01 m** (a constraint row activates when gap < 0.01), **penetration slop
0.005 m** (Baumgarte pushes only beyond it, §3). **Restitution 0,
frictionless** — normal rows only, λ ≥ 0. At the reference pose no live
pair is within the margin (verified: min gap 0.08, thigh–thigh).

## 3. Solver spec

`DT = 1/30` (control/obs/scoring tick, unchanged). `N_SUB = 8` substeps per
tick at `h = 1/240` s. Applied joint torques are constant across a tick's
8 substeps. **The following per-substep pipeline and its ordering are
NORMATIVE — f64 cross-implementation parity depends on every step
happening in exactly this order:**

1. **Kinematics**: body poses from (q_root, q₂₉).
2. **CRBA** → mass matrix `M(q)` (35×35, symmetric PD; root 6 + 29).
3. **RNEA(q, v, 0)** → bias forces `C(q, v)` (Coriolis + gravity,
   g = 9.81 −y).
4. Applied generalized force `τ = τ_gear − b ∘ q̇` (actuation §4 + per-joint
   damping; root rows 0).
5. **Dense Cholesky** `M = L·Lᵀ` (lower, no pivoting).
6. Unconstrained velocity `v* = v + h·M⁻¹(τ − C)`.
7. **Collision detection** over the 55-pair table (margin 0.01).
8. **Constraint solve (REVISED at P2 — Divergences §15, §16): exactly
   12 iterations, fixed order, no warm starting** (λ = 0 each substep —
   determinism/parity simplicity). Row set:
   - rows 0–5: foot_l weld (3 linear, then 3 angular),
   - rows 6–11: foot_r weld (same layout),
   - one normal row per ACTIVE contact, in pair-id order (§2),
   - one unilateral joint-limit row per joint whose NEARER stop is
     within `h·ω_cap[j]` of `q_j`, joint order 0..28 (ties → lower
     stop; row +e_j at the lower stop / −e_j at the upper).
   Each of the 12 iterations, in order:
   (a) the 12 bilateral weld rows solved EXACTLY as a dense block —
       `Δλ_w = A_ww⁻¹(b_w − J_w·v)` with `A_ww = J_w M⁻¹ J_wᵀ`
       no-pivot-Cholesky-factored once per substep;
   (b) contact rows then limit rows, projected Gauss-Seidel to λ ≥ 0,
       in row order;
   (c) ω-cap pass, joint order 0..28: hard clamp of `q̇_j` to ±ω_cap
       via momentum-consistent impulses (exact assignment).
   `M⁻¹Jᵀ` columns and effective masses via Cholesky solves of THIS
   substep's factor. Limit-row and ω-cap impulse directions are
   projected into the WELD NULL SPACE:
   `c_j = M⁻¹e_j − M⁻¹J_wᵀ·A_ww⁻¹·(J_w M⁻¹ e_j)` (so `J_w·c_j = 0` —
   limit/cap impulses can never break the welds); a direction whose
   composite mobility ≤ **1e-4** × its plain `(M⁻¹)_jj` is immobile
   given the welds and its row is inert (REVISED at P3 from 1e-10 —
   Divergences §19a: the threshold is an impulse-amplification bound,
   not a numerical-noise floor).
   **Right-hand sides** (`J·v = b` welds; `J·v ≥ b`, λ ≥ 0 unilaterals):
   - welds: `b = −(β_weld/h)·C_pose`, β_weld = **0.2** (position error
     3-vector; orientation error via the log map, 3-vector),
   - contacts: `b = −gap/h` while gap ≥ 0 (speculative activation
     inside the margin — Divergences §16), else
     `(β_c/h)·max(penetration − slop, 0)`, β_c = **0.2**, slop 0.005,
   - limits: same shape with slop 0 and β_weld pushback.
9. `v ← v* + M⁻¹Jᵀλ`.
10. **Semi-implicit Euler positions**: `q₂₉ ← q₂₉ + h·q̇₂₉`; root position
    `p ← p + h·v_lin`; root quaternion via exp map of the WORLD angular
    velocity: `q_root ← normalize(exp(ω·h/2) ⊗ q_root)` with
    `exp(u) = (cos|u|, sin|u|·û)`, Taylor guard `|u| < 1e-12 → identity`.
11. **Post-solve clamps as momentum-consistent impulses** (REVISED at
    P1 — Divergences §13; the v9-style direct coordinate write is
    NORMATIVELY RETIRED, it detonates the free-floating body):
    a. position pass, joint order 0..28: if `q_j` is outside
    `[lo_j, hi_j]`, clamp it and mark the joint LIMIT-HIT;
    b. **8 Gauss-Seidel passes** over joints 0..28 (early-out when a
    pass changes nothing): target = 0 for limit-hit joints, else
    `clamp(q̇_j, ±ω_cap[j])`; when `q̇_j ≠ target`, apply the internal
    impulse `v ← v + M⁻¹e_j·(target − q̇_j)/(M⁻¹)_jj`, then
    `q̇_j ← target` exactly. `M⁻¹e_j` columns use THIS substep's
    Cholesky factor, solved lazily once per joint and cached;
    c. guarantee pass, joint order 0..28: `q̇_j ← clamp(q̇_j, ±ω_cap)`,
    then `q̇_j ← 0` for limit-hit joints (direct write — residuals at
    this point are near-null-mode noise with negligible momentum).
    Dissipative as before.
12. **Root velocity envelope guarantee (ADDED at P3 — Divergences
    §19b; supersedes step 11's "the root is never clamped"; AMENDED
    2026-08-03 — Divergences §23)**: the envelope is applied **TWICE
    per substep — immediately BEFORE step 10, and again after step
    11** (both applications are the identical expression tree): if
    `|ω_root| > 96` rad/s, rescale the root angular 3-vector onto 96;
    if `|v_root| > 48` m/s, rescale the linear 3-vector onto 48. The
    welded feet + capped joints kinematically bound the true root
    velocity (6-joint leg chain × 16 rad/s; ≤ ~0.5 m root–foot
    levers) — beyond the envelope is integrator/solver artifact, not
    physics; ordinary motion never approaches it.
    The **pre-step-10** application is what bounds the INTEGRATED root
    motion: `‖Δp‖ ≤ h·48` and `|Δθ| ≤ h·96` per substep,
    unconditionally, and hence the next substep's Baumgarte weld bias.
    The **post-step-11** application is what bounds the STORED
    velocity (step 11's momentum-consistent joint impulses
    `Δv = M⁻¹e_j·λ` have non-zero ROOT rows and can re-exceed the
    envelope after step 10 has already run). Both are no-ops on any
    state ordinary motion reaches — a converged v10 policy measured
    **0 engagements in 204,800 env-ticks** — so the amendment is
    BITWISE transparent on healthy trajectories.
    **Engagements of the pre-step-10 application are COUNTED** (per
    (env, substep); `pop_envelope_hits()` on RawBeatSim / VecBeatSim /
    CudaBeatVecEnv, the `pop_pd_hits` pattern) and an
    env that engages it during a control tick **TERMINATES that tick**
    with `done = true` and its **reward set to exactly 0** (§4): the
    trajectory is bounded but no longer physical, so none of it may
    reach the policy. The post-step-11 application is deliberately not
    counted (it is the measured subset; see Divergences §23).

Feet weld definition: foot body frames held at their reference-pose world
poses — foot_l at (−0.10, 0.07, −0.65), foot_r at (+0.10, 0.07, −0.65),
identity orientation (6 bilateral rows each).

## 4. Actuation & effort

- **Action**: `MultiDiscrete([n_gears] × 29)`, joint order = §1.4. Gear
  tables verbatim from v8 (`constants.gear_table` / `GEAR_SIGNED`): id 0 =
  coast, 1/2 = ±full at every K; K=5 appends ±1/4; K=7 appends ±1/3, ±1/8.
  Default `--gears 5` carries over.
- **Torque**: `τ_j = gear_signed[a_j] × τ_max[j]`, held constant for the
  tick's 8 substeps. No feed-forward gravity compensation — holding a pose
  against gravity costs gear effort (by design).
- **Effort cost** (replaces the v7.9 EFFORT_COST table): positive
  mechanical work,
  `effort_tick = effort_coef × Σ_{s=1..8} Σ_{j=0..28} max(τ_j·q̇_j^{(s)}, 0)·h`
  with `q̇^{(s)}` the post-solve substep velocity. (The per-substep sum IS
  the work integral; τ is constant over the tick.) Reward −= effort_tick.
- **effort_coef = 4e-4** (per joule). Calibration: a representative
  v9-style swing — shoulder chain to ω ≈ 8 rad/s (half cap), elbow ride,
  wrist snap to ≈ 20 rad/s — costs
  `½·0.7115·8² + ½·0.1196·8² + ½·0.00266·20² ≈ 22.8 + 3.8 + 0.5 ≈ 27.1 J`
  of positive work; the same swing in v9 billed ≈ 0.01 (full-gear Σ
  EFFORT_COST ≈ 9.4e-4/tick × ~10–15 active ticks). 0.01 / 27.1 J =
  3.7e-4 → frozen at 4e-4. Episode check: ~43 swings × 27.1 J × 4e-4 ≈
  0.47 — inside v9's 0.5–1.0 per-episode band. Flagged for P7 re-tune (the
  gravity/work economy is a known behavioral discontinuity).
- **Root-escape termination (ADDED 2026-08-03 — §3 step 12, Divergences
  §23)**: if the pre-step-10 root-velocity envelope engages on ANY of a
  tick's 8 substeps, that tick's reward is **exactly 0** (it overrides
  every other term — effort, cuts, misses, follow-through) and the
  episode ends (`done = true`, auto-reset as usual). The state left the
  kinematically reachable envelope, so the tick's observation and reward
  are solver artifact, not play. Measured incidence: 0 on a converged
  policy over 204,800 env-ticks.

Env-layer geometry that was constant in v9 becomes live-pose-derived
(judgment formulas otherwise unchanged): broad-phase reach gate
`note_z ≤ max_hand(shoulder_z(t)) + 0.65 + 1.0 + 0.75` (v9's REACH_Z 1.75
at the reference pose), miss boundary `z < head_z(t)` (v9's −Z_MISS at
reference). Scoring, hitboxes, note look-at (billboarding to the LIVE head
position), swing tracking, rewards, chart generation: SPEC.md verbatim.

## 5. Observation layout (f32, order is a contract)

Total **OBS_DIM = 340**. Block offsets:

| offset | len | block |
|---|---|---|
| 0 | 3 | root position: x/1, (y − 0.94)/0.75, (z + 0.65)/1 |
| 3 | 6 | root orientation, 6D two-column: [R00,R10,R20, R01,R11,R21] of the world-from-body rotation |
| 9 | 3 | root linear velocity / 6 |
| 12 | 3 | root angular velocity (world) / 16 |
| 15 | 31 | q block (29 joints; wrist rolls as cos/sin → 31 lanes; see below) |
| 46 | 29 | q̇ block: q̇_j / ω_cap[j], joint order §1.4 |
| 75 | 10 | hand L derived: wrist x/1, (y−1.4)/0.75, z/1; wrist_vel(3)/6 (finite-diff across the 30 Hz tick); blade dir(3); swing meter min(swing_acc/PRE_FULL, 2) |
| 85 | 10 | hand R derived (same) |
| 95 | 3 | head position (FK): x/1, (y − 1.70)/0.75, (z + 0.65)/1 |
| 98 | 2 | contact bits [L, R]: 1.0 if any substep this tick had an active contact row involving that arm's colliders (uarm/farm/fist of the side; a cross-arm contact sets both) |
| 100 | 224 | notes 16 × 14 — v9 NOTE_FEAT verbatim (SPEC.md obs v9) |
| 324 | 16 | note mask — v9 verbatim |

q block (offsets 15–45): normalized angle per joint in §1.4 order; the two
wrist rolls keep v9's cos/sin coding (the only 2-lane joints — no other
joint wraps):

| lanes | content | normalizer |
|---|---|---|
| 15–17 | spine pitch, yaw, roll | /0.7853981633974483, /0.6981317007977318, /0.5235987755982988 |
| 18–25 | L arm: pitch, yaw, twist, elbow, cos ψ, sin ψ, bend, dev | v9 verbatim: /(π/2), /1.7453292519943295, /(π/2), (2q−E)/E with E=2.6179938779914944, cos/sin, /1.6580627893946132, /0.5235987755982988 |
| 26–33 | R arm (same coding) | v9 verbatim |
| 34–39 | L leg: hip_pitch, hip_yaw, hip_roll, knee, ankle_pitch, ankle_roll | /2.0943951023931953, /0.7853981633974483, /0.7853981633974483, (2q−E)/E, /0.8726646259971648, /0.3490658503988659 |
| 40–45 | R leg (same) | same |

New-joint normalizer rule (how the table was built): `q / max(|lo|, |hi|)`,
except one-way knees which use the elbow-style `(2q − hi)/hi`.

Wrist position/velocity, blade dir, swing meter, and all note features keep
their exact v9 semantics (they feed the unchanged judgment code). Informative
(policy-side, finalized in P4): recommended token layout
`[saber_L, saber_R, body, note_0..15]` (19 tokens) — saber tokens from each
hand's q/q̇ slices + derived block + hand one-hot, the ONE body token from
root block + spine/leg q/q̇ + head + contact bits; 29 action heads via the
fused (trunk → 29·K + 1) GEMM.

## 6. Replay v10 (JSONL)

Header line: `{"header": {"version": "v10", "record_rev": 2,
"config": {...}, "chart": [...]}}` — chart entries unchanged from v7+.
Viewers reject non-"v10" versions (all v9 replays/checkpoints retire,
house pattern). `version` names the BODY generation only; the per-tick
record grows ADDITIVELY under `record_rev` (absent/1 = pre-`head`,
2 = `head` present), so a new field never retires a reader.

Per-tick record:

```json
{"t": 0, "time": 0.0,
 "q": [px, py, pz, qw, qx, qy, qz, <29 joint angles, order §1.4>],
 "twist": [vx, vy, vz, wx, wy, wz],
 "omega": [<29 joint velocities>],
 "sabers": [[wx, wy, wz, bx, by, bz], [wx, wy, wz, bx, by, bz]],
 "head": [px, py, pz, qw, qx, qy, qz],
 "action": [<29 gear ids>],
 "reward": 0.0,
 "contacts": [{"pair": 12, "p": [x, y, z], "j": 0.42}],
 "events": [ ... ]}
```

- `q`: 36 position coords — root position, root quaternion **wxyz (scalar
  first)**, then the 29 joints. `twist`: root world-frame linear + angular
  velocity. `omega`: the 29 joint rates.
- `sabers`: per hand [L, R], FK wrist position + unit blade dir — viewer
  convenience (the v9 fields the viewer actually consumes); the viewer does
  full-skeleton FK from `q`.
- `head` (record_rev 2, ADDITIVE): the FK head frame's world origin and
  world-from-body rotation, quaternion **wxyz** like `q`'s root block.
  Same "viewer convenience" role as `sabers` — derivable from `q`, logged
  because the head is welded to the chest (§1.2) and every exporter needs
  its ROTATION: the ScoreSaber `.dat` head frame drives
  watch.scoresaber.com's first-person camera AND its note-arrival plane.
  Head local +z is the gaze, matching `blade` = hand frame +z (§1.3).
- `contacts`: one entry per pair with nonzero tick-summed normal impulse:
  normative `pair id` (§2), world contact point (last substep's), scalar
  impulse `j` = Σ_substeps λ_n (N·s). Empty list omitted-or-empty (writer
  emits `[]`).
- `events`: score events UNCHANGED from v7+ (`cut` with
  points/pre/acc, `post`, `miss`, `bomb`).

## 7. Test gates (P1–P6)

House style: gates are pytest files; a phase does not merge until its gate
is green plus all previous gates stay green. f64 parity = EXACT equality
(identical expression trees), the existing discipline.

**P1 — core math** (`rust` unit tests + `python/tests/test_v10_math.py`):
- Arm-chain FK vs v9 `fk()` on 10k random in-limit poses (root at
  reference, spine 0): |Δ| ≤ 1e-12 per component (wrist, elbow, blade).
  Spine/leg FK vs an independent Python mirror: ≤ 1e-12.
- M(q): exact symmetry (built symmetric), Cholesky succeeds on 10k random
  in-limit states; CRBA columns vs RNEA(q, 0, e_j) ≤ 1e-10.
- RNEA(q, 0, 0) vs hand-derived gravity torques on 1- and 2-body chains
  ≤ 1e-10.
- Single pendulum: v10 pipeline vs a 50-digit mpmath implementation of the
  SAME discrete map, 240 substeps: |Δθ| ≤ 1e-10. Convergence: error vs the
  elliptic-integral closed form at h, h/2, h/4 shrinks at O(h).
- Double pendulum energy (gravity on, damping/clamps off):
  |E(t) − E(0)|/|E(0)| ≤ 1e-3 over 1 s at h = 1/240, and halves with h/2.
- Momentum (gravity off, free base, no constraints; REVISED at P1 —
  Divergences §14): symplectic Euler in these coordinates conserves
  neither linear nor angular momentum exactly (the momentum map A(q)v
  has q-dependent coefficients; per-step error O(h²), fixed-horizon
  drift O(h)). Gate:
  linear + angular relative drift ≤ 2e-2 over 2 s of tumbling at
  h = 1/240 (measured 1.32e-2), halving with h (measured ratio 2.20).

**P2 — constraints + contacts** (RE-SCOPED at P2 — Divergences §17:
gates live in the Rust crate, `rust/src/dynamics/tests_{collide,
constraints,oracle}.rs`, until P3 exposes the engine to Python; several
bounds re-read against measured physics, reasoning in §17):
- Capsule–capsule distance on 10k random pairs incl. parallel and
  degenerate (sphere) cases: brute-force-sampling upper bound + convex
  box-KKT certificate ≤ 1e-9 (measured 8.2e-16).
- Welded stance + constant full 35-DoF RNEA(q_ref, 0, 0) compensation
  incl. the root wrench: **2 s**, max joint drift ≤ 1e-6 rad (measured
  3.1e-13), weld violation ≤ 1e-8 m (measured 6.2e-15). Muscle-tone
  stance: vertical weld force = body weight ±5 % (measured 739.45 N vs
  735.95).
- Weld under load: scripted two-arm swinging ON THE MUSCLE-TONE STANCE,
  10 s: weld pose violation ≤ 1e-4 m / 1e-4 rad (measured 1.2e-6 m /
  1.7e-6 rad).
- Weld under torture: 10 s per-tick random FULL-gear torques, all 29:
  weld violation < 1e-2 m / 5e-2 rad at every substep (measured 7.5e-3
  / 1.8e-2 — the β = 0.2 Baumgarte floor, §17), no NaN, root within the
  kinematic envelope.
- Non-penetration: oracle contact-probe presses: max penetration
  < 2×slop = 0.01 m (measured 0.0066 / 0.0081 — equal to MuJoCo's),
  λ ≥ 0, complementarity ≤ 1e-3, zero impulse once separated.
- Baumgarte energy: contact-rich scripted rollout: per-substep
  `ΔE − W_applied ≤ Σ max(λ·b, 0)` (the Baumgarte work bound; max
  injection measured 1.6e-2 J), and E(t) − W_applied(t) never rises by
  > 1 J over any 1 s window (measured −0.037 J).
- Oracle trajectory bands (all 5 refdata scenarios, stabilizer kd
  implicit — §17): max |Δq| ≤ 0.05 rad smooth / ≤ 0.15 rad random
  (measured 0.0068 / 0.0165 / 0.0452 / 0.0200 / 0.0484).
- Determinism: contact-rich trajectory + impulse stream bitwise
  identical across runs.

**P2b — MuJoCo oracle** (`python/tests/test_v10_oracle.py`, skipped when
mujoco is absent; `mjcf/body_v10.xml`):
- FK cross-check: mj_kinematics vs our FK, 1k random poses, ≤ 1e-8 m —
  MANDATORY FIRST (catches MJCF multi-joint composition-order mistakes).
- Gravity load: MuJoCo inverse dynamics at (q, 0, 0) vs our RNEA ≤ 1e-6
  N·m per joint (identical explicit inertias).
- Trajectory bands (different solvers — bands, not parity): 0.5 s scripted
  torques, no contacts, ω/limit clamps DISABLED in ours: per-joint |Δq|
  ≤ 0.02 rad; with welds + a scripted self-contact: ≤ 0.05 rad.

**P3 — env integration** (`python/tests/test_v10_env.py`): obs length 340 +
every §5 offset; determinism (seed + action stream ⇒ identical rewards/
events, all object kinds); effort vs a hand-computed one-joint case
≤ 1e-12; live REACH/miss gates move with a crouched/leaning pose; replay
v10 round-trip (counters + score match the sim); scripted-torque chart
smoke: ≥ 1 good cut. Scenario tests re-authored on the dynamic body (v9's
`set_joints` pose-family tests are retired with the kinematic body; the
replacement drives torques).

**P4 — torch mirror** (`python/tests/test_v10_parity_torch.py`): f64
EXACT (bitwise) obs/reward/done/event parity vs Rust across the parity
config matrix (gears 3/5/7, contacts on, welds on) over full episodes;
f32 bounded per the existing suite's tolerances. **RETIRED 2026-08-03**
— the gate was met (and is recorded in Divergences §20), but the mirror
it gated (`torch_env.py` + `torch_dynamics.py`, `--env torch`) had no
consumer left besides its own tests once P5 took training, so each
physics change was paying a third tier. Both modules and the P4 test
module are deleted; see the live contract below.

**PARITY CONTRACT (live, from 2026-08-03).** The **Rust crate is THE
reference sim** — every rule change lands in `rust/src/` first. The
**CUDA backend is the one gated mirror** (`--env cuda`, the training
backend), gated by `python/tests/test_v10_parity_cuda.py` per P5. There
is no bitwise cross-implementation tier any more: the CUDA f64 branch
is horizon-bounded tight-tolerance by construction (Divergences §21 —
libdevice is 1 ulp off the CRT libm and nvcc contracts FMAs), so the
f64 tier is single-tick/short-horizon free-running at the seed band
PLUS lockstep-resync whole episodes where counters / score
decomposition / per-note status / done timing are EXACT. Bitwise
regression coverage of the dynamics kernels themselves lives in the
Rust cargo suites (`dynamics/tests_dyn.rs`, `tests_soak.rs`,
`env_tests`), which is where a bitwise claim is actually checkable.

**P5 — CUDA mirror** (`python/tests/test_v10_parity_cuda.py`): the f64
tier above / f32 bounded matrix vs Rust; fused-policy kernels guarded to
the v10 layout (29 heads, obs 340) with junk-inertness tests.

**P6 — viewer + bot** (`viz/app` node tests + `python/tests/
test_v10_bot.py`): JS full-body FK vs sim-logged `sabers` wrist ≤ 1e-6;
replay v10 parse + version rejection; per-joint τ/ω gauges + foot
reaction-force display render (screenshot smoke); gravity-compensated
PD-over-IK bot ≥ 50 % good cuts on the default chart (v9 solvability bar).

## 8. Decisions record (locked 2026-07-31)

1. **Hand-rolled engine, MuJoCo as tolerance-band oracle** (not MJX
   backend) — triple-mirror f64 parity is the house quality mechanism.
2. **Feet welded** in v10; ungluing is v11+ (same constraint machinery).
3. **Hard ω clamps kept** (v9-style, post-solve, dissipative); motor
   curves later. DRAG → physical joint damping b = 0.6·I_ref.
4. **Spine 3-DoF, neck fixed** (head welded to chest).
5. **75 kg / de Leva masses on v9 anchor geometry** (shoulders 1.47, head
   1.70 preserved exactly; legs/torso derived, §1.1).
6. **τ_max from subtree-inertia × v9-α continuity** (arms verbatim;
   spine/legs clamped into [1×, 2×] human-table band, §1.5).
7. **30 Hz control retained**, physics substepped 8 × 1/240 s.
8. **Torch-first training, CUDA trailing** (P4 before P5; P5 may trail
   P6).

## Divergences

Deviations from the proposal / v9 conventions made at P0, with reasons:

1. **Segment COMs at geometry centroids**, not de Leva's proximal-offset
   COM fractions (~40–45 %). One source of truth (the capsule) for COM +
   inertia + collider; masses carry the anthropometric fidelity. Affects
   gravity torques by a few percent per segment.
2. **Long-axis τ reference poses**: shoulder twist (elbow 90°), wrist roll
   (bend 90°), hip yaw (knee 90°). At neutral these subtrees are on-axis —
   the derivation degenerates to near-zero torque. Flexed-reference is the
   pose where the joint actually moves mass.
3. **Legs + spine τ clamped into [1×, 2×] the human-torque table.** The
   pure derivation is ill-posed for stance joints under the foot weld: the
   ankle's distal subtree is the 1 kg foot (1.5 N·m — useless), the knee's
   is shank+foot (70 N·m), while spine pitch/roll (arms + head + chest at
   full lever) blow past 2× human (1150 / 973 N·m). Floors: hip_yaw 100,
   knee 250, ankle 180/50; ceilings: spine_pitch 800, spine_roll 500,
   hip_roll 300. Arms untouched (continuity is the contract).
4. **Pendulum gate precision re-read**: semi-implicit Euler at h = 1/240
   is O(h) — 1e-10 vs the CONTINUOUS closed form is unattainable. The gate
   is 1e-10 vs a high-precision implementation of the same discrete map,
   plus an O(h) convergence check against the elliptic solution.
5. **Live pair count 55**, not the proposal's ~25–30 estimate: the full
   non-adjacent closure of the 12-collider roster. All 55 are cheap
   (~55 segment–segment tests/substep); pruning would buy nothing and cost
   an arbitrary-looking table.
6. **Torso–thigh pairs excluded** beyond strict parent–child (they
   interpenetrate at the neutral pose: chest lower cap vs thigh upper cap).
7. **Pelvis has no collider** — proposal roster verbatim; it sits inside
   the torso/thigh envelope.
8. **effort_coef 4e-4** — derived (§4), but the work economy under gravity
   is new; explicitly flagged for P7 re-tuning rather than pretending the
   calibration is exact.
9. **Obs total 340**, below the proposal's 380–420 ballpark (that estimate
   assumed wider cos/sin use; only the two wrist rolls wrap).
10. **MJCF multi-joint composition order**: the twin declares stacked
    joints outer→inner assuming MuJoCo applies same-body joints in
    declaration order. The P2b FK cross-check runs FIRST and catches a
    reversed convention before any trajectory test is trusted.
11. **Stance width = hip width (feet at x ±0.10)** — the proposal left
    stance unspecified; matching the hip anchors keeps thigh/shank capsules
    vertical at neutral.
12. **Armature added (P1, 2026-07-31)**: `armature[j] = I_ref[j]/100`
    on every actuated DoF (§1.5). The P0 spec's "M symmetric PD,
    no-pivot Cholesky" is FALSE without it: the shoulder's
    yaw→pitch→twist stack gimbal-locks at pitch ±90°, arm-pitch's lower
    limit is exactly −90°, and the limit clamp parks joints exactly
    there (first blow-up observed after 849 substeps of constant
    torque; failing pivot −4.3e-18). Armature enters CRBA's diagonal
    and RNEA's q̈ term identically in every mirror; the MJCF twin adds
    the `armature` attribute (P2b re-check).
13. **Step-11 clamps are impulses, not coordinate writes (P1)**: the
    v9-style `q̇_j ← 0` at a torque-pinned limit discards the joint's
    momentum share while the solve's reaction stays in the root — a
    measured pump of ~0.5 rad/s root spin per substep under 1 N·m on
    every joint, driving the free body past the explicit-Coriolis
    stability bound (h·ω ≈ 1, ~240 rad/s at h = 1/240) in under a
    second of ordinary bang-bang actuation. Replaced by internal
    impulses `Δv = M⁻¹e_j·λ` (zero root rows ⇒ momentum map intact),
    8 Gauss-Seidel passes + a direct guarantee pass (§3.11). One pass
    is not enough: near-collinear axis pairs (straight arm:
    twist/roll) ping-pong the violation. With welds (P2) the same
    rows may fold into the main PGS; row order 0..28 is the contract.
14. **P1 momentum gate re-read (like Divergence §4)**: exact discrete
    momentum conservation is not a property of symplectic Euler in
    root-frame coordinates — the P0 gate's 1e-10 was aspirational.
    Replaced by an O(h)-convergence gate with a measured band (§7 P1).
    Sustained net free moments in unsupported free fall (e.g. constant
    τ_max on all 29 joints for seconds) legitimately spin the root
    past h·ω ≈ 1 where NO fixed-step explicit integrator can follow;
    the welded v10 env cannot produce that input, and the P1 stability
    test uses per-tick random full gears instead (10 s clean, root ω
    settling ~15–25 rad/s).
15. **P2 solver revision — weld block + limit/cap rows folded into the
    main solve (step 8 as now written; solver.rs)**. Three measured
    failures of the P0 "pure per-row PGS-12 + separate step-11 clamps"
    design forced it, each kept deterministic and mirrorable:
    a. **Welds as an exact 12×12 block, not GS rows.** The two feet's
       vertical rows are near-parallel through the root at the
       straight-knee stance; 12 GS sweeps leave a 0.14 m/s weld-row
       residual → ~3 mm steady weld violation (blows the 1e-4 load
       gate by 40×). The block solve (`A_ww` no-pivot Cholesky, 12×12,
       once per substep) is machine-tight (measured 1.1e-16 post-solve
       residual) at negligible cost.
    b. **Joint-limit rows join the solve** (the §3.11/Divergence-§13
       anticipation "may fold into the main PGS", exercised). The
       neutral stance parks knee, elbow and wrist-roll exactly ON a
       0-stop; with limits handled only post-hoc (step 11), the weld
       solve assigns flexion rate and the position clamp teleports q
       back — a measured 5.8e-4 m of foot error injected per substep
       (3.7 mm steady violation). Speculative activation: a row exists
       whenever the nearer stop is within `h·ω_cap` (the farthest the
       capped joint can travel in one substep).
    c. **Limit/ω-cap directions projected into the weld null space.**
       Even in the solve, plain GS between a pinned joint and the weld
       block does not converge in 12 iterations (near-parallel
       directions in the M metric — knee motion IS foot motion on a
       straight leg; measured up to 38 mm foot drift under full random
       gears). With `c_j = M⁻¹e_j − M⁻¹J_wᵀA_ww⁻¹(J_w M⁻¹e_j)` the
       limit/cap impulses cannot break the welds by construction, and
       the physical interpretation is exact: the stop's reaction is
       transmitted to the ground through the welded feet.
    **Step 11 remains, verbatim, as the outer guarantee** (its
    impulses are near-null in ordinary motion now). Final normative
    substep order: steps 1–6; step 7 narrow phase (pair-id order);
    step 8 = 12 × [weld block → contacts (pair-id order) → limit rows
    (joint order) → ω-cap pass (joint order)]; step 9 implicit in the
    incremental impulses; step 10 integrate; step 11 clamp passes
    (joint order 0..28). Parity mirrors replicate this literally.
16. **Contact RHS completed with speculative activation (P2)**: the P0
    text only defined the Baumgarte push-out term. Activating a hard
    `J·v ≥ 0` row at any gap < margin would make the 1 cm margin a
    standoff force field; instead an open gap allows approach at
    exactly `gap/h` (Catto-style speculative contact: the pair may
    close the remaining gap this substep, no more). Penetration then
    stays bounded near the slop; measured press penetrations equal
    MuJoCo's (0.0066 / 0.0081 m vs oracle's 0.0066 / 0.0081).
17. **P2 gate re-readings (like §4/§14; measured numbers in §7 P2)**:
    a. **Static stance horizon 10 s → 2 s** (the oracle protocol's
       horizon). The welded stance under CONSTANT full-RNEA
       compensation is a knife-edge equilibrium (inverted-pendulum
       modes under constant torque); rounding-noise seeds (~1e-17 —
       the weld targets' decimal literals vs the FK's binary sums)
       e-fold every ~0.35 s, so a 10 s horizon amplifies any seed by
       e²⁸ regardless of solver quality. MuJoCo's b2 check uses 2 s
       for the same reason. Measured drift at 2 s: 3.1e-13 rad.
    b. **Weld-under-load scenario runs on the muscle-tone body**: the
       raw open-loop body falls over in ~1 s (README: chaotically
       unintegrable without the stabilizer), after which "weld
       violation under arm swinging" measures a collapsing tangle, not
       the weld. Gate value kept at 1e-4 (passes at 1.2e-6).
    c. **Random-full-gear weld drift 2 mm → 10 mm / 0.05 rad**: with
       leg joints riding their 16 rad/s caps, the velocity-level
       constraint's per-substep linearization error is O(h²·q̇²·L)
       ≈ 1.5 mm and the Baumgarte steady state is that /β. Measured
       7.5 mm at β = 0.2 and 2.5 mm at β = 1 (β-sensitivity confirms
       the floor); β stays at the normative 0.2, the gate moves to the
       floor. Orientation measured 1.8e-2 rad (the original 0.02
       bound holds).
    d. **Oracle-band harness applies the stabilizer kd IMPLICITLY**
       (MuJoCo-Euler damping semantics: `v⁺ = v + h(M+h·diag(kd))⁻¹
       (τ − C − kd∘q̇)`; harness-only path, the env's own pipeline is
       untouched). Explicit application is unstable at h = 1/240: at
       the straight-arm pose shoulder-twist/wrist-roll are collinear,
       `(M⁻¹)_jj ≈ 1/armature`, and `h·kd·(M⁻¹)_jj ≫ 2` — the wrists
       rattle at the ±40 rad/s cap (measured; the cap-less MuJoCo
       equivalent NaNs, which is exactly why MuJoCo integrates joint
       damping implicitly and why the README's stabilizer exists).
       With matched semantics all 5 bands pass at the ORIGINAL gates
       (no widening): 0.0068–0.0484 rad.
    e. P2 gates are cargo tests in the Rust crate until P3 exposes the
       engine to Python (`test_v10_contacts.py` then wraps them).
18. **P3 env-integration surface (2026-07-31)** — the hard-replace of
    the v9 body in `rust/src/lib.rs`, decisions made at integration:
    a. **v9 `fk()` survives as a `#[cfg(test)]` reference
       implementation** in lib.rs (with its SHOULDER/L_UPPER/L_FORE
       literals) solely for the P1 continuity gate
       (`tests_fk::v9_fk_continuity` references `crate::fk`); nothing
       live calls it. `kinematics.py` remains its Python mirror,
       banner-marked v9-legacy (its IK feeds the P6 bot rebuild).
    b. **Effort's "post-solve substep velocity" = the END-of-substep
       velocity** (after the step-11 clamps and step-12 envelope) —
       the velocity that integrates positions next substep. Pinned
       bitwise against an independent pipeline mirror in
       `lib.rs env_tests::effort_is_positive_work`; the Python gate
       checks the reward identity `r_effort = −coef × last_effort`.
    c. **Obs contact bits count ASSEMBLED rows** (gap < margin, λ may
       be 0 — the §5 "active contact row" as the solver defines
       active); replay `contacts` keep the stricter nonzero
       tick-summed-λ rule (§6).
    d. **Episode tail**: v9's `0.5 + Z_MISS/min_speed` becomes
       `0.5 + 1.5/min_speed` — the miss boundary is the LIVE head
       plane now, and head_z ≥ −1.5 bounds every reachable pose (the
       welded body cannot fold the head further back), so final
       misses always register inside the tail.
    e. **FFI surface**: `set_joints` is the FULL-state hook (q36 =
       root pos + quat wxyz + 29 joints; v35), used by the pose
       tooling and every re-authored scenario test. RawBeatSim gains
       `obs()` (no-step introspection), `qpos()/twist()/omega()/
       sabers()/contacts()/last_effort()`; `sabers()` returns the §6
       viewer form `[wrist(3), blade(3)]` per hand (the v9 17-wide
       joint-state form died with the kinematic body).
    f. **effort_coef defaults to 4e-4** (the §4 literal) in the Rust
       ctors and Python wrappers; 0 still = free movement (knob
       semantics carried).
    g. **P2 gates stay cargo tests** — P3 integrates the engine
       BEHIND the env FFI without raw dynamics bindings, so the
       planned `test_v10_contacts.py` wrapper (§17e) is dropped;
       Python-side physics visibility arrives with P4's parity suite.
       (P3 adds `tests_soak.rs`: baked wild-pose regression states
       from 512-env random-gear soaks, §19.)
    h. **Tooling**: `convert_scoresaber.py` consumes replay v10
       (direct wrist+blade from `sabers`; version gate "v10"; head
       pose frame stays the reference head until P6's full-body FK
       mirror) and takes `first_note_s` as a parameter (default 2.0,
       the bench convention).
19. **P3 solver-stability revisions (2026-07-31)** — two measured
    blow-ups from 512-env random-full-gear soaks (~10⁶ env-ticks),
    both ending in a mass-matrix PD panic within one control tick:
    a. **COMPOSITE_EPS 1e-10 → 1e-4** (§3 step 8). The projected
       direction is `c_j = S·e_j` with
       `S = M⁻¹ − M⁻¹J_wᵀA_ww⁻¹J_wM⁻¹` symmetric PSD, so enforcing a
       cap/limit on joint j moves any other coordinate by
       `Δv_i = S_ij·Δv_j/S_jj ≤ sqrt(S_ii/S_jj)·Δv_j`. At 1e-10,
       joints in the near-immobile band (S_jj/plain ∈ (1e-10, 1e-4);
       contorted poses reach it) amplified cap assignments into the
       root by up to ~1e5× per application — measured |ω_root|
       15 → 5.5e5 rad/s in one tick. At 1e-4 the amplification is
       bounded by ~100; joints below the threshold are owned by the
       welds and their bounds fall to the step-11 clamps (plain
       momentum-consistent directions, always well-conditioned).
    b. **Step 12 root envelope** (§3). Cap-riding contorted poses
       spike the EXPLICIT Coriolis term (unconstrained Δv ≈ 150 per
       substep at |q̇| pinned to the 16/40 caps — the §14 h·ω ≈ 1
       instability, which P0 wrongly assumed the welded env could not
       produce); the solve re-caps every joint but the root rows kept
       the artifact velocity and the next substep's Coriolis fed on
       it. The envelope (|ω_root| ≤ 96 = 6×16 chain bound, |v_root| ≤
       48) is a kinematic consequence of welds + caps — the rescale
       only ever removes non-physical velocity. Regression state
       baked in `dynamics/tests_soak.rs`; 2000 × 512 random-gear soak
       clean after both fixes.
20. **P4 torch-mirror notes (2026-07-31; the mirror was REMOVED
    2026-08-03 — this entry is the record of what it proved, not a
    live tier)** — the f64 CPU parity bar was
    met EXACTLY (bitwise obs/reward/done/status/state/effort/contact
    impulses across the gears 3/5/7 matrix, trajectories and full
    episodes); the transcription discipline required two measured
    torch-CPU deviations to be engineered around, and one Rust
    observable is represented differently:
    a. **torch CPU f64 `scalar / tensor` division is not correctly
       rounded** (reciprocal path; 28 % of random inputs 1 ulp off,
       measured) — the mirror materializes numerator tensors
       (tensor/tensor division IS IEEE). torch's CPU f64
       sin/cos/acos/**sqrt** kernels (SLEEF) differ from the platform
       CRT libm by 1 ulp on ~1.5–7 % of inputs; the f64 CPU path
       routes them through numpy, which defers to the same CRT libm
       Rust's f64 methods call (verified bitwise on 2 000 points).
       f32/GPU paths use native torch ops (tolerance class).
    b. **Post events carry no note ids in the torch mirror**: the
       follow-through trackers are aggregated ring slots (same-tick
       same-saber trackers share their arc, hence their points), and
       reward pays each tracker individually in spawn order — bitwise
       — but WHICH note a post belonged to is not represented. The
       parity gate pins post totals via reward + score_post + status;
       replay/bench event streams stay on the Rust sim.
    c. **Sign-of-exact-zero relaxation**: multiplications by the model
       tables' exact 0.0/±1.0 constants are constant-folded, and the
       fixed 8 clamp passes skip Rust's early-out, so ±0.0 values may
       differ in sign from Rust's. IEEE equality (the comparison the
       gate uses) treats them equal, and no expression in the pipeline
       is zero-sign-sensitive (no divisions or sign extractions of
       exact zeros).
    d. **Fused fp16 attention is ineligible at 19 tokens** (the kernel
       requires an even token count); the v10 policy's body token makes
       T = 19, so fp16 rollouts fall back to eager attention until the
       P5 kernel pass revisits the layout.
    e. **f32 Cholesky pivot floor (training path only)**: M(q)'s
       condition reaches ~3e5 at gimbal-adjacent poses (min eigenvalue
       ≈ the wrist armature 2.7e-5), so f32 factorization pivots go
       non-positive under contorted random-gear poses where f64 stays
       PD (first observed within 4 ticks of full-random gears at
       N = 1024). The f32 path floors pivots at 1e-8 and counts
       engagements (`pd_net_hits`); the caps + step-11 clamps +
       step-12 envelope bound the resulting velocities — the same
       stability-net role §19 assigns them. The f64 parity path kept
       the strict Rust semantics (raise = Rust's panic) until §22
       floored the f64 tier everywhere (same net, eps-scaled floor).
21. **P5 CUDA-mirror notes (2026-07-31)** — the CUDA backend
    (`csrc/beat_arena_cuda.cu`, one THREAD per env, the Rust loops
    transcribed serially per thread so every accumulation keeps Rust's
    order) meets a TOLERANCED f64 tier, not the P4 bitwise bar, for a
    measured and fundamental reason:
    a. **CUDA f64 transcendental provenance** (the §20a follow-up):
       device sqrt and division are IEEE-correctly-rounded and match
       the CRT libm BITWISE (measured over 10⁵ inputs); device
       sin/cos/acos (libdevice) differ from the CRT by exactly 1 ulp
       on 10.5 % / 14.0 % / 4.2 % of inputs, and nvcc's default FMA
       contraction re-rounds multiply-add chains. There is no
       practical way to reproduce the closed-source UCRT rounding in
       device code, so the CUDA f64 path carries ≤1-ulp-per-op seeds.
    b. **Seeds are chaotically amplified by design**: the v10 dynamics
       e-fold perturbations every ~0.35 s (§17a) and the solver's
       knife edges (contact margin, limit-row reach, the §19a
       COMPOSITE_EPS band — the stance parks knee/elbow/wrist-roll ON
       stops, permanently near-band) flip discretely. Measured: RUST
       ITSELF, given a 1-ulp state perturbation, diverges to
       macroscopic scale by ~tick 10–20 under gentle gears (t≈3 under
       full random gears). No free-running full-episode tolerance can
       therefore exist for ANY non-bitwise mirror; the v9 CUDA tier's
       full-episode 1e-6 bounds relied on the v9 kinematic body being
       contractive (DRAG), which died with it.
    c. **The P5 gate** (`test_v10_parity_cuda.py`) is accordingly:
       the transcendental measurement itself (a); reset/set_state
       bitwise-or-1e-12; single-tick + short-horizon parity in the
       seed band (obs 1e-5, reward 1e-9) with a small deterministic
       quota of knife-edge envs; LOCKSTEP-RESYNC full episodes (the
       CUDA env re-takes Rust's exact state each tick) with counters /
       score decomposition / per-note status / done timing EXACT and
       rewards inside the integer-rounding flip band; and an f32
       well-formedness tier. Since the torch mirror's removal
       (2026-08-03) this IS the cross-implementation gate — there is
       no bitwise Python tier left; bitwise regression lives in the
       Rust cargo suites. The CUDA backend's charter is f32 training
       throughput with the §20e pivot-floor net mirrored
       (`pop_pd_hits`, reported per update).
    d. **Throughput** (GTX 1080, f32, kernel-only / full training):
       ~4.7 k / ~4.5 k env-steps/s at N = 1024, plateauing at ~9 k /
       ~9.7 k by N = 4096 (vs the torch mirror's 107 at N = 1024 —
       42–90×). Profile: insensitive to block size and register
       budget; bound by per-env serial latency + local-memory traffic
       (51 KB stack/thread — working set ≫ L2). The identified next
       lever is intra-env parallelism (warp per env, factor in shared
       memory) if P7 needs more than ~25 h per 400 M at N = 1024.
    e. **fp16-all loss scale**: at the v10 obs/reward scale the
       default 131072 static loss scale overflows every minibatch
       until the halving audit walks it down — fp32 and fp16-update
       train cleanly. Flagged for the P7 re-tune alongside §8's
       effort economy.
22. **f64 Cholesky pivot floor (2026-08-02)** — found post-training:
    benching v10_gears5_400m ckpt_upd0762 (a mid-surge, flaily greedy
    policy) crashed VecBeatSim step_into with the Rust "mass matrix
    not positive definite" PANIC — a contorted gimbal-adjacent state
    (the §12/§21 conditioning family: shoulder stacks parked at the
    pitch −90° lock under cap-riding rates, min pivot ≈ the wrist
    armature 2.7e-5, cond(M) ~3e5) pushed an f64 pivot into the
    rounding-noise band. A bench must degrade gracefully, never
    panic; the panic is retired on every tier:
    a. **`PD_FLOOR_F64 = 1e-17`** (algo.rs): the §20e CUDA f32 net's
       1e-8 scaled by the machine-epsilon ratio (1e-8 · eps_f64 /
       eps_f32 ≈ 1.9e-17, rounded down a decade). The smallest
       PHYSICAL pivot is 12 decades above the floor, so a healthy
       factorization is BITWISE identical to the strict one; the
       floor only absorbs the singular-pose noise band (P1-measured
       gimbal pivot −4.3e-18 clamps cleanly).
    b. **Semantics**: `chol_factor_floored` clamps each pivot with a
       simple `max(d, PD_FLOOR_F64)` (fixed expression, no fastmath
       ambiguity — exactly reproducible) and counts engagements into
       `Workspace::pd_hits`; both former panic sites use it (step 5
       mass matrix AND the §15a weld block A_ww). Counts pop via the
       additive FFI getter `pop_pd_hits()` on RawBeatSim/VecBeatSim
       (the CUDA `pop_pd_hits` naming; no existing FFI shape
       changed); bench_scoresaber reports run + file totals
       (`pd_floor_hits` in the results JSON — reported, not hidden).
    c. **Parity**: the .cu f64 template branch floors identically
       (same constant, same max clamp; its f32 branch keeps the §20e
       1e-8). Because the floor is bit-transparent for healthy pivots,
       the P5 CUDA tier gates are unchanged and stay green. (The torch
       mirror carried the same clamp in `chol_factor` / `_chol12` with
       a `pd_net_hits` counter until its 2026-08-03 removal; the
       forced-engagement regression now lives only in
       `tests_dyn::substep_survives_pd_loss_and_counts`, which drives
       the exactly-singular armature-less matrix through the real
       substep.)
    d. **Post-floor safety**: the ω caps + step-11 clamps + step-12
       envelope bound the velocities a floored (near-singular) solve
       can emit — the same stability-net role §19/§20e assign them.
       Regression: `tests_dyn::pivot_floor_clamps_singular_gimbal_matrix`,
       `tests_dyn::substep_survives_pd_loss_and_counts` (armature-less
       gimbal state — the exactly-singular §12 matrix — through the
       full substep: no panic, counter increments, state finite), and
       `tests/test_v10_pd_floor.py` (FFI end-to-end + the cross-tier
       floor-constant contract).
23. **§19b root envelope applied BEFORE integration; the escape
    mechanism re-diagnosed (2026-08-03).** §19b attributed the root
    escape to the explicit Coriolis term at `h·ω ≈ 1`. **That
    attribution is wrong** (measured, research/2026-08-03-root-escape-
    design.md): escapes are born at `h·|ω_root| = 0.005…0.11`, the
    Coriolis contribution to the root's Δω at the birth substep is
    0.07–0.3 rad/s against the solver's 1e4–1e5 rad/s, and deleting the
    velocity-product bias term entirely does not prevent them (it makes
    the gears-3 escape set worse: 339.7 m → 1.4e8 m).
    a. **Mechanism.** The spike is born INSIDE step 8's ω-cap/limit
       stage. The §15c weld-null-space projection makes that stage the
       operator `Q_j = I − (S e_j)e_jᵀ/S_jj` with
       `S = M⁻¹ − M⁻¹J_wᵀA_ww⁻¹J_w M⁻¹`. `Q_j` is idempotent but NOT
       M-self-adjoint (`M S` is not symmetric), so it is an OBLIQUE
       projector with `‖Q_j‖_M ≈ (S_jj/(M⁻¹)_jj)^{-1/2}` — measured
       **90–116** for the four ankle joints when their mobility ratio
       sits just above the §19a `COMPOSITE_EPS = 1e-4`. Twelve
       iterations of a gain > 1 diverge geometrically: measured
       ×2.2–4.7 per iteration; `|v|` inside the PGS 387 (2 iterations)
       → 8.5e3 (6) → 9.3e5 (12) → 8.3e21 (24) on the same state, in
       f64. §19a bounded the per-APPLICATION amplification; it did not
       bound the LOOP. Trigger: a weld-owned joint riding its ω cap at
       `r_j ∈ [1.1e-4, 1.6e-4]` — reached by contorted cap-riding
       poses, i.e. random and mid-training policies, not converged ones
       (measured incidence: 0.005–0.015 % of random-policy env-ticks,
       0.00049 % at `ckpt_upd0762`, **0** at `final.pt`).
    b. **The fix (A1).** The envelope is therefore ALSO applied
       immediately before step 10 (§3 step 12). This bounds the
       integrated root motion at `‖Δp‖ ≤ h·48 = 0.2 m` per substep and
       hence the next substep's Baumgarte weld bias at `β/h·0.2 = 9.6`
       m/s instead of the measured **1.9e3 m/s**, which kills the
       cascade. Measured: the captured escape corpus goes from 339.7 m
       (gears 3) / 34.07 m (gears 5) worst per-substep `|Δp|` to
       **0.2 m exactly** in every configuration, and a 192-state ×
       16-substep adversarial sweep (60 % of joints on a stop, all
       rates saturated, welds pre-violated 0.10 m) that went NON-FINITE
       in all six f32 configurations pre-fix stays finite and inside the
       0.2 m bound post-fix. The alternative "rewind p on engagement"
       form was rejected: its bound leaks by construction (29–33 % of
       over-envelope integrations never reach the step-12 test; measured
       `|Δp|` up to 0.975 m against the same 0.2 m bound).
    c. **Bitwise transparency.** `torch.where`/`if` returns the input
       unchanged when nothing engages, and nothing engages on healthy
       play: a `ckpt_upd3048` rollout of 512 envs × 400 ticks = 204,800
       env-ticks in the CUDA f32 env is **bitwise identical** (obs,
       reward, done, state, score, contacts) before and after this
       amendment, with `envelope_hits = 0`. Under a UNIFORM-RANDOM
       policy the suites do contain escapes, and there the only rows
       that change are exactly the escaping envs from exactly the tick
       they escape (256-env × 200-tick f32/f64 suites: 1–4 envs differ,
       every other env bitwise, all four suites).
    d. **C backstop.** An engagement of the pre-step-10 application
       terminates the env's episode that tick with the tick's reward
       set to exactly 0. Rationale: A1 makes the trajectory BOUNDED, it
       does not make it MEANINGFUL, and feeding a physically
       meaningless observation/reward to the policy is the harm the
       hunt started from (the fp16 loss-scale collapse and the §22
       bench panic were both downstream of it). Auto-reset handles the
       done exactly like any other. The trigger is "any engagement",
       per the design doc: the cap itself is already a 4.41× margin over
       the worst converged-policy `|ω_root|` (21.76 vs 96) and 2.91× in
       `|v_root|`, and the measured healthy engagement rate is 0, so a
       larger multiplier would only let more meaningless physics through
       without protecting any real play.
    e. **Counter.** `Workspace::env_hits` (Rust), `Dyn::env_n` +
       the `env_hits` device tensor (CUDA), popped by
       `pop_envelope_hits()` on every backend with identical
       per-(env, substep) semantics (pinned exactly Rust↔CUDA-f64 on
       the escape fixture; the retired torch mirror was pinned
       bitwise there until 2026-08-03).
       The trainer prints it per update; bench_scoresaber writes
       `envelope_hits` into the results JSON next to `pd_floor_hits` —
       reported, never hidden.
    f. **NOT fixed here.** The oblique projection itself. Making the
       cap/limit rows use the plain (M-orthogonal, `‖Q‖_M = 1`) `M⁻¹e_j`
       direction, or folding the saturated caps into the exact weld
       block, would remove the geometric gain rather than bounding its
       consequences — but the affected joints (the four ankles) are
       engaged on **88 %** of healthy trained-policy substeps, so either
       change alters every trajectory and belongs to a v11 body revision
       with a full physics + recipe re-gate. Also declined there:
       implicit/semi-implicit Coriolis (does not fix this bug, +30–60 %
       substep cost, and it retires the symmetric-PD/no-pivot-Cholesky
       invariant the tri-tier bitwise method rests on), raising
       COMPOSITE_EPS (works — 339.7 m → 0.275 m — but silently
       reclassifies the ankles on 88 % of healthy substeps), and
       adaptive substepping (measured WORSE: the divergence lives inside
       one substep's 12-iteration loop, and halving `h` doubles the
       Baumgarte gain `β/h`).
    g. **Regression**: `dynamics::tests_soak::{root_step_bounded_on_the
       _escape_state, adversarial_states_stay_finite_and_bounded,
       healthy_motion_never_engages_the_envelope,
       soak_state_engages_the_envelope_and_stays_bounded}`,
       `env_tests::{escaped_env_terminates_with_zero_reward,
       healthy_env_never_escapes}` (Rust), and
       `tests/test_v10_root_escape.py` (Rust FFI + CUDA f64/f32 tiers
       on the escape fixture: `done`, zeroed reward, engagement count
       and the ‖Δp‖ ≤ N_SUB·h·ROOT_V_MAX bound all pinned exactly, the
       state to the single-tick f64 band. The retired torch mirror
       additionally pinned the escape tick BITWISE until 2026-08-03).
