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
8. **Constraint solve — PGS, exactly 12 iterations, fixed row order**:
   - rows 0–5: foot_l weld (3 linear, then 3 angular),
   - rows 6–11: foot_r weld (same layout),
   - then one normal row per ACTIVE contact, in pair-id order (§2).
   Bilateral weld rows unclamped; contact rows projected to λ ≥ 0. Effective
   masses via `J M⁻¹ Jᵀ` (Cholesky solves). **No warm starting** (λ = 0
   each substep — determinism/parity simplicity).
   **Baumgarte** bias velocities on the right-hand side:
   - welds: `v_bias = −(β_weld/h)·C_pose`, β_weld = **0.2** (position error
     3-vector; orientation error via the log map, 3-vector),
   - contacts: `v_bias = −(β_c/h)·max(penetration − slop, 0)`, β_c =
     **0.2**, slop 0.005.
9. `v ← v* + M⁻¹Jᵀλ`.
10. **Semi-implicit Euler positions**: `q₂₉ ← q₂₉ + h·q̇₂₉`; root position
    `p ← p + h·v_lin`; root quaternion via exp map of the WORLD angular
    velocity: `q_root ← normalize(exp(ω·h/2) ⊗ q_root)` with
    `exp(u) = (cos|u|, sin|u|·û)`, Taylor guard `|u| < 1e-12 → identity`.
11. **Post-solve clamps, in joint-index order 0..28** (v9-style,
    dissipative):
    a. `q̇_j ← clamp(q̇_j, −ω_cap[j], +ω_cap[j])`;
    b. if `q_j < lo_j` or `q_j > hi_j`: `q_j ← clamp(q_j, lo_j, hi_j)` and
    `q̇_j ← 0`.
    The root is never clamped.

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

Header line: `{"header": {"version": "v10", "config": {...},
"chart": [...]}}` — chart entries unchanged from v7+. Viewers reject
non-"v10" versions (all v9 replays/checkpoints retire, house pattern).

Per-tick record:

```json
{"t": 0, "time": 0.0,
 "q": [px, py, pz, qw, qx, qy, qz, <29 joint angles, order §1.4>],
 "twist": [vx, vy, vz, wx, wy, wz],
 "omega": [<29 joint velocities>],
 "sabers": [[wx, wy, wz, bx, by, bz], [wx, wy, wz, bx, by, bz]],
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
- Momentum (gravity off, free base, no constraints): linear momentum
  relative drift ≤ 1e-10 over 1 s; angular ≤ 1e-4 with O(h) convergence.

**P2 — constraints + contacts** (`python/tests/test_v10_contacts.py`):
- Capsule–capsule distance vs brute-force sampling ≤ 1e-9 on 10k random
  pairs, including parallel and degenerate (sphere) cases.
- Welded stance + per-substep gravity-compensation torque
  τ = RNEA(q_ref, 0, 0): 10 s, max joint drift ≤ 1e-6 rad, weld violation
  ≤ 1e-8 m.
- Weld under load: scripted two-arm swinging, 10 s: weld pose violation
  ≤ 1e-4 m / 1e-4 rad (Baumgarte holds).
- Non-penetration: 30 s random-gear rollout: max contact penetration
  ≤ 0.01 m.
- Baumgarte energy: contact-rich scripted rollout: E(t) − W_applied(t)
  never rises by > 1 J over any 1 s window.

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
f32 bounded per the existing suite's tolerances.

**P5 — CUDA mirror** (`python/tests/test_v10_parity_cuda.py`): same f64
exact / f32 bounded matrix vs Rust; fused-policy kernels guarded to the
v10 layout (29 heads, obs 340) with junk-inertness tests.

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
