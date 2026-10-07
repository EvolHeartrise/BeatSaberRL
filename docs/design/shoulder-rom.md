# Shoulder ROM design: position-dependent axial limits and a passive shoulder tendon

Branch `exp/shoulder-rom` (from `origin/exp/wrist-tendon`, the v65 body lineage).
**Status: PHASE 2 IMPLEMENTED (2026-10-07)** with the recommended defaults below, plus the user's decision 6: the twist box's internal side is widened to **−190° (left) / +190° (right)**, so every human IR limit of the fit is reachable: the worst direction (abduction plane, ~40° elevation) needs −184.3°, rounded out to −190° for ~6° of margin. See the PR for test, parity and throughput results. The phase-1 text below is kept as written; where it says "phase 2" it describes what was built.
Date: 2026-10-07.

Prototype scripts that check every number below are in [`docs/design/shoulder_rom/`](shoulder_rom/). How to run them is in §9.

---

## 0. Summary

**Coordinates.** Our shoulder stack `R_y(yaw)·R_{−x}(pitch)·R_z(twist)` is *exactly* the ISB Y-X-Y humerothoracic globe (verified to 5e-16), with:
- elevation = pitch + 90°;
- plane of elevation = yaw + 90° (left arm; the right arm is mirrored);
- globe axial rotation = −twist − 90° (internal rotation positive).

Because twist is the globe's axial angle, today's fixed twist box is a fixed range in ISB axial rotation: **ER 160° … IR 0° at every arm direction**, plus a further 145° of axial rotation from yaw near the hanging pole (Codman's paradox).

**Axial coordinate.** The design uses the swing-twist angle about the hanging reference:
- `τ = yaw − twist` (IR+), which equals ISB plane + ISB axial.
- It is linear in q, has a constant Jacobian (1, 0, −1), and is regular at the hanging pole.

**Limits.** Human limits are smooth functions of the arm direction:
- `f(d) = A(cos θ) + B·(d_lat − d_fwd) + (1 − cos θ)·φ`, one function for the ER limit and one for the IR limit;
- fitted to 4 literature anchors (Vairo 2012, Roy 2009, Wang 1998, Humphries 2015, Novotny 2000);
- regular at both the hanging and the overhead pole.

**Passive term.** A soft, potential-based axial tendon `E = T·w·G(u)`, written in the wrist tendon's design language:
- `u = (τ − mid(d)) / halfrange(d)`;
- zone start s = 0.3;
- peak T = 0.30 × default-body twist τmax = **5.093 N·m** at the human limit;
- steeper growth past the limit (one extra T per 25% of the half range, squared);
- linear-torque continuation past u = 1.25 (bounded stiffness, still rising);
- full twist gear can hold the arm only up to about u = 1.29, i.e. 14–24° past the human limit.

**Torque mapping and damping.**
- Joint torques τ = −∂E/∂q are analytic, with no special case at the poles.
- **Implicit rank-1 edge damping** acts on τ̇ = ẏaw − ṫwist, with b_pk = T/16 rad/s = 0.318 N·m·s/rad.

**Stiffness.**
- Worst ω·h is 0.57 at h = 1/240 and 0.29 at 1/480, using the model's own M⁻¹, over replay poses and the whole joint box. The explicit tendon is fine.
- Damping b·h/I_ax reaches up to 0.83, so the damping must be implicit, as on the wrist.

**Mirror-exact by construction.** The right arm is evaluated at the canonical angles (−yaw, pitch, −twist), and its yaw and twist torques are negated.

**Impact (re-scored v61/v64/v65 replays, open loop).**
- v65 spends **11.7%** of arm-ticks beyond the human *external*-rotation limit for its arm direction (median 9.5° past, p90 27°), and 0% beyond IR, which the box already blocks.
- v61: 8.4%. v64: 10.0%.
- In 1.6% of v65 ticks the passive torque would exceed the twist actuator's full torque.
- The dominant uncertainty is the ER limit at 90° flexion: v65's share is 3.8% / 11.7% / 23% for anchors −45° / −25° / −10°.

**Cone.** No term is needed. The yaw/pitch box already sits inside the human sinus; it is *more* restrictive, with no extension, no horizontal extension, and a 155° elevation cap.

**Forearm coupling.** No coupled term is needed (see §3.6). Forearm roll sits on its 0 stop in 55–64% of ticks (71% with a straight arm). The policies do saber roll with the shoulder.

---

## A. Coordinates

### A.1 Our frames (from `rust/src/dynamics/model.rs`)

- Chest frame: y up, z forward (the zombie zero pose points the arms along +z). The left shoulder sits at x = −0.20, so −x is lateral for the left arm.
- Upper arm along local +z. Orientation (intrinsic, leftmost first):
  `M = R_y(yaw) · R_x(−pitch) · R_z(twist)`. The axes are (0,1,0), (−1,0,0), (0,0,1). The stack order is yaw → pitch → twist.
  - Actuated indices, left: pitch 3, yaw 4, twist 5. Right: 10, 11, 12.
  - Elbow 6/13, wrist roll 7/14, bend 8/15, dev 9/16.
- Humerus direction: `d = M e_z = (cos p · sin y, sin p, cos p · cos y)`.
  - Pitch is latitude above horizontal (box −90°…+65°).
  - Yaw is azimuth from forward (left box −100°…+45°; negative is lateral).
- Mirror (PR #57 / `mirror_ops.py`): arm signs [pitch +, yaw −, twist −, elbow +, roll −, bend +, dev −]. The right box is the left box negated on yaw and twist.

**Canonical (left-frame) evaluation.** For the right arm use `(y_c, p, t_c) = (−yaw_R, pitch_R, −twist_R)`.

### A.2 Exact identity with the ISB globe (Wu et al. 2005)

Write pitch = e − 90°, with e = elevation from hanging, and note `R_x(90°)·R_z(t) = R_y(−t)·R_x(90°)`:

```
M = R_y(yaw) · R_x(−e) · R_y(−twist) · B0,       B0 = R_x(90°)  (hanging reference: arm local z → world −y)
  = [R_y(yaw) R_x(−e) R_y(−yaw)] · R_y(yaw − twist) · B0      (swing · twist about the hanging axis)
```

This is checked numerically in `shoulder_rom/verify_coords.py` (max error 5.4e-16 over 2000 random poses). Therefore:

| quantity | formula (left-canonical) | meaning |
|---|---|---|
| elevation θ | `p + 90°` (0 = hanging, 155° = box max) | ISB elevation |
| plane φ | `y + 90°` (0 = abduction/coronal, 90 = flexion/sagittal, 135 = cross-body) | ISB plane of elevation |
| globe axial ψ_ISB | `−t − 90°`, IR + | ISB axial rotation: 0 at 90° abduction with the forearm forward |
| **swing-twist axial τ** | **`y − t` = φ + ψ_ISB**, IR + | rotation about the humerus after the minimal swing from hanging. 0 = forearm forward at the side; at 90° abduction it equals the clinical goniometric rotation (forearm forward = 0, forearm up = ER 90). At 90° flexion τ = 0 means forearm *up* |

The reset pose (pitch −80°, yaw 0, twist 0, elbow 100°) has τ = 0: forearm forward, neutral.

### A.3 Codman's paradox, and what is wrong with a fixed twist box

Codman's paradox:
- Start at the side with the forearm forward.
- Flex 90° (pitch −90° → 0), abduct horizontally 90° (yaw 0 → −90°), then lower to the side (pitch 0 → −90°).
- The humerus now points down again, but it is **externally rotated 90°**, although the twist joint never moved: τ = y − t went 0 → −90°.
- The globe's third angle ψ is rotation relative to a frame that rotates with the plane of elevation. It is not a property of the humerus's orientation alone.
- At the hanging pole the plane is undefined and **only φ + ψ = τ is defined** (yaw and twist are anti-parallel there, the gimbal lock the armature exists for).

A fixed box on ψ (our twist) therefore allows a τ window `[yaw − 70°, yaw + 90°]` (left) that **slides with yaw**. From `shoulder_rom/box_geometry.py`:

| arm direction | box τ window (ER … IR) | human (fit below) |
|---|---|---|
| hanging (pole; yaw is axial too) | **ER 170 … IR 135 (305° wide)** | ER 80 … IR 77 (157°) |
| 90° abduction | ER 160 … IR 0 | ER 100 … IR 60 |
| 90° abduction, scapular plane (30°) | ER 130 … IR 30 | ER 75 … IR 84 |
| 90° flexion | ER 70 … IR 90 | ER 25 … IR 130 |
| cross-body (plane 135) | ER 25 … IR 135 | τ +17 … +171 (internal only) |

**In ISB terms the box is axial ψ ∈ [−160°, 0°] at every direction (ER 160 to neutral).** The human range is about ψ ∈ [−100…−118, +36…+62] at 90° elevation. So in the policies' working region the box lets the arm go 45–60° past the human ER limit, and it blocks 40–55° of human IR. Its width (160°) is about right; its *centre* is about 50° too far toward ER. Near the side the box is twice as wide as a human shoulder.

### A.4 Exact formulas (no singular case needed)

```
# left-canonical (y, p, t); right arm: y = -yaw_R, t = -twist_R
d_l = -cos p sin y,  d_u = sin p,  d_f = cos p cos y      # lateral, up, forward
c   = cos θ = -sin p                                       # θ = p + 90°
φ   = y + 90°                                              # plane (box: −10° … 135°)
τ   = y − t                                                # axial, IR +
```

- **Hanging pole (p = −90°).** τ is linear in q. The limits are smooth in q because they depend on q only through d and c, plus the gauge `(1 − c)·φ`, which vanishes to second order at c = 1.
- **Consistency at the pole.** E and the axial torque depend only on y − t there. Verified: four (y, t) pairs with the same y − t give identical E and τ_twist. τ_pitch traces a pure sinusoid in y: the same physical force expressed along a yaw-rotated pitch axis.
- **No division anywhere** except by w (≥ 47°) and constants.
- **The only singular point** of the coordinate set is the *overhead* pole (p = +90°), 25° outside the pitch box. It is not reached, and even there every expression stays finite.
- **Reporting.** Plane and elevation are reported by telemetry and tools only. φ never needs to be "defined" at the pole for the physics.

---

## B. Human ROM data

### B.1 Axial rotation range

| source | posture | ER | IR | arc | notes |
|---|---|---|---|---|---|
| Wang et al. 1998, J Biomech 31:899, [doi:10.1016/S0021-9290(98)00098-0](https://doi.org/10.1016/S0021-9290(98)00098-0) | whole sinus cone, 7 men, polynomial surfaces | – | – | **157° arm down → 94° arm vertical** | Abstract plus a secondary implementation (Wilhelms & Van Gelder 2001). ZY′X″ convention. "As the shoulder is horizontally flexed, the IR limit decreases while the ER limit increases" |
| Vairo et al. 2012, SMARTT 4:33, [doi:10.1186/1758-2555-4-33](https://doi.org/10.1186/1758-2555-4-33) | n = 622, supine goniometry | at the side: active 76.5, passive **82.3**. At 90° abduction: active 98.8, passive **104.6** | at 90° abduction: active 54.7, passive **58.9** | at 90° abduction: passive 163.6 | full text |
| Roy et al. 2009, SMARTT 1:4, [doi:10.1186/1758-2555-1-4](https://doi.org/10.1186/1758-2555-1-4) | n = 130 | at the side, active 67.7. At 90° abduction, passive 88.7 | at 90° abduction, passive 67.5 | – | full text |
| Moromizato et al. 2016, J Physiol Anthropol, [doi:10.1186/s40101-016-0112-8](https://doi.org/10.1186/s40101-016-0112-8) | n = 78 | 92.9 | 62.5 | – | arm posture not stated |
| Humphries et al. 2015, JSES 24:1974, [doi:10.1016/j.jse.2015.07.007](https://doi.org/10.1016/j.jse.2015.07.007) | 60/90/120° elevation in coronal, scapular and sagittal planes, passive end range at 4 N·m | **max ER independent of humeral position** | **max IR reduced in the sagittal plane and at 120°** | – | numbers only in a supplement (not obtained) |
| Kolz et al. 2021, J Biomech 117:110266, [doi:10.1016/j.jbiomech.2021.110266](https://doi.org/10.1016/j.jbiomech.2021.110266) | biplane fluoroscopy | ER at the top of scapular-plane abduction 85±14° (X-Z′-Y″) | – | 117 (at the side, active, forearm-on-abdomen start) | convention-dependent |
| Chang et al. 2005, BME Appl Basis Comm 17:275 | bone pin, n ≈ 1 | – | – | glenohumeral 143° at 90° abduction, 69° at max abduction | abstract |
| Novotny et al. 2000, J Orthop Res 18:190, [doi:10.1002/jor.1100180205](https://doi.org/10.1002/jor.1100180205) | in vivo, 45° abduction, ±4 N·m | – | – | 139±40 at ±4 N·m. Neutral zone (±1 N·m) 78±46 | see B.3 |

Hill et al. 2008 (Gait Posture 27:177) and Lugo et al. 2008 contain no usable axial-range numbers. Doorenbosch et al. 2003 (JRRD 40:147) is n = 1 and gives functional planes (+101° to −68°), not limits. MoBL-ARMS (Holzbaur 2005, Saul 2015) uses the same Y-(−X)-Y sequence. `shoulder_rot` is −90…20° (v3.3) or −90…120° (a 4.1 copy), IR+, with **no elevation coupling**. Its CoordinateLimitForce values (stiffness "100", transition 485–545°) look like unvalidated tuning, not data.

### B.2 Anchors used (τ, IR +, left arm) and the fit

| anchor | θ, φ | ER limit τ_lo | IR limit τ_hi | basis |
|---|---|---|---|---|
| side | 0°, – | −80 | +77 | ER0: Vairo passive 82 / active 77, Roy 68. Arc 157 (Wang, arm down). IR at the side is really trunk-limited; the collision table handles the forearm against the trunk |
| 90° abduction | 90°, 0° | −100 | +60 | Vairo passive 105/59, Roy 89/68, Moromizato 93/63 |
| 90° flexion | 90°, 90° | **−25** | +130 | ψ_ISB ER 115 / IR 40. ER is position-independent (Humphries) or rises toward flexion (Wang); IR is reduced in the sagittal plane (Humphries, and sleeper-stretch IR at 90° flexion is about 30–50°). **Lowest confidence; see the decision list** |
| top | 155°, 40° | −75 | +20 | arc about 95 at the top (Wang 94, Chang 69 glenohumeral), mostly ER (Kolz) |

Fit form (`srom_model.py`, `fit_coeffs`), per limit k ∈ {lo, hi}:

```
f_k = a0_k + a1_k c + a2_k c² + B_k (d_l − d_f) + (1 − c) φ          [rad]
lo: a0 −107.500°, a1 38.624°, a2 −11.124°, B 7.500°
hi: a0   50.000°, a1 74.637°, a2 −47.637°, B 10.000°
```

Design notes on the form:
- **The gauge `(1 − c)·φ`.** It is 0 at the side, φ at the horizontal and about 2φ at the top. τ is the regular axial coordinate at the hanging pole, ψ_ISB is the natural one at the horizontal, and ψ_ISB − φ is the regular one overhead (the globe's top singularity). The gauge interpolates between the three, so the limits are smooth on SO(3) wherever the box can go.
- **One plane harmonic `B·(d_l − d_f)`.** At the horizontal it gives ψ-limits a0 ± B (abduction / flexion), monotone in between. A free cos/sin pair floated a0 and made a spurious 20–40° IR dip between the anchors.

The fitted limits reproduce the anchors exactly (`limit_table.py`):

| θ \ φ | −10 | 30 | 60 | 90 | 135 |
|---|---|---|---|---|---|
| 0 | −80…77 | −80…77 | −80…77 | −80…77 | −80…77 |
| 30 | −79…83 | −77…85 | −76…85 | −74…86 | −70…90 |
| 60 | −88…80 | −74…94 | −63…102 | −53…112 | −33…131 |
| 90 | −109…52 | −75…84 | −50…106 | −25…130 | 17…171 |
| 120 | −137…−4 | −82…49 | −42…88 | −1…127 | 64…191 |
| 155 | −167…−71 | −93…2 | −38…56 | 17…111 | 101…195 |

The table gives τ ER-limit…IR-limit in degrees.

Checks against the data:
- **Arc:** 157 at the side, 160–169 at 30–60°, 154–160 at 90°, 128–133 at 120°, 93–96 at 155°. Wang gives 157 → 94; Humphries says IR shrinks at 120°.
- **At 90° elevation in ψ_ISB:** ER 99–118 across all planes (Humphries: position-independent). IR falls from 62 (abduction) to 36 (cross-body) (Wang and Humphries: IR decreases toward horizontal flexion).

**Where the data is thin:** flexion and cross-body planes, and high elevation. There the fit is a conservative smooth extrapolation; §E gives the sensitivity.

### B.3 Passive torque data

| source | finding |
|---|---|
| Novotny 2000 (in vivo, 45° abduction) | Neutral zone (±1 N·m) is 78° of a 139° arc at ±4 N·m. ER flexibility from 1–4 N·m is 20 °/N·m (≈ 2.9 N·m/rad); IR is about 4× stiffer (≈ 11 N·m/rad) |
| Beardsley et al. 2010, Clin Biomech, [doi:10.1016/j.clinbiomech.2010.06.006](https://doi.org/10.1016/j.clinbiomech.2010.06.006) | Torque-angle curve is a 4-parameter logistic: compliant middle, steep ends. Neutral zone about 1.4 N·m |
| Southgate et al. 2009, J Biomech 42:1307 (cadaver capsule) | Toe region ends at about ±0.5 N·m. Arc 140° at ±1 N·m; it shifts toward ER with abduction |
| Humphries 2015/2017 | Clinical passive end range defined at **4 N·m** |
| Roy 2009 active strength (for scale) | IR 45 (men) / 21 (women) N·m; ER 34 / 17 N·m. Passive end-range torque ≈ 10% of active strength |
| Engin 1979 / Engin & Chen 1986 II / Peindl & Engin 1987 | Beyond-sinus restoring-moment maps exist; I only had the abstracts, so no numbers |

### B.4 Direction envelope (sinus)

| motion | human | our box |
|---|---|---|
| flexion | 165–176 (Vairo, Moromizato, Soucie) | max elevation 155 |
| abduction | 150–180 (Gill, Moromizato) | 155 |
| extension | 56–66 | **not reachable** (plane ≥ −10°) |
| horizontal flexion (cross-body at 90°) | 134±9 | plane 135 (on the limit) |
| horizontal extension | 56 | plane −10 (≈ 10°) |

Engin & Chen 1986 I ([doi:10.1115/1.3138605](https://doi.org/10.1115/1.3138605)), the only true 3D sinus, is paywalled; I could only read the abstract. The box's direction set lies **inside** the human sinus. It is a restriction, not a non-human allowance, so no cone term is needed (§C.4). Every direction in the box also admits at least 46° of human axial rotation inside the twist box (`overlap` scan), so the axial term never squeezes the arm against a box stop.

---

## C. Passive torque model

### C.1 Energy (per arm, left-canonical)

```
m = (f_hi + f_lo)/2,  w = (f_hi − f_lo)/2                 # mid-range and half-range (rad), w ∈ [47°, 84°]
u = (τ − m)/w,  a = |u|                                   # normalized excursion; neutral = MID-range
z = ((a − s)/(1 − s))₊,  x = ((a − 1)/wb)₊
g(a) = z² + kb·x²                     for a ≤ uc          # pure-axis torque / T
g(a) = g_c + g'_c (a − uc)            for a > uc          # linear continuation (bounded stiffness)
G(a) = ∫₀ᵃ g                                              # closed form, piecewise cubic/quadratic
E = T · w · G(a)
dE = sign(u) T g (dτ − dm) + T (G − a g) dw               # exact
τ_joint = −dE/dq,  dτ/dq = (1, 0, −1) on (yaw, pitch, twist);
dm, dw from df_k/dq (analytic: d, c, φ derivatives — srom_model.py `limits`)
```

- **Pure axial torque:** `−sign(u)·T·g(a)`, i.e. zero inside |u| < s, T at the human limit, then rising steeply.
- **Torques on yaw and pitch:** from ∂m/∂d and ∂w/∂d. If the arm is rotated in a way that is fine at its current direction but not at a neighbouring one, moving toward that neighbour costs energy. This is the capsule resisting combined positions, and it is what makes the light twist DOF follow a moving limit.
- **Gates (`check_grad.py`):**
  - −∇E vs central differences: worst relative error 3e-6;
  - closed-loop work / path work: 4e-7 (trapezoid-limited);
  - pole consistency;
  - bitwise mirror.

### C.2 Recommended numbers

| parameter | value | rationale |
|---|---|---|
| T (peak at the human limit) | **0.30 × 19.45 × 0.8729 = 5.093 N·m**, FIXED (not scaled by τ draws or tau_fit) | Same fraction as the wrist. The clinical end range is defined at 4 N·m (Humphries), and Novotny's arcs are at ±4 N·m; we hit 4 N·m at u = 0.92 |
| s (zone start) | **0.3** | 1 N·m is reached at u = 0.61, so the neutral zone covers 61% of the arc (Novotny 56%). The shoulder has a large free middle, unlike the always-centring wrist. Mean slope from 1 to 4 N·m ≈ 7.5 N·m/rad, between Novotny's ER (2.9) and IR (≈ 11) |
| kb, wb (beyond the limit) | **1, 0.25** | Adds T·((a − 1)/0.25)²: 7.5 N·m at u = 1.1, 11.7 at 1.2 |
| uc (linear past) | **1.25** (14.5 N·m) | Slope g'_c = 11.9 T per unit u. Full twist gear (16.98 N·m) is matched at u ≈ 1.29, i.e. 14–24° past the human limit, so the actuator cannot hold the arm deeper. Uncapped, the box corners (u up to 3.7) would reach about 1000 N·m; an earlier uncapped variant (wb = 0.15) measured ω·h ≈ 1.2 |
| edge damping | **b = b_pk·z(min(a, uc))²**, b_pk = T/16 rad/s = **0.318 N·m·s/rad**, on τ̇ | Onset at the zone start, b_pk at the limit, capped at 0.586 past uc. At the limit and moving at the 16 rad/s cap, the damping torque equals T (the wrist uses a 30 rad/s reference). ζ at the limit ≈ 0.8 with a straight arm (I_ax 0.0035) and ≈ 0.26 with a bent one (0.035); joint damping adds about 0.06–0.2 |

Torque profile (pure axis): u 0.5 / 0.61 / 0.8 / 0.92 / 1.0 / 1.1 / 1.25 / 1.5 / 2.0 gives 0.42 / 1.0 / 2.6 / 4.0 / 5.09 / 7.5 / 14.5 / 29.6 / 59.9 N·m.

### C.3 Soft vs hard q-dependent limit rows: **soft**

1. **Physiology.** The human end range is a stiffening (logistic) curve, not a stop (Beardsley).
2. **Solver.** A hard row `u(q) ≤ 1` has a Jacobian across yaw, pitch and twist. That means a new nonlinear row type in the PGS limit block *and* in the step-11 clamp passes, which are per-coordinate and orbit-paired (`pair_box_targets`). A coupled 3-DoF row needs its own mirror pairing and a CUDA twin.
3. **Moving limits.** The limit moves when yaw and pitch move. A hard row would fire impulses into the strong yaw/pitch actuators (about 80 N·m) on every elevation change. The potential instead lets the light twist (I_ax ≈ 0.003–0.05) comply.
4. **Hard stop.** The existing box stays the hard stop and stays unchanged.
5. **Mirror.** The soft term is mirror-exact for free under canonical evaluation.

### C.4 Cone term: **not needed**

The box is inside the human sinus (B.4). A cone term only becomes necessary if the yaw/pitch box is widened, for example to allow extension or horizontal extension (decision 7).

### C.5 Edge damping: **yes, implicit, rank 1**

- `B = b·JᵀJ` with J = (1, 0, −1) on (yaw, pitch, twist).
- The right arm has J = (−1, 0, 1), which gives the same B.
- Add h·b to M[yaw][yaw] and M[tw][tw], and −h·b to M[yaw][tw] and M[tw][yaw]. The yaw frame is an ancestor of the twist frame, so this entry is inside the tree profile (CUDA `d.m[fcb[F_UARM] + 2]`: slot 0 is the twist itself, +1 the pitch row, +2 the yaw row).
- RHS: −b·τ̇ on the yaw row, +b·τ̇ on the twist row.
- PSD, so dissipative and unconditionally stable.

### C.6 Forearm roll coupling: **no coupled term**

**Anatomy.** Humeral axial rotation is limited by the glenohumeral capsule and cuff. Pronation and supination are limited by the radioulnar joints and the interosseous membrane. Only minor biarticular coupling exists (the biceps long head). No published coupled limit was found (Wang limits the humerus only).

**Total hand roll with a straight arm** is the sum of two ranges:
- human: 157 (side) / 160 (90° abduction) / 95 (overhead) plus 160 forearm, i.e. 255–320° (Chang: 376° at 90° abduction including the scapula);
- today's model: 305 + 180 = **485° at the side**;
- with this design: 157 + 180 = 337°.

The position-dependent axial limit already bounds the "screw".

**Speed is not the anatomical problem.** Axial I_eff of a straight arm is 0.0029–0.0041 kg·m² (model M; physically about right), so the twist reaches the 16 rad/s cap in about 3 ms. The non-human part is *where* the screw happens (ER past the human limit at forward arm directions), and that is what this term targets.

**Observation.** Forearm roll sits on its 0 stop in 55–64% of ticks (71% with a straight arm), and its τmax is only 1.26 N·m. The shoulder does the saber roll. A separate forearm-roll tendon, or a forearm-strength review, is a candidate follow-up (decision 8).

---

## D. Numerics

Stiffness was evaluated with the model's own M(q): CRBA plus armature, dumped by a temporary Rust probe (`shoulder_rom/srom_probe.rs.txt`). For each pose the shoulder 3×3 block of the **full M⁻¹** is taken (root and all joints free), then `ω² = max eig((M⁻¹)_sh·K)` with K = Hessian(E).

Poses:
- 600 replay poses × twist placed at u = ±1, ±1.25, ±1.5 and at both twist box stops;
- 3000 uniform box poses (straight and bent arms).

| state | ω·h at 1/240: median / p99 / max | max at 1/480 | b·h/I_ax max | k_max (N·m/rad): median / max |
|---|---|---|---|---|
| ER limit (u = −1) | 0.17 / 0.21 / 0.22 | 0.11 | 0.33 | 19.5 / 39 |
| u = −1.25 | 0.33 / 0.43 / 0.44 | 0.22 | 0.61 | 75 / 168 |
| twist box ER stop | 0.32 / 0.43 / 0.44 | 0.22 | 0.62 | 71 / 174 |
| uniform box (deep corners, u up to 3.7) | 0.12 / 0.55 / **0.57** | 0.29 | **0.83** | 10 / 323 |

Effective axial inertia `I_ax = 1/(J M⁻¹ Jᵀ)`:

| arm | I_ax (kg·m²) |
|---|---|
| straight | 0.0029–0.0041 |
| elbow 15° | 0.0056–0.0071 |
| elbow 90° | 0.010–0.049 (smallest overhead) |

- **The explicit tendon is stable.** ω·h ≤ 0.57 is below the wrist's accepted 0.71, and symplectic Euler needs < 2. An earlier uncapped variant (wb = 0.15) reached ω·h ≈ 1.2 at the box stops; that is why the torque is continued linearly past uc.
- **Edge damping must be implicit.** b·h/I_ax reaches 0.83 on reachable box states and 0.61 at ordinary overshoot.
- **The Hessian is indefinite** in deep corners. This is the physical "fall off the ridge" of a position-dependent potential (rate ≤ 38 s⁻¹), not an integrator instability; the wrist had the same.
- **Pole.** There is no division by a vanishing quantity and the torque is continuous (C¹ in q). The gauge term's Hessian is bounded but discontinuous at the pole (it behaves like r²·atan2). The constant-Jacobian τ makes the damping exactly regular there.

---

## E. Impact on the v61/v64/v65 replays

- Data: 129 replays from R2, scratch copies. v65: 56 replays, 732k arm-ticks. v64: 55 replays, 721k arm-ticks. v61: 18 replays, 202k arm-ticks. Both arms are folded into the left frame.
- Method: open-loop re-scoring of the recorded poses (`impact.py`), with no re-simulation.

**Where the policies live:**
- elevation p5–p95 is 50–80° (median 64°);
- plane is 45–117° (forward);
- τ p5/p50/p95 is −66/−15/+55° for v65.

So they work at the forward-flexed directions where the human ER limit is about −50…−60°.

| | v61 | v64 | v65 |
|---|---|---|---|
| beyond the human ER limit for their direction | **8.4%** | **10.0%** | **11.7%** |
| beyond the IR limit | 0.0% | 0.01% | 0.0% |
| > 25% of the half range past the limit | 2.2% | 3.5% | 2.3% |
| degrees past when beyond: median / p90 | 11.6 / 31 | 14.0 / 34 | 9.5 / 27 |
| passive twist torque would exceed full twist gear | 1.7% | 2.8% | 1.6% |
| mean / p95 / p99 passive twist torque (N·m) | 1.8 / 7.8 / 21 | 2.0 / 10.4 / 25 | 2.2 / 8.9 / 20 |
| p99 passive pitch / yaw torque (N·m) | 11.7 / 10.6 | 26.7 / 7.5 | 13.6 / 10.6 |
| beyond, by elevation 45–60 / 60–75 / 75–90° | 5 / 9 / 12% | 2 / 9 / 39% | 7.5 / 13.6 / 13.3% |
| per-replay beyond%: median (min–max) | 8.3 (2.7–15) | 9.8 (4.5–19) | 11.4 (5.6–20) |
| edge-damping power, open loop | 8.9 W | 6.2 W | 8.0 W |

Other observations:
- Twist sits at its 16 rad/s cap in 16% / 13% / 20% of ticks (v61/v64/v65).
- The beyond share is about the same at the cap as overall (8.8–9.3%). The screw happens across the whole range, not specifically at its ER end.
- Cone: 0% outside the sinus (the box is inside it). The yaw box stops are hit in ≤ 0.1% of ticks.
- Re-scoring is open loop, so a trained policy would adapt. These numbers measure how non-human current behaviour is, not post-training cost.

**Sensitivity of the v65 beyond share:**

| variant | v65 beyond% |
|---|---|
| default | 11.7% |
| 90° flexion ER anchor −45° (Wang-style: more ER in flexion) | 3.8% |
| 90° flexion ER anchor −10° (strict Humphries) | 23.1% |
| ER +10° at the side and at 90° abduction | 9.3% |
| s = 0 instead of 0.3 | same beyond share; mean twist torque 2.6 instead of 2.2 N·m |

---

## F. Implementation plan (phase 2, after review)

1. **`rust/src/dynamics/model.rs`**
   - `pub struct PassiveShoulder { arms: [(yaw, pitch, twist); 2] = [(4,3,5), (11,10,12)], coef_lo/coef_hi: [Real; 4], tau_peak, s, kb, wb, uc, b_peak }` and `Model::passive_shoulder: Option<PassiveShoulder>`. It is `Some` on HUMANOID and `None` on the pendulums.
   - `pub fn shoulder_rom(y, p, t, &PassiveShoulder) -> (τ_y, τ_p, τ_t, u, b)` in left-canonical coordinates, transcribing `srom_model.py` (expression tree as in §C.1), plus `shoulder_rom_energy` for tests.
   - Constants: `SROM_FRAC = 0.30`, `TAU_BODY_DEFAULT_ARM_TWIST = 0.8729`, `SROM_TAU_PEAK = (0.30 × 19.45 × 0.8729) as Real` (f64, rounded once, like the wrist), and `SROM_S`, `SROM_KB`, `SROM_WB`, `SROM_UC`, `SROM_DAMP_REF_SPEED = 16`, `SROM_COEF_*` (deg → rad in f64).
2. **`rust/src/dynamics/step.rs`, new step 4d after the wrist's 4c.** Per arm:
   - Read the canonical angles (right: −yaw, −twist) and evaluate.
   - Negate τ_y and τ_t for the right arm.
   - Re-assemble the three rows as `((τ_gear + τ_p) − damping) − b_joint·q̇ − C`.
   - Add the rank-1 implicit damping to `ws.m` (both off-diagonals) before step 5.
   - Telemetry: `ws.shoulder[s] = [τ_axial, u, zone, P_diss]`.
   - Mirror: the right arm reads exactly negated inputs, so its outputs are bitwise the left's. No `sin(−x) = −sin(x)` assumption is needed.
3. **CUDA (`beat_arena_cuda.cu`).**
   - Device `shoulder_rom` (T-templated, `T(double constexpr)` constants), plus the same 4d block.
   - Diagonals at `d.m[TREE_C.fcb[4|6|11|13]]`, off-diagonal at `d.m[TREE_C.fcb[6|13] + 2]` (slot +1 is the pitch row; pinned by the `srom_slots_ok` static_assert).
   - `pop_shoulder` telemetry.
   - Parity test against Rust on random states (pod: no GPU on the laptop).
4. **Python.**
   - `constants.py`: `SROM_*` plus a reference `shoulder_rom()`, with asserts tying T to `TAU_MAX[5] × TAU_BODY_DEFAULT[arm_twist]`.
   - `RawBeatSim.set_passive_shoulder(bool)`, a test-only switch like the wrist's.
   - Replay `header.body["shoulder_rom"]`, bench results key.
   - `[shoulder]` log line, `metrics.jsonl` keys `srom_L/R`, `szone_L/R`, `sout_L/R`, `sdiss_L/R`, and dash panels.
   - Not muscle effort: never in `Body::tau`, the energy price, `[torque]` or the replay torque fields.
5. **Tests (`rust/src/shoulder_tests.rs`, mirroring `tendon_tests.rs`).**
   - Pure-axis profile.
   - FD gradient, including across the pole, u = 0, s, 1 and uc.
   - Closed-loop work, using exact segment integrals of E.
   - Pole consistency, i.e. (y, t) ~ (y + a, t + a) at p = −90°.
   - Anchor reproduction.
   - 100k-pose bitwise mirror.
   - Stiffness budget with the model's M (assert ω·h < 1).
   - Implicit-damping dissipation (never gains energy with T = 0).
   - Release-from-limit vibration and ζ.
   - 60 s hard-driven soaks at 1/240 and 1/480 (finite, 0 pivot-floor, 0 envelope).
   - Exclusion from effort and price.
   - Telemetry wiring.
   - The mirror regressions (`tick_commutes_with_the_mirror`, symmetric-stays-symmetric) re-run with the term on.
6. **Fixture retunes to expect.** Scripted-torque and whip tests that pose the arm at twist values past the new ER limits (as the wrist branch had).
7. **Training.** A fresh trunk, or a fine-tune from v65 (decision 9). Observations are unchanged unless the twist box is widened (decision 6).

---

## G. Open decisions for the user

1. **ER limit at 90° flexion** (the dominant uncertainty). The default is −25°, from Humphries, scored with my convention mapping. Alternatives are −10° (strict) and −45° (Wang-style). v65 beyond share: 23% / 11.7% / 3.8%.
2. **Zone start s.** 0.3 is recommended (matches the human neutral zone). 0 would match the wrist's "always centring".
3. **Peak fraction.** 0.30 (5.09 N·m) is recommended. 0.25 (4.2 N·m) matches the clinical 4 N·m end-feel exactly.
4. **Beyond-limit growth.** kb = 1, wb = 0.25, linear past uc = 1.25, so full gear holds about 1.29 (14–24° past). Stiffer means about 1.15 holding, at a cost in ω·h.
5. **Edge damping.** Reference speed 16 rad/s with onset at s is recommended. Alternatives are damping from 0 (wrist-style) or none.
6. **Twist box.**
   - Keep it (recommended for phase 2): no observation-scale change.
   - Or widen the IR side (left twist lo −90° → about −130°), so that human IR in forward and cross-body postures (τ up to 130° at 90° flexion) becomes reachable. This needs an observation-normalization change.
7. **Yaw/pitch box and cone.** Keep it, with no cone term. Or widen it for extension and horizontal extension, which would then need a soft cone term.
8. **Forearm.** No coupled term. Optionally a separate forearm-roll tendon or strength review, since roll sits on its 0 stop in more than half the ticks.
9. **Training plan.** A fresh trunk, as for the wrist tendon, or a fine-tune from v65, which starts 11.7% outside the human range.

---

## 9. Prototype scripts (`docs/design/shoulder_rom/`)

| script | purpose |
|---|---|
| `verify_coords.py` | Euler ≡ globe ≡ swing-twist identity; forearm directions at key poses |
| `srom_model.py` | reference model: limits, energy, analytic torques, damping |
| `check_grad.py` | FD gradient, closed-loop work, pole consistency, mirror |
| `limit_table.py` | fitted coefficients, anchors, limit table vs the box |
| `box_geometry.py` | today's box τ windows |
| `load_replays.py`, `distrib.py`, `impact.py`, `sensitivity.py` | replay decode (cached npz in the scratch dir), occupancy, impact, sensitivity. `SROM_REPLAYS` sets the replay directory |
| `mass_probe.py`, `stiff.py`, `srom_probe.rs.txt` | M(q) dump via a temporary Rust test, effective inertias, ω·h and damping budget |

Run with `python/.venv/bin/python` from that directory. numpy only; the replay scripts import `beat_arena_env.replay_format`.
