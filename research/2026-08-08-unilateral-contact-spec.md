# Unilateral foot contact (env v11 physics) — normative spec

2026-08-08, user decision: "let's go unilateral. keep 5 gear lower
body." Replaces the two 6-DOF foot welds with unilateral frictional
ground contact. Everything not specified here is UNCHANGED: obs-340,
K=5 gears, effort-as-work, charts, scoring, self-collision, joint
limits. This is a physics-regime break (like the 2026-07-30 speed_scale
removal): checkpoints and bench numbers do not carry across.

## 1. Mode flag

`foot_mode`: 0 = weld (v10, DEFAULT — every existing test keeps
passing), 1 = contact. Plumbed: Rust `BeatSim`/`VecBeatSim` ctor kwarg
`foot_contact: bool`, python `RustBeatEnv(foot_contact=)`,
`CudaBeatVecEnv(foot_contact=)`, trainer `--foot-contact`. In contact
mode the 12 weld rows are dropped from the constraint set.

## 2. Contact geometry

Ground plane y = 0, static. Contact POINTS are the 8 sole-box corners
(SPEC-v10 §1.2 foot box: center (0, −0.035, 0.05), half (0.045, 0.035,
0.125), in the foot frame):

  corner order (NORMATIVE, fixed): foot_l then foot_r; within a foot
  (x−,z−), (x−,z+), (x+,z−), (x+,z+) with x = ±0.045,
  y = −0.070 (sole plane), z ∈ {−0.075, +0.175}.

Gap = world y of the corner. A corner is ACTIVE when gap < MARGIN
(reuse collide::MARGIN = 0.01). Reference pose puts all 8 corners at
exactly y = 0 (ankle at y = 0.07): active and touching at reset.

## 3. Constraint rows (extends the existing PGS block)

Row order per active corner, corners in normative order, appended
where the weld rows used to live (before self-collision normals, so
the existing §2 pair-id order is untouched):

  1 normal row  N: J = d(corner·ŷ)/dv, unilateral λ_n ≥ 0,
                   rhs Baumgarte beyond SLOP = 0.005 (same β pipeline
                   as the self-collision normals, §3 step 8);
  2 friction rows T1 (world x̂), T2 (world ẑ) at the same corner:
                   BOX-clamped |λ_t| ≤ MU · λ_n against the SAME
                   sweep's current λ_n (Gauss-Seidel coupling,
                   re-clamped every iteration), rhs = 0 (no Baumgarte,
                   friction stabilizes nothing).

MU = 0.9. No restitution (velocity-level PGS, rest contact).
Iteration count and sweep order: IDENTICAL to the existing
contact/limit PGS — fixed count, fixed order, no data-dependent early
exit (parity + the K13 warp-divergence rule both require it).
Warm-start policy: same as the existing unilateral rows (whatever
self-collision does today — mirror it, do not innovate).

## 4. Termination (fall)

Contact mode adds one termination: root y < 0.55 m ⇒ episode
terminates, reward 0 for the step, counted and REPORTED like the
root-escape envelope (SPEC-v10 §20e "reported, never hidden"):
`[fall]` line per update in the trainer, `pop_fall_hits()` on the CUDA
env, `falls` counter in Rust info. No other envelope changes.

## 5. Parity contract

Rust is the reference. CUDA transcribes SERIALLY in the same order —
same corner order, same row order, same iteration count, same clamp
sequence. Gate tiers as v10: cargo bitwise suites for Rust; CUDA f64
lockstep tight-tolerance over bounded horizons; f32 well-formedness.
New tests (both engines where applicable):
  a. settle: reference pose, zero gears, 5 s → stands (|root_y −
     0.94| < 0.05, no NaN, all 8 corners stay active);
  b. no-pull: every λ_n ≥ 0 every substep; total vertical contact
     force over a settled second ≈ 736 N (= 75.02 kg · g) within 2%;
  c. static friction: settled stance + 100 N lateral root push for
     0.5 s → root x moves < 1 cm (μ=0.9 holds; 800 N slides it);
  d. fall: hard lean (full +hip_pitch gears 1 s) → termination fires,
     `falls` counts it;
  e. determinism: same seed, same actions, two runs → bitwise-equal
     q trajectories (f64);
  f. weld regression: foot_mode=0 → every existing test bitwise
     unchanged.

## 6. Explicitly out of scope (recorded, not forgotten)

Contact-state observations (blind-feet first, per user's "see if it's
possible"); K=7 lower body; taxing limit-row impulses; foot colliders
vs. self-collision (sole corners only touch the GROUND); non-flat
ground.
