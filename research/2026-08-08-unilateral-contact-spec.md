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
terminates, counted and REPORTED like the root-escape envelope
(SPEC-v10 §20e "reported, never hidden"): `[fall]` line per update in
the trainer, `pop_fall_hits()` on the CUDA env, `falls` counter in
Rust info. No other envelope changes.

Fall reward (v11.1, 2026-08-09 — replaces the original "reward 0"):
the fall step's reward is REWARD_MISS (−0.5) × the count of scoreable
notes still pending (status 0, kind ≠ bomb) at the fall tick — the
lump-sum cost of lying on the floor missing the rest of the chart.
Pending-note STATUSES ARE NOT REWRITTEN (judgment counters stay
actually-judged; the fall is reported via its own counter). Rationale:
the 8h run proved reward-0 termination is an escape hatch that bounds
the miss stream — ep_return converged to −0.13 (mean episode 1.2 s,
death before first_note_s) and held for 3.4B steps. Pricing the
remaining chart removes the bound exactly, with no tuned constant, and
keeps episodes short. Root-escape termination keeps its reward-0
semantics unchanged (physics violation, not game failure).

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

K=7 lower body; taxing limit-row impulses; foot colliders vs.
self-collision (sole corners only touch the GROUND); non-flat ground.
(Contact-state observations were out of scope for v11's blind-feet run
and are now IN scope — §7.)

## 7. env v12 (2026-08-09): contact observations + survival gradient

Added after the v11.1 run's falls plateaued ~5,100/update with episodes
ending well before the first note: blind feet gave the policy nothing
to balance WITH, and the v11.1 fall pricing (constant before the first
note) gave survival time no gradient to climb.

a. **Ground-contact obs.** OBS_DIM 340 → 344. Lanes 340..343 =
   [L_heel, L_toe, R_heel, R_toe]: the tick-summed ground NORMAL
   impulse (N·s) of each foot's back/front sole-corner pair (§2 corner
   order: odd k = z+ = toe), divided by GROUND_IMP_NORM = 24.5 — the
   spec literal ≈ M_TOTAL·g·DT (24.53 N·s), so a settled two-foot
   stance sums to ≈ 1.0 across the four lanes. All-zero in weld mode
   (lanes 0..340 stay the v10 obs verbatim); zeroed on reset; a
   set_state refresh reads the LAST tick's sums in both engines. The
   policy feeds the four lanes to the BODY token (BODY_FEAT 50 → 54)
   in policy.py and both packed CUDA policy kernels.

b. **Survival gradient.** ALIVE_BONUS = 0.002 per tick, contact mode
   only, added AFTER the follow-through drain and BEFORE the
   escape/fall overrides — a terminal fall/escape tick never collects
   it. Sizing: 0.06/s ≈ +11 over a 3-min chart, an order under the
   note stream (~|70|+), so it cannot out-bid hitting notes or revive
   the loiter exploit; it exists to make "die at 1.9 s" beat "die at
   0.5 s" before the first note is ever reachable.

c. **Parity.** The §5 tiers gate the new lanes as part of the full obs
   and the bonus as part of reward parity; a settled-stance lane test
   (tests/test_v11_contact_parity.py, CUDA; tests/test_v12_contact_obs.py,
   Rust) pins the bucket ORDER — nonzero lanes on both sides, so a
   permutation cannot hide in zeros. Weld regression unchanged
   (test_v10_parity_cuda.py), new lanes constant 0 there.

## 8. env v13 (2026-08-09): explicit body-point observations

User-directed FK shortcuts: the engines compute every frame position
each tick; handing six of them to the policy spares network capacity
for control instead of kinematics.

a. **Lanes.** OBS_DIM 344 → 362. Lanes 344..361 = world positions of
   [l_elbow, r_elbow, l_knee, r_knee, l_foot, r_foot] (frames 7, 14,
   21, 27, 23, 29 — knee = shank origin, foot = ankle origin), three
   lanes each, head-style normalization x/1, (y−Y0)/0.75,
   (z−PLANE_Z0)/1. Y0: elbows share Y_CENTER = 1.40 (they live in the
   note field); KNEE_Y0 = 0.50 (hip 0.94 − thigh 0.44); FOOT_Y0 = 0.07
   (§2 ankle reference height). Present in EVERY mode (weld included —
   positions are mode-independent); reference-pose values pinned by
   tests/test_v13_body_pts.py.

b. **Tokenization.** Each elbow goes to its own saber token
   (SABER_FEAT 25 → 28); knees + feet go to the body token (BODY_FEAT
   54 → 66). Mirrored in policy.py and both packed CUDA policy kernels
   (packed FRONT 18752 → 19712).

c. **Parity.** The §5 lockstep tiers gate the lanes as part of the
   full obs; the Rust-side reference-pose test pins frame choice,
   order, and normalization literals exactly.
