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

## 9. env v14 (2026-08-10): reciprocal joint stops — the phantom-force repair

A three-Opus-agent audit (user-triggered, on the v13 trestle pose)
proved the step-11 post-solve clamp block manufactured external force:
the direct-write guarantee pass (v ← 0 at stops, no M⁻¹e_j reaction)
and the routine position teleport (q ← clamp(lo,hi)) leaked momentum
whenever joints rode their stops — +250-335 N (27-46% of body weight)
sustained under the trained pose, 15.6 m of free-fall deviation over
4 s mid-air. The trained policy had converged INTO the bug (pinning
8-13 joints maximizes clamp traffic). v11-v13 contact results are
void in the braced regime.

The v14 contract, both engines:
a. **No reaction-free writes.** The guarantee pass is deleted; the
   position teleport survives only as a deep-violation backstop
   (LIMIT_TELEPORT_BACKSTOP = 0.35 rad — unreachable by one-substep
   overshoot, root-envelope class). CLAMP_PASSES 8 → 64 (early-out
   keeps typical substeps unchanged).
b. **Stops are reciprocal-impulse springs.** The step-11 rate target
   at a violated stop is a Baumgarte recovery toward the bound
   (BETA_LIMIT = 0.2) with a rest dead-band (SLOP_LIMIT = 2 mrad):
   inside the slop a parked joint targets 0 and fires NO impulses, so
   steady stop-resting leaks nothing. All corrections flow through
   Δv = M⁻¹e_j·λ.
c. **Semantics.** Transient penetration ≈ h·(cap + h·τ/M) on arrival
   (≤ ~0.12 rad legs, ~0.24 rad wrists), rest penetration ≤ 2 mrad.
   The P1 path (no solver limit rows) under sustained full torque is a
   bounded limit cycle around the stop; the env path (solver rows)
   parks inside the slop.
d. **Gates.** tests_dyn `v14_momentum_conserved_with_joints_parked_on_
   stops` (airborne press-and-park: CoM rides the free-fall parabola
   to < 10 mm / 2 s; measures 3.4 mm — vs ~0.3-3 m pre-v14) and
   tests_ground `v14_settled_cop_matches_com_projection` (CoP inside
   the hull and within 4 cm of the CoM projection — the moment-balance
   gate the suite never had). KNOWN separate effect, not covered: the
   explicit-Coriolis discretization drift at cap-speed flailing
   (~1.6-3% g, ∝ v², the integrator's documented h·ω bound) — real,
   velocity-dependent, vanishes when quasi-static, orthogonal to this
   repair.
e. v11-era trajectory gates re-pinned where semantics moved (limit
   crossing tolerance 1e-9 → 2.1e-3; P1 full-press = bounded cycle).

## 10. env v15 (2026-08-10): head-anchored note track

User-identified exploit in the first honest-physics (v14) replays: the
policy shimmied forward in z to meet notes early — the track targets a
FIXED world plane (hit plane z = 0) while the body stands at ~−0.65,
so every step forward was real distance bought and earlier reward.
Real Beat Saber is room-anchored: you cannot walk the notes closer.

Fix (user-specified semantics): every note's world z carries the live
offset `z_off = head_z − PLANE_Z0` (zero at the reference pose, where
head z = −0.65). Walking toward the notes shifts the whole track away
by exactly the walked amount.

a. **Sites** (both engines, three each): the note obs z lane (offset
   added before Z_CLIP), the broad-phase sorted-break bound (z_off is
   shared by every note, so the bound shifts with it and the
   sorted-by-hit_time argument is unchanged), and the judgment z.
   x/y stay world-fixed (lanes are body-independent).
b. **Consequences.** Miss timing becomes a pure function of chart
   time: z < head_z ⇔ base z < PLANE_Z0, i.e. 0.65/speed seconds
   past hit_time, invariant to any body motion. Leaning forward now
   costs reach (the head's full z displacement moves the track away;
   the shoulders move less) — leaning-in as a reach tactic is gone
   along with walking. Head bob makes the track jitter by the same
   mm — harmless, and faithful to the head-anchor semantics.
c. **Gates.** tests/test_v15_note_offset.py (Rust reference): per-tick
   obs identity note_z = base + (head_z − PLANE_Z0) under live tipping
   (offset exercised > 2 mm), and analytic miss-tick invariance under
   a spine lunge. CUDA is gated by the §5 lockstep tiers (note lanes
   are full-obs; judgment feeds reward). Python constants export
   PLANE_Z0.
d. Physics untouched — this is a scoring/obs change; v14 checkpoints
   remain physically valid but their walk-forward habit is
   reward-dead under v15.

## 11. env v16 (2026-08-12): discounted fall pricing + segment lead

Motivated by the 8h-trunk survival-drift postmortem (leaf value
peaked at upd ~8k then decayed while balance kept improving; user
suspects #1 and #4 pre-registered in sessions/2026-08-10-wsd).

a. **Discounted fall pricing.** The v11.1 lump sum is return-
   equivalent to standing-and-missing, but NOT under discounting: it
   lands the whole remaining penalty undiscounted at the fall tick —
   a manufactured risk premium on reaching. v16: the fall charges
   REWARD_MISS × Σ_pending fall_gamma^k_i, k_i = min(ceil(max(0,
   t_miss_i − t_now) / DT), 600), t_miss_i = hit_time + (−PLANE_Z0)/
   speed (the §10b miss boundary). `fall_gamma` is a sim ctor kwarg
   (RawBeatSim / VecBeatSim / RustBeatEnv / BeatVecEnv /
   CudaBeatVecEnv), DEFAULT 1.0 = the v11.1 count bitwise (every
   factor is 1.0); the trainer passes its --gamma so the charge is
   return-equivalent under the agent's own discount. Sequential
   multiply in chart order, cap 600 ticks (0.965^600 < 5e-10):
   deterministic, transcribed identically by the CUDA mirror
   (identical f64 division/ceil/multiply order — parity-safe without
   pow). Gates: tests/test_v16_fall_pricing.py (formula identity at
   1e-12; gamma=1.0 exactness); §5 lockstep tiers gate the mirror.
b. **Segment lead (`lead_s`).** ranked_pool's k>0 window head gap
   (was the LEAD_S=1.5 literal) is now a parameter, plumbed as
   trainer --lead-s (default 1.5, cache-keyed). Hypothesis #4:
   falls spike → more pre-first-note episodes → data mix drifts to
   survival → consolidation consolidates survival; a shorter lead
   keeps note data in the mix. Probe value: 0.5.
c. Scoring, obs, physics all untouched; OBS_DIM 362 unchanged.

## 12. env v17 (2026-08-13): walls + distance-sorted entity window

User-driven after reviewing v16 leaf replays. Two coupled changes:
real-map obstacles enter the sim, and the note window becomes a
shared distance-sorted entity window so walls and notes ride the
same 16 slots (one transformer token stream, same self-attention).

a. **Wall geometry (ScoreSaber watch parity).** Parsed from all
   three schemas exactly as research/vendor/scoresaber-watch does:
   v2 `_obstacles` (`_type` 0 → y=0,h=5 full wall; `_type` 1 →
   y=2,h=3 crouch wall; ME `_type` ≥ 1000 skipped — ME/Noodle stay
   out of scope exactly like notes), v3 `obstacles` {b,x,y,d,w,h},
   v4 index-indirected obstacles+obstaclesData. Conversion
   (WallManager.cs): y clamped (0..2 v2/v3, 0..4 v4), h capped at
   5−y; world width = w·0.6 (LaneWidth), world height = h·0.6
   (WallHScale); center x = −0.9 + x·0.6 + (width−0.6)/2, center
   y = y·0.6 + height/2; |width|,|height| floored at 0.06
   (MinWallSize); ClampPlayerHeight: bottom edge ≥ 0, top ≤ 3 m
   (shrink + recenter); negative durations flipped; length =
   duration_s · NJS, floored at 0.06 m. Front face runs 0.25 m
   BEHIND the note plane (viewer front-face alignment quirk), so
   the wall front reaches the head 0.4/NJS s after its nominal
   time. Walls are head-anchored in z like notes (§10): dz_front =
   (t_start − t)·speed + 0.4 is head-z-invariant; the wall spans
   dz ∈ [dz_front, dz_front + length].
b. **Head-in-wall penalty.** Head CENTER point vs wall AABB each
   tick (no sweep): inside iff dz_front ≤ 0 ≤ dz_front + length and
   |head_x − cx| ≤ width/2 and |head_y − cy| ≤ height/2. Each tick
   inside ANY wall: reward += WALL_PEN = −0.05 (once per tick, not
   per wall — overlapping walls are one hazard), event code 7,
   `walls_in` episode counter. No termination, no combo/score
   interaction, no saber/wall collision.
c. **Distance-sorted shared entity window.** The 16 obs slots are
   filled nearest-first by 3D distance from the head center, notes
   and walls in one pool. Note distance: Δx = x − head_x, Δy = y −
   head_y, dz = (hit − t)·speed + 0.65 (head-invariant; 0 exactly
   at the §10b miss boundary = the head plane). Wall distance:
   clamp-based nearest point on the AABB (per-axis separation,
   x/y world, z via dz_front/length) — cheaper and tighter than
   any corner. Entities BEHIND the head consume no slots: notes
   with dz < 0 (⇔ t > t_miss, already judged), walls with
   dz_front + length < 0. Note scan early-terminates via the
   sorted chart: once (hit_i − t)·min_speed exceeds the current
   16th-best distance no later note can qualify. Ties break by
   chart index (stable) — the CUDA mirror transcribes the same
   comparison order for §5 parity.
d. **Token layout.** ENT_FEAT 14 → 18: the v9 note lanes verbatim
   (x, y, z, vz, colorL, colorR, cosθ, sinθ, is_bomb,
   is_chain_head, is_chain_link, free_pre, free_post, valid) then
   [is_wall, width/X_NORM, height/Y_NORM, length/Z_SCALE]. Wall
   tokens: x = center_x/X_NORM, y = (center_y − Y_CENTER)/Y_NORM,
   z = front-face obs z ((t_start−t)·s − 0.25 + z_off)/Z_SCALE
   (same frame as note z lanes), vz = −speed/Z_SCALE; color/dir/
   kind/free lanes zero. OBS_DIM 362 → 426 (100 + 16·18 + 16 mask
   + 4 ground + 18 body pts). NEW ERA: v16-and-earlier checkpoints
   cannot resume or warm-start (input width changed).
e. Sources: real BeatSaver maps only — the procedural generator
   emits no walls (recorded for later). Replay JSONL header gains
   a `walls` array (viewer renders translucent boxes on the
   anchored track).
