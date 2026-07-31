# Proposal: env v10 — full body, full dynamics, self-collision (one jump)

Status: PROPOSAL — nothing implemented. Written 2026-07-31 after three design
discussions (feet-glued kinematic body, self-collision, and the user's call to
skip intermediate stages: "if we do collision with our existing body system,
we're probably going to need to throw it all out").

## 0. Framing — what the jump skips and what it keeps

The previously discussed ladder was: (A) kinematic pelvis-virtual-root body →
(B) RNEA reaction coupling → (C) full forward dynamics → (D) unglued feet.
This proposal collapses A–C into one env version. What dies with the jump:
the kinematic virtual-root parameterization and the rejection-based collision
response (both version-specific). What survives from the ladder design work:
the capsule collider roster and pair table, the substepping insight, the
skeleton/joint-set choices, the anthropometric mass tables, and the entire
verification philosophy. Feet stay WELDED in v10 (bilateral constraints, not
glue-by-construction) — ungluing (balance, stepping, falling) is v11+ and
requires zero rework under this architecture, because welds and contacts are
the same constraint machinery.

Two-way force flow is not a feature to add — it is what a correct articulated
dynamics solver does by construction: the shoulder reaction of an arm swing is
Newton's third law inside the solver; leg/hip torque pumped through welded
feet adds energy up the chain into hand speed (real whole-body swing
technique). If the solver is right, force flow is right.

## 1. The body (skeleton spec)

Floating base + 29 actuated DoF. Tree, root = pelvis:

| chain | joints | DoF |
|---|---|---|
| pelvis (root) | free joint (3 pos + quat) | 6 (unactuated) |
| spine (pelvis→chest) | waist pitch/yaw/roll | 3 |
| head | welded to chest (no neck DoF in v10) | 0 |
| arm ×2 (chest→hand) | shoulder 3 + elbow 1 + wrist roll/bend/dev 3 | 7 each — **identical joint set to v9** |
| leg ×2 (pelvis→foot) | hip 3 + knee 1 + ankle pitch/roll 2 | 6 each |

Total: 35 velocity coordinates (36 position coords with quat).
Action: `MultiDiscrete([n_gears] × 29)` — same gear semantics, more heads.

Dimensions: KEEP the v9 arm exactly (L_UPPER 0.35, L_FORE 0.30, blade 1.0,
shoulder separation 0.4) and derive torso/legs to reproduce the current
anchors at neutral stance: shoulder height 1.47, head 1.70 → ~1.78 m body.
Continuity of reach geometry is worth more than fresh anthropometry.

Masses/inertia from de Leva (1996) segment fractions, 75 kg reference:
trunk ~32.6 kg, thigh ~10.6, shank ~3.2, foot ~1.0, upper arm ~2.0,
forearm ~1.2, hand ~0.46, head ~5.2. Blade + controller: 0.15 kg point mass
at the grip (the blade itself is a massless hologram — consistent with it
not colliding). Segment inertia as capsules/boxes from the same tables.

Joint limits: arms carry over v9 verbatim; legs/spine from anatomical range
tables (knee 0–150° one-way, ankle dorsi/plantar +20/−50°, etc.) — the
"realistic, fixed law" doctrine.

## 2. Dynamics core

- **Substepped physics under the 30 Hz env**: action/obs/scoring stay at
  DT = 1/30; physics runs N_SUB = 8 substeps at h = 1/240 s. Substepping is
  load-bearing three ways: integrator stability with contacts, tunneling
  elimination (per-substep motion ≪ capsule diameters), and constraint drift.
- **Algorithm per substep**: CRBA → mass matrix M(q) (35×35, symmetric PD);
  RNEA with q̈=0 → bias/gravity forces; dense Cholesky of M; velocity-level
  constraint solve (foot welds = 12 bilateral rows + active contacts) via
  projected Gauss–Seidel impulses, FIXED iteration count (start 12) and FIXED
  constraint ordering — determinism and cross-impl parity require both;
  Baumgarte stabilization for weld drift and contact penetration;
  semi-implicit Euler; quaternion root integrated by exp map.
  (ABA is the O(n) alternative but handles loop constraints awkwardly; at
  n=35 the dense route is ~15k flops for the factorization — cheap — and much
  easier to keep parity-exact. Revisit only if profiling says so.)
- **Actuation**: gear fraction × τ_max[j]. Derive τ_max for continuity:
  τ_max[j] = I_subtree(neutral pose, about j) × α_v9[j], so unloaded joint
  accel matches the current feel (v8 "effective in-game motion" doctrine
  carries over automatically). Examples at 75 kg: wrist ≈ 1.2 N·m
  (I_hand+controller ~0.003 × 400), elbow ≈ 12 N·m (~0.076 × 160),
  shoulder ≈ 65 N·m (~0.4 × 160) — all inside plausible human ranges.
- **ω caps**: keep the hard clamp per joint tier in v10 (dissipative,
  parity-safe, behavioral continuity). Physical motor curves are a later
  refinement. DRAG is replaced by physical joint damping (small viscous term).
- **Gravity ON.** Consequence to socialize: coast no longer holds a pose —
  arms fall. Holding guard position costs continuous torque. This is fidelity,
  and it makes the effort economy real.
- **Effort cost**: replace hand-tuned EFFORT_COST with physical positive
  mechanical work per tick: effort_coef × Σ_j max(τ_j·ω_j, 0). One constant
  instead of a per-joint table.

## 3. Collision (design carried from 2026-07-31 discussion)

- Roster: REAL body segments only — upper arms, forearms, fists (spheres),
  torso capsule/box, head sphere, thighs, shanks. **Blades collide with
  nothing** (real-BS accurate: sabers are holograms).
- Capsule–capsule = segment–segment closest distance minus radii (Ericson
  §5.1.9). Pair table: exclude parent–child; ~25–30 live pairs. Contact margin
  activates constraints slightly before touch.
- Response is now physical: frictionless normal contact impulses in the PGS
  solve (restitution 0). No response-rule design debt — this is the piece the
  kinematic ladder would have thrown away.

## 4. Build vs buy

Recommendation: **hand-roll, MuJoCo as oracle.** Rationale: the triple-mirror
f64 parity discipline (Rust/CUDA/torch, five re-parameterizations, zero
parity-test edits) is the project's proven quality mechanism and no external
engine fits inside it; the torch training regime and CUDA fused-policy path
are load-bearing; owned sim = every past exploit fix was possible because we
own the step function. Cost: this is a small rigid-body engine,
~1.5–2.5k lines per mirror — the largest env build in project history
(~3–5× v6).

Alternative (recorded honestly): adopt MuJoCo/MJX as the physics backend.
Pros: correct contact physics for free, battle-tested. Cons: JAX (MJX) vs
torch integration friction, loses exact-parity testing, loses the Rust bench
FFI path, and the sim stops being ours. Not recommended, but it is the real
fork.

Oracle plan regardless: author the SAME body as an MJCF file. Use MuJoCo for
(a) trajectory tolerance tests (short horizons, tolerance bands — exact match
across engines is impossible), (b) static equilibrium and gravity-load
checks, (c) sanity on contact behavior. Exactness lives in our own f64
triple parity, as always.

## 5. Verification ladder (gates between phases)

1. Spatial-algebra + FK unit tests; M symmetric PD; RNEA(q,0,0) = gravity
   torques; CRBA-vs-RNEA consistency (M columns via RNEA unit accels).
2. Analytic dynamics: single + double pendulum vs closed form / high-precision
   reference; energy conservation (gravity on, no damping, no contacts) to
   integrator order; momentum conservation (gravity off, free floating base).
3. Constraint tests: welded-feet static stance holds under gravity with zero
   actuation drift; contact non-penetration bound; Baumgarte doesn't inject
   energy above threshold.
4. MuJoCo oracle comparisons (tolerance bands).
5. f64 exact parity Rust↔CUDA↔torch (fixed PGS order/iterations make this
   attainable); the existing identical-expression-tree discipline.
6. Scripted bot clears easy charts (solvability gate before any training).

## 6. Env-layer changes

- Obs: per-joint q (cos/sin where wrapping) + normalized q̇; root pose
  (position + 6D orientation) + root lin/ang velocity; head position (drives
  look-at + future walls); per-hand FK wrist/blade features as today; contact
  summary (start minimal: per-arm contact bit). Ballpark 380–420 f32 total
  with the unchanged 16×14 note block + mask. Exact layout at spec time.
- Policy: 29 heads (fused GEMM reshape generalizes); token layout decision at
  spec time — recommend adding ONE body token (root/spine/leg state) beside
  the two hand tokens; note tokens unchanged.
- Replay v10: q(36) + root twist + contact events per tick; viewer does full
  FK. ALL v9 checkpoints/replays retire (established pattern). Fresh
  training, no warm start (head-count and obs both change; retraining is the
  cheap part by every ledger lesson).
- Scripted bot: rewrite as gravity-compensated PD controller tracking IK
  targets (arms: existing two-link IK; legs: stance targets). PD-over-IK is
  the standard and reuses kinematics.py. Nontrivial tuning — budget for it.
- Scoring/notes/generator: UNTOUCHED. REACH_Z / Z_MISS become functions of
  live head/shoulder pose instead of constants.
- Crouch walls: NOT in v10 (one variable at a time — the body is the
  variable). They are the natural v10.1 once the body exists; under dynamics
  legs already have pressure (stabilize reaction wrenches, pump energy) even
  welded.

## 7. Implementation phases

- **P0 — spec freeze**: joint table, limits, masses, capsule set, τ_max
  derivation, obs/action layout, replay format. MJCF twin authored here.
  AskUserQuestion menus on the open decisions (§8).
- **P1 — Rust core math**: FK (generalized tree replaces hand-rolled chains),
  CRBA, RNEA, Cholesky, integrator. Gate: verification 1–2.
- **P2 — constraints + contacts in Rust**: capsule distance, pair table,
  PGS, welds, Baumgarte, substepping. Gate: verification 3–4.
- **P3 — env integration in Rust**: 30 Hz wrapper, gear→τ, obs writer,
  effort-as-work, replay v10. Existing scenario tests re-authored for the
  dynamic body. Gate: sim plays a chart with scripted torques.
- **P4 — torch mirror** (training home first): same expression trees, f64
  parity suite. First smoke training possible after this phase.
- **P5 — CUDA mirror + fused-policy update**: the biggest port line-item;
  one-thread-per-hand becomes per-body-partition or serial-per-env — decide
  by profiling, not upfront. Can trail P6 if torch throughput suffices for
  first runs.
- **P6 — viewer + bot**: full skeleton, contact markers, per-joint τ/ω
  gauges + foot reaction-force display (the gauge culture has caught model
  errors twice — instrument generously); PD bot; solvability refs.
- **P7 — first runs**: 50M smoke → 400M protocol; registered predictions per
  house rules.

## 8. Open decisions for the user (P0 menu material)

1. Hand-roll + oracle (recommended) vs MJX/MuJoCo backend.
2. Feet welded in v10 (recommended) — unglue as v11?
3. ω hard-clamp continuity (recommended) vs physical motor curves now.
4. Spine 3-DoF (recommended) vs 2 vs 0; neck fixed (recommended)?
5. Reference body mass 75 kg / height from current anchors (recommended) vs
   re-derived anthropometry.
6. τ_max continuity derivation (recommended) vs biomechanics-table torques.
7. 30 Hz control retained (recommended; 60 Hz axis still unexplored — keep
   orthogonal).
8. Torch-first training with CUDA trailing (recommended) vs CUDA-gated start.

## 9. Performance expectations (to be benchmarked in P1, not promised)

Per env-step cost ≈ 8 substeps × (CRBA+RNEA ~O(n) + 35×35 Cholesky ~15k flops
+ PGS 12 it × ~40 rows) ≈ low-hundreds of kflops — roughly 3–10× the current
kinematic step. MJX/Brax precedent says thousands of batched humanoids on one
GPU is routine. Expect first-run throughput in the several-k steps/s range
rather than 20k+; treat a P1 microbenchmark as a gate before committing to
the CUDA port order.

## 10. Risks

- Scope: largest env build to date; the phase gates exist so failure is
  caught at the math layer, not after a 400M run.
- PGS in f32 on GPU: convergence/parity risk — mitigation: f64 tests, fixed
  iterations/order, Baumgarte tuning tests.
- Bot solvability under gravity+torque control may lag badly at first (PD
  tuning); training has repeatedly beaten the bot anyway, but the solvability
  gate is still the honest check.
- Behavioral discontinuity: gravity + effort-as-work changes the game's
  economy; all v9-era intuitions about effort_coef need re-tuning.
- The 6-initializer CUDA zero-fill class of bug: generalized tables (no more
  hand-counted arrays) are themselves the mitigation.
