# Beat Arena — Interface Specification (v0)

Single source of truth for all components. Do not deviate from constants, layouts, or file formats without updating this file.

## Purpose

Toy RL environment modeled on Beat Saber: timed note blocks approach a cut
plane; the agent controls two sabers (left/red, right/blue) and must swing
through each note fast enough, in the note's required direction, with the
matching hand, inside a timing window. Built to learn **temporal/sequence
transformers** and per-tick motion control under a schedule. Pure
Python/NumPy sim, Gymnasium API, PufferLib hookup, PPO harness.

Milestones: v0 (easy one-hand charts, pool baseline + transformer both train)
→ v1 (two hands, direction variety, flow patterns) → v2 (difficulty
curriculum + generalization eval) → v3 (parse real BeatSaver `.dat` maps,
sight-read eval) → v4 (dynamic gym: continuous angles, per-note speeds)
→ v5 (blade sabers: wrist rotation DOFs, real saber length, 115-point
graded swing scoring; v5.3 = notes are rotated boxes)
→ v6 (kinematic upper body: shoulder→elbow→wrist arm chains)
→ **v7 (real Beat Saber geometry & judgment: real lanes/rows/hitboxes,
swept-surface collision, integer 115, bombs/chains/arcs — current)**.
v7's constants are sourced from the ArcViewer + BeatLeader code
inventory: `research/2026-07-22-bs-mechanics-inventory.md`.

## Layout

```
BeatSaberRL/
  SPEC.md
  README.md
  python/
    beat_arena_env/      # chart gen, constants, gym wrappers, FK mirror
    policy.py            # NotePolicy (transformer)
    train_batch.py       # batched CPU PPO (the only trainer)
    make_top10_replays.py # checkpoint replay recorder (Rust event stream -> JSONL)
    tests/               # pytest
  rust/                  # beat-arena-rs: THE sim implementation
```

**Single-implementation policy (since 2026-07-20, user direction):** the
Python sim, the pool encoder, and the serial trainer were deleted to cut
maintenance surface. `rust/src/lib.rs` is the only physics
implementation; `beat_arena_env/constants.py` mirrors its constants for
tooling and this SPEC remains the written contract. Rule changes land in
lib.rs and are gated by the scenario tests in `tests/test_env.py`
(which drive the Rust sim directly via its `set_joints` hook and event
stream).

Uses the repo venv `python/.venv` (torch-cpu,
gymnasium, pytest; see README Quick start). No new dependencies in v0. No GPU;
paths contain a space — quote everything.

## Sim rules (`rust/src/lib.rs`) — env v7: real geometry & judgment

Env v7 (2026-07-22, research-driven): the v6 body plays in the REAL
Beat Saber world. Geometry, hitboxes, scoring arithmetic, and the new
object types are sourced from the ArcViewer/BeatLeader inventory
(`research/2026-07-22-bs-mechanics-inventory.md`). All v6
checkpoints/replays retire (hard replace).

- Fixed tick `DT = 1/30` s. One policy action per tick.
- **World in METERS, floor-based** (y = 0 is the floor). The hit plane
  is `z = 0`; notes fly in from +z. 4 columns on the real **0.6 m lane
  pitch**: column x = `(col−1.5)·0.6` ∈ {−0.9, −0.3, 0.3, 0.9}. 3 rows
  at the real **non-uniform at-player heights** y = 0.85 / 1.40 / 1.90.
  Notes spawn at their final heights (the real rise-in-flight gravity
  arc is documented in the research doc, deliberately not modeled yet).
  `BLOCK = 0.5` m is now VISUAL only (the real game's cube; 0.4 until
  2026-07-30) — collision uses the real hitboxes below. Objects carry explicit world positions `(x, y)` in the chart
  (chain links sit off-grid on their Bézier).
- **Body (v6 chains, real placement)**: a 1.8 m player — shoulder
  anchors at `(∓0.2, 1.47, −0.65)` (0.4 m shoulder width, head at the
  real 0.65 m cut-plane distance); `L_UPPER = 0.35` m, `L_FORE = 0.30`
  m, `SABER_LEN = 1.0` m.
- **Joints** per hand, in order `q = (pitch, yaw, twist, elbow, roll,
  bend)`; all-zero = arm straight forward (+z), blade its extension:
  - `pitch`, **env v7.2: `[−90°, 65°]`** (+ = arm up, about x —
    straight-overhead arms are gone; straight-down unchanged).
  - `yaw`: + = toward +x; outward 100° / cross-body 45° per hand →
    right `[−45°, 100°]`, left `[−100°, 45°]`.
  - `twist` (humeral rotation), **env v7.1: right `[−70°, 90°]` / left
    mirrored** (outward rotation keeps its full range, inward capped):
    rotates the arm about its own axis, orienting the elbow's bend
    plane; doesn't move the elbow.
  - `elbow` (flexion) ∈ [0°, 150°]: hinge; curls toward the arm-local
    +y (twist picks the world direction of that).
  - `roll`/`bend`: the v5.2 wrist in the FOREARM frame — **env v7.1
    (user-specified, found via the pose tool): roll right `[−180°, 0°]`
    / left mirrored** — the old 280° arcs were compensation for the
    missing arm joints and let training cheat; with bend +90° a wrist
    roll now covers only a 180° blade fan, so 100°+ swing arcs must
    come from the arm. Bend ±95°. Blade dir =
    `M_f · d(ψ, φ)` with `d(ψ, φ) = (−sinψ·sinφ, cosψ·sinφ, cosφ)`:
    the neutral blade is the forearm's extension (v5.2's law
    generalized to a moving forearm).
  - `dev` (env v9, user-caught gap): **wrist radial/ulnar deviation**
    — the orthogonal bending axis roll+bend lacked. Range
    user-specified in RIGHT-hand terms: right `[−15°, +30°]`, left
    mirrored `[−30°, +15°]` (deviation is antisymmetric under the
    body-midplane mirror, like yaw/twist/roll; obs normalizer 30°). Blade dir = `M_f · Rz(ψ)·Ry(v)·Rx(−φ)·e_z`; dev = 0
    reproduces the v5.2-v8 wrist exactly, and at bend = 0 deviation
    moves the blade at full rate (the roll/bend spherical
    parametrization had out-of-plane speed `roll_rate·sin(bend)` — a
    pole singularity at straight wrists).
  - **Forward kinematics** (the contract; `kinematics.py` and the
    viewer mirror it): arm frame `M = R_y(yaw)·R_x(−pitch)·R_z(twist)`
    (local +z = upper arm); elbow `E = S + L_UPPER·M·e_z`; forearm dir
    `f = M·(0, sin e, cos e)`; wrist `W = E + L_FORE·f`; wrist frame
    `M_f = M·R_x(−elbow)`.
- **Tiered dynamics** (env v8, 2026-07-30, user-directed — the v6
  anatomical values DOUBLED on both axes): shoulder + elbow `α = 160`
  rad/s², `|ω| ≤ 16` rad/s; wrist roll + bend `α = 400`, `|ω| ≤ 40`
  (was 80/8 and 200/20 — calibrated to EFFECTIVE in-game hand motion:
  real players beat biomechanics via controller-tracking prediction).
  Per joint per tick: `ω += frac·α·DT` (frac from the gear table),
  `ω *= (1 − DRAG)` (`DRAG = 0.02`), clamp, `q += ω·DT`, limit-clamp
  (zeroing ω). Start pose: all zeros, at rest.
- **Action** per tick, per hand: 7 joint heads with K = `gears` options
  each (env v9; `MultiDiscrete([K]×14)`, left hand first, joint order as
  above, no masks). Gear table (`constants.gear_table`): id 0 = coast,
  1/2 = `±α` at EVERY K (the ternary era is the K=3 prefix — old action
  streams stay valid); finer ± pairs append:
  K=5 adds `±α/4`, K=7 adds `±α/3` and `±α/8`. Effort cost scales by
  |fraction|. Checkpoints stamp `n_gears` in `arch`; env and policy
  tables must match (trainer enforces).
- **Objects** (env v7): chart = list of `(hit_time, x, y, color, angle,
  speed, any_dir, kind, free_pre, free_post)` sorted by `hit_time`;
  positions in meters; **`z(t) = (hit_time − t) · speed`**. `kind`:
  0 = note, 1 = bomb, 2 = chain head, 3 = chain link. `free_pre` (arc
  tail) grants the full 70 pre-swing; `free_post` (arc head) grants the
  full 30 follow-through at the cut tick.
- **Swing tracking** per hand (feeds the score): each tick, with
  `Δ = ∠(d_prev, d_cur)` (3D geodesic) and rotation axis
  `a = d_prev × d_cur` (normalized):
  - `Δ < SWING_BREAK = 0.005` rad/tick → the swing chain **breaks**
    (accumulator `swing_acc` resets to 0).
  - stored axis exists and `a · a_stored < 0` → **reversal**:
    `swing_acc` restarts at `Δ`.
  - otherwise `swing_acc += Δ`; axis stored. `swing_acc` therefore holds
    the arc of the current continuous same-direction swing, INCLUDING
    the current tick.
- **Hitboxes** (env v7 dims, **env v7.6 orientation**): the real boxes,
  NOT the visual cube, on the note's FULL transform. The frame = the
  cut-direction z-roll (`rot = angle − 270°`; arrow notes/chain heads
  by their angle, chain links by their tangent, dots roll-free —
  env v7.4, user-caught) **blended toward FACING THE PLAYER'S HEAD as
  the note approaches (env v7.6, the ArcViewer/BeatLeader look-at)**:
  head anchor `HEAD = (0, 1.70, −0.65)`, head.y damped 80% toward the
  note's y (`LOOK_VERT`), forward nlerped from +z to the head→note ray
  with weight `λ = clamp(1 − z/LOOK_DIST(8), 0, 1)` (our analog of
  jump progress — full head-facing at the plane), roll riding as the
  frame's up. λ = 0 (chain heads/links, far notes) is exactly the
  v7.4 z-roll frame. Plain notes only (dots billboard too, roll-free);
  bombs are spheres. The good box's 0.25 m shift is along the note's
  LOCAL z (it extends toward the head for edge notes — early cuts
  reach farther there). No freeze-after-pass (moot at Z_MISS 0.65)
  and no spawn-settle info timeline (deliberately skipped). Boxes:
  - **Good-cut box** (matching saber only): `0.8 × 0.5 × 1.0` m,
    center shifted `0.25` m toward the player — spans z
    `[note_z − 0.75, note_z + 0.25]`, so early and off-center swings
    connect. Dot notes are `0.8` tall (modern "new dots"); chain links
    `0.2` tall.
  - **Small box** (wrong-color saber only): `0.35³` (links
    `0.35 × 0.1 × 0.35`), centered — smaller than the visual cube, so
    off-hand corner grazes are forgiven.
  - **Bomb**: sphere `r = 0.18`, either saber.
- **Saber sweep** (env v7): the full swept SURFACE between the previous
  and current blade segments — quad `(w0→t0, w1→t1)` split into
  triangles `(w0, t0, t1)` and `(w0, t1, w1)` — tested against the
  boxes with the Akenine-Möller triangle-AABB SAT (no sampled-segment
  tunneling; thin chain links stay hittable). An object is testable
  while `−Z_MISS ≤ z ≤ REACH_Z = 1.75` (max wrist z 0.0 + blade 1.0 ≥
  good-box near face at `note_z − 0.75`).
- **Contact point & velocity**: `s*` = the blade sample
  (`N_BLADE_SAMPLES = 8`) whose displacement segment passes closest to
  the object center; `v_c = (p1(s*) − p0(s*)) / DT`. `|v_c| ≥ V_MIN =
  1` m/s or the cut is `slow`. **Direction check in the NOTE'S LOCAL
  frame (env v7.6)** — the game transforms the cut velocity by the
  note rotation: `cosv = −v_ly / ‖(v_lx, v_ly)‖ ≥ 0.5` where
  `v_l = frameᵀ·v_c` (required direction ≡ local −y'); at λ = 0 this
  is exactly the old world in-plane check. Dots and chain links
  exempt. Near the plane the billboarded frame folds forward velocity
  into local x — forward-heavy swings at edge notes lose direction
  credit, as in the real game.
- **Judgment order** per pending object: bombs — either saber's sweep
  within `0.18` of center → **bomb hit** (event code 6). Notes/heads/
  links — matching saber vs good box → judged (slow/angle/good);
  matching saber missed → wrong-color saber vs SMALL box →
  `wrong_hand`. Any cut consumes the object.
- **Score** per good cut — real integer Beat Saber points, every
  component `floor(x + 0.5)`-rounded:
  - **pre-swing** `round(70 · min(swing_acc / 100°, 1))`; `free_pre`
    (arc tail) → flat 70.
  - **accuracy** `round(15 · (1 − clamp(d / 0.3, 0, 1)))` where `d` =
    distance from the object center to the **infinite cutting plane**
    — the plane through the blade spanned by the blade direction and
    the tip's tick movement (`|n̂·(center − wrist)|`,
    `n̂ = blade × tip_move` normalized), the real game's
    `cutDistanceToCenter` (replays record `cutPoint` + `cutNormal`:
    the plane is the primitive). Timing-independent: an EARLY cut
    through the deep box still scores full accuracy if the swing
    plane is laterally centered. Degenerate stab (tip motion
    ~parallel to the blade, `‖n‖ ≤ 1e-9`): distance to the blade LINE
    instead. **Corrected 2026-07-29 (user-caught, v7.4-class)**:
    v7.0–v7.10 measured the closest approach of the one-tick swept
    surface, which conflated depth/timing with lateral error and
    floored early cuts at ~0 — the source of the 0.870 (= 100/115)
    accuracy ceiling. Bench results carry `scoring_version` (2 since
    this fix); versions never merge.
  - **follow-through** `round(30 · min(post_arc / 60°, 1))` — tracker
    unchanged (break/reversal ends it, `FOLLOW_TIMEOUT = 15`);
    `free_post` (arc head) → flat 30 paid at the cut tick, no tracker.
  - **Chain heads**: pre + accuracy only, max 85 — no follow-through.
  - **Chain links**: flat 20 on any valid cut (color + speed;
    direction exempt) — no pre/acc/post.
- **Reward per tick**: good cut pays `pts/115` (so a link is +20/115, a
  head ≤ 85/115); bad cut `−0.5`; miss `−0.5`; **bomb hit `−1.5` (env
  v7.8, 2026-07-24, user-priced)** — a deliberate proxy for the
  UNMODELED combo-multiplier break (a break costs ~4 notes of
  multiplier on ScoreSaber; at `−0.5` a bomb was the CHEAPEST gym error
  — miss = penalty + ~1.0 forgone points, bomb = penalty only — and
  two very different policies equilibrated at ~12% bombs eaten,
  budget-invariant). A bomb passing by is free and silent. A
  rotation-free "stab" scores only accuracy (≤ 15/115) — swinging is
  the game. **Effort cost (env v7.9, 2026-07-24, user-designed)**:
  every non-coast joint action costs its tier per tick — shoulder
  pitch/yaw/twist `2e-4`, elbow `1.3e-4`, wrist roll/bend `0.7e-4`
  (`EFFORT_COST`), scaled by the env's `effort_coef` (default 1.0;
  0 = free movement). **RETIRED 2026-09-13** (with its v10 work-price
  successor): the live movement prices are `--energy-coef` (env v43 torque energy, Σ|τ|; replaced `--iso-coef`)
  and `--limit-coef --limit-arms` (limit rest); the text below is
  history. A metabolic proxy (proximal joints move more
  mass): calibrated so one accuracy point (1/115 ≈ 0.0087) dwarfs the
  effort of earning it — a typical played episode's total effort is
  ~0.5-1.0 against a ~+40 return; stillness becomes the cheap default
  once scoring saturates.
- **Miss**: the object passes `z < −Z_MISS = 0.65` m uncut — the HEAD
  plane. **DELIBERATE DEVIATION from the real game** (which judges
  misses at despawn ~2 m behind the head; v7.0-v7.2 used 2.65),
  user-specified for credit assignment: the −0.5 lands ~5 ticks after
  the play plane instead of ~20, and a passed note is certainly lost,
  never ambiguously rescuable. Objects stay cuttable until the head
  plane. Bombs passing are consumed silently (status 4, no event). The
  episode tail after the last `hit_time` is `0.5 + Z_MISS / min_speed`
  so final misses always register.
- **No reward shaping** (mechanism REMOVED 2026-07-21, user direction):
  rewards come only from the sim's scoring, like real Beat Saber. The
  potential-based shaping (Ng et al. 1999) used in v0's earliest runs was
  already off by default in the pure-reward era (2026-07-20); its code
  and the `shaping_coef` knob are now deleted everywhere.
- **Termination**: all objects consumed AND no follow-through pending
  (so the last note's post-swing isn't truncated), or
  `t > last hit_time + tail` (tail = `0.5 + Z_MISS/min_speed`), or tick
  count = `max_steps` (default `chart_len_s·30 + 90`); the hard ends
  force-resolve pending follow-throughs with their partial arcs. All
  ends are termination (no bootstrap-relevant truncation — episodes are
  natural-length).
- **RNG**: `numpy.random.Generator(PCG64(seed))`; all randomness (chart
  gen) flows from it. Same seed + same action sequence ⇒ identical episode.

## Flux game mode (`rust/src/flux.rs`) — 2026-09-26

A SECOND game mode, alongside (never replacing) the standard game
above. User-specified rules; everything not listed here is the
standard env verbatim: the v10 body and physics, 30 Hz ticks, the 29
gear heads, the head-anchored track (`z = (hit_time − t)·speed +
z_off`), the energy / limit / self-contact / foot terms, the fall and
tilt terminations, the episode end and tail rules, and the combo /
multiplier rules (event codes are the standard ones, so ComboWalk and
the bench walk run unchanged). The standard game is bit-identical with
Flux compiled in (pinned by a before/after fingerprint of seeded
rollouts on both builds and both foot modes).

- **Mode selection**: `RawBeatSim(..., flux=True)` / `VecBeatSim(...,
  flux=True)`, fixed per sim. A Flux sim takes `set_flux_chart(chart)`
  (`set_chart` refuses) and vice versa. Walls, bombs, chains and arcs
  do not exist in Flux.
- **Hands**: no sabers. Each hand is its **palm centre**, a point
  `PALM_OFF = 0.06` m along the hand axis (the blade direction of the
  standard game) from the wrist — the model's fist-collider centre.
  The **Flux** is the finite segment between the two palms, always
  live. No swing chain, no follow-through trackers.
- **Chart tuple**: `(hit_time, x, y, kind, color, rot_rad, speed,
  length, sweet)`, sorted by hit_time; kind 4 = circle, 5 = bar.
  Positions are world metres in the standard grid frame (object
  centre at the hit plane).
- **Circle** (`KIND_CIRCLE`, colour 0 left / 1 right): a flat disc
  facing the player, hitbox radius `CIRCLE_R = 0.175` (the ~35 cm
  visual disc), with the standard good box's depth convention: 1.0 m
  deep, shifted 0.25 m toward the player (`z ∈ [z − 0.75, z + 0.25]`).
  No look-at tilt. Contact = the palm's path over the tick (last
  tick's palm → this tick's) clipped to the slab passes within
  `CIRCLE_R` of the centre in-plane. **Matching palm → good**:
  `50 + round(10 · (1 − clamp(d / 0.3, 0, 1)))`, `d` = that in-plane
  closest approach (the standard accuracy formula at 10 points; note
  the hitbox radius caps `d` at 0.175, so the accuracy floor is 4).
  **Other palm → bad hit** (event 3 wrong_hand, −0.5); the matching
  palm wins a same-tick double touch. No speed or direction
  requirement.
- **Bar** (`KIND_BAR`, colourless): a thin rectangle in the plane
  facing the player, width `BAR_WIDTH = 0.05`, length `length` m,
  rotated `rot` about z: axis `u = (−sin rot, cos rot)` — 0 = vertical,
  CCW from the player's view (the note-angle convention). Hitbox: the
  box `[0.025, length/2, 0.5]` half-extents in that z-roll frame,
  centred 0.25 m toward the player (the circle slab). Contact = the
  **swept Flux** — the quad from last tick's segment to this tick's,
  triangulated `(L0, R0, R1), (L0, R1, L1)` exactly like the saber
  sweep — overlaps the box (Akenine-Möller SAT). Score:
  `100 + round(15 · (1 − clamp(d / 0.3, 0, 1)))` with `d` = the
  distance along `u` from where THIS tick's Flux LINE (infinite, in
  the bar's xy plane) crosses the bar axis — clamped onto the bar — to
  the **sweet spot** at `sweet · length/2` (sweet ∈ [−1, 1]: 0 = the
  middle, +1 = the `+u` end, "top" at rot 0). A Flux parallel to the
  bar (or collapsed to a point) uses the point of its projected span
  nearest the sweet spot. **Crossing angle** (user rule, the note
  swing tolerance's twin): this tick's Flux line, projected onto the
  bar's xy plane, may deviate at most 60° from the bar's perpendicular
  `(cos rot, sin rot)` — undirected, so 0–90°: `(f · n)² ≥ 0.25 |f|²`.
  A crossing beyond that is a **bad cut** (event 2 "angle", −0.5,
  combo break); a collapsed Flux (palms coincident in xy) fails too.
- **Misses**: an untouched circle or bar passing the head plane is a
  miss (event 4, −0.5), exactly like a note.
- **Reward / score**: good = `pts / 115` (circle max 60/115, bar
  115/115); bad and miss −0.5. `score_components()` returns (base
  contact points, accuracy points, 0); a good event's `pre` slot
  carries the base (50/100) and `acc` the accuracy points.
- **Reach gate**: `max(shoulder_z) + 0.65 + PALM_OFF + 0.75` (the
  slab's near face) replaces the saber reach.
- **Obs** (same OBS_DIM and offsets as standard; Flux lane meanings,
  `constants.FLUX_LANE_*`): hand blocks = palm x/y head-relative, palm
  z world, palm velocity /V_NORM, hand axis (3), lanes 9–10 zero.
  Entity slots (nearest-first by head distance, no walls): 0–3 as
  standard (x, y head-relative; z lane; −speed/Z_SCALE), 4/5 circle
  colour one-hot, 6/7 bar axis u, 8 is_circle, 9 is_bar, 10 bar length
  (m), 11 sweet offset, 12/13 sweet-spot head-relative x/y, 14–17
  zero. Previous-note blocks record the last resolved CIRCLE of that
  colour (x/y, valid; direction lanes zero); bars never record.
- **Policy**: the same NotePolicy network, trained from scratch, with
  arch marker `game = "flux"`; `load_policy(ckpt, game=...)` refuses a
  checkpoint of the other mode (absent marker = standard), so the
  standard bench/replay tools never load a Flux policy. Under
  `--note-geo` a bar's target point is its sweet spot (lanes 12/13)
  instead of its centre.
- **Generator f1** (`beat_arena_env/flux_chart.py`; no official Flux
  map data exists yet): generator v2's clock (jittered lattice,
  interval × note_speed_max / walk speed, one shared speed walk,
  default 10–18 m/s) plus HAND FLOW — two imaginary palms walk the
  play rectangle (default = the grid span, x ∈ [−0.9, 0.9], y ∈ the
  row heights; one step per tick uniform in a disc of radius
  `hand_speed · dt`, 2.5 m/s; left kept ≥ `min_sep` 0.3 m left of
  right) and objects are placed ON that trajectory: each hand spawns a
  circle at its palm with P = `circle_prob` (0.3) per tick; a bar
  (P = `bar_prob`, 0.15) gets its sweet spot on the palm-to-palm
  segment (fraction U(0.2, 0.8)), rotated within ±`bar_dev_max` (45°)
  of square to the palm-to-palm line (so it is clearable under the 60°
  rule) and flipped end-for-end with P = ½, length U(0.3, 1.2) m,
  sweet U(−1, 1), centre = sweet point − sweet·length/2·u.
- **Envs / trainer**: `flux_env.RustFluxEnv` (gym, JSONL recording with
  header `"game": "flux"`, the Flux chart, and per-tick `"palms"`),
  `flux_env.FluxVecEnv` (batched Rust), `CudaBeatVecEnv(game="flux")`;
  `train_batch.py --game flux` (ranked maps and ep-stats switch off;
  `--circle-prob`, `--bar-prob`, per-mode speed defaults).
  Tests: `tests/test_flux.py`, `rust/src/flux.rs` unit tests,
  `tests/test_flux_parity_cuda.py`.
- **Viewer** (`viz/app`, header `"game": "flux"`): palms (hand-coloured
  orbs at wrist + PALM_OFF along the hand axis, from the body FK) and
  the Flux beam replace the sabers; circles draw as coloured discs with
  a rim, bars as thin strips rolled by `rot` with a gold sweet-spot
  notch; 'h' shows the circle cylinders / bar boxes; touches burst and
  fade (no cut plane to slice), bad touches and misses leave an outline
  + mark. Fixture: `python/tools/make_flux_fixture.py` →
  `viz/app/test/fixtures/flux_scripted.jsonl`; tests
  `test/flux.test.js` (npm test) and `test/flux-browser.mjs`.
- **Code layout** (2026-09-27): the rules are three body-free files —
  `rust/src/geom.rs` (Note, the hitbox / sweep geometry, judge
  constants, chart tail / cursor / end rule), `rust/src/flux.rs` (palm,
  circle / bar tests, `flux_note`, `judge`), `rust/src/combo.rs`
  (ComboWalk) — compiled into BOTH the env crate and the WebXR game's
  wasm crate (below). `rust/src/flux_sim.rs` is the env-only glue
  (chart install, the per-tick `flux::judge` call on the body's palms
  plus the env's own tallies from its events, the Flux obs blocks).
  The split was bitwise-neutral: seeded standard + Flux rollouts on the
  f64 and f32 builds, single and vector, hash identically before and
  after.

## Flux WebXR game (`viz/app`, `rust/flux-wasm`) — 2026-09-27

A person plays Flux in a headset, judged by the env's own code. There is
no port of the rules: `rust/flux-wasm` compiles the env crate's
`geom.rs` / `flux.rs` / `combo.rs` (via `#[path]`) to
`wasm32-unknown-unknown` with a dependency-free C ABI, and
`FluxGame::step` runs the Flux tick in the env's order (`Core::step_inner`):
tick end time `(tick + 1)·DT`, `flux::judge` on last tick's → this
tick's palms, `tick += 1`, the cursor, `geom::chart_done` (no trackers;
a person has no fall / tilt / escape), then the bench's ScoreSaber
`ComboWalk` for combo / multiplier / score. f64: the game matches the f64
Rust REFERENCE env bit for bit; the agent trains on the f32 CUDA kernel,
which is tied to that reference by the existing CUDA parity tests
(tolerance bands, not bits).

- **Deterministic trig**: a bar's rotation is installed as (sin, cos) via
  `geom::det_sin_cos` (a pure-Rust copy of musl's algorithm, < 1 ulp)
  in the f64 build, not the platform libm — glibc (native env) and
  musl's port (wasm32) disagree in the last bit for ~6% of angles, which
  would let the two builds install different bars. Nothing else in the
  judge calls a transcendental (sqrt is IEEE-exact). The f32 build keeps
  sinf/cosf (no game counterpart). Introducing it moved no seeded
  fingerprint (standard or Flux, f64 or f32).

- **The player is the body**: palms = the controller grip-space origin
  (the handle centre the palm wraps), or for tracked hands without a
  grip space the midpoint of wrist and middle knuckle; head = the viewer
  pose; shoulder z (reach gate) = head z (the env's reference pose has
  both at PLANE_Z0). Approximations, not rule changes: the headset pose
  is the eyes, a few cm in front of the body's head-frame origin, so the
  miss plane (head z) sits that much further forward than on the body;
  and with shoulder z = head z the reach gate stops judging objects more
  than ~1.46 m past the eyes (a palm > ~0.7 m in front of the face) —
  about the body's own reach. Object positions are the env's absolute world
  metres (no height scaling). The player's floor origin sits at world
  z = PLANE_Z0, where the body stands (an `<XROrigin>` at scene
  z = 0.65 for the view, the same offset added to every pose read).
- **Rendering vs judging**: everything is DRAWN at the headset's display
  rate (72–120 Hz): objects on continuous chart time, the palms / beam
  from each frame's pose. The JUDGE runs at `judge_hz` = 30·k — default
  120 Hz, selectable 60, or 30 = the env's own tick (like-for-like with
  the agent). The judge rate is only the scoring clock (k judge ticks per
  30 Hz env tick; unrelated to the body sim's physics substeps — the
  game has no body): the same `flux::judge` on a finer sampling of the
  player's path, so hits resolve (and burst, and sound) within 1/120 s of
  happening instead of at the env tick's end. At 30 Hz it is the env's
  tick bit for bit (the parity gate runs there); above it the game judges
  a finer path than the env's 30 Hz discretization of the body — same
  rules, finer sampling, so scores across rates are not one leaderboard
  (bests are kept per chart and rate).
- **Ticks**: poses arrive at the display rate; judge tick j's palms /
  head are the pose track linearly interpolated to its end time
  (j + 1)·DT/k (a sample exactly on a tick is used verbatim), and
  p0 is the previous tick's palms (the palms at chart time 0 for the
  first tick). The recording keeps 30 Hz frames (the viewer's clock);
  every event carries its exact judge time `t`, which the viewer uses
  for the hit burst / removal (agent replays: the tick end, as before);
  header `judge_hz`. A 3 s countdown precedes chart time 0; the clock pauses
  while the session is hidden (headset off / system menu). With no pose
  (no headset at all, or tracking lost) the last pose — else a resting
  stance — is HELD, so the chart clock still runs to its end; on a
  resume or a regained track the pose track REBASES (the next tick's
  swept Flux starts at the new palms, never across the gap). A single
  hand that loses tracking keeps its last position and does sweep to its
  new one on reacquisition.
- **Charts**: `viz/app/src/lib/fluxcharts.data.js`, generated by
  `python/tools/make_flux_chart_pack.py` from generator f1 at full
  precision (the "Training" entries are its default config — the
  agent's training distribution — plus slow warm-ups, a 60 s marathon
  and a bars-only chart), and the chart of any agent Flux replay open
  when play mode starts (race the agent on its own level).
- **Recording**: a run is a replay (`fluxgame.js` LiveReplay; header
  `"game": "flux"`, `"player": "human"`, `"result"`; per tick `palms`,
  `head` [x, y, z, qx, qy, qz, qw] (env frame), `reward` (judge terms
  only), `events` in the env recorder's shape). The viewer renders the
  live game from it (Notes / Flux unchanged) and plays saved runs back
  (palms + beam + a headset marker; first-person = the head track).
- **Play**: `cd viz/app && npm run xr` (HTTPS dev server on the LAN,
  self-signed; WebXR needs a secure context), open
  `https://<laptop-ip>:5173` on the headset, "🥽 play Flux", "enter VR";
  trigger / pinch starts, grip changes chart; results offer save
  (.jsonl) and watch-in-viewer.
- **Parity gates**: `viz/app/test/flux-parity.test.js` replays env
  episodes (`python/tools/make_flux_game_fixture.py`: every tick's full
  judge input — both ticks' palms, head, shoulder z — at full precision,
  plus the env's events and each chart's INSTALLED (sin_a, cos_a)) through
  the wasm judge and requires the same install bit for bit, identical
  events tick for tick, the same end tick and tallies, and every Flux
  outcome exercised; a 400-rotation trig probe (≈ 23 angles where the
  platform libms differ) pins the install beyond the episodes' few bars; npm test runs it on the committed fixture,
  `python/tests/test_flux_game_parity.py` on one recorded fresh from the
  env under test (and pins the chart pack to the generator). Game
  plumbing: `test/flux-game.test.js`; the built app:
  `test/flux-play-browser.mjs` (scripted poses) and
  `test/flux-xr-browser.mjs` (a real WebXR session on an emulated
  Quest 3, iwer). CI: `.github/workflows/ci.yml`.

## Chart generation (`beat_arena_env/chart.py`) — generator v2

**Generator v2 (2026-07-24, user-designed, hard replace):** the old
rate-scheduled uniform stream (note_rate/dirs/continuous_dirs/
min_hand_gap_s/bomb_rate/chain_prob/arc_prob) is GONE. Notes now come
from a **jittered tick lattice** walked by **three independent paths**
— left hand, right hand, bombs.

`ChartConfig`: `chart_len_s` (default 20.0), `colors` (default `"both"`
since 2026-07-26 — the old `"right"` default silently trained gen3's 5B
run one-handed, the whole real-map transfer regression; `"right"` = all
blue, `"both"`, default `"right"`), `first_note_s` (1.0), `cols`/`rows`
(allowed positions — still the spatial-curriculum dial), and per-tick
dials `note_prob` (P(note) per hand path per tick, 0.25), `extra_prob`
(0.125), `dot_prob` (0.1), `bomb_prob` (0.125), plus the speed-walk
clamp `note_speed_min`/`note_speed_max` (2.0 / 8.0 m/s, user-picked).

Generation, per tick (first at `t = first_note_s`, then
`t += (0.15 + U(−0.03, 0.03)) · (note_speed_max / walk_speed)` —
`TICK_DT`/`TICK_JITTER` scaled by the shared speed walk's current
value (user-directed: top speed keeps the base lattice, slow sections
spread out — equivalently the tick-to-tick z-gap is speed-invariant,
~1.2 m of approach distance) — until `chart_len_s`):

- **Each hand path** independently spawns a note with P = `note_prob`.
  Direction keeps **parity flow**: the path's previous ORIGINAL angle
  flipped 180° plus `N(0, ANGLE_SIGMA = 45°)` (user-picked σ) — mostly
  clean opposing swings, occasional big reversals; the path's first
  note draws `U(0, 2π)`. Position: x, y ~ U over the allowed-cols/rows
  rectangle padded by `POS_PAD = 0.2` m (env v7.5 semantics unchanged;
  defaults x [−1.1, 1.1], y [0.65, 2.1]; real BeatSaver maps keep grid
  cells; col/row are nearest-cell metadata). Speed: ONE shared random
  walk across ALL paths — previous spawned object's speed
  `+ U(−0.4, 0.4)` (`SPEED_DEV`), clamped to
  [`note_speed_min`, `note_speed_max`]; first draw uniform in the range
  (shared lookback regardless of color keeps the hands' speeds
  similar).
- **Extra note** (P = `extra_prob` after each spawn): starts as an
  exact replica; placed `d ~ U(−1.5, 1.5)` m (`EXTRA_PATH`) along the
  original's swing line (ahead/behind on the swing path); DISCARDED if
  that along-line point leaves the spawn rectangle; survivors then get
  a `U(−45°, 45°)` rotation offset (`EXTRA_ROT`), a `U(−0.5, 0.5)` m
  offset perpendicular to the line (`EXTRA_PERP` — the final position
  may overhang the rectangle by ≤ 0.5 m), and a `U(−0.1, 0.1)` m z
  offset (`EXTRA_DZ`, implemented as the equivalent arrival-time shift
  `dz/speed`). Extras NEVER feed the parity chain (originals only) and
  never roll their own extras.
- **Dot roll**: at the end of each tick every generated note (originals
  and extras) independently becomes a dot with P = `dot_prob`
  (`any_dir`, parser-convention angle 0; the parity chain keeps the
  pre-dot angle).
- **Bomb path**: same spawn logic (its own parity chain steers extra
  placement, shared speed walk) at P = `bomb_prob` per tick; emitted
  objects become directionless bombs (`any_dir`, angle 0); no dot roll.

There is NO min-hand-gap guarantee anymore — pacing is emergent from
the lattice and spawn odds (same-hand originals are ≥ 1 tick apart;
extras land essentially on their original's tick: one swing can take
both). Chains and arcs are NOT generated procedurally — real BeatSaver
maps are their only source; `chart.chain_links` (real quadratic Bézier,
BeatLeader beat.js:424-463) survives for the parser. Reference density
(20 s chart): right-only ≈ 26 notes + 11 bombs (tests/curricula only)
(~1.3 notes/s); both ≈ 43 notes + 11 bombs (~2.3 notes/s), ~9% dots —
and it varies WITHIN a chart with the speed walk: measured local
density ~1.4 notes/s around sub-4 m/s notes vs ~3.2 around 6+ m/s
notes (pinned by test).

## Observation layout (exact, all f32) — order is a contract

Obs v6 (env v7 — 16 object slots × 14 features; sabers unchanged from
obs v5 except y normalization):

- `sabers: [f32; 46]` = per hand (left then right), 23 features:
  `[pitch/90°, yaw/100°, twist/90°, (2·elbow−150°)/150°, cosψ, sinψ,
  bend/95°, ω(6)/Ω_joint, wrist x/1, (y−1.4)/0.75, z/1, wrist_vel(3)/6,
  blade dir(3), min(swing_acc/100°, 2)]` — joint pose (bounded scalars
  normalized to their ranges; roll keeps its v5 cos/sin coding),
  per-joint angular velocities, then the DERIVED hand state the reward
  actually runs on: FK wrist position (3D; y centered on the middle
  row), finite-difference wrist velocity, unit blade direction, and the
  live pre-swing meter (1.0 = full credit banked; clipped at 2).
- `notes: [[f32; 14]; 16]` — the next 16 *pending* objects (notes,
  bombs, chain heads/links) in arrival order, slot 0 = soonest. Per
  object: `[x/1, (y−1.4)/0.75, z/8 (clipped to [−3, 12] m — the
  negative tail keeps "behind the plane but still cuttable" visible),
  vz/8 (= −speed/8, per-object), is_left, is_right, dir_cos, dir_sin,
  is_bomb, is_chain_head, is_chain_link, free_pre, free_post,
  valid=1]`. Direction stays the unit vector; dot notes encode the ZERO
  vector; **chain links carry their Bézier tangent** (path information
  — `is_chain_link` disambiguates; they still judge as any-direction).
  Bombs have no color one-hot and no direction. Empty slots all-zero.
  Slots SHIFT as objects are consumed (no stable identity; 16 slots so
  a chain burst doesn't evict upcoming notes).
- `note_mask: [f32; 16]` — 1.0 = slot holds a pending object.

**Flat vector:** `sabers(50) ++ notes row-major(224) ++ note_mask(16)` →
**length 290** (env v9: the saber block gains the wrist-deviation angle
+ omega → 25/hand). Offsets: notes at 50, mask at 274. Checkpoints
from all earlier obs versions (incl. 286-dim v7/v8 and the 166-dim v6
lineage) are incompatible.

There is deliberately NO time feature: with constant speed,
`time_until = z/|vz|` — the approach axis carries the temporal ordering
spatially, as a player perceives it. `vz` is included so timing stays
recoverable if per-note speeds ever vary (then z alone is an insufficient
statistic for arrival). Both properties are pinned by tests.

## Gymnasium env (`beat_arena_env/rust_env.py`)

`RustBeatEnv(colors, chart_len_s, cols, rows, note_speed_min,
note_speed_max, note_prob, extra_prob, dot_prob, bomb_prob, seed)`:
`observation_space = Box(-2, 2, (290,))` (bounds loose),
`action_space = MultiDiscrete([K]×14)` (per hand: pitch, yaw, twist,
elbow, roll, bend, dev; left first; K = `gears`). New chart every `reset(seed)` (chart
gen stays in Python — single source for charts). Per-step `info["events"]` =
(note_idx, reason, points) from the Rust event stream (reason `post` =
follow-through resolution, note already counted; `bomb` = bomb hit);
final step adds `good_cuts, bad_cuts, misses, n_notes, score,
bombs_hit` (n_notes = scoreable objects, bombs excluded; score max =
115·notes + 85·heads + 20·links). `set_record(path)` dumps replay JSONL
v7 built from those events. Batched training path:
`beat_arena_env/vec_env.py: BeatVecEnv` over `VecBeatSim`.

## Replay log (JSONL, version 7)

Version 7 (env v7): chart entries carry EXPLICIT world positions and
object kinds; saber entries keep the v6 arm state — 15 numbers per
hand, `q(6) = [pitch, yaw, twist, elbow, roll, bend]` ++ `om(6)` ++
the FK `wrist [x, y, z]` (logged so viewers/tools can validate their FK
mirrors); cut events carry `points` (pre+accuracy at the cut,
follow-through via a separate `post` event); bomb hits are `bomb`
events; the action is the 12-int vector. v6 replays and older are
rejected.

```json
{"header": {"version": 7, "config": {..., "note_speed_min": 2.0, "note_speed_max": 8.0, "note_prob": 0.25, "extra_prob": 0.125, "dot_prob": 0.1, "bomb_prob": 0.125},
 "chart": [{"t":1.0,"x":0.3,"y":1.4,"color":1,"angle":4.712,"speed":4.0,"any":false,"kind":0,"fpre":false,"fpost":false}, ...]}}
{"t": 0, "time": 0.0, "sabers": [[pitch,yaw,twist,elbow,roll,bend, om...6, wx,wy,wz],[...]], "action": [0,1,0,2,1,0, 0,0,1,0,2,0],
 "reward": 0.0, "events": [{"cut": {"note": 3, "good": true, "reason": "ok", "points": 78}} | {"cut": {"note": 4, "good": false, "reason": "slow|angle|wrong_hand", "points": 0}} | {"post": {"note": 3, "points": 26}} | {"miss": {"note": 5}} | {"bomb": {"note": 6}}]}
```

**Viewer** (`viz/replay.html`, since 2026-07-21 a **Svelte 5 + Threlte 8
(three.js) SPA** compiled to that single static HTML file — still a
double-clickable, drop-a-`.jsonl`-in, zero-server page): real WebGL 3D
with **OrbitControls** (drag = orbit, wheel = zoom, right-drag = pan).
Since 2026-07-22 the viewport IS the page (fullscreen canvas) and all
DOM chrome floats on top of it: title + file pill at the top,
HUD/controls/help in translucent panels at the bottom
(`pointer-events: none` containers so drags reach the canvas). The
camera-attached gauge sprites anchor to the actual viewport corners
(aspect-aware) and sit above the floating bottom UI.
Source in `viz/app/` (Vite + vite-plugin-singlefile); rebuild with
`npm install && npm run build`, then copy `dist/index.html` over
`../replay.html`. `npm test` runs the pure-logic tests (node) and
`node test/browser.test.js` drives the built file in headless Edge
(loads a real replay via the file input, asserts playback advances with
zero page errors, screenshots to `test/out/`). Scene (world → three.js:
`(x, y, z) → (x, y, −z)`, camera on +z, orbit target y 1.4): a FLOOR
plane at y = 0, tunnel rails + depth rings to `z = 12` m, the 4×3 grid
at the real v7 lane/row positions, the blade-reach region
(z −2.65..1.75) drawn faintly, and per-kind objects at their explicit
chart `(x, y)`: 40 cm note cubes visually rotated to their cut
direction with arrows / dot discs, **bombs** (black spheres, red flash
+ "bomb!" on hit, silent despawn at z −2.65), **chain heads**
(half-height cubes), **chain links** (thin tinted slabs, shrink/pop +
"+20" on cut), and **arc hints** (faint tube from each fpost head to
its fpre tail + "free pre/post" glyphs). `h` toggles the REAL hitbox
wireframes (shifted good boxes, dot/link variants, small wrong-color
box, bomb sphere). Since env v6 the **whole upper body** renders: a
torso (shoulder crossbar at 1.47, trunk + legs hint to the floor,
head sphere) and per hand the upper-arm/forearm segments with
shoulder/elbow/wrist joints (positions from `armFK`, the JS mirror of
the sim's FK — validated against the sim-logged wrist in the node
tests) plus the blade wrist→tip as a cylinder (white-hot core when the
TIP is contact-capable ≥ 1 m/s via the finite-difference velocity
semantics) with 10-tick tip trails.
Note glyphs are Beat Saber-style low-poly white chevrons / dot discs;
a good cut slices the ACTUAL cube along the cut plane at its cut-tick
z — the two colored half-cubes fly apart perpendicular to the swing,
carried along it, tumbling and fading — with "+N" score popups,
follow-through "+N follow" popups, bad cuts/misses labeled with reason.
**Per-hand arm gauges** (toggle `g`): camera-attached panels showing
the roll dial (needle = blade azimuth inside the hand's allowed 280°
arc, forbidden zone dark, limit ticks), the bend dial (±95°, up =
toward the notes), and four bars for pitch/yaw/twist/elbow inside their
anatomical ranges, with numeric angles + angular velocities; needles
and bar markers turn orange within 10° of a joint limit. All in-scene text/panels are
canvas-texture sprites (not DOM overlays) so exported videos keep them
and the file works fully offline. Controls: space play/pause, ←/→
tick-step, +/− speed, r restart, g gauges, scrub bar, ⏺ export video
(MediaRecorder → .webm). Note positions and outcomes are DERIVED from
the chart + event stream (single source of truth), not re-simulated.
Replays come from `python/make_top10_replays.py`, which sets up the
production physics and plays a checkpoint on the bench's real maps
(the procedural-chart `record_replay.py` scripted/random recorder was
removed 2026-09-22 with the scripted bot).
Every interesting run should be recorded.

## Scripted reference bot (removed 2026-09-22)

The v7-v10 scripted reference bot (`beat_arena_env/scripted.py`: plant
the wrist by closed-form two-link IK, re-aim the blade, whip through
the note; the P6 re-author added gravity-compensated PD over that IK
on the dynamic body) was the solvability proof and eyeball baseline
before learned policies surpassed it. It was deleted together with
the arm IK only it used (`kinematics.ik_arm` / `wrist_for_blade`),
its solvability test and the `record_replay.py` recorder; last
measured numbers are in the README ("Historical reference points").

## Policy (`python/policy.py`)

`NotePolicy()` — transformer only (the pool baseline was deleted
2026-07-20 after the easy-config studies concluded; checkpoints still
record `encoder: "transformer"`):

- Per-note MLP `14 → 64 → 64` (shared across the 16 object slots).
- Per-saber token MLP `25 → 64 → 64` on the 23 obs-v6 arm features +
  `[is_left, is_right]` (hand identity via one-hot, weights shared).
- Token sequence = `[saber_L, saber_R, note_0..15]` (18 tokens); key mask
  `[1, 1, note_mask]`. **2 × full pre-LN transformer blocks** (LayerNorm →
  single-head QKV self-attention → residual; LayerNorm → FFN 64→128→64 →
  residual). Readout = concat of the two saber tokens' output embeddings
  (128) → trunk.
- Trunk `→ 128 → 128`, heads (unchanged since env v6): 12 × 3-way — per hand
  `pitch, yaw, twist, elbow, roll, bend`, left first — plus `value(1)`.
  All heads always execute; `logprob = Σ` over the twelve factored heads
  (pure independent factorization, as before).
- **Fused head + last-block pruning (2026-07-25, exact-math rewrites of
  the forward — the four-agent optimization survey's top find):**
  (a) the 12 head Linears + value head run as ONE concatenated
  `128 → 37` GEMM (`forward_stacked` → logits `(B,12,3)` + value; the
  13 modules stay separate in the state_dict — weights are `torch.cat`'d
  per call — so checkpoints are untouched), and sampling/logprob/entropy
  are computed from one `log_softmax` + `multinomial` instead of 12
  `Categorical`s (~120 tiny kernels gone; the sampling RNG STREAM
  changed — same distribution, the caveat the old graphed rollout
  already carried). (b) The FINAL transformer block computes only the
  2 readout tokens' outputs (`Block(..., n_queries=2)`): the attention
  mask is key-side only, so this is exactly the full block restricted
  to those rows (−29% forward MACs). Verified ≤5e-6 vs the old math on
  the record checkpoint; `forward()` still returns the 12-logit list
  (views) for existing callers. Measured on the GTX 1080: e2
  production training 39.6k → **62.5k steps/s (1.58×)**.

## Training (`python/train_batch.py` — the only trainer)

Batched PPO: `BeatVecEnv` (N Rust envs, one FFI call per vector step),
one policy forward per vector step; GAE(γ=0.99, λ=0.95), dual-clip
(ε=0.2, c=3), ent 0.01, vf 0.5, Adam 3e-4, grad clip 0.5. Flags:
`--num-envs --horizon --minibatches --epochs --total-steps --time-cap
--ent-coef --seed --ckpt-every --run-dir --init-from --resume --bench`
+ env config (`--colors --cols --rows --chart-len --note-speed-min
--note-speed-max --note-prob --extra-prob --dot-prob --bomb-prob`).
Checkpoints since 2026-07-24 carry Adam state + the LR-schedule anchors:
`--resume <ckpt>` continues a stopped run EXACTLY (weights, optimizer
moments, schedule position — lr/lr-final/lr-decay/num_updates come from
the checkpoint, CLI values ignored; the env/chart stream restarts
fresh, so only the optimization state is exact). `--init-from` remains
the weights-only warm-start with a fresh schedule; older checkpoints
without optimizer state are --init-from only. **`--cuda-graphs` REMOVED
(2026-07-25)**: capture existed to hide the 12-head/12-Categorical
launch storm; the fused head + last-block pruning (§Policy) removed
the storm at the source, after which graphs measured as pure overhead
(static-buffer copies + capturable Adam) at every shape tried. The
capture driver survives in `cuda_graphs.py` for the bench's argmax
fallback + the forward microbench; graphs-era checkpoints load
unchanged (they saved the normalized eager Adam form). Production
throughput after the rewrite: **62.5k steps/s** at
`--env cuda --num-envs 4096 --minibatches 16 --epochs 2` (was 39.6k
eager / 46.2k fused+graphs best-ever). **`--env cuda` (2026-07-25,
user-ordered, subagent-built): a custom CUDA-kernel sim**
(`beat_arena_env/cuda_env.py` + `csrc/beat_arena_cuda.cu`, JIT-built
via torch cpp_extension for sm_61; needs the CUDA 12.6 toolkit —
found at `D:\Cuda-12-6` — and VS2022) — ONE kernel launch per vector
step (block per env, 64 threads; note-ordered reward accumulation for
bitwise reproducibility; the torch env's aggregated tracker ring;
chart-pool indices so reset copies nothing). A deliberate THIRD sim
implementation: Rust stays the reference; parity suite mirrors the
torch env's (f64 EXACT on all six config×policy cases, f32 bounded;
`tests/test_cuda_env.py`; suite 97 green). **The torch env itself
(`--env torch`, torch_env.py) was REMOVED 2026-07-25** — surface-area
reduction once the CUDA kernel became the production backend;
`_charts_to_tensors` (the single chart-tensorization source) moved to
cuda_env.py. Measured at removal time: env step
**~170× the torch env** (5.8M steps/s at N=4096), end-to-end
**1.65×** (13.6k → 22.5k steps/s; 24.6k with the then-extant
--cuda-graphs; all pre-fused-head numbers) — the Amdahl cap: rollout
share fell 49% → 14% and the env is now ~3% of rollout; the wall is
the update path (82% after the fused-head rewrite).
Batch = num_envs×horizon;
NOTE: batch shape changes the optimization regime (gradient passes per
env step) — cross-shape learning comparisons need retuning.
**`--epochs 1 --minibatches 8` is the DEFAULT (2026-07-26, user
decision)**: same 8 gradient steps/update as the old e2/mb4 at half the
update compute — the paired 50M gate measured **+110.9 wpp (dAcc
t=+12.88, score_acc +32.5%) AND 1.61x wall** (~250k steps/s at the
production config). Overturns the 07-21 "single-pass fails" study,
which conflated sample reuse with gradient-step density.
**`--fp16 all` is the production DEFAULT for training
legs (2026-07-26, user decision)**: fp16-storage activations everywhere
(fp32 cuBLAS compute, fp32 master weights/checkpoints, static loss
scale 2^17), fused
single-head attention on the fp16 path, chart pool 65536 — ~140k
steps/s vs 78k fp32-era (1.8x) at the production config.
`--graph-rollout` was part of that default until **2026-08-31, when it
was REMOVED**: at the production width (N=32,768) the graphed forward
measured 0.99x eager and the end-to-end rollout/update split was
identical with and without it, so the capture/refresh machinery and
its eager-divergent sampling RNG stream were pure cost. The 1.20x it
was adopted for was a small-N launch-overhead win. The ScoreSaber
bench keeps its OWN graphed argmax (`cuda_graphs.py`, opt out with
`--no-graphs`); that path is unchanged and bench JSONs still stamp
`forward`. The 2026-07-26
graph gate measured the all-fp16 graphed arm at **+72.4 wpp (paired
dAcc t=+6.63, score_acc +8.7%)** over fp32-rollout 'update' mode — the
epoch-1 ratio diagnostic (r1 2.0e-3 -> 6.7e-4) attributes it to
rollout/update numerical CONSISTENCY, which reframes the earlier
borderline fp16-update gate (−0.8% wpp, score_acc −12%): that arm
paired an fp32 rollout with an fp16 update. fp32 ('off') remains the
record-attempt fallback. Graph caveats: sampling RNG stream differs
from eager; auto-downgrades with a note on CPU/rust env,
--fused-update, or capture failure. fp16 mutually exclusive with
--fused-update.

## ScoreSaber bench (`python/bench_scoresaber.py`, 2026-07-23)

Standalone checkpoint benchmark (NOT part of training): every ranked
ScoreSaber leaderboard, one greedy argmax rollout each (deterministic
sim + fixed chart ⇒ exact), no replays. `--stars MIN MAX` restricts the
range; runs on the same checkpoint MERGE into one JSON per checkpoint
(default `bench/<run>_<ckpt>.json`): evaluated levels are skipped on
rerun (star ranges accumulate), skipped levels (parse/network) are
retried, `--force` re-evaluates, and a model-hash mismatch (sha256 over
weight names+shapes+bytes) is an error, not a merge.

Machinery (`beat_arena_env/scoresaber.py`):

- **Catalog**: `/api/leaderboards?ranked=true` paginated to completion
  (~6.1k SoloStandard difficulties), cached at
  `maps/scoresaber_catalog.json`; per-leaderboard stars + maxPP +
  maxScore from the API. Map chart data downloads from BeatSaver by
  hash into `maps/scoresaber/<hash>/` — extracted, media stripped
  (~100 KB vs ~10 MB per song). Downloads run on a producer thread
  while rollouts proceed (N envs, one batched policy forward).
- **Combo multiplier** (bench-only — the training gym deliberately
  stays multiplier-free): tiers 1x/2x/4x/8x, progress bar 2/4/8 hits,
  the promoted multiplier applies to the hit that fills the bar
  (decoded from ScoreSaber's own max-score formula 115·(8n−63), n ≥ 14
  plain notes — reproduced exactly by our walk, pinned by tests AND
  validated live: computed max == API maxScore on every leaderboard
  checked). Combo breaks (bad cut, miss, bomb hit) halve the
  multiplier and reset the bar. A note's full points (cut +
  follow-through) commit at its cut-time multiplier. Units: note 115,
  chain head 85, link 20; bombs score nothing.
- **PP**: accuracy = multiplied score / multiplied max score;
  pp = curve(acc) × maxPP where maxPP = 42.117208413 × stars and the
  curve is ScoreSaber's official 37-point piecewise-linear accuracy
  curve (fetched from `/api/v2/realms/1/pp-curve`, 2026-07-23;
  curve(95%) = 1, curve(100%) = 5.37). Total = Σ ppᵢ·0.965^(rank−1),
  best first (wiki.scoresaber.com/ranking-system).

Result JSON: checkpoint path/name, model_hash, arch, global_step,
scoring_version, updated timestamp, total_weighted_pp, n_evaluated,
n_skipped, and `levels[]` — per level: leaderboard_id, song_hash, song,
mapper, difficulty, stars, max_pp, duration_s, n_notes, score,
max_score, api_max_score, accuracy, pp, good_cuts, bad_cuts, misses,
bombs_hit, highest_combo, full_combo (skipped levels carry a `skipped`
reason instead of scores).

**Vectorized rollout (2026-07-24, the former "future unlock", built):**
`VecBeatSim.step` now also returns the flat per-env ORDERED event
stream `(env_idx, note_idx, code, points)` per vector step (order
within an env preserved — the combo/multiplier walk depends on it;
vec ≡ serial pinned by tests), so the bench steps every active map in
ONE FFI call. Idle slots hold an empty chart (done every step,
eventless, ~free); a done env's counters are read before its slot is
recycled. Measured (GTX 1080, cached maps, steady-state): **~60k
ticks/s ≈ 800 maps/min ≈ ~8 min for the full 6.1k catalog** at
`--num-envs 512` (256 ≈ 520/min; 1024 is SLOWER on partial runs — the
idle tail wastes fixed-N forwards). Was 38-39k / 445/min / 14 min on
the old serial per-env loop; the wall is now the per-step policy
forward, not Python. Fixed-N forwards also make results run-to-run
deterministic (the old variable-batch forward could flip near-tie
argmaxes — observed once in 200 maps vs old recorded results, same cut
counts, Δscore 0.45%). **2026-07-25 rungs harvested**: the argmax
forward is CUDA-graph captured (`_GraphedArgmax`, capture driver now
in `cuda_graphs.py`, eager fallback + `--no-graphs`) — ~20%
steady-state, 800 cached maps 67.2 s → **57.3 s**, full catalog
**~7 min**; and the
combo/multiplier walk moved INTO Rust (`ComboWalk` in `VecBeatSim`,
`combo_walk=True`, the ONLY walk — the Python event loop is gone;
Python `ComboTracker` survives for `max_multiplied_score` and as the
pinned test reference; Rust ≡ Python pinned by test, identical
results verified on 800 maps). The walk itself bought ~0.3% — the
forward was the wall, as Amdahl said. **Fused single-kernel forward
(2026-07-25, subagent-built, `--fused-forward`)**: the whole 107k-param
policy inference as ONE 128-thread-block-per-sample launch
(`csrc/policy_forward.cu` + `policy_cuda.py`; weights packed/pre-
transposed into one L2-resident buffer, 13 heads fused to a 128→37
matmul, LN/attention/masking fused, in-kernel argmax; fp32,
arch-pinned to embed64/trunk128/1-head, JIT-built, graceful fallback)
— **3.4× the graphed forward at N=512** (620 µs vs 2,130), 3.0× at
N=4096, parity at ~4e-7 max logit delta with bitwise mask-junk
inertness (`tests/test_policy_cuda.py`); 800-map bench 64.0 → 45.3 s
(**1.41×**; the forward was ~40% of the loop). Measurement note: the
agent's controlled µs benchmark found graphed ≈ eager for this forward
— it is small-GEMM EXECUTION-bound, so cuBLAS was the real incumbent;
the earlier 67→57 s attribution to graphs is partly superseded.
**Bench pipeline pass (2026-07-25 evening, from the four-agent
optimization survey — the wall was PYTHON PARSING, ~50% of a warm run,
GIL-serialized against the loop despite the producer thread):**
(a) **parsed-chart cache** — the producer caches
`(note tuples, max_multiplied_score, duration_s, last_hit_s)` per
`(hash, difficulty, parser-source-hash)` as pickles under
`--chart-cache` (default `maps/chart_cache/`, ~380 MB full catalog);
the source hash (beatsaver+chart+scoresaber .py bytes) auto-invalidates
on any parser edit. (b) **longest-first task ordering** off a
`durations.json` sidecar the producer fills — long maps can no longer
straggle at the tail on near-empty slot vectors (unknown durations
sort first). (c) **timed saves** — `save_results` every ~30 s wall
instead of every 25 finishes (the rewrite is linear in levels-on-file,
so every-25 was quadratic in catalog size). (d) **rayon in
`VecBeatSim::step`** (see §Rust sim core). Measured on the 4.0-4.5★
band (731 levels, N=512, `--fused-forward`): cold 34.6 s → warm
**20.2 s (~36 maps/s ⇒ full catalog ~3-4 min warm)**; cold==warm
per-level EXACT, and diffs vs the pre-fused-forward committed catalog
are only the known ~4e-7 argmax near-tie flips of `--fused-forward`
itself. Remaining rungs if ever needed: skip-idle-row forwards,
star-range sharding (top-300 carries 99.998% of weighted pp).

## Rust sim core (`rust/`, crate `beat-arena-rs`)

THE physics implementation (PyO3 0.22 + numpy 0.22; build with
`VIRTUAL_ENV=<venv> maturin develop --release` from `rust/`). Charts are
injected per episode via `set_chart` (env v7 tuples: `(t, x, y, color,
angle, speed, any_dir, kind, free_pre, free_post)`).
`RawBeatSim.step(actions)` (12 ints, per hand pitch/yaw/twist/elbow/
roll/bend) returns `(obs, reward, done, events)` — events = `(note_idx,
code, points, pre_pts, acc_pts)` with codes 0 ok / 1 slow / 2 angle /
3 wrong_hand / 4 miss / 5 post (follow-through resolution) / 6 bomb
hit; **the component split (2026-07-25, user-directed)**: a good cut's
points = pre + acc (+30 on arc-headed free_post grants; links/posts/
errors carry zeros), and all three sims track per-episode
`score_pre/score_acc/score_post` totals (Rust `score_components()` /
`VecBeatSim.components(i)`; torch/CUDA env tensors; f64-parity-pinned
across all three; replay cut events carry `pre`/`acc`; bench level
entries carry the totals) — plus getters
`sabers()` (per hand `q(6) ++ om(6) ++ wrist(3)`), `blades()` (unit
blade dirs), `status()` (0 pending / 1 good / 2 bad·bomb-hit / 3 missed
/ 4 bomb passed), `bombs_hit`, `n_notes` (scoreable, bombs excluded),
`time/score` and a `set_joints(s, q, om)` test hook (FK cache refreshes
immediately). `VecBeatSim` steps N sims per FFI call (actions `(N, 12)`
i64) returning `(obs, rewards, dones, events)` — events = flat
`(env_idx, note_idx, code, points)` with per-env order preserved
(2026-07-24, for the vectorized bench); `counters(i)` = `(good, bad,
miss, n_scoreable, score, bombs_hit)`. **`VecBeatSim::step` is
rayon-parallel over envs since 2026-07-25** (envs are independent;
per-env event vecs concatenate in index order, so the output is
IDENTICAL to the old serial loop — pinned by the vec≡serial and
Rust≡Python-walk tests) and releases the GIL for the span, so Python
threads (the bench's parse/download producer) genuinely overlap. The
scenario tests in `tests/test_env.py` gate every rule change.

Measured 2026-07-20 (8-core CPU): Python `env.step` 137 µs vs Rust
2.18 µs (**63×**; raw sim call 1.19 µs — the difference is gym-wrapper
glue). End-to-end training (8 serial envs): pool **1.6×**, transformer
**~1.3×** — the env is now ~0.3% of wall-clock and the torch policy is
the whole bottleneck; large parallel-env counts are effectively free on
the sim side (128 envs ≈ 0.28 ms/vector-step).

## Batched rollouts (`VecBeatSim` + `python/train_batch.py`)

`beat_arena_rs.VecBeatSim`: N sim cores stepped in ONE FFI call per
vector step (obs returned flat, reshaped Python-side; `obs_one_into(i, buf)` for
cheap post-reset rows). `beat_arena_env/vec_env.py: BeatVecEnv` adds
auto-reset with Python-side chart gen and per-episode info dicts.
`train_batch.py` = same PPO math as train_smoke, one policy forward per
vector step; batch = num_envs × horizon. Parity: VecBeatSim(1) ≡
RawBeatSim (tested); the batched trainer reproduces the serial path's
learning curve on the easy reference config (validated pre-v5, when
shaping still existed).

Throughput measured 2026-07-20 (8-core CPU, hard config, T=128):

| num_envs | transformer | pool |
|---|---|---|
| 8 (serial-rust reference) | 1.4k | 4.7k |
| 32 | **2.0k** | **10.6k** |
| 128 | 1.8k | 10.4k |
| 1024 | 1.6k | 8.7k |
| 128, `--epochs 1` | **2.9k** | **24.1k** |

Update phase is 85–93% of wall-clock once batched (rollout ≤15%): the
CPU ceiling is torch fwd+bwd per sample, not the env (sim ≈ free at
1.2 µs/step). Best N ≈ 32–128; larger batches lose to cache effects.
`--epochs` trades sample reuse for throughput. Overnight capacity at
these rates: ~84M steps (transformer) / ~700M (pool, epochs 1).
Further gains need a GPU or a cheaper policy, not more envs.

## Tests (`python/tests/`, pytest)

1. Obs layout: length 290, offsets, arm-state block, kind/arc flags,
   mask semantics; the Python FK mirror (`kinematics.py`) pinned
   against the sim's getters on random poses.
2. Determinism: same seed + fixed action stream (all object kinds in
   the chart) ⇒ identical rewards and events across two runs.
3. Cut detection unit scenarios driven on the RUST sim via `set_joints`
   + the event stream, mostly in the "aligned wrist frame" pose family
   (`pitch = −elbow + δ, yaw = twist = 0` makes the wrist frame a pure
   `R_x(−δ)`, which fixes the world x axis — a ±x blade stays exactly
   in a z = const plane; δ = 0.7 parks it at row-1 height): rotational
   good cut with sensible INTEGER points; too-slow → `slow`; opposite
   swing → `angle`; wrong hand only via the SMALL box (`wrong_hand`;
   the same placement that the good box catches is out of the small
   box's reach); the good box's 0.75 m front extension allows early
   cuts (~0.5 m out) but not too early (~0.8 m); dot notes have the
   0.8 m-tall box (arrow notes don't); untouched → miss exactly at the
   z = −2.65 exit; a pure SHOULDER-pitch pendulum cuts and its accuracy
   drops off-column; full pre-swing wind-up → 70 + follow-through 30;
   chain links pay flat 20 from any direction but still need speed;
   chain heads never emit `post` (max 85); `free_pre`/`free_post` grant
   their components; bombs: sweep-through = code-6 event and −0.5,
   pass-by = silent (status 4); the small bomb sphere ignores a blade
   0.38 m above it; the arm-forward start pose contacts early notes a
   flat blade can't reach; every joint's limit clamps and kills its ω.
   Recording round-trip (JSONL v7 counters + score match the sim) and
   VecBeatSim(1) ≡ RawBeatSim (6-tuple counters).
4. Chart gen (generator v2): tick-lattice spawn counts track the
   per-tick probabilities; parity flow (consecutive same-hand originals
   mostly oppose, offsets real and continuous); the speed walk stays in
   its clamp and steps ≤ 0.4 m/s across colors AND bombs (shared
   walk); extra notes are same-speed replicas along the swing line
   (rotation ≤ 45°, perpendicular ≤ 0.5 m, along-line point inside the
   rectangle) and discards happen; dot fraction ~1/10 with angle 0;
   positions continuous inside the padded rectangle; chain-link Bézier
   layout (parser-only now: times strictly increasing to the tail,
   positions marching along the curve).
5. Policy junk-leak: garbage in masked note slots changes no
   logprob/entropy/value (key-masking).
6. (Retired 2026-09-22: the scripted-bot solvability gate, removed
   with the bot.)
7. ScoreSaber bench (offline, network monkeypatched): PP-curve anchors
   (0.95→1, 1.0→5.367…, interpolation, monotonicity), 0.965-weighted
   totals, the multiplier walk vs ScoreSaber's 115·(8n−63) formula,
   break-halving + cut-time-multiplier commit semantics, per-kind max
   walk, catalog pagination/filtering, media-stripping map cache,
   weight-hash stability, results save/merge (hash mismatch refuses),
   and a full greedy-rollout smoke on a procedural chart.
8. BeatSaver parser fixtures: v2 (notes + bombs + dots), v3 (chains
   with the link Bézier, arcs as grants, precision angles, BPM change
   events), v4 (index-indirected arrays, dot chain heads), and the
   paired-note ANGLE SNAP rule (snapped diagonal pair; same-column
   control pair unsnapped).
