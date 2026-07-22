# Beat Arena — Interface Specification (v0)

Single source of truth for all components. Do not deviate from constants, layouts, or file formats without updating this file.

## Purpose

Toy RL environment modeled on Beat Saber: timed note blocks approach a cut
plane; the agent controls two sabers (left/red, right/blue) and must swing
through each note fast enough, in the note's required direction, with the
matching hand, inside a timing window. Built to learn **temporal/sequence
transformers** (vs kiting-arena's set encoder) and per-tick motion control
under a schedule. Pure Python/NumPy sim, Gymnasium API, PufferLib hookup,
same PPO harness style as `../kiting-arena`.

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
inventory: `../research/2026-07-22-bs-mechanics-inventory.md`.

## Layout

```
projects/beat-arena/
  SPEC.md
  README.md
  python/
    beat_arena_env/      # chart gen, constants, gym wrappers, scripted bot
    policy.py            # NotePolicy (transformer)
    train_batch.py       # batched CPU PPO (the only trainer)
    record_replay.py     # replay recorder (Rust event stream -> JSONL v2)
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

Uses kiting-arena's venv: `../kiting-arena/python/.venv` (torch-cpu,
gymnasium, pufferlib 3.0, pytest). No new dependencies in v0. No GPU;
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
  `BLOCK = 0.4` m is now VISUAL only — collision uses the real hitboxes
  below. Objects carry explicit world positions `(x, y)` in the chart
  (chain links sit off-grid on their Bézier).
- **Body (v6 chains, real placement)**: a 1.8 m player — shoulder
  anchors at `(∓0.2, 1.47, −0.65)` (0.4 m shoulder width, head at the
  real 0.65 m cut-plane distance); `L_UPPER = 0.35` m, `L_FORE = 0.30`
  m, `SABER_LEN = 1.0` m.
- **Joints** per hand, in order `q = (pitch, yaw, twist, elbow, roll,
  bend)`; all-zero = arm straight forward (+z), blade its extension:
  - `pitch` ∈ ±90°: + = arm up (about x).
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
  - **Forward kinematics** (the contract; `kinematics.py` and the
    viewer mirror it): arm frame `M = R_y(yaw)·R_x(−pitch)·R_z(twist)`
    (local +z = upper arm); elbow `E = S + L_UPPER·M·e_z`; forearm dir
    `f = M·(0, sin e, cos e)`; wrist `W = E + L_FORE·f`; wrist frame
    `M_f = M·R_x(−elbow)`.
- **Tiered anatomical dynamics** (user-picked): shoulder + elbow
  `α = 80` rad/s², `|ω| ≤ 8` rad/s; wrist roll + bend keep v5's
  `α = 200`, `|ω| ≤ 20`. Per joint per tick: `ω += ±α·DT`,
  `ω *= (1 − DRAG)` (`DRAG = 0.02`), clamp, `q += ω·DT`, limit-clamp
  (zeroing ω). Start pose: all zeros, at rest.
- **Action** per tick, per hand: 6 ternary joint heads (0 = coast,
  1 = `+α`, 2 = `−α`). Full action = `MultiDiscrete([3]×12)`, left hand
  first, joint order as above. No action masks. The 9-way translation
  head is GONE — all motion is joint-driven.
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
- **Hitboxes** (env v7 — the real ones, axis-aligned, NOT the visual
  cube; they do NOT rotate with the cut direction — v5.3's rotated
  collision cube is retired, the real game is axis-aligned and
  forgiving):
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
  1` m/s or the cut is `slow`; in-plane direction check
  `cos∠((v_c.x, v_c.y), dir_vec) ≥ 0.5` (dots and chain links exempt).
- **Judgment order** per pending object: bombs — either saber's sweep
  within `0.18` of center → **bomb hit** (event code 6). Notes/heads/
  links — matching saber vs good box → judged (slow/angle/good);
  matching saber missed → wrong-color saber vs SMALL box →
  `wrong_hand`. Any cut consumes the object.
- **Score** per good cut — real integer Beat Saber points, every
  component `floor(x + 0.5)`-rounded:
  - **pre-swing** `round(70 · min(swing_acc / 100°, 1))`; `free_pre`
    (arc tail) → flat 70.
  - **accuracy** `round(15 · (1 − clamp(d_min / 0.3, 0, 1)))` where
    `d_min` = closest approach of the swept surface to the object
    center (the real 0.3 m normalizer; a cut far from the center plane
    — including EARLY cuts through the deep box — floors at 0).
  - **follow-through** `round(30 · min(post_arc / 60°, 1))` — tracker
    unchanged (break/reversal ends it, `FOLLOW_TIMEOUT = 15`);
    `free_post` (arc head) → flat 30 paid at the cut tick, no tracker.
  - **Chain heads**: pre + accuracy only, max 85 — no follow-through.
  - **Chain links**: flat 20 on any valid cut (color + speed;
    direction exempt) — no pre/acc/post.
- **Reward per tick**: good cut pays `pts/115` (so a link is +20/115, a
  head ≤ 85/115); bad cut `−0.5`; miss `−0.5`; **bomb hit `−0.5`**;
  a bomb passing by is free and silent. A rotation-free "stab" scores
  only accuracy (≤ 15/115) — swinging is the game.
- **Miss**: the object passes `z < −Z_MISS = 2.65` m uncut (~2 m behind
  the head — the real game judges misses at despawn; objects stay
  CUTTABLE until then). Bombs passing are consumed silently (status 4,
  no event). The episode tail after the last `hit_time` is
  `0.5 + Z_MISS / min_speed` so final misses always register.
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

## Chart generation (`beat_arena_env/chart.py`)

`ChartConfig`: `note_rate` (notes/s, default 1.0), `chart_len_s` (default
20.0), `colors` (`"right"` = all blue, `"both"`, default `"right"`),
`dirs` (allowed subset of 0..7, default `[6]` = down-swings),
`min_hand_gap_s` (default 0.5), `cols`/`rows` (allowed positions —
restricting them is a spatial curriculum dial), `note_speed` (m/s toward
the plane, default 4.0), and the **env v7 object dials** (all default
0): `bomb_rate` (bombs/s), `chain_prob` (P(non-dot note → chain)),
`arc_prob` (P(consecutive same-color pair → arc)).

Generation: first note at `t = 1.0`; successive intervals
`(1/note_rate)·U(0.8, 1.2)`; color drawn 50/50 in `"both"` mode but a note
is pushed later if its hand's previous note is closer than `min_hand_gap_s`
(physical-feasibility guarantee — the same hand is never asked to be in two
places at once faster than it can travel). Col/row uniform over the grid.
Chains: sc ∈ 3..6 pieces, tail `U(0.10, 0.25)` s later, 1-2 cells along
the head's direction; links laid on the real quadratic Bézier
(`chart.chain_links`: control point = head + dirVec·dist/2, link i at
curve parameter `(i/(sc−1))·squish`, time lerped unsquished, angle =
curve tangent — BeatLeader beat.js:424-463); the hand is busy until the
tail. Bombs: an independent `(1/bomb_rate)·U(0.8, 1.2)` stream, cell
re-rolled up to 4× to avoid sitting within 0.3 m / 0.25 s of a note.
Arcs: consecutive same-color plain-note pairs ≤ 2 s apart get
`free_post` (head) + `free_pre` (tail).

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

**Flat vector:** `sabers(46) ++ notes row-major(224) ++ note_mask(16)` →
**length 286**. Offsets: notes at 46, mask at 270. Checkpoints from all
earlier obs versions (incl. the 166-dim v6 lineage and its 99.5%
record) are incompatible.

There is deliberately NO time feature: with constant speed,
`time_until = z/|vz|` — the approach axis carries the temporal ordering
spatially, as a player perceives it. `vz` is included so timing stays
recoverable if per-note speeds ever vary (then z alone is an insufficient
statistic for arrival). Both properties are pinned by tests.

## Gymnasium env (`beat_arena_env/rust_env.py`)

`RustBeatEnv(note_rate, colors, dirs, continuous_dirs, chart_len_s,
min_hand_gap_s, cols, rows, note_speed, note_speed_max, bomb_rate,
chain_prob, arc_prob, seed)`:
`observation_space = Box(-2, 2, (286,))` (bounds loose),
`action_space = MultiDiscrete([3]×12)` (per hand: pitch, yaw, twist,
elbow, roll, bend; left first). New chart every `reset(seed)` (chart
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
{"header": {"version": 7, "config": {..., "note_speed": 4.0, "bomb_rate": 0.0, "chain_prob": 0.0, "arc_prob": 0.0},
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
`python/record_replay.py` records episodes from a checkpoint / the
scripted bot / random (`--tag` prefixes filenames); logged rewards are
the raw interpretable ones.
Every interesting run should be recorded.

## Scripted reference bot (`beat_arena_env/scripted.py`)

Per hand, against its color's earliest pending scoreable object (bombs
are never targets; chain links are targeted with their TANGENT as the
swing direction, so one whip sweeps consecutive links). Env v7 changes:
the wrist plants at `PLANT_Z = −0.25` (shoulders sit 0.65 m behind the
plane — an on-plane wrist would be at full stretch), and the whip is
timed to pass through the contact angle `EARLY_S = 0.10` s BEFORE
`hit_time` — the elongated good box reaches the blade zone early, and a
plane-timed swing gets caught mid-wind-up (`angle`/`slow`); cutting
early inside the deep box is exactly what real players do. Otherwise
the v6 plan survives (plant, cock up to `SWING_ARC = 130°`, whip
through; the leftover rotation is the follow-through):

1. **Plant**: the wrist goal slides along the ray
   `center − s·(cosθ_c, sinθ_c, 0)` + `(0, 0, PLANT_Z)` (both
   perpendicular candidates, s ∈ [0.3, 0.95], ideal s = 0.75) to the
   most reachable point; `kinematics.ik_arm(hand, goal, swivel="back")`
   solves closed-form two-link IK with the elbow pushed toward −z,
   which keeps the FOREARM pointing at the notes — roll-swings then
   sweep near the hit plane.
2. **Re-aim**: the contact blade orientation is `θ_aim` = wrist→note
   from the pose the arm will actually reach (exact when the plant is
   reachable, best-effort otherwise — the 60° cut tolerance absorbs the
   residual). `kinematics.wrist_for_blade` inverts the forearm-frame
   blade formula; BOTH wrist representations (`(ψ, φ)` ≡ `(ψ±π, −φ)`)
   and both sweep senses are ranked: direction error ≤ 55° (hard cap),
   then wind-up room ≥ arc and follow-through room ≥ 45° inside the
   roll range (v5.2's room ranking, in roll space).
3. **Steering**: every joint is an independent force-free integrator,
   so each is bang-banged toward its target (desired ω = gain·error).
   The whip is FULL roll accel in the swing sense, timed by the wrist's
   `α/Ω` physics so the blade sweeps THROUGH `θ_aim` at contact.

Stateless per tick; solvability proof and eyeball baseline, not
optimal. Must land ≥ 80% good cuts on the default easy chart (tested),
with mean score well above the 15-point stab ceiling. Its flat-blade
technique caps accuracy (~5 of 15 — the blade plane sits ~0.2 m behind
the note center; forward-TILTED blades sweeping through the box depth
are the real-BS high-accuracy technique, open to the learned policy).
The realistic body costs it dearly on harder configs (see README
reference numbers) — by design ("that's the game").

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

## Training (`python/train_batch.py` — the only trainer)

Batched PPO: `BeatVecEnv` (N Rust envs, one FFI call per vector step),
one policy forward per vector step; GAE(γ=0.99, λ=0.95), dual-clip
(ε=0.2, c=3), ent 0.01, vf 0.5, Adam 3e-4, grad clip 0.5. Flags:
`--num-envs --horizon --minibatches --epochs --total-steps --time-cap
--ent-coef --seed --ckpt-every --run-dir --init-from --bench` + env
config (`--note-rate --colors --dirs --cols --rows --chart-len
--min-hand-gap --note-speed`) + curriculum mode
(`--curriculum` + friends, see §Curriculum training). Batch = num_envs×horizon;
NOTE: batch shape changes the optimization regime (gradient passes per
env step) — cross-shape learning comparisons need retuning.

## Curriculum training (`train_batch.py --curriculum`, 2026-07-20)

Alternative to the procedural generator: a **ladder of real BeatSaver
maps ordered by ScoreSaber star rating** (the community difficulty
scale; each ranked (map, difficulty) leaderboard carries a `stars`
value). Flow:

1. Start: one random ranked map near `--start-star` (default 1.0★),
   found via the ScoreSaber API (`beat_arena_env/scoresaber.py`,
   stdlib urllib only), downloaded from BeatSaver by song hash into
   `--map-dir` (default `maps/curriculum/`, a shared cache), parsed by
   `beatsaver.py` with `--speed-scale` (default 0.3, same as sightread).
   Unparseable maps (v4 schema, no Standard set) and maps longer than
   `--max-song-s` (default 400 s, which also sets the sim tick cap) are
   skipped; sparse star bands auto-widen ±0.1★ at a time (0.9–1.1★
   holds only ~5 ranked leaderboards).
2. Every episode's chart is drawn from the ladder: the newest (un-FC'd)
   map with probability `--new-map-weight` (default 0.5), the remainder
   uniform over the FC'd rungs — old maps stay in the mix so they don't
   regress, without the newest map's gradient share shrinking as the
   ladder grows.
3. Every `--fc-eval-every` updates (default 10) the newest map is played
   once **greedy** (argmax both heads — deterministic sim + fixed chart
   ⇒ one rollout is the exact answer), aborting on the first combo
   break. **Full combo** (= every note a good cut: zero bad cuts, zero
   misses) ⇒ checkpoint `fc_<stars>star.pt`, mark the rung FC'd, fetch a
   new random map `--star-step` (default 0.2) harder, keep training.
4. Ladder state (stages, hashes, FC flags) persists as JSON
   (`--curriculum-state`, default `<run-dir>/curriculum.json`); pointing
   a new run at an old state file resumes the ladder (zips re-download
   by hash if the cache was purged). Promotion failing on network error
   is retried at the next FC check.

Implementation: `beat_arena_env/curriculum.py` (`Curriculum` — ladder
state, weighted `sample(i, rng)` hook, per-stage rolling stats printed
at each FC check) + `BeatVecEnv(chart_sampler=..., max_steps=...)`
(sampler replaces `generate_chart` at auto-reset; the tick cap must
cover the longest map). Known limits: star ratings measure *human*
difficulty (NJS/nps patterns), which correlates only loosely with what
the agent finds hard at 30 Hz point-mass control; ultra-short ranked
memes (e.g. Dogbass, 17 notes / 4 s, 0.99★) are legitimate rungs.

## Rust sim core (`rust/`, crate `beat-arena-rs`)

THE physics implementation (PyO3 0.22 + numpy 0.22; build with
`VIRTUAL_ENV=<venv> maturin develop --release` from `rust/`). Charts are
injected per episode via `set_chart` (env v7 tuples: `(t, x, y, color,
angle, speed, any_dir, kind, free_pre, free_post)`).
`RawBeatSim.step(actions)` (12 ints, per hand pitch/yaw/twist/elbow/
roll/bend) returns `(obs, reward, done, events)` — events = `(note_idx,
code, points)` with codes 0 ok / 1 slow / 2 angle / 3 wrong_hand /
4 miss / 5 post (follow-through resolution) / 6 bomb hit — plus getters
`sabers()` (per hand `q(6) ++ om(6) ++ wrist(3)`), `blades()` (unit
blade dirs), `status()` (0 pending / 1 good / 2 bad·bomb-hit / 3 missed
/ 4 bomb passed), `bombs_hit`, `n_notes` (scoreable, bombs excluded),
`time/score` and a `set_joints(s, q, om)` test hook (FK cache refreshes
immediately). `VecBeatSim` steps N sims per FFI call (actions `(N, 12)`
i64); `counters(i)` = `(good, bad, miss, n_scoreable, score,
bombs_hit)`. The scenario tests in `tests/test_env.py` gate every rule
change.

Measured 2026-07-20 (8-core CPU): Python `env.step` 137 µs vs Rust
2.18 µs (**63×**; raw sim call 1.19 µs — the difference is gym-wrapper
glue). End-to-end training (8 serial envs): pool **1.6×**, transformer
**~1.3×** — the env is now ~0.3% of wall-clock and the torch policy is
the whole bottleneck; large parallel-env counts are effectively free on
the sim side (128 envs ≈ 0.28 ms/vector-step).

## Batched rollouts (`VecBeatSim` + `python/train_batch.py`)

`beat_arena_rs.VecBeatSim`: N sim cores stepped in ONE FFI call per
vector step (obs returned flat, reshaped Python-side; `obs_one(i)` for
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

1. Obs layout: length 286, offsets, arm-state block, kind/arc flags,
   mask semantics; the Python FK/IK mirror (`kinematics.py`) pinned
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
4. Chart gen: respects `min_hand_gap_s` per hand and the `dirs` subset;
   the v7 dials produce their kinds; chain-link Bézier layout (times
   strictly increasing to the tail, positions marching along the curve).
5. Policy junk-leak: garbage in masked note slots changes no
   logprob/entropy/value (key-masking).
6. Scripted bot ≥ 80% good cuts on default easy chart (solvability).
7. Curriculum (offline, network monkeypatched): state save/load
   round-trip, newest-map sample weighting, per-stage episode
   bookkeeping, promotion appends the next rung, `chart_sampler` path
   through `BeatVecEnv`, `greedy_fc_eval` counters (bombs break FC).
   NOTE: the ladder tests skip when the local (gitignored) fixture zips
   are absent.
8. BeatSaver parser fixtures: v2 (notes + bombs + dots), v3 (chains
   with the link Bézier, arcs as grants, precision angles, BPM change
   events), v4 (index-indirected arrays, dot chain heads), and the
   paired-note ANGLE SNAP rule (snapped diagonal pair; same-column
   control pair unsnapped).
