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
→ **v5 (blade sabers: wrist rotation DOFs, real saber length, 115-point
graded swing scoring — current)**.

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
(which drive the Rust sim directly via its `set_saber` hook and event
stream).

Uses kiting-arena's venv: `../kiting-arena/python/.venv` (torch-cpu,
gymnasium, pufferlib 3.0, pytest). No new dependencies in v0. No GPU;
paths contain a space — quote everything.

## Sim rules (`rust/src/lib.rs`) — env v5: blade sabers, graded swing scoring

Env v5 (2026-07-21, user-designed): sabers stop being points. Each hand
is a **wrist** (point mass on the hit plane, exactly the old point-saber
dynamics) plus a rigid **blade** of real length whose 3D direction is
driven by two wrist-rotation DOFs. Cuts are blade-through-block sweeps,
scored with the real Beat Saber 115-point system (pre-swing / follow-
through / accuracy). All v4 checkpoints/replays retire (hard replace).

- Fixed tick `DT = 1/30` s. One policy action per tick.
- **World in METERS**, matching real Beat Saber block dimensions. The hit
  plane is `z = 0`, centered on the origin. 4 columns × 3 rows grid on a
  `CELL = 0.5` m lane pitch: cell `(col,row)` center =
  `((col−1.5)·0.5, (row−1.0)·0.5)`, grid spans `[-1,1] × [-0.75,0.75]`
  (y up). Notes are `BLOCK = 0.4` m cubes; cut radius `R = 0.28` m
  (≈ circumscribed circle of the 0.4 m face). Notes fly in from +z.
- **Wrists**: 2 (index 0 = left/red, 1 = right/blue). Point mass ON the
  hit plane: position clamped to `[-1.5,1.5] × [-1.25,1.25]` (clamping
  zeroes the clamped velocity component), velocity `v`, speed cap
  `V_MAX = 6` m/s, accel `A = 30` m/s², drag `DRAG = 0.02` per tick.
  Start at rest at `(∓0.5, 0)`.
- **Blade** per hand (env v5.2, 2026-07-21 — forearm-anchored
  parameterization; supersedes the v5.0/v5.1 azimuth+tilt model, whose
  neutral was in-plane and which let blades point straight back at the
  player): rigid segment of length `SABER_LEN = 1.0` m (real Beat
  Saber) from the wrist along
  `d(ψ, φ) = (−sinψ·sinφ, cosψ·sinφ, cosφ)` — the neutral blade is the
  forearm's extension, **pointing at the incoming notes (+z)**:
  - `ψ` = **roll**: forearm CW/CCW rotation about the forward axis
    (pronation/supination). It orients the plane the wrist bends in; it
    does NOT move a straight-forward blade (`d` is ψ-independent at
    φ = 0 — spinning a forward saber about its own axis, correct
    physics). Anatomical limits from the palm-down neutral (ψ = 0):
    right 90° CCW / 190° CW → `ψ_R ∈ [−190°, 90°]`; mirrored left
    `ψ_L ∈ [−90°, 190°]`. 280° arcs, (cos ψ, sin ψ) unique, no wrap;
    hitting a limit zeroes `ω_ψ`.
  - `φ` = **bend**: wrist deflection away from straight-forward
    (flexion/extension), signed, clamped to `±BEND_LIM = 95°`
    (1.65806 rad). `φ = 0` points at the notes; `|φ| = 90°` lays the
    blade flat in the hit plane (in-plane azimuth = `ψ + 90°` for
    `+φ`, `ψ − 90°` for `−φ` — each azimuth is reachable by flexion OR
    extension, like a real wrist); `|φ| = 95°` reaches 5° past the
    plane. **The blade can therefore NEVER point back at the player**:
    `d_z ≥ cos 95° ≈ −0.087` by construction. Hitting the clamp zeroes
    `ω_φ`.
  - Angular dynamics per DOF mirror the linear ones:
    `ω += ±ALPHA_ROT·DT` (`ALPHA_ROT = 200` rad/s²), `ω *= (1 − DRAG)`,
    clamp `|ω| ≤ OMEGA_MAX = 20` rad/s, `angle += ω·DT`.
    Start pose: `ψ = 0, φ = 0` (both blades pointing at the notes),
    `ω = 0`. Checkpoints/replays from v5.0–v5.1 retire (the 98.7%
    v5.1 policy demonstrably held blades tilted BACKWARD — the exploit
    this model closes).
- **Action** per tick, per hand: `(move, roll, bend)` —
  `move` one of 9 (0 = coast, `k ∈ 1..8` = accelerate at `A` toward
  `(k−1)·45°`); `roll`/`bend` each one of 3 (0 = coast, 1 = `+α`,
  2 = `−α`; roll positive = CCW from the player's view, bend positive =
  toward the `+φ` flexion side). Full action =
  `MultiDiscrete([9,3,3, 9,3,3])`, left hand first.
  Integration order per tick: translation (`v += a·DT`; drag; cap;
  `p += v·DT`; clamp) then rotation (per-DOF as above). No action masks.
- **Notes** (unchanged from env v4): chart = list of
  `(hit_time, col, row, color, angle, speed, any_dir)` sorted by
  `hit_time`; a note's 3D position is `(cell center, z)` with
  **`z(t) = (hit_time − t) · speed`**.
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
- **Cut evaluation** (no more slab): a note is testable while
  `−Z_MISS ≤ z ≤ REACH_Z = 1.28` m (max tip z = a straight-forward
  blade, 1 m, + R; the reach window replaces the old ±0.6 m slab —
  timing tolerance is physical: a flat blade covers |z| ≤ R ≈ ±0.07 s
  at 4 m/s, pointing forward buys earlier contact, and behind the plane
  the blade reaches only `cos 95°·1 m + R ≈ 0.37 m`). Each tick the blade sweep is
  approximated by `N_BLADE_SAMPLES = 8` points at `s·SABER_LEN`,
  `s = 1/8 .. 8/8`: each sample's displacement segment (prev wrist+pose →
  current wrist+pose, full 3D) is tested against the note center (3D
  point-segment distance ≤ `R` = crossed). Matching-color saber first;
  contact sample `s*` = the sample with minimum distance `d_min`.
  - Contact velocity `v_c = (vx, vy, 0) + s*·SABER_LEN·ḋ` where
    `ḋ = θ̇·(−cosφ·sinθ, cosφ·cosθ, 0) + φ̇·(−sinφ·cosθ, −sinφ·sinθ, cosφ)`
    — wrist translation plus blade rotation at the contact point.
  - Wrong-color saber crosses (checked only if matching didn't) →
    **bad cut** (`wrong_hand`). Matching saber: `|v_c| < V_MIN = 1` m/s
    → `slow`; else in-plane direction check
    `cos∠((v_c.x, v_c.y), dir_vec) ≥ 0.5` (dot notes exempt) → **good
    cut**, otherwise `angle`. Any cut consumes the note.
- **Score** per good cut — real Beat Saber 115: 
  - **pre-swing** `70 · min(swing_acc / 100°, 1)` — the wind-up arc.
  - **accuracy** `15 · (1 − d_min/R)` — how centered the cut was.
  - **follow-through** `30 · min(post_arc / 60°, 1)` — a per-note
    tracker opens at the cut and accumulates the hand's subsequent swing
    arc under the same chain rules (break/reversal ends it); it resolves
    when 60° is reached, the chain ends, `FOLLOW_TIMEOUT = 15` ticks
    pass, or the episode terminates.
- **Reward per tick**: good cut pays `(pre + accuracy)/115` at the cut
  tick and `follow-through/115` at its resolution tick (max total +1.0
  per note, same scale as v4); bad cut `−0.5`; miss `−0.5`. A
  rotation-free "stab" (translation only) is a valid good cut but scores
  only accuracy (≤ 15/115) — swinging is now the whole game.
- **Miss**: the note passes `z < −Z_MISS = 0.6` m uncut → consumed as
  missed.
- **No reward shaping** (mechanism REMOVED 2026-07-21, user direction):
  rewards come only from the sim's scoring, like real Beat Saber. The
  potential-based shaping (Ng et al. 1999) used in v0's earliest runs was
  already off by default in the pure-reward era (2026-07-20); its code
  and the `shaping_coef` knob are now deleted everywhere.
- **Termination**: all notes consumed AND no follow-through pending (so
  the last note's post-swing isn't truncated), or
  `t > last hit_time + 0.5 s`, or tick count = `max_steps` (default
  `chart_len_s·30 + 30`); the hard ends force-resolve pending
  follow-throughs with their partial arcs. All ends are termination (no
  bootstrap-relevant truncation — episodes are natural-length).
- **RNG**: `numpy.random.Generator(PCG64(seed))`; all randomness (chart
  gen) flows from it. Same seed + same action sequence ⇒ identical episode.

## Chart generation (`beat_arena_env/chart.py`)

`ChartConfig`: `note_rate` (notes/s, default 1.0), `chart_len_s` (default
20.0), `colors` (`"right"` = all blue, `"both"`, default `"right"`),
`dirs` (allowed subset of 0..7, default `[6]` = down-swings),
`min_hand_gap_s` (default 0.5), `cols`/`rows` (allowed positions —
restricting them is a spatial curriculum dial), `note_speed` (m/s toward
the plane, default 4.0).

Generation: first note at `t = 1.0`; successive intervals
`(1/note_rate)·U(0.8, 1.2)`; color drawn 50/50 in `"both"` mode but a note
is pushed later if its hand's previous note is closer than `min_hand_gap_s`
(physical-feasibility guarantee — the same hand is never asked to be in two
places at once faster than it can travel). Col/row uniform over the grid.

## Observation layout (exact, all f32) — order is a contract

Obs v4 (env v5 blade state; note block unchanged from v3):

- `sabers: [f32; 20]` = per hand (left then right), 10 features:
  `[x/1, y/0.75, vx/6, vy/6, cosψ, sinψ, φ/BEND_LIM, ωψ/20, ωφ/20,
  min(swing_acc/100°, 2)]` — wrist state, joint pose (roll as its unit
  vector — wrap-free and unique on a 280° range; signed normalized
  bend), angular velocities, and the live pre-swing meter (1.0 = full
  pre-swing credit banked; clipped at 2 — reward-relevant state the
  policy must see).
- `notes: [[f32; 9]; 12]` — the next 12 *pending* (uncut, unmissed) notes
  in arrival order, slot 0 = soonest. Per note:
  `[cx/1, cy/0.75, z/8 (z clipped to [−1, 12] m — the negative tail keeps
  "past the plane but still in the slab" visible), vz/8 (= −speed/8,
  PER-NOTE since env v4 — vz is a live feature under variable speeds),
  is_left, is_right, dir_cos, dir_sin, valid=1]`. The direction is its
  UNIT VECTOR (cos θ, sin θ) — obs v3, 2026-07-20, replacing the 8-dim
  one-hot: continuous precision angles drop in without an obs change, and
  the network receives the exact object the cut rule uses
  (`cos∠ = v·d/|v|`); adjacent directions share representation. Empty
  slots all-zero.
  Slots SHIFT as notes are consumed (no stable identity — nothing targets
  a slot by index; contrast kiting-arena's frozen slots).
- `note_mask: [f32; 12]` — 1.0 = slot holds a pending note.

**Flat vector:** `sabers(20) ++ notes row-major(108) ++ note_mask(12)` →
**length 140**. Offsets: notes at 20, mask at 128. Checkpoints from obs
v1 (188), v2 (200), and v3 (128, point sabers) are all incompatible —
the v3 lineage (98.2% dynamic-gym record) ended at env v5.

There is deliberately NO time feature: with constant speed,
`time_until = z/|vz|` — the approach axis carries the temporal ordering
spatially, as a player perceives it. `vz` is included so timing stays
recoverable if per-note speeds ever vary (then z alone is an insufficient
statistic for arrival). Both properties are pinned by tests.

## Gymnasium env (`beat_arena_env/rust_env.py`)

`RustBeatEnv(note_rate, colors, dirs, continuous_dirs, chart_len_s,
min_hand_gap_s, cols, rows, note_speed, note_speed_max,
seed)`:
`observation_space = Box(-2, 2, (140,))` (bounds loose),
`action_space = MultiDiscrete([9, 3, 3, 9, 3, 3])` (per hand: move, rot,
tilt; left first). New chart every `reset(seed)` (chart gen stays in
Python — single source for charts). Per-step `info["events"]` =
(note_idx, reason, points) from the Rust event stream (reason `post` =
follow-through resolution, note already counted); final step adds
`good_cuts, bad_cuts, misses, n_notes, score` (score = summed points;
max = 115 · n_notes). `set_record(path)` dumps replay JSONL v4 built
from those events. Batched training path:
`beat_arena_env/vec_env.py: BeatVecEnv` over `VecBeatSim`.

## Replay log (JSONL, version 5)

Version 5 (env v5.2): saber entries carry the joint pose
`[x, y, vx, vy, roll, bend, om_roll, om_bend]`; cut events carry
`points` (pre+accuracy at the cut, follow-through via a separate `post`
event); the action is the 6-int vector. v4 replays (the retired v5.0/
v5.1 azimuth+tilt parameterization) and older are rejected.

```json
{"header": {"version": 5, "config": {..., "note_speed": 4.0}, "chart": [{"t":1.0,"col":2,"row":1,"color":1,"angle":4.712,"speed":4.0,"any":false}, ...]}}
{"t": 0, "time": 0.0, "sabers": [[x,y,vx,vy,roll,bend,om_roll,om_bend],[...]], "action": [0,1,0,5,2,0],
 "reward": 0.0, "events": [{"cut": {"note": 3, "good": true, "reason": "ok", "points": 78.4}} | {"cut": {"note": 4, "good": false, "reason": "slow|angle|wrong_hand", "points": 0}} | {"post": {"note": 3, "points": 26.1}} | {"miss": {"note": 5}}]}
```

**Viewer** (`viz/replay.html`, since 2026-07-21 a **Svelte 5 + Threlte 8
(three.js) SPA** compiled to that single static HTML file — still a
double-clickable, drop-a-`.jsonl`-in, zero-server page): real WebGL 3D
with **OrbitControls** (drag = orbit, wheel = zoom, right-drag = pan).
Source in `viz/app/` (Vite + vite-plugin-singlefile); rebuild with
`npm install && npm run build`, then copy `dist/index.html` over
`../replay.html`. `npm test` runs the pure-logic tests (node) and
`node test/browser.test.js` drives the built file in headless Edge
(loads a real replay via the file input, asserts playback advances with
zero page errors, screenshots to `test/out/`). Scene (world → three.js:
`(x, y, z) → (x, y, −z)`, camera on +z): tunnel rails + depth rings to
`z = 12` m, the 4×3 plane grid, the blade-reach region (z −0.6..1.28)
drawn faintly, 40 cm note cubes with required-direction arrows / dot
discs (white edge outline while in reach), **blades drawn wrist→tip as
cylinders** (white-hot core when the TIP is contact-capable ≥ 1 m/s)
with 10-tick tip trails, wrist discs + velocity vectors, good cuts as
split halves drifting apart with "+N" score popups, follow-through
"+N follow" popups, bad cuts/misses labeled with reason. **Per-hand wrist
gauges** (toggle `g`): camera-attached panels showing the roll dial
(needle = blade azimuth inside the hand's allowed 280° arc, forbidden
zone dark, limit ticks) and the tilt dial (±95°, right = toward the
notes), with numeric angles + angular velocities; needles turn orange
within 10° of a joint limit. All in-scene text/panels are
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

Per hand, against its color's earliest pending note. The cut config: at
contact the blade must be PERPENDICULAR to the required direction `e`
(swinging sweeps the contact point along `e`). Two candidates — CCW
swing (`θ_c = angle − 90°`) or CW (`θ_c = angle + 90°`), each with wrist
goal `center − 0.75·SABER_LEN·(cosθ_c, sinθ_c)` (note passes the blade
at 75% of its length). Env v5.2: the bot swings FLAT — it holds the
bend at `φ = +90°` (blade in the hit plane, 5° shy of the clamp) and
swings by ROLLING, so the blade's in-plane azimuth is `ψ + 90°` and the
reachable azimuth window equals `[ψ_lo, ψ_hi] + 90°` (= [0°, 280°] left
/ [−100°, 180°] right, the same windows as v5.1). Each candidate's
`θ_c` is fitted into that window (mod 2π; at least one of the two
opposite orientations always fits a 280° arc); candidates are ranked by
full-swing feasibility (wind-up room ≥ `SWING_ARC = 130°` AND
follow-through room ≥ 45° inside the limits), then wrist-goal
in-bounds, then wrist distance. The wind-up target is
`θ_c − sense·min(SWING_ARC, room_back − 5°)` and the whip starts at a
per-arc swing time (accel/Ω_MAX physics + 0.02 s margin). **Wind-up
phase**: wrist steers to the goal (arrival steering) while the roll
steers (bang-bang toward a desired ω via the 3-way roll action, direct
unwrapped error — the short way around may cross a limit) to the
wind-up target and the bend steers to `+90°`. **Swing phase**: full
roll accel in the swing sense until the note resolves; the leftover
rotation after contact is the follow-through.
Stateless per tick; solvability proof and eyeball baseline, not
optimal. Must land ≥ 80% good cuts on the default easy chart (tested),
with mean score well above the 15-point stab ceiling.

## Policy (`python/policy.py`)

`NotePolicy()` — transformer only (the pool baseline was deleted
2026-07-20 after the easy-config studies concluded; checkpoints still
record `encoder: "transformer"`):

- Per-note MLP `9 → 64 → 64` (shared across the 12 slots).
- Per-saber token MLP `12 → 64 → 64` on the 10 obs-v4 saber features +
  `[is_left, is_right]` (hand identity via one-hot, weights shared).
- Token sequence = `[saber_L, saber_R, note_0..11]` (14 tokens); key mask
  `[1, 1, note_mask]`. **2 × full pre-LN transformer blocks** (LayerNorm →
  single-head QKV self-attention → residual; LayerNorm → FFN 64→128→64 →
  residual). Readout = concat of the two saber tokens' output embeddings
  (128) → trunk.
- Trunk `→ 128 → 128`, heads (env v5): `left_move(9), left_rot(3),
  left_tilt(3), right_move(9), right_rot(3), right_tilt(3)`, `value(1)`.
  All heads always execute; `logprob = Σ` over the six factored heads
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
injected per episode via `set_chart` (~0.5 µs/step amortized).
`RawBeatSim.step(al_move, al_rot, al_tilt, ar_move, ar_rot, ar_tilt)`
returns `(obs, reward, done, events)` — events = `(note_idx, code,
points)` with codes 0 ok / 1 slow / 2 angle / 3 wrong_hand / 4 miss /
5 post (follow-through resolution) — plus getters
`sabers()` (per hand `[x,y,vx,vy,roll,bend,om_roll,om_bend]`),
`status()/time/score/counters` and a `set_saber(s, x, y, vx, vy, roll,
bend, om_roll, om_bend)` test hook. `VecBeatSim` steps N sims per FFI call
(actions `(N, 6)` i64); `counters(i)` = `(good, bad, miss, n_notes,
score)`. The scenario tests in `tests/test_env.py` gate every rule
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

1. Obs layout: length 140, offsets, saber pose block, mask semantics.
2. Determinism: same seed + fixed action stream ⇒ identical rewards and
   events across two runs.
3. Cut detection unit scenarios driven on the RUST sim via `set_saber`
   + the event stream: rotational good cut with sensible points;
   too-slow → `slow`; opposite swing → `angle`; wrong hand →
   `wrong_hand`; untouched → miss exactly at the z = −0.6 exit; a
   translation-only stab is good but scores ≤ accuracy (15); full
   pre-swing wind-up ≈ 70+acc points; follow-through emits a `post`
   event (total ≈ 115); accuracy decreases off-center; tilted blade
   reaches a note beyond the old slab, unreachable note untouched.
   Recording round-trip (JSONL v4 counters + score match the sim) and
   VecBeatSim(1) ≡ RawBeatSim.
4. Chart gen: respects `min_hand_gap_s` per hand and the `dirs` subset.
5. Policy junk-leak: garbage in masked note slots changes no
   logprob/entropy/value (key-masking).
6. Scripted bot ≥ 80% good cuts on default easy chart (solvability).
7. Curriculum (offline, network monkeypatched): state save/load
   round-trip, newest-map sample weighting, per-stage episode
   bookkeeping, promotion appends the next rung, `chart_sampler` path
   through `BeatVecEnv`, `greedy_fc_eval` counters.
