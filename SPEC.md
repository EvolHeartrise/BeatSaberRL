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
sight-read eval) → v4 ideas (continuous control, graded swing scoring,
energy-bar fail, visualizer).

## Layout

```
projects/beat-arena/
  SPEC.md
  README.md
  python/
    beat_arena_env/      # package: chart gen, sim core, gymnasium env, scripted bot
    policy.py            # NotePolicy: pool | transformer encoder
    train_smoke.py       # CPU PPO (mirrors kiting-arena harness)
    tests/               # pytest
```

Uses kiting-arena's venv: `../kiting-arena/python/.venv` (torch-cpu,
gymnasium, pufferlib 3.0, pytest). No new dependencies in v0. No GPU;
paths contain a space — quote everything.

## Sim rules (`beat_arena_env/sim.py`) — world v2: metric, 3D notes

- Fixed tick `DT = 1/30` s. One policy action per tick.
- **World in METERS**, matching real Beat Saber block dimensions. The hit
  plane is `z = 0`, centered on the origin. 4 columns × 3 rows grid on a
  `CELL = 0.5` m lane pitch: cell `(col,row)` center =
  `((col−1.5)·0.5, (row−1.0)·0.5)`, grid spans `[-1,1] × [-0.75,0.75]`
  (y up). Notes are `BLOCK = 0.4` m cubes; cut radius `R = 0.28` m
  (≈ circumscribed circle of the 0.4 m face). All lengths/speeds/accels
  are the former unit system scaled by 0.5 — a pure unit change,
  dynamics identical (verified: scripted bot 97/88/66% across
  difficulties, matching pre-metric 96/88/65%).
- **Sabers**: 2 (index 0 = left/red, 1 = right/blue). Point mass on the
  hit plane: position clamped to `[-1.5,1.5] × [-1.25,1.25]` (0.5 m
  wind-up margin; clamping zeroes the clamped velocity component),
  velocity `v`, speed cap `V_MAX = 6` m/s. Start at rest at `(∓0.5, 0)`.
- **Action** per tick, per saber: one of 9 — `0` = coast,
  `k ∈ 1..8` = accelerate at `A = 30` m/s² in direction `(k−1)·45°`
  (0° = +x, counter-clockwise). Integration order per tick:
  `v += a·DT`; `v *= (1 − DRAG)` with `DRAG = 0.02`; clamp `|v| ≤ V_MAX`;
  `p += v·DT`; clamp `p`. No action masks — every action is always legal.
- **Notes**: chart = list of `(hit_time, col, row, color, dir)` sorted by
  `hit_time`; `color ∈ {0 left/red, 1 right/blue}`, `dir ∈ 0..7` = required
  swing direction `(dir)·45°` (so `2` = upward swing, `6` = downward).
  A note's 3D position is `(cell center, z)` with
  **`z(t) = (hit_time − t) · note_speed`** — z is derived from the
  schedule, so time and space can never disagree. `note_speed` default
  `4` m/s (config).
- **Cut window is SPATIAL**: a note is *active* while inside the slab
  `|z| ≤ Z_WINDOW = 0.6` m (at 4 m/s that is ±0.15 s of travel; faster
  notes get proportionally less time — tested).
- **Cut evaluation**, each tick, for each active uncut note: a saber
  *crosses* the note if its displacement segment `p_prev → p` comes within
  `R` of the note center. Matching-color saber is evaluated first.
  - Matching saber crosses with `|v| ≥ V_MIN = 1` m/s (real Beat Saber
    scores gentle swings — changed from 3.0 on 2026-07-20; results
    recorded before then used 3.0 and are not comparable) and
    `cos∠(v, dir_vec) ≥ 0.5` (within 60°) → **good cut**.
  - Matching saber crosses but too slow / wrong angle → **bad cut**
    (reason `slow` / `angle`). Wrong-color saber crosses (only checked if
    the matching one didn't) → **bad cut** (reason `wrong_hand`).
  - Any cut (good or bad) consumes the note. Crossing outside the window
    does nothing (the block "isn't there yet").
- **Miss**: the note exits the slab past the plane (`z < −Z_WINDOW`)
  uncut → consumed as missed.
- **Reward per tick**: `+1.0` per good cut, `−0.25` per bad cut, `−0.5`
  per miss. (Bad cut > miss: engaging wrongly beats not engaging — a
  deliberate shaping gradient. Graded swing quality is v4.)
- **Potential-based shaping** (added after the first smoke run showed the
  raw reward is too sparse — random motion never discovers the
  place×speed×direction×window conjunction, entropy never leaves uniform):
  `reward += 0.99·Φ(s′) − Φ(s)` with
  `Φ = −shaping_coef · Σ_hands min(dist(saber, its earliest pending
  matching note center), 2)/2` (in-plane distance, meters),
  `Φ(terminal) = 0`. `shaping_coef` default
  `0.2`, `0` disables. Potential-based form per Ng et al. 1999 — provably
  cannot change the optimal policy; it only densifies the gradient.
  Shaping must never change the event stream (tested).
- **Termination**: all notes consumed, or `t > last hit_time + 0.5 s`, or
  tick count = `max_steps` (default `chart_len_s·30 + 30`). All ends are
  termination (no bootstrap-relevant truncation in v0 — episodes are
  natural-length).
- **RNG**: `numpy.random.Generator(PCG64(seed))`; all randomness (chart
  gen) flows from it. Same seed + same action sequence ⇒ identical episode.

## Chart generation (`beat_arena_env/chart.py`)

`ChartConfig`: `note_rate` (notes/s, default 1.0), `chart_len_s` (default
20.0), `colors` (`"right"` = all blue, `"both"`, default `"right"`),
`dirs` (allowed subset of 0..7, default `[6]` = down-swings),
`min_hand_gap_s` (default 0.5), `cols`/`rows` (allowed positions —
restricting them is a spatial curriculum dial), `note_speed` (m/s toward
the plane, default 4.0), `shaping_coef` (see §Sim rules).

Generation: first note at `t = 1.0`; successive intervals
`(1/note_rate)·U(0.8, 1.2)`; color drawn 50/50 in `"both"` mode but a note
is pushed later if its hand's previous note is closer than `min_hand_gap_s`
(physical-feasibility guarantee — the same hand is never asked to be in two
places at once faster than it can travel). Col/row uniform over the grid.

## Observation layout (exact, all f32) — order is a contract

Obs v2 (spatial notes — no explicit time feature):

- `sabers: [f32; 8]` = per saber `[x/1, y/0.75, vx/6, vy/6]`, left then
  right (normalizers = grid half-extents and `V_MAX`; signed, symmetric).
- `notes: [[f32; 15]; 12]` — the next 12 *pending* (uncut, unmissed) notes
  in arrival order, slot 0 = soonest. Per note:
  `[cx/1, cy/0.75, z/8 (z clipped to [−1, 12] m — the negative tail keeps
  "past the plane but still in the slab" visible), vz/8 (= −note_speed/8),
  is_left, is_right, dir one-hot (8), valid=1]`. Empty slots all-zero.
  Slots SHIFT as notes are consumed (no stable identity — nothing targets
  a slot by index; contrast kiting-arena's frozen slots).
- `note_mask: [f32; 12]` — 1.0 = slot holds a pending note.

**Flat vector:** `sabers(8) ++ notes row-major(180) ++ note_mask(12)` →
**length 200**. Offsets: notes at 8, mask at 188. Pre-v2 (188-dim)
checkpoints are incompatible.

There is deliberately NO time feature: with constant speed,
`time_until = z/|vz|` — the approach axis carries the temporal ordering
spatially, as a player perceives it. `vz` is included so timing stays
recoverable if per-note speeds ever vary (then z alone is an insufficient
statistic for arrival). Both properties are pinned by tests.

## Gymnasium env (`beat_arena_env/env.py`)

`BeatEnv(note_rate, colors, dirs, chart_len_s, min_hand_gap_s, seed)`:
`observation_space = Box(-2, 2, (200,))` (bounds loose),
`action_space = MultiDiscrete([9, 9])`. New chart every `reset(seed)`.
`set_record(path)` dumps replay JSONL. `info` on final step carries
`good_cuts, bad_cuts, misses, n_notes`.

## Replay log (JSONL, version 2)

Version 2 = metric world; `config` carries `note_speed`, and viewers
derive each note's `z` from the schedule (`z = (t_note − t) · speed`).
The 3D viewer rejects v1 files (pre-metric coordinates).

```json
{"header": {"version": 2, "config": {..., "note_speed": 4.0}, "chart": [{"t":1.0,"col":2,"row":1,"color":1,"dir":6}, ...]}}
{"t": 0, "time": 0.0, "sabers": [[x,y,vx,vy],[x,y,vx,vy]], "action": [0,5],
 "reward": 0.0, "events": [{"cut": {"note": 3, "good": true, "reason": "ok"}} | {"cut": {"note": 4, "good": false, "reason": "slow|angle|wrong_hand"}} | {"miss": {"note": 5}}]}
```

**Viewer** (`viz/replay.html`): self-contained HTML/canvas **3D**
viewer, zero dependencies — open in a browser, drop a replay `.jsonl`
in. Hand-rolled perspective projection (camera slightly above/behind the
plane): tunnel rails + depth rings to `z = 12` m, 40 cm note cubes flying
in at `note_speed` with required-direction arrows on the front face
(white outline while inside the cuttable slab, which is drawn faintly),
the 4×3 plane grid with correct occlusion (notes past the plane draw in
front), saber discs with trails + velocity vectors and a white ring when
cut-capable (≥ 1 m/s), good cuts as split halves drifting apart,
bad cuts/misses labeled with reason, HUD counters. Controls: space
play/pause, ←/→ tick-step, +/− speed, r restart, scrub bar. Note
positions and outcomes are DERIVED from the chart + event stream (single
source of truth), not re-simulated. `python/record_replay.py` records
episodes from a checkpoint / the scripted bot / random (`--tag` prefixes
filenames), always with `shaping_coef=0` so logged rewards are the raw
interpretable ones. Every interesting run should be recorded.

## Scripted reference bot (`beat_arena_env/scripted.py`)

Per hand, against its color's earliest pending note: **wind-up phase**
(steer to `center − dir_vec·0.9` via arrival steering: desired velocity
toward goal with slowdown near it, pick the discrete accel minimizing the
gap to desired velocity) then **swing phase** (when `t_until ≤ 0.22` s,
full accel along `dir_vec`) until the note resolves; repeat. This is the
solvability proof and eyeball baseline, not optimal. Must land ≥ 80% good
cuts on the default easy chart (tested).

## Policy (`python/policy.py`)

`NotePolicy(encoder="pool"|"transformer")`, obs contract offsets mirrored:

- Per-note MLP `15 → 64 → 64` (shared across the 12 slots).
- Per-saber token MLP `6 → 64 → 64` on `[x/1, y/0.75, vx/6, vy/6, is_left,
  is_right]` (hand identity via one-hot, weights shared).
- `encoder="pool"`: masked max-pool over note embeddings (64) ++ raw saber
  features (8) → trunk. Saber–note interaction happens only in the trunk.
- `encoder="transformer"`: token sequence = `[saber_L, saber_R,
  note_0..11]` (14 tokens); key mask `[1, 1, note_mask]`. **2 × full
  pre-LN transformer blocks** (LayerNorm → single-head QKV self-attention
  → residual; LayerNorm → FFN 64→128→64 → residual) — unlike kiting-arena's
  deliberately minimal arm, this applies its lesson: give attention its
  standard optimization aids. Readout = concat of the two saber tokens'
  output embeddings (128) → trunk.
- Trunk `→ 128 → 128`, heads: `left(9)`, `right(9)`, `value(1)`.
  Both heads always execute; `logprob = logp_left + logp_right` (true
  independence factorization — no conditional gating, unlike kiting).

## Training (`python/train_smoke.py`)

Same harness shape as kiting-arena: PufferLib
`GymnasiumPufferEnv` + `vector.Serial`, 8 envs × 128 horizon, 3 epochs ×
4 minibatches, GAE(γ=0.99, λ=0.95), dual-clip PPO (ε=0.2, c=3), ent 0.01,
vf 0.5, Adam 3e-4, grad clip 0.5. Flags: `--total-steps --time-cap
--ckpt-every --run-dir --seed --encoder --note-rate --colors --dirs
--cols --rows --chart-len --shaping-coef --ent-coef --init-from`.
Checkpoints record the encoder arm. Eval reports
random / trained / **scripted** mean return over the same eval seeds —
scripted is the ceiling reference.

## Tests (`python/tests/`, pytest)

1. Obs layout: length 200, offsets, mask semantics.
2. Determinism: same seed + scripted bot ⇒ identical rewards and event
   stream across two runs.
3. Cut detection unit scenarios (crafted saber trajectories): good cut
   registers; too-slow → bad `slow`; wrong angle → bad `angle`; wrong hand
   → bad `wrong_hand`; untouched → miss exactly when the note exits the
   slab; early crossing (before the slab) does nothing.
   Plus obs-v2 pins: `time_until = z/|vz|` holds at every step; faster
   note_speed shortens the slab's time window.
4. Chart gen: respects `min_hand_gap_s` per hand and the `dirs` subset.
5. Policy junk-leak (both encoders): garbage in masked note slots changes
   no logprob/entropy/value (transformer key-masking + pool masking).
6. Scripted bot ≥ 80% good cuts on default easy chart (solvability).
