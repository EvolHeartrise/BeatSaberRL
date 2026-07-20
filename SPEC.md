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

## Sim rules (`beat_arena_env/sim.py`)

- Fixed tick `DT = 1/30` s. One policy action per tick.
- **Cut plane**: 4 columns × 3 rows grid; continuous coords `x ∈ [0,4]`,
  `y ∈ [0,3]` (y up). Note cell `(col,row)` has center `(col+.5, row+.5)`,
  cut radius `R = 0.55`.
- **Sabers**: 2 (index 0 = left/red, 1 = right/blue). Point mass:
  position `p` clamped to `[-1,5] × [-1,4]` (1 unit of wind-up room beyond
  the grid; clamping zeroes the clamped velocity component), velocity `v`,
  speed cap `V_MAX = 12` u/s.
- **Action** per tick, per saber: one of 9 — `0` = coast,
  `k ∈ 1..8` = accelerate at `A = 60` u/s² in direction `(k−1)·45°`
  (0° = +x, counter-clockwise). Integration order per tick:
  `v += a·DT`; `v *= (1 − DRAG)` with `DRAG = 0.02`; clamp `|v| ≤ V_MAX`;
  `p += v·DT`; clamp `p`. No action masks — every action is always legal.
- **Notes**: chart = list of `(hit_time, col, row, color, dir)` sorted by
  `hit_time`; `color ∈ {0 left/red, 1 right/blue}`, `dir ∈ 0..7` = required
  swing direction `(dir)·45°` (so `2` = upward swing, `6` = downward).
- **Cut window**: a note is *active* while `|t − hit_time| ≤ W`, `W = 0.15` s.
- **Cut evaluation**, each tick, for each active uncut note: a saber
  *crosses* the note if its displacement segment `p_prev → p` comes within
  `R` of the note center. Matching-color saber is evaluated first.
  - Matching saber crosses with `|v| ≥ V_MIN = 6` u/s and
    `cos∠(v, dir_vec) ≥ 0.5` (within 60°) → **good cut**.
  - Matching saber crosses but too slow / wrong angle → **bad cut**
    (reason `slow` / `angle`). Wrong-color saber crosses (only checked if
    the matching one didn't) → **bad cut** (reason `wrong_hand`).
  - Any cut (good or bad) consumes the note. Crossing outside the window
    does nothing (the block "isn't there yet").
- **Miss**: window expires with the note uncut → note consumed as missed.
- **Reward per tick**: `+1.0` per good cut, `−0.25` per bad cut, `−0.5`
  per miss. (Bad cut > miss: engaging wrongly beats not engaging — a
  deliberate shaping gradient. Graded swing quality is v4.)
- **Potential-based shaping** (added after the first smoke run showed the
  raw reward is too sparse — random motion never discovers the
  place×speed×direction×window conjunction, entropy never leaves uniform):
  `reward += 0.99·Φ(s′) − Φ(s)` with
  `Φ = −shaping_coef · Σ_hands min(dist(saber, its earliest pending
  matching note center), 4)/4`, `Φ(terminal) = 0`. `shaping_coef` default
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
restricting them is a spatial curriculum dial), `shaping_coef` (see §Sim
rules).

Generation: first note at `t = 1.0`; successive intervals
`(1/note_rate)·U(0.8, 1.2)`; color drawn 50/50 in `"both"` mode but a note
is pushed later if its hand's previous note is closer than `min_hand_gap_s`
(physical-feasibility guarantee — the same hand is never asked to be in two
places at once faster than it can travel). Col/row uniform over the grid.

## Observation layout (exact, all f32) — order is a contract

- `sabers: [f32; 8]` = per saber `[x/4, y/3, vx/12, vy/12]`, left then right.
- `notes: [[f32; 14]; 12]` — the next 12 *pending* (uncut, unmissed) notes
  in hit-time order, slot 0 = soonest. Per note:
  `[t_until/2 (t_until clipped to [−0.2, 3.0] s — the negative tail keeps
  "already past hit-time but still in window" visible), cx/4, cy/3, is_left, is_right,
  dir one-hot (8), valid=1]`. Empty slots all-zero. Slots SHIFT as notes
  are consumed (no stable identity — nothing targets a slot by index, so
  stability is not needed; contrast kiting-arena's frozen slots).
- `note_mask: [f32; 12]` — 1.0 = slot holds a pending note.

**Flat vector:** `sabers(8) ++ notes row-major(168) ++ note_mask(12)` →
**length 188**. Offsets: notes at 8, mask at 176.

`t_until` is the *temporal position* of the note — the transformer gets
time as an explicit input feature rather than a sinusoidal positional
encoding added to embeddings (same job, simpler and exact for this env).

## Gymnasium env (`beat_arena_env/env.py`)

`BeatEnv(note_rate, colors, dirs, chart_len_s, min_hand_gap_s, seed)`:
`observation_space = Box(-2, 2, (188,))` (bounds loose),
`action_space = MultiDiscrete([9, 9])`. New chart every `reset(seed)`.
`set_record(path)` dumps replay JSONL. `info` on final step carries
`good_cuts, bad_cuts, misses, n_notes`.

## Replay log (JSONL, version 1)

```json
{"header": {"version": 1, "config": {...}, "chart": [{"t":1.0,"col":2,"row":1,"color":1,"dir":6}, ...]}}
{"t": 0, "time": 0.0, "sabers": [[x,y,vx,vy],[x,y,vx,vy]], "action": [0,5],
 "reward": 0.0, "events": [{"cut": {"note": 3, "good": true, "reason": "ok"}} | {"cut": {"note": 4, "good": false, "reason": "slow|angle|wrong_hand"}} | {"miss": {"note": 5}}]}
```

**Viewer** (`viz/replay.html`): self-contained HTML/canvas, zero
dependencies — open in a browser, drop a replay `.jsonl` in. Renders the
grid, saber trails + velocity vectors, a white ring when a saber is
cut-capable (≥ 6 u/s), note fly-in over the 2 s before hit-time (white
outline while the cut window is open), required-direction arrows, good
cuts as split halves, bad cuts/misses with reason labels, HUD counters.
Controls: space play/pause, ←/→ tick-step, +/− speed, r restart, scrub
bar. Note outcomes are DERIVED from the event stream (single source of
truth), not re-simulated. `python/record_replay.py` records episodes from
a checkpoint / the scripted bot / random, always with `shaping_coef=0` so
logged rewards are the raw interpretable ones. Every interesting run
should be recorded.

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

- Per-note MLP `14 → 64 → 64` (shared across the 12 slots).
- Per-saber token MLP `6 → 64 → 64` on `[x/4, y/3, vx/12, vy/12, is_left,
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

1. Obs layout: length 188, offsets, mask semantics.
2. Determinism: same seed + scripted bot ⇒ identical rewards and event
   stream across two runs.
3. Cut detection unit scenarios (crafted saber trajectories): good cut
   registers; too-slow → bad `slow`; wrong angle → bad `angle`; wrong hand
   → bad `wrong_hand`; untouched → miss exactly when window expires; early
   crossing (before window) does nothing.
4. Chart gen: respects `min_hand_gap_s` per hand and the `dirs` subset.
5. Policy junk-leak (both encoders): garbage in masked note slots changes
   no logprob/entropy/value (transformer key-masking + pool masking).
6. Scripted bot ≥ 80% good cuts on default easy chart (solvability).
