# Architecture brainstorm — policy/critic changes for the accuracy ceiling (2026-09-14)

Scope: MODEL changes only (policy.py / train_batch.py). Reward, gamma/lambda,
env physics, obs lanes produced by the env are out of scope. Everything is
ranked by expected value per pod-hour at the standard probe shape.

## The regime this is written against (measured facts, sources in brackets)

- Model: 27 tokens (saber_L, saber_R, body, 24 entity slots × 18 lanes) → 3
  pre-LN blocks (embed 192, 4 heads, ffn ×4) → readout of the 3 lead tokens
  → plain 2-layer ReLU trunk (576 → 384 → 384) → one packed (384 → 29·5)
  head GEMM; adapool_v1 critic (body-derived query over the resolved token
  stream + 3-layer MLP on [pooled, trunk]) = 2,414,930 params, 1.90M actor /
  0.52M critic. [policy.py]
- The width step 128/256 → 192/384 (1.09M → 2.41M) under Muon at matched
  1.12B samples: +0.45 pt trunk acc, +3.4% bench upp, gain GROWING with star
  rating (+0.2 pt at 4–6★, +0.9 at 12–14★), falls halved, +15% wall.
  [sessions/2026-09-12-five-credit-horizon.md]
- Trunk-feature effective rank (99% energy) ≈ 60% of width at BOTH widths
  (158/256, 232/384 at upd 451); Muon holds it, Adam collapsed 165 → 50 over
  4.7B. Dormant fraction 0.0 everywhere. The rank is measured on the OUTPUT
  of the plain trunk MLP. [same session; v40-probes]
- Residual bench loss = (a) mid/top-row cut accuracy on hard maps (the
  ACCURACY component of the 115, i.e. cut-plane distance to note centre),
  (b) early terminations (falls) on specific maps. [v40-probes bench read;
  five-credit wrap-up]
- Every good cut lands ~3 ticks early with elbows near-straight: γ pays
  +3.6%/tick for earliness; γ changes lost badly (0.98: −1.6 pt warm-start;
  0.99 arms −5 pt from scratch, policy never sharpens, ent 26–28 vs 18).
  **Not an architecture problem** — no model change below targets it; the
  user has ruled out reward-timing and posture fixes. [five-credit]
- Critic: ev 0.992–0.994 at the recipe γ; the critic's gradient share
  measurably steers the shared trunk when its targets get noisier (γ 0.99
  arms: ev 0.965–0.978 AND the actor stopped committing). `--critic-detach`
  (PPG-style stop-grad at the trunk boundary) and the gradcos/srank
  `[diag]` line already exist. [train_batch.py 1063, 1675–1770]
- fp16: rollout/update ratio mismatch r1 = 3.8e-3 at width 384 vs 1.6e-3 at
  256, rising ~linearly over the trunk; clipfrac_first exactly 0, so it has
  not reached the clip window. Trigger for action: r1 > 1e-2. [five-credit]
- Dead gears (K=5: 0 coast, 1 +1, 2 −1, 3 +¼, 4 −¼): ±1.0 is <1% used on
  spine (0.0%), hips, ankles, knee-extend; knees run on −¼ (72–82%); arms use
  all five. Per-head entropy floors at trunk end: legs 0.55–0.63 nats, spine
  0.36–0.38 — the entropy bonus keeps paying for mass on fall-inducing
  gears. [five-credit "Dead gears"]
- Wall: env kernel 70%, policy forward 9%, update 21%; ~190–220k steps/s at
  32,768 × T=32; VRAM 32 GB (peak ~13 GiB at mb 16,384 on the 4090-era
  measurement). Update is clip-bound (clipfrac_last ≈ 0.6). Adv-filter 0.25
  = free −18% wall (untested at recipe γ).
- Noise floor: trunk-acc gaps < 0.3 pt are seed noise; bench MDD ≈ 98k upp
  (σ_diff ≈ 35k); benches compare SAME POD ONLY (up to +4.5% across hosts
  on identical weights). The current pod (hjwiiue72y2uz3) is not the pod
  v40_w192 was benched on → any bench comparison needs a re-bench of
  `runs/v40_w192_cd300/leaf_from_upd01068.pt` on it first.

Standard probe (used for every "test" below): 32,768 envs × T=32, 1,068
updates = 1.12B samples, Muon, lr 5.7e-4 (critic ×3), seed 0, adv-filter
0.5, from scratch, vs v40_w192 (trunk last-50 0.9466; cd300 leaf 0.9670;
bench 2,353,043 upp / 16,420 wpp same-pod). Readouts: trunk acc windows
(201–250, 401–450, 601–650, 1019–1068), falls/M, ent, ev, srank/psrank,
r1, steps/s; then cd300 leaf (seed 1) + bench + per-star Δacc + per-row
acc on the 5-map greedy set. ≈ 100 min trunk + 25 min leaf + 20 min bench.

## Ranking (expected value per pod-hour)

| # | Proposal | Symptom it targets | Expected Δ (trunk acc, honest) | Code cost | Wall cost | Ckpt compat | Literature |
|---|---|---|---|---|---|---|---|
| 1 | Hand-relative geometric lanes on the note tokens (policy-side, obs unchanged) | mid/top-row cut accuracy; every hand↔note relation is currently learned through attention alone | +0.3–0.8 early, +0.1–0.4 at plateau | ~50 lines policy.py | ≈0 | warm-start via zero-padded columns | frame-choice evidence (Frame Mining); no direct PPO-humanoid ablation |
| 2 | SimBa-style residual+LN trunk (replace the plain 2-layer ReLU trunk) | srank 60% of width on exactly this MLP's output; Adam-era collapse to 50/256 | +0.1–0.4; better long-trunk rank | ~25 lines | ≈0 (param-matched) / +5% (ffn×4) | trunk reinit or partial load | SimBa, SimBaV2, Creus Castanyer (PPO) |
| 3 | Lower-body gear heads K=3 over {coast, ±¼} (policy-side id remap) + per-group entropy coefficients | dead full-torque gears; entropy bonus paying for fall-inducing mass; falls on specific maps | +0–0.3 acc, falls −20–40% | ~40 lines policy + ~20 trainer | ≈0 | exact warm start by slicing head rows | unsupported (no paper); follows from the census |
| 4 | Next width/depth rung + fp32 head GEMM | proven width gain growing with difficulty; r1 growth | +0.2–0.4 | 0 (flags) + ~10 lines fp32 heads | +20–30% | new arch (leaves load) | Hilton scaling; SONIC; Muon width transfer |
| 5 | Per-limb structured heads (arm_L / arm_R / lower-body sub-trunks off the readout) | cross-limb interference in one 384-wide bottleneck; leg-fall gradients vs arm-accuracy gradients | +0.1–0.4 | ~80 lines (packed head cache) | ≈0 | new heads/trunk | Body Transformer, SMP/MetaMorph (morphology work, not ceiling) |
| 6 | Multi-horizon auxiliary value heads (γ 0.9 / 0.965 / 0.995 targets, actor untouched) | falls: long-horizon fall anticipation without the γ 0.99 actor-target noise | +0–0.3; falls ↓ | ~60 lines trainer, 1 line policy | +1% | critic head reinit | Fedus 2019 (Rainbow aux horizons); PPG aux value |
| 7 | vf-coef ladder (0.5 → 0.25 / 0.1) and `--critic-detach` probe | critic gradient share steering the shared trunk | ±0.3 (mostly a diagnostic) | 0 | 0 | none | DNA, PPG §3.6, Andrychowicz; critic-design topic |
| 8 | HL-Gauss categorical critic | fall outliers (−5 terminal) dominate the MSE value gradient into the shared trunk | +0–0.3 | ~50 lines | ≈0 | critic head reinit | Farebrother 2024; DreamerV3 twohot; not tested in PPO |
| 9 | Autoregressive limb-group heads (spine → arms → legs) | independent 29-way factorization samples incoherent exploration noise | +0–0.5, could be negative | ~120 lines; sequential sampling in rollout | +2–4% | new heads | AlphaStar AR heads; Metz 2017; Tang & Agrawal 2020 |
| 10 | Chunked actions (decide at 15 Hz, 2 ticks of gears per forward) | none measured; temporally-extended exploration argument | −0.5 to +0.3 | ~150 lines rollout/GAE | −4% (fewer forwards) | new heads | ACT/Q-chunking are IL/offline-TD; frame-skip precedent only |
| 11 | Self-supervised next-state aux loss (predict saber/body lanes at t+1 from trunk + action) | rank utilization; sample efficiency | +0–0.3 early, ~0 at plateau | ~60 lines | +5–8% update | none | SPR; DreamerV3; weak in data-rich on-policy |
| 12 | Attention tweaks: `--n-heads 8`; learned time-to-plane attention bias | multi-note/wall resolution | ±0.2 | 0 / ~15 lines | ≈0 | heads: new arch | ALiBi-style bias; unsupported here |
| 13 | Previous-action input token (29 gear fractions of t−1 fed to the saber/body tokens) | policy cannot see its own last command except via q̇ | +0–0.2 | ~40 lines trainer/policy | ≈0 | encoder width change | common in locomotion obs (legged_gym); cheap substitute for memory |
| 14 | Recurrence (MinGRU/LSTM after the trunk, truncated BPTT over T=32) | nothing measured — obs is near-Markov with a 2 s window | ±0.3 | ~300 lines (sequence minibatches, state carry, resets) | +15–30% update, rollout state I/O | new arch | OpenAI Five / Dactyl LSTM; Memory Gym; Ni 2023 |
| 15 | Next-note-outcome prediction aux head | representation shaping | +0–0.3 | needs env label routing (out of scope) | — | — | unsupported |
| 16 | Small-init (×0.01) policy heads | early trajectory only | +0–0.2 at 200M, ~0 at 1.12B | 3 lines | 0 | from-scratch only | Andrychowicz 2020 |

Two things deliberately NOT ranked: a full separate critic transformer
(critic-design topic: ruled out at this scale, +13% wall for a baseline
whose ev is already 0.993) and bf16 activations (the whole fp16 machinery —
fp16_linear kernels, flat replicas, LT epilogues — is half-typed; a bf16
port is a week of kernel work for an r1 that has not reached the clip
window; the fp32 head GEMM in #4 is the 10-line insurance).

## One paragraph each

### 1. Hand-relative geometric lanes on the note tokens

Mechanism: today a note token knows its own world pose (x, y, z/8, vz/8),
cut direction (dir_cos, dir_sin) and type; a saber token knows its wrist
pose/velocity and blade direction. Everything the policy needs at the cut —
the wrist→note offset, time-to-plane, whether the current swing is aligned
with the required cut direction — has to be assembled by attention across
3 blocks at d=192 (softmax mixing is a poor way to subtract two vectors).
Compute those quantities explicitly per (hand, note) pair from lanes the
obs ALREADY carries and append them to the 18 note lanes before `note_mlp`:
Δp (3, de-normalized then /2), |Δp|, t_plane = z/|vz| (clipped), and two
alignment dots — wrist-velocity direction · cut direction and blade
direction · cut direction — per hand (7 × 2 = 14 lanes), plus 2 head-
relative wall lanes (wall_y − head_y, wall_x − head_x) for ducking; wall
rows get the note-only dots masked to 0 by (1 − is_wall). Why here: the
residual bench loss is the cut-plane-distance accuracy component on mid/top
rows of hard maps, i.e. precision of the wrist→note relation under time
pressure; the width step's gain grew with map difficulty, which is what you
expect when the relation is being approximated by capacity. Literature: the
closest quantitative evidence is Liu et al., "Frame Mining"
(https://arxiv.org/abs/2210.07442) — coordinate-frame choice for point-
cloud manipulation RL swings success rates by tens of points; hand/object-
relative features are standard in dexterous-hand PPO (DeXtreme
https://arxiv.org/abs/2210.13702, object pose relative to the hand). No
paper ablates it for a token humanoid policy — unsupported in our exact
form, but the mechanism is first-principles. Expected: the largest early
gain of anything here (the first 400 updates should move visibly, like the
width step's +1.5 pt at 201–250); plateau gain +0.1–0.4. Cost: the
elementwise ops on (B, 24, ·) are ~1% of the note_mlp GEMM; obs, env, kernel
untouched. Risk: none to throughput; checkpoint shape change on
`note_mlp.0.weight` only — zero-init the 16 new columns and the function
is bit-identical to the old model, so a WARM-START probe off
v40_w192/trunk_final.pt is possible (300 updates vs a plain continued-
training control, like the γ-ramp design). Test: standard from-scratch
probe; secondary: greedy per-row acc on the 5-map set (low/mid/top
0.694/0.625/0.637 for the muon trunk was the last reading) — the top-row
number is the one to move.

### 2. SimBa-style residual + LayerNorm trunk

Mechanism: the transformer blocks are pre-LN residual, but the trunk that
produces the 384-dim feature `h` on which srank is measured is a plain
`Linear(576,384) → ReLU → Linear(384,384)` — the one place in the network
with no normalization and no skip. SimBa (https://arxiv.org/abs/2410.09754)
and SimBaV2 (https://arxiv.org/abs/2502.15280) show that plain MLPs lose
monotone width scaling in RL while LN + residual blocks keep it; Creus
Castanyer et al. (https://arxiv.org/abs/2506.15544) report the same for
shared-trunk PPO on IsaacGym humanoids (multi-skip residuals + LN fix the
size-collapse mode); Kumar et al. (https://arxiv.org/abs/2010.14498)
tie rank collapse to bootstrapped losses through exactly such layers.
Proposal: `Linear(576,384)` projection → one residual block `x + W2·relu(W1·LN(x))`
→ final LN, with hidden 384 (param-matched, +147k) as the isolating probe
and hidden 1536 (ffn ×4, +1.0M) as a rung variant. The fused
`FusedLayerNorm.forward_add` already implements residual-add + LN in one
kernel on both streams, so the fp16 path needs no new kernel. Expected:
srank/psrank should rise from ~60% toward 75–85% of width and the long-
trunk decay (the thing that bit the 4.7B Adam trunks) should flatten; acc
+0.1–0.4 at 1.12B, more on long trunks. Risk: the final LN changes the
logit scale the heads see (head-norm growth was the candidate entropy-
decline mechanism in the norm census) — watch `ent` and clipfrac on the
first 50 updates; the diag `srank` line needs `h` to remain the stashed
tensor. Checkpoint: trunk weights reinit (or load the first Linear and
reinit the block). Test: standard probe; the readout that matters beyond
acc is srank at upd 451 and 1068 vs 232/384.

### 3. Lower-body gear heads over {coast, ±¼} + per-group entropy coefficients

Mechanism: the census says the ±1.0 gears are functionally dead on the 15
spine/leg joints (a full-torque trunk gear is a fall) yet the policy must
keep assigning them probability under the entropy bonus, and every rollout
tick samples 15 chances at a fall-inducing action. Policy-side fix with the
env unchanged: those 15 heads become 3-way over LOCAL options mapped to env
ids [0, 3, 4]; the 14 arm heads stay 5-way. Exact warm start: slice rows
[0, 3, 4] of each old 5-row head — the renormalized old policy conditioned
on "not full torque", which differs from the old policy by <1% mass. The
trainer stores local indices in `b_act` and maps to env ids at the step
call (one gather on a (29,) LUT per head). Entropy ceiling drops from
29·ln5 = 46.7 to 14·ln5 + 15·ln3 = 39.0 nats, so `ent` numbers stop being
comparable — log per-group entropy (already in `_ent_h`). The loss-only
variant to run FIRST because it is 10 lines: per-group entropy
coefficients (`--ent-group-mult spine=0.5,legL=0.5,legR=0.5`) using the
existing `_ent_h` (B, 29) stash — it stops the bonus from paying for mass
on the dead gears without changing head sizes. Expected: falls/M down
(the leg entropy floor 0.55–0.63 nats is where the rollout's fall
exploration lives), acc +0–0.3 through fewer terminations and cleaner
advantage estimates on hard maps. Unsupported by literature (no paper
prunes a factored discrete action space from a usage census); the closest
is the general finding that entropy regularization over irrelevant actions
hurts (e.g. per-head clip literature in `sessions/2026-09-12-per-head-
clip-literature.md`). Risk: knees actually use −1.0 ~17% of the time
(hard squats/wall ducks); keep knees at K=5 or map knee to {0, 2, 4}. Test:
standard probe; readouts falls/M, per-group entropy, leaf falls (0.8/M for
w192_cd300), duck hit rate on the 8 forced-duck maps.

### 4. Next rung: width/depth, with an fp32 head GEMM

Candidate shapes (exact counts from `policy.py`, gears 5):
embed 256 / trunk 512 / 3 blocks = 4,252,050 (3.33M actor / 0.92M critic);
embed 192 / 384 / 5 blocks = 3,304,658; 192 / 384 / ffn ×8 = 3,301,970;
224 / 448 / 3 = 3,268,978; 192 / 384 / 4 blocks = 2,859,794. Given the
rank readout (utilization proportional at both widths, so width is being
used, not wasted) and that the geometry computation lives in attention,
the two arms worth pod time are 256/512/3 (the SONIC/Hilton whole-net
pattern, lr 5.7e-4·√(192/256) = 4.9e-4) and 192/384/5 (depth, lr unchanged
— Muon's per-matrix RMS matching makes depth transfer safer than Adam's).
Skip ffn ×8 (SimBa/LLM evidence says ffn ratio is the least sensitive
axis). Expected +0.2–0.4 pt at 1.12B with the gain again concentrated on
hard maps; Hilton et al. (https://arxiv.org/abs/2301.13442) predict
diminishing returns per doubling at fixed compute. Cost: 0 code (flags
exist), +20–30% wall (the 128→192 step cost +15%; the model-bound share
scales sub-linearly with FLOPs on the 5090). fp16: r1 rose 1.8× with the
last width step and rises through the trunk; the head GEMM is the fp32
boundary and it is tiny ((16,384 × 384) × (384 × 145)), so add
`--fp32-heads` (cast `h` to fp32 and run `F.linear` against the fp32
masters; the rollout cache path does the same) and enable it for any run
at width ≥ 384 — no measurable throughput cost, removes the one term of
r1 that grows with sharper logits. bf16 is not worth it (see above).
Test: two standard probes; also log `steps/s` and peak VRAM (the 4.25M
model's update activations at mb 16,384 — mb-accum 2 exists as the
fallback).

### 5. Per-limb structured heads

Mechanism: the 29 heads share one 384-wide readout `h`. Arm heads want
hand↔note precision features; leg/spine heads want balance features; the
γ-noisy fall advantages and the dense cut advantages both flow into the
same 384 dims. Split the readout: arm_L sub-trunk on [saber_L, body] (384
→ 256), arm_R on [saber_R, body] (weights optionally SHARED with arm_L via
the mirror map — see `sessions/2026-09-07-symmetry-methods.md`; the
residual L/R asymmetry there was ~0.2 pp so sharing is a small
regularizer, not a win), lower-body sub-trunk on [body, saber_L, saber_R]
(576 → 384). Literature: Body Transformer (https://arxiv.org/abs/2408.06316)
shows embodiment-masked attention beating a vanilla transformer on
single-body PPO tasks; SMP (https://arxiv.org/abs/2007.04976) and
MetaMorph (https://arxiv.org/abs/2203.11931) are morphology-generalization
work and do not speak to a single body's ceiling. Expected +0.1–0.4 pt;
the more likely visible effect is the per-group gradient norms in `[diag]`
decoupling. Cost: the packed-head machinery (`_refresh_head_cache`,
`flatten_fp16` tail group, `fp16_packed_linear` over `list(self.heads)`)
assumes one (145, 384) GEMM; three sub-trunks mean three head GEMMs and
three tail groups — ~80 lines and the tests in test_policy/test_gears.
Throughput: three small GEMMs vs one, negligible. Test: standard probe.

### 6. Multi-horizon auxiliary value heads

Mechanism: the γ 0.99 arms failed because the ACTOR's advantage targets got
noisy, not because long-horizon features are useless — falls are exactly a
long-horizon event (a fall the critic saw coming 0.3 s out is credited to
the wrong action under γ 0.965). Give the critic extra output heads that
regress returns at γ ∈ {0.9, 0.995} (each with its own value-norm stats;
targets from the existing `out=` GAE scan run with λ=1 at those γ), sum
their losses into the critic loss at a small weight (0.1–0.25 each), and
leave the actor's GAE at γ 0.965 / λ 0.98 untouched. The shared trunk is
then shaped by long-horizon prediction while the policy gradient is not.
Literature: Fedus et al., hyperbolic discounting / multi-horizon aux
(https://arxiv.org/abs/1902.06865) — auxiliary horizons improved Rainbow
on Atari as representation shaping; PPG (https://arxiv.org/abs/2009.04416)
shows aux value phases help shared-trunk PPO. On-policy humanoid evidence:
none. Expected +0–0.3 acc; the honest primary readout is falls/M and
early-termination count on the bottom-3 bench maps. Cost: ~60 lines in
train_batch (two extra return scans, two norm EMAs, a (B, 3) value output),
`critic_mlp[-1]` → `Linear(384, 3)`. Risk: this is the "critic steers the
trunk" channel by design — if #7 shows detaching helps, run this with the
aux heads on the detached path only.

### 7. vf-coef ladder and `--critic-detach` probe

Zero code. `--vf-coef 0.5` is the recipe; the critic-design topic notes
working shared-trunk transformer PPO implementations run 0.1, and the γ
0.99 arms are direct evidence that the critic's gradient share can hurt
the actor through the trunk. Run `--vf-coef 0.25` (one probe) and
`--critic-detach` (one probe: value loss stops at the trunk boundary, the
critic trains on detached features — PPG §3.6's approximation of a dual
network at 1× params). Interpretation matrix: detach ≈ control → the trunk
is actor-owned and #6/#8 must be evaluated with detach on; detach worse →
TD gradients are useful representation pressure here (Lyle et al.
https://arxiv.org/abs/2206.02126) and #6 is the way to get more of it;
vf 0.25 better → adopt, and the shared-trunk question is answered without
a separate net. Literature: DNA (https://arxiv.org/abs/2206.10027, 12.6×
gradient-noise-scale gap between policy and value losses), Andrychowicz
et al. (https://arxiv.org/abs/2006.05990). Expected ±0.3 (noise-level);
its value is that it gates #5, #6, #8 for two probes' worth of pod time.

### 8. HL-Gauss categorical critic

Mechanism: the value target distribution has a heavy negative tail (−5
terminal falls, −1.5 walls/bombs on a ~+0.18/tick stream); MSE puts the
largest gradients on exactly those outliers, and they flow into the shared
trunk. Cross-entropy over a histogram (HL-Gauss: Gaussian-smoothed two-hot
over ~101 bins spanning the normalized return range ±6σ) bounds the
per-sample gradient and, per Farebrother et al.
(https://arxiv.org/abs/2403.03950), scales better with network size in
value-based RL; DreamerV3 (https://arxiv.org/abs/2301.04104) uses twohot +
symlog for its actor-critic critic. Not tested in PPO by either paper —
unsupported in our setting. The value used for GAE is the expectation over
bins (fp32), so the trainer's value-norm/GAE code is unchanged past that
one line. Expected +0–0.3, ev already 0.993 so do not expect it to move;
the gain if any is in acc via a calmer trunk. Cost ~50 lines
(`critic_mlp[-1]` → `Linear(384, 101)`, target smoothing, CE loss, value
expectation). Risk: bin range must follow the value-norm EMA — clamp
targets and log the clamp rate.

### 9. Autoregressive limb-group heads

Mechanism: the joint policy is a product of 29 independent categoricals;
the exploration noise sampled each tick is 29 independent draws, mostly
incoherent on the body, and the ratio for PPO is the product of 29 ratios
(the per-head clip work found the joint ratio to be the right object). An
autoregressive order over 5 groups — spine → arm_L → arm_R → leg_L → leg_R
— feeds the SAMPLED gears of earlier groups (embedded, 29 × 5 one-hot →
64) into later groups' heads: coherent whole-limb exploration and an
explicit spine→arm dependency. Update-time forward is one pass (teacher
forcing on the stored actions); rollout sampling becomes 5 dependent tiny
GEMMs — +2–4% wall in a latency-bound forward. Literature: AlphaStar's AR
action heads (https://www.nature.com/articles/s41586-019-1724-z), Metz et
al. sequential discrete actions (https://arxiv.org/abs/1705.05035), Tang &
Agrawal discrete factorization for on-policy control
(https://arxiv.org/abs/1901.10500 — the precedent for our current design,
which they found competitive WITHOUT AR on MuJoCo). Expected +0–0.5 but
could be negative (AR heads sharpen faster; the entropy schedule was tuned
for the independent product). Cost ~120 lines and a new head layout. Rank
below the cheaper structural change (#5), which it can be layered on.

### 10. Chunked actions (decide at 15 Hz)

Mechanism: predict gears for ticks t and t+1 from obs_t (58 heads), act on
both, value/GAE on the 2-tick macro-step (reward summed, γ²). Halves policy
forwards (−4% wall), doubles the correlation time of exploration noise (the
ACT / Q-chunking argument: https://arxiv.org/abs/2304.13705,
https://arxiv.org/abs/2507.07969 — both imitation or offline-to-online TD,
not on-policy from scratch; the only on-policy precedent is Atari frame-
skip, which is an env hyperparameter). Against: reaction latency 33 → 67 ms
on a task where one tick is 0.63 m of note travel at 19 m/s and where
balance recovery is tick-level; the policy already commits swings over
~10 ticks, so the exploration-coherence gain is speculative. Expected −0.5
to +0.3. Cost ~150 lines across rollout storage, GAE indexing and the
env-step loop. Rank low; only worth it if #9 shows coherence matters.

### 11. Self-supervised next-state auxiliary loss

Predict the 32 saber lanes × 2 and 78 body lanes at t+1 from [h, embedded
action_t] (b_obs[t+1] is in the buffer; mask across episode boundaries).
SPR (https://arxiv.org/abs/2007.05929) and DreamerV3 show representation
gains in data-limited regimes; at 200k samples/s with 1.12B per probe, data
is not the constraint and aux losses add trunk gradient traffic of the kind
#7 is testing. Expected +0–0.3 early, ~0 at plateau; +5–8% update time.
Rank low; revisit only if srank stays at 60% after #2.

### 12. Attention tweaks

`--n-heads 8` at embed 192 (dh 24) is a zero-code arm: more, narrower
heads resolve more simultaneous entities (two notes + a wall). A learned
per-head additive bias on attention scores of −β·t_plane (ALiBi-style,
https://arxiv.org/abs/2108.12409) would let the sabers attend by imminence
without spending block capacity on it — ~15 lines in `Block.forward`
(the `attn_bias` tensor is already built once per forward). Both
unsupported here and within seed noise in expectation (±0.2). Adding slots
(24 → 32) is an env/obs change and out of scope; the eviction study made
24 the knee.

### 13. Previous-action input

The policy sees q̇ but not its own last gear command; feeding action_{t−1}
(29 fractions) into the saber/body token encoders is the cheap version of
memory and is standard in locomotion obs vectors (legged_gym). ~40 lines
(trainer carries `b_act[t−1]`, zeros at reset, passes it alongside obs).
Expected +0–0.2; unsupported in our regime (q̇ at 30 Hz already encodes
most of it).

### 14. Recurrence

What it could buy: entities evicted past 24 slots (rare after the window
study), wall duration state, integration over the policy's own sampling
noise. What the obs already gives: full body state with velocities and a
~2 s entity window, so the task is close to Markov at the 1 s causal
horizon the γ experiments confirmed. PPO cost: minibatches become
(envs × T) sequences (16,384 rows = 512 envs × 32 ticks — 64× fewer
independent episodes per minibatch, a real variance cost at our
clip-bound updates), hidden-state carry per env with reset masking, stale
initial states across updates (R2D2 burn-in), the update's recurrent
segment runs sequentially over 32 ticks (MinGRU's parallel scan,
https://arxiv.org/abs/2410.01201, removes the sequential dependence but
not the sequence-minibatch structure), and the adv-filter (top-50% |A|
rows) becomes incompatible with sequence minibatches. Literature:
OpenAI Five (https://arxiv.org/abs/1912.06680) and Dactyl
(https://arxiv.org/abs/1808.00177) used LSTMs under partial observability;
Memory Gym (https://arxiv.org/abs/2309.17207) and Ni et al.
(https://arxiv.org/abs/2307.03864) find memory architectures matter only
when the task has memory demands. Expected ±0.3; cost ~300 lines and
+15–30% update time. Lowest EV per pod-hour on this list.

### 15–16. Next-note-outcome aux head; small-init heads

The outcome-prediction head needs per-slot labels (which note in which
slot got which cut) that the obs does not carry — an env telemetry change,
out of scope. Small-init heads (Andrychowicz item 7) are a 3-line
from-scratch tweak that only affects the early trajectory; bundle it with
whichever from-scratch probe runs next rather than spending a probe on it.

## Do first

Order: #1 → #2 → #3 (loss-side variant first), with the #7 vf-coef arm
slotted wherever a pod would otherwise idle (zero code). Each is one
standard probe; #1 also has a 300-update warm-start variant off
`runs/v40_w192/trunk_final.pt` for a fast signal. Re-bench
`runs/v40_w192_cd300/leaf_from_upd01068.pt` on the current pod BEFORE the
first leaf bench so the reference is same-host.

### 1. `--note-geo` — hand-relative lanes (policy.py)

```python
# policy.py — module constants
NOTE_GEO_FEAT = 16          # 7 per hand (dp3, |dp|, t_plane, v·d, blade·d) + 2 head/wall
# saber_feat lanes (SABER_FEAT=32 order: q8, qd7, derived11, elbow3, elbow_vel3)
_S_WPOS, _S_WVEL, _S_BLADE = slice(15, 18), slice(18, 21), slice(21, 24)
# note lanes: x, y, z/8, vz/8, is_left, is_right, dir_cos, dir_sin, ..., valid=13, is_wall=14

def _note_geo(notes, sabers, body, dt):
    """(B,24,18), (B,2,32), (B,78) -> (B,24,16). Same normalizers as the obs
    (x/1, (y-1.4)/0.75, z/8 vs wrist z/1) so the subtraction is done in
    metres and re-normalized."""
    npos = torch.stack([notes[..., 0], notes[..., 1] * 0.75, notes[..., 2] * 8.0], -1)  # (B,24,3) m
    cutd = torch.stack([notes[..., 6], notes[..., 7], torch.zeros_like(notes[..., 6])], -1)
    not_wall = 1.0 - notes[..., 14:15]
    feats = []
    for hnd in range(2):
        w = sabers[:, hnd]                                   # (B,32)
        wpos = torch.stack([w[:, 15], w[:, 16] * 0.75, w[:, 17]], -1)      # m
        dp = (npos - wpos[:, None]) * 0.5                    # (B,24,3), /2 m
        dist = dp.norm(dim=-1, keepdim=True)
        tpl = (notes[..., 2:3] / notes[..., 3:4].abs().clamp_min(1e-3)).clamp(-3, 12) / 4.0
        v = w[:, 18:21][:, None]; vhat = v / (v.norm(dim=-1, keepdim=True) + 0.05)
        vdot = (vhat * cutd).sum(-1, keepdim=True) * not_wall
        bdot = (w[:, 21:24][:, None] * cutd).sum(-1, keepdim=True) * not_wall
        feats += [dp, dist, tpl, vdot, bdot]
    head = body[:, _HEAD_IN_BODY:_HEAD_IN_BODY + 3]          # head x,y,z lanes in _BODY_IDX order
    feats += [(notes[..., 1:2] - head[:, None, 1:2]) * (1 - not_wall),
              (notes[..., 0:1] - head[:, None, 0:1]) * (1 - not_wall)]
    return torch.cat(feats, -1).to(dt)

# NotePolicy.__init__: note_geo flag in self.arch; encoder input widens
self.note_mlp = _mlp([NOTE_FEAT + (NOTE_GEO_FEAT if note_geo else 0), embed, embed])
# forward_stacked, before `emb = _seq(self.note_mlp, notes)`:
if self.note_geo:
    notes = torch.cat([notes, _note_geo(notes.float(), sabers.reshape(B, 2, SABER_FEAT),
                                        body, dt)], -1)
```

Warm start: in `load_policy`, if the checkpoint's `note_mlp.0.weight` is
(192, 18) and the model's is (192, 34), copy into the first 18 columns and
leave the rest zero — the function is then bit-identical to the old model,
so `--init-from` off v40_w192 is a legitimate warm-start probe. Trainer:
`--note-geo` flag → `NotePolicy(note_geo=True)`, arch marker stamped, an
`assert` that `_BODY_IDX` still places the head lanes where `_note_geo`
reads them (pin it in tests/test_policy.py against constants.py the way
`HEAD_GROUPS` is pinned). Head lanes: obs offset 97..100 (the 598 layout;
`_BODY_IDX` = 18 root/spine-q + 15 leg-q/spine-q̇ + 12 leg-q̇ + [97..102)
…), so `_HEAD_IN_BODY = _BODY_IDX.index(97)` = 45 — derive it, never
hard-code it. Also de-normalize head y the same way as the notes
((y − 1.70)/0.75 for the head vs (y − 1.4)/0.75 for notes: Δy_m =
0.75·(ny − hy) − 0.30).

### 2. `--trunk-arch simba` — residual + LN trunk (policy.py)

```python
class ResBlock(nn.Module):           # SimBa block: x + W2 relu(W1 LN(x))
    def __init__(self, d, hidden):
        super().__init__()
        self.ln = FusedLayerNorm(d)
        self.ffn = _mlp([d, hidden, d])
    def forward(self, x, delta=None):
        # delta: previous un-added residual (fused add+LN, same trick as Block)
        x, h = (self.ln.forward_add(x, delta) if delta is not None else (x, self.ln(x)))
        return x, _seq(self.ffn, h)

# __init__ (trunk_arch in self.arch; "mlp" = today's layout, default)
if trunk_arch == "simba":
    self.trunk_in = nn.Linear(3 * embed, trunk_dim)
    self.trunk_blocks = nn.ModuleList(ResBlock(trunk_dim, trunk_hidden) for _ in range(trunk_depth))
    self.trunk_ln = FusedLayerNorm(trunk_dim)
# forward_stacked, replacing `h = _seq(self.trunk, ...)`:
x = _lin(self.trunk_in, tokens.reshape(B, -1)); delta = None
for rb in self.trunk_blocks:
    x, delta = rb(x, delta)
h = self.trunk_ln(x + delta)      # the stashed diag tensor stays `h`
```

Two arms: `trunk_hidden=384, depth=1` (param-matched, +147k) and
`trunk_hidden=1536, depth=1` (+1.0M). `flatten_fp16` picks the new
Linears up via `modules()`; the heads tail group is unchanged. The critic
reads `h` and `x_res` as before. Readouts: srank/psrank at upd 451 and
1068 vs 232/384 and the entropy trajectory over the first 50 updates
(final-LN logit-scale check).

### 3. Dead-gear heads — loss-side first, then head sizes

Loss-side (train_batch.py, ~10 lines, run first):

```python
p.add_argument("--ent-group-mult", default="",   # e.g. "spine=0.5,legL=0.5,legR=0.5"
               help="per-HEAD_GROUPS multiplier on the entropy bonus")
# after parsing: ent_w = torch.ones(29); for nm,a,b in HEAD_GROUPS: ent_w[a:b] *= mult[nm]
# in the minibatch loop, replace `ent_loss = entropy.mean()` with
ent_loss = (policy._ent_h * ent_w).sum(-1).mean()     # _ent_h is (B,29), set by need_entropy
```

(`_ent_h` is detached today for the diag stash — for this path it must be
the live tensor; keep the detached copy for the diag and return the live
per-head entropy from `get_action_and_value` when a `head_ent=True` flag
is set, mirroring `head_logp`.)

Head-size change (policy.py, `--gears-lower 3`):

```python
LOWER_IDS = (0, 3, 4)     # coast, +1/4, -1/4  (knees: consider (0, 2, 4) — they use -1.0 17%)
self.head_sizes = tuple(n_gears if 3 <= h < 17 else n_lower for h in range(29))
self.register_buffer("_act_map", torch.tensor(<per-head local->env id table, (29, 5) padded>))
# forward_stacked: the packed head GEMM output is (B, sum(head_sizes)); split ragged
# instead of view(B, 29, K) -> pad to (B, 29, 5) with -inf on the missing local
# columns so log_softmax / gather / entropy stay one kernel each (mask bias
# reuse: MASK_BIAS). Sampled local ids -> env ids via _act_map before the env
# step; b_act stores LOCAL ids (the gather index). Warm start: slice rows
# [0,3,4] of each 5-row head weight/bias in load_policy when the ckpt is K=5.
```

The padded-(B, 29, 5) trick keeps `get_action_and_value`, the per-head
entropy diag and `--head-clip` unchanged; the only trainer touch is the
`_act_map` gather at the env-step call and the `--gears` consistency check
(`train_batch.py` ~1068). Readouts: falls/M (trunk 5.5, leaf 0.8 for
w192), per-group entropy, leaf duck hit rate on the 8 forced-duck maps.

### Parallel zero-code arms (fill idle pod slots)

`--vf-coef 0.25`; `--critic-detach`; `--n-heads 8`; and the next rung
`--embed 256 --trunk-dim 512 --lr 4.9e-4` — the last one only after
`--fp32-heads` (10 lines: `h.float()` + `F.linear` against the fp32
masters in both the rollout cache path and `fp16_packed_linear` site)
exists, and with adv-filter 0.25 if the af25-at-recipe-γ probe confirms it
is free.
