# PufferLib 5.0 — algorithmic ideas for train_batch.py (2026-09-14)

Scope: the PPO recipe inside PufferLib 5.0, read from source, compared
against our trainer. The C/CUDA runtime was evaluated and declined on
2026-09-10 (`sessions/2026-09-10-pufferlib-5-evaluation.md`); this note is
only about what the *learner* does differently and whether any of it should
be probed here. Fixed constraints honoured: no reward-timing changes, no
posture/effort costs, T=32 floor (T=64 allowed), target = accuracy on real
maps.

## What was read

- PufferLib `5.0` branch at `adbe5cbb` (2026-09-10, "Trailer"; scratchpad
  checkout). Head today is `6ffa5b10` (2026-09-13, 17 commits later); the
  `src/` diff since our checkout is 14 lines in `pufferl.cu` (log-reduce
  agent count, selfplay bot-ladder env count) — **nothing algorithmic
  moved**. Files: `src/pufferl.cu` (`train_epoch_gpu`, `train_impl`,
  `run_train` async loop, `sample_logits`, `cosine_annealing`),
  `src/algo.cu` (`ppo_loss_compute`, `puff_advantage`, `cache_imp_and_v`,
  `muon_step`), `config/default.ini` + all 86 env configs (Protein-tuned
  values), and the 4.0 torch trainer (`pufferlib/torch_pufferl.py` on
  `origin/4.0`) for lineage.
- "5.0 Experiments" GitHub release (2026-09-13): pretrained models and
  Protein experiment logs only; **no algorithmic release notes exist**.
  puffer.ai/docs.html (5.0) says only "PPO-variant with improvements based
  on our own research … synchronous and 1-epoch asynchronous training as in
  CleanBa". The "Puffing up PPO" X thread (3.0, June 2025) is paywalled to
  fetch; its claims are quoted from search snippets and cross-checked
  against the 4.0 code.
- Ours: `python/train_batch.py` (GAE at 1606–1614, value-norm 1634–1647,
  filter 1656–1668, minibatch loop 1771–1980, `pg_loss_fn` 90,
  `adv_filter_*` 125–161, `Muon` 678–760), README, the 09-12/09-13 probe
  notes.

## PufferLib 5.0's learner, as it actually is

| item | PufferLib 5.0 (code path) | ours |
|---|---|---|
| batch shape | 4,096 agents × 64 steps = 262k (default); mb 8,192 = 128 agent-segments × 64 | 32,768 × 32 = 1.05M; mb 16,384 flat shuffled samples |
| epochs | `replay_ratio` × batch / mb minibatches, walked **sequentially over contiguous agent segments, no shuffle** (`train_epoch_gpu`: `dest_off = (mb*mb_segs) % n_rows`); fractional allowed (moba 0.83 = trains on 83% of rows; drone 3.98 = 4 passes) | 1 epoch, device shuffle, top-50% \|A\| filter → 32 steps |
| advantage | `puff_advantage` = GAE with optional V-trace ρ̄/c̄ clips on δ and on the λ-product; **recomputed per minibatch on the LIVE critic** (`cache_imp_and_v` writes `mb_gae_v`, then `puff_advantage(values=mb_gae_v…)`); returns = V_live + A; last step of each segment gets A = 0 (no extra bootstrap forward) | GAE once per update on the rollout critic, extra forward for V_T |
| V-trace | `vtrace = 0` default → ρ = c = 1 → plain GAE. **0 of 86 shipped configs set vtrace = 1** — the "puffer advantage" correction is dormant; replay_ratio > 1 runs uncorrected | n/a (1 epoch) |
| adv normalization | **none** (`ppo_loss_compute`: `adv_for_pg = g.advantages[nt]` raw). 4.0 normalized per minibatch × prio weight; 5.0 dropped it | per-minibatch mean/sd (full-batch stats under the filter, 09-02 audit) |
| reward | hard clamp to [−1, 1] in `train_epoch_gpu` (`clamp_precision_kernel`) | raw score/115, −1.5 bomb, −5 fall |
| value | raw return units, **clipped** loss 0.5·max(unclipped, clipped) around the frozen rollout V, `vf_clip_coef` 0.2 default (Protein picks 0.01–5), `vf_coef` 2.0 | normalized target (EMA μ/σ, β 0.01), unclipped, vf_coef 0.5, critic lr ×3 |
| policy loss | standard PPO clip, `clip_coef` 0.2 default; Protein-picked values across configs span 0.01–0.92, 22 of 50 above 0.2 (drone 0.32, moba 0.57, breakout 0.81); no dual-clip, no KL stop, no per-head anything | ε 0.2 joint (constant, no flag), dual-clip c = 3 |
| entropy | `ent_coef` 0.001 (single head sum), optional cosine anneal (`anneal_ent_coef`, on in 3 configs) | 0.03 → exp-decay to floor (`--ent-tau`), leaf `--ent-final` |
| lr | cosine to `min_lr_ratio`·lr (default → 0) over the whole run | WSD: warmup → constant trunk → 1-sqrt cooldown leaf |
| optimizer | Muon on **every ≥2-D param** (encoder, decoder/heads, value column too), 1-D params get the same Nesterov-momentum step un-orthogonalized, global grad clip 1.5 **before** momentum, μ 0.95 (Protein picks 0.5–0.997, median ≈ 0.93), lr 0.015, wd 0, HeavyBall per-step NS coefficients, aspect scale √max(1, R/C); no Adam, no critic lr split | hybrid: Muon hidden matrices, Adam on encoder inputs / 29 heads / critic final / 1-D; RMS-matched lr 5.7e-4, critic ×3, clip 0.5 |
| γ / λ | 0.995 / 0.90 default; Protein picks λ 0.2–0.995 (median 0.92), γ mostly 0.99–0.9996 | 0.965 / 0.98 (2×2 at T=64 settled: γ 0.99 −5 pt, λ 1.0 −0.7 pt) |
| recurrence | MinGRU stack with highway; state carried across horizons (`reset_every_horizon = 0`), per-slot initial states snapshotted at t = 0 and restored for training; whole-horizon segments as minibatch rows | none (transformer over entity slots) |
| async | `async = 1` default (83 of 86 configs): 2 rollout slots; slot k+1 is collected with a **one-update-stale actor snapshot** on stream 0 while the learner trains slot k on its own stream, same GPU; no importance correction unless `vtrace = 1` | synchronous |
| multi-GPU | NCCL all-reduce of the flat grad per minibatch (data parallel), schedules per rank | single GPU |
| exploration floor | `verb_eps`: ε-mixture with uniform-over-legal on NetHack's verb head, applied in sampling AND in the training log-prob (a proper mixture policy); off elsewhere | none |
| precision | bf16 activations + obs, fp32 master weights, no loss scaler | fp16 activations + dynamic loss scale, fp32 obs → fp16 replica |

New in 5.0 versus 4.0's torch trainer: async Cleanba slots, MinGRU state
carry, selfplay pool + bot ladder, NCCL DP, bf16, entropy anneal option,
V-trace behind a flag — and two **removals**: advantage-weighted segment
prioritization (4.0 `torch_pufferl.py:291–296`: multinomial over
Σ|A|^α per segment, importance weight (N·p)^−β with β annealed) and
per-minibatch advantage normalization. The 3.0 blog's "prioritized replay
matters a lot for mazes" did not survive into the CUDA trainer.

## 1. Ranked candidates

Detection rule for every probe: from-scratch wide recipe (embed 192 / trunk
384, Muon, lr 5.7e-4, 32k × 32, af 0.5), 1,068 updates = 1.12B samples,
same pod as a fresh control (pod-to-pod is ~1 pt; the seed-1 replication
gap is 0.3 pt). Read the last-100 training acc + `return`, KL(mb1/mbN),
clipfrac_last, ent, ev, falls/M, pn actor/critic. **acc gaps < 0.3 pt are
seed noise**; a cd300 leaf + bench (σ_cut ≈ 25k upp, MDD ≈ 98k) only for a
winner.

| # | change | mechanism (PufferLib path) | expected effect here, and why | cost in train_batch.py | one-probe design |
|---|---|---|---|---|---|
| 1 | **Joint clip ε ladder: 0.3, then 0.4** | `ppo_loss_compute` clips on `clip_coef`; default 0.2 but Protein lands 22/50 configs above 0.2 under Muon with 1–4 replay passes — the only external Muon+PPO evidence that a wider trust region is often the optimum | Our progress per update is measured clip-bound (clipfrac_last ≈ 0.6, KL/upd 0.08–0.10; the 2M-batch probe moved the policy far less than 2× per update). A wider ε lets each of the 32 steps keep gradient on more rows. The per-head ε 0.2 collapse (v40_headclip) was a *looser* per-head constraint, not this; joint ε 0.3 is one step wider on the joint ratio and dual-clip c = 3 stays as the floor guard. Honest prior: +0.3–0.8 pt at matched samples if the clip-bound reading is right, P ≈ 0.45; downside is faster entropy decline (watch ent < 15 before upd 600) | ~5 lines: `--clip` flag feeding `pg_loss_fn(clip=)`, `pg_loss_heads`, and the clipfrac diagnostic (`CLIP` constant at line 54 is used in 4 places); checkpoint `schedule` should record it | From scratch, `--clip 0.3` vs same-pod control. Read: acc/return last-100, KL(mbN) (expect 0.12–0.15), clipfrac_last (expect < 0.45), ent trajectory, falls, pn. Go to 0.4 only if 0.3 ≥ control + 0.3 pt with ent ≥ control − 2 nats; stop the ladder at the first rung that loses |
| 2 | **Stale-actor emulation → decide 2-GPU async** | `run_train` (pufferl.cu 3077–3107): rollout slot k+1 with `actor_weights` (one update old) while the learner trains slot k; `pufferl_forward_step` picks `actor_weights` when `async`. On one GPU it needs co-resident kernels (measured impossible for our env kernel, 09-10); on **two GPUs** it is a pure pipeline: GPU-A env + forward (79% of our wall), GPU-B update (21%), ceiling 1/0.79 = **1.27× wall**, batch transfer ~0.75 GB/update (fp16 obs 340 lanes) ≈ 30 ms over PCIe — negligible | The cost is algorithmic and unmeasured: rollout k+1 is collected under θ_k while the update starts from θ_{k+1}; with our KL/upd ≈ 0.1 the first minibatch already sits where the previous update's clipfrac_last (0.6) left it — most rows outside ±ε before any step. Cleanba tolerates 1-step staleness because its per-update KL is 5–10× smaller. Prior: −0.3 to −1.5 pt at matched samples, P(< 0.3 pt) ≈ 0.3. If cheap, the second GPU also gives a free bench/leaf lane 73% idle. If expensive, the fix is IMPALA-shaped: recompute "old" log-probs under θ_{k+1} at update start (one value+logp forward over the batch, ~5% wall) and correct the advantage with ρ̄-clipped V-trace = exactly `puff_advantage` with `importance = π_{θ_{k+1}}/π_{θ_k}` — the one place their kernel becomes relevant to us | Emulation: ~30 lines — keep a rollout copy of the policy (`copy.deepcopy` + `load_state_dict` once per update, before `opt.step` loop, from the *previous* update's weights), use it in the rollout forward; fp16 replica cache (`bump_weights_version`) must key on the copy. No throughput change, no second GPU needed. The DP alternative (NCCL, 2 × 32k envs = 2M batch) is already priced: batch doubling costs 1.67× samples for equal accuracy (v40_t64b) — ≈ break-even wall, so **async, not DP**, is the 2-GPU design | From scratch, `--stale-actor 1` vs control. Read: acc gap at 300M / 700M / 1.12B, KL(mb1) (expect ≈ previous KL(mbN) ≈ 0.1 instead of ~0), clipfrac_first (expect 0.4–0.6 instead of 0). Decision: gap < 0.3 pt → rent a 2-GPU pod and build the pipeline; 0.3–1 pt → build the recompute-old-logp + V-trace fix and re-probe; > 1 pt → close |
| 3 | **λ ladder downward: 0.95, 0.90** | `default.ini` λ 0.90 with γ 0.995; Protein's median λ ≈ 0.92, many envs 0.4–0.85 — short traces are what their sweeps keep choosing | Our trace-only arm (λ 1.0) lost 0.7 pt and sharpened less; the derivative at 0.98 points *down*, and we never tested below (0.98 came from the 07-27 sweep, pre-value-norm). Mechanism from the 09-13 horizon note: the trace imports other notes'/hand's randomness; the critic carries the long-range channel. Prior: +0.2–0.5 pt at 0.95, wash at 0.90, P(≥ +0.3) ≈ 0.4. Watch ev (mechanically rises with shorter trace) and falls (fall credit lands on the critic's forecast, not the trace) | 0 — `--gae-lambda` exists | From scratch `--gae-lambda 0.95` vs control; 0.90 only if 0.95 ≥ control. Read acc/return, ev, falls/M, ent |
| 4 | **Two passes over the top-25%** (`--epochs 2 --adv-filter 0.25`) | `replay_ratio` in `train_epoch_gpu` re-walks the same rows (drone 3.98, craftax 1.17, nmmo3 1.43) with no IS correction — their only data-reuse mechanism, guarded by the PPO clip alone | Same 32 optimizer steps as the recipe, spent twice on the most informative rows. The af ladder says af25 ≈ af50 at half the steps (γ 0.99 trunk; untested at recipe γ); the question is whether the second pass still carries gradient after 16 steps (clipfrac_last ≈ 0.6 says half the rows are already outside ε). Prior: wash to +0.3, P(≥ +0.3) ≈ 0.25; also a free measurement of how clip-bound we are. Check the run archive for any `--epochs 2` run first (none recorded in the notes) | 0 — both flags exist; the shuffle re-permutes `keep_idx` per epoch already | From scratch `--epochs 2 --adv-filter 0.25` vs control (same wall). Read acc, KL(mbN) of epoch 2, clipfrac_last; pair with an `--adv-filter 0.25` single-pass arm (the 18%-wall saving, untested at recipe γ) |
| 5 | **Muon momentum 0.9 rung** | `muon_clip_nesterov` μ = 0.95 default; Protein lands μ below 0.95 in 25 of 44 swept configs (median ≈ 0.93, drone 0.88, moba 0.84, nmmo3 0.86) | Their regime has hundreds of small updates per horizon; ours has 32 steps per update with a fresh batch, so momentum spans ~1.5 updates at 0.95 and ~0.3 at 0.9. The 09-11 note argued *higher* β under noisy gradients; the sparse −5 fall term is our noise source. Prior: wash, P(≥ +0.3) ≈ 0.15 — included because it is free and closes the question | 0 — `--muon-momentum` exists | From scratch `--muon-momentum 0.9`; read acc, KL, gnorm, pn (norm walk should slow), falls |
| 6 | **Mid-update advantage recompute** (live-critic GAE, cheap variant) | `train_epoch_gpu`: GAE per minibatch on `mb_gae_v` (live V) — the one genuinely different estimator in their trainer | Their per-minibatch form needs contiguous env-segment minibatches (mb = 512 envs × 32), which kills the flat shuffle and the sample-level \|A\| filter — too structural. The cheap version: after step 16 of 32, re-run the value forward over the batch (fp16, no grad, ~4–5% wall), redo GAE + normalization stats + keep_idx for the second half. Prior at 1 epoch: wash (Andrychowicz 2020 finds per-epoch recompute a small gain only with ≥ 3 epochs), P(≥ +0.3) ≈ 0.15; it becomes relevant if #4 or a 2-GPU pipeline makes the critic move more within an update | ~40 lines (a `recompute_adv()` closure around lines 1592–1668 called at `s == train_n // 2`); the value-norm EMA must NOT be re-stepped mid-update | From scratch `--adv-recompute 1` vs control; read acc, ev, v_loss first/last mb |
| 7 | **Full-Muon split** (orthogonalize heads, encoder inputs, critic final; 1-D on Nesterov SGD) | `muon_step`: every `ndim ≥ 2` param through NS, 1-D params keep the momentum step at the same lr | Against the Moonshot/Jordan rule we followed; the only mechanism in its favour is the census's head-norm growth (×5.2, entropy-decline candidate): a fixed-spectral-norm head step makes head growth linear and lr-bounded instead of Adam-scale-free. Their 1-D-on-SGD part is a footgun (LN gains at 1.5e-2 raw lr). Prior: wash to −0.5, P(≥ +0.3) ≈ 0.15 | ~10 lines in `muon_param_names` behind `--muon-all` (keep 1-D on Adam) | From scratch; read acc, head-group pn, ent slope, max attention logit |
| 8 | **EMA-scale advantage normalization** (their "no normalization", made safe) | `ppo_loss_compute` uses raw A | Under Muon the hidden-matrix step is scale-free, so per-batch renormalization only sets the PG : entropy : value mix — and it *hides* advantage shrinkage as the policy converges (the PG term is re-inflated every update while the entropy bonus is constant). Normalizing by a slow EMA of sd(A) (β 0.01, like the value norm) keeps the early scale and lets the PG term decay naturally late, which is what their tiny `ent_coef` + raw A does implicitly. Prior: wash on the trunk, possible small leaf gain (sharpening), P ≈ 0.2 | ~10 lines: `adv_sd_ema` tensor next to `ret_var`, used in place of `a_sd`/`af_sd`; `ent_coef` unchanged | From scratch; read acc, ent trajectory vs control at equal ent schedule, leaf acc |

Not ranked, already ours: Muon (adopted 09-12), Protein (vendored),
top-|A| filtering (their 4.0 prio replay was the segment-level cousin and
was removed in 5.0), entropy anneal (ours is exp-to-floor + leaf target).

## 2. Do NOT copy

- **Reward clamp to [−1, 1]** (`train_epoch_gpu`): would flatten the −1.5
  bomb and −5 fall penalties and the 1.0 full-note cut into one bin. Their
  envs emit pre-scaled rewards; ours is the score.
- **Value clipping** (`vf_clip_coef`): Engstrom et al. 2020 and
  Andrychowicz et al. 2020 find it neutral-to-harmful; it exists in
  PufferLib because they have no value normalization. We normalize
  (T1, 08-19) and have had no value blow-up since; a clip in *normalized*
  units at critic lr ×3 would just slow the critic.
- **Raw (unnormalized) advantages + `vf_coef` 2.0 + `ent_coef` 0.001**:
  a coupled retune of three coefficients with nothing to gain under a
  scale-free optimizer (see #8 for the one safe piece).
- **Cosine lr to zero over the run**: needs the run length in advance and
  forbids extension; WSD trunk/leaf is what lets us fork, extend, and cut
  cd200/cd300 leaves. Same for cosine entropy anneal.
- **γ 0.995 / λ 0.90 defaults and Protein's high-γ preference**: our
  γ/λ 2×2 at T=64 is the direct test (γ 0.99 −5 pt, still 5 pt behind
  after 2.24B). The causal window is ~1 s; only λ *down* (#3) is open.
- **Contiguous, unshuffled agent-segment minibatches** and the
  whole-horizon rows: required by their MinGRU carry state; for a
  memoryless transformer it only correlates each minibatch and breaks the
  sample-level filter.
- **Last-step advantage = 0 / no V_T forward** (`puff_advantage` "Last
  index H-1 left 0"): drops 1/T of the batch's policy gradient and uses
  V[T−1] as the bootstrap; our extra bootstrap forward is correct and
  costs < 0.3% wall.
- **V-trace as shipped**: dormant (0/86 configs). Only relevant to us as
  the correction inside the async fix in #2, never as a default.
- **Muon on 1-D params at raw lr** and **no critic lr split**: LN gains
  and biases on Nesterov SGD at 1.5e-2 is their small-net setting; the
  census showed our LN gains already drift 1.0 → 0.32 under Adam.
- **`verb_eps` exploration floor**: an ε-mixture over legal actions would
  keep sampling exactly the dead full-torque spine/hip gears (09-14 dead
  gears note) — the opposite of what the ε floor is for there.
- **NCCL data-parallel as the 2-GPU use**: it doubles the batch, and the
  batch-doubling cost is measured (1.67× samples for equal accuracy) —
  ≈ break-even wall. Async (#2) keeps the batch and can win 1.27×.
- **bf16 observations**: 8× coarser mantissa on metre-scale positions,
  unvalidated (09-10 note); our fp16 path has the r1 / clipfrac_first
  watch items and a working loss scaler.
- **Their throughput numbers as evidence for any of the above**: every
  Protein result above was swept jointly with lr, batch, and net size on
  microsecond envs; only the *direction* of the picks (clip > 0.2, λ < 0.95,
  μ < 0.95) is informative, never the values.

## 3. Sources

- PufferLib repo, `5.0` branch: `src/pufferl.cu` (`train_epoch_gpu`
  1461–1569, `train_impl` 1571–1645, `run_train` async loop 3077–3107,
  `sample_logits` ~600–704, `pufferl_forward_step` 784–878,
  `cosine_annealing` 1441, `create_pufferl` hypers 1780–1812);
  `src/algo.cu` (`muon_step` 1144–1211, `cache_imp_and_v` 1389–1457,
  `ppo_loss_compute` 1478–1607, `puff_advantage` 1698–1747);
  `config/default.ini` and the 86 env `.ini` files (checkout `adbe5cbb`,
  2026-09-10; head `6ffa5b10`, 2026-09-13, `src/` diff non-algorithmic).
  https://github.com/PufferAI/PufferLib
- PufferLib `origin/4.0`: `pufferlib/torch_pufferl.py` 253–354
  (prioritized segment sampling, per-minibatch normalization, reward
  clamp, cosine lr) — the lineage 5.0 pruned.
- GitHub release "5.0 Experiments" (2026-09-13):
  https://github.com/PufferAI/PufferLib/releases — pretrained models and
  Protein logs, no release notes. Docs: https://puffer.ai/docs.html
  ("PPO-variant … synchronous and 1-epoch asynchronous training as in
  CleanBa"; no hyperparameter or algorithm detail).
- "Puffing up PPO" (3.0 announcement, J. Suarez, X, 2025-06):
  https://x.com/jsuarez5341/status/1937554394700231105 — Puffer Advantage
  = GAE at ρ̄ = c̄ = ∞, = V-trace at λ = 1; prioritized replay with α, β₀
  swept; fetch paywalled, quoted via search snippets and checked against
  the 4.0 code.
- Huang et al. 2023, *Cleanba: A Reproducible and Efficient Distributed RL
  Platform*, arXiv 2310.00036 (1-step-stale actor/learner pipeline).
  Espeholt et al. 2018, *IMPALA*, arXiv 1802.01561 (V-trace ρ̄/c̄).
  Engstrom et al. 2020, *Implementation Matters in Deep RL*, arXiv
  2005.12729; Andrychowicz et al. 2020, *What Matters in On-Policy RL*,
  arXiv 2006.05990 (value clipping, advantage recompute, λ/γ sensitivity).
- Ours: `sessions/2026-09-10-pufferlib-5-evaluation.md` (runtime verdict,
  profile 70/9/21, Muon deep dive), `sessions/2026-09-12-v40-probes.md`
  (Muon, wd, per-head clip, self-contact), `sessions/2026-09-12-five-
  credit-horizon.md` (γ/λ 2×2, T=64, width, filter ladder, dead gears,
  2-GPU hardware Q&A), `python/train_batch.py` at HEAD `8b2c59c`.
