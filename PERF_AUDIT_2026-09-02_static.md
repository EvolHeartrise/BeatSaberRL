# Static optimization audit — 2026-09-02

Three independent static-analysis passes over the training stack (no
code was run; every number is derived from the source plus the
measurements already on record in `PERF_AUDIT_2026-08-16.md`, the
in-code comment history, and the 2026-08-31 session logs). Production
shape assumed throughout: N=32768 × T=32, minibatch 16384,
`--minibatches 64 --mb-accum 2`, `--fp16 all`, adapool critic, RTX 5090.

Angles: **A** update phase (fwd/bwd/optimizer), **B** rollout phase
(env kernel / inference / buffers / GAE), **C** PPO algorithmics
(estimators, optimization dynamics, architecture levers).

One arithmetic correction that matters for everything below: with
`--minibatches 64`, `opt.step()` runs once per minibatch
(`train_batch.py` minibatch loop), so the unfiltered update is **64
optimizer steps**, not 32; `--mb-accum 2` only splits each backward.
`--adv-filter 0.5` makes it 32.

---

## Synthesis — what to do, in order

### Tier 0: measure or log before anything else (zero science risk)

| # | item | why first | source |
|---|------|-----------|--------|
| 0.1 | Check `BEAT_ENV_ORDER=1` is exported on the production pod; promote to a trainer flag defaulting ON and print it on the banner | +5.4% steps/s measured at exactly this shape (2026-08-21), bitwise identical, gated by an env var no launcher sets | B §1 |
| 0.2 | `train_batch --bench` at production shape, `--mb-accum 1` vs `2` | accum 2 was adopted for graph-pool OOMs that no longer exist; it doubles the per-minibatch enqueue. Decides whether the update is CPU-enqueue-bound, which gates A §4/§6/§7/§10 | A §2 |
| 0.3 | Log clipfrac + approx-KL (first and last minibatch), pre-clip grad norm / `clip_coef`, per-group and min-head entropy, and a full-batch entropy under the filter | The joint 29-head ratio leaves ±0.2 at a per-head change of 0.63%; nothing measures how much of each update is zero-gradient. `total_norm` is computed and discarded. Under the filter the printed `ent` is the kept-row entropy and is not comparable to unfiltered runs | C §1.2, §2.2, §5 |

**All three executed 2026-09-02 — see "Tier 0 results" below.** 0.1 was NOT set on the pod, is now the `--env-order` flag (default ON), and measures **NULL** here (+0.1%, not +5.4%). 0.2: the update phase is **GPU-bound**, accum 2 costs 1.0% of the update phase / 0.26% wall and saves 1.65 GiB — production default left at 2, and every Tier-1/2 item gated on "if CPU-bound" takes its low end. 0.3 shipped, bit-identical; last-minibatch clipfrac reaches 0.237 (0.249 under `--adv-filter 0.5`) by update 5 and is still rising, so §2.1's gate is essentially met; the 0.5 grad clip stops binding after update 2.

### Tier 1: confirmed-by-code defects and free wins

| # | item | expected | risk | source |
|---|------|----------|------|--------|
| 1.1 | **The critic runs in fp32 through stock torch.** `policy.py:481` adds the fp32 `ffn.2.bias` Parameter to the fp16 residual stream → `x_res` promotes to fp32 → fp32 LN, stock `F.linear` (non-tensor-core), two hidden 113 MB `reshape` copies in the pooling matmuls, fp32 `critic_mlp`. Fix: reuse the final block's already-resolved fp16 `x` (bit-consistent with what its own k/v saw), SDPA with Tq=1 for the pool, pack critic k/v into one GEMM | ~8–12% of update GPU time (~3–4% wall), ~1.5% of rollout, **−1.5 to −1.8 GiB peak** | value changes at fp16-rounding level (class already accepted for the actor); `r1` unaffected | A §1, §9; B §7 |
| 1.2 | `del p_obs, p_act, p_logp, p_adv, p_ret, perm` after the epoch loop — the 1.25 GB permuted obs copy and 243 MB action copy currently live through the entire next rollout | **−1.5 GiB peak** at filter 1.0 (−0.75 at 0.5 — this exactly explains the −0.70 GiB the filter measured) | none, bit-identical | A §3 |
| 1.3 | `b_act` int64 → uint8 (gears ≤ 7), `.long()` per micro-batch | −213 MB resident, −213 MB on the perm copy | none, integers | A §3, B §5 |
| 1.4 | `--adv-filter` × per-minibatch advantage normalization: the kept set `{|A| ≥ median|A|}` has σ ≈ 1.36σ_full, so normalized advantages shrink 0.73× relative to entropy and value terms — implicitly `ent_coef ×1.36`, value-gradient share into the shared trunk ×2.2. Fix: compute μ/σ once per update over the full `f_adv` and use them when filtering (bit-identical when off) | removes a confound from the 0.5 probe's "parity" reading; 3 lines | none when filter off | C §1.1 |
| 1.5 | `relu_bwd`: `at::zeros` → `at::empty` (`fp16_gemm.cu:821`); every column is written | ~900 dead memsets/update | none | A §8 |

### Tier 2: real levers, each gated on a Tier-0 measurement

| # | item | ceiling | gate | source |
|---|------|---------|------|--------|
| 2.1 | Per-head PPO clip (or wider joint ε) | unknown — potentially the largest hidden loss of effective batch in late minibatches | clipfrac on the last minibatch > ~0.25 (0.3) | C §1.2 |
| 2.2 | Critic on a side stream so it overlaps `step_kernel` (bit-identical, ~30 lines) | ~0.8% wall alone; it is the go/no-go for 2.3 | do after 1.1 | B §2 |
| 2.3 | Two-chain rollout: N split in half on two streams, env(A) ‖ policy(B) then swap; T stays 32, batch unchanged | 14% wall (whole forward hidden); realistic 4–8% | 2.2 measures > 0; needs nsys timeline | B §3 |
| 2.4 | CUDA-graph the whole update minibatch (2 micro fwd+bwd + loss-scale + clip + capturable Adam). Differs from the retired rollout graph: fixed-shape, grad-enabled, and the enqueue/GPU ratio has changed. All custom kernels are capturable; `cpu_opt_state()` already normalizes `capturable`/tensor lr | 0 if GPU-bound; 8–12% wall if 0.2 says CPU-bound | 0.2 | A §4 |
| 2.5 | cuBLASLt fused bias+ReLU epilogue (`torch._addmm_activation`) — the un-executed half of the 08-16 audit's Tier 1.1 | 3–5% of update | sm_120 Lt kernel runs at plain-GEMM speed (sm_61 didn't) | A §5 |
| 2.6 | T=64 + `--adv-filter 0.5`: 64 steps/update on 1M kept rows — same step count and minibatch size as today's unfiltered recipe, mean captured GAE window 10.1 → 13.5 ticks, ≥50%-mass samples 62% → 81% | sample-efficiency lever; rescale `--ent-tau` 62, ckpt cadence, `VALUE_NORM_BETA` 0.02; b_obs 2.5 GB | Tier 1 VRAM items first | C §1.3 |
| 2.7 | Ragged (CSR) chart pool — 10 fp32 + 9 bool lanes padded to C≈1.5k for every one of ~92k segments ≈ 6.9 GB at ~7% occupancy | **~−6 GB VRAM**, 0 wall, bit-identical by construction | P unverified — read the `[ranked-pool] … MB` log line | B §4 |
| 2.8 | Adaptive-KL trunk lr (rl_games-style ×1.5 / ÷1.5 per update on mean KL) — the principled form of the lr-ladder finding | self-tuning trunk lr; must be checkpointed | 0.3 logging, threshold set from measured joint-29-head KL | C §2.3 |
| 2.9 | `wgrad` with beta=1 straight into the flat-grad views (skip 62 `AccumulateGrad` adds per micro; packed-head dW into the (145,256) flat view in one GEMM) | ~1% update GPU, 2–3% if CPU-bound | 0.2; re-measure (split-K kernel choice) | A §7 |
| 2.10 | ~20 bookkeeping launches per minibatch (`v_tgt` normalization per micro is constant per update; duplicate `found_inf` ops; 10-kernel loss-scale adaptation chain) | 0.3–2% of update depending on 0.2 | 0.2 | A §6 |
| 2.11 | `torch.compile` only the pure-torch loss tail (post-`forward_stacked`); Dynamo cannot trace the custom `autograd.Function`s | 0.5–3% depending on 0.2 | 0.2 | A §10 |

### Cross-checks between the three passes

- B's roofline estimate (forward ~65–70% of roofline) was made before A found the fp32 critic; ~3 ms of the 26.7 ms `bench_policy_forward` figure is that promotion, so the actor path is closer to roofline than B credited and the critic is the slack. Do A §1 before B §2/§3, and re-baseline the forward afterwards.
- A and B independently arrived at `b_act` uint8 and at packing critic k/v.
- C's §1.1 (filter normalization) and A's §3 (perm copies) both explain parts of the 2026-08-31 probe numbers: the −0.70 GiB is the half-size perm copy, not an update-activation saving; the "parity" was measured with an implicit +36% entropy pressure.
- Explicitly **not** worth doing, all three agree: fusing/parallelizing the GAE scan (<0.5 ms/update), the sampling path, obs dtype casts, per-tick Python overhead (refuted by the graph measurement), value clipping, per-batch vs per-minibatch normalization without the filter (n=16384 → 0.55% σ error), the env kernel internals (the 08-16 do-not-retry list stands).

### Diagnostics worth adding (all fit in the once-per-update print sync)

Approx-KL and clipfrac (first/last mb); `clip_coef` mean; per-group and min-head entropy; full-batch entropy under the filter; bootstrap calibration (`b_val[T]` from update u vs realized 32-tick return from u+1 — the non-structural critic-quality metric EV cannot give); critic skill vs a note-counter baseline; kept/full advantage σ ratio when filtering; diag on kept rows; per-position advantage variance. (C §5)

### Tier 0 results (measured 2026-09-02)

All numbers from ONE session on the RTX 5090 pod `194.14.47.19` (torch
2.8.0.dev+cu128), production shape `--num-envs 32768 --horizon 32
--minibatches 64` (batch 1,048,576, minibatch 16,384), the T=32 campaign
argv. `--bench` caps at 4 updates and is warmup-inflated — the totals
below (~115.6k steps/s) sit ~7% under the 124,224 steps/s 50-update
reference **from the previous pod**, and are NOT comparable to it;
every comparison here is within this pod and this session. Run-to-run
spread over identical arms: **0.31%** on total steps/s (115,300 →
115,655 over 5 reps of accum 2), 0.37–0.74% on the phase seconds.

#### 0.1 `BEAT_ENV_ORDER` — NOT set on the pod; promoted to `--env-order`; the +5.4% does NOT reproduce (NULL)

(a) Confirmed unset: `echo $BEAT_ENV_ORDER` is empty and the name appears
in none of `/root/.bashrc`, `/root/.profile`, `/etc/profile`,
`/etc/environment`. Also confirmed the var still *works*: with
`BEAT_ENV_ORDER=1`, `_env_order_on` is True and `env.env_order` is
non-identity after 40 ticks (identity without it) — so the null below is
a real null, not a knob that failed to engage.

(b) Shipped as `--env-order` / `--no-env-order`, **default ON**,
resolved as *explicit flag > shell `BEAT_ENV_ORDER` > ON* and re-exported
into `os.environ` before the env is constructed (`cuda_env.py`'s reader
is untouched, so the standalone env-var contract and
`tests/test_env_order_bitwise.py` are unaffected). It is in the resolved
`args.json` config dump and on the `=== batched PPO` banner, read off
`venv._env_order_on` rather than off `args` with an assert between them.
Pinned by `tests/test_env_order_flag.py` (7 tests).

(c) Measured, `--bench`, both code versions (env var on the old build,
flag on the new — same numbers):

| arm | steps/s (reps) | mean | Δ |
|---|---|---|---|
| accum 2, env-order **off** | 115,628 / 115,411 / 115,456 / 115,300 | 115,449 | — |
| accum 2, env-order **on** | 115,560 / 115,477 / 115,555 / 115,628 / 115,655 | 115,575 | +0.11% |
| accum 1, env-order **off** | 115,712 / 115,747 / 115,677 / 115,856 | 115,748 | — |
| accum 1, env-order **on** | 115,850 / 115,893 / 116,009 / 115,830 / 115,747 | 115,866 | +0.10% |

The sharp instrument is the **rollout phase alone** (the permutation
touches nothing else), from the new exec-true `[rollout s / update s]`
split: **on 25.425 s (n=6) vs off 25.450 s (n=4) over 4 updates —
+0.10%**.

**Verdict: NULL on this pod.** +0.1%, inside the 0.31% run-to-run
spread, against the +5.4% (123,081 vs 116,804) measured at nominally
this shape on 2026-08-21 on the *previous* pod. Not explained here; the
candidates are the pod change itself and the env changes since (v35
`--limit-arms`, v38 `--crouch-start-prob`, `--wall-despawn`), which move
the ground-contact branch profile the dynamics key sorts on. **Budget
nothing for env-order.** The flag stays and stays default ON — it is
bitwise identical, free, and now visible, which is the actual fix: the
knob was dark for 12 days and no log said which ordering a run used.

#### 0.2 `--mb-accum 1` vs `2` — the update phase is **GPU-bound**, not CPU-enqueue-bound

`train_batch --bench`, production shape, env-order held constant per pair:

| arm | steps/s (reps) | mean | update phase, s / 4 upd (reps) | mean | peak VRAM |
|---|---|---|---|---|---|
| `--mb-accum 2` | 115,555 / 115,628 / 115,655 / 115,456 / 115,300 | 115,519 | 10.81 / 10.79 / 10.81 / 10.82 / 10.87 | **10.820** | **13.78 GiB** |
| `--mb-accum 1` | 116,009 / 115,830 / 115,747 / 115,677 / 115,856 | 115,824 | 10.69 / 10.71 / 10.70 / 10.73 / 10.71 | **10.708** | **15.43 GiB** |

Rollout phase is unchanged across the pair (25.41 vs 25.44 s), as it must
be — that agreement is what says the ~0.4% timer resolution is real.

Isolated GPU-side cost, `bench_policy_update.py --fp16` (best of 3 × 50,
`--mb-accum` knob added for this), two independent invocations each:

| mb | accum | ms/update | peak MB |
|---|---|---|---|
| 8192 | 1 | 20.32 / 20.35 | 2487.0 |
| 16384 | 1 | 39.53 / 39.55 | 4974.0 |
| 16384 | 2 | 39.93 / 39.94 | 2487.2 |

`2 × t(8192) = 40.67` vs `t(16384) = 39.54` → **+2.9%**, but that arm
pays two optimizer steps; the direct knob (two micros, one step) is
**39.94 vs 39.54 = +1.01%**.

**Verdict: NOT CPU-enqueue-bound.** Accum 2 roughly doubles the fwd/bwd
launches per minibatch (~690 vs ~345, the census above) at unchanged
GPU FLOPs. If enqueue were binding, the update phase would grow by
something like that ratio — this section predicted **10–30% wall**. It
grows **1.04%** end-to-end (10.820 → 10.708 s) and **1.01%** in the
isolated instrument, which is the size of the GEMM-efficiency loss from
halving M (221k → 110k rows), not of a doubled enqueue. Two more
consistency checks: the trainer's per-minibatch update time is 42.27 ms
(accum 2) and 41.83 ms (accum 1) against the isolated 39.94 / 39.54 —
the loss-scale + flat-grad + gather machinery costs **+5.8% in both
arms**, which it could not if either were enqueue-starved. The 4090-era
census (CPU 30 ms vs GPU 48 ms/minibatch at accum 1) does **not**
transfer to this pod.

Consequences for the tiers below — take the **GPU-bound branch** of every
item gated on 0.2:
- **A §4** (CUDA-graph the update minibatch): "0 if GPU-bound" → **drop**.
- **A §6** (~20 bookkeeping launches/minibatch), **A §7 / 2.9** (`wgrad`
  beta=1), **A §10 / 2.11** (`torch.compile` the loss tail): take their
  low ends (~0.3%, ~1% of update GPU, ~0.5%), not their CPU-bound ceilings.
- The wins left in the update phase are the ones that move **bytes and
  FLOPs**: A §1 (the fp32 critic), A §3, A §5.

**Production default unchanged** (`--mb-accum 2`), as briefed. For the
record the trade is +0.26% wall for **+1.65 GiB peak** (13.78 → 15.43),
which is not worth taking on its own; revisit after Tier 1.1 + 1.2 + 1.3
free ~3 GiB, when accum 1 becomes free.

#### 0.3 Logging — shipped, bit-identical, and the first real clipfrac/KL numbers

Added to the trainer, all device-side and read at the **existing**
once-per-update print sync (one `torch.cat` → one `.tolist()` → one
sync, exactly as before; the minibatch loop still has **zero** host
syncs): k3 approx-KL and clipfrac on the first and last minibatch, mean
pre-clip grad norm and mean `clip_coef` over the update's optimizer
steps, per-limb-group and min-head entropy, and a full-batch `entfull`.
`get_action_and_value` now exposes per-head entropy (`policy._ent_h`);
the split is a rename of the two reductions that were already there, so
`entropy` is bitwise unchanged. `HEAD_GROUPS` (spine 3 / armL 7 / armR 7
/ legL 6 / legR 6) is derived from the env's actuated-order contract and
pinned against `beat_arena_env.constants` in
`tests/test_head_entropy.py` (14 tests). Gate: **268 passed** (was 247).

`entfull` is taken from the **t=0 rollout forward** — the option that
adds no extra forward at production width (the rollout forward happens
anyway; it costs ~4 kernels and one `(N,29,3)` buffer once per update).
It is therefore over all 32,768 envs regardless of `--adv-filter`, and
it reads the policy at the *start* of update u, so it lags the printed
`ent` by one update: measured `entfull(u=5) = 46.529` vs
`ent(u=4) = 46.530`.

**Bit-identity confirmed.** 5-update runs at the same seed, before and
after, `--no-env-order` on both: `acc` / `accL` / `accR` / `return` /
`ent` / `ev` identical to every printed digit on all five updates
(update 5: `acc 0.6582 accL 0.6502 accR 0.6658 return -15.585 ent 46.485
ev 0.512`), peak VRAM identical at 13.78 GiB, throughput 116,285 vs
116,257 steps/s (−0.02%, inside noise). `ep_return(last100)` differs, as
it already did between env-order arms — it is a mean over whichever 100
episodes the ring buffer popped, which the thread→env assignment
reorders.

A real line (update 5, unfiltered, 5-update run from a fresh init):

```
[kl] upd   5  kl(mb1/mbN) 2.00e-10/1.45e-02  clipfrac 0.0000/0.2365  gnorm 0.180  clipc 1.000  ent/head spine 1.591  armL 1.606  armR 1.607  legL 1.605  legR 1.606  min 1.567 (h0)  entfull 46.529
```

**What the first numbers say (against the C §1.2 prediction):**

| upd | clipfrac mb1 / mbN | k3 KL mb1 / mbN | gnorm | clipc |
|---|---|---|---|---|
| 1 | 0.0000 / 0.0901 | 5.9e-10 / 7.05e-03 | 0.385 | 0.913 |
| 2 | 0.0000 / 0.1093 | 1.6e-10 / 8.04e-03 | 0.228 | 0.969 |
| 3 | 0.0000 / 0.2134 | 1.2e-10 / 1.30e-02 | 0.155 | 1.000 |
| 4 | 0.0000 / 0.2031 | 1.8e-10 / 1.27e-02 | 0.198 | 1.000 |
| 5 | 0.0000 / 0.2365 | 2.0e-10 / 1.45e-02 | 0.180 | 1.000 |

- **C §1.2 is confirmed in direction and is at its own threshold in
  magnitude.** The first minibatch is exactly the same-policy fixed
  point (clipfrac 0, k3 KL ~2e-10, consistent with `r1` ~1e-6), so
  *all* of the clipping is manufactured inside one update by the 63
  preceding Adam steps. Last-minibatch clipfrac climbs 0.090 → 0.237
  over five updates and is **still rising**. The §2.1 gate was
  "last-mb clipfrac > ~0.25 (0.3)": at update 5, unfiltered, it is
  0.2365 — within noise of the gate, on a policy that is still
  essentially uniform (entropy 46.5 of 46.67 max). Read it again at
  update 50 and 500 before committing to per-head clipping, but the
  expectation should now be that the gate is met.
- Under `--adv-filter 0.5` the same series is **higher** despite half
  the optimizer steps (32 vs 64): 0.105 → 0.131 → 0.209 → 0.246 →
  **0.249**. That is C §1.1's normalization confound made visible — the
  kept set's larger |A| spread means each of the 32 steps moves the
  policy further than each of the 64 does.
- **Joint per-update KL is large.** 1.45e-02 (k3, last minibatch)
  against the ~0.008–0.01 an rl_games-style adaptive-KL controller
  targets. C §2.3 has something to act on, and the threshold it needs
  can now be set from data.
- **C §2.2 answered: the 0.5 grad clip is inert in the steady state.**
  `clipc` is 0.913 at update 1 and 0.969 at update 2 — i.e. it binds
  during lr warmup — then **exactly 1.000 from update 3 onward**, with
  `gnorm` 0.15–0.31 against `MAX_GRAD_NORM` 0.5. It is not a live
  regularizer at this shape; it is a warmup guard.
- **Entropy: only the spine has started to move.** Every group sits
  within 0.02 of the `--gears 5` ceiling ln 5 = 1.6094 at update 5, and
  the *only* one falling is spine (1.603 → 1.591). The weakest head is
  **h0 = `spine_pitch`** (1.597 → 1.567); armL / armR / legL / legR are
  pinned at the uniform ceiling. The 29-head sum (46.53 of 46.67) could
  never have shown that — this is the min-head instrument earning its
  place on its first run.


### Tier 1 results (measured 2026-09-02)

Same pod (`194.14.47.19`, RTX 5090, torch 2.8.0.dev+cu128), same
session, same production argv as Tier 0 (`--num-envs 32768 --horizon 32
--minibatches 64 --mb-accum 2`, `--env-order` now default ON). The
baseline was RE-MEASURED at `cb339ca` in this session, not taken from
the Tier 0 table (which it reproduces: 115.3k steps/s, 13.78 GiB,
identical 5-update prints). Every item shipped alone, measured, then
committed, in the order 1.2 → 1.3 → 1.5 → 1.1 so the one
numerics-changing item lands last. 1.4 (adv-filter normalization) was
skipped by brief. Instruments per item: full gate (`pytest tests/`),
5-update same-seed production-shape run for bit-identity / numerics,
`train_batch --bench` × 2 reps (exec-true phase seconds, peak VRAM),
`bench_policy_forward --fp16 --ns 32768` × 2, `bench_policy_update
--fp16 --mbs 16384 --mb-accum 2` × 2. Run-to-run spread on steps/s is
0.31% (Tier 0); anything inside it is called a null below.

#### Cumulative table

| state | commit | steps/s (2 reps) | rollout s / 4 upd | update s / 4 upd | peak GiB (rollout / update) | BPF ms @32768 | BPU ms 16384/accum 2 | gate |
|---|---|---|---|---|---|---|---|---|
| baseline `cb339ca` | — | 115,331 / 115,199 | 25.48 / 25.54 | 10.83 / 10.81 | 13.78 | 26.09 / 26.21 | 40.00 / 39.96 | 268 |
| + 1.2 `del` perm copies | `decd53a` | 115,231 / 115,265 | 25.52 / 25.48 | 10.82 / 10.84 | **13.01** (12.60 / 13.01) | 26.09 / 26.22 | 39.96 / 39.96 | 268 |
| + 1.3 `b_act` uint8 | `54f0bc9` | 115,121 / 115,210 | 25.52 / 25.50 | 10.85 / 10.84 | **12.61** (12.17 / 12.61) | 26.10 / 26.23 | 39.98 / 39.99 | 268 |
| + 1.5 `relu_bwd` `at::empty` | `9436ba3` | 115,186 / 115,435 | 25.53 / 25.43 | 10.81 / 10.84 | 12.61 (12.17 / 12.61) | 26.08 / 26.22 | 39.96 / 39.94 | 268 |
| + 1.1 fp16 critic | `f4f6011` | **125,992 / 125,999** | **24.42 / 24.42** | **8.80 / 8.80** | 12.60 (12.17 / 12.60) | **18.42 / 18.50** | **32.22 / 32.17** | 279 |

Net over the four items: **+9.3% steps/s** (115.3k → 126.0k), update
phase **−18.7%** (10.82 → 8.80 s), rollout **−4.2%** (25.5 → 24.4 s),
peak **−1.18 GiB** (13.78 → 12.60), all of the wall gain from 1.1 and
all of the VRAM gain from 1.2 + 1.3.

#### Per-item verdicts

**1.2 `del` the permuted copies — −0.77 GiB, wall null, bit-identical.**
The audit said −1.5 GiB. The new `--bench` per-phase attribution
(`[bench-mem]` lines: resident at rollout start / rollout peak / update
peak, host-side allocator reads at the syncs `--bench` already does)
explains the difference. Before the `del`, update 1 read
`resident 9.11 / rollout 12.32 / update 13.01` and every later update
`10.58 / 13.78 / 13.78`: the 1.46 GiB of perm copies (p_obs 1.17 +
p_act 0.23 + small) raised the rollout peak from 12.32 to 13.78, and
the update bracket ALSO peaked at 13.78 because its first op is the
GAE bootstrap forward at the same N — the same 3.21 GiB forward
transient as the rollout. After the `del`: `9.40 / 12.60 / 13.01`. So
the audit's inference "the peak is in the rollout" was right, but the
minibatch loop (perm copies 1.46 + saved activations ~2.15 over the
9.4 resident) sat only 0.77 GiB under it and becomes the new peak. A
second leak found on the way: the `_` local holding the last micro's
action slice is a VIEW of `p_act` and kept its storage alive past the
`del` (+0.23 GiB at the next rollout start); it is in the `del` list now.

**1.3 `b_act` uint8 — −0.40 GiB, wall null, bit-identical.** −0.20 on
the buffer (resident at update 1: 9.11 → 8.91) and −0.20 on its perm
copy under the update-phase peak. The cast back to int64 happens per
micro-batch inside `get_action_and_value` (gather needs a long index).

**1.5 `relu_bwd` `at::zeros` → `at::empty` — NULL, bit-identical.**
Verified from the launch geometry first: threads `0..D2C-1` of every
launched block store both floats of their column pair, `2*D2C == d`
for every dispatch, and a thread whose grid-stride loop never ran
still stores its `0.f` partials, so every `(block, column)` of `part`
is written. The JIT rebuild fired on the pod (srchash `d74f6e…` →
`4dffc3…`, `.so` rebuilt inside the gate run). Every instrument came
back inside the spread (steps/s 115,186 / 115,435, update 10.81 /
10.84 s, BPU 39.96 / 39.94 ms). Kept because it is correct and free;
~900 tiny memsets per update are not a byte/FLOP item in a GPU-bound
phase, exactly as Tier 0.2 predicted for launch-count items.

**1.1 The critic on the fp16 stream — the whole Tier 1 wall gain, and
bigger than the audit's estimate.**

*Diagnosis confirmed before the change* (`x_res` probe at N=8192 on
the fp16 stream): `x_res` **fp32**, `critic_ln` input/output **fp32**,
and the profile of a value-loss fwd+bwd showed `aten::linear` ×7 and
fp32 `sgemm_largek` kernels — i.e. the stock non-tensor-core path.
After: every one of those reads fp16, the same probe's device time
fell 69.0 → 56.4 ms, and `_scaled_dot_product_efficient_attention`
runs ×4 (3 blocks + the critic pool; not the math fallback).

*What shipped:* the final block returns its already-resolved fp16
residual stream (`want_resolved`, the `x` its fused `ln_attn` pass
writes before the `n_queries` slice) instead of `forward_stacked`
recomputing `tokens + delta + delta_b`; `critic_ln` runs the fp16
kernel; `critic_q`/`critic_out` go through `fp16_linear` (the body-row
query slice needed a `.contiguous()` guard in `_Fp16Linear.forward` —
a `(B,1,K)` slice reshapes to a strided view that `wgrad` rejects);
`critic_k`/`critic_v` are one packed GEMM via the existing group
machinery (a flat group in `flatten_fp16`, a version-keyed cat cache
otherwise); the pool is `F.scaled_dot_product_attention` with Tq=1
under `BEAT_ATTN=sdpa` (the legacy matmul path stays as the escape
hatch, and it now covers the critic too); `critic_mlp` takes the
`bias_relu` / `relu_bwd` epilogues. The fp32 record path is
bit-identical (the fused `ln_fwd_add`'s `x` IS `a + b` in fp32, which
is what the old recompute produced there — pinned by
`tests/test_critic_fp16.py::test_fp32_path_matches_old_recompute_bitwise`).

*Measured:* `bench_policy_forward --ns 32768`: 26.1 → **18.4 ms
(−29%)**; `bench_policy_update 16384/accum 2`: 39.95 → **32.2 ms
(−19%)**, peak 2487 → 2475 MB; `train_batch --bench`: update phase
10.82 → **8.80 s (−18.7%)**, rollout 25.5 → **24.4 s (−4.2%)**, total
**115.3k → 126.0k steps/s (+9.3%)**, two reps within 7 steps/s of each
other. The audit estimated 8–12% of update GPU time and ~1.5% of the
rollout; the update came in at 19% and the rollout at 4%, so the fp32
critic cost more than the byte census gave it — plausibly the
non-tensor-core SGEMMs and the m=1 batched GEMMs, which the census
priced at "~1 ms". **VRAM did NOT move** (12.61 → 12.60): the rollout
peak stayed at 12.17 (3.20 GiB over resident) with the fp32 critic
transients gone, so that transient is something else (the env kernel's
own allocations, or the actor's attention/FFN buffers at N=32768 —
not attributed here); the audit's −1.5 to −1.8 GiB for this item was
wrong.

*Numerics gate:* (a) gate **279 passed** (268 + 11 new pin tests);
(b) `tests/test_policy.py` unchanged and passing (its value check is
the fp32 path, which is bit-identical); (c) 5-update same-seed run vs
the pre-change run from the same session, update 5: `acc 0.6584` vs
0.6582, `return −15.607` vs −15.585, `ent 46.484` vs 46.485, **`ev
0.502` vs 0.512** (the brief's ±0.02), `r1 1.0e-06` vs 1.6e-06 —
update 1 is identical in acc/return (the first rollout samples from
the same init policy; the value is not in the sampling path), and the
divergence from update 2 on is the trajectory noise of a changed
value gradient, not a shift; (d) `perf_numerics compare` against the
pre-change dump: logits and head grads **bit-identical** (actor path
untouched), value max-diff 5.6e-5 on 5.5e-2, value-MLP / `critic_out`
grads ≤1% max-rel, `critic_v` 10%, and the score-path grads
(`critic_q`, `critic_k`, `critic_ln`) at O(1) max-rel — which needed
the isolation probe below before accepting.

*Isolation probe (grads of a value-only loss, 4096 rows, five fp16
arms vs the fp32 path on the same weights):* packed vs unpacked k/v is
**bit-identical in the forward** and in every grad downstream of `hv`;
`dW_k`/`dW_v` differ at 1e-12–1e-10 (wgrad accumulation order at N=256
vs 128) and `critic_ln`'s grad at 5e-3 rel (dhv is one rounded GEMM
instead of two plus an add) — routing is correct. SDPA vs the legacy
fp16 matmul pool: value max-rel 5.5e-3 vs 4.3e-3; downstream critic
grads cos > 0.998 in both. The score-path grads are at the init noise
floor: `|dW_k|max` ≈ 7e-6, `|dW_q|max` ≈ 7e-6 (`critic_k.bias`'s grad
is mathematically 0 — a per-key constant is softmax-invariant), where
cos vs fp32 is 0.3–0.4 for either fp16 arm; scaling the critic q/k
weights ×8 lifts them to cos 0.85–0.88 (SDPA) / 0.96–0.97 (legacy).
So SDPA-Tq=1's backward is somewhat less faithful on the score path
than the fp16 matmuls at init, both are in the fp16 class the actor's
2026-08-15 SDPA ladder already accepted, and the run-level gate (c)
shows no EV/acc effect. Kept SDPA; `BEAT_ATTN=legacy` is the bisect
hatch if a long run ever disagrees. The pin test compares fp16 vs
fp32 only on the grads that carry signal at init (v / out / value MLP,
cos > 0.98) and packed vs unpacked for routing.

*Also landed with 1.1:* `perf_numerics.py` had crashed on every
adapool policy since 2026-08-19 (it probed `value_head`); it now
probes the critic's packed k/v, body-row query and value-MLP grads
instead, which is how (d) was run.

#### What Tier 1 changes for the rest of the plan

- The update phase is now 8.80 s / 4 updates against the rollout's
  24.4 (26% / 74% exec-true). A §5 (cuBLASLt bias+ReLU epilogue) and
  A §7 (`wgrad` beta=1) are the remaining byte/FLOP items there and
  should be re-estimated against 8.8 s, not 10.8.
- `--mb-accum 1` costs +1.65 GiB (Tier 0.2) — at 12.60 peak on 32 GB
  that is now free VRAM-wise; it buys ~1% of the update phase.
- The rollout-phase peak (12.17, 3.20 GiB over resident) is NOT the
  critic and is now the thing to attribute if VRAM matters (T=64 /
  2.6 wants b_obs 2.5 GB); `torch.cuda.memory._record_memory_history`
  over one rollout tick would settle it.
- B §2 / 2.2 (critic on a side stream) is now cheaper to test — the
  critic is ~a third of the forward it was.

### Architecture notes (C §3)

- `trunk = Linear, ReLU, Linear` then Linear heads: the actor's readout ends in two linear layers in a row, so `trunk.2` adds optimization geometry but no capacity, and `dormant` measured on its signed linear output is structurally ~0 — the instrument is placed where it cannot read. Measure dormancy on `trunk.0`'s ReLU output or FFN hiddens; a ReLU after `trunk.2` is a probe, not a patch.
- Action-head init and temperature: not a lever (entropy 42 of 46.7 max at start; CleanRL's 0.01 head gain is irrelevant after ~50 updates).
- Critic head is sound; shared-trunk interference is not the bottleneck (gradcos ≈ 0, |g_v| ≪ |g_pi|) — until the filter's ×2.2 value share (1.4).

---

## Pass A — update phase (full report)

Scope: `python/train_batch.py` main() from the flatten (L1046) through `opt.step()` (L1277), `policy.py` forward/critic, `fp16_linear.py`, `fused_ln.py`, `csrc/layer_norm.cu`, `csrc/fp16_gemm.cu`, `bench_policy_update.py`, `PERF_AUDIT_2026-08-16.md`, and the 2026-08-31 session note. Static only; every number below is derived from bytes/launch counts in the code and labelled as an estimate.

### Baseline census

Per micro-batch (B=8192, fp16 stream, grad enabled), counted from the code paths actually taken:

- Forward ≈ 125 launches: obs split + 3 encoder MLPs ≈ 14; attn_bias 3 (L464–467); block 0/1 ≈ 11 each (ln, packed qkv addmm, SDPA + mask pad, att reshape copy, out_proj, ln_fwd_add, ffn mm + `bias_relu_` + nobias mm); `x_res` 2; final block ≈ 12; trunk 3; packed heads 1 + `.float()` 1; **critic ≈ 20** (see §1); log_softmax/logprob/entropy ≈ 7; loss + diagnostics ≈ 30 (L1177–1205).
- Backward ≈ 220 (autograd ≈ 1.7× fwd, plus **138 `AccumulateGrad` in-place adds** — one per parameter tensor, including 58 for the 29 head weight/bias slices returned by `_Fp16PackedLinear.backward`, L316–324 — plus 14 `pdw/pdb.sum(0)` for the 7 LNs and 7 `part.sum(0)` for relu sites).
- Per minibatch: 2 micro × ~345 + ~35 fixed (zero 1, loss-scale bookkeeping ≈ 24, clip 5, fused Adam 4, replica refresh 1) ≈ **725 launches**; ≈ **46k launches/update** at 64 minibatches.

Saved-activation footprint per micro-batch ≈ 1.85 GB actor (fp16; the (B·27, 512) ffn hidden is 226 MB per block) + ≈ 0.45 GB critic (fp32, see §1) ≈ 2.3 GB.

The prior census (08-16 audit Tier 2.1, 4090, **mb 16384 / accum 1**) measured CPU enqueue 30 ms vs GPU 48 ms per minibatch. Production now runs **accum 2** on a 5090 whose GPU time is ~40 ms/minibatch (39.7 ms measured in `bench_policy_update`, which has no accum). Accum 2 roughly doubles the fwd/bwd enqueue per minibatch (~55 ms if the 4090-pod CPU figure transfers) against ~40 ms GPU → the update may now be **CPU-enqueue-bound**. This can't be settled statically; it changes the payoff of half the items below, so §2 is the first measurement to make.

### A §1. The critic runs in fp32 through stock torch — unintended dtype promotion

**Anchor:** `policy.py:481-482`

```python
x_res = (x_res + delta if delta_b is None
         else x_res + delta + delta_b)
```

`delta_b` is `ffn.2.bias`, an **fp32 (d,) Parameter** (the fold-tail path, `_seq(..., fold_tail=True)` L88–94, default on). A dimensioned fp16 tensor + a dimensioned fp32 tensor promotes to **fp32** (only 0-dim tensors/Python scalars are lower priority). So on the default fp16 stream `x_res` is fp32 (B,27,128), and everything downstream follows:

- `critic_ln(x_res)` (L535) → `FusedLayerNorm.forward` takes the **fp32** `ln_fwd_kernel` (the deliberately non-register-cached, non-specialised one, `layer_norm.cu:54`), output fp32.
- `_lin(critic_q/k/v/out, hv)` (L536–538, 546) → `_lin` dispatches on `x.dtype is torch.float16` (L73) → **stock `nn.Linear` / `F.linear` in fp32**, with stock autograd backward (fp32 dX GEMM, fp32 dW GEMM, `sum(0)` db). No TF32 anywhere in the repo, so these are full-rate FP32 SGEMMs, not tensor-core.
- `torch.cat([a, h])` (L547) promotes `h` (fp16 trunk readout) to fp32 → `critic_mlp` runs the stock fp32 path too (`_run` L106 only takes the `bias_relu` epilogue for fp16).
- `(q*scale) @ k.transpose(-2,-1)` and `softmax @ v` (L543–545): 4-D `matmul` with batch dims (B, 4) whose strides (3456, 32) are not collapsible → `reshape` **copies k and v** (113 MB fp32 each, forward and again in backward). The actor blocks avoid this because SDPA accepts the strided 4-D layout directly.

The comments say "the critic pays it once, one (B,27,d) add" (L475–478) and `val.float()` (L548) expects an fp16 value — this is almost certainly unintended. The critic-design topic never declares an fp32 critic; `git log -S x_res` shows it landed with the adapool commit (8d36d47) untouched since.

**Estimated cost (per micro-batch, B=8192):** forward ≈ 1.2 GB traffic (two adds 340 MB, LN 226 MB, k/v GEMMs 450 MB, two bmm copies 450 MB), backward ≈ 2.5 GB (k/v dX+dW+db ≈ 1.1 GB, grad accumulation of dhv from q/k/v incl. a full-size zero-fill from the `hv[:, 2:3]` slice backward ≈ 450 MB, fp32 ln_bwd 340 MB, the add-backward cast + reductions ≈ 400 MB). At B=16384 (the bench shape) ≈ 7.4 GB → ~5 ms at ~1.5 TB/s effective, + ~1 ms of full-fp32 GEMM compute, + inefficient m=1 batched GEMMs. **≈ 6–7 ms of the measured 39.7 ms/minibatch (~15%).** On the fp16 stream ≈ 2–2.5 ms.

**Expected gain:** ~8–12% of the update phase (≈ 3–4% wall), plus a rollout-side bonus: the same promotion runs in the no-grad rollout forward at N=32768 (≈ 4.8 GB extra traffic ≈ 3 ms of the 26.7 ms `bench_policy_forward` figure, ~1.5% of the rollout phase). **VRAM:** in the rollout at N=32768 the fp32 critic transients (x_res, hv, k, v, k-copy, v-copy ≈ 6 × 453 MB ≈ 2.7 GB live) drop to ≈ 0.9 GB; in the update ≈ −0.25 GB per micro-batch of saved tensors. If the 13.78 GiB peak lands in the rollout (likely — see suspicious list), this is −1.5 to −1.8 GiB.

**Fix (preferred):** don't recompute `x_res` at all. The final block's `ln_attn.forward_add(x, delta, delta_bias)` (L249) already writes the resolved fp16 residual stream `x = round_h(a + (b + bb))` — bit-identical to what its own k/v see — before the `x[:, :n_queries]` slice at L283. Return/stash that `x` from the final block and feed it to `_critic`. This deletes 2 kernels (340 MB), makes the critic's input *more* consistent with the actor, and the whole critic then flows through the existing fp16 machinery: fp16 `ln_fwd_h_kernel<2>`, `fp16_linear` for q/k/v/out (custom `wgrad`), `_run`'s `bias_relu` epilogue for `critic_mlp`, `relu_bwd` for its two ReLUs (d=256 is in `_RELU_BWD_DIMS`). Replace the two `matmul`s with `F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.unsqueeze(1))` (Tq=1). Autograd: the critic's gradient on `x` then enters `_LnAddFn.backward` as `gx`, which is already the fused path (`fused_ln.py:563-578`).

**Numerics/repro risk:** the value prediction changes at the fp16-rounding level. `r1` is unaffected (value isn't in the ratio). Gate: the 50-update smoke reading EV (0.98 at upd 776 on the af50 probe is the reference); `tests/test_policy.py:59-63` (atol 1e-5 on `v`) is a same-mode consistency test and should still pass. **Effort:** ~half a day + one pin test asserting `x_res.dtype is torch.float16` on the fp16 stream. **Verify:** `bench_policy_update.py --fp16 --mbs 8192 16384` (ms and peak MB) and `bench_policy_forward.py --ns 32768` before/after; `train_batch --bench` at production shape for the wall number.

### A §2. `--mb-accum 2`: its justification was removed with `--graph-rollout` — re-test accum 1 (zero code)

**Anchor:** `v27_critic_probe.py:6-9` — accum 2 was adopted because "plain N=32k OOMs on **graph pools** + the env's non-torch state", with `expandable_segments`. The graphed rollout and its private pools were deleted 2026-08-31. Every current driver (`v27`–`v30_*_probe.py:51`) still carries `--mb-accum 2` verbatim.

**Mechanism:** accum 2 doubles fwd/bwd launches per minibatch (~690 vs ~345), halves GEMM M (221k → 110k rows), and doubles the CPU enqueue per optimizer step — the ratio that decides whether §4 is worth anything. Accum 1 is never slower on the GPU side; the only price is VRAM: +≈ 2.3 GB of saved activations (13.78 → ~16 GiB on 32 GB — fits, and §1/§3 buy that back).

**Expected gain:** 0–3% of the update if it is GPU-bound; **10–30%** if the accum-2 enqueue exceeds GPU time. `help` text calls accum "same math (…modulo fp association)", so accum 1 vs 2 is the same near-tie class already accepted.

**Verify (this also gates §4, §6, §7):** `train_batch --bench` at production shape with `--mb-accum 1` vs `2` (exec-true split); `bench_policy_update.py --fp16 --mbs 8192 16384` — if 2·t(8192) ≈ t(16384) the GPU side of micro-batching is free and any production gap is CPU. Add a `--mb-accum` knob to `bench_policy_update.py` for the isolated number.

### A §3. VRAM: the permuted copies outlive the update; per-micro gathers; int64 actions

**Anchors:** `train_batch.py:1157-1158` (`p_obs, p_act, p_logp, p_adv, p_ret = f_*[perm]`), L786 (`b_act` int64), L1279–1283 (end of update — no `del`).

- `p_obs` (1,048,576 × 598 fp16 = **1.25 GB**) and `p_act` (× 29 int64 = **243 MB**) are Python locals assigned inside the epoch loop and **stay allocated through the entire next rollout and GAE** until reassigned on the next update. That matches the adv-filter measurement exactly: at 0.5, `train_n` halves → p_obs+p_act shrink by 0.63 + 0.12 GB = 0.75 GB ≈ the −0.70 GiB observed. So the peak is (or is within ~1 GB of) the rollout, and these buffers sit under it doing nothing. `del p_obs, p_act, p_logp, p_adv, p_ret, perm` after the epoch loop: **−1.5 GiB peak at filter 1.0, −0.75 at 0.5**, zero risk, bit-identical.
- Going further: gather per micro-batch instead of per epoch — `torch.index_select(f_obs, 0, perm[sl], out=static_obs)` (and likewise act). Same total bytes moved, identical rows in identical order → bit-identical. Costs +2 launches per micro, removes the 1.5 GB resident copies entirely, and provides the static input buffers §4 needs. Differs from the retired host-side shuffle (device `randperm`, device `index_select`, no H2D).
- `b_act` is int64 for values in [0, 5). Store uint8 and `.long()` per micro: **−213 MB** resident, −213 MB on the perm copy. Bit-identical.

Wall-clock: ~0. VRAM: −1.5 to −2 GiB combined. Effort: 1 hour.

### A §4. CUDA-graph the whole minibatch — gated on §2

How this differs from what was retired: the rollout graph captured a no-grad forward at N=32768 — one large launch stream that was already GPU-bound. The 08-16 audit Tier 2.1 refuted graphing the update at **mb 16384 / accum 1 on a 4090** (CPU 63% busy). The case here rests on two things that changed since: accum 2 doubled the enqueue per step, and the 5090 cut the GPU time ~20%. If §2 shows the update is GPU-bound at the chosen accum, **skip this**.

Feasibility with the custom stack (all checked statically):
- `ln_fwd/ln_fwd_add/ln_bwd`, `bias_relu_`, `relu_bwd`: plain stream launches on `getCurrentCUDAStream()`, capturable.
- `wgrad` uses `at::cuda::getCurrentCUDABlasHandle()` (`fp16_gemm.cu:860`) — torch's own handle with its per-stream workspace (warm up before capture).
- SDPA mem-efficient backward is capturable.
- Fused Adam: needs `capturable=True` and tensor `lr` — the trainer sets `g["lr"]` as a Python float every update (L914–915); make it a per-group 0-dim device tensor updated by `.fill_()`. `cpu_opt_state()` (L432–446) already normalises `capturable`/`lr` for checkpoints.
- Loss-scale state is already fully device-side (L1217–1254); `_amp_foreach_non_finite_check_and_unscale_` is graph-safe.
- `flat_refresh()` (`fp16_linear.py:93-98`) is version-keyed in Python: capture right after a `bump_weights_version()` so the `copy_` is inside the graph.
- `ratio_diag` (L1188–1193) is assigned only on the first minibatch — compute it every replay into a static tensor.
- Inputs: static micro-batch buffers filled by `index_select(out=)` (§3).
- Autograd accumulation into the pre-existing flat-grad views is in-place (the `flat_guard` at L1256–1266 proves it); run the guard before capture.

Launch count per minibatch goes from ~725 to ~3. **Gain: 0 if GPU-bound; plausibly 20–30% of the update (≈ 8–12% wall) in the accum-2 CPU-bound scenario.** Numerics: bit-identical replay. Effort: 2–3 days + tests. Verify: `train_batch --bench` exec-true split; GPU-busy % in the update via `nsys profile --stats=true` or `torch.profiler`.

### A §5. cuBLASLt fused bias+ReLU epilogue — the un-executed half of the 08-16 Tier 1.1

**Anchor:** `fp16_linear.py:416-422` (`torch.mm` bias-free, then `_load().bias_relu_(y, b)` — a second full pass over the output). The split was justified by sm_61 addmm-epilogue slowness. The width fix was done 2026-08-15; the Lt re-probe is not in the A/B ladder.

**Mechanism:** `torch._addmm_activation(b16, x2, wh.t(), use_gelu=False)` routes to cublasLt's `RELU_BIAS` epilogue with contiguous fp16 inputs and a 1-D bias. Removes the `bias_relu_` pass: per micro-batch ≈ 1.5 GB less traffic ≈ **~1 ms/micro ≈ 3–5% of the update** if the Lt kernel runs at plain-GEMM speed on sm_120.

**Numerics:** one rounding instead of two, bias enters as fp16 — fp16 last-rounding class. Backward unchanged. Effort: half a day behind a `BEAT_LT_EPILOGUE=1` toggle. Verify: `bench_policy_update --fp16 --mbs 16384` A/B, then the grad-probe rel-error gate.

### A §6. Per-minibatch bookkeeping: ~20 pure-overhead launches per minibatch

All in the L1159–1278 loop; payoff depends on §2 (≈ 0.3% of the update if GPU-bound, ≈ 1.5–2% if CPU-bound). All bit-identical:

- L1245–1246 recompute `found_inf[0].clamp(max=1.0)` and `1.0 - overflow` that L1223 already computed as `ok` (−2).
- L1217 `inv_scale.copy_(fp16_scale.reciprocal())` → `torch.reciprocal(fp16_scale, out=inv_scale)` (−1).
- L1197 `v_tgt = (p_ret[sl] - ret_mu) / ret_sig` is constant per update: normalise `f_ret` once instead of 2 × 128 (−254/update).
- L1194 `(a_sd + 1e-8)` per micro → per minibatch.
- L1185–1187 `logratio.abs().max()` + `maximum(out=)`: `torch.linalg.vector_norm(logratio, ord=inf)` is 1 launch.
- L1248–1252 scale-adaptation chain (≈ 10 kernels): 3–4 with `torch.where` on a stacked pair.

### A §7. `wgrad` with beta=1 straight into the flat-grad views

**Anchors:** `fp16_gemm.cu:858-859` (`c = at::empty(...)`, `beta = 0`), `fp16_linear.py:313-317, 364, 396, 443`, `train_batch.py:750-757` (every `p.grad` is a flat view).

Every weight grad is produced as a fresh fp32 temporary and then added into the flat view by `AccumulateGrad` — 62 weight tensors per micro. Passing `beta=1` with `c = p.grad` (the head weights are laid out consecutively in `flatten_fp16_replicas`, so the packed head dW can accumulate into the (145, 256) flat view in **one** GEMM) and returning `None` removes 62 add kernels + 62 temporaries per micro ≈ 8k launches/update. **Gain ≈ 1% of the update GPU time; 2–3% if CPU-bound.** Differs from the retired packed stack: autograd still owns the graph; only weight-grad accumulation moves into the GEMM epilogue. Numerics: last-rounding class; cuBLAS may pick a split-K kernel with beta≠0, so re-measure. The diag block's two `retain_graph` backwards rely on accumulation into `flat_grad` — beta=1 preserves that.

### A §8. `relu_bwd`: `at::zeros` → `at::empty`

**Anchor:** `fp16_gemm.cu:821`. Every block writes all `d` columns of its `part` row, so the memset is dead. 7 relu sites × 128 micro ≈ 900 memset launches/update. `ln_bwd` already uses `at::empty` (L583–584).

### A §9. Once the critic is on the fp16 stream: pack critic k/v, SDPA for the pool

`critic_k`/`critic_v` share input `hv` (L537–538) — the qkv-group machinery turns them into one GEMM (N=256) + one `wgrad` + one bias sum. `critic_q` runs on the single body row, so it stays separate. SDPA with Tq=1 replaces the two batched matmuls and their hidden copies. Together ≈ −8 launches and ≈ −0.5 GB traffic per micro on top of §1 — ~1–2% of the update. Only sensible as part of §1.

### A §10. torch.compile: only the pure-torch loss tail is viable

Dynamo cannot trace through `_Fp16Linear*`, `_LnFn`, `_LnAddFn` (custom `autograd.Function`s wrapping extension calls) without custom-op registration. `mode="reduce-overhead"` is CUDA graphs, i.e. §4 with less control. Viable: compile `(logits, action, old_logp, adv, a_mu, a_sd, newval, v_tgt, ent_coef_t, scale_t) → loss, entropy_mean, logr_max`: ~30 small fwd kernels + ~40 bwd on (8192, 29, 5) tensors fuse to ~6–8. Pass `ent_now`/scale as tensors. **Gain ≈ 0.5–1% if GPU-bound, 2–3% if CPU-bound.**

### A — suspicious but not confirmable statically

1. **Update CPU-bound at accum 2?** Every update-phase number in the comments (39.70 ms × 64) is at accum 1; production is accum 2.
2. **Which SDPA backend actually runs** (`policy.py:306-307`): fp16, dh=32, T=27, Tq=3, additive mask (B,1,1,27). A silent fall-through to the math path would materialise (B,4,Tq,27) scores. Check with `sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])` around one forward.
3. **Hidden copies in the critic's 4-D `matmul`s** — derived from stride collapsibility; `torch.profiler` on one minibatch shows `copy_` adjacent to `bmm`.
4. **Where the 13.78 GiB peak lands.** `max_memory_allocated` excludes the env kernel's ~1.4 GB of driver-managed local memory and cuBLAS/SDPA workspaces. `torch.cuda.memory._record_memory_history()` over one update would settle it.
5. **`torch._foreach_mul_(grads, ok)`** (L1224) with a 0-dim CUDA tensor: on older builds the scalar path would `.item()` — a hidden sync per minibatch.
6. **`x[:, :n_queries].contiguous()` backward** (`policy.py:283`): SliceBackward zero-fills a full (B,27,128) fp16 tensor per micro. ~0.1 ms/micro.
7. The `.half()` at `policy.py:440` is a no-op on fp16 obs, so the comment "the cast IS the copy" is stale — the copy happens later, once, in the Function.

### A — already well-optimised

No host syncs in the minibatch loop (verified line by line); flat grad buffer with single-tensor unscale/check/clip; fused Adam over 2 flat groups; one replica refresh per step, version-keyed; packed qkv and packed 29-head GEMMs; fused residual-add + LN; ffn.2 bias fold; register-cached fp16 LN at d=128; fused `relu_bwd`; custom `wgrad`; final-block query pruning; block-invariant `attn_bias` hoist; device-side hoisted GAE; fp16 obs buffer; adv-filter composed lazily into the shuffle index.

---

## Pass B — rollout phase (full report)

### B §0. Where the rollout tick goes (derived)

Per tick at N=32768 on the 5090, using the repo's own numbers (`train_batch.py:960-965`: env kernel ~43% of a 5.96 s update = 2.56 s / 32 ticks; `bench_policy_forward --ns 32768` = 26.72 ms):

| item | ms/tick | share of tick | share of wall |
|---|---|---|---|
| `step_kernel` (one launch, 8 substeps) `beat_arena_cuda.cu:5983-5990` | ~80 | ~75% | ~43% |
| policy forward (`policy.py:432-520` + `_critic` 522-548) | ~26.7 | ~25% | ~14% |
| sampling: log_softmax, exponential_, log, sub, argmax, gather, sum (`policy.py:622-634`) | ~0.1 | <0.1% | ~0 |
| buffer writes: `b_obs[t]=obs` (78 MB fp32 read + 39 MB fp16 write), `b_act` 7.6 MB, four 131 KB vectors (`train_batch.py:929-938`) | ~0.1 | <0.1% | ~0 |
| `_split` gathers: 2 `index_select` over 64 lanes + `notes.half()` (`policy.py:424-440`) | ~0.15 | <0.2% | ~0 |
| env wrapper: `randint` cand, optional `_kick`, `_pose_jit.uniform_`, optional argsort (`cuda_env.py:660-725`) | <0.1 | — | ~0 |

Total ≈ 107 ms/tick; 32 ticks ≈ 3.43 s ≈ 57% of 5.96 s. Everything outside the two big kernels is well under 1% of the tick. GAE is 64 launches on 131 KB tensors plus ~8 elementwise passes over 4 MB tensors: well under 0.5 ms per update. Roughly ~100 launches per tick; CPU enqueue is ~20-30x faster than GPU execution here, which is exactly why the whole-rollout graph measured 0.99x.

Policy forward roofline: 884k tokens × ~590k MACs/token ≈ 1.0 TFLOP, final block n_queries=3 so ≈ 0.8 TFLOP → ~8 ms of compute at ~105 TFLOPS. Activation traffic ≈ 14-15 GB ≈ 8 ms at 1.8 TB/s. Floor ≈ 16-18 ms vs 26.7 measured: ~65-70% of roofline (before accounting for Pass A's fp32-critic finding, which is ~3 ms of the gap). The only way to make the forward "free" is to run it concurrently with the env kernel (§2, §3).

Obs path: the kernel writes obs directly into `venv._obs` (fp32, `cuda_env.py:615`, `beat_arena_cuda.cu:4903-4905`), NOT into `b_obs[t]`; the trainer copies it with an fp32→fp16 cast (`train_batch.py:929`). The obs buffer is already fp16 under `--fp16 all` (1.25 GB). The critic runs at full width every tick and must: `b_val[t]` feeds GAE. It costs roughly 5-8% of the forward.

### B §1. Confirm `BEAT_ENV_ORDER=1` is actually exported on the production pod

- Anchors: `cuda_env.py:608`, `cuda_env.py:713-724` (dynamics-key argsort), `beat_arena_cuda.cu:5772`.
- The R12 dynamics-key permutation measured +5.4% end-to-end at exactly this shape on the 5090 (08-16 audit addendum, 2026-08-21), is per-env bitwise identical (`tests/test_env_order_bitwise.py`), and is default OFF behind an env var. The v30 campaign launcher (`v30_package_probe.py:50-62`) does not set it; the v34 trunk note says it was exported by hand.
- Change: promote to a trainer flag defaulting ON (or set `os.environ` in the launchers) and print its state on the `=== batched PPO` line.
- Expected gain: 0% if already on; +5.4% steps/s if not. Risk: none. Effort: 10 lines. Verify: `--bench` with/without; 123,081 vs 116,804 from 2026-08-21 is the reference.

### B §2. Run the critic on a side stream so it overlaps the env kernel (bit-identical probe)

- Anchors: `policy.py:519-520` (critic called inline, after heads), `train_batch.py:930-936`, `cuda_env.py:725` (`_launch`).
- Mechanism: only the actor path is on the critical path to `venv.step`. The critic's inputs exist before the heads GEMM; its output is not needed until GAE. On a side stream the critic's GEMMs + LN + softmax run concurrently with `step_kernel`.
- Why concurrency should help here: `step_kernel` is latency-bound and parallelism-starved (08-16 Tier 4: 4-6 warps/SM, ~0.1% FLOP utilization). At N=32768: 1024 blocks × 32 threads, 254 regs → ~6 blocks/SM = 192 threads/SM, tensor cores idle, and a straggler tail (thread-per-env with divergent contact counts). A concurrent kernel fills the tail and idle issue slots. This is a GPU-side concurrency change, not a launch-overhead change, so the `--graph-rollout` 0.99x verdict does not apply.
- Expected gain: ceiling ~1.5-2 ms/tick ≈ 0.8% wall. Its real value is as the cheapest go/no-go for §3. If this measures ~0, do not build §3.
- Risk: bit-identical (same kernels, same batch; actions unchanged). Watch stream semantics (`record_stream` / keep the join inside the tick). Register pressure may prevent GEMM CTAs from co-residing with 6 env blocks/SM — then only the tail overlaps.
- Effort: ~30 lines. Verify: `--bench` A/B; `perf_profile.py --phase env` (kernel time sum unchanged, wall down); gate; r1 stays 0.

### B §3. Two-chain rollout: split N into two N/2 halves on two streams, alternate policy/env

- Anchors: `train_batch.py:928-947`, `cuda_env.py:349-660` (env instance = per-N state; the pool dict is shareable), `beat_arena_cuda.cu:5983-5990`.
- Two `CudaBeatVecEnv(N/2, pool=pool)` instances on two streams; per tick, chain A's env kernel runs while chain B's policy forward runs, then swap. Same `(T,N,...)` buffers written at `[:N/2]` / `[N/2:]`; GAE and the update unchanged; T stays 32 per chain.
- How it differs from retired ideas: not `--graph-rollout`; not stale-policy pipelining (each chain strictly on-policy). `design/WARP_RESCUE_LEGACY_SIDE.md:279-287` capped this at ~7% of wall when step_kernel was 90% of the rollout on the 4090 at N=16k. On the 5090 at N=32768 the forward is ~25% of the rollout (14% of wall) — the hideable slice doubled.
- Expected gain: ceiling 14% wall. Realistic 4-8%, only if both halves' env launches overlap each other (512+512 blocks fit in one 1360-slot wave). If they serialize, the N-scaling curve (2x envs = 1.56x time) makes two half launches cost ~1.28x one full launch and the whole thing loses. Gate on §2.
- Risk: not bit-reproducible vs today (cuBLAS may pick different kernels at batch N/2). Run-to-run deterministic. `pop_*` counters merged across two instances (every pop returns a sum). Effort: ~150 lines, one day plus measurement.
- Verify: `--bench`; an nsys timeline (does not need the locked counters that killed ncu on RunPod); then a 50M learning gate.

### B §4. Ragged (CSR) chart pool: ~6 GB VRAM, zero wall change, bit-identical

- Anchors: `cuda_env.py:161-244` (`_charts_to_tensors`, every field padded to (P, C)), `ranked_pool.py:52` (`MAX_OBJECTS = 1536`), `ranked_pool.py:233-247`, kernel index sites `beat_arena_cuda.cu:5015, 5074` (`cb = cc * C + i`), walls at `5031, 5045, 5633`, status stride at `4840, 5016` (`e * C + i`).
- Sizes: per (chart, slot) 10 fp32 lanes + 9 bools = 49 B. With `--chart-pool 999999 --mirror-augment` P ≈ 2 × 46k ≈ 92k segments and C ≈ 1.5k → ~141M slots ≈ 6.9 GB, of which the ~100-note median segment uses ~7%. Likely the single largest tensor in the peak. P unverified statically; the `[ranked-pool] ... MB` line settles it.
- Change: store notes flat with a per-chart `p_off[P]` and read `cb = p_off[cc] + i`; keep `status`/`wall_passed` at their per-env strides. Values, arithmetic, scan order untouched: bit-identical by construction.
- Expected gain: ~−6 GB VRAM, 0 wall (the residency probe showed step time flat across pool sizes 2 MB..7 GB). Value: headroom for N=65536 or larger pools.
- Risk: low; parity suite + a pool-equality test. Effort: half a day (kernel indexing at ~8 sites, pool builder, npz cache key bump).

### B §5. `b_act` as uint8: −213 MB VRAM, free

`train_batch.py:786`: `b_act` int64 (T,N,29) = 243 MB at T=32. gears ≤ 7 fit in uint8. Bit-exact. 3 lines.

### B §6. Collapse the per-update telemetry syncs (hygiene, ~0.02% wall)

`train_batch.py:1284-1360` → `cuda_env.py:787-901`: ~19 host syncs per update after the update, where the queue is already short. A single `torch.stack([...]).tolist()` is cleaner but will not move steps/s.

### B §7. Pack `critic_k`/`critic_v` into one GEMM (~0.1% wall)

`policy.py:536-538` — three separate `_lin` calls over `hv`; k and v read the full tensor twice. The actor blocks already do this. Fold into §2 / Pass A §1.

### B — not worth doing

GAE fusion/parallel scan (<0.5 ms/update); sampling path (~0.1 ms/tick); obs copies/casts (~0.3 ms/tick); Python/launch overhead in the T-loop (refuted by the graph measurement); value-norm EMA; the value head at full width (required by GAE); bootstrap forward reuse (algorithm change for <0.5%); env kernel internals (08-16 do-not-retry list stands).

### B — suspicious but not confirmable statically

Whether `BEAT_ENV_ORDER=1` is in the pod environment; actual pool VRAM (read the log line); whether the 26.72 ms forward figure was taken with `--fp16`; which SDPA backend runs on sm_120 with an additive (B,1,1,27) mask; the env kernel's straggler tail size (nsys timeline of one tick); sector efficiency of the two 64-lane `index_select` gathers; `allow_fp16_reduced_precision_reduction` left at default True (numerics observation).

### B — already well-optimized

fp16 obs buffer with the exact-rounding argument; `b_val` shifted-view GAE with zero allocations; flat fp16 replica mirror with a single refresh kernel; fused-head GEMM; packed qkv; SDPA; fused LN+residual passes; bias+ReLU epilogues; `n_queries=3`; `need_entropy=False`; in-kernel auto-reset; device-side episode-return ring; no host syncs in the tick; dynamics-key env permutation available; telemetry syncs deferred out of the rollout→GAE boundary.

---

## Pass C — PPO algorithmics (full report)

Scope: `train_batch.py`, `policy.py`, the CUDA env's terminal seam (`beat_arena_cuda.cu:5698-5747`), SPEC/README/PERF_AUDIT, and the session logs. Anchors are `train_batch.py:LINE` unless stated.

### C §1.1 Advantage normalization under `--adv-filter` silently re-weights entropy and value loss

**Where.** `a_mu`/`a_sd` at 1166-1167 are computed over the kept minibatch (`p_adv` is already the `keep_idx`-gathered set); `a_mb` at 1194 divides by that. Entropy (1199) and value loss (1198) are not rescaled.

**Mechanism.** The kept set is `{|A| ≥ median|A|}`. For A ~ N(0, σ²), with c = 0.6745σ:
- E[A² | |A|>c] = 1 + c·φ(c)/(1−Φ(c)) = **1.857σ²** → σ_kept = **1.363σ** (Laplace tails give 1.39; robust at ~1.36-1.4).
- E[|A| | |A|>c] = 1.271σ vs unconditional 0.798σ.

Per kept sample the normalized advantage magnitude is 0.933 vs 0.798 unfiltered (+17% per-sample signal), but the PG gradient's scale relative to the entropy and value terms drops by 1/1.363 = **0.73**. Under Adam this is purely a change of direction — equivalent to running the filtered recipe with `ent_coef × 1.36` and `vf_coef × 1.36`. The value term compounds: the critic now trains on the half with the largest residuals (mean |residual| 1.59× the full-batch mean), so the value-gradient share into the shared trunk rises by 1.59/0.73 ≈ **2.2×**. The `[diag]` block runs on unfiltered rows, so it cannot see this.

**Fix.** Normalize by full-batch statistics computed once per update from `f_adv` and use them at 1194 when `adv_filter_on`. Bit-identical when off.

**Probe.** Resume `trunk_upd01000.pt`, 300 updates, two arms differing only in the σ source. Predicted: full-σ arm runs lower `ent` and slightly higher `maxlr`; acc at matched updates is the verdict.

### C §1.2 PPO clip on the joint 29-head ratio is extremely tight per head — clipfrac is not logged

**Where.** `logprob` is the sum over 29 heads (`policy.py:635`); `pg_loss_fn` clips the joint ratio at 1±0.2.

**Mechanism.** The joint ratio leaves the trust window when the geometric-mean per-head ratio exceeds 1.2^(1/29) = **1.0063**. With independent per-head log-ratio perturbations of RMS s, the joint log-ratio has RMS √29·s = 5.4s, so half the batch is outside ±0.18 once s ≈ 0.034. After 64 Adam steps at 7e-4, mid-to-late minibatches likely have a large fraction of adv>0 samples with r > 1.2 and therefore **zero policy gradient**. `maxlr` (2-4) is the max over 1M samples of a 29-head sum and says nothing about the typical sample.

**Fix candidates (contingent on measurement).** (a) Per-head clipping — apply `pg_loss_fn` per head with the same A; standard in multi-discrete implementations but the surrogate is no longer a bound on the joint policy's change. (b) Widen ε for the joint ratio (0.3-0.4).

**Probe.** Step 1: log clipfrac and mean |log r| on the first and last minibatch, 50 updates. Step 2 only if last-mb clipfrac > ~0.25: 500-update A/B, per-head vs joint.

### C §1.3 GAE at T=32: the average credit window is ~10 ticks, not 18

Captured trace mass for a sample at rollout position t = 1−(γλ)^(T−t) with γλ = 0.9457. Half the mass needs T−t ≥ 12.4 ticks; 90% needs ≥ 41.3.

| T | mean captured window (ticks) | samples with ≥50% mass | ≥90% |
|---|---|---|---|
| 8 | 4.0 | 0% | 0% |
| **32** | **10.1** | **62%** | **0%** |
| 64 | 13.5 | 81% | 36% |

A full pre-swing is ~10-15 ticks, so for over a third of the batch the cut reward lands outside the window. The T=8→32 move recovered only 55% of the nominal 1/(1−γλ) window.

**Proposal.** T=64 at N=32768 (`--minibatches 128`, batch 2.1M) with `--adv-filter 0.5`: 64 steps/update on 1M kept rows — identical step count and minibatch size to today's unfiltered recipe. Rescale `--ent-tau` 62, checkpoint cadence, `VALUE_NORM_BETA` 0.01→0.02. `b_obs` doubles to 2.5 GB.

**Probe.** 300 T=64 updates vs 600 T=32 updates from scratch, acc and ep_return at matched samples, falls/1M, maxlr. The ent anneal must be matched in samples.

### C §1.4 Non-issues, checked

Deltas use unnormalized V (consistent to within β = 1%). Time-limit truncation: `max_steps = seg_len·30 + 90 = 840` exceeds the natural end, so the cap essentially never fires; segment ends are genuinely terminal; rollout truncation at T is bootstrapped through `b_val[T]` — correct. Fall/tilt replaces the tick's reward with −fall_const — intentional. `LOG_RATIO_CLAMP 20` is inert (no gradient flows above r = 3 in any branch). Dual-clip c = 3 engages only in the tail. Entropy summed over heads is the right reduction for factored heads. No value clipping — correct. Reward scale is handled by value + advantage normalization. Per-minibatch normalization at n=16384 → σ estimate error 0.55%, equivalent to per-batch. `--mb-accum 2` is gradient-identical to K=1.

### C §2.1 Effective step size: 64 Adam steps at 7e-4 is a hot regime; the drift instrument is wrong

Max drift per update = 64 × 7e-4 = 0.045 vs default-init weight std 0.026-0.051 — drift ceiling ~1-2× initial weight scale per update; realized ~10-50%. The lr ladder put the optimum at 7e-4 with hotter rungs "trained fine but produced worse policies" — the signature of drift. The instrument that would confirm this (mean approx-KL per update, mean |log r| on the last minibatch) is not logged; `r1` measures fp16 precision at the first minibatch only.

### C §2.2 Grad clip at 0.5: whether it binds is computed and discarded

`total_norm` (1270) is never read. Adam is invariant to a constant rescale, so the clip only acts through intermittent binding. Log `clip_coef` mean per update.

### C §2.3 Adaptive-KL lr on the trunk (rl_games-style)

Multiply lr by 1.5 or 1/1.5 per update depending on whether mean KL is below thr/2 or above 2·thr. Self-tunes the trunk lr per phase, never discards rollout data (unlike KL early-stop). Compatible with WSD if the leaf decay is relative to the trunk's live lr. Must be checkpointed. **Do not deploy before §1.2's logging exists** — the threshold has to be set from measured KL under the joint 29-head ratio. Probe: 500 updates from `trunk_upd01000.pt`; abort if lr leaves [3.5e-4, 1.4e-3].

### C §2.4-2.6

Critic lr 3x, vf 0.5 — keep (|g_v| ≪ |g_pi|, gradcos ≈ 0); re-run the diag on `keep_idx` rows when filtering. Warmup — fine. Step-shape probes the filter result motivates: (a) 2 epochs on the top 50% (64 steps; the clip becomes live on epoch 2); (b) §1.3's T=64 + filter. Run (a) only after §1.1's fix.

### C §3 Architecture-level levers

- **§3.1** `trunk = Linear, ReLU, Linear` (`policy.py:372`) then Linear heads: the actor's last nonlinearity is `trunk.0`; `trunk.2` adds no capacity. `dormant` on this signed linear output is structurally ~0 — every log reads `dormant 0.000`. Measure dormancy on `trunk.0`'s ReLU output or FFN hiddens; a ReLU after `trunk.2` is a probe, not a patch.
- **§3.2** Action-head init — not a lever.
- **§3.3** Critic head sound; the filter is the only thing that changes its gradient share. The 5-gear categorical ignores ordinal structure — noted, not a problem.

### C §4 Already good — don't touch

Dual-clip with the log-ratio clamp; value normalization T1; no value clipping; `done` masking of bootstrap and trace; per-minibatch advantage normalization at n=16384; `--mb-accum` as a pure memory lever; 1 epoch × 64 minibatches (density finding stands); WSD with the entropy curve as a function of absolute update index; critic lr split + vf 0.5; fp16 storage with fp32 masters and fp32 PPO math; two-sided |A| filter over signed.

### C §5 Things the code should log but doesn't

1. Mean approx-KL (k3) and clipfrac on the first and last minibatch.
2. Pre-clip gradient norm / `clip_coef` mean.
3. Per-group entropy (spine / armL / armR / legL / legR) and min-head entropy.
4. Full-batch entropy under the filter (the printed `ent` is kept-row entropy).
5. Bootstrap calibration: `b_val[T]` from update u vs realized discounted 32-tick return from u+1 for the same envs.
6. Critic skill score vs a trivial baseline (per-update least-squares on 2-3 obs lanes).
7. Kept-set/full-set advantage σ ratio when filtering.
8. Diag on kept rows when filtering.
9. Per-position advantage variance (by t within the rollout).

---

### Tier 1 results — 1.4 + probe re-run (2026-09-02/03)

Same pod (`194.14.47.19`, RTX 5090), code at `97af37c` (= `f4f6011`
+ 1.4). Two things: item 1.4 shipped and measured, then the 2026-08-31
top-|A| 50% probe was re-run under the fixed normalization against a
matched control on this pod and this code, so the four arms in the
table are two different pods/codes but the two NEW arms differ only
in `--adv-filter 0.5`.

#### 1.4 — full-batch advantage stats under the filter (`97af37c`)

When `adv_filter_on`, `a_mu`/`a_sd` are computed ONCE per update over
the full flat `f_adv` (`adv_filter_stats`) and used in every minibatch;
filter off keeps the per-minibatch stats. The kept/full σ ratio rides
the once-per-update print sync as `afsd` on the `[kl]` line (C §5
item 7); the banner says `full-batch adv norm`. No new host syncs.

- **Filter-off bit-identity:** same-seed 5-update production-shape
  runs before/after, every `update` and `[kl]` line identical to the
  printed digit (upd 5: `acc 0.6480 return -16.910 ent 46.521 ev 0.412
  r1 2.6e-06`, `kl 6.00e-10/1.18e-02 clipfrac 0.0000/0.1884 gnorm 0.368
  clipc 1.000 entfull 46.547`); `ep_return(last100)` differs as it
  always does (ring-buffer sampling). Gate **283** (279 + 4 pins).
- **σ_kept/σ_full, measured:** 0.175 at update 1 (the untrained critic
  makes every advantage a large negative and the top-|A| half is the
  TIGHT cluster — the old normalization scaled that update's PG term
  UP ×5.7, the one update where it went the other way), then 1.343 /
  1.353 / 1.358 / 1.360 over updates 2–5, 1.385 at 10–100, **1.395 at
  300 → 1.404 at 1068**. C §1.1's Gaussian 1.363 is right at the
  start; the batch's tails thicken toward Laplace (1.39) as the policy
  sharpens. The confound was real and its magnitude was predicted.
- **Old-norm vs new-norm at update 5 (`--adv-filter 0.5`, same
  session, same seed):** `gnorm` 0.386 → 0.580, `clipc` 1.000 →
  0.870, last-mb clipfrac 0.123 → 0.138, k3 KL 8.6e-3 → 9.1e-3,
  `ent` 46.476 → 46.467. (The Tier 0 "0.249 under the filter" was a
  1-update-warmup run; the matched pair here uses the campaign's
  `--warmup-updates 34`.)

#### The re-run: recipe and ops

Both arms: the 2026-08-31 argv verbatim (v38 T=32 recipe, seed 0,
`--ckpt-every 125`) to **1068 updates** (`--total-steps 1119879168`,
the update the old probe was paused at) with **`--warmup-updates 34`**
— the old runs' `--total-steps 2359296000` gave warmup = round(0.015 ×
2250) = 34, and warmup is the only thing `--total-steps` feeds on a
trunk (constant lr after it; the ent curve is absolute-indexed), so
this reproduces the old schedule exactly and still exits at 1068. Arm
P (`runs/af50n_trunk`, `--adv-filter 0.5`) ran 22:10–23:41 UTC, Arm C
(`runs/ctrl_trunk`) 23:41–01:33, chained by `/root/af50n.sh` (on the
pod; not committed — it is the af50.sh pattern with the C leg and an
nvidia-smi sampler). Both trunks' 9 checkpoints + logs are on the pod
and in `~/beat-arena-volume-backup/beat-arena-af50n/` (5-min rsync
loop, stopped after the final pull). No leaf, no bench (per brief).

#### Matched-update table

`ctrl-old` = `v38t32_trunk` (`beat-arena-v38t32/logs/t32_trunk.log`),
`probe-old` = the 08-31 probe under the OLD normalization
(`beat-arena-af50/logs/af50_trunk.log`) — both a different pod and
pre-fp16-critic / pre-env-order / pre-uint8 code, and their logs
predate the `[kl]` line, so `entfull`/clipfrac/KL are `–` there and
their `ent` under the filter is KEPT-row entropy. `ctrl-new` = Arm C,
`probe-new` = Arm P. `clipfrac`/`k3` are the last minibatch of the
update; `afsd` = σ_kept/σ_full.

| upd | arm | acc | return | ep_return(last100) | ev | ent | entfull | clipfrac | k3 KL | maxlr | afsd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | ctrl-old | 0.8325 | 2.045 | 32.6 | 0.883 | 38.61 | – | – | – | 5.3 | – |
| 300 | probe-old | 0.8358 | 2.236 | 35.0 | 0.881 | 37.35 | – | – | – | 4.1 | – |
| 300 | **ctrl-new** | 0.8354 | 2.067 | 43.1 | 0.890 | 38.54 | 38.51 | 0.590 | 8.4e-2 | 3.8 | – |
| 300 | **probe-new** | 0.8324 | 1.971 | 28.2 | 0.861 | 36.31 | 36.56 | 0.590 | 8.0e-2 | 3.8 | 1.395 |
| 500 | ctrl-old | 0.8708 | 3.651 | 75.7 | 0.958 | 33.42 | – | – | – | 4.5 | – |
| 500 | probe-old | 0.8751 | 3.903 | 77.5 | 0.957 | 32.35 | – | – | – | 4.8 | – |
| 500 | **ctrl-new** | 0.8752 | 3.895 | 78.3 | 0.961 | 33.75 | 33.71 | 0.566 | 7.7e-2 | 4.7 | – |
| 500 | **probe-new** | 0.8677 | 3.913 | 87.3 | 0.953 | 30.97 | 31.32 | 0.636 | 1.07e-1 | 7.0 | 1.399 |
| 776 | ctrl-old | 0.9009 | 4.883 | 102.3 | 0.983 | 28.14 | – | – | – | 7.5 | – |
| 776 | probe-old | 0.9011 | 4.839 | 101.3 | 0.981 | 28.28 | – | – | – | 3.2 | – |
| 776 | **ctrl-new** | 0.9013 | 4.815 | 97.0 | 0.981 | 28.10 | 28.07 | 0.593 | 8.8e-2 | 6.5 | – |
| 776 | **probe-new** | 0.9053 | 4.856 | 94.6 | 0.981 | 25.86 | 26.00 | 0.606 | 9.5e-2 | 4.0 | 1.403 |
| 1068 | ctrl-old | 0.9137 | 5.137 | 132.1 | 0.987 | 25.07 | – | – | – | 5.3 | – |
| 1068 | probe-old | 0.9161 | 5.233 | 111.4 | 0.989 | 24.08 | – | – | – | 5.8 | – |
| 1068 | **ctrl-new** | 0.9105 | 5.175 | 105.8 | 0.987 | 25.55 | 25.50 | 0.574 | 7.7e-2 | 8.0 | – |
| 1068 | **probe-new** | **0.9210** | 5.244 | 117.8 | 0.989 | 22.14 | 22.37 | 0.618 | 9.8e-2 | 6.7 | 1.404 |

Single-update `acc` has an update-to-update sd of 0.001–0.003, so
the 30-update-window means are the numbers to compare: at 1068,
**probe-new 0.9195 vs ctrl-new 0.9115 (+0.0080)**; at 776, 0.9029 vs
0.8998 (+0.0031); at 500, 0.8664 vs 0.8720 (−0.0056); at 300, 0.8274
vs 0.8322 (−0.0048). `ep_return(last100)` is a ±20% instrument (the
two controls, same recipe and seed, read 132 vs 106 at 1068) and
carries no signal here.

**Wall and VRAM (this pod):**

| arm | 1068 upd wall | s / 100 upd (upd 900→1000, steady) | steps/s (1068-upd total) | rollout / update s (enqueue-attributed) | peak-vram (allocator) | nvidia-smi max |
|---|---|---|---|---|---|---|
| ctrl-new (C) | 6,690 s | 559 | 167,408 | 3,491 / 3,168 | **12.60 GiB** | 24,242 MiB |
| probe-new (P) | **5,493 s** | **456** | **203,864** | 3,423 / 2,039 | **12.17 GiB** | 24,222 MiB |

P is **−17.9% wall** (−18.4% steady-state, +21.8% steps/s) — the
08-31 number (−18% same-pod) reproduces exactly; the update phase
halves (3,168 → 2,039 s; 32 vs 64 optimizer steps, the rest is the
GAE/perm work that does not halve). Peak −0.43 GiB: with 1.2 + 1.3
landed, C's peak is the update-phase 12.60 and P's half-size perm
copies pull that under the 12.17 rollout peak, so P's peak IS the
rollout. Driver-level VRAM (ranked pool 7.6 GB + env kernel) is
identical.

#### Verdict

**Sample efficiency at matched updates: parity through ~700 updates,
then a small lead for the filter.** Per-100-update mean of (P − C)
acc: −0.005 / −0.008 / −0.004 / −0.005 / −0.005 / −0.005 / −0.003
for updates 1–700, then +0.002 / +0.003 / +0.003 / **+0.006** for
701–1068, i.e. the sign flips at ~700 and the lead is widening at the
end (+0.0105 at the 1068 point, +0.0080 on the 30-update window).
The noise floor I can measure: `ctrl-new − ctrl-old` (same recipe
and seed, different pod and code) drifts **within ±0.004 per
100-update window** (mean +0.0016), and the OLD probe's lead over its
control was +0.002–0.003 from update 200 on — inside that floor. So
the old "parity" reading held up as a null, and the new late lead is
~2× the drift floor from ONE seed per arm with a mid-run sign flip:
**call it "at least parity, probably slightly better late", not a
win**. One caveat that matters for "better": P's entropy at 1068 is
3.1 nats below C's (22.4 vs 25.5), so it is also further along the
entropy anneal — part of the late acc lead may be the policy being
sharper at that step rather than a better trunk; the leaf cut is the
instrument for that and was not run. Combined with the −18% wall and
−0.43 GiB, the filter stays in the recipe.

**Did the signatures move as C §1.1 predicted?** Mixed:

- **σ ratio — yes.** 1.36 predicted, 1.34–1.36 measured at updates
  2–5 and 1.39–1.40 in steady state.
- **Entropy — direction yes, framing no.** Removing the implicit
  `ent_coef ×1.36` did lower the probe's entropy (P `entfull` 22.4 vs
  the old probe's `ent` 24.1 at 1068, on a control that reads the
  same, 25.5 vs 25.1), but the brief's expectation that P would sit
  CLOSER to C than the old probe did is the opposite of what
  happened: old probe − old control was −1.0 to −1.3 nats (−1.3 /
  −1.1 / +0.1 / −1.0 at 300/500/776/1068); P − C is **−2.0 / −2.4 /
  −2.1 / −3.1**. The old normalization's extra entropy pressure was
  partly MASKING the filter's own entropy-lowering effect (the kept
  rows are exactly the high-|A| states where the PG gradient is
  strongest); with the PG term at full scale the policy sharpens
  faster than the control on every group (spine 0.55 vs 0.69, legs
  0.69–0.71 vs ~0.85 at 1068).
- **Clipfrac / KL — no.** The prediction was a LOWER last-mb clipfrac
  than the old probe's 0.25 at update 5. At update 5 it is 0.138 vs
  0.123 (old-norm, same session). In the campaign both arms live at
  **0.57–0.64** from update 300 on, and P is 0.00–0.07 ABOVE C with
  k3 KL 8.0e-2 → 9.8e-2 vs C's 7.7e-2 → 8.8e-2 (+15–35%) — P moves
  the policy further per update with half the optimizer steps, more
  so than the old-norm probe did. In hindsight that is what full-scale
  kept advantages must do under Adam: each kept sample carries +17%
  per-sample signal (E|A| 0.93 vs 0.80) and the PG term's share of
  the step rises; nothing in the fix reduces per-step movement.
- **Grad clip (C §2.2, revisited).** Tier 0 read the 0.5 clip as an
  inert warmup guard from `clipc` 1.000 at updates 3–5. Past update
  ~100 it binds on essentially every step in BOTH arms: C `gnorm`
  1.3–1.6 / `clipc` 0.32–0.39, P `gnorm` 3.0–3.6 / `clipc`
  0.14–0.17 (P's gradient is ~2.3× C's, more than the 1.36 × 1.17 the
  normalization alone accounts for). A uniform scale is nearly
  invisible to Adam, so this is not a live regularizer either way,
  but the "inert" reading was an update-≤5 artifact.
- **Critic.** P's `ev` lags C early (0.861 vs 0.890 at 300, 0.953 vs
  0.961 at 500) and matches from 776 on (0.981 / 0.989 vs 0.981 /
  0.987) — the same shape the old probe showed; training the critic
  on the top-|A| half only costs it early calibration.
- `maxlr` carries no signal (3.8/7.0/4.0/6.7 vs 3.8/4.7/6.5/8.0).

**Not done:** leaf cut and bench on either trunk (resumable from
`trunk_upd01000.pt` / `trunk_final.pt` in both run dirs); a second
seed, which is what the +0.008 needs before anyone calls it a win;
C §5 item 8 (diag on kept rows) is still open.

### 2.7 results (measured 2026-09-03)

Same pod (`194.14.47.19`, RTX 5090), same session, same production
argv as Tier 0/1 (`--num-envs 32768 --horizon 32 --minibatches 64
--mb-accum 2 … --chart-pool 999999 --mirror-augment`, env-order
default ON). Baseline RE-MEASURED in this session at the pod's tree
(= laptop HEAD `545590b`, every relevant file md5-matched), not
taken from the Tier 1 table (which it reproduces: 126.0k, 12.60 GiB).
Instruments: `train_batch --bench` (exec-true phase seconds, `[bench-
mem]` per-phase peaks), 5-update same-seed run for bit-identity, full
gate, plus an old-vs-new A/B with the kernel rebuilt each way.

**Change (`cuda_env.py`, `ranked_pool.py`, `beat_arena_cuda.cu`).**
Per-object pool fields are flat — one array per field holding every
segment's objects back to back, `note_off[P+1]` / `wall_off[P+1]`
int64 exclusive prefix sums — and the kernel reads `cb = p_note_off[cc]
+ i` (was `cc * C + i`) at the 4 note sites and `wb = p_wall_off[cc] +
wi` (was `cc * P.W + wi`) at the 3 wall sites. Per-env `status` /
`wall_passed` keep their (N, C) / (N, W) strides with C = W = max
objects over the pool (what the padded width was: C=1489, W=444).
`valid` / `wall_valid` are kept as all-true flat lanes so the kernel's
branch structure is untouched. Built through the unchanged padded
builders (`_charts_to_padded` / `_walls_to_padded`) + `pad_to_ragged`
(mask-select in (p, j) order = CSR order), chunk-stitched by
`cat_pools`; `load_ranked_pool`'s seeded subsample gathers object
ranges (`_subset_ragged`). Cache keys bumped: ranked v5 → v6,
procedural v11 → v12. `pool=` still accepts either layout (a padded
dict is converted at construction), so pool sharing for 2.3 is intact.
Tensor-list contract: +2 slots (`PNOFF`, `PWOFF`), with a
`TORCH_CHECK` that the pool is ragged.

#### Table (2 baseline reps + 2 A/B old reps vs 4 A/B new reps + 2 final reps)

| state | steps/s | rollout s / 4 upd | update s / 4 upd | peak GiB (rollout / update / resident@rollout-start) | `[ranked-pool]` | gate |
|---|---|---|---|---|---|---|
| baseline (padded) | 126,219 / 126,341 / 126,537 / 126,372 — mean **126,367** | 24.42 / 24.37 / 24.34 / 24.38 | 8.75 / 8.77 / 8.74 / 8.75 | **12.60** (12.17 / 12.60 / 8.97) | `93472 segments (C=1489, 7655 MB)` | 283 |
| ragged v1 (base loaded at top of `step_env`) — NOT shipped | 125,712 / 125,613 / 125,667 / 125,393 — mean 125,596 (**−0.6%**) | 24.57 / 24.60 / 24.59 / 24.63 | 8.73 / 8.73 / 8.72 / 8.75 | 5.76 (5.33 / 5.76 / 2.13) | `645 MB` | 293 |
| **ragged final** (base loaded next to `nn`) | 126,396 / 126,307 / 126,399 / 126,285 / 126,244 / 126,361 — mean **126,332 (−0.03%, null)** | 24.34 / 24.37 / 24.39 / 24.39 / 24.38 / 24.39 | 8.77 / 8.77 / 8.73 / 8.76 / 8.78 / 8.74 | **5.76** (5.33 / 5.76 / 2.13) | `93472 segments (C=1489, 13596318 notes + 196250 walls ragged, 645 MB)` | **293** |

- **Pool VRAM 7,655 → 645 MB (−7,010 MB, −91.6%).** 13.6M notes + 196k
  walls over 93,472 segments (mean 145 objects vs C=1489: the padded
  layout was 9.8% occupied on notes, 0.5% on walls). The audit's
  "~0.5–0.7 GB" and "P ≈ 92k" both held.
- **Peak 12.60 → 5.76 GiB (−6.84 GiB)**, every phase moving by the
  same amount (rollout 12.17 → 5.33, update 12.60 → 5.76, resident at
  rollout start 8.97 → 2.13): the pool was the single largest tensor,
  as B §4 guessed, and nothing else changed. At N=32768 the trainer now
  peaks at 5.76 GiB on a 32 GB card.
- **steps/s: NULL.** The first cut cost a real −0.6% (4 reps, all in
  the rollout: 24.4 → 24.6 s, update flat) — ptxas showed 12 B of spill
  stores/loads in `step_kernel<float,true>` (41,504 B stack) where the
  old build had 0 (41,488 B): `nb0 = p_note_off[c0]` loaded at the top
  of `step_env` stayed live across the 8 dynamics substeps in a
  255-register kernel. Loading it next to `nn = p_n_notes[c0]` (the old
  layout's first pool touch) restored 41,488 B / 0 spill and 126.3k on
  6 reps. So the residency probe's "step time flat across pool sizes"
  held — the layout is free — but a live int64 across the physics
  section is not. The `write_obs` bases (`nb`, `wbase`) sit next to
  the existing `nn` / `msp` loads and did not need moving.
- **Bit-identity: PASS.** 5-update same-seed run (`runs/r27_base_five`
  vs `runs/r27_final_five`): all 10 `update` + `[kl]` lines identical
  in every printed digit (`acc accL accR return ent ev r1 maxlr`,
  `kl clipfrac gnorm clipc ent/head min entfull`). The only differing
  column is `ep_return(last100)`, which is scheduling noise between
  identical runs (the Tier-1 `t1_s11` log, same code as this
  baseline, prints −6.510 at update 5 where the baseline printed
  −6.558 with every other column equal — the ring-buffer pop order,
  Tier 0.1 note).
- **Pool equality: PASS, two ways.** (a) `tests/test_ragged_pool.py`
  (7 tests): every per-object field equal per segment per object
  (notes and walls, empty segments included), offsets consistent,
  C/W = the padded width, `ragged_to_padded` round-trips exactly, a
  padded dict handed to `pad_to_ragged` equals the native ragged
  build, the subsample gather equals a from-scratch build of the
  chosen segments, and on CUDA an env fed the padded dict and one fed
  the ragged dict step bitwise-identically for 120 ticks (obs / reward
  / done / status / wall_passed / counters). (b) On the pod, the OLD
  8.0 GB v5 npz converted by `pad_to_ragged` vs the fresh v6 build
  from the pickles: all 32 fields equal on `tobytes()`. (The new
  ragged npz is 677 MB and built in 54 s; the padded one was 8.0 GB.)
  `test_ranked_pool_chunked_pack.py` now pins both the padded chunk
  contract and `cat_pools` (3 more parametrized cases).
- **Rebuild confirmed.** `jit_cache` srchash `263a1b4f…` (.so of
  09-02 19:50) → `246f75df…` (v1, 03:56:22) → `d3788ca2…` (final,
  04:12:25); the final gate (293 passed, 04:15–04:17) and the final
  five/bench runs print `d3788ca2…` at start and end, i.e. ran against
  the new .so. `fast_load` picked up the .cu edit on the first import
  each time (ninja build ~2.5 min).
- Gate: **293 passed** (283 + 10 new).

*Not done / notes:* the v5 padded npz (`ranked_e52ac58c96855bee.npz`,
8.0 GB) is left in the pod's shared `pool_cache` (not this item's to
delete); `MAX_OBJECTS` stays 1536 and the (N, C) status block stays
48.8 MB at N=32768 — a follow-up could shrink C by capping the
densest segments, but there is nothing left to gain from the pool
itself. N=65536 / larger pools / `--mb-accum 1` now have ~20 GB of
headroom.

### 2.2 results (measured 2026-09-03)

Same pod (`194.14.47.19`, RTX 5090, 170 SMs, torch 2.8.0.dev+cu128),
same session, same production argv as 2.7 (`--num-envs 32768 --horizon
32 --minibatches 64 --mb-accum 2 … --chart-pool 999999
--mirror-augment`, env-order default ON). Baseline RE-MEASURED in this
session at the pod's tree (= laptop HEAD `f2efa71`, every relevant file
md5-matched); it reproduces the 2.7 table (126.3k, 5.76 GiB, identical
5-update prints). Instruments: `train_batch --bench` (exec-true phase
seconds, `[bench-mem]` peaks), 5-update same-seed run for bit-identity,
full gate, `bench_policy_forward --fp16 --ns 32768`, and — the point of
the item — an `nsys profile -t cuda` timeline of ONE update per arm
(`nsys` 2024.6.2 apt-installed on the pod; `cuda_gpu_trace` CSV, an
interval-intersection script over stream ids).

**Change (`policy.py`, `train_batch.py`, `tests/test_critic_stream.py`,
flag `--critic-stream` / `--no-critic-stream`, default OFF).**
`forward_stacked(critic_async=True)` under no_grad records an event on
the current stream right after the trunk readout `h` (so before the
heads GEMM — the critic's inputs `x_res`, `h`, `attn_bias` all exist),
a dedicated `torch.cuda.Stream` waits on it and runs `_critic` there,
a second event is recorded on the side stream, and the value is
returned unjoined; the trainer enqueues `venv.step(a)` without waiting
and calls `policy.critic_join()` (a `wait_event` on the main stream, no
host sync) right before `b_val[t]` is written — which moved from before
the env launch to after it (same copy, one launch later, no other
reader until GAE). Allocator semantics: the three main-pool inputs are
HELD on `policy._critic_pending` until the join (a Python free before
it would let the caching allocator hand their blocks to the env step
while the side stream still reads them); the side-pool value gets
`record_stream(main)`. The update-phase forward (grad enabled) is
untouched and ignores `critic_async`; a missed join raises on the next
forward. `BEAT_CRITIC_LATE=1` is the 2.3 probe (below): the side-stream
launch is deferred to `critic_launch_deferred()`, which the trainer
calls after sampling, right before the env launch, so the critic and
`step_kernel` become eligible at the same instant. Gate **304 passed**
(293 + 11), run on the pod against the final tree.

#### Table (3 baseline reps vs 3 ON reps + 2 OFF reps; the probe's 2+2)

| state | steps/s | rollout s / 4 upd | update s / 4 upd | peak GiB (rollout / update / resident@start) | BPF ms @32768 |
|---|---|---|---|---|---|
| baseline (`f2efa71`) | 126,344 / 126,185 / 126,365 — mean **126,298** | 24.38 / 24.41 / 24.40 | 8.75 / 8.77 / 8.73 | 5.76 (5.33 / 5.76 / 2.13) | 18.39 / 18.49 |
| `--critic-stream` ON | 125,921 / 125,754 / 126,021 — mean **125,899 (−0.32%)** | 24.43 / 24.47 / 24.39 (+0.1%) | 8.82 / 8.81 / 8.83 (**+0.8%**) | 5.77 (5.33 / 5.77 / 2.14) | — |
| `--no-critic-stream` OFF (new code) | 126,290 / 126,309 | 24.41 / 24.40 | 8.74 / 8.75 | 5.76 (5.33 / 5.76 / 2.13) | — |
| probe v1 (`BEAT_CRITIC_LATE=1`, event after the env launch) | 125,893 / 125,923 | 24.40 / 24.40 | 8.85 / 8.84 | 5.77 | — |
| probe v2 (`BEAT_CRITIC_LATE=1`, event before the env launch) | 125,780 / 126,027 | 24.46 / 24.42 | 8.82 / 8.80 | 5.77 | — |

- **steps/s: NULL to slightly negative.** Rollout seconds — the sharp
  instrument, the change touches nothing else — move +0.03 s on 24.4
  (+0.1%, inside the 0.4–0.7% phase spread). The −0.3% on steps/s is
  the update phase: +0.07 s / 4 updates (+0.8% of the phase) on all 5
  side-stream reps vs all 5 inline reps (8.80–8.85 vs 8.73–8.77).
  The update never touches the side stream; the likely cause is the
  second allocator pool (the critic's ~0.7 GB of intermediates now
  live in a side-stream pool the update's main-stream allocations
  cannot reuse — peak 5.76 → 5.77), not chased. **Default OFF**; the
  OFF arm reproduces the baseline (126,290 / 126,309, 24.41 / 24.40 s)
  so the moved `b_val[t]` write is free.
- **Bit-identity: PASS**, three ways. ON vs baseline 5-update runs
  (`runs/s22_on_five` vs `runs/s22_base_five`): all 10 `update` +
  `[kl]` lines identical in every printed digit (only
  `ep_return(last100)` differs — the ring-buffer pop order, as in
  every previous item). The probe-v1 5-update run: identical too.
  `tests/test_critic_stream.py` (11 tests): `torch.equal` on logits,
  value, sampled actions and logprobs between the side-stream and
  inline paths on both dtype streams, flat and non-flat; the side
  stream is actually used and the inputs are held until the join; the
  grad-enabled forward ignores `critic_async`; a missed join raises;
  24 ticks of main-stream allocate/fill/free churn between the async
  forward and the join cannot corrupt the value; the deferred probe
  path is grad-free and bit-identical.
- **Ops note.** Probe v1's first cut called `critic_launch_deferred()`
  outside the trainer's no_grad block: the critic built an autograd
  graph, `b_val[t].copy_(v)` chained every tick's graph onto `b_val`
  through CopySlices, and the first rollout OOM'd at 20 GiB by tick
  ~20. The method now enforces `no_grad` itself and `_critic_enqueue`
  asserts it.

#### Timeline evidence (nsys, 1 update per arm, N=32768)

Where the tick goes NOW (the B §0 table was derived from a 5.96 s
update and is stale by ~1.7×): **tick 183.2 ms** = `step_kernel`
164.4 ms (90%; 1024 × 32 threads, 255 regs, 0 smem; excl. the 1.9 ms
reset launch; first post-reset tick 81.6 ms, steady 165–177) + 18.9 ms
forward + sampling + env-step prep (10%). So the hideable slice for a
two-chain design is ~10% of the rollout ≈ 7.5% of wall, not the 14%
B §3 assumed.

| arm | side-stream kernels | overlap with `step_kernel` | overlap with other main-stream kernels | where the critic sits | inter-step gap (fwd + tail) | tick |
|---|---|---|---|---|---|---|
| OFF | — (all 4,713 ops on stream 7) | — | — | inline, in the forward | 18.87 ms | 183.19 ms |
| ON | 480 ops on stream 13 (15/tick), 64.4 ms busy over the 5.86 s rollout window | **0.00 ms, 32/32 ticks** | 5.66 ms total = **0.18 ms/tick** | starts 2.30 ms before the env launch, span 2.14 ms, **ends 0.15 ms before `step_kernel` starts** | 18.78 ms | 183.45 ms |
| probe v1 (event after the env launch) | same | 0.00 ms | 0.00 ms | starts **+0.000 ms after each step end** — an event recorded after `step_kernel` completes when it does; a probe of event semantics, not of the scheduler; discarded | — | 183.31 ms |
| probe v2 (critic and `step_kernel` eligible at the same instant) | same | **0.000 ms, 31/31 ticks** | 3.97 ms | **critic first**: starts −2.08 ms, span 2.07 ms, `step_kernel` starts 0.01 ms after the critic's last kernel | 18.70 ms | 183.06 ms |

Side-stream kernel census (ON arm, per rollout; regs/thread, grid ×
block, dyn smem): `fmha_cutlassF_f16_aligned_64x64_rf_sm80` 31 × =
30.5 ms (118 regs, 1×4×32768 CTAs × 32×4 threads, 19 KB) — the Tq=1
SDPA pool is HALF the critic; `cutlass_80_tensorop_f16_s16816gemm_relu
_f16_64x64_32x6_tn_align8` 93 × = 17.6 ms (88 regs, 1024 × 128, 49 KB:
critic_ln→q/kv and the value MLP); `ln_fwd_h_kernel` 31 × = 10.5 ms
(38 regs, 1024 × 256); `…wmma…32x32_128x1` 1.5 ms (96 regs, 8192 ×
128); `…256x128_32x3` 1.4 ms (232 regs, 256 × 256, 74 KB); `cat` 1.3;
`bias_relu` 0.6; `gemvx` 0.5 (216 regs, 8192 × 32); copies 0.4.

Reading: (1) in the shipped arm the critic is not on the env kernel's
critical path at all — it runs BEFORE the env launch, entirely within
the actor's own post-trunk tail (heads GEMM, log_softmax,
exponential_, argmax, gather, randint, the env-order argsort, the
b_obs cast), and that tail is throughput-bound: 2.1 ms of critic
moved off the main stream bought 0.09 ms/tick (18.87 → 18.78), i.e.
the two streams interleaved at kernel boundaries rather than
co-executed (0.18 ms/tick of measured concurrency). (2) When the
critic and `step_kernel` were handed to the hardware at the same
instant (probe v2) it ran the critic grid to completion and only then
dispatched the env wave — zero co-residency in 31/31 ticks, and no
fill of the env kernel's straggler tail either (the critic was long
gone). Across the three timelines: 0.00 ms of critic-beside-env
execution in 95/95 ticks.

#### Occupancy arithmetic (why the hardware cannot co-schedule here)

`step_kernel` at N=32768: 1024 blocks × 1 warp, `__launch_bounds__
(32, 1)` → 255 regs/thread → 8,192 regs per warp (256 × 32 after
allocation rounding). Register cap: 65,536 / 8,192 = **8 warps/SM** →
1,360 slots ≥ 1,024 blocks: the whole grid is ONE wave, 6 blocks/SM on
166 SMs and 7 on 4 SMs, using 6 of 48 warp slots and 49,152 of 65,536
registers (16,384 free), 0 of ~100 KB smem. By the SM-level count a
critic CTA would fit: the 64×64 GEMM needs 4 warps × 3,072 = 12,288
regs (+49 KB smem), the fmha 4 × 3,840 = 15,360, `ln_fwd` 8 × 1,280 =
10,240 — only the 256×128 GEMM (8 × 7,424 = 59,392) does not. But the
register file is physically four 16,384-register sub-partitions, a warp
is pinned to one, and a CTA's warps are spread one per sub-partition:
six 8,192-register env warps land 2/2/1/1, so **two sub-partitions are
full** (three on the 7-block SMs) and no 4- or 8-warp CTA can be placed
until env blocks retire. The only critic kernel with 1-warp CTAs is the
0.5 ms `gemvx`. That matches the timeline exactly: nothing co-resides
with the resident env wave, and the hardware's choice at a tie is to
finish one grid before starting the other.

#### 2.3 verdict

**NO-GO for 2.3 (two-chain rollout).** Its mechanism — env(A) executing
while policy(B)'s GEMM/attention CTAs co-reside on the same SMs — does
not exist on this GPU at this kernel's register footprint: measured
0.00 ms of overlap in every configuration (ON, probe v1, probe v2;
95/95 ticks), with the register-file geometry above as the reason
(6–7 resident 255-register warps fill 2–3 of the 4 sub-partitions per
SM). Two chains could at best fill the env kernel's straggler tail,
which none of the timelines shows being filled, and B §3's own
N-scaling penalty (two N/2 launches ≈ 1.28 × one N launch if they
serialize — and everything here serializes) makes it a likely net loss
on top of the ~150-line, non-bit-reproducible rewrite. The hideable
slice is also smaller than assumed: the forward + tail is 10% of the
183 ms tick (≈ 7.5% of wall), not 14%. The only route to hiding the
forward would be to change the env kernel's residency (fewer registers
per env thread, or a warp-per-env layout freeing sub-partitions) —
that is the retired candidate-B / BEAT_MIN_BLOCKS territory, not a
trainer-side stream change.

*Kept:* `--critic-stream` (default OFF, bit-identical, −0.3%) and the
`BEAT_CRITIC_LATE` probe as the record of the experiment; the `b_val[t]`
write now sits after the env launch in both modes (measured free).
*Not done:* attributing the +0.8% update-phase cost of the second
allocator pool (irrelevant with the flag off); a tail-fill probe would
need the critic eligible mid-kernel (a device-side spin, not an event)
and is not worth building given the sub-partition arithmetic.

### 2.5 results (measured 2026-09-03)

Same pod (`194.14.47.19`, RTX 5090 sm_120, torch 2.8.0.dev20250319+cu128,
cuBLAS 12.8), same session, same production argv as 2.2/2.7
(`--num-envs 32768 --horizon 32 --minibatches 64 --mb-accum 2 …`,
env-order default ON, critic-stream OFF). Baseline RE-MEASURED in this
session at the pod's tree (= laptop HEAD `193b381`, every relevant file
md5-matched); it reproduces the 2.2 table (126.3k, 5.76 GiB, identical
5-update prints). Instruments: `train_batch --bench` ×3 (exec-true
phase seconds, `[bench-mem]` peaks), `bench_policy_update --fp16 --mbs
8192 16384 --mb-accum 2` ×2, `bench_policy_forward --fp16 --ns 32768`
×2, a 5-update same-seed run per arm, full gate under both toggle
values, `perf_numerics` dump/compare (random init AND a trained
checkpoint), a shape probe with `torch.profiler` kernel names, and a
profiled production minibatch (mb 16384, accum 2) in both modes.

**Change (`fp16_linear.py`, `policy.py` comment, `tests/test_lt_epilogue.py`,
toggle `BEAT_LT_EPILOGUE`, default ON).** `_Fp16LinearReLU.forward`
and the no-grad `fp16_linear_relu` fast path call `_relu_forward`,
which runs `torch._addmm_activation(b16, x2, w16.t())` — cublasLt's
`RELU_BIAS` epilogue, bias = the flat fp16 replica view — when
`_lt_epilogue_ok` holds (contiguous row-major x/w, contiguous 1-D
bias, 16-B-aligned pointers; every flat replica view is 32-B aligned
by `flatten_fp16_replicas`' ALIGN=16-element contract), else the old
two-pass bias-free `torch.mm` + `bias_relu_(y, b_fp32)`. Backward
untouched (`relu_bwd` / `wgrad` from the saved post-ReLU `y`).
`BEAT_LT_EPILOGUE=0` is the bisect hatch. The sm_61 history (round-4
#2's 33–59% addmm-epilogue cost, the 08-16 width fix, the never-run Lt
re-probe) is written into the toggle's comment so it is not re-run.

#### Table (3 baseline reps vs 3 ON reps + 2 OFF reps)

| state | steps/s | rollout s / 4 upd | update s / 4 upd | peak GiB (rollout / update / resident@start) | BPF ms @32768 | BPU ms 8192 / 16384 (accum 2) | gate |
|---|---|---|---|---|---|---|---|
| baseline (`193b381`) | 126,146 / 126,270 / 126,439 — mean **126,285** | 24.44 / 24.40 / 24.36 | 8.74 / 8.75 / 8.75 | 5.76 (5.33 / 5.76 / 2.13) | 18.42 / 18.50 | 24.81, 22.77 / 32.12, 32.14 | 304 |
| **`BEAT_LT_EPILOGUE=1` (new default)** | 129,193 / 129,260 / 128,968 — mean **129,140 (+2.3%)** | **24.01 / 23.98 / 24.04 (−1.5%)** | **8.40 / 8.40 / 8.42 (−4.0%)** | 5.76 (5.33 / 5.76 / 2.13) | **15.88 / 16.05 (−13.5%)** | 21.67, 22.06 / **30.90, 30.91 (−3.8%)** | **321** (304 + 17) |
| `BEAT_LT_EPILOGUE=0` (new code) | 126,333 / 126,122 | 24.38 / 24.38 | 8.76 / 8.81 | 5.76 (5.33 / 5.76 / 2.13) | — | — | 321 |

- **steps/s +2.3% (126.3k → 129.1k), every rep of the ON arm above
  every rep of both OFF arms** (spread 0.2%; the Tier-0 run-to-run
  figure is 0.31%). The update phase gives −0.34 s / 4 upd (−4.0%,
  the audit's 3–5% ceiling), the rollout −0.38 s (−1.5%): the
  rollout forward runs the same nine ReLU sites at N=32768, where
  the three ffn.0 `bias_relu_` passes over (32768·27, 512) were ~2.5 ms
  of the 18.4 ms forward (BPF −13.5%). VRAM unchanged (the fused
  output is the same tensor the two-pass path wrote in place).
- **OFF reproduces the baseline** (126,333 / 126,122; 24.38 s; 8.76 /
  8.81 s) and its 5-update run is bit-identical to the baseline's in
  every printed `update` + `[kl]` digit — the toggle is the only
  difference.

#### Kernel evidence (`torch.profiler`, CUDA activities)

*Shape probe, isolated ffn.0 site (M = 8192·27, K = 128, N = 512),
fp16, 100-iter event timing:* two-pass **508.7 µs** (= GEMM 235 µs +
`bias_relu_kernel_p2<256>` 288 µs) vs fused **224.4 µs** vs bare
`torch.mm` 217.9 µs. The fused call is ONE kernel,
`cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_tn_align8`
(238 µs) — the same cutlass kernel cuBLAS already ran for the
bias-free mm (235 µs), now with its epilogue switched on: **the Lt
RELU_BIAS kernel runs at plain-GEMM speed on sm_120**, which is the
gate the item was conditioned on. At M = 16384·27: two-pass 1037 µs
vs fused 443 µs vs mm-only 725 µs (the plain-mm heuristic picks a
worse algo at that M; Lt does not). trunk.0 / critic_mlp.0 (8192,
384 → 256): 35.5 → 21.0 µs (Lt picks a `64x256_32x4` variant). The
odd-K encoder first layers (K = 18 / 34 / 78): also one `align2`
cutlass kernel each, 0.55–0.64× the two-pass time, so the gate does
not exclude them. A bias view offset by ONE fp16 element (2 B) still
fuses but drops to a `cutlass_75 … align1` kernel at 394 µs (1.8×), so
`_lt_epilogue_ok` requires 16-B pointer alignment (satisfied by every
flat view; pinned by the test).

*Production minibatch (mb 16384, accum 2, fp16 flat, one fwd+bwd+clip+
Adam), two profiles per mode in one process:* OFF **1021 kernels /
30.97 ms** GPU; ON **1003 kernels / 29.70 ms (−1.27 ms, −4.1%)**. The
18 missing launches are the nine ReLU sites × 2 micro: `bias_relu_
kernel_p2` 18 launches / **1.24 ms → 0**. The ffn.0 GEMMs are the same
`…s16816gemm_relu_f16_64x64_32x6_tn_align8` kernel at 176–222 µs in
both modes (the 66 "gemm_relu"-named GEMMs total 5.68 vs 5.65 ms — the
name is cuBLAS's epilogue-capable kernel, used for plain mm too);
no `relu_`/`addmm` fallback kernels appear. Everything else in the
top-14 (fmha fwd/bwd, the nt/nn GEMMs, reduce, cat, `relu_bwd`,
`ln_bwd`) is identical to the launch and to ±0.02 ms.

#### Numerics gate

Class: fp16 last-rounding, NOT bit-identical (one rounding instead of
two; bias as fp16). On the ffn.0 probe: max|two-pass − fused| 1.95e-3
at output scale ~3 (6.6e-4 rel); vs an fp32 reference the fused result
is the CLOSER one (max 1.28e-3 vs 2.12e-3 for the two-pass path).

- (a) Gate: **321 passed** under the default (= ON) and **321** under
  `BEAT_LT_EPILOGUE=0`, both on the pod against the final tree.
- (b) `perf_numerics compare` (production fp16 flat config, 4096-row
  synthetic minibatch). *Trained weights* (`runs/ctrl_trunk/trunk_
  final.pt`): **PASS, worst_rel 1.75e-3** — logits max-diff 1.25e-1 on
  168 (7.4e-4 rel), value 9.8e-3 on 7.15 (1.4e-3), every one of the 22
  grad probes ≤ 1.75e-3 rel. *Random init*: logits 4.9e-4 on 0.377
  (1.3e-3), value 1.7e-3, actor/critic grads 4e-3–4e-2 rel, but the
  scale-relative verdict is a formal **FAIL (worst_rel 0.128)** on
  `critic_q`/`critic_k` — tensors whose max|grad| is 1.0e-5 / 9.8e-6
  (abs diff 1.3e-6 / 1.2e-6). That is the init noise floor 1.1's
  isolation probe documented (|dW_k|max ≈ 7e-6 at init, cos 0.3–0.4
  vs fp32 for EITHER fp16 arm); the trained-weight run is the one that
  measures routing, and it passes.
- (c) 5-update same-seed run, ON vs baseline (`runs/s25_on_five` vs
  `runs/s25_base_five`), update 5: `acc 0.6555` vs 0.6584 (−0.003,
  inside the ±0.004 single-seed drift floor from the 1.4 re-run),
  `return −15.580` vs −15.607, `ent 46.486` vs 46.484, **`ev 0.502` vs
  0.502**, `kl(mbN) 1.34e-2` vs 1.31e-2, `clipfrac 0.213` vs 0.208.
  Update 1 differs already (`acc 0.6429` vs 0.6432): the rollout
  forward's numerics changed, so a few rounding-boundary samples
  differ from the first tick.
  **`r1` moved — to exactly 0.0 (from 1.0e-6–2.4e-6) at every update,
  and `kl(mb1)` 0.00e+00 (from ~1e-10).** `r1` is the mean |ratio − 1|
  on the first minibatch, i.e. the rollout-vs-update logprob
  consistency. It did not break; it became exact: the probe shows the
  plain-mm heuristic choosing DIFFERENT cutlass kernels at the rollout
  M (32768·27) and the update M (8192·27), and the Lt heuristic
  choosing the same one, so the two forwards now agree bit-for-bit on
  the ReLU sites — the ~1e-6 residual was algorithm choice by M, not
  fp16 itself. (The non-ReLU GEMMs evidently already agreed.)
- (d) `tests/test_lt_epilogue.py` (17 tests): the gate per production
  site and its rejection of a 2-B-offset bias / strided x; fused vs
  two-pass within 2e-3 rel and both within 2e-3 of fp32 at every site
  (ReLU-zero disagreements only where the fp32 pre-activation is
  < 2e-3); the backward against a same-mask reference in both modes
  (dx 2e-3, dW/db 1e-4); whole-policy logits/value within 2e-2 and
  NOT bit-identical (the toggle is live); on sm_80+ the fused call is
  one `gemm` kernel with no `bias_relu` for ffn.0 and for the K=18
  encoder shape, while a gate-rejected bias view still takes the
  two-pass pair.

#### Default: ON. Why

Every instrument agrees and none is inside noise: +2.3% steps/s on 3
vs 5 reps with zero overlap, −4.0% update phase, −1.5% rollout, −3.8%
BPU, −13.5% BPF, one fewer full-tensor pass per ReLU site with the
kernel evidence that the epilogue is free on this GPU. Numerics are
the accepted fp16 last-rounding class, closer to fp32 than before, the
trained-weight routing gate passes at 1.75e-3, EV/acc/return are at
parity in the 5-update run, and the rollout/update consistency
diagnostic improved. The sm_61 finding that motivated the split is
recorded in the toggle comment and is not contradicted — it was a
Pascal-era cuBLAS epilogue cost that sm_120 + cuBLAS 12.8 does not
have.

*Not done / notes:* `bias_relu_` and its `BEAT_GEMM_LEGACY` dispatch
stay in `fp16_gemm.cu` (they are the OFF path and the non-16-B-aligned
fallback); the 5090-only default has not been re-validated on the
4090 pod (bench results are not comparable across pods anyway — the
kernel-name probe is the 30-second check to run there). The
random-init `perf_numerics` verdict is scale-relative per tensor and
will keep flagging the ~1e-5 critic q/k grads for any fp16-class
change; a trained checkpoint should be the documented input for that
gate from now on.

### 2.6 T=64 probes (2026-09-03)

Two from-scratch T=64 arms on the re-bootstrapped pod (`194.14.47.19`,
port 22705, RTX 5090, code = `0fdf055` + `a12c491`), compared at
matched SAMPLES against the two T=32 runs of the 1.4 re-run
(`ctrl_trunk`, `af50n_trunk`, previous container of the same pod,
code `97af37c` — pre-2.7/2.2/2.5, so their VRAM is not comparable and
their steps/s only roughly). Verdict first: **T=64 at matched samples
LOSES to the T=32 filtered run at every point (−0.012 to −0.023 acc,
3–5× the ±0.004 floor, one seed), and γ=0.98 loses further (−0.03 vs
af50n) with a 4× fall plateau at 0.2–0.4 B samples.** The
credit-window claim (C §1.3) does not survive its first test in this
design; the design confounds T with the number of policy iterations
(534 vs 1068), see "what this does and does not say".

#### Recipe and ops

`--value-norm-beta` flag added (`a12c491`, default 0.01 = the old
module constant, bit-identical, 2 pins; pod gate **336** = 321 + 13
`test_metrics_log` + 2). Base argv = `af50n_trunk`'s verbatim
(v38 T=32 recipe, `--adv-filter 0.5`, seed 0) re-shaped to T=64:
`--horizon 64 --minibatches 128` (batch 2,097,152, minibatch 16,384;
filter keeps 1,048,576 rows → **64 optimizer steps/update**, the
unfiltered T=32 recipe's count), `--mb-accum 2`, and the
update-indexed schedules halved because each update holds 2× the
samples: `--ent-tau 62 --warmup-updates 17 --ckpt-every 63
--value-norm-beta 0.02`; `--total-steps 1119879168` = 534 updates =
the 1068-update T=32 runs' 1.12 B samples. Arm A `--gamma 0.965
--run-dir runs/t64_g965`, Arm B `--gamma 0.98 --run-dir runs/t64_g98`,
chained by `/root/t64.sh` (A 18:24–19:55 UTC, B 19:55–21:26). The ent
COEFFICIENT is matched in samples by construction (T=64's exponent
is −u/62 = −2u/124 vs T=32's −2u/125; ratio 0.986–1.000 at every
comparison point). Both trunks' 9 checkpoints + `metrics.jsonl` +
logs are on the pod and in `~/beat-arena-volume-backup/beat-arena-t64/`
(`trunk_final.pt` md5 verified both sides; 5-min rsync loop stopped
after the final pull). No leaf cut, no bench. Pre-flight: a 5-update
T=64 smoke (13.8 s/upd, peak 7.00 GiB, banner `warmup 17`, `tau=62`).
Side confirmation for `0fdf055`: `metrics.jsonl` appeared next to
`args.json` in both run dirs.

#### Matched-sample table

Comparison points: T=32 updates 300 / 500 / 776 / 1068 = T=64
updates 150 / 250 / 388 / 534. "Window mean" = 30 T=32 updates = 15
T=64 updates = 31.5 M samples (within-window acc sd 0.0006–0.0015).
`return` is `b_rew.sum()/N` (per-env reward summed over the T-tick
rollout), so the T=64 values are shown ÷2 as "return / tick ×32".
`falls / 1 M steps` = `[fall]` count ÷ batch, over the window.
`clipfrac`/`k3` are the last minibatch of the update; `afsd` =
σ_kept/σ_full.

| samples | metric | ctrl_trunk | af50n_trunk | T=64 γ=0.965 | T=64 γ=0.98 |
|---|---|---|---|---|---|
| **0.31 B** | update | 300 | 300 | 150 | 150 |
|  | acc (point) | 0.8354 | 0.8324 | 0.8185 | 0.8112 |
|  | acc (31.5 M-sample window mean) | **0.8322** | **0.8274** | **0.8167** | **0.8075** |
|  | return / tick ×32 (per-env) | 2.067 | 1.971 | 1.163 | -1.517 |
|  | ep_return(last100) | 43.1 | 28.2 | 19.8 | -3.4 |
|  | ev | 0.890 | 0.861 | 0.833 | 0.823 |
|  | entfull | 38.51 | 36.55 | 38.35 | 38.35 |
|  | last-mb clipfrac | 0.590 | 0.590 | 0.577 | 0.519 |
|  | last-mb k3 KL | 8.4e-02 | 8.0e-02 | 7.5e-02 | 5.6e-02 |
|  | gnorm / clipc | 1.30 / 0.393 | 3.23 / 0.161 | 3.12 / 0.165 | 3.23 / 0.162 |
|  | maxlr | 3.8 | 3.8 | 3.0 | 4.3 |
|  | σ ratio (afsd) | – | 1.395 | 1.393 | 1.384 |
|  | falls / 1 M steps (window) | 259 | 377 | 868 | 8321 |
| **0.52 B** | update | 500 | 500 | 250 | 250 |
|  | acc (point) | 0.8752 | 0.8677 | 0.8536 | 0.8416 |
|  | acc (31.5 M-sample window mean) | **0.8720** | **0.8664** | **0.8496** | **0.8383** |
|  | return / tick ×32 (per-env) | 3.895 | 3.913 | 3.390 | 2.783 |
|  | ep_return(last100) | 78.3 | 87.3 | 70.3 | 52.5 |
|  | ev | 0.961 | 0.953 | 0.924 | 0.919 |
|  | entfull | 33.71 | 31.32 | 34.60 | 34.78 |
|  | last-mb clipfrac | 0.566 | 0.636 | 0.593 | 0.574 |
|  | last-mb k3 KL | 7.7e-02 | 1.1e-01 | 8.2e-02 | 7.3e-02 |
|  | gnorm / clipc | 1.64 / 0.323 | 3.20 / 0.166 | 3.24 / 0.164 | 3.07 / 0.168 |
|  | maxlr | 4.7 | 7.0 | 4.3 | 4.2 |
|  | σ ratio (afsd) | – | 1.399 | 1.400 | 1.397 |
|  | falls / 1 M steps (window) | 12 | 40 | 82 | 124 |
| **0.81 B** | update | 776 | 776 | 388 | 388 |
|  | acc (point) | 0.9013 | 0.9053 | 0.8859 | 0.8719 |
|  | acc (31.5 M-sample window mean) | **0.8998** | **0.9029** | **0.8845** | **0.8709** |
|  | return / tick ×32 (per-env) | 4.815 | 4.856 | 4.457 | 3.849 |
|  | ep_return(last100) | 97.0 | 94.6 | 91.2 | 85.4 |
|  | ev | 0.981 | 0.981 | 0.968 | 0.955 |
|  | entfull | 28.07 | 26.00 | 30.11 | 31.48 |
|  | last-mb clipfrac | 0.593 | 0.606 | 0.571 | 0.583 |
|  | last-mb k3 KL | 8.8e-02 | 9.5e-02 | 7.5e-02 | 8.1e-02 |
|  | gnorm / clipc | 1.52 / 0.342 | 3.57 / 0.144 | 3.71 / 0.144 | 2.99 / 0.174 |
|  | maxlr | 6.5 | 4.0 | 3.9 | 4.7 |
|  | σ ratio (afsd) | – | 1.403 | 1.402 | 1.397 |
|  | falls / 1 M steps (window) | 15 | 9 | 22 | 34 |
| **1.12 B** | update | 1068 | 1068 | 534 | 534 |
|  | acc (point) | 0.9105 | 0.9210 | 0.9076 | 0.8849 |
|  | acc (31.5 M-sample window mean) | **0.9115** | **0.9195** | **0.9049** | **0.8861** |
|  | return / tick ×32 (per-env) | 5.175 | 5.244 | 5.127 | 4.527 |
|  | ep_return(last100) | 105.8 | 117.8 | 112.1 | 92.9 |
|  | ev | 0.987 | 0.989 | 0.984 | 0.972 |
|  | entfull | 25.50 | 22.37 | 25.93 | 28.73 |
|  | last-mb clipfrac | 0.574 | 0.618 | 0.602 | 0.551 |
|  | last-mb k3 KL | 7.7e-02 | 9.8e-02 | 9.2e-02 | 6.9e-02 |
|  | gnorm / clipc | 1.33 / 0.388 | 3.06 / 0.169 | 2.71 / 0.189 | 2.91 / 0.179 |
|  | maxlr | 8.0 | 6.7 | 6.9 | 7.5 |
|  | σ ratio (afsd) | – | 1.404 | 1.400 | 1.402 |
|  | falls / 1 M steps (window) | 8 | 9 | 7 | 7 |

Falls per 1 M steps per ~105 M-sample window (100 T=32 updates / 50 T=64 updates), window END in B samples:

| window end | ctrl_trunk | af50n_trunk | T=64 γ=0.965 | T=64 γ=0.98 |
|---|---|---|---|---|
| 0.10 B | 21200 | 23040 | 27013 | 21839 |
| 0.21 B | 4276 | 6510 | 12213 | 12653 |
| 0.31 B | 463 | 746 | 2347 | 10039 |
| 0.42 B | 93 | 141 | 479 | 2639 |
| 0.52 B | 23 | 49 | 128 | 229 |
| 0.63 B | 22 | 45 | 76 | 83 |
| 0.73 B | 16 | 17 | 28 | 42 |
| 0.84 B | 17 | 12 | 30 | 34 |
| 0.94 B | 16 | 16 | 15 | 26 |
| 1.05 B | 8 | 13 | 12 | 10 |
| 1.12 B | 8 | 10 | 8 | 8 |

| arm | batch | updates | wall first→last update (s) | s / 100 M samples (total) | THROUGHPUT steps/s (total) | steady s/upd → steps/s (last 20% of updates) | steady s / 100 M | peak-vram (allocator) | nvidia-smi max MiB |
|---|---|---|---|---|---|---|---|---|---|
| ctrl_trunk | 1,048,576 | 1068 | 6,690 | 597 | 167,408 | 5.57 → 188,219 | 531 | 12.60 GiB | 24242 |
| af50n_trunk | 1,048,576 | 1068 | 5,493 | 491 | 203,864 | 4.59 → 228,602 | 437 | 12.17 GiB | 24222 |
| T=64 γ=0.965 | 2,097,152 | 534 | 5,449 | 487 | 205,519 | 9.14 → 229,349 | 436 | 7.00 GiB | 18468 |
| T=64 γ=0.98 | 2,097,152 | 534 | 5,470 | 488 | 204,737 | 9.71 → 215,868 | 463 | 7.00 GiB | 18468 |

Per-105 M-sample window MEANS of `acc` (window end in B samples), the
number that carries the verdict; `ctrl−af50n` is the two-pod /
two-code drift the 1.4 re-run used as its noise floor:

| window end | ctrl | af50n | T=64 γ=.965 | T=64 γ=.98 | γ.965 − af50n | γ.98 − af50n | γ.98 − γ.965 | ctrl − af50n |
|---|---|---|---|---|---|---|---|---|
| 0.10 B | 0.6965 | 0.6913 | 0.6796 | 0.6515 | −0.0117 | −0.0398 | −0.0281 | +0.0052 |
| 0.21 B | 0.7829 | 0.7751 | 0.7526 | 0.7097 | −0.0225 | −0.0654 | −0.0429 | +0.0078 |
| 0.31 B | 0.8220 | 0.8179 | 0.8048 | 0.7928 | −0.0131 | −0.0252 | −0.0120 | +0.0040 |
| 0.42 B | 0.8488 | 0.8441 | 0.8268 | 0.8207 | −0.0173 | −0.0234 | −0.0061 | +0.0047 |
| 0.52 B | 0.8658 | 0.8613 | 0.8449 | 0.8349 | −0.0164 | −0.0264 | −0.0100 | +0.0045 |
| 0.63 B | 0.8817 | 0.8769 | 0.8602 | 0.8467 | −0.0166 | −0.0302 | −0.0136 | +0.0048 |
| 0.73 B | 0.8939 | 0.8908 | 0.8746 | 0.8579 | −0.0162 | −0.0329 | −0.0167 | +0.0031 |
| 0.84 B | 0.8996 | 0.9016 | 0.8838 | 0.8699 | −0.0177 | −0.0317 | −0.0140 | −0.0020 |
| 0.94 B | 0.9027 | 0.9059 | 0.8922 | 0.8763 | −0.0138 | −0.0296 | −0.0159 | −0.0033 |
| 1.05 B | 0.9105 | 0.9130 | 0.8992 | 0.8818 | −0.0138 | −0.0312 | −0.0174 | −0.0025 |
| 1.12 B | 0.9117 | 0.9179 | 0.9033 | 0.8860 | −0.0146 | −0.0319 | −0.0173 | −0.0061 |

Whole-run falls: ctrl 2.74 M (2,447 / M), af50n 3.21 M (2,865 / M),
T=64 γ=.965 4.44 M (3,965 / M), T=64 γ=.98 4.99 M (4,457 / M); all
four end at 7.5–10 / M in the last 105 M window.

Wall and VRAM: the T=32 rows are the previous container (pre-2.7
ragged pool, pre-2.2, pre-2.5), so their allocator peaks (12.60 /
12.17 GiB) are not comparable — T=32 on THIS code is **5.76 GiB**
(bootstrap smoke, `129k` steps/s over 5 warm updates). Per-phase VRAM
attribution needs `--bench`, which was not run (per brief); the
allocator peak is the whole-run max. `steady` = last 20% of updates
from the log timestamps. The two T=64 arms cost the SAME wall as
af50n for the same samples (5,449 / 5,470 vs 5,493 s; 229k vs 229k
steady steps/s): rollout cost per sample is unchanged and the
filtered update does 64 steps per 2.1 M samples = af50n's 32 per
1.05 M. Peak +1.24 GiB over T=32 (obs buffer doubles), as predicted.
`nvidia-smi` max 18,468 MiB on both T=64 arms (24,2xx on the old
container, ragged pool not yet landed).

#### Verdicts (one seed per arm; floor ±0.004 per window)

**(1) T=64 at γ=0.965 vs the T=32 filtered run at matched samples:
NO — it loses, and not marginally.** Window-mean acc is −0.012 to
−0.023 below `af50n_trunk` in all 11 windows and −0.007 to −0.030
below `ctrl_trunk` in 10 of 11 (the exception, 0.84 B, is −0.016 vs
ctrl too — every window is negative against both controls). The
point values at 1.12 B: 0.9076 vs 0.9210 / 0.9105. That is 3–5× the
drift floor with a constant sign, so it is a negative result, not a
null. `ep_return(last100)` agrees in direction where it says anything
(112 vs 118 / 106; a ±20% instrument). The early-phase mechanism is
visible: T=64 has **2–3× the falls of af50n from 0.2 to 0.6 B samples**
(12,213 / 2,347 / 479 / 128 vs 6,510 / 746 / 141 / 49 per M) and a
lower `ev` at every window (0.819 vs 0.839 at 0.31 B, 0.956 vs 0.972
at 0.73 B, 0.982 vs 0.989 at 1.12 B) — the longer trace bought a
noisier target (Var(returns) grows with T, C §1.3's own caveat), the
critic fits it later, and the policy spends 0.2–0.5 B samples falling
more, which is where the acc gap opens (−0.0225 at 0.21 B) and then
never closes. The gap is flat at ≈ −0.015 from 0.3 B on, i.e. the
T=64 arm is not converging toward af50n at the end of the budget.

*What this does and does not say.* The design matched SAMPLES and
optimizer steps, so it also **halved the number of policy iterations**
(534 vs 1068 rollout→update cycles). At matched samples "longer
credit window" and "half as many policy updates" are the same knob
here, and the result says the pair is net negative; it does not
isolate the window. The clean isolation is `--num-envs 16384
--horizon 64 --minibatches 64 --adv-filter 0.5` to 1068 updates: same
batch (1,048,576), same update count, same steps, same schedules as
af50n, only T (and the per-env sample count) change — that is the
probe C §1.3 should have specified, at the same 1.5 h cost. Until it
runs, T=32 stays the recipe.

**(2) Does γ=0.98 add anything? NO — it costs accuracy and falls.**
Window-mean acc is −0.006 to −0.043 below the γ=0.965 arm in every
window (−0.017 over the last four) and −0.032 below af50n at 1.12 B
(0.8849 vs 0.9210 point). Falls are the headline: a **plateau at
10,039 / M in the 0.21–0.31 B window vs 2,347 (γ=.965) / 746
(af50n)**, still 2,639 vs 479 / 141 at 0.42 B, converging only after
0.6 B and matching at the end (8 / M). `return / tick` is NEGATIVE at
0.31 B (−1.5 vs +1.2 / +2.0): the policy was accepting falls. This is
the stated confound in action, not fixed here: at γ=0.98 the
discounted note-reward stream is ~1.75× larger and the flat fall
penalty (5.0, sized at γ=0.965) is relatively that much cheaper, and
the value-norm EMA (halflife ~35 updates at β=0.02) spends the first
~100 updates re-centring on the larger targets — `ev` is **0.50 in the
first 105 M window vs 0.69 (γ=.965) / 0.71 (af50n)**, i.e. the critic
took ~2× longer to become useful, exactly when the fall/no-fall
distinction is being learned. The session's logged prediction ("γ .98
falls −20–40%, acc ±0.003") was for the resume-from-trained design
and does not transfer to from-scratch; here it is falls +330% at
the 0.31 B window and acc −0.017. A from-scratch γ=0.98 arm would need
the fall penalty rescaled (≈ ×1.75) to be a fair test; the
resume-at-1000 ladder from the session note remains the right way to
test the horizon on a trained agent.

**(3) Entropy / clipfrac / KL signatures.**

- **σ ratio:** identical, 1.384–1.402 on all three filtered arms —
  the kept/full σ does not depend on T or γ.
- **Entropy:** the filter's extra entropy drop at T=32 (af50n 3 nats
  under ctrl from 0.5 B on) **does not appear at T=64**: the γ=0.965
  arm tracks the UNFILTERED control, not af50n (`entfull` 26.5 vs
  25.5 ctrl vs 22.7 af50n at 1.12 B; 34.0 / 32.7 / 29.6 at 0.63 B),
  despite the same ent coefficient at every sample count. So the
  extra sharpening the 1.4 re-run attributed to "kept rows are the
  high-|A| states" is not intrinsic to the filter — it needs the
  1068-iteration T=32 cadence to show, which is consistent with (1):
  the number of policy iterations, not the trace, is what the T=64
  arm gave up. γ=0.98 stays softer still (29.2 at 1.12 B, +2.7 over
  γ=.965); its per-group entropies at 534 are all higher (spine 0.72
  vs 0.68, legs 0.85/0.92 vs 0.76/0.79).
- **Clipfrac / KL / gnorm:** γ=0.965 at T=64 is af50n's signature
  unchanged — last-105 M means clipfrac 0.606 vs 0.611, k3 9.2e-2 vs
  9.7e-2, gnorm 3.07 / clipc 0.174 vs 3.19 / 0.166. The update-phase
  dynamics are the filter's and are T-invariant. γ=0.98 moves the
  policy **less** per update (clipfrac 0.527, k3 6.1e-2, −13% /
  −34%) at the same lr and gnorm — the larger-scale, noisier
  advantages dilute the per-sample signal after normalization.
- **Critic:** `ev` ordering is af50n ≥ ctrl > γ.965 > γ.98 at every
  window after 0.2 B (1.12 B: 0.989 / 0.987 / 0.982 / 0.969); the
  T=64 arms never catch up, and γ=0.98's early 0.50 is the value-norm
  transient.
- `maxlr` carries no signal (6.9 / 7.5 vs 6.7 / 8.0 at the end).

**Not done:** the isolating probe (N=16384 × T=64 at 1068 updates); a
second seed on anything here; leaf cut and bench on either T=64 trunk
(resumable from `trunk_upd00504.pt` / `trunk_final.pt` in both run
dirs, on the pod and in the backup); a fall-penalty-rescaled γ=0.98.
