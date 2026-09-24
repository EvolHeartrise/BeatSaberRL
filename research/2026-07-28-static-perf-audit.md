# Static performance audit — training-run speed (2026-07-28)

Five independent Opus 5 analysts, static analysis only (no code executed; s1 sweep was live).
Angles: PPO loop, CUDA kernels, env/sim layer, sweep orchestration, cross-cutting pipeline.
Config assumed = production sweep default (`sweep_protein.py:81`): N=1024, T=128, mb=8,
epochs=1, `--fp16 all`, `--graph-rollout`, `--env cuda`. Measured split ~16% rollout / 84% update,
~526 ms/update at 249k steps/s.

Consensus verdict: **the training loop and pipeline are already unusually tight** (fully
on-device rollout, one graph capture, copy-free weight propagation, zero hidden syncs in the
minibatch loop). The recoverable time is concentrated in three places:

1. **Sweep orchestration fixed costs** — ~21% of sweep wall is non-training; ~14% is a
   structurally-broken pool cache. Biggest, cheapest, and recurs every sweep.
2. **`fused_attn.cu` kernel defects** — the update path is 84% of wall and its attention
   kernel spends ~90-95% of its score phase in shared-memory bank-conflict stalls.
3. **~5-8% of wall in small training-loop waste** — launch storms, dead diagnostics,
   syncs at the worst boundary.

Findings independently corroborated by 2-3 agents (high confidence): discarded entropy inside
the captured rollout graph; the 128-iteration Python GAE scan against an empty GPU queue;
`pop_stats()` double-sync at the rollout→GAE boundary; fp16 `b_obs`; `(T+1,N)` `b_val`;
`done.to()` alloc; unbounded `ep_returns`; the `C`-sized dynamic shmem occupancy trap.

---

## Tier 1 — sweep-level wall clock (largest absolute, lowest risk)

### 1.1 Chart-pool cache never hits — 101 s/leg × 99 legs = 175.6 min (14.2% of s1 sweep)
`cuda_env.py:183-196` keys the pool cache on `repr(ChartConfig)`, and `bomb_prob` is both a
ChartConfig member (`chart.py:80-90`) and a swept dimension (`sweep_protein.py:68`, continuous
log-uniform). Measured from disk: 97 distinct bomb_prob values in 99 legs → cache hit ~never;
103 × ~317 MB npz files (31 GB) written, none ever read back.
- **(a)** Quantize bomb_prob in the cache key to a 10% log grid (keep exact value for the env):
  97 values → 34 buckets; saves ~115 min/99-leg sweep, ~100% on later sweeps. 10% bomb-density
  resolution is below single-seed noise.
- **(b)** Parallelize `generate_chart` (pure per-chart work; re-seed `default_rng(seed*P+i)`,
  multiprocessing.Pool of 8): 101 s → ~15 s; ~140 min alone, ~48 min on top of (a).
- **(c)** Cap/evict `pool_cache/` (no eviction path exists; D: has 98 GB free — one more sweep
  eats a third; a full-disk `savez` raises inside env init → spurious TRAIN FAILED → poisons
  the GP with a false failure).

### 1.2 Early stopping: fully implemented, configured, never called — ~89 min/sweep
`sweep_protein.py:59` sets `early_stop_quantile: 0.3`; `protein_sweep.py:964-993` implements
`should_stop()`/`early_stop()`; the cost-threshold model is refit every `suggest()`
(`protein_sweep.py:808`). Zero callers. `run_leg` is an unconditional `subprocess.run`.
Measured: 17/99 legs finished below 60% of median score (worst ten: 0–128k vs median 791k) —
total collapses detectable at a 25% rung. Fix: stream trainer stdout (drop
`capture_output=True`), parse the existing `ep_return(last100)` print, call
`protein.should_stop()` at ~25% of updates, kill on hit. **Validate ep_return ↔
catalog_unweighted_pp ordering correlation on observations.jsonl before arming.**

### 1.3 Bench per-leg overhead — 5-10 min/sweep, one-line fixes
`sweep_protein.py:102-107` omits `--quiet` → 6.1k per-level prints formatted+flushed and
discarded. `bench_scoresaber.py:576-578` re-serializes a 3.67 MB indent=1 JSON mid-run on a
30 s timer whose resumability purpose is void (run dir is rmtree'd on retry). Add `--quiet`
and gate the interval save.

### 1.4 Resumability — up to ~37 min per interruption
Driver never passes `--ckpt-every`; `run_leg` rmtree's the dir. `train_batch.py:255-268`
already implements exact resume. Pass `--ckpt-every` ≈5 min and prefer resume when `args.json`
matches exactly (preserves the fresh-model-per-leg invariant; cf. commit d4a1808 / run 003).

Verified non-problems: JIT cache is armed for all six extensions (3.4 s median inter-leg gap);
GP proposal cost negligible; parsed-chart cache hits 100%; checkpoint I/O negligible.

## Tier 2 — update-path kernels (`fused_attn.cu` is the headline; update = 84% of wall)

### 2.1 HIGH — unpadded smem row stride: ~18-way bank conflicts in the score phase
`fused_attn.cu:59-63` strides `sq/sk/sv` by D=64. In the score loop (`:97-105`) all 32 lanes
read `sk[j*64+c]` — same bank, 18 addresses → 18 replays/load; score phase ~90-95% stall.
`policy_forward.cu:70` already has the fix (`SROW = D+1`) with an explanatory comment;
`fused_attn.cu` never received it. Also in the backward `dp` loop (`:190-198`). Occupancy
cost verified zero (fwd 15,192→15,408 B, still 6 blocks/SM). ~10 lines. **Largest kernel win;
hits both update (8 mb × 2 blocks × fwd+bwd) and rollout (128 graphed forwards).**

### 2.2 HIGH — score dot is one 64-long dependent FMA chain
Same loop: single `acc` accumulator, no cross-dot ILP. `policy_forward.cu:271-291` documents
the fix (interleave the thread's 2-3 dots as independent accumulators, bit-identical), or split
partials (allowed — `fused_attn.cu:16-17` declares the non-bitwise tolerance class). Compounds
with 2.1.

### 2.3 MED-HIGH — `p_out` store uncoalesced from Tq threads (`fused_attn.cu:120-124`)
36 B deposited across ~18 sectors per warp-store, from 18 (block 0) or 2 (block 1) of 128
threads. Store cooperatively from all threads after the `:126` barrier.

### 2.4 MED-HIGH — bias-grad reductions re-read ~1.5 GB/update (`fused_attn.py:83-88`)
`dq/dk/dv.sum((0,1))` re-reads three (16384,18,64) fp16 tensors per backward to make six
64-float vectors whose summands are already in registers at the kernel's store sites
(`fused_attn.cu:211,244,266`). Accumulate channel sums in-kernel (strided partials or
per-block atomicAdd of 3×64 floats).

### 2.5 MED — s64 modulo per element in `bias_relu_kernel` (`fp16_gemm.cu:33`)
`int64 % int` → tens-of-instruction emulation on Pascal, ~48 M/update-minibatch-forward, in a
2-load/2-FMA/1-store kernel. Fix: int32 index; template on d2∈{32,64} → mask; best: 2-D grid
(also kills the 8 B-bias-per-4 B-data reload). Also process 2×`__half2`/thread (`:51`).

### 2.6 MED — ReLU backward never got the epilogue treatment (`fp16_linear.py:185,192`)
`threshold_backward` + separate `sum(0)` = ~302 MB/site/minibatch at ffn.0. Mirror the forward
fusion: in-place mask + column-sum partials in one pass.

### 2.7 MED — softmax on Tq of 128 threads (`fused_attn.cu:109`, bwd `:215-222`)
2 active threads for block 1 while 126 park at the barrier. Warp-per-row via
`__shfl_xor_sync` (primitive exists at `layer_norm.cu:40-44`). Related: `__expf`/`__frcp_rn`
(or `-use_fast_math` on fused_attn **only** — layer_norm/policy kernels' bit-exactness
contracts forbid it) — the `expf` sits on the serialized softmax critical path.

### 2.8 MED — block-1 (Tq=2) launch wastes ~75% of threads; smem k/v staging is a net loss
there (each element read twice). Batch 4 samples/block or a Tq≤2 specialization.

### 2.9 MED-LOW — `layer_norm.cu:520-522,564`: 2 extra launches + 2 allocs per bwd call ×32
per update (single `{2,blocks,d}` partial + one `sum(1)`). Also `:151/:166`, `:268-288`
possible re-loads across shuffles — check `cuobjdump -sass` on the existing build_ln pyd first.

Eval-path only (bench, not training): `policy_forward.cu` k-loops want `#pragma unroll 4`
(bit-exactness survives, k stays ascending); 2-samples-per-block experiment (identical
occupancy, halves weight re-reads). `policy_backward.cu` is dead in this sweep
(`--fused-update` off); its ~1.8 G atomics/backward + 12.5% occupancy findings are recorded
in the agent report for if that path returns.

Non-issues verified: no tensor cores on GP104 (fp16-as-storage is correct); cuBLAS already
does the GEMMs (`fp16_gemm.cu` cublasGemmEx choice is right); `beat_arena_cuda.cu` launch
bounds/atomics clean; layer_norm fp16 vectorization already optimal; `-O3`/PTX flags moot.

## Tier 3 — training-loop waste (~5-8% of wall aggregate, mostly trivial diffs)

- **3.1** `pop_stats()` forces 2 blocking syncs (`cuda_env.py:356-368`) at the exact
  rollout→GAE boundary (`train_batch.py:487`), draining the queue before the most
  launch-latency-sensitive region; a 3rd sync at `:606-607`. Also corrupts the published
  rollout/update timer split (t_rollout absorbs the previous update's tail; t_update is
  enqueue-only) — the 16/84 vs 67/33 confusion is exactly this. Fix: defer stats one update
  via pinned non_blocking copy + event, read at the existing log sync; fix docstring (claims
  1 sync, does 2).
- **3.2** GAE scan: 3 kernels × 128 steps on (1024,) tensors, CPU-dispatch-bound against an
  empty queue (`train_batch.py:509-514`). Cheap fix: `out=`-based mul/add (bit-exact, 3
  lines); proper fix: one custom `gae_scan` kernel (1 thread/env, absorbs `nonterm/deltas/
  coef/returns` lines 505-514). Avoid `addcmul` (FMA contraction breaks bit-exactness claim).
- **3.3** Entropy computed and discarded on every rollout step, baked inside the captured
  graph (`policy.py:337-348`; callers `rollout_graph.py:54`, `train_batch.py:470`); plus
  duplicate `logsm.exp()`. Add `need_entropy=False`; bootstrap `train_batch.py:501` should be
  `forward_stacked(obs)[1]` (currently runs full multinomial+entropy for one number, and
  consumes graph-registered RNG).
- **3.4** Flat-buffer refactor for the eager update path: 66-tensor `nan_to_num_` loop
  (528 launches/update, `train_batch.py:566-584`) + 34-launch fp16 replica refresh per
  opt.step (`fp16_linear.py:84-93`) + per-minibatch `clip_grad_norm_` regrouping → flat fp32
  master / fp16 replica / grad buffers with param views: ~880 launches → ~30. The one real
  refactor recommended; do behind a paired gate.
- **3.5** Dead diagnostic: `frac_active`/`dual_frac` computed every minibatch, never read
  (`train_batch.py:72-73,553`). Delete. Free.
- **3.6** `b_val` as `(T+1,N)` kills the per-update full `torch.cat` (`train_batch.py:506`).
  `b_done[t].copy_(done)` instead of `done.to(dtype)` (`:464` etc). Cache `torch.eye` /
  write attn_bias directly into a cached fp16 buffer (`policy.py:263-272`; `key_mask` is
  never actually consumed by blocks when attn_bias is supplied). Local `import fp16_linear`
  inside per-forward functions (`policy.py:305,324`). `.sum(-1).sum(-1)` → `.sum((-2,-1))`.
  Minibatch gathers → permute-once + contiguous slices (40 gathers → 5, epochs=1 partitions
  exactly).
- **3.7** fp16 `b_obs` under `--fp16 all`: bit-identical (same rounding at store vs use),
  halves the 150 MB buffer and the per-minibatch gather; must stay fp32 for `--fused-update`.
- **3.8** Gumbel-max instead of `torch.multinomial` (distribution-identical, graph-safe,
  deletes the exp) — changes RNG stream, so gate like the fused-head commit.

## Tier 4 — robustness items that have already cost multi-billion-step runs

- **4.1** `ep_returns` unbounded Python list (`train_batch.py:423`): ~250 MB host on a 5B run;
  only `[-100:]` ever read. `deque(maxlen=…)`. (gen3/gen4 5B runs both died ~3.8B.)
- **4.2** `GraphedRollout.refresh()` gated on mutable `policy.fp16`, which the trainer flips
  twice per update — any future early-exit between `train_batch.py:566-591` silently trains
  the rollout on frozen weights. Record `captured_fp16` at capture, assert in refresh/step.
- **4.3** Dynamic shmem = C × sizeof(scalar) (`beat_arena_cuda.cu:967`) vs
  `__launch_bounds__(THREADS,16)` budget of ~6 KB/block: C is a max over the 65536-chart pool
  (monotone in pool size); crossing 1536 f32 silently craters rollout occupancy. Print/assert
  at env construction.
- **4.4** `cuda_graphs._Graphed` has no input-shape guard (`cuda_graphs.py:34-48`) — safe for
  its one current consumer, silent-stale-output for any future one. One-line assert.

## Env layer (correctly deprioritized: env ≈ 0.5-0.9% of wall)

Real but ceiling-bounded kernel wins recorded for when the env matters again (denser charts,
bigger N): no broad-phase AABB reject before the full swept-surface SAT
(`beat_arena_cuda.cu:560-685`, 3-10× on Phase C, conservative bound = bit-identical; same fix
in `lib.rs:791-970` where the bench feels it); 64 threads/env where 1-2 do the serial work
(`:387`, invert to thread-per-env, 3-8×); high-water bound recomputed per step and 3-4× loose
(`:415-427`, per-env cursor + per-note suffix-min r_i); note-obs template `(P,C,14)` — 13 of
14 features are chart constants (`:861-879`); f32 reciprocal-multiply for the ~100
non-pow2-constant divides in the serial phases (stays in the documented drift class).

Reaches beyond the env ceiling:
- Obs layout: emit hand-id one-hot + 2 mask lanes from the kernel (kills per-forward
  `eye`+`cat`+`ones`+`cat` in `policy.py:243-272`, which run epochs×mb× fwd+bwd in the 84%);
  `nb[13]` is provably redundant with the mask lane (OBS_DIM 286→270, needs retrain — bank
  with next arch change).
- `--env rust` path (and shared Core the bench uses): `vec_env.py` ignores existing zero-copy
  `step_into`/`obs_one_into` (`lib.rs:1313,1345`); Python chart gen runs inside the vector
  step (~2.2 ms serial GIL per step — adopt the CUDA env's pool design); `lib.rs:1392` copies
  the action array per step; `reset_state` re-allocates status/trackers per episode
  (`lib.rs:580-596`); `write_obs` runs for done envs then is overwritten.
- `step()` pybind takes `std::vector<at::Tensor>` by value: 51 tensor casts/refcounts per call
  (`beat_arena_cuda.cu:894`) — dead under graph, real for eager/tests/eval; build Sim once.

Closed questions: kinematics.py not on any step path; no Python/Rust/CUDA per-step duplication;
zero H↔D per step on the CUDA path; chart upload once; lookahead already cursor-based; rayon
parallelism correct; no fp64 in production.

## Non-goals ruled out by structure

- No legal rollout/update stream overlap (strict two-way dependency) — don't chase
  double-buffering or multi-stream pipelining.
- Whole-horizon graph capture: CPU is ~10× ahead of GPU at N=1024; back-pocket for small-N.
- Note-slot sparsity compaction: slots are ~full except episode tails; not worth ragged
  batching.
- Tensor cores / GEMM library changes on GP104: n/a, already correct.

## Suggested execution order

1. Sweep fixes (1.1a+b, 1.3): ~2.9 h per 99-leg sweep, no science change, recurring.
2. `fused_attn.cu` SROW padding + accumulator split (2.1, 2.2): largest kernel win, ~10 lines
   each, inside the declared tolerance class. Gate on bench_policy_update + paired learning
   check.
3. Trivial loop fixes batch (3.5, 3.6, 3.3, 3.1, 4.1, 4.2): an afternoon, ~2-3% wall +
   honest timers + two failure modes closed.
4. Early stopping (1.2) after proxy-correlation validation.
5. GAE kernel (3.2), attention bias-grads (2.4), bias_relu modulo (2.5), relu-bwd fusion
   (2.6): each contained, each ~0.5-1%.
6. Flat-buffer refactor (3.4) last, behind a paired gate.

Full agent transcripts: session task outputs 2026-07-28 (PPO loop, CUDA kernels, env, sweep
orchestration, pipeline).

---

## Implementation results (2026-07-28, byte-identical pass)

Constraint discovered at implementation time: gen5_5b is paused at 0.5B awaiting
BYTE-IDENTICAL optimizations, so only the bit-exact subset of Tiers 2+3 landed.
Gate: 100-update production-shape run (N=1024, seed 123) — weights AND Adam-state
sha256 must equal baseline (run-to-run determinism verified first: two baseline
runs hashed identical).

| stage | steps/s | vs baseline | weights hash |
|---|---|---|---|
| baseline | 209,416 | — | b37b54ba… |
| + fused_attn SROW padding + bias_relu p2 kernel | 227,153 | +8.5% | identical |
| + Tier 3 batch A (GAE out= scan, (T+1,N) b_val, dead diag, eye cache, F.pad bias, copy_ done) | 229,025 | +9.4% | identical |
| + batch B (need_entropy=False, permute-once slices, pop_stats deferral) | 229,153 | +9.4% | identical |
| + fp16 b_obs | 229,533 | **+9.6%** | identical |

Kernel-level (best-of-5, 16384-row update shapes): attn fwd b0 3946→2158 µs
(1.83×), bwd b0 5285→3420 µs (1.55×), rollout fwd 270→148 µs (1.83×);
bias_relu ~3% (memory-bound — the s64 modulo was ALU shadow, not the roofline).

Measured NEGATIVE results (reverted, documented in fused_attn.cu):
- 2.2 score-dot accumulator interleave: fwd b0 +9%, Tq=2 +43% SLOWER post-padding
  (6 blocks/SM already hides the FMA chain; policy_forward.cu's 4-block context
  doesn't transfer).
- 2.3 cooperative coalesced p_out store: ~1.3% slower on b0; strided store was
  already hidden.

Validation: 123/123 tests pass; gen5 ckpt_upd3815 re-bench reproduces
bench/gen5_5b_ckpt_upd3815.json exactly (model hash, all 6101 per-level results,
both pp totals; levels-array order differs only among equal-stars ties — the
bench's stable sort keeps completion order, pre-existing nondeterminism).

Deferred (NOT landed — would change bits; revisit at a natural trajectory break):
2.4 in-kernel attention bias-grads; 2.6 fused ReLU-bwd+bias-sum; 2.7 warp softmax
+ __expf; 2.8 Tq=2 sample batching; 3.4 flat grad/replica buffers (a byte-safe
subset exists: flat nan_to_num + single replica copy while keeping per-tensor
clip_grad_norm_); 3.8 Gumbel-max; bootstrap via forward_stacked (drops an RNG
consumer — shifts the sampling stream).

## Implementation results, pass 2 (2026-07-28, bits-allowed)

User released the byte-identity constraint (gen5 restarts fresh; the 0.5B is
written off). The deferred set landed, each E2E-gated (same 100-update run,
speed-only gate now) + tolerance-checked (near-tie class) + deterministic
run-to-run (no atomics anywhere).

| stage | steps/s | delta |
|---|---|---|
| pass-1 head | 229,533 | — |
| 2.4 in-kernel attention bias-grad partials | 233,032 | +1.5% |
| 2.7 __expf softmax + warp-reduced bwd ds | 233,467 | +0.2% |
| 2.6 fused ReLU-bwd + bias-sum partials | 237,826 | +1.9% |
| 3.4 flat grad buffer (zero/unscale/nan_to_num/clip as 1-tensor ops) | 237,696 | ~0 (kept: CPU-side headroom, no regression) |
| 3.8 Gumbel-max sampling + bootstrap forward_stacked | 242,883 | +2.2% |

**Cumulative: 209.4k -> 242.9k steps/s = +16.0% over the session baseline.**

Design notes that mattered:
- 2.4 v1 (post-hoc global readback, 1 warp) regressed the bwd kernel 48%;
  the landed v2 accumulates channel partials in REGISTERS inside the
  dv/dq/dk loops (THREADS % D2 == 0 makes each lane's channel
  loop-invariant), stages 4 warps' partials in dead smem (sB), combines in
  fixed warp order, and finishes with one torch sum(0) over (B,3,64).
  System-level: 834+672 us of python sums -> ~190 us kernel + 150 us
  partial-sum, ~7.8 ms/update.
- 2.7: warp-per-row softmax in the FORWARD measured slower (+2.7% b0,
  +11% rollout shape) — kept serial rows + __expf there; the warp version
  survives only in the backward's ds phase. dbk is mathematically zero
  (softmax shift-invariance), so its "error" vs torch sums is noise ~0.05.
- 2.6: relu_bwd kernel 1.43-1.50x over threshold_backward+sum, gm bitwise
  exact, db within 1.2e-4.
- 3.8: exponential-race form argmax(logp - log E); distribution verified
  at n=200k (0.7842/0.1758/0.0400 vs target 0.7856/0.1753/0.0391).
- Validation: 123/123 tests; r1 diagnostic 2.2-3.1e-4 throughout (same
  band as before); trajectory now legitimately differs from gen5's.
Still open (measured-out or unattempted): 2.8 Tq=2 sample batching (untried).

## Implementation results, pass 3 (2026-07-28): flat fp16 master/replica buffers

Landed: all Linear fp32 masters repacked into ONE contiguous buffer (p.data =
views; Adam/checkpoints/state_dict unchanged) with a flat fp16 mirror —
replica refresh is ONE cast kernel (was ~26 copies x 9 refresh sites/update),
and the heads+value are laid out LAST in cat order so the fused-head rollout
cache is a pair of static views, never rebuilt. 16-element slice alignment
(64 B fp32 / 32 B fp16) — fused_attn/bias_relu reinterpret masters as
float2*/__half2*, and cuBLAS selects GEMM paths by pointer alignment (the
first 4-element cut measured ~0.2% slower; alignment bump recovered it).
Tail-bias group stays internally contiguous (torch-only consumers) so the
(37,) view exists. Trainer-only (flatten runs before graph capture; bench and
tests keep the per-tensor path).

Speed: statistically neutral — baseline samples {242.9k, 242.0k} vs flat
{242.3k, 242.5k, 242.6k} steps/s (update is GPU-bound; the removed ~230
launches/update were CPU-side and already hidden). Kept for the structural
wins: 1-kernel refresh, static head views, less WDDM launch pressure, and the
fragile per-tensor version-keyed cache retired from the trainer path.
Correctness: forward bit-identical pre/post-flatten, refresh verified, grads
flow to all 66 params, same-seed trajectory identical to pass-2 head,
123/123 tests.
