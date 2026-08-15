# GPU performance audit — 2026-08-16 (Opus 5 static audit)

Headline: no single large win statically visible. Plan = (a) re-measure
first, (b) a stack of 0.5–3% policy-stack fixes that are REGRESSIONS
introduced by the 2026-08-13 6.2x scale-up, (c) two >10% bets needing a
profile. Structural fact: EVERY perf verdict in the repo predates the
current 861k/27-token config (the 4090 round measured the 138k arch;
the kernel header's numbers are sm_61).

## MEASURED — 2026-08-15 ladder (isolated 4090 pod, ranked-pool shape)

Tier 0 answers:
- 0.1 split at production shape (N=16384 T=8, exec-true): **71%
  rollout / 29% update** — the "update grew to 35–50%" estimate was
  WRONG; 2.1's upside re-scales to ~1–3.5% of wall.
- Torch-profiler census: **step_kernel = 90% of the rollout phase
  (~146 ms/tick at 16384 envs) ≈ 64% of total wall.** The env kernel
  is the dominant lever; every policy-stack item combined caps at ~10%.
- 0.2 sm_89 census (cold build): step_kernel<float> 254 regs (write)
  / 168 (read-only), 42 KB stack frame, ZERO spills; fp64 variants
  spill (parity-test-only, don't care).
- 0.3 ncu: **ERR_NVGPUCTRPERM — counters locked on RunPod secure
  cloud.** Bandwidth-vs-latency stays unresolved by counters; census
  proxies: 8-warp reg ceiling + 42 KB/thread local frame both stand.
  Tier 3.3 (warp-coherence permutation) is justified either way.
- relu_bwd<256> measures ~1.3 TB/s effective — at the DRAM roofline,
  already optimal.

Tier 1 A/B (ms/update mb=16384 | graphed fwd µs N=16384 | prod steps/s):
- A0 pre-audit baseline: 49.23 | 16,149 | 80.6k
- A1 gemm widths:        48.32 | 16,150 | 79.8k (update −1.8%, ~0.5%
  wall — real but small; bit-identical, kept)
- fold: 48.55; sdpa: 42.24 | 12,482 (**−23% fwd**); qkv: 45.83; heads:
  48.32 (0 alone — kept for the launch-count cut, bit-identical)
- **A7 sdpa+qkv+heads: 40.33 (−18%) | 11,900 (−26%) | 85.0k
  (+5.4–6.9% end-to-end). ADOPTED as defaults 2026-08-15.** Numerics
  gate: worst 5.9e-3 rel (fp16 last-rounding class); r1 becomes
  exactly 0 (rollout and update share the SDPA kernels). Escape
  hatches: BEAT_ATTN=legacy|fold, BEAT_QKV_PACKED=0,
  BEAT_PACKED_HEADS=0, BEAT_GEMM_LEGACY=1, BEAT_NO_WIDE_RELU_BWD=1.
- The sm_61 SDPA refutation indeed did not transfer (audit 1.3
  vindicated); old fused-attn deletion re-confirmed dead.

Tier 3.3 warp-coherence permutation: **MEASURED NULL, reverted**
(implemented behind BEAT_ENV_PERM, bitwise-identity proven — 86
parity/env tests green with the permutation forced on — then:
step_kernel 134.07 -> 133.75 ms/tick (−0.24%, noise), --bench 86.5k
vs 87.0k (noise)). The header's 3.5x identical-envs figure is another
sm_61 verdict that did not transfer: sm_89 cost is dominated by the
42 KB/thread local frame, whose traffic is order-invariant. DO NOT
RETRY without counters showing divergence stalls. (Mechanism lives in
git history at 1c548d1 if a future arch changes the calculus.)

Tier 2.1 graphed update: **REFUTED by the census** — update-phase CPU
enqueue is 30 ms/iter vs 48 ms/iter GPU execution (63%), the queue
stays full, the phase is GPU-bound not launch-bound at mb 16384.
Graphs would buy ~0 at production shape. Skip.

Remaining open: 3.1/3.2 (pool layout / status transpose) — decided by
the pool-residency probe (tiny L2-resident pool vs full 7.1 GB pool:
if step time is flat, pool reads are not the bottleneck and Tier 3
layout work is refuted too); then 2.3 bf16 behind a learning gate
(user decision — it changes numerics and deletes the loss-scale
apparatus).

## Tier 0 — measure first (~1h)
- 0.1 `train_batch.py --bench` at production shape: get the TRUE
  rollout/update split (est. update now 35–50% of wall, was 19%).
- 0.2 Regenerate ptxas census on sm_89 (capture cold-build log).
- 0.3 ONE Nsight Compute run on step_kernel: decides bandwidth-bound
  (41,872 B/thread local frame => ~5.4 MB/SM vs 128 KB L1) vs
  latency-bound (4 warps/SM against a reg ceiling of 8). This selects
  Tier 3.1/3.2 (layout) vs 3.3 (divergence).
- 0.4 Restore bench_policy_forward/update.py from bd07669^ (deleted
  with the fused stack; they're the A/B harness for Tier 1–2).

## Tier 1 — scale-up regressions (bit-identical or near-tie)
- 1.1 fp16_gemm.cu bias+ReLU fast path only handles widths 128/64;
  the ffn hidden (512, the model's LARGEST tensor: 453 MB) and trunk
  (256) fall to the slow s64-modulo kernel. Fix: p2<256>/p2<128>
  instantiations (bit-identical, ~5 lines) AND re-probe cuBLASLt
  fused RELU_BIAS (refuted at d=64; ~2% of wall at d=512).
- 1.2 relu_bwd hard-gated to 64/128 => ffn/trunk backward pays an
  extra full-tensor pass + separate bias-grad re-read (~0.5%).
  Mechanical fix.
- 1.3 4-head attention is hand-rolled and unfused (the fused kernel
  only ever ran at n_heads==1 — DEAD CODE at production config).
  (a) baddbmm fold of scale+bias (5 lines); (b) F.scaled_dot_product_
  attention (mask is key-side, no all-masked rows => safe). ~2–3%.
  Old SDPA refutation was sm_61/d64/T19/1-head — does not transfer.
- 1.4 Pack q/k/v into one GEMM (flatten_fp16_replicas already
  supports contiguous groups). ~1.5–2%.
- 1.5 30-way action-head cat runs every update minibatch (~480
  launches/update); static view pair fixes it. <0.5%.

## Tier 2 — re-price old refutations
- 2.1 CUDA-graph the UPDATE phase: loss-scale machine is already
  fully device-side; blockers are small (capturable fused Adam +
  device lr tensor, cpu_opt_state capturable flag, flat16 copy
  instead of host version check, minibatch offset staging). Was
  "1–4% wall", now 4–12% if update share grew.
- 2.2 A plain-torch + torch.compile arm (F.linear/nn.LayerNorm/SDPA
  under autocast): most likely to subsume 1.3–1.5 in one step. The
  custom stack was justified by sm_61 measurements.
- 2.3 bf16: deletes the loss-scale apparatus AND the inf x 0 = NaN
  class that killed both gen-3 5B runs; Ada runs bf16 at fp16 rate.
  Needs a 50M learning gate (r1 is the instrument).

## Tier 3 — env kernel, parity-preserving (gated on 0.3)
- 3.1 Repack chart pool SoA -> AoS (P, C, 8): obs writer reads 13
  scattered arrays per note today. Bit-identical layout change.
- 3.2 Transpose status to (C, N) and/or widen the 512-byte reset
  memset to 32-bit stores (~3% sector efficiency today).
- 3.3 Warp-coherence env permutation (sort envs by pending-count per
  tick, indirect e = env_order[gid]): per-env bitwise identical;
  header measured 3.5x identical-envs vs production traffic;
  realistic 10–25% of env kernel. One-day spike.
- 3.4 Free: -lineinfo in jit_cache (forces one rebuild), randint out=.
- 3.5 N=32768: measured +10% only; costs a learning gate. Skip.

## Tier 4 — parity-breaking (do not do casually)
fast-math/ftz (breaks bitwise-CRT claim + parity test), warp-per-env,
frame reduction via NUM_PAIRS/packed triangle (measure realized nc
histogram first), stale-policy pipelining (algorithm change),
two-chain rollout graph (RNG-stream shift only — accepted class).

## Verified non-problems
No hidden syncs in hot loops; GAE fully on GPU; auto-reset device-side;
optimizer already effectively packed-leaf (docstring stale); pop_*
syncs land inside the print block. Do-not-retry list (measured
negative): write_obs split, width ladders, c_J dot narrowing, c_minv
transpose, maxrregcount.

## Suggested order
Tier 0 -> 1.1+1.2 -> 1.3a/b -> (0.3 decides 3.1/3.2 vs 3.3) -> 2.2
measurement arm -> 2.1 -> 2.3 behind a learning gate.

Stale comments to fix while there: policy.py header (12-slot/19-token/
2-block/1-head claims), beat_arena_cuda.cu "16 slots"/"obs 340",
cuda_env.py OFF_BODY_PTS comment.
