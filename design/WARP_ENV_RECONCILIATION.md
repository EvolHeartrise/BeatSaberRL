# Warp-per-env — reconciliation of the two designs (2026-08-15)

Both docs agree on the load-bearing decisions: one env per warp
(e = block*4 + warp, lane = tid&31), 128 threads / 4 envs per block,
NO __syncthreads ever (whole-warp liveness; a uniform early return
per warp only), no float atomics, fixed shuffle trees (run-to-run
bitwise determinism is a hard gate), f32-only warp kernel behind
`if constexpr` inside the existing AT_DISPATCH (legacy kernel stays
byte-identical as the f64 parity oracle + escape hatch), BEAT_ENV_WARP
default OFF, PGS row order / Cholesky column order / CRBA merge order
are SEMANTIC, per-stage ptxas census gates, stage-1 and stage-2 gate
on BITWISE identity to the legacy kernel.

Resolved divergences:

1. **Mass matrix placement**: SHARED (shell's plan = solver core's own
   fallback: 5.9-6.1 KB/env, column-major stride 35, unpadded — 35 is
   coprime with 32 so both walk directions are bank-conflict-free).
   Both configs reach 16 warps/SM; shared is simpler and frees ~32
   regs/lane. The registers-resident factor (solver config B) is held
   as a stage-7 occupancy-ladder probe, not the baseline.
2. **Contact capacity**: hold NUM_PAIRS=55-sized rows through stage 5
   (shell's discipline: every A/B measures the mapping, nothing else);
   probe NC_CAP ∈ {55, 16, 12} at stage 7 WITH the overflow-counter
   tensor (report-don't-hide). Note: this is occupancy-relevant
   capacity (registers/arena), distinct from the refuted thread-per-env
   frame cut.
3. **FK delivery** (solver's "main ask"): decided AT STAGE 5/6, when
   CRBA/RNEA go lane-parallel — until then FK stays serial-on-lane-0
   and the question is moot. The 450-shuffle vs 1.8 KB/env cost table
   from the solver doc is the decision input.

Combined staging (S-numbers below are the project's; each stage lands
alone, gates ptxas census BEFORE timing, and reverts on census fail):

- S0  measure realized `ng` (ground rows) under production traffic —
      hist-patch instrumentation, folded into the S1 pod gate. Sizes
      the S2 fused-RHS sweep (1 + 3·ng + nc + nl).
- S1  warp shell: new step_kernel_warp<float,FC>, lane-0 runs the
      existing serial body via the existing device functions, W1
      launch config, selection plumbing (pybind trailing
      kernel_variant + BEAT_ENV_WARP + env kwarg). GATE: bitwise
      identity vs legacy f32 + run-to-run determinism. Expect a
      SLOWDOWN (scaffold).
- S2  fused multi-RHS triangular solves (one 70-level sweep for all
      ~33 RHS of a substep). GATE: bitwise (parallelism is over rows,
      not the summation index). First expected win.
- S3  PGS row-body shuffle butterflies (first deliberate
      reassociation). GATE: tolerance ladder (1/8/30/240 ticks,
      distributional past 30) + determinism.
- S4  Cholesky right-looking warp factor. S5 CRBA column emissions.
      S6 RNEA/FK frame-per-lane (+ FK-delivery decision).
- S7  occupancy ladder (reg-resident factor probe, NC_CAP probe,
      24/32-warp configs) + the LEARNING GATE (3k probe, leaf-end
      return) before any campaign adopts the warp kernel.
- STOP RULE (solver doc): if S2+S3 do not beat the legacy kernel
  end-to-end, stop — S4-S6 cover only ~14% of today's ops.

## S0 result (2026-08-15, 39.3M samples/arm, trained policy, full pool)

Realized ground-row count ng per (env,substep): production
{0: 4.4%, 1: 17.7%, 2: 45.7%, 3: 24.9%, 4: 6.8%, >=5: 0.5%}, max 8;
stress pushes shift mass toward 0/1 (airborne) but same max 8. So the
typical fused-sweep width is 1 + 3*ng + nc + nl ≈ 8-15 RHS (ONE round
of 32 lanes); the ~33-RHS two-round case is the contact-rich tail.
Stage-1 floor recorded: 614.0 vs 122.5 ms/tick (5.01x, by design).
