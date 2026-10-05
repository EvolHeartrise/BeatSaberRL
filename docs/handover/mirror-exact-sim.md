# Handover: mirror-exact simulator (branch `sim/mirror-exact`)

Goal (user-approved): make the sim mirror-exact BY CONSTRUCTION —
env(M s, M a) = M env(s, a) for the x -> -x reflection M — with no runtime
flag / alternating order. Body constants unchanged; solver ordering only.

## Done
- **Rust reference (commit e73767a; fast path added later), all tests green** (`cd rust && cargo test --release`: 98 passed).
  - ground: right foot walks mirror corner order (`collide.rs ground_corner_k`, k^2), feet are a
    Jacobi pair (one combined `v += G_L L_L + G_R L_R`);
  - contacts: narrow phase emits `PAIR_ORBIT_ORDER`; adjacent mirror pair = one Jacobi PGS step;
  - limit rows: joint orbit order (0,1,2,3,10,4,11,...,22,28); mirror pair = Jacobi step;
  - omega-cap pass + step-11 clamps: orbit order; a mirror pair is projected jointly by the exact
    2-variable box LCP `solver.rs pair_box_targets` (continuous, momentum-consistent);
  - `Model.mirror` field (+ `HUMANOID_MIRROR`); pendulum test models get identity maps.
  - regression tests `rust/src/mirror_tests.rs` (soak one-tick gap, symmetric stays symmetric,
    mean yaw/roll/lateral over symmetric states, table symmetry).
  - `python/beat_arena_env/mirror_ops.py` (operator M on qpos/qvel/actions/630-lane obs) +
    `python/tests/test_mirror_ops.py`; `tests/test_self_contact.py` made terminal-tick aware.
  - Python suite on the laptop against the new build: 543 passed, 109 skipped (CUDA).
- **CUDA kernel transcription** in `python/beat_arena_env/csrc/beat_arena_cuda.cu`: PAIR_MIRROR /
  PAIR_ORBIT / JORB tables (+ static_assert), narrow phase orbit order, limit-row orbit assembly +
  pair Jacobi, ground feet Jacobi + kk^2 corner walk, contact pair Jacobi, cap pass + step 11 over
  JORB with device `pair_box_targets`. Compiles; parity verified on the pod (RTX 5090, 2026-10-04).
- **Throughput fast path (both engines)**: `single_box_target` — when only ONE twin of a mirror pair
  is off its interval, its single update (the LCP's own first step) is taken whenever it leaves the
  other twin inside its interval, reading the coupling from the firing twin's column (`ca[vb]`), so
  the twin's column is not built. Mirror-equivariant (in the mirrored state the other twin plays a);
  differs from the full LCP only by m_ab = ca[vb] vs the 0.5(ca[vb] + cb[va]) average (rounding).
  The #6e column prediction is mirror-closed too (a predicted joint / limit row predicts its twin).
- **Tests**: `test_self_contact.py::test_cuda_pricing_matches_telemetry` made terminal-tick aware like
  its Rust twin; `test_wrist_resets.py::test_cuda_pop_and_episode_record` at 1024 envs (random play
  lands ~1-2 good cuts per 256 envs in base and fix alike, so 256 was a coin flip).

## Verification (pod, 2026-10-04)
- `cargo test --release`: 98 passed. Mirror tests unchanged by the fast path (numbers below).
- Full Python suite with CUDA: 732 passed; the 5 failures + 2 collection errors are exactly the
  ones `origin/main` has on the same pod (flux scripted 114 vs 115 expectation — known flaky;
  i07 blade-geo flips; v43 energy ratio; two viz/-fixture tests and two publish_replays imports
  that need `viz/`, absent from the pod snapshot).
- CUDA vs Rust, 32 tame random states, one tick: 29/32 within 1e-9 (base 30/32), inside the gate.
- GOTCHA found: base and fix shared one cargo target dir; the copied fix sources kept older mtimes,
  so cargo reused base's artifacts and the fix pylib held the BASE .so (the first parity run failed
  32/32 for that reason, not the kernel). Build each tree with its own CARGO_TARGET_DIR.

## Throughput (RTX 5090, N=32768, f32, `tools/bench_env_kernel.py time --sample`)
| tree | stdtr (trained trunk, sampled) | standard (random actions) |
|---|---|---|
| base (origin/main) | 15.67 ms/tick | 28.88 ms/tick |
| mirror-exact, plain pair LCP | 20.51 (+31%) | 46.27 (+60%) |
| **+ single_box_target (this PR)** | **18.37 (+17%)** | **37.72 (+31%)** |

- Training (bend7 flags, 40 fresh updates, rate over updates 10-40): base 151.7k steps/s, plain pair
  LCP 143.4k (-5.5%; early training = near-random actions). With a trained policy the env step is
  ~0.5 of ~6.6 us per training step, so the fast-path build costs roughly +1.3% training time.
- Ablation of the plain-LCP build (random actions, env steps/s; base 1.12M, fix 0.70M): reverting
  step 11 alone -> 0.92M, the cap pass alone -> 0.75M, limit-row Jacobi alone -> 0.72M, feet +
  contacts alone -> 0.71M, all four -> 1.12M. The cost is in the paired cap / clamp passes. The
  mirror-closed prediction alone took 0.70M -> 0.74M; adding single_box_target -> 0.88M.
- f32 step_kernel ptxas resources are identical to base (stack 34096 B, 124 B spills); SASS
  +17% instructions. Nsight counters are locked on the pod (ERR_NVGPUCTRPERM).
- Remaining overhead lives in the both-twins-fire LCP and the twin-at-a-stop step-11 case (a free
  twin's single update moves a parked equality twin, so it takes the joint solve).

## Left
1. Merge (CI green, `gh pr merge --merge`, no --auto).
2. Follow-up: merge main into exp/bend7, rebuild /workspace/bend7, resume bend7_trunk with
   `python train_batch.py --resume runs/bend7_trunk/trunk_upd03250.pt --run-dir runs/bend7_trunk --total-steps 13893632000 --ckpt-every 250 --gamma 0.975 --n-blocks 10 --catalog both --mb-accum 2 --depth-init --muon-enc-hidden`
   using `/workspace/bend7.sh`'s env (PATH=/root/venv/bin:/usr/local/cuda/bin, CUDA_HOME=/usr/local/cuda,
   PYTHONPATH=/workspace/bend7/pylib, cwd /workspace/bend7/python).

## Commands
- Rust: `cd rust && cargo test --release` (mirror tests: `cargo test --release mirror -- --nocapture`).
- Python (laptop, no GPU): build the .so's (`cargo build --release` in rust/ and rust32/), put them on
  PYTHONPATH ahead of the venv, `python -m pytest -q tests/`.
- Pod: `ssh -p 34270 root@180.189.55.43`; env
  `export PATH=/root/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda PYTHONPATH=/workspace/symfix/fix/pylib`,
  `cd /workspace/symfix/fix/python && python -m pytest -x -q tests/test_v10_parity_cuda.py`.

## Numbers so far (Rust f64)
- One-tick mirror gap, 500 soak states (mirror_tests): median 0.46 -> 1.9e-13; p90 5.2 -> 1.0e-12;
  p99 26 -> 4e-8 (continuous states). ~7% of soak states sit on intrinsic discontinuities (a joint
  exactly on a stop flips step-11 `hit`); a 1e-12 nudge of the UN-mirrored state jumps by the same
  amount there, so the gap at those states is rounding, not handedness (the test checks this).
- Coasting reset stance: |s - Ms| 0.28 after 10 ticks -> 3e-14.
- Symmetric near-stance states, one tick, mean: lateral -6.0 +- 1.2 mm/s -> +1.0 +- 0.3 mm/s;
  yaw rate (audit, python) +0.136 rad/s -> +0.0012 +- 0.0009; roll within noise. RESIDUAL: the
  lateral ~1 mm/s is deterministic rounding asymmetry (CRBA sibling merge order, tree Cholesky
  left-subtree-first) amplified by the unstable light wrist stack at ~7% of states. Bit-exact
  symmetric linear algebra (pairwise sibling accumulation in crba/rnea/TreeChol, Rust + CUDA) would
  remove it — not done.
- Solver quality (scratch harness, 40 soak + 10 stance episodes; base -> fix): ground unmet mean
  2.30e-2 -> 2.30e-2 m/s, contact unmet 1.39e-3 -> 1.09e-3, limit-row unmet 1.41e-3 -> 1.27e-3,
  penetration mean 3.11 -> 3.04 mm (max 14.8 -> 12.8), limit overshoot max 1.0e-2 -> 5.4e-3,
  stance cap overshoot post-solve 4.6e-3 -> 0; fall rate over 450 soak episodes 39 -> 40 (same).
- Physical change on asymmetric states (one tick vs old solver): max|dv| median 0.41 rad/s
  (~0.9% of the tick's own velocity change), p90 3.0; max|dq| median 7 mrad.
- Mirror tests with the fast path: one-tick gap median 1.85e-13, p90 1.0e-12, p99 4.3e-8; coast
  3.4e-14, double swing 1.3e-12 over 10 ticks; lateral +1.0 +- 0.3 mm/s, yaw +2.8e-3 +- 1.9e-3 rad/s.

## Pod state
- `/workspace/symfix/base` = origin/main snapshot, `/workspace/symfix/fix` = this branch, each with its
  own `pylib` and kernel build dir; cargo targets `/root/cargo-target-symfix` (base) and
  `/root/cargo-target-symfix-fix` (fix). `maps` in both trees symlink `/workspace/wb/python/maps`;
  `pool_cache` holds per-file symlinks to bend7's. Scratch: `/workspace/symfix/abl` (ablation trees),
  `*.log` / `*.txt` result files. All deletable after the merge. bend7 training is still paused.

## Known issues
- The lateral-velocity residual above (test bound is 3 SE + 2 mm/s, rejects the pre-fix -6 mm/s).
- Env-kernel overhead vs base: +17% on trained traffic (see Throughput).
- Scratch audit scripts (laptop): /tmp/claude-1000/-home-evolheartrise-git-projects-BeatSaberRL/dcb1a0a2-73f1-4c8b-9ec9-f6f9911b2ca5/scratchpad/sym_audit/
