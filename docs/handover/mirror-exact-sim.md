# Handover: mirror-exact simulator (branch `sim/mirror-exact`)

Goal (user-approved): make the sim mirror-exact BY CONSTRUCTION —
env(M s, M a) = M env(s, a) for the x -> -x reflection M — with no runtime
flag / alternating order. Body constants unchanged; solver ordering only.

## Done
- **Rust reference (commit e73767a), all tests green** (`cd rust && cargo test --release`: 98 passed).
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
- **CUDA kernel transcription (commit after e73767a, UNTESTED — never compiled yet)** in
  `python/beat_arena_env/csrc/beat_arena_cuda.cu`: PAIR_MIRROR/PAIR_ORBIT/JORB tables (+ static_assert),
  narrow phase orbit order, limit-row orbit assembly + pair Jacobi, ground feet Jacobi + kk^2 corner
  walk, contact pair Jacobi, cap pass + step 11 over JORB with device `pair_box_targets`.

## Left
1. Compile + run CUDA parity on the pod (was just started, see below); fix compile errors.
   `tests/test_v10_parity_cuda.py`, `tests/test_production_lockstep.py`, `tests/test_v11_contact_parity.py`,
   `tests/test_flux_parity_cuda.py`, then the whole suite with CUDA.
2. Throughput: short train_batch run (few updates) base vs fix at the bend7 recipe flags
   `--gamma 0.975 --n-blocks 10 --catalog both --mb-accum 2 --depth-init --muon-enc-hidden`
   (training was ~152k steps/s). Needs maps (symlink `/workspace/wb/python/maps`, as bend7 does) and a
   pool_cache (per-file symlinks to `/workspace/bend7/python/pool_cache/*.npz` into the symfix dir —
   don't write into bend7). Pod disk is tight (3.8 GB free).
3. Mark PR ready, CI green, `gh pr merge --merge` (no --auto).
4. Follow-up: merge main into exp/bend7, rebuild /workspace/bend7, resume bend7_trunk with
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
- Throughput: NOT measured yet.

## Pod state
- `/workspace/symfix/base` = origin/main snapshot, `/workspace/symfix/fix` = this branch (CUDA WIP),
  each with its own `pylib` (maturin wheels installed with --target) and its own kernel build dir
  (`python/beat_arena_env/csrc/build`). Cargo target `/root/cargo-target-symfix` (109 MB, deletable).
- Running at handover: `python -m pytest -x -q tests/test_v10_parity_cuda.py` in /workspace/symfix/fix/python
  (nohup, log `/workspace/symfix/parity1.log`) — first kernel compile. bend7 training is paused
  (no `^python train_batch.py` process).

## Known issues
- The lateral-velocity residual above (test bound is 3 SE + 2 mm/s, rejects the pre-fix -6 mm/s).
- CUDA changes not yet compiled; register pressure of the feet-Jacobi block (2x6 vf/lf arrays) and
  the inlined pair LCP in the unrolled cap / step-11 loops may cost throughput — measure.
- Scratch audit scripts (laptop): /tmp/claude-1000/-home-evolheartrise-git-projects-BeatSaberRL/dcb1a0a2-73f1-4c8b-9ec9-f6f9911b2ca5/scratchpad/sym_audit/
