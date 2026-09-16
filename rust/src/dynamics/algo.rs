//! CRBA (mass matrix), RNEA (inverse dynamics / bias forces), dense
//! Cholesky (SPEC-v10 §3 steps 2, 3, 5), and momentum/energy
//! diagnostics. World-frame 3-vector formulation throughout.
//!
//! Free-base velocity coordinates (rows 0..6): [v_lin of the root
//! frame origin (world), omega (world)]; conjugate generalized forces:
//! [total force, total moment about the root origin]. Actuated
//! coordinate j lives at row voff + j.
//!
//! Straightforward loops, fixed evaluation order — the torch/CUDA
//! mirrors replicate these expression trees literally.
//!
//! R23 (2026-08-07) — this file carries two of the dead-store sweep's
//! three landed items; the round's headline numbers, both new rules and
//! the two recorded negatives live in the solver.rs header.
//!   · **R23b** (`crba`): the mass matrix's STRICT UPPER triangle has no
//!     reader in the shipping build, so the mirror stores are compiled
//!     out — but only where the lower/upper roles are COMPILE-TIME
//!     known (the free-base coupling rows and the root skew block, 183
//!     of ~300 entries). **+0.28%** on top of the package. The
//!     ancestor-loop half needs a runtime min/max select because `coord`
//!     is not monotone in frame index; that form measured −0.18% and is
//!     NOT shipped — see the note at the site.
//!   · **R23e** (`rnea`): the frame-origin VELOCITY chain is not
//!     propagated or stored any more. The force/moment block uses
//!     ao/om/al only, and `ws.vo`'s two real consumers (`momentum`,
//!     `energy`) each rebuild the chain in their own topological loop.
//!     **+0.45%** measured alone (22,330 → 22,430 env-steps/s at
//!     N = 256, 3 interleaved ABBA rounds, 6/6 paired wins,
//!     non-overlapping).
//! Both are gated by `tests_constraints::r23_dead_regions_have_no_reader`,
//! which NaN-poisons each region and requires the trajectory back bit
//! for bit (the R13 `poison_weld_scratch` idiom); all four poisonings
//! were confirmed to BITE against deliberately reintroduced readers
//! before the items landed.

use crate::Real;
use super::fk::{add3, cross3, dot3, mat_vec, scale3, sub3, Mat3};
use super::model::Model;
use super::Workspace;

/// I_world = R · diag(i) · R^T (about the body COM). Upper triangle
/// computed, lower mirrored — bitwise symmetric by construction (the
/// two orders (r_a·i)·r_b vs (r_b·i)·r_a round differently).
///
/// R12 REVERTED (2026-08-06, measured negative): `crba` and `rnea` each
/// call this once per massive frame per substep on the SAME FK, so the
/// second call is redundant — caching it in the Workspace (a 30×Mat3
/// array written by `crba`, read by `rnea`, i.e. the CUDA K4e port,
/// which paid +1.15% there) measured **−0.37%** on the CPU
/// (21,282 → 21,204 env-steps/s, N=256, 3 interleaved A/B rounds, every
/// round negative). On this machine 30 nine-word stores plus 30 cold
/// nine-word loads cost more than recomputing 6 three-term dots from an
/// `ws.r[k]` that `rnea` is loading anyway (`e = R·com` two lines up).
/// Do not retry without a register/cache-economics change.
fn inertia_world(r: &Mat3, i: [Real; 3]) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in a..3 {
            let v =
                r[a][0] * i[0] * r[b][0] + r[a][1] * i[1] * r[b][1] + r[a][2] * i[2] * r[b][2];
            out[a][b] = v;
            out[b][a] = v;
        }
    }
    out
}

/// Parallel-axis dyad: m·((d·d)I − d dᵀ), added into `i`.
fn add_shift(i: &mut Mat3, m: Real, d: [Real; 3]) {
    let dd = dot3(d, d);
    for a in 0..3 {
        for b in 0..3 {
            i[a][b] += m * (if a == b { dd } else { 0.0 } - d[a] * d[b]);
        }
    }
}

fn mat_add(a: &mut Mat3, b: &Mat3) {
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] += b[i][j];
        }
    }
}

/// R23b (2026-08-07): CRBA's STRICT UPPER triangle of `ws.m` has no
/// reader in the shipping build, so the mirror stores are compiled out
/// of it. The only consumer of the mass matrix is the in-place
/// factorization two lines later in `step::substep_opts`, and
/// `chol_factor_floored_inplace` touches `a[j·n+j]` (diagonal),
/// `a[j·n..j·n+j]` (row j below the diagonal) and `a[i·n..i·n+j]` with
/// i > j — lower and diagonal ONLY; its own header note already states
/// this (it is why the strict upper triangle stopped being zeroed in
/// 2026-08-03). After the factor, `ws.m` holds L and every `chol_*`
/// reader is likewise lower-only. The harness implicit-damping pass
/// touches diagonal entries alone.
///
/// `MIRROR_M_UPPER` keeps the mirrors for `cargo test` /
/// `--features introspect` / debug builds, so tests_dyn's BITWISE
/// M-symmetry test still has both triangles to compare. Same cfg set
/// (and same reasoning) as solver.rs's `DIAG_ALL_ROWS` / `LIMIT_J_ROWS`.
/// Worth ~300 stores per substep: 117 ancestor entries, 174 free-base
/// coupling entries, 9 in the root skew block.
const MIRROR_M_UPPER: bool = cfg!(any(test, feature = "introspect", debug_assertions));

/// CRBA: fills ws.m (nv×nv, row-major) from the FK in ws. Composite
/// rigid bodies accumulated leaf→root; column j of M = the generalized
/// momentum produced by unit rate of coordinate j (moment of momentum
/// about each ancestor pivot along its axis). Both triangles written
/// from the same value — M is symmetric BITWISE by construction (and
/// bitwise symmetric is what the LOWER triangle alone would be, so
/// R23b's gated mirrors do not weaken the property; see above).
pub fn crba(model: &Model, ws: &mut Workspace) {
    let nf = model.frames.len();
    let nv = model.nv();
    let voff = model.voff();
    ws.m[..nv * nv].fill(0.0);

    // per-frame composites seeded with the frame's own body (if any)
    for k in 0..nf {
        let fr = &model.frames[k];
        ws.cm[k] = fr.mass;
        ws.cc[k] = if fr.mass > 0.0 {
            add3(ws.p[k], mat_vec(&ws.r[k], fr.com))
        } else {
            ws.p[k]
        };
        ws.ci[k] = if fr.mass > 0.0 {
            inertia_world(&ws.r[k], fr.inertia)
        } else {
            [[0.0; 3]; 3]
        };
    }
    // base composite (pelvis body; fixed-base models: mass 0)
    let mut bm = model.base_mass;
    let mut bc = if bm > 0.0 {
        add3(ws.p0, mat_vec(&ws.r0, model.base_com))
    } else {
        ws.p0
    };
    let mut bi = if bm > 0.0 {
        inertia_world(&ws.r0, model.base_inertia)
    } else {
        [[0.0; 3]; 3]
    };

    // leaf→root accumulation (frames are in topological order), with
    // column fill for revolute frames BEFORE merging into the parent —
    // composite k is final once every higher-index frame is processed.
    for k in (0..nf).rev() {
        let fr = &model.frames[k];
        let (m, c, i) = (ws.cm[k], ws.cc[k], ws.ci[k]);
        if fr.coord >= 0 && m > 0.0 {
            let j = voff + fr.coord as usize;
            let a = ws.axw[k];
            // unit qdot_j: omega = a, com velocity = a × (c − o_j)
            let hl = scale3(cross3(a, sub3(c, ws.p[k])), m); // linear momentum
            let ha = mat_vec(&i, a); // angular momentum about c
            // self term: moment of momentum about o_j along a
            ws.m[j * nv + j] = dot3(a, add3(ha, cross3(sub3(c, ws.p[k]), hl)));
            // ancestors: walk the parent chain
            let mut anc = fr.parent;
            while anc >= 0 {
                let fa = &model.frames[anc as usize];
                if fa.coord >= 0 {
                    let ia = voff + fa.coord as usize;
                    let val = dot3(
                        ws.axw[anc as usize],
                        add3(ha, cross3(sub3(c, ws.p[anc as usize]), hl)),
                    );
                    // R23b DELIBERATELY DOES NOT GATE THIS PAIR, and the
                    // reason is a rule: `coord` is NOT monotone in frame
                    // index (the arm stacks put the yaw frame's HIGHER
                    // coord above the pitch frame's), so which of these
                    // two entries is the lower triangle is a RUNTIME
                    // question. The gated form
                    //   let (lr, lc) = if ia < j { (j, ia) } else { (ia, j) };
                    //   ws.m[lr*nv+lc] = val;
                    //   if MIRROR_M_UPPER { ws.m[lc*nv+lr] = val; }
                    // was built and measured: **−0.18%** against the
                    // shipping build (22,772 → 22,731 env-steps/s at
                    // N = 256, 3 interleaved ABBA rounds, 4/6 paired
                    // losses, ranges overlapping) — the compare + two
                    // index selects cost about what the 117 removed
                    // stores save. A dead-store deletion pays only when
                    // the address is a COMPILE-TIME constant of the loop;
                    // buying it with a runtime select is a wash.
                    ws.m[ia * nv + j] = val;
                    ws.m[j * nv + ia] = val;
                }
                anc = fa.parent;
            }
            if model.free_base {
                // root rows: translation conjugate = h_lin, rotation
                // conjugate = moment of momentum about the root origin
                let lrow = add3(ha, cross3(sub3(c, ws.p0), hl));
                // R23b: rows t and 3+t are free-base rows (< voff ≤ j),
                // so `m[t·nv+j]` / `m[(3+t)·nv+j]` are statically the
                // STRICT UPPER copies.
                for t in 0..3 {
                    ws.m[j * nv + t] = hl[t];
                    ws.m[j * nv + (3 + t)] = lrow[t];
                }
                if MIRROR_M_UPPER {
                    for t in 0..3 {
                        ws.m[t * nv + j] = hl[t];
                        ws.m[(3 + t) * nv + j] = lrow[t];
                    }
                }
            }
        }
        // merge composite k into its parent (or the base)
        let (pm, pc, pi) = if fr.parent < 0 {
            (&mut bm, &mut bc, &mut bi)
        } else {
            let pk = fr.parent as usize;
            // split borrow: parent index < k, disjoint from (m, c, i) copies
            let (cmp, ccp, cip) = (&mut ws.cm[pk], &mut ws.cc[pk], &mut ws.ci[pk]);
            (cmp, ccp, cip)
        };
        let tm = *pm + m;
        if tm > 0.0 {
            let tc = scale3(add3(scale3(*pc, *pm), scale3(c, m)), 1.0 / tm);
            add_shift(pi, *pm, sub3(*pc, tc));
            add_shift(pi, m, sub3(c, tc));
            mat_add(pi, &i);
            *pm = tm;
            *pc = tc;
        }
    }

    // armature (reflected rotor) inertia on every actuated diagonal —
    // keeps M PD at the shoulder gimbal-lock pose (see model.rs)
    for fr in model.frames.iter() {
        if fr.coord >= 0 {
            let j = voff + fr.coord as usize;
            ws.m[j * nv + j] += model.armature[fr.coord as usize];
        }
    }

    if model.free_base {
        // root 6×6 from the total composite (bm, bc, bi about bc):
        //   trans-trans  m·I3
        //   rot-trans    unit p̈0 = e_j accelerates everything uniformly;
        //                moment about p0 = (c−p0) × m e_j = m·[d]x e_j
        //   rot-rot      unit alpha = e_j about p0: L(p0) =
        //                [I_c + m·((d·d)I − d dᵀ)] e_j
        let d = sub3(bc, ws.p0);
        let mut rr = bi;
        add_shift(&mut rr, bm, d);
        // R23b: the translation block is bm·I3 — its six off-diagonal
        // entries are still the `fill(0.0)` at the top of this function
        // (nothing between writes rows/cols 0..3 at cols/rows 0..3), so
        // storing 0.0 over them was dead in BOTH triangles.
        for i in 0..3 {
            ws.m[i * nv + i] = bm;
        }
        for i in 0..3 {
            for j in 0..3 {
                ws.m[(3 + i) * nv + (3 + j)] = rr[i][j];
            }
        }
        let sk = [
            [0.0, -d[2], d[1]],
            [d[2], 0.0, -d[0]],
            [-d[1], d[0], 0.0],
        ];
        // rows 3+i are strictly below columns j < 3, so the first store
        // is the lower copy and the second the R23b mirror.
        for i in 0..3 {
            for j in 0..3 {
                let val = bm * sk[i][j];
                ws.m[(3 + i) * nv + j] = val;
                if MIRROR_M_UPPER {
                    ws.m[j * nv + (3 + i)] = val;
                }
            }
        }
    }

    // R23b TEST HOOK: emulate the shipping build, where the strict upper
    // triangle carries no CRBA output at all.
    #[cfg(test)]
    if ws.poison_m_upper {
        for r in 0..nv {
            for c in (r + 1)..nv {
                ws.m[r * nv + c] = Real::NAN;
            }
        }
    }
}

/// RNEA inverse dynamics: generalized force required to produce
/// coordinate accelerations `acc` (None = all-zero: the bias-force
/// call C(q, v)) at state velocity `v`, INCLUDING gravity
/// (model.gravity enters as F = m·(a_com − g)). Requires fk_all in ws;
/// result in ws.c.
pub fn rnea(model: &Model, ws: &mut Workspace, v: &[Real], acc: Option<&[Real]>) {
    let nf = model.frames.len();
    let nv = model.nv();
    let voff = model.voff();
    let g = model.gravity;
    debug_assert_eq!(v.len(), nv);
    debug_assert!(acc.map_or(true, |a| a.len() == nv));
    let a_of = |i: usize| acc.map_or(0.0, |a| a[i]);

    // base kinematics: material point at the root origin.
    //
    // R23e (2026-08-07): RNEA does NOT propagate the frame-origin
    // VELOCITY chain any more. Nothing downstream of the forward pass
    // reads it — the per-body force/moment block below uses `ao`, `om`
    // and `al` only, and the backward pass reads `ws.f` / `ws.n` /
    // `ws.axw` — while `ws.vo`'s two real consumers, `momentum` and
    // `energy`, each recompute the whole chain in their OWN topological
    // loop (they write `ws.vo[k]` from `ws.vo[parent]` in ascending k,
    // seeded from the base, so they never observe what RNEA left there).
    // `ws.vo` is kept as a field for exactly those two.
    // Sizing: `cross3(oml, d)` is common to `vo` and `ao`'s inner cross
    // and was CSE'd already, so what goes is the `add3`, the 30 three-
    // word stores and the 30 parent loads, not the cross product.
    let (om0, al0, ao0) = if model.free_base {
        (
            [v[3], v[4], v[5]],
            [a_of(3), a_of(4), a_of(5)],
            [a_of(0), a_of(1), a_of(2)],
        )
    } else {
        ([0.0; 3], [0.0; 3], [0.0; 3])
    };

    // forward pass: velocities + accelerations, then per-body inertial
    // force F (about COM) and moment n (about the frame origin)
    for k in 0..nf {
        let fr = &model.frames[k];
        let (oml, all, aol, ol) = if fr.parent < 0 {
            (om0, al0, ao0, ws.p0)
        } else {
            let pk = fr.parent as usize;
            (ws.om[pk], ws.al[pk], ws.ao[pk], ws.p[pk])
        };
        let d = sub3(ws.p[k], ol);
        let ao = add3(
            add3(aol, cross3(all, d)),
            cross3(oml, cross3(oml, d)),
        );
        let (om, al) = if fr.coord >= 0 {
            let qd = v[voff + fr.coord as usize];
            let qdd = a_of(voff + fr.coord as usize);
            let aw = ws.axw[k];
            (
                add3(oml, scale3(aw, qd)),
                add3(add3(all, scale3(aw, qdd)), scale3(cross3(oml, aw), qd)),
            )
        } else {
            (oml, all)
        };
        ws.om[k] = om;
        ws.al[k] = al;
        ws.ao[k] = ao;
        if fr.mass > 0.0 {
            let e = mat_vec(&ws.r[k], fr.com); // com − origin
            let ac = add3(add3(ao, cross3(al, e)), cross3(om, cross3(om, e)));
            let iw = inertia_world(&ws.r[k], fr.inertia);
            let fb = scale3(sub3(ac, g), fr.mass);
            let nb = add3(mat_vec(&iw, al), cross3(om, mat_vec(&iw, om)));
            ws.f[k] = fb;
            ws.n[k] = add3(nb, cross3(e, fb));
        } else {
            ws.f[k] = [0.0; 3];
            ws.n[k] = [0.0; 3];
        }
    }

    // base body's own inertial force/moment (about the root origin)
    let (mut f0, mut n0) = if model.base_mass > 0.0 {
        let e = mat_vec(&ws.r0, model.base_com);
        let ac = add3(
            add3(ao0, cross3(al0, e)),
            cross3(om0, cross3(om0, e)),
        );
        let iw = inertia_world(&ws.r0, model.base_inertia);
        let fb = scale3(sub3(ac, g), model.base_mass);
        let nb = add3(mat_vec(&iw, al0), cross3(om0, mat_vec(&iw, om0)));
        (fb, add3(nb, cross3(e, fb)))
    } else {
        ([0.0; 3], [0.0; 3])
    };

    // backward pass: accumulate into parents, project onto joint axes;
    // the armature term a_j·qdd_j mirrors its M-diagonal contribution
    // (keeps CRBA column j == RNEA(0, e_j) − RNEA(0, 0) exact)
    ws.c[..nv].fill(0.0);
    for k in (0..nf).rev() {
        let fr = &model.frames[k];
        if fr.coord >= 0 {
            let c = fr.coord as usize;
            ws.c[voff + c] =
                dot3(ws.axw[k], ws.n[k]) + model.armature[c] * a_of(voff + c);
        }
        if fr.parent < 0 {
            f0 = add3(f0, ws.f[k]);
            n0 = add3(n0, add3(ws.n[k], cross3(sub3(ws.p[k], ws.p0), ws.f[k])));
        } else {
            let pk = fr.parent as usize;
            let lever = sub3(ws.p[k], ws.p[pk]);
            let fk_ = ws.f[k];
            ws.f[pk] = add3(ws.f[pk], fk_);
            ws.n[pk] = add3(ws.n[pk], add3(ws.n[k], cross3(lever, fk_)));
        }
    }
    if model.free_base {
        ws.c[0] = f0[0];
        ws.c[1] = f0[1];
        ws.c[2] = f0[2];
        ws.c[3] = n0[0];
        ws.c[4] = n0[1];
        ws.c[5] = n0[2];
    }
}

/// f64 Cholesky pivot floor (SPEC-v10 Divergences §22) — the f64
/// analogue of the CUDA f32 training net's 1e-8 (§20e), scaled by the
/// machine-epsilon ratio (1e-8 · eps_f64/eps_f32 ≈ 1.9e-17, rounded
/// down a decade). The smallest PHYSICAL pivot (the wrist armature,
/// 2.7e-5) sits 12 decades above the floor, so any legitimate
/// factorization is BIT-IDENTICAL to the unfloored one (max(d, FLOOR)
/// returns d untouched); the floor only absorbs the singular-pose
/// noise band (measured gimbal-lock pivots ~ −4.3e-18, §12).
/// The .cu f64 path carries the SAME constant and clamp expression
/// (pinned from Python by tests/test_v10_pd_floor.py); Rust owns the
/// value. (The retired torch mirror carried it too until 2026-08-03.)
pub const PD_FLOOR_F64: f64 = 1e-17;

/// f32 Cholesky pivot floor (SPEC-v10 Divergences §20e) — the CUDA
/// kernel's `PD_FLOOR_F32`, used by the `f32` feature build.
pub const PD_FLOOR_F32: f32 = 1e-8;

/// Floor one pivot: returns (floored pivot, counted engagement). f64:
/// `!(d >= 1e-17)` floors and counts (NaN included). f32 mirrors the
/// .cu f32 branch exactly — counts only `d <= 0`, clamps through the
/// NaN-to-floor max at 1e-8.
#[inline(always)]
fn floor_pivot(d: Real) -> (Real, bool) {
    #[cfg(not(feature = "f32"))]
    {
        if !(d >= PD_FLOOR_F64) { (PD_FLOOR_F64, true) } else { (d, false) }
    }
    #[cfg(feature = "f32")]
    {
        let hit = d <= 0.0;
        (if !(d >= PD_FLOOR_F32) { PD_FLOOR_F32 } else { d }, hit)
    }
}

/// `chol_factor` with the §22 pivot floor: a pivot below PD_FLOOR_F64
/// is clamped to it (simple max — exactly reproducible, no fastmath
/// ambiguity) and counted. Returns the number of floored pivots (0 =
/// bitwise-identical to `chol_factor` succeeding). The factor is
/// always produced; the caller reports the count instead of panicking.
pub fn chol_factor_floored(n: usize, m: &[Real], l: &mut [Real]) -> u32 {
    l[..n * n].copy_from_slice(&m[..n * n]);
    chol_factor_floored_inplace(n, &mut l[..n * n])
}

/// `chol_factor_floored` writing the factor OVER its input (LAPACK
/// `potrf` convention) — the shipping form at both production call
/// sites (2026-08-03, R1 item 1e). Halves the factor stage's working
/// set from two nv×nv buffers to one, which matters at nv = 35 where
/// the solver is fighting for L1 against `j` and `minvjt`.
///
/// SAFE IN PLACE, and BITWISE the two-buffer form: entry (i, j) is read
/// as the seed and written as the result within the SAME j iteration,
/// and column j is untouched by every earlier iteration (iteration j'
/// writes only column j'), so every operand — the seed `a[i·n+j]`, the
/// pivot `a[j·n+j]`, and the factor entries `a[·][k<j]` — carries
/// exactly the value the separate-input version would have read, in the
/// same order. The strict UPPER triangle is left holding the input's
/// upper triangle and is never read (see the note below).
pub fn chol_factor_floored_inplace(n: usize, a: &mut [Real]) -> u32 {
    let mut hits = 0u32;
    for j in 0..n {
        let mut d = a[j * n + j];
        {
            let lj = &a[j * n..j * n + j]; // row j, columns 0..j
            for k in 0..j {
                d -= lj[k] * lj[k];
            }
        }
        // NEGATED comparison, not `d < FLOOR`: a NaN pivot must be
        // floored and counted too (`NaN < x` is false, so the plain
        // form let NaN through unfloored and poisoned the whole
        // factor). `!(d >= FLOOR)` is true for NaN and identical to
        // `d < FLOOR` for every non-NaN d — this matches the CUDA
        // tmax/fmax semantics the .cu mirror carries (NaN → floor).
        let (fd, hit) = floor_pivot(d);
        hits += hit as u32;
        d = fd;
        let dj = d.sqrt();
        a[j * n + j] = dj;
        // NOTE (2026-08-03): the strict upper triangle is NOT zeroed
        // any more. Nothing reads it — `chol_solve` touches only
        // l[i*n+k] with k ≤ i (forward) and l[k*n+i] with k > i, i.e.
        // row k column i with i < k, also lower — so the O(n²) zeroing
        // pass was pure dead stores. `chol_factor` below dropped it
        // too; the one test that compared whole factors now compares
        // the lower triangle (tests_dyn.rs).
        //
        // UNROLL-AND-JAM by 4 over the row index i: rows i..i+4 share
        // one pass over the k-chain. Each s_x still subtracts its terms
        // in ASCENDING k — bit-for-bit the one-row-at-a-time result —
        // and no jammed row reads a value another writes (the k-chain
        // stops at k < j, the writes land at column j).
        //
        // WIDTH 8 REVERTED (2026-08-06, R15 — the post-R9 re-probe the
        // solver header's ORDER-DEPENDENCE rule asks for): an 8-wide
        // rung above this one (ladder 8 → 4 → 1, bitwise-clean) measured
        // **−1.00%** end to end (22,273 → 22,050 env-steps/s at N = 256,
        // 4 interleaved ABBA pairs, every pair negative). Unlike the
        // `weld_axpy12!` jams next door this k-chain is a DEPENDENT
        // scalar recurrence — it cannot vectorize, so the 8 accumulators
        // and 8 row pointers are 8 real live registers competing with
        // the address arithmetic, and at n = 35 the halved `lj[k]` load
        // count is not worth what spills. 4 is the width.
        let (head, tail) = a.split_at_mut((j + 1) * n);
        let lj = &head[j * n..j * n + j];
        let mut i = j + 1;
        while i + 4 <= n {
            let (r0, r1, r2, r3) = {
                let b = (i - j - 1) * n;
                (b, b + n, b + 2 * n, b + 3 * n)
            };
            // seeds read from the SAME cells the results are written
            // back to (in place) — still the untouched input values,
            // column j being written only in this j iteration
            let (mut s0, mut s1, mut s2, mut s3) =
                (tail[r0 + j], tail[r1 + j], tail[r2 + j], tail[r3 + j]);
            {
                let t = &*tail;
                let (q0, q1) = (&t[r0..r0 + j], &t[r1..r1 + j]);
                let (q2, q3) = (&t[r2..r2 + j], &t[r3..r3 + j]);
                for k in 0..j {
                    let ljk = lj[k];
                    s0 -= q0[k] * ljk;
                    s1 -= q1[k] * ljk;
                    s2 -= q2[k] * ljk;
                    s3 -= q3[k] * ljk;
                }
            }
            tail[r0 + j] = s0 / dj;
            tail[r1 + j] = s1 / dj;
            tail[r2 + j] = s2 / dj;
            tail[r3 + j] = s3 / dj;
            i += 4;
        }
        while i < n {
            let r = (i - j - 1) * n;
            let mut s = tail[r + j];
            {
                let q = &tail[r..r + j];
                for k in 0..j {
                    s -= q[k] * lj[k];
                }
            }
            tail[r + j] = s / dj;
            i += 1;
        }
    }
    hits
}

/// Dense Cholesky M = L·Lᵀ (lower, no pivoting, fixed order — SPEC-v10
/// §3 step 5). Returns false if a pivot is not strictly positive.
pub fn chol_factor(n: usize, m: &[Real], l: &mut [Real]) -> bool {
    for j in 0..n {
        let mut d = m[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d <= 0.0 {
            return false;
        }
        let dj = d.sqrt();
        l[j * n + j] = dj;
        // strict upper triangle left untouched — see the note in
        // `chol_factor_floored`: no reader, so zeroing it was dead
        for i in (j + 1)..n {
            let mut s = m[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / dj;
        }
    }
    true
}

/// In-place solve M x = b via the factor L (forward then back subst).
/// (2026-08-03) Both k-chains are expressed over EXACT-LENGTH slices so
/// LLVM can hoist/erase the per-element bounds checks and vectorize the
/// unit-stride forward pass. Pure indexing restructure — no `unsafe`,
/// no `get_unchecked`, no `mul_add`: identical operands in identical
/// order, so the result is bitwise the naive form's.
pub fn chol_solve(n: usize, l: &[Real], x: &mut [Real]) {
    for i in 0..n {
        let lrow = &l[i * n..i * n + i]; // row i, columns 0..i
        let mut s = x[i];
        {
            let xr = &x[..i];
            for k in 0..i {
                s -= lrow[k] * xr[k];
            }
        }
        x[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = x[i];
        {
            let xr = &x[..n];
            // rows i+1..n, column i — chunks_exact keeps k ascending,
            // the same order (and therefore the same partial sums) as
            // the l[k*n + i] indexing it replaces
            for (k, lrow) in l.chunks_exact(n).enumerate().take(n).skip(i + 1) {
                s -= lrow[i] * xr[k];
            }
        }
        x[i] = s / l[i * n + i];
    }
}

/// Batched in-place solve of W right-hand sides sharing ONE factor.
/// `x` holds the W RHS ROW-MAJOR — W consecutive rows of `n` doubles,
/// exactly the `minvjt` layout, so nothing ripples into the A_ww
/// assembly / weld-null projection / PGS axpy consumers.
///
/// SHAPE (2026-08-03, R1): UNROLL-AND-JAM over the RHS index with the
/// solve index i kept OUTER. Every factor load (`lrow[k]` forward,
/// `lrow[i]` back) is shared by all W accumulators while each RHS
/// stream stays UNIT-STRIDE — the form the small-items pass established
/// as the winner on this CPU. The opposite arrangement (RHS outer, i
/// inner, or an RHS-minor `x[i*W + r]` transpose) was rejected for the
/// same reason the weld-PGS interchange was reverted: at nv = 35 the
/// whole working set is L1-resident, so there is no streaming win to
/// buy and strided inner loads only cost the unit-stride AVX2
/// vectorization.
///
/// BITWISE: accumulator r subtracts exactly the terms `chol_solve`
/// would subtract for RHS r, with the same operands in the same
/// ascending-k order, so every partial sum matches bit for bit; the
/// closing divide is per-lane IEEE (`vdivpd` is correctly rounded per
/// element — no reciprocal approximation), so a batched divide equals
/// W scalar divides exactly.
pub fn chol_solve_x4(n: usize, l: &[Real], x: &mut [Real]) {
    debug_assert_eq!(x.len(), 4 * n);
    let (a, rest) = x.split_at_mut(n);
    let (b, rest) = rest.split_at_mut(n);
    let (c, d) = rest.split_at_mut(n);
    for i in 0..n {
        let lrow = &l[i * n..i * n + i]; // row i, columns 0..i
        let (mut s0, mut s1, mut s2, mut s3) = (a[i], b[i], c[i], d[i]);
        {
            let (ar, br) = (&a[..i], &b[..i]);
            let (cr, dr) = (&c[..i], &d[..i]);
            for k in 0..i {
                let lk = lrow[k];
                s0 -= lk * ar[k];
                s1 -= lk * br[k];
                s2 -= lk * cr[k];
                s3 -= lk * dr[k];
            }
        }
        let dg = l[i * n + i];
        a[i] = s0 / dg;
        b[i] = s1 / dg;
        c[i] = s2 / dg;
        d[i] = s3 / dg;
    }
    for i in (0..n).rev() {
        let (mut s0, mut s1, mut s2, mut s3) = (a[i], b[i], c[i], d[i]);
        {
            let (ar, br) = (&a[..n], &b[..n]);
            let (cr, dr) = (&c[..n], &d[..n]);
            // rows i+1..n, column i — chunks_exact keeps k ascending,
            // the same order (and partial sums) as `chol_solve`
            for (k, lrow) in l.chunks_exact(n).enumerate().take(n).skip(i + 1) {
                let lki = lrow[i];
                s0 -= lki * ar[k];
                s1 -= lki * br[k];
                s2 -= lki * cr[k];
                s3 -= lki * dr[k];
            }
        }
        let dg = l[i * n + i];
        a[i] = s0 / dg;
        b[i] = s1 / dg;
        c[i] = s2 / dg;
        d[i] = s3 / dg;
    }
}

/// Batched solve of 12 right-hand sides held RHS-MINOR: `t` is n rows
/// of 12, so `t[i*12 + r]` is entry i of RHS r.
///
/// Why this layout exists ALONGSIDE `chol_solve_x4` (2026-08-03, R1
/// 1a): under strict IEEE semantics LLVM may NOT vectorize the k-chain
/// of a row-major batch — that would reassociate one accumulator's
/// additions. Here the 12 lanes are 12 INDEPENDENT accumulators, one
/// per RHS, so a single AVX2 FMA advances four of them at once and the
/// reassociation question never arises: lane r still subtracts its own
/// terms in ascending k. Same reason the divide is safe — `vdivpd` is
/// correctly rounded per lane. The k-chain load `t[k*12 + r]` is a
/// CONTIGUOUS 12-wide fetch, not a strided one, so this is not the
/// reverted interchange in disguise.
///
/// The caller transposes in and out (an exact copy, bitwise neutral)
/// rather than storing `minvjt` this way — the RHS-minor layout would
/// otherwise ripple into the A_ww assembly, the weld-null projections
/// and the PGS axpy, all of which want unit-stride rows.
pub fn chol_solve_t<const W: usize>(n: usize, l: &[Real], t: &mut [Real]) {
    debug_assert_eq!(t.len(), W * n);
    for i in 0..n {
        let lrow = &l[i * n..i * n + i];
        let mut s = [0.0 as Real; W];
        s.copy_from_slice(&t[i * W..i * W + W]);
        for k in 0..i {
            let lk = lrow[k];
            let xr = &t[k * W..k * W + W];
            for r in 0..W {
                s[r] -= lk * xr[r];
            }
        }
        let dg = l[i * n + i];
        let row = &mut t[i * W..i * W + W];
        for r in 0..W {
            row[r] = s[r] / dg;
        }
    }
    for i in (0..n).rev() {
        let mut s = [0.0 as Real; W];
        s.copy_from_slice(&t[i * W..i * W + W]);
        for (k, lrow) in l.chunks_exact(n).enumerate().take(n).skip(i + 1) {
            let lki = lrow[i];
            let xr = &t[k * W..k * W + W];
            for r in 0..W {
                s[r] -= lki * xr[r];
            }
        }
        let dg = l[i * n + i];
        let row = &mut t[i * W..i * W + W];
        for r in 0..W {
            row[r] = s[r] / dg;
        }
    }
}

/// The BACK half of `chol_solve_t` — for right-hand sides whose
/// forward pass ran separately (the limit columns: each has its own
/// pivot row, so their forward passes stay sparse and row-major, but
/// the dense back pass they share is worth the transpose).
pub fn chol_back_t<const W: usize>(n: usize, l: &[Real], t: &mut [Real]) {
    debug_assert_eq!(t.len(), W * n);
    for i in (0..n).rev() {
        let mut s = [0.0 as Real; W];
        s.copy_from_slice(&t[i * W..i * W + W]);
        for (k, lrow) in l.chunks_exact(n).enumerate().take(n).skip(i + 1) {
            let lki = lrow[i];
            let xr = &t[k * W..k * W + W];
            for r in 0..W {
                s[r] -= lki * xr[r];
            }
        }
        let dg = l[i * n + i];
        let row = &mut t[i * W..i * W + W];
        for r in 0..W {
            row[r] = s[r] / dg;
        }
    }
}

/// item 3 (2026-09-16): `chol_fwd_unit` + `chol_back_t` for W unit
/// right-hand sides at once, RHS-MINOR. The caller seeds `t` with +0.0
/// everywhere except `t[m_r·W + r] = sign_r`, with every pivot
/// `m_r >= m0`. The forward pass runs rows i ≥ m0 and chains k ≥ m0 for
/// ALL lanes, which is bitwise `chol_fwd_unit` per lane:
///  * i < m0: never touched, the seed's +0.0 == the `fill(0.0)`;
///  * m0 ≤ i < m_r: seed +0.0, every term is l·(+0.0) = ±0.0 (inductively
///    x[k] = +0.0 for k < i), `+0.0 − (±0.0) = +0.0`, and +0.0 / (positive
///    finite pivot, §22) = +0.0 — the fill's bytes;
///  * i = m_r: `sign − (±0.0)…` is exactly `sign`, so `sign / l_mm`;
///  * i > m_r: seed +0.0, the k in m0..m_r terms are ±0.0 and leave it
///    +0.0, then k = m_r..i is `chol_fwd_unit`'s own chain in order.
/// Lanes are independent accumulators (the `chol_solve_t` argument), so
/// the SIMD lane-vectorization is legal under strict IEEE.
pub fn chol_solve_unit_t<const W: usize>(n: usize, l: &[Real], m0: usize, t: &mut [Real]) {
    debug_assert_eq!(t.len(), W * n);
    for i in m0..n {
        let lrow = &l[i * n..i * n + i];
        let mut s = [0.0 as Real; W];
        s.copy_from_slice(&t[i * W..i * W + W]);
        for k in m0..i {
            let lk = lrow[k];
            let xr = &t[k * W..k * W + W];
            for r in 0..W {
                s[r] -= lk * xr[r];
            }
        }
        let dg = l[i * n + i];
        let row = &mut t[i * W..i * W + W];
        for r in 0..W {
            row[r] = s[r] / dg;
        }
    }
    chol_back_t::<W>(n, l, t);
}

/// `chol_solve_x4` at width 2 — the remainder rung of the batch ladder.
pub fn chol_solve_x2(n: usize, l: &[Real], x: &mut [Real]) {
    debug_assert_eq!(x.len(), 2 * n);
    let (a, b) = x.split_at_mut(n);
    for i in 0..n {
        let lrow = &l[i * n..i * n + i];
        let (mut s0, mut s1) = (a[i], b[i]);
        {
            let (ar, br) = (&a[..i], &b[..i]);
            for k in 0..i {
                let lk = lrow[k];
                s0 -= lk * ar[k];
                s1 -= lk * br[k];
            }
        }
        let dg = l[i * n + i];
        a[i] = s0 / dg;
        b[i] = s1 / dg;
    }
    for i in (0..n).rev() {
        let (mut s0, mut s1) = (a[i], b[i]);
        {
            let (ar, br) = (&a[..n], &b[..n]);
            for (k, lrow) in l.chunks_exact(n).enumerate().take(n).skip(i + 1) {
                let lki = lrow[i];
                s0 -= lki * ar[k];
                s1 -= lki * br[k];
            }
        }
        let dg = l[i * n + i];
        a[i] = s0 / dg;
        b[i] = s1 / dg;
    }
}

/// BACK substitution alone — verbatim the second half of `chol_solve`
/// (same operands, same ascending-k order), split out so a right-hand
/// side whose FORWARD pass has a closed form (`chol_fwd_unit`) can skip
/// straight to it.
pub fn chol_back(n: usize, l: &[Real], x: &mut [Real]) {
    for i in (0..n).rev() {
        let mut s = x[i];
        {
            let xr = &x[..n];
            for (k, lrow) in l.chunks_exact(n).enumerate().take(n).skip(i + 1) {
                s -= lrow[i] * xr[k];
            }
        }
        x[i] = s / l[i * n + i];
    }
}

/// `chol_back` jammed over 2 — the remainder rung.
pub fn chol_back_x2(n: usize, l: &[Real], x: &mut [Real]) {
    debug_assert_eq!(x.len(), 2 * n);
    let (a, b) = x.split_at_mut(n);
    for i in (0..n).rev() {
        let (mut s0, mut s1) = (a[i], b[i]);
        {
            let (ar, br) = (&a[..n], &b[..n]);
            for (k, lrow) in l.chunks_exact(n).enumerate().take(n).skip(i + 1) {
                let lki = lrow[i];
                s0 -= lki * ar[k];
                s1 -= lki * br[k];
            }
        }
        let dg = l[i * n + i];
        a[i] = s0 / dg;
        b[i] = s1 / dg;
    }
}

/// EXACT sparse forward substitution for the unit right-hand side
/// `sign·e_m` (`sign` = ±1.0), writing the full length-`n` result.
/// Mirrors the CUDA `chol_solve_unit_n` template.
///
/// Why the leading rows can be skipped rather than computed
/// (BITWISE argument — the RHS here is produced by `jr.fill(0.0)`
/// followed by a single `jr[m] = sign` STORE, so every non-pivot entry
/// is a genuine `+0.0` and no `-0.0` ever seeds the solve; note the row
/// is never formed by multiplying a zero row by −1, which is the one
/// construction that would introduce `-0.0` seeds):
///
///  * The diagonal is unconditionally positive and finite. §22's pivot
///    floor is applied through the NEGATED comparison `!(d >= FLOOR)`,
///    so even a NaN pivot is clamped to `PD_FLOOR_F64` — hence
///    `l[i·n+i] = sqrt(d) >= sqrt(1e-17) ≈ 3.2e-9 > 0` for EVERY i, no
///    precondition on the pose. (This is what makes the argument hold
///    unconditionally; before that fix a NaN pivot could reach here.)
///  * Row i < m: `s` starts at `+0.0` and each term subtracts
///    `l[i][k]·(+0.0) = ±0.0`. IEEE round-to-nearest gives
///    `+0.0 − (+0.0) = +0.0` AND `+0.0 − (−0.0) = +0.0`, so the sign of
///    the factor entry cannot leak in: `s` stays exactly `+0.0` and
///    `+0.0 / (positive finite) = +0.0`. The memset therefore
///    reproduces the dense bytes exactly, not merely the values.
///  * Row i = m: `s` starts at `sign` and subtracts m terms each
///    `±0.0`; subtracting a zero from a non-zero finite value is exact,
///    so `s == sign` bit for bit and `x[m] = sign / l[m·n+m]`.
///  * Row i > m: `s` starts at `+0.0` and the k < m terms leave it at
///    `+0.0` by the first bullet, so beginning the accumulation at
///    k = m gives the identical partial-sum chain. (The `s` seed is
///    kept as an explicit `+0.0` rather than negating the first
///    product — `+0.0 − (+0.0)` is `+0.0` where `−(+0.0)` would be
///    `-0.0`.)
pub fn chol_fwd_unit(n: usize, l: &[Real], m: usize, sign: Real, x: &mut [Real]) {
    debug_assert_eq!(x.len(), n);
    debug_assert!(m < n);
    x[..m].fill(0.0);
    x[m] = sign / l[m * n + m];
    for i in (m + 1)..n {
        let lrow = &l[i * n..i * n + i]; // row i, columns 0..i
        let mut s = 0.0;
        {
            let xr = &x[..i];
            for k in m..i {
                s -= lrow[k] * xr[k];
            }
        }
        x[i] = s / l[i * n + i];
    }
}

/// Full solve for a unit right-hand side `sign·e_m`: the sparse
/// forward pass above then the ordinary dense back substitution
/// (the back pass has no exploitable structure — by the time it runs
/// every entry above m is generally non-zero).
pub fn chol_solve_unit(n: usize, l: &[Real], m: usize, sign: Real, x: &mut [Real]) {
    chol_fwd_unit(n, l, m, sign, x);
    chol_back(n, l, x);
}

/// Total (linear, angular-about-world-origin) momentum at the FK in ws.
pub fn momentum(model: &Model, ws: &mut Workspace, v: &[Real]) -> ([Real; 3], [Real; 3]) {
    let nf = model.frames.len();
    let voff = model.voff();
    let (om0, vo0) = if model.free_base {
        ([v[3], v[4], v[5]], [v[0], v[1], v[2]])
    } else {
        ([0.0; 3], [0.0; 3])
    };
    let mut pl = [0.0; 3];
    let mut am = [0.0; 3];
    let mut add_body =
        |m: Real, com: [Real; 3], inertia: [Real; 3], r: &Mat3, o: [Real; 3], om: [Real; 3], vo: [Real; 3]| {
            if m > 0.0 {
                let e = mat_vec(r, com);
                let c = add3(o, e);
                let vc = add3(vo, cross3(om, e));
                let iw = inertia_world(r, inertia);
                pl = add3(pl, scale3(vc, m));
                am = add3(am, add3(mat_vec(&iw, om), cross3(c, scale3(vc, m))));
            }
        };
    let r0 = ws.r0;
    add_body(model.base_mass, model.base_com, model.base_inertia, &r0, ws.p0, om0, vo0);
    for k in 0..nf {
        let fr = &model.frames[k];
        let (oml, vol, ol) = if fr.parent < 0 {
            (om0, vo0, ws.p0)
        } else {
            let pk = fr.parent as usize;
            (ws.om[pk], ws.vo[pk], ws.p[pk])
        };
        let d = sub3(ws.p[k], ol);
        let vo = add3(vol, cross3(oml, d));
        let om = if fr.coord >= 0 {
            add3(oml, scale3(ws.axw[k], v[voff + fr.coord as usize]))
        } else {
            oml
        };
        ws.om[k] = om;
        ws.vo[k] = vo;
        let rk = ws.r[k];
        add_body(fr.mass, fr.com, fr.inertia, &rk, ws.p[k], om, vo);
    }
    (pl, am)
}

/// Total energy (kinetic + gravitational potential) at the FK in ws,
/// including the armature rotor term ½·a_j·q̇_j².
/// U = −Σ m g⃗·c (zero at the world origin).
pub fn energy(model: &Model, ws: &mut Workspace, v: &[Real]) -> Real {
    let nf = model.frames.len();
    let voff = model.voff();
    let g = model.gravity;
    let (om0, vo0) = if model.free_base {
        ([v[3], v[4], v[5]], [v[0], v[1], v[2]])
    } else {
        ([0.0; 3], [0.0; 3])
    };
    let mut e_tot = 0.0;
    let mut add_body =
        |m: Real, com: [Real; 3], inertia: [Real; 3], r: &Mat3, o: [Real; 3], om: [Real; 3], vo: [Real; 3]| {
            if m > 0.0 {
                let e = mat_vec(r, com);
                let c = add3(o, e);
                let vc = add3(vo, cross3(om, e));
                let iw = inertia_world(r, inertia);
                e_tot += 0.5 * m * dot3(vc, vc) + 0.5 * dot3(om, mat_vec(&iw, om));
                e_tot -= m * dot3(g, c);
            }
        };
    let r0 = ws.r0;
    add_body(model.base_mass, model.base_com, model.base_inertia, &r0, ws.p0, om0, vo0);
    for k in 0..nf {
        let fr = &model.frames[k];
        let (oml, vol, ol) = if fr.parent < 0 {
            (om0, vo0, ws.p0)
        } else {
            let pk = fr.parent as usize;
            (ws.om[pk], ws.vo[pk], ws.p[pk])
        };
        let d = sub3(ws.p[k], ol);
        let vo = add3(vol, cross3(oml, d));
        let om = if fr.coord >= 0 {
            add3(oml, scale3(ws.axw[k], v[voff + fr.coord as usize]))
        } else {
            oml
        };
        ws.om[k] = om;
        ws.vo[k] = vo;
        let rk = ws.r[k];
        add_body(fr.mass, fr.com, fr.inertia, &rk, ws.p[k], om, vo);
    }
    for j in 0..model.nq {
        e_tot += 0.5 * model.armature[j] * v[voff + j] * v[voff + j];
    }
    e_tot
}
