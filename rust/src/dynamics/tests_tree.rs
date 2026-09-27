//! audit #10 gates — the tree-sparse, fill-free factorization of M
//! (algo.rs `TreeChol`, SPEC-v10 Divergences §25):
//!   · the humanoid's elimination layout (post-order, permutation,
//!     profile, nnz 341, chains ≤ 16);
//!   · BITWISE: the tree factor / solves are the dense Cholesky of the
//!     permuted matrix P·M·Pᵀ with its structural zeros skipped (and
//!     those zeros are exactly +0.0);
//!   · BITWISE: on `TreeChol::dense` the tc_* routines ARE the old dense
//!     root-first arithmetic (`chol_factor_floored` / `chol_solve` /
//!     `chol_solve_unit`) — the reference the per-substep gate uses;
//!   · ROUNDING-LEVEL: tree solves vs the old root-first dense solves on
//!     random SPD mass matrices from real states;
//!   · the per-substep old-vs-new solver comparison (#9's discipline),
//!     contact mode and the free-floating (no ground) harness mode;
//!   · the pivot floor on the new pivots (singular matrix).

use super::algo::{
    chol_factor, chol_factor_floored, chol_solve, chol_solve_unit, crba, tc_factor,
    tc_solve_unit_v, tc_solve_v, TreeChol,
};
use super::fk::fk_all;
use super::model::{HUMANOID, NQ, NV};
use super::solver::ConstraintSet;
use super::step::substep_wrench;
use super::testutil::{random_state, Rng};
use super::{State, Workspace};

const H: f64 = 1.0 / 240.0;

/// P·M·Pᵀ as a dense n×n row-major matrix (both triangles).
fn permuted(tc: &TreeChol, m: &[f64]) -> Vec<f64> {
    let n = tc.n;
    let mut p = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let (a, b) = (tc.e2v[i], tc.e2v[j]);
            let (r, c) = (a.max(b), a.min(b));
            p[i * n + j] = m[r * n + c];
        }
    }
    p
}

#[test]
fn humanoid_tree_layout() {
    let tc = TreeChol::tree(&HUMANOID);
    assert_eq!(tc.n, NV);
    // e-order = reversed frame (DFS pre-)order: r leg, l leg, r arm,
    // l arm, spine, then the base v0..v5
    #[rustfmt::skip]
    let want: [usize; NV] = [
        34, 33, 32, 31, 30, 29,        // r leg  (frames 29..24)
        28, 27, 26, 25, 24, 23,        // l leg  (frames 23..18)
        21, 22, 20, 19, 18, 16, 17,    // r arm  (frames 17..11: bend, dev, roll, elbow, twist, pitch, yaw)
        14, 15, 13, 12, 11, 9, 10,     // l arm  (frames 10..4)
        8, 7, 6,                       // spine  (frames 2, 1, 0)
        0, 1, 2, 3, 4, 5,              // base
    ];
    assert_eq!(tc.e2v, want.to_vec());
    #[rustfmt::skip]
    let lo: [usize; NV] = [
        0, 0, 0, 0, 0, 0,
        6, 6, 6, 6, 6, 6,
        12, 12, 12, 12, 12, 12, 12,
        19, 19, 19, 19, 19, 19, 19,
        12, 12, 12,
        0, 0, 0, 0, 0, 0,
    ];
    assert_eq!(tc.lo, lo.to_vec());
    assert_eq!(tc.nnz(), 341, "structural non-zeros of L");
    let longest = (0..NV).map(|e| tc.anc_off[e + 1] - tc.anc_off[e] + 1).max().unwrap();
    assert_eq!(longest, 16, "deepest chain: a wrist (7 arm + 3 spine + 6 base)");
    // the permutation is the stacking/actuation swap and nothing else
    // inside each chain: v2e ∘ e2v = id
    for e in 0..NV {
        assert_eq!(tc.v2e[tc.e2v[e]], e);
    }
    // factor MAC count (Σ over columns of |desc|·(|anc|+1)) — the audit
    // estimate ~1.6k vs ~7.1k dense
    let macs: usize = (0..NV)
        .map(|j| (j - tc.lo[j]) * (tc.anc_off[j + 1] - tc.anc_off[j] + 1))
        .sum();
    let dense = TreeChol::dense(NV);
    let dmacs: usize = (0..NV)
        .map(|j| (j - dense.lo[j]) * (dense.anc_off[j + 1] - dense.anc_off[j] + 1))
        .sum();
    let solve: usize = (0..NV).map(|e| e - tc.lo[e]).sum::<usize>() + tc.anc.len();
    println!("factor MACs tree {macs} vs dense {dmacs}; dense-RHS solve {solve} vs {}", NV * (NV - 1));
    assert_eq!((macs, dmacs, solve), (1742, 7140, 612));
}

/// The tree factor IS the dense no-pivot Cholesky of P·M·Pᵀ, bit for
/// bit, with every skipped entry an exact +0.0 of the dense factor; the
/// tree solves (dense and unit RHS) likewise bitwise the dense permuted
/// solves. 300 random in-limit states (random root pose too).
#[test]
fn tree_factor_matches_dense_permuted() {
    let m = &HUMANOID;
    let n = NV;
    let mut ws = Workspace::new(m);
    let mut rng = Rng(0x5EED_0000_0000_A010);
    let mut esc = vec![0.0; n];
    for pose in 0..300 {
        let st = random_state(m, &mut rng, false);
        fk_all(m, &st, &mut ws);
        crba(m, &mut ws);
        let tc = ws.tc.clone();
        assert_eq!(tc_factor(&tc, &ws.m, &mut ws.l), 0);
        let mp = permuted(&tc, &ws.m);
        let mut lp = vec![f64::NAN; n * n];
        assert_eq!(chol_factor_floored(n, &mp, &mut lp), 0);
        for i in 0..n {
            for k in 0..=i {
                if k >= tc.lo[i] {
                    assert_eq!(
                        ws.l[i * n + k].to_bits(),
                        lp[i * n + k].to_bits(),
                        "pose {pose}: L({i},{k})"
                    );
                } else {
                    assert_eq!(lp[i * n + k].to_bits(), 0.0f64.to_bits(), "pose {pose}: fill at ({i},{k})");
                }
            }
        }
        // dense-RHS solve
        let b: Vec<f64> = (0..n).map(|_| rng.range(-3.0, 3.0)).collect();
        let mut x = b.clone();
        tc_solve_v(&tc, &ws.l, &mut x, &mut esc);
        let mut y: Vec<f64> = (0..n).map(|e| b[tc.e2v[e]]).collect();
        chol_solve(n, &lp, &mut y);
        for e in 0..n {
            assert_eq!(x[tc.e2v[e]].to_bits(), y[e].to_bits(), "pose {pose}: solve word {e}");
        }
        // unit-RHS solves (both signs)
        for (vm, sign) in [(6 + (pose % NQ), 1.0), (6 + ((pose * 7) % NQ), -1.0), (pose % 6, 1.0)] {
            tc_solve_unit_v(&tc, &ws.l, vm, sign, &mut x, &mut esc);
            chol_solve_unit(n, &lp, tc.v2e[vm], sign, &mut y);
            for e in 0..n {
                assert_eq!(x[tc.e2v[e]].to_bits(), y[e].to_bits(), "pose {pose}: unit {vm} word {e}");
            }
        }
    }
}

/// On the DENSE layout the tc_* routines are the old root-first
/// arithmetic bit for bit — what licenses `TreeChol::dense` as the
/// old-formulation reference in the per-substep gates below.
#[test]
fn dense_layout_is_the_old_arithmetic() {
    let m = &HUMANOID;
    let n = NV;
    let mut ws = Workspace::new(m);
    let dense = TreeChol::dense(n);
    let mut rng = Rng(0x5EED_0000_0000_A011);
    let mut esc = vec![0.0; n];
    for pose in 0..100 {
        let st = random_state(m, &mut rng, false);
        fk_all(m, &st, &mut ws);
        crba(m, &mut ws);
        let mut lo = vec![0.0; n * n];
        assert_eq!(chol_factor_floored(n, &ws.m, &mut lo), 0);
        assert_eq!(tc_factor(&dense, &ws.m, &mut ws.l), 0);
        for i in 0..n {
            for k in 0..=i {
                assert_eq!(ws.l[i * n + k].to_bits(), lo[i * n + k].to_bits(), "pose {pose}: L({i},{k})");
            }
        }
        let b: Vec<f64> = (0..n).map(|_| rng.range(-3.0, 3.0)).collect();
        let (mut x, mut y) = (b.clone(), b.clone());
        tc_solve_v(&dense, &ws.l, &mut x, &mut esc);
        chol_solve(n, &lo, &mut y);
        assert!(x.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()), "pose {pose}: solve");
        for vm in [pose % n, 6 + (pose % NQ)] {
            tc_solve_unit_v(&dense, &ws.l, vm, -1.0, &mut x, &mut esc);
            chol_solve_unit(n, &lo, vm, -1.0, &mut y);
            assert!(x.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()), "pose {pose}: unit {vm}");
        }
    }
}

/// ROUNDING-LEVEL agreement with the old root-first dense Cholesky on
/// random SPD mass matrices from real states: M⁻¹b (dense RHS) and
/// M⁻¹e_j (every coordinate), relative to the solution's max-norm.
/// Also the relative residual ‖M x − b‖∞/‖b‖∞ of both orders (the same
/// order of magnitude: both are backward-stable no-pivot Cholesky
/// factorizations of the same SPD matrix).
#[test]
fn tree_solves_match_old_dense_to_rounding() {
    let m = &HUMANOID;
    let n = NV;
    let mut ws = Workspace::new(m);
    let mut rng = Rng(0x5EED_0000_0000_A012);
    let mut esc = vec![0.0; n];
    let (mut worst, mut worst_unit) = (0.0f64, 0.0f64);
    let (mut res_old, mut res_new) = (0.0f64, 0.0f64);
    let mut lo = vec![0.0; n * n];
    let resid = |mm: &[f64], x: &[f64], b: &[f64]| {
        let mut r = 0.0f64;
        let mut s = 0.0f64;
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                let (a, c) = (i.max(j), i.min(j));
                acc += mm[a * n + c] * x[j];
            }
            r = r.max((acc - b[i]).abs());
            s = s.max(b[i].abs());
        }
        r / s
    };
    for _pose in 0..300 {
        let st = random_state(m, &mut rng, false);
        fk_all(m, &st, &mut ws);
        crba(m, &mut ws);
        assert!(chol_factor(n, &ws.m, &mut lo));
        assert_eq!(tc_factor(&ws.tc.clone(), &ws.m, &mut ws.l), 0);
        let b: Vec<f64> = (0..n).map(|_| rng.range(-100.0, 100.0)).collect();
        let (mut x, mut y) = (b.clone(), b.clone());
        tc_solve_v(&ws.tc, &ws.l, &mut x, &mut esc);
        chol_solve(n, &lo, &mut y);
        let sc = y.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let d = x.iter().zip(&y).fold(0.0f64, |a, (p, q)| a.max((p - q).abs()));
        worst = worst.max(d / sc);
        res_old = res_old.max(resid(&ws.m, &y, &b));
        res_new = res_new.max(resid(&ws.m, &x, &b));
        for vm in 0..n {
            tc_solve_unit_v(&ws.tc, &ws.l, vm, 1.0, &mut x, &mut esc);
            chol_solve_unit(n, &lo, vm, 1.0, &mut y);
            let sc = y.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            let d = x.iter().zip(&y).fold(0.0f64, |a, (p, q)| a.max((p - q).abs()));
            worst_unit = worst_unit.max(d / sc);
        }
    }
    println!(
        "audit #10 vs root-first dense: M⁻¹b worst rel {worst:.3e}, M⁻¹e_j worst rel {worst_unit:.3e}; \
         residual old {res_old:.3e} new {res_new:.3e}"
    );
    assert!(worst < 1e-11 && worst_unit < 1e-11);
}

/// The pivot floor on the NEW pivots: the armature-less gimbal-locked
/// matrix with the dependent rows duplicated EXACTLY (tests_dyn's
/// §22 construction) is singular in any elimination order, so the tree
/// factor must engage the floor, count it, and stay finite; the solve
/// through it stays finite. (Where it engages moves with the order —
/// see the algo.rs header: that is the one semantic difference, and it
/// is confined to matrices the floor already reports as degenerate.)
#[test]
fn tree_pivot_floor_on_singular_matrix() {
    let m = &HUMANOID;
    let nv = m.nv();
    let mut ws = Workspace::new(m);
    let mut st = State::neutral(m);
    st.q[3] = m.limits[3].0;
    st.q[10] = m.limits[10].0;
    st.q[4] = 0.3;
    st.q[5] = -0.2;
    fk_all(m, &st, &mut ws);
    crba(m, &mut ws);
    let mut mm = ws.m.clone();
    for j in 0..NQ {
        mm[(6 + j) * nv + (6 + j)] -= m.armature[j];
    }
    for (a, b) in [(10usize, 11usize), (17, 18)] {
        for k in 0..nv {
            mm[b * nv + k] = mm[a * nv + k];
            mm[k * nv + b] = mm[k * nv + a];
        }
        mm[b * nv + b] = mm[a * nv + a];
    }
    let tc = ws.tc.clone();
    let hits = tc_factor(&tc, &mm, &mut ws.l);
    println!("tree factor floor engagements on the doubly-singular matrix: {hits}");
    assert!(hits >= 1, "floor must engage on the singular matrix");
    for e in 0..nv {
        for k in tc.lo[e]..=e {
            assert!(ws.l[e * nv + k].is_finite(), "factor must stay finite at ({e},{k})");
        }
    }
    let mut rhs: Vec<f64> = (0..nv).map(|i| 0.1 * (i as f64 + 1.0)).collect();
    let mut esc = vec![0.0; nv];
    tc_solve_v(&tc, &ws.l, &mut rhs, &mut esc);
    assert!(rhs.iter().all(|x| x.is_finite()), "solve must stay finite");
}

/// audit #10 EQUIVALENCE GATE, #9's discipline: at every substep of
/// random-gear trajectories (contact mode, both amplitudes, and the
/// free-floating harness mode — the tier the v10 weld mode held until
/// 2026-09-27) the SAME pre-substep state is stepped twice — once with the
/// tree layout, once with `TreeChol::dense` (= the old root-first
/// arithmetic, bitwise — `dense_layout_is_the_old_arithmetic`) — and
/// the velocity leaving the step-7..9 constraint stage (which includes
/// the step-5/6 factor and v* solve) is compared, before steps 10/11
/// whose clamps are discontinuous. The reference trajectory then
/// continues from the TREE result, so every substep compares identical
/// inputs. Exact arithmetic makes the two identical; checked to 1e-9
/// (|Δv| / max(|v|, 1)), observed worst printed.
#[test]
fn tree_matches_dense_per_substep() {
    let m = &HUMANOID;
    let mut report = Vec::new();
    for (mode, seed, amp) in [
        ("contact", 0x5EED_0000_0000_A0C1u64, 1.0),
        ("contact", 0x5EED_0000_0000_A0C2, 0.25),
        ("floating", 0x5EED_0000_0000_A0C3, 1.0),
    ] {
        let mk = || ConstraintSet::new(m, true, mode == "contact");
        let mut ws = Workspace::new(m);
        let mut ws_ref = Workspace::new(m);
        ws_ref.tc = TreeChol::dense(m.nv());
        let (mut cs, mut cs_ref) = (mk(), mk());
        let mut st = State::neutral(m);
        let mut rng = Rng(seed);
        let mut tau = [0.0; NQ];
        let (mut worst, mut n, mut same_end) = (0.0f64, 0usize, 0usize);
        for tick in 0..480 {
            if tick % 8 == 0 {
                for j in 0..NQ {
                    let g = [0.0, 1.0, -1.0, 0.25, -0.25][rng.next_u64() as usize % 5];
                    tau[j] = amp * g * m.tau_max[j];
                }
            }
            let mut st_ref = st.clone();
            let run = |st: &mut State, ws: &mut Workspace, cs: &mut ConstraintSet| {
                let mut out = Vec::new();
                cs.h = H;
                let mut hook = |mm: &super::model::Model, w: &mut Workspace, q: &[f64], v: &mut [f64]| {
                    cs.solve(mm, w, q, v);
                    out = v.to_vec();
                };
                substep_wrench(m, st, &tau, &super::step::ZERO_WRENCH, H, ws, Some(&mut hook));
                out
            };
            let va = run(&mut st, &mut ws, &mut cs);
            let vb = run(&mut st_ref, &mut ws_ref, &mut cs_ref);
            let e = va.iter().zip(&vb).fold(0.0f64, |a, (p, q)| a.max((p - q).abs() / p.abs().max(1.0)));
            assert!(e < 1e-9, "{mode} seed {seed:x} substep {tick}: tree vs dense |dv| {e:.3e}");
            worst = worst.max(e);
            n += 1;
            let de = st.v.iter().zip(&st_ref.v).fold(0.0f64, |a, (p, q)| a.max((p - q).abs() / p.abs().max(1.0)));
            same_end += (de < 1e-9) as usize;
        }
        assert_eq!(ws.pd_hits + ws_ref.pd_hits, 0);
        report.push(format!("{mode}/{amp}: {n} substeps worst {worst:.3e} ({same_end} post-clamp states within 1e-9)"));
    }
    println!("audit #10 per-substep tree vs dense: {}", report.join("; "));
}
