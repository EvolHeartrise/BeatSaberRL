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

use super::fk::{add3, cross3, dot3, mat_vec, scale3, sub3, Mat3};
use super::model::Model;
use super::Workspace;

/// I_world = R · diag(i) · R^T (about the body COM). Upper triangle
/// computed, lower mirrored — bitwise symmetric by construction (the
/// two orders (r_a·i)·r_b vs (r_b·i)·r_a round differently).
fn inertia_world(r: &Mat3, i: [f64; 3]) -> Mat3 {
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
fn add_shift(i: &mut Mat3, m: f64, d: [f64; 3]) {
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

/// CRBA: fills ws.m (nv×nv, row-major) from the FK in ws. Composite
/// rigid bodies accumulated leaf→root; column j of M = the generalized
/// momentum produced by unit rate of coordinate j (moment of momentum
/// about each ancestor pivot along its axis). Both triangles written
/// from the same value — M is symmetric BITWISE by construction.
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
                    ws.m[ia * nv + j] = val;
                    ws.m[j * nv + ia] = val;
                }
                anc = fa.parent;
            }
            if model.free_base {
                // root rows: translation conjugate = h_lin, rotation
                // conjugate = moment of momentum about the root origin
                let lr = add3(ha, cross3(sub3(c, ws.p0), hl));
                for t in 0..3 {
                    ws.m[t * nv + j] = hl[t];
                    ws.m[j * nv + t] = hl[t];
                    ws.m[(3 + t) * nv + j] = lr[t];
                    ws.m[j * nv + (3 + t)] = lr[t];
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
        for i in 0..3 {
            for j in 0..3 {
                ws.m[i * nv + j] = if i == j { bm } else { 0.0 };
                ws.m[(3 + i) * nv + (3 + j)] = rr[i][j];
            }
        }
        let sk = [
            [0.0, -d[2], d[1]],
            [d[2], 0.0, -d[0]],
            [-d[1], d[0], 0.0],
        ];
        for i in 0..3 {
            for j in 0..3 {
                let val = bm * sk[i][j];
                ws.m[(3 + i) * nv + j] = val;
                ws.m[j * nv + (3 + i)] = val;
            }
        }
    }
}

/// RNEA inverse dynamics: generalized force required to produce
/// coordinate accelerations `acc` (None = all-zero: the bias-force
/// call C(q, v)) at state velocity `v`, INCLUDING gravity
/// (model.gravity enters as F = m·(a_com − g)). Requires fk_all in ws;
/// result in ws.c.
pub fn rnea(model: &Model, ws: &mut Workspace, v: &[f64], acc: Option<&[f64]>) {
    let nf = model.frames.len();
    let nv = model.nv();
    let voff = model.voff();
    let g = model.gravity;
    debug_assert_eq!(v.len(), nv);
    debug_assert!(acc.map_or(true, |a| a.len() == nv));
    let a_of = |i: usize| acc.map_or(0.0, |a| a[i]);

    // base kinematics: material point at the root origin
    let (om0, al0, vo0, ao0) = if model.free_base {
        (
            [v[3], v[4], v[5]],
            [a_of(3), a_of(4), a_of(5)],
            [v[0], v[1], v[2]],
            [a_of(0), a_of(1), a_of(2)],
        )
    } else {
        ([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3])
    };

    // forward pass: velocities + accelerations, then per-body inertial
    // force F (about COM) and moment n (about the frame origin)
    for k in 0..nf {
        let fr = &model.frames[k];
        let (oml, all, vol, aol, ol) = if fr.parent < 0 {
            (om0, al0, vo0, ao0, ws.p0)
        } else {
            let pk = fr.parent as usize;
            (ws.om[pk], ws.al[pk], ws.vo[pk], ws.ao[pk], ws.p[pk])
        };
        let d = sub3(ws.p[k], ol);
        let vo = add3(vol, cross3(oml, d));
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
        ws.vo[k] = vo;
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

/// Dense Cholesky M = L·Lᵀ (lower, no pivoting, fixed order — SPEC-v10
/// §3 step 5). Returns false if a pivot is not strictly positive.
pub fn chol_factor(n: usize, m: &[f64], l: &mut [f64]) -> bool {
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
        for i in (j + 1)..n {
            l[j * n + i] = 0.0; // keep the strict upper triangle clean
        }
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
pub fn chol_solve(n: usize, l: &[f64], x: &mut [f64]) {
    for i in 0..n {
        let mut s = x[i];
        for k in 0..i {
            s -= l[i * n + k] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = x[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
}

/// Total (linear, angular-about-world-origin) momentum at the FK in ws.
pub fn momentum(model: &Model, ws: &mut Workspace, v: &[f64]) -> ([f64; 3], [f64; 3]) {
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
        |m: f64, com: [f64; 3], inertia: [f64; 3], r: &Mat3, o: [f64; 3], om: [f64; 3], vo: [f64; 3]| {
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
pub fn energy(model: &Model, ws: &mut Workspace, v: &[f64]) -> f64 {
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
        |m: f64, com: [f64; 3], inertia: [f64; 3], r: &Mat3, o: [f64; 3], om: [f64; 3], vo: [f64; 3]| {
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
