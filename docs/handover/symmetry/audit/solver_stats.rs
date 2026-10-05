// Solver quality stats (scratch harness, identical in the base and fixed builds).
use super::*;
use crate::dynamics::step::{substep_wrench, ZERO_WRENCH};
use crate::dynamics::collide::{GROUND_CORNERS, GROUND_FRAMES};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> Real { (self.next() >> 11) as Real / (1u64 << 53) as Real }
}

#[derive(Default, Debug)]
struct Acc { n: f64, sum: f64, max: f64 }
impl Acc { fn add(&mut self, x: f64) { self.n += 1.0; self.sum += x; self.max = self.max.max(x); }
  fn s(&self) -> String { format!("mean {:.3e} max {:.3e} (n {})", self.sum / self.n.max(1.0), self.max, self.n) } }

#[test]
#[ignore]
fn zz_solver_stats() {
    let gear = joint_gear_rows(5);
    for (wname, arm_p, eps) in [("soak (random arm gears, leg hold)", 0.5, 40usize), ("stance (leg hold, idle arms)", 0.0, 10)] {
    let mut rng = Rng(7);
    let mut g_unmet = Acc::default(); let mut g_comp = Acc::default();
    let mut c_unmet = Acc::default(); let mut l_unmet = Acc::default();
    let mut cap_over = Acc::default(); let mut pen = Acc::default(); let mut pen_max = Acc::default();
    let mut lim_over = Acc::default(); let mut cap_after = Acc::default();
    let mut ticks = 0; let mut falls = 0;
    let mut tgt = [0.0 as Real; NQ];
    for (j, v) in Q_START_LEG { tgt[j] = v; }
    for _ in 0..eps {
        let mut b = Body::new(); b.reset();
        for _ in 0..40 {
            let mut a = [0u8; NQ];
            for j in 17..NQ { let u = -(b.st.q[j] - tgt[j]) * 3.0 - 0.3 * b.st.v[6 + j];
                a[j] = if u > 0.3 { 3 } else if u < -0.3 { 4 } else { 0 }; }
            for j in 3..17 { if rng.uniform() < arm_p { a[j] = 1 + (rng.next() % 4) as u8; } }
            let mut tau = [0.0 as Real; NQ];
            for j in 0..NQ { tau[j] = gear[j][a[j] as usize] * MODEL.tau_max[j]; }
            for _ in 0..8 {
                let cs = &mut b.cs;
                let mut hook = |m: &dynamics::model::Model, w: &mut dynamics::Workspace, q: &[Real], v: &mut [Real]| {
                    cs.h = dynamics::step::H_SUB;
                    cs.solve(m, w, q, v);
                    let ng = cs.ground_rows();
                    for gi in 0..cs.ground_contacts.len() {
                        let row = cs.ground_row(gi);
                        let jv = cs.jv(row, v);
                        let unmet = (cs.bias[row] - jv).max(0.0);
                        g_unmet.add(unmet);
                        g_comp.add(cs.lambda[row] * (jv - cs.bias[row]).max(0.0));
                    }
                    for ci in 0..cs.contacts.len() {
                        let row = ng + ci;
                        c_unmet.add((cs.bias[row] - cs.jv(row, v)).max(0.0));
                    }
                    let nd = ng + cs.contacts.len();
                    for (li, &(j, sign)) in cs.limit_set().iter().enumerate() {
                        let jv = sign * v[6 + j];
                        l_unmet.add((cs.bias[nd + li] - jv).max(0.0));
                    }
                    for j in 0..NQ { cap_over.add((v[6 + j].abs() - m.omega_cap[j]).max(0.0)); }
                };
                substep_wrench(MODEL, &mut b.st, &tau, &ZERO_WRENCH, dynamics::step::H_SUB, &mut b.ws, Some(&mut hook));
                // post-substep: penetration & limit overshoot
                dynamics::fk::fk_all(MODEL, &b.st, &mut b.ws);
                let mut pm: f64 = 0.0;
                for fr in GROUND_FRAMES { let (r, p) = (&b.ws.r[fr], &b.ws.p[fr]);
                    for c in &GROUND_CORNERS { let y = p[1] + r[1][0]*c[0] + r[1][1]*c[1] + r[1][2]*c[2];
                        if y < 0.0 { pen.add(-y); } pm = pm.max(-y); } }
                pen_max.add(pm.max(0.0));
                for j in 0..NQ { let (lo, hi) = MODEL.limits[j];
                    lim_over.add((lo - b.st.q[j]).max(b.st.q[j] - hi).max(0.0));
                    cap_after.add((b.st.v[6 + j].abs() - MODEL.omega_cap[j]).max(0.0)); }
            }
            b.refresh_fk();
            ticks += 1;
            if b.st.p[1] < 0.25 { falls += 1; break; }
        }
    }
    println!("== {wname}: {ticks} ticks, {falls} falls");
    println!("  ground normal unmet (m/s)   {}", g_unmet.s());
    println!("  ground complementarity      {}", g_comp.s());
    println!("  contact unmet (m/s)         {}", c_unmet.s());
    println!("  limit-row unmet (rad/s)     {}", l_unmet.s());
    println!("  cap overshoot post-solve    {}", cap_over.s());
    println!("  penetration (m, per corner) {}", pen.s());
    println!("  max penetration per substep {}", pen_max.s());
    println!("  limit overshoot post-step   {}", lim_over.s());
    println!("  cap overshoot post-step     {}", cap_after.s());
    }
}
