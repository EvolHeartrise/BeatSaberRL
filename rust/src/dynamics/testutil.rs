//! Shared test helpers: a seeded splitmix64 RNG (no external deps),
//! random in-limit humanoid states, and the P2 "muscle tone" stabilized
//! body (the oracle protocol's spring-damper stance — oracle README:
//! kd REPLACES the spec damping b_j, so the muscle model zeroes b).

use super::algo::{crba, rnea};
use super::fk::fk_all;
use super::model::{Model, HUMANOID, NQ};
use super::{State, Workspace};

pub static ZERO_DAMP29: [f64; NQ] = [0.0; NQ];

/// HUMANOID with joint damping removed — the muscle-tone kd is the
/// TOTAL dof damping and is applied as part of the actuated torque.
pub fn muscle_model() -> Model {
    let mut m = HUMANOID;
    m.damping = &ZERO_DAMP29;
    m
}

/// Oracle-protocol stabilizer gains: kp = τ_max/0.2, kd = max(b_j,
/// 2·√(kp·M_jj)) with M_jj from our CRBA at the neutral pose (the
/// refdata JSONs carry MuJoCo's values of the same formula).
pub fn muscle_gains() -> ([f64; NQ], [f64; NQ]) {
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let st = State::neutral(m);
    fk_all(m, &st, &mut ws);
    crba(m, &mut ws);
    let nv = m.nv();
    let mut kp = [0.0; NQ];
    let mut kd = [0.0; NQ];
    for j in 0..NQ {
        kp[j] = m.tau_max[j] / 0.2;
        let mjj = ws.m[(6 + j) * nv + (6 + j)];
        kd[j] = m.damping[j].max(2.0 * (kp[j] * mjj).sqrt());
    }
    (kp, kd)
}

/// Full 35-DoF gravity-compensation force at pose `q29` (v = 0):
/// (root wrench, joint torques) = RNEA(q, 0, 0) split.
pub fn gravity_comp(q29: &[f64; NQ]) -> ([f64; 6], [f64; NQ]) {
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let mut st = State::neutral(m);
    st.q.copy_from_slice(q29);
    fk_all(m, &st, &mut ws);
    let v0 = vec![0.0; m.nv()];
    rnea(m, &mut ws, &v0, None);
    let mut w = [0.0; 6];
    let mut t = [0.0; NQ];
    w.copy_from_slice(&ws.c[..6]);
    t.copy_from_slice(&ws.c[6..]);
    (w, t)
}

pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
}

/// Random state: root pose perturbed around the reference, every joint
/// uniform WITHIN its limits, velocities optionally randomized.
pub fn random_state(model: &Model, rng: &mut Rng, with_vel: bool) -> State {
    let mut st = State::neutral(model);
    if model.free_base {
        for i in 0..3 {
            st.p[i] = model.base_pos0[i] + rng.range(-0.5, 0.5);
        }
        let mut q = [0.0f64; 4];
        loop {
            for x in q.iter_mut() {
                *x = rng.range(-1.0, 1.0);
            }
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            if n > 0.1 {
                st.quat = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
                break;
            }
        }
    }
    for j in 0..model.nq {
        let (lo, hi) = model.limits[j];
        st.q[j] = rng.range(lo, hi);
    }
    if with_vel {
        for v in st.v.iter_mut() {
            *v = rng.range(-2.0, 2.0);
        }
    }
    st
}
