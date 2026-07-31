//! Shared test helpers: a seeded splitmix64 RNG (no external deps) and
//! random in-limit humanoid states.

use super::model::Model;
use super::State;

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
