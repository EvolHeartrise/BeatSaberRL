//! Env v10 rigid-body dynamics core (P1). Contract: SPEC-v10.md — §1
//! model tables (model.rs), §1.3 tree FK (fk.rs), §3 solver pieces CRBA
//! / RNEA / dense Cholesky (algo.rs) and the per-substep pipeline
//! (step.rs). f64 throughout; no external math deps; expression trees
//! written for literal torch/CUDA mirroring (P4/P5).
//!
//! P2 adds §2 collision (collide.rs: 12 colliders, 55-pair table,
//! capsule narrow phase) and §3 steps 7–9 (solver.rs: foot welds +
//! contact normal rows, PGS-12) plugged into step.rs's constraint seam
//! — `solver::substep_constrained` is the full v10 substep.

pub mod algo;
pub mod collide;
pub mod fk;
pub mod model;
pub mod solver;
pub mod step;

#[cfg(test)]
mod testjson;
#[cfg(test)]
mod testutil;
#[cfg(test)]
mod tests_collide;
#[cfg(test)]
mod tests_constraints;
#[cfg(test)]
mod tests_dyn;
#[cfg(test)]
mod tests_fk;
#[cfg(test)]
mod tests_oracle;
#[cfg(test)]
mod tests_pendulum;
#[cfg(test)]
mod tests_soak;

use model::Model;

/// Full generalized state. `q` is the 29 (nq) actuated angles in §1.4
/// order; `v` is the nv velocity vector — free base: [v_lin world of
/// the root origin (3), omega world (3), qdot (nq)]; fixed base: qdot
/// only. Root quaternion wxyz (scalar first), replay v10 convention.
#[derive(Clone)]
pub struct State {
    pub p: [f64; 3],
    pub quat: [f64; 4],
    pub q: Vec<f64>,
    pub v: Vec<f64>,
}

impl State {
    /// Reference (neutral) state: base at base_pos0, identity
    /// orientation, all angles/velocities zero.
    pub fn neutral(model: &Model) -> State {
        State {
            p: model.base_pos0,
            quat: [1.0, 0.0, 0.0, 0.0],
            q: vec![0.0; model.nq],
            v: vec![0.0; model.nv()],
        }
    }
}

/// Preallocated buffers for one model — no allocation inside a substep.
pub struct Workspace {
    // FK
    pub r0: fk::Mat3,
    pub p0: [f64; 3],
    pub r: Vec<fk::Mat3>,   // world rotation per frame
    pub p: Vec<[f64; 3]>,   // world origin per frame
    pub axw: Vec<[f64; 3]>, // world revolute axis per frame
    // dense matrices (row-major nv×nv)
    pub m: Vec<f64>, // mass matrix
    pub l: Vec<f64>, // Cholesky factor (lower)
    // vectors (nv)
    pub c: Vec<f64>,     // RNEA output / bias forces
    pub rhs: Vec<f64>,   // tau − c, then the accel after the solve
    pub vstar: Vec<f64>, // unconstrained post-solve velocity
    pub sol: Vec<f64>,      // M⁻¹e_j scratch (clamp impulses, P2 rows)
    pub hit: Vec<bool>,     // per-joint limit-hit flags (step 11)
    pub cached: Vec<bool>,  // step 11: M⁻¹e_j column cached?
    pub solcache: Vec<f64>, // step 11: cached M⁻¹e_j columns (nq×nv)
    // RNEA / velocity-pass scratch, per frame
    om: Vec<[f64; 3]>, // body angular velocity
    al: Vec<[f64; 3]>, // body angular acceleration
    vo: Vec<[f64; 3]>, // frame-origin (material point) velocity
    ao: Vec<[f64; 3]>, // frame-origin acceleration
    f: Vec<[f64; 3]>,  // backward pass: net force
    n: Vec<[f64; 3]>,  // backward pass: net moment about frame origin
    // CRBA composite scratch, per frame
    cm: Vec<f64>,       // composite mass
    cc: Vec<[f64; 3]>,  // composite COM (world)
    ci: Vec<fk::Mat3>,  // composite inertia about the COM (world)
}

impl Workspace {
    pub fn new(model: &Model) -> Workspace {
        let nf = model.frames.len();
        let nv = model.nv();
        Workspace {
            r0: fk::IDENT3,
            p0: [0.0; 3],
            r: vec![fk::IDENT3; nf],
            p: vec![[0.0; 3]; nf],
            axw: vec![[0.0; 3]; nf],
            m: vec![0.0; nv * nv],
            l: vec![0.0; nv * nv],
            c: vec![0.0; nv],
            rhs: vec![0.0; nv],
            vstar: vec![0.0; nv],
            sol: vec![0.0; nv],
            hit: vec![false; model.nq],
            cached: vec![false; model.nq],
            solcache: vec![0.0; model.nq * nv],
            om: vec![[0.0; 3]; nf],
            al: vec![[0.0; 3]; nf],
            vo: vec![[0.0; 3]; nf],
            ao: vec![[0.0; 3]; nf],
            f: vec![[0.0; 3]; nf],
            n: vec![[0.0; 3]; nf],
            cm: vec![0.0; nf],
            cc: vec![[0.0; 3]; nf],
            ci: vec![fk::IDENT3; nf],
        }
    }
}
