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
pub(crate) mod tests_soak;

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

/// R26 per-substep instrumentation counters. The STRUCT is always
/// compiled (so the FFI getter can hand back a zeroed one in the default
/// build), but nothing ever WRITES to it unless the `stats` feature is
/// on — see `Workspace::stats`, which only exists under that feature.
///
/// Every `*_sum` is a running total over substeps and every `*_max` the
/// worst single substep; divide a sum by `nsubsteps` for the mean. The
/// counters are cumulative until the owner pops them (lib.rs
/// `pop_dbg_stats`), exactly like `pd_hits` / `env_hits`.
///
/// `nsubsteps` is incremented once per completed `step::substep_opts`,
/// which is the denominator for ALL five rates: under bench traffic
/// every substep runs the constraint hook (`substep_constrained`), so
/// the solver-side counters share it. A P1-style unconstrained caller
/// would inflate the denominator for `nlim`/`ncon`/`cap_solves` — don't
/// mix the two in one collection window.
#[derive(Clone, Copy, Default)]
pub struct DbgStats {
    /// completed substeps (the denominator)
    pub nsubsteps: u64,
    /// solver.rs: `limset.len()` — joint-limit rows assembled
    pub nlim_sum: u64,
    pub nlim_max: u64,
    /// solver.rs: `contacts.len()` — active contact rows
    pub ncon_sum: u64,
    pub ncon_max: u64,
    /// solver.rs: `chol_solve_unit` calls actually made by the ω-cap
    /// pass (the engage branch only — a `limpos` dedupe hit or a cached
    /// column does NOT count)
    pub cap_solves_sum: u64,
    pub cap_solves_max: u64,
    /// step.rs: `chol_solve_unit` calls actually made by the step-11
    /// clamp block (the cache-miss branch only)
    pub s11_solves_sum: u64,
    pub s11_solves_max: u64,
}

impl DbgStats {
    /// Fold `o` in: sums add, maxes take the max. Used to aggregate the
    /// per-env counters of a vec-env into one report.
    pub fn merge(&mut self, o: &DbgStats) {
        self.nsubsteps += o.nsubsteps;
        self.nlim_sum += o.nlim_sum;
        self.nlim_max = self.nlim_max.max(o.nlim_max);
        self.ncon_sum += o.ncon_sum;
        self.ncon_max = self.ncon_max.max(o.ncon_max);
        self.cap_solves_sum += o.cap_solves_sum;
        self.cap_solves_max = self.cap_solves_max.max(o.cap_solves_max);
        self.s11_solves_sum += o.s11_solves_sum;
        self.s11_solves_max = self.s11_solves_max.max(o.s11_solves_max);
    }

    /// (name, value) pairs in a fixed order — the FFI layer turns these
    /// into the Python dict so the key set lives in exactly one place.
    pub fn as_pairs(&self) -> [(&'static str, u64); 9] {
        [
            ("nsubsteps", self.nsubsteps),
            ("nlim_sum", self.nlim_sum),
            ("nlim_max", self.nlim_max),
            ("ncon_sum", self.ncon_sum),
            ("ncon_max", self.ncon_max),
            ("cap_solves_sum", self.cap_solves_sum),
            ("cap_solves_max", self.cap_solves_max),
            ("s11_solves_sum", self.s11_solves_sum),
            ("s11_solves_max", self.s11_solves_max),
        ]
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
    // f64 Cholesky pivot-floor engagements (SPEC-v10 Divergences §22),
    // cumulative until the owner pops it (lib.rs pop_pd_hits)
    pub pd_hits: u64,
    // PRE-integration root-velocity-envelope engagements (SPEC-v10
    // Divergences §23 / step 12's first application), counted per
    // (env, substep) and cumulative until the owner pops it
    // (lib.rs pop_envelope_hits). A nonzero count means the root left
    // the kinematically reachable envelope — a solver artifact, i.e.
    // the trajectory stopped being physics (the env layer terminates
    // the episode on it; see lib.rs Body::escaped).
    pub env_hits: u64,
    // R26 instrumentation counters — EXISTS ONLY under the `stats`
    // feature so the shipping build's Workspace layout is untouched.
    #[cfg(feature = "stats")]
    pub stats: DbgStats,
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
            pd_hits: 0,
            env_hits: 0,
            #[cfg(feature = "stats")]
            stats: DbgStats::default(),
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
