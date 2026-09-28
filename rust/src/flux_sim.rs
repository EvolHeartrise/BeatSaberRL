//! The Flux game mode's ENV side (SPEC.md §Flux): the Core glue around
//! the shared rules in `flux.rs` — chart install, the per-tick judge call
//! on the body's palms plus the env's own tallies, and the Flux obs
//! blocks. Everything that decides an outcome is in `flux.rs` (shared
//! with the WebXR game's wasm build); nothing here may.

use super::*;

impl Core {
    /// Install a Flux chart (flux sims only). Walls do not exist in
    /// Flux; any previous ones are cleared.
    pub(crate) fn set_flux_chart(&mut self, chart: Vec<flux::FluxTuple>) {
        assert!(self.flux, "set_flux_chart needs a sim built with flux=True");
        self.set_walls(Vec::new());
        self.chart = chart
            .into_iter()
            .map(|t| flux::flux_note(t).unwrap_or_else(|e| panic!("{e}")))
            .collect();
        self.finish_chart_install();
    }

    /// The Flux object pass of `step_inner` (replaces the saber cut
    /// loop): `flux::judge` on the body's palms, then the env's tallies
    /// (counters, score components, the per-colour previous-circle
    /// record) from the events it emitted. prev_* are last tick's
    /// wrist/hand-axis caches.
    pub(crate) fn flux_judge(
        &mut self,
        reward: &mut Real,
        prev_pos: &[[Real; 3]; 2],
        prev_dir: &[[Real; 3]; 2],
    ) {
        let p0 = [flux::palm(prev_pos[0], prev_dir[0]),
                  flux::palm(prev_pos[1], prev_dir[1])];
        let p1 = [
            flux::palm(self.body.wrist[0], self.body.blade[0]),
            flux::palm(self.body.wrist[1], self.body.blade[1]),
        ];
        for s in 0..2 {
            for k in 0..3 {
                self.palm_vel[s][k] = (p1[s][k] - p0[s][k]) / self.body.rate.dt;
            }
        }
        let reach_z = flux::reach_z(
            self.body.shoulder_z[0].max(self.body.shoulder_z[1]));
        let first = self.events_buf.len();
        flux::judge(&self.chart, &mut self.status, self.first_pending,
                    self.min_speed, self.time, DT, self.body.head, reach_z,
                    &p0, &p1, reward, &mut self.events_buf);
        for k in first..self.events_buf.len() {
            let (i, code, pts, pre, acc) = self.events_buf[k];
            match code {
                0 => {
                    // integer points: the f32 event values are exact
                    self.good_cuts += 1;
                    self.score += pts as Real;
                    self.score_pre += pre as Real;
                    self.score_acc += acc as Real;
                }
                4 => self.misses += 1,
                _ => self.bad_cuts += 1, // 2 angle, 3 wrong hand
            }
            // env v45 previous-note record: every resolved CIRCLE, on
            // its own colour (good, wrong hand or miss); bars never
            let n = &self.chart[i as usize];
            if n.kind == KIND_CIRCLE {
                self.prev_idx[n.color as usize] = i;
                self.prev_valid[n.color as usize] = true;
            }
        }
    }

    /// Flux hand blocks (HAND_FEAT lanes each): palm x/y head-relative,
    /// palm z world, palm velocity /V_NORM, hand axis; lanes 9–10 zero
    /// (the standard swing meter / follow-through lanes have no Flux
    /// meaning).
    pub(crate) fn write_flux_hands(&self, obs: &mut [f32]) {
        for (s, base) in [(0usize, OFF_HAND_L), (1usize, OFF_HAND_R)] {
            let b = &self.body;
            let p = flux::palm(b.wrist[s], b.blade[s]);
            obs[base] = ((p[0] - b.head[0]) / X_NORM) as f32;
            obs[base + 1] = ((p[1] - b.head[1]) / Y_NORM) as f32;
            obs[base + 2] = p[2] as f32;
            for k in 0..3 {
                obs[base + 3 + k] = (self.palm_vel[s][k] / V_NORM) as f32;
                obs[base + 6 + k] = b.blade[s][k] as f32;
            }
        }
    }

    /// Flux entity window: the standard nearest-first selection over
    /// pending objects (no walls in Flux), Flux lane meanings (SPEC.md
    /// §Flux obs): 0–3 as standard, 4/5 circle colour, 6/7 bar axis u,
    /// 8 is_circle, 9 is_bar, 10 bar length (m), 11 sweet offset,
    /// 12/13 sweet-spot head-relative x/y; 14–17 zero.
    pub(crate) fn write_flux_entities(&self, obs: &mut [f32]) {
        let z_off = self.body.head[2] - PLANE_Z0;
        let z_lo = self.body.head[2];
        let head = &self.body.head;
        let mut sel = [(Real::INFINITY, false, 0usize); N_NOTE_SLOTS];
        let mut n_sel = 0usize;
        for i in self.first_pending..self.chart.len() {
            if self.status[i] != 0 {
                continue;
            }
            let n = &self.chart[i];
            let dz_lb = (n.hit_time - self.time) * self.min_speed - PLANE_Z0;
            if n_sel == N_NOTE_SLOTS && dz_lb > 0.0
                && dz_lb * dz_lb >= sel[N_NOTE_SLOTS - 1].0
            {
                break;
            }
            let dz = (n.hit_time - self.time) * n.speed - PLANE_Z0;
            if dz < 0.0 {
                continue;
            }
            let dx = n.x - head[0];
            let dy = n.y - head[1];
            slot_insert(&mut sel, &mut n_sel, (dx * dx + dy * dy + dz * dz, false, i));
        }
        for slot in 0..n_sel {
            let (_, _, idx) = sel[slot];
            let base = OFF_NOTES + slot * NOTE_FEAT;
            let n = &self.chart[idx];
            let zu = (n.hit_time - self.time) * n.speed + z_off;
            obs[base] = ((n.x - head[0]) / X_NORM) as f32;
            obs[base + 1] = ((n.y - head[1]) / Y_NORM) as f32;
            obs[base + 2] = z_lane(zu, z_lo) as f32;
            obs[base + 3] = (-n.speed / Z_SCALE) as f32;
            if n.kind == KIND_CIRCLE {
                obs[base + 4 + n.color as usize] = 1.0;
                obs[base + 8] = 1.0;
            } else {
                let (ux, uy) = (-n.sin_a, n.cos_a);
                let s_sweet = n.sweet * 0.5 * n.len;
                obs[base + 6] = ux as f32;
                obs[base + 7] = uy as f32;
                obs[base + 9] = 1.0;
                obs[base + 10] = n.len as f32;
                obs[base + 11] = n.sweet as f32;
                obs[base + 12] = ((n.x + s_sweet * ux - head[0]) / X_NORM) as f32;
                obs[base + 13] = ((n.y + s_sweet * uy - head[1]) / Y_NORM) as f32;
            }
            obs[OFF_NOTE_MASK + slot] = 1.0;
        }
    }
}

