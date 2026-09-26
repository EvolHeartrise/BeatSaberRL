//! Flux game mode (SPEC.md §Flux) — a second Beat Saber mode on the
//! same body, physics, track and reward terms as the standard game.
//!
//! No sabers: each hand is its PALM CENTRE, a point PALM_OFF along the
//! hand axis from the wrist (the model's own fist-collider centre). The
//! "Flux" is the finite segment between the two palms, always live.
//!
//! Objects ride the same head-anchored track as notes
//! (`z = (hit_time − t)·speed + z_off`) and miss at the head plane:
//! - **Circle** (KIND_CIRCLE, has a colour): a flat disc facing the
//!   player, radius CIRCLE_R, with the good box's depth convention
//!   (z ∈ [z − 0.75, z + 0.25]). Touching it with the matching palm
//!   scores CIRCLE_PTS + round(10·(1 − clamp(d/0.3, 0, 1))), d = the
//!   in-plane distance from the centre at the palm path's closest
//!   approach inside the slab. The other palm touching it is a bad hit
//!   (event 3, wrong hand). No speed or direction requirement.
//! - **Bar** (KIND_BAR, colourless): a thin rectangle in the plane
//!   facing the player — width 2·BAR_HALF_W, length `len`, rotated by
//!   θ about z (axis u = (−sin θ, cos θ): θ = 0 vertical, CCW from the
//!   player's view, the note-angle convention), depth as circles. The
//!   swept Flux (quad between last tick's and this tick's segment, the
//!   saber-sweep triangulation) touching its box scores BAR_PTS +
//!   round(15·(1 − clamp(d/0.3, 0, 1))), d = the distance along u from
//!   where this tick's Flux LINE crosses the bar axis (clamped onto the
//!   bar) to the sweet spot at `sweet·len/2`. The crossing must be
//!   near-square: this tick's Flux line may deviate at most 60 deg from
//!   the bar's perpendicular (2D, undirected), else the touch is a bad
//!   cut (event 2 "angle") — the note swing tolerance's twin.
//!
//! Events reuse the standard codes (0 good, 3 wrong hand, 4 miss) so
//! ComboWalk runs unchanged; a good event's `pre` slot carries the base
//! contact points (50/100) and `acc` the accuracy points.

use super::*;

/// Palm centre along the hand axis from the wrist (m) — the fist
/// sphere / hand CoM offset in mjcf/body_v10.xml.
pub(crate) const PALM_OFF: Real = 0.06;
/// Circle hitbox radius (m): the ~35 cm visual disc.
const CIRCLE_R: Real = 0.175;
const CIRCLE_PTS: Real = 50.0;
const CIRCLE_ACC_PTS: Real = 10.0;
const BAR_PTS: Real = 100.0;
const BAR_ACC_PTS: Real = 15.0;
/// Bar half width (m): 5 cm wide.
const BAR_HALF_W: Real = 0.025;
/// Hitbox depth slab relative to the object centre — the standard good
/// box's convention (1.0 m deep, shifted 0.25 m toward the player).
const SLAB_LO: Real = GOOD_Z_SHIFT - GOOD_HALF_Z; // -0.75
const SLAB_HI: Real = GOOD_Z_SHIFT + GOOD_HALF_Z; // +0.25
/// Live broad-phase reach: arm (0.65) + palm + the slab's near face.
const FLUX_REACH_PAST_SHOULDER: Real = 0.65 + PALM_OFF - SLAB_LO;

/// set_flux_chart tuple: (hit_time, x, y, kind, color, rot_rad, speed,
/// length, sweet).
pub(crate) type FluxTuple = (Real, Real, Real, u8, u8, Real, Real, Real, Real);

pub(crate) fn palm(wrist: [Real; 3], dir: [Real; 3]) -> [Real; 3] {
    [
        wrist[0] + PALM_OFF * dir[0],
        wrist[1] + PALM_OFF * dir[1],
        wrist[2] + PALM_OFF * dir[2],
    ]
}

/// Palm path a → b (world) against the circle centred at `c`: the path
/// is clipped to the depth slab, then its closest in-plane approach to
/// the centre is found. Some(d) when that approach is within CIRCLE_R.
fn circle_touch(a_w: [Real; 3], b_w: [Real; 3], c: [Real; 3]) -> Option<Real> {
    let a = sub3(a_w, c);
    let b = sub3(b_w, c);
    let dz = b[2] - a[2];
    let (mut t0, mut t1) = (0.0 as Real, 1.0 as Real);
    if dz.abs() < 1e-12 {
        if a[2] < SLAB_LO || a[2] > SLAB_HI {
            return None;
        }
    } else {
        let ta = (SLAB_LO - a[2]) / dz;
        let tb = (SLAB_HI - a[2]) / dz;
        t0 = t0.max(ta.min(tb));
        t1 = t1.min(ta.max(tb));
        if t0 > t1 {
            return None;
        }
    }
    let ex = b[0] - a[0];
    let ey = b[1] - a[1];
    let ee = ex * ex + ey * ey;
    let t = if ee < 1e-12 {
        t0
    } else {
        (-(a[0] * ex + a[1] * ey) / ee).clamp(t0, t1)
    };
    let qx = a[0] + t * ex;
    let qy = a[1] + t * ey;
    let d = (qx * qx + qy * qy).sqrt();
    if d <= CIRCLE_R {
        Some(d)
    } else {
        None
    }
}

/// Distance along the bar axis from the Flux crossing to the sweet
/// spot. The crossing is where this tick's Flux LINE (through the palms
/// l, r, projected onto the bar's xy plane) meets the bar axis
/// c + s·u, clamped onto the bar; a Flux parallel to the bar (or
/// collapsed to a point) takes the point of its projected span nearest
/// the sweet spot instead.
fn bar_sweet_dist(n: &Note, u: [Real; 2], l: [Real; 3], r: [Real; 3]) -> Real {
    let half = 0.5 * n.len;
    let s_sweet = n.sweet * half;
    let fx = r[0] - l[0];
    let fy = r[1] - l[1];
    let lx = l[0] - n.x;
    let ly = l[1] - n.y;
    let cr = u[0] * fy - u[1] * fx;
    let ff = fx * fx + fy * fy;
    let s = if ff > 1e-12 && cr * cr > 1e-12 * ff {
        (lx * fy - ly * fx) / cr
    } else {
        let sa = lx * u[0] + ly * u[1];
        let sb = (r[0] - n.x) * u[0] + (r[1] - n.y) * u[1];
        s_sweet.clamp(sa.min(sb), sa.max(sb))
    };
    (s.clamp(-half, half) - s_sweet).abs()
}

/// Bar crossing-angle check (user rule 2026-09-26, the note swing
/// tolerance's twin): this tick's Flux LINE, projected onto the bar's xy
/// plane, may deviate at most 60 deg from the bar's perpendicular
/// n = (cos θ, sin θ) (a line is undirected, so the deviation is 0-90
/// deg): |f̂ · n| >= COS_TOL, squared to stay sqrt-free. A collapsed
/// Flux (palms coincident in xy) has no direction and fails.
fn bar_angle_ok(n: &Note, l: [Real; 3], r: [Real; 3]) -> bool {
    let fx = r[0] - l[0];
    let fy = r[1] - l[1];
    let ff = fx * fx + fy * fy;
    if ff <= 1e-12 {
        return false;
    }
    let dn = fx * n.cos_a + fy * n.sin_a;
    dn * dn >= COS_TOL * COS_TOL * ff
}

impl Core {
    /// Install a Flux chart (flux sims only). Walls do not exist in
    /// Flux; any previous ones are cleared.
    pub(crate) fn set_flux_chart(&mut self, chart: Vec<FluxTuple>) {
        assert!(self.flux, "set_flux_chart needs a sim built with flux=True");
        self.set_walls(Vec::new());
        self.chart = chart
            .into_iter()
            .map(|(t, x, y, kind, color, rot, speed, len, sweet)| {
                assert!(kind == KIND_CIRCLE || kind == KIND_BAR,
                        "flux chart kinds are 4 (circle) / 5 (bar)");
                assert!(color <= 1, "colour must be 0 (left) or 1 (right)");
                assert!(speed > 0.0, "speed must be positive");
                if kind == KIND_BAR {
                    assert!(len > 0.0, "bar length must be positive");
                    assert!((-1.0..=1.0).contains(&sweet),
                            "sweet-spot offset must be in [-1, 1]");
                }
                let is_bar = kind == KIND_BAR;
                Note {
                    hit_time: t,
                    x,
                    y,
                    color: if is_bar { 0 } else { color },
                    sin_a: if is_bar { rot.sin() } else { 0.0 },
                    cos_a: if is_bar { rot.cos() } else { 1.0 },
                    speed,
                    any_dir: true,
                    kind,
                    free_pre: false,
                    free_post: false,
                    len: if is_bar { len } else { 0.0 },
                    sweet: if is_bar { sweet } else { 0.0 },
                }
            })
            .collect();
        self.finish_chart_install();
    }

    /// The Flux object pass of `step_inner` (replaces the saber cut
    /// loop): miss boundary, circle touches, bar crossings. prev_* are
    /// last tick's wrist/hand-axis caches.
    pub(crate) fn flux_judge(
        &mut self,
        reward: &mut Real,
        prev_pos: &[[Real; 3]; 2],
        prev_dir: &[[Real; 3]; 2],
    ) {
        let p0 = [palm(prev_pos[0], prev_dir[0]), palm(prev_pos[1], prev_dir[1])];
        let p1 = [
            palm(self.body.wrist[0], self.body.blade[0]),
            palm(self.body.wrist[1], self.body.blade[1]),
        ];
        for s in 0..2 {
            for k in 0..3 {
                self.palm_vel[s][k] = (p1[s][k] - p0[s][k]) / DT;
            }
        }
        // the swept Flux: quad (L0 -> R0) to (L1 -> R1), triangulated
        // like the saber sweep
        let tris = [[p0[0], p0[1], p1[1]], [p0[0], p1[1], p1[0]]];

        let reach_z = self.body.shoulder_z[0].max(self.body.shoulder_z[1])
            + FLUX_REACH_PAST_SHOULDER;
        let head_z = self.body.head[2];
        let head = self.body.head;
        let z_off = head_z - PLANE_Z0;

        for i in self.first_pending..self.chart.len() {
            if self.status[i] != 0 {
                continue;
            }
            let n = &self.chart[i];
            if (n.hit_time - self.time) * self.min_speed + z_off > reach_z {
                break; // sorted-chart bound, as in the saber loop
            }
            let z = (n.hit_time - self.time) * n.speed + z_off;
            if z < head_z {
                self.status[i] = 3;
                self.misses += 1;
                *reward += REWARD_MISS;
                self.events_buf.push((i as u32, 4, 0.0, 0.0, 0.0));
                if n.kind == KIND_CIRCLE {
                    let h = n.color as usize;
                    self.prev_idx[h] = i as u32;
                    self.prev_valid[h] = true;
                }
                continue;
            }
            if z > reach_z {
                continue;
            }
            let center = [n.x, n.y, z];

            if n.kind == KIND_CIRCLE {
                let m = n.color as usize;
                if let Some(d) = circle_touch(p0[m], p1[m], center) {
                    let acc = round_pts(
                        CIRCLE_ACC_PTS * (1.0 - (d / ACC_NORM).clamp(0.0, 1.0)));
                    let pts = CIRCLE_PTS + acc;
                    self.status[i] = 1;
                    self.good_cuts += 1;
                    *reward += pts / SCORE_MAX;
                    self.score += pts;
                    self.score_pre += CIRCLE_PTS;
                    self.score_acc += acc;
                    self.events_buf.push((i as u32, 0, pts as f32,
                                          CIRCLE_PTS as f32, acc as f32));
                    self.prev_idx[m] = i as u32;
                    self.prev_valid[m] = true;
                } else if circle_touch(p0[1 - m], p1[1 - m], center).is_some() {
                    self.status[i] = 2;
                    self.bad_cuts += 1;
                    *reward += REWARD_BAD;
                    self.events_buf.push((i as u32, 3, 0.0, 0.0, 0.0));
                    self.prev_idx[m] = i as u32;
                    self.prev_valid[m] = true;
                }
            } else {
                // bar: z-roll frame (no look-at), local y' = the axis u
                let frame = note_frame(center, n.cos_a, n.sin_a, 0.0, head);
                let box_c = [center[0], center[1], center[2] + GOOD_Z_SHIFT];
                let h = [BAR_HALF_W, 0.5 * n.len, GOOD_HALF_Z];
                if sweep_hits_box(&tris, box_c, h, &frame) {
                    if !bar_angle_ok(n, p1[0], p1[1]) {
                        // crossed, but more than 60 deg off square: a bad
                        // cut (event 2 "angle", like an off-direction note)
                        self.status[i] = 2;
                        self.bad_cuts += 1;
                        *reward += REWARD_BAD;
                        self.events_buf.push((i as u32, 2, 0.0, 0.0, 0.0));
                        continue;
                    }
                    let u = [-n.sin_a, n.cos_a];
                    let d = bar_sweet_dist(n, u, p1[0], p1[1]);
                    let acc = round_pts(
                        BAR_ACC_PTS * (1.0 - (d / ACC_NORM).clamp(0.0, 1.0)));
                    let pts = BAR_PTS + acc;
                    self.status[i] = 1;
                    self.good_cuts += 1;
                    *reward += pts / SCORE_MAX;
                    self.score += pts;
                    self.score_pre += BAR_PTS;
                    self.score_acc += acc;
                    self.events_buf.push((i as u32, 0, pts as f32,
                                          BAR_PTS as f32, acc as f32));
                }
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
            let p = palm(b.wrist[s], b.blade[s]);
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

#[cfg(test)]
mod tests {
    use super::*;

    const C: [Real; 3] = [0.3, 1.4, 0.5];

    fn bar(x: Real, y: Real, rot: Real, len: Real, sweet: Real) -> Note {
        Note {
            hit_time: 0.0, x, y, color: 0, sin_a: rot.sin(), cos_a: rot.cos(),
            speed: 10.0, any_dir: true, kind: KIND_BAR, free_pre: false,
            free_post: false, len, sweet,
        }
    }

    #[test]
    fn circle_static_palm_inside_the_slab_touches() {
        let p = [0.35, 1.45, 0.3]; // 7 cm off-centre, 0.2 m in front
        let d = circle_touch(p, p, C).expect("touch");
        assert!((d - (0.05f64.powi(2) * 2.0).sqrt() as Real).abs() < 1e-6);
    }

    #[test]
    fn circle_misses_outside_radius_and_outside_the_slab() {
        let wide = [C[0] + 0.18, C[1], C[2]];
        assert!(circle_touch(wide, wide, C).is_none());
        let behind = [C[0], C[1], C[2] + 0.3]; // past the far face (+0.25)
        assert!(circle_touch(behind, behind, C).is_none());
        let deep = [C[0], C[1], C[2] - 0.8]; // before the near face (-0.75)
        assert!(circle_touch(deep, deep, C).is_none());
    }

    #[test]
    fn circle_fast_pass_through_is_caught_by_the_sweep() {
        // both endpoints 0.5 m either side of the centre, in the slab
        let a = [C[0] - 0.5, C[1] + 0.02, C[2] - 0.2];
        let b = [C[0] + 0.5, C[1] + 0.02, C[2] - 0.2];
        let d = circle_touch(a, b, C).expect("swept touch");
        assert!((d - 0.02).abs() < 1e-9);
    }

    #[test]
    fn circle_closest_approach_is_taken_inside_the_slab_only() {
        // the path passes the centre line only beyond the far face
        // (unclipped closest approach at t = 0.5, beyond the far face)
        let a = [C[0] - 0.2, C[1], C[2] - 0.5];
        let b = [C[0] + 0.2, C[1], C[2] + 1.5];
        // clipped at z = C+0.25: t = 0.75/2 = 0.375 -> x = -0.2 + 0.15
        let d = circle_touch(a, b, C).expect("touch");
        assert!((d - 0.05).abs() < 1e-9, "{d}");
    }

    #[test]
    fn bar_sweet_distance_perpendicular_crossing() {
        // vertical bar at x=0, Flux horizontal at y = 1.5
        let n = bar(0.0, 1.4, 0.0, 0.6, 0.0);
        let (l, r) = ([-0.3, 1.5, 0.0], [0.3, 1.5, 0.0]);
        let u = [-n.sin_a, n.cos_a];
        assert!((bar_sweet_dist(&n, u, l, r) - 0.1).abs() < 1e-9);
        let top = bar(0.0, 1.4, 0.0, 0.6, 1.0); // sweet at y = 1.7
        assert!((bar_sweet_dist(&top, u, l, r) - 0.2).abs() < 1e-9);
        let bottom = bar(0.0, 1.4, 0.0, 0.6, -1.0); // sweet at y = 1.1
        assert!((bar_sweet_dist(&bottom, u, l, r) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn bar_sweet_distance_rotated_and_clamped() {
        // bar rotated 90 deg CCW: axis u = (-1, 0), sweet +1 at x = -0.3
        let n = bar(0.0, 1.4, std::f64::consts::FRAC_PI_2 as Real, 0.6, 1.0);
        let u = [-n.sin_a, n.cos_a];
        // Flux vertical through x = -0.1: crossing s = +0.1 along u
        let d = bar_sweet_dist(&n, u, [-0.1, 1.0, 0.0], [-0.1, 1.8, 0.0]);
        assert!((d - 0.2).abs() < 1e-6, "{d}");
        // crossing off the bar's end (x = -0.5) clamps to the end
        let d = bar_sweet_dist(&n, u, [-0.5, 1.0, 0.0], [-0.5, 1.8, 0.0]);
        assert!(d.abs() < 1e-6, "{d}");
    }

    #[test]
    fn bar_sweet_distance_parallel_flux_uses_its_span() {
        let n = bar(0.0, 1.4, 0.0, 0.6, 1.0); // sweet at y = 1.7
        let u = [-n.sin_a, n.cos_a];
        // Flux lying along the bar from y = 1.2 to 1.5: nearest 1.5
        let d = bar_sweet_dist(&n, u, [0.0, 1.2, 0.0], [0.0, 1.5, 0.0]);
        assert!((d - 0.2).abs() < 1e-9, "{d}");
        // collapsed Flux (palms together) at y = 1.3
        let d = bar_sweet_dist(&n, u, [0.0, 1.3, 0.0], [0.0, 1.3, 0.0]);
        assert!((d - 0.4).abs() < 1e-9, "{d}");
    }

    #[test]
    fn bar_angle_allows_60_deg_off_square_and_no_more() {
        let n = bar(0.0, 1.4, 0.0, 0.6, 0.0); // vertical: square = horizontal
        let flux = |deg: f64, len: f64| {
            let (s, c) = (deg.to_radians() as Real).sin_cos();
            let h = 0.5 * len as Real;
            ([-h * c, 1.4 - h * s, 0.0], [h * c, 1.4 + h * s, 0.0])
        };
        for (deg, ok) in [(0.0, true), (30.0, true), (59.0, true),
                          (61.0, false), (90.0, false), (115.0, false),
                          (125.0, true), (150.0, true), (180.0, true),
                          (-59.0, true),
                          (-61.0, false)] {
            let (l, r) = flux(deg, 0.6);
            assert_eq!(bar_angle_ok(&n, l, r), ok, "{deg} deg");
            // undirected: swapping the palms changes nothing
            assert_eq!(bar_angle_ok(&n, r, l), ok, "{deg} deg swapped");
        }
        // rotated bar: square follows the bar
        let rot = bar(0.0, 1.4, 1.0, 0.6, 0.0); // normal at 1 rad
        let (l, r) = flux(57.29577951308232 + 55.0, 0.6);
        assert!(bar_angle_ok(&rot, l, r));
        let (l, r) = flux(57.29577951308232 + 65.0, 0.6);
        assert!(!bar_angle_ok(&rot, l, r));
        // collapsed Flux has no direction
        let p = [0.1, 1.4, 0.0];
        assert!(!bar_angle_ok(&n, p, [0.1, 1.4, 0.3]));
    }

    #[test]
    fn swept_flux_catches_a_thin_bar_between_ticks() {
        // vertical 5 cm bar at x = 0; the Flux (horizontal, y = 1.4)
        // jumps from x in [-0.8, -0.2] to [0.2, 0.8]: neither tick's
        // segment touches the bar, the swept quad does
        let n = bar(0.0, 1.4, 0.0, 0.6, 0.0);
        let frame = note_frame([0.0, 1.4, 0.0], n.cos_a, n.sin_a, 0.0,
                               [0.0, 1.7, -0.65]);
        let h = [BAR_HALF_W, 0.3, GOOD_HALF_Z];
        let bc = [0.0, 1.4, GOOD_Z_SHIFT];
        let (l0, r0) = ([-0.8, 1.4, -0.2], [-0.2, 1.4, -0.2]);
        let (l1, r1) = ([0.2, 1.4, -0.2], [0.8, 1.4, -0.2]);
        let stat = [[l0, r0, r0], [l0, r0, l0]];
        assert!(!sweep_hits_box(&stat, bc, h, &frame));
        let stat1 = [[l1, r1, r1], [l1, r1, l1]];
        assert!(!sweep_hits_box(&stat1, bc, h, &frame));
        let swept = [[l0, r0, r1], [l0, r1, l1]];
        assert!(sweep_hits_box(&swept, bc, h, &frame));
        // a static Flux crossing the bar hits; one above its top misses
        let (la, ra) = ([-0.3, 1.5, -0.2], [0.3, 1.5, -0.2]);
        assert!(sweep_hits_box(&[[la, ra, ra], [la, ra, la]], bc, h, &frame));
        let (lb, rb) = ([-0.3, 1.75, -0.2], [0.3, 1.75, -0.2]);
        assert!(!sweep_hits_box(&[[lb, rb, rb], [lb, rb, lb]], bc, h, &frame));
    }
}
