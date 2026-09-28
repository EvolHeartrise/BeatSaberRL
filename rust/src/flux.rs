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
//!
//! **Shared with the WebXR game.** This file holds the Flux RULES only —
//! no body, no obs, no pyo3 — and is compiled into both the Python env
//! crate (`mod flux`, driven by `flux_sim.rs`) and the browser game's
//! wasm crate (`rust/flux-wasm`, fed by tracked controller palms). One
//! source, two builds: the game cannot judge a touch differently from
//! the env the agent trains on. It may only use `crate::Real` and the
//! other shared files (`geom.rs`, `combo.rs`).

use crate::geom::*;
use crate::Real;

/// Palm centre along the hand axis from the wrist (m) — the fist
/// sphere / hand CoM offset in mjcf/body_v10.xml.
pub(crate) const PALM_OFF: Real = 0.06;
/// Circle hitbox radius (m): the ~35 cm visual disc.
pub(crate) const CIRCLE_R: Real = 0.175;
pub(crate) const CIRCLE_PTS: Real = 50.0;
pub(crate) const CIRCLE_ACC_PTS: Real = 10.0;
pub(crate) const BAR_PTS: Real = 100.0;
pub(crate) const BAR_ACC_PTS: Real = 15.0;
/// Bar half width (m): 5 cm wide.
pub(crate) const BAR_HALF_W: Real = 0.025;
/// Hitbox depth slab relative to the object centre — the standard good
/// box's convention (1.0 m deep, shifted 0.25 m toward the player).
pub(crate) const SLAB_LO: Real = GOOD_Z_SHIFT - GOOD_HALF_Z; // -0.75
pub(crate) const SLAB_HI: Real = GOOD_Z_SHIFT + GOOD_HALF_Z; // +0.25
/// Live broad-phase reach: arm (0.65) + palm + the slab's near face.
pub(crate) const FLUX_REACH_PAST_SHOULDER: Real = 0.65 + PALM_OFF - SLAB_LO;

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

/// One judgment event: (chart index, code, points, base, accuracy) —
/// code 0 good, 2 bad "angle", 3 wrong hand, 4 miss (the standard codes).
pub(crate) type Event = (u32, u8, f32, f32, f32);

/// A bar's rotation as (sin, cos). The f64 build — the env reference AND
/// the WebXR game's wasm — uses geom::det_sin_cos, so the two install
/// bit-identical bars (the platform libms differ in the last bit). The f32
/// build (the CUDA kernel's precision twin, no game counterpart) keeps
/// the f32 libm.
#[cfg(not(feature = "f32"))]
fn bar_sin_cos(rot: Real) -> (Real, Real) {
    det_sin_cos(rot)
}
#[cfg(feature = "f32")]
fn bar_sin_cos(rot: Real) -> (Real, Real) {
    (rot.sin(), rot.cos())
}

/// Chart tuple -> the sim's object record, with the install-time checks
/// (the env panics on an Err, the wasm game reports it).
pub(crate) fn flux_note(t: FluxTuple) -> Result<Note, &'static str> {
    let (hit_time, x, y, kind, color, rot, speed, len, sweet) = t;
    if kind != KIND_CIRCLE && kind != KIND_BAR {
        return Err("flux chart kinds are 4 (circle) / 5 (bar)");
    }
    if color > 1 {
        return Err("colour must be 0 (left) or 1 (right)");
    }
    // written so a NaN fails too (as the env's assert!(speed > 0.0) did)
    if !(speed > 0.0) {
        return Err("speed must be positive");
    }
    let is_bar = kind == KIND_BAR;
    let (sin_r, cos_r) = if is_bar { bar_sin_cos(rot) } else { (0.0, 1.0) };
    if is_bar {
        if !(len > 0.0) {
            return Err("bar length must be positive");
        }
        if !(-1.0..=1.0).contains(&sweet) {
            return Err("sweet-spot offset must be in [-1, 1]");
        }
    }
    Ok(Note {
        hit_time,
        x,
        y,
        color: if is_bar { 0 } else { color },
        sin_a: sin_r,
        cos_a: cos_r,
        speed,
        any_dir: true,
        kind,
        free_pre: false,
        free_post: false,
        len: if is_bar { len } else { 0.0 },
        sweet: if is_bar { sweet } else { 0.0 },
    })
}

/// Live broad-phase reach gate from the forward-most shoulder's z.
pub(crate) fn reach_z(shoulder_z_max: Real) -> Real {
    shoulder_z_max + FLUX_REACH_PAST_SHOULDER
}

/// The Flux object pass of one tick (replaces the saber cut loop): miss
/// boundary, circle touches, bar crossings. `p0` / `p1` are last tick's
/// and this tick's palms [left, right] (world), `head` this tick's head
/// position, `time` this tick's END time, `reach_z` the live reach
/// gate. Resolves objects in `status` (1 good, 2 bad, 3 missed), adds
/// the reward terms to `reward` in chart order and appends one Event per
/// resolution; the caller keeps its own tallies from the events.
#[allow(clippy::too_many_arguments)]
pub(crate) fn judge(
    chart: &[Note],
    status: &mut [u8],
    first_pending: usize,
    min_speed: Real,
    time: Real,
    head: [Real; 3],
    reach_z: Real,
    p0: &[[Real; 3]; 2],
    p1: &[[Real; 3]; 2],
    reward: &mut Real,
    events: &mut Vec<Event>,
) {
    // the swept Flux: quad (L0 -> R0) to (L1 -> R1), triangulated
    // like the saber sweep
    let tris = [[p0[0], p0[1], p1[1]], [p0[0], p1[1], p1[0]]];

    let head_z = head[2];
    let z_off = head_z - PLANE_Z0;

    for i in first_pending..chart.len() {
        if status[i] != 0 {
            continue;
        }
        let n = &chart[i];
        if (n.hit_time - time) * min_speed + z_off > reach_z {
            break; // sorted-chart bound, as in the saber loop
        }
        let z = (n.hit_time - time) * n.speed + z_off;
        if z < head_z {
            status[i] = 3;
            *reward += REWARD_MISS;
            events.push((i as u32, 4, 0.0, 0.0, 0.0));
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
                status[i] = 1;
                *reward += pts / SCORE_MAX;
                events.push((i as u32, 0, pts as f32, CIRCLE_PTS as f32, acc as f32));
            } else if circle_touch(p0[1 - m], p1[1 - m], center).is_some() {
                status[i] = 2;
                *reward += REWARD_BAD;
                events.push((i as u32, 3, 0.0, 0.0, 0.0));
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
                    status[i] = 2;
                    *reward += REWARD_BAD;
                    events.push((i as u32, 2, 0.0, 0.0, 0.0));
                    continue;
                }
                let u = [-n.sin_a, n.cos_a];
                let d = bar_sweet_dist(n, u, p1[0], p1[1]);
                let acc = round_pts(
                    BAR_ACC_PTS * (1.0 - (d / ACC_NORM).clamp(0.0, 1.0)));
                let pts = BAR_PTS + acc;
                status[i] = 1;
                *reward += pts / SCORE_MAX;
                events.push((i as u32, 0, pts as f32, BAR_PTS as f32, acc as f32));
            }
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

    /// ulp distance between two finite f64s of the same sign region
    fn ulps(a: f64, b: f64) -> u64 {
        let key = |x: f64| {
            let b = x.to_bits() as i64;
            if b < 0 { i64::MIN - b } else { b }
        };
        (key(a) - key(b)).unsigned_abs()
    }

    #[test]
    fn det_sin_cos_is_within_one_ulp_of_libm_everywhere_a_chart_goes() {
        // dense sweep over [-4π, 4π] (bars live in [0, 2π)) plus the
        // octant boundaries, where the range reduction switches branches
        let mut worst = 0;
        let mut pts: Vec<f64> = (-400_000..=400_000)
            .map(|k| k as f64 * (4.0 * std::f64::consts::PI / 400_000.0))
            .collect();
        for k in -16..=16 {
            let b = k as f64 * std::f64::consts::FRAC_PI_4;
            pts.extend([b, f64::from_bits(b.to_bits() + 1), f64::from_bits(b.to_bits().wrapping_sub(1))]);
        }
        for x in pts {
            let (s, c) = det_sin_cos(x);
            // near a zero of sin/cos the ulp scale collapses: compare in
            // absolute terms there
            for (got, want) in [(s, x.sin()), (c, x.cos())] {
                if want.abs() < 1e-8 {
                    assert!((got - want).abs() < 1e-16, "{x}: {got} vs {want}");
                } else {
                    worst = worst.max(ulps(got, want));
                }
            }
        }
        assert!(worst <= 1, "worst {worst} ulp");
        assert_eq!(det_sin_cos(0.0), (0.0, 1.0));
        assert!(det_sin_cos(f64::NAN).0.is_nan() && det_sin_cos(f64::INFINITY).1.is_nan());
        // huge angles fold by the exact remainder
        let (s, c) = det_sin_cos(1.0e7);
        assert!((s * s + c * c - 1.0).abs() < 1e-15);
    }

    #[test]
    fn bars_install_with_the_deterministic_trig() {
        for rot in [0.3, 1.0, 2.5, 4.0, 5.9] {
            let n = flux_note((1.0, 0.0, 1.4, KIND_BAR, 0, rot, 10.0, 0.5, 0.0)).unwrap();
            let (s, c) = det_sin_cos(rot);
            assert_eq!((n.sin_a.to_bits(), n.cos_a.to_bits()), (s.to_bits(), c.to_bits()));
        }
        // circles carry the identity rotation
        let n = flux_note((1.0, 0.0, 1.4, KIND_CIRCLE, 1, 2.0, 10.0, 0.0, 0.0)).unwrap();
        assert_eq!((n.sin_a, n.cos_a), (0.0, 1.0));
    }
}
