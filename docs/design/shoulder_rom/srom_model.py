"""Prototype of the passive shoulder axial-rotation limit (design phase 1).

Canonical (LEFT-arm) coordinates; the right arm is evaluated at the
mirrored angles (yaw, twist negated) and its yaw/twist torques negated —
exactly the sim's mirror map, so mirror-exactness is by construction.

  humerus direction d (chest frame, left-canonical, lateral/up/forward):
      d_l = -cos p sin y,  d_u = sin p,  d_f = cos p cos y
  elevation from hanging      theta = p + 90 deg,  c = cos theta = -sin p
  plane of elevation          phi = y + 90 deg  (0 abduction, 90 flexion)
  axial rotation (IR +)       tau = y - t       (swing-twist from hanging;
                                                 = the ISB globe psi + phi - 90)
  limits (deg of tau)         f_k = A_k(c) + B_k (d_l - d_f) + (1 - c) phi,
                              A_k quadratic in c  (k = lo (ER), hi (IR));
                              (1 - c) phi is the gauge that makes f regular
                              at BOTH poles: tau-type at the side (c = 1),
                              globe-psi-type at the horizontal, (psi - phi)-
                              type at the top (the regular coordinate there)
  mid / half range            m = (f_hi + f_lo)/2,  w = (f_hi - f_lo)/2
  normalized excursion        u = (tau - m)/w,  a = |u|
  zone                        z = ((a - s)/(1 - s))_+          (s: zone start)
  beyond                      x = ((a - 1)/wb)_+               (past the limit)
  g(a) = z^2 + kb x^2         (pure-axis torque / T), for a <= uc;
                              linear continuation g_c + g'_c (a - uc) beyond
                              uc (bounded stiffness; still rising)
  energy                      E = T w G(a),  G' = g,  G(0) = 0
  pure axial torque           -dE/dtau = -sign(u) T g(a)
  joint torques               -dE/dq = -[sign(u) T g (dtau - dm) + T (G - a g) dw],
                              analytic; no singularity: tau = y - t is linear
                              in q and the limits are smooth in q inside the
                              box (the hanging pole needs no special case).
"""
import numpy as np

DEG = np.pi / 180.0

# twist tau_max of the DEFAULT body: TAU_MAX[5] * TAU_BODY_DEFAULT[arm_twist]
TWIST_TAU_DEFAULT = 19.45 * 0.8729

# Limit anchors, tau degrees (IR +), left arm: name -> (theta, phi, ER-side
# limit tau_lo, IR-side limit tau_hi). Sources in docs/design/shoulder-rom.md
# section B. tau = phi_ISB + psi_ISB, so at 90 deg abduction tau is the
# clinical rotation (0 = forearm forward) and at 90 deg flexion tau = 0 is
# forearm UP.
ANCHORS = {
    "side":     (0.0,   0.0,  -80.0,  77.0),   # ER0 ~80 (Vairo passive 82); arc 157 (Wang)
    "abd90":    (90.0,  0.0, -100.0,  60.0),   # Vairo passive 105/59, Roy 89/68
    "flex90":   (90.0, 90.0,  -25.0, 130.0),   # psi_ISB ER 115 / IR 40 (Humphries, sleeper)
    "top":      (155.0, 40.0, -75.0,  20.0),   # arc ~95 at the top (Wang 94), mostly ER
}


def basis(theta_deg, phi_deg):
    """Row of the limit basis [1, c, c^2, d_l - d_f] and the gauge term
    lambda(c)*phi (deg), lambda = 1 - c. The single plane harmonic
    B (d_l - d_f) = B sin(theta) (cos phi - sin phi) makes the horizontal
    limit (in globe psi) a0 + B at abduction, a0 - B at flexion and
    monotone in between (a free cos/sin pair let a0 float and dipped
    20-40 deg between the two anchors)."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    c = np.cos(th)
    return np.array([1.0, c, c * c, np.sin(th) * (np.cos(ph) - np.sin(ph))]), \
        (1.0 - c) * phi_deg


def fit_coeffs(anchors=None):
    """Least-squares (exact for 4 anchors) fit of f_k = A(c) + B (d_l - d_f)
    + (1 - c) phi, per limit k. Returns {k: (a0, a1, a2, B) in rad}."""
    an = anchors or ANCHORS
    rows, gauge = zip(*(basis(a[0], a[1]) for a in an.values()))
    X = np.array(rows)
    out = {}
    for i, k in ((2, "lo"), (3, "hi")):
        yv = np.array([a[i] for a in an.values()]) - np.array(gauge)
        coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        out[k] = coef * DEG
    return out


class ShoulderROM:
    def __init__(self, anchors=None, frac=0.30, s=0.3, kb=1.0, wb=0.25, uc=1.25,
                 tau_ref=TWIST_TAU_DEFAULT, damp_ref_speed=16.0):
        self.cf = fit_coeffs(anchors)
        self.T = frac * tau_ref
        self.s, self.kb, self.wb, self.uc = s, kb, wb, uc
        self.bpk = self.T / damp_ref_speed

    # ---- limits -------------------------------------------------------
    def limits(self, y, p):
        """(f_lo, f_hi) and their gradients wrt (y, p)."""
        sy, cy, sp, cp = np.sin(y), np.cos(y), np.sin(p), np.cos(p)
        dl, df, c = -cp * sy, cp * cy, -sp
        dl_y, dl_p = -cp * cy, sp * sy
        df_y, df_p = -cp * sy, -sp * cy
        c_p = -cp
        lam, lam_p = 1.0 - c, cp            # lambda = 1 - c = 1 + sin p
        phi, phi_y = y + np.pi / 2, 1.0
        res = {}
        for k, (a0, a1, a2, B) in self.cf.items():
            C = -B
            Av = a0 + c * (a1 + c * a2)
            Ad = a1 + 2 * c * a2
            f = Av + B * dl + C * df + lam * phi
            f_y = B * dl_y + C * df_y + lam * phi_y
            f_p = Ad * c_p + B * dl_p + C * df_p + lam_p * phi
            res[k] = (f, f_y, f_p)
        return res

    def _raw(self, a):
        s, kb, wb = self.s, self.kb, self.wb
        z = np.maximum((a - s) / (1 - s), 0.0)
        x = np.maximum((a - 1.0) / wb, 0.0)
        g = z * z + kb * x * x
        G = (1 - s) / 3 * z ** 3 + kb * wb / 3 * x ** 3
        dg = 2 * z / (1 - s) + 2 * kb * x / wb
        return g, G, dg, z

    def shape(self, a):
        """g(a) (torque / T), G(a) (energy / (T w)) and the zone z, with the
        linear-torque continuation past uc."""
        g, G, _, z = self._raw(a)
        gc, Gc, dgc, _ = self._raw(np.asarray(self.uc, float))
        d = a - self.uc
        beyond = d > 0
        g = np.where(beyond, gc + dgc * d, g)
        G = np.where(beyond, Gc + gc * d + 0.5 * dgc * d * d, G)
        return g, G, z

    # ---- energy and torques ---------------------------------------------
    def eval(self, y, p, t):
        """Returns E, (tau_y, tau_p, tau_t) = -grad E, u, z."""
        L = self.limits(y, p)
        flo, flo_y, flo_p = L["lo"]
        fhi, fhi_y, fhi_p = L["hi"]
        m = 0.5 * (fhi + flo)
        w = 0.5 * (fhi - flo)
        m_y, m_p = 0.5 * (fhi_y + flo_y), 0.5 * (fhi_p + flo_p)
        w_y, w_p = 0.5 * (fhi_y - flo_y), 0.5 * (fhi_p - flo_p)
        tau = y - t
        u = (tau - m) / w
        a = np.abs(u)
        g, G, z = self.shape(a)
        T = self.T
        E = T * w * G
        sg = np.sign(u)
        # dE = sg T g (dtau - dm) + T (G - a g) dw
        e_m = -sg * T * g
        e_w = T * (G - a * g)
        e_tau = sg * T * g
        dE_y = e_tau * 1.0 + e_m * m_y + e_w * w_y
        dE_p = e_m * m_p + e_w * w_p
        dE_t = e_tau * -1.0
        return E, (-dE_y, -dE_p, -dE_t), u, z

    def damping(self, u):
        """Axial edge damping coefficient b (N.m.s/rad) on tau-dot; the
        joint-space matrix is b * J^T J, J = (1, 0, -1) on (yaw, pitch,
        twist) — rank 1, PSD, applied implicitly."""
        a = np.minimum(np.abs(u), self.uc)      # capped like the stiffness
        z = np.maximum((a - self.s) / (1 - self.s), 0.0)
        return self.bpk * z * z


def anat(y, p, t):
    """Reporting angles (deg): elevation, plane, tau (IR+), globe psi (IR+)."""
    return (np.degrees(p) + 90, np.degrees(y) + 90, np.degrees(y - t), -np.degrees(t))
