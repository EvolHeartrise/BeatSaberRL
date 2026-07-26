// Beat Arena sim step as ONE CUDA kernel launch (env v7.9).
//
// A semantics-exact port of rust/src/lib.rs step_inner + build_obs — the
// Rust crate remains THE reference; this kernel exists so training
// rollouts run wide (N in the thousands) with the whole step (integrate +
// judge + auto-reset + obs) in a single launch. Parity is pinned by
// tests/test_cuda_env.py (f64 exact vs Rust, f32 bounded drift).
//
// Kernel architecture:
// - One thread block (64 threads) per env. Threads 0/1 each own a hand for
//   the serial arm phase; all threads stride the chart for the expensive
//   per-object swept-surface SAT phase; thread 0 runs the control logic
//   (trackers, done, ordered reward sum) — Rust's loops verbatim.
// - Env state is a structure-of-arrays of torch tensors mutated in place;
//   charts live in a padded (P, C) pool, each env holds only a pool index
//   (reset = swap the index; no chart copying).
// - Per-object reward contributions go to a dynamic-shared scratch row and
//   are summed by thread 0 IN NOTE ORDER (Rust's accumulation order, and
//   bitwise-deterministic across runs — no float atomics in the reward
//   path). Counters use integer atomics; score adds integer-valued floats
//   (exact at any order).
// - Follow-through trackers use an aggregated ring (design note at
//   cuda_env.py N_TRACKERS): same-tick trackers share (arc, age) forever,
//   so one slot with a note count is EXACT; K = FOLLOW_TIMEOUT + 1 slots
//   makes slot reuse provably safe.
// - Expression trees mirror lib.rs term for term (the since-removed torch
//   port proved this is what makes f64 parity exact); Rust's early returns
//   stay real branches here (no side effects, same booleans).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <limits>
#include <vector>

namespace {

constexpr int N_JOINTS = 6;
constexpr int N_TRACKERS = 16;  // FOLLOW_TIMEOUT + 1 (ring; see cuda_env)
constexpr int N_NOTE_SLOTS = 16;
constexpr int NOTE_FEAT = 14;
constexpr int SABER_FEAT = 23;
constexpr int OFF_NOTES = 2 * SABER_FEAT;                    // 46
constexpr int OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT;  // 270
constexpr int OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS;        // 286
constexpr int THREADS = 64;

// --- constants (exact f64 mirror of rust/src/lib.rs) -------------------------
constexpr double DT = 1.0 / 30.0;
constexpr double DRAG = 0.02;
constexpr double V_NORM = 6.0;
constexpr double V_MIN_CUT = 1.0;
constexpr double ACC_NORM = 0.3;
constexpr double COS_TOL = 0.5;
constexpr double Y_CENTER = 1.40;
constexpr double REWARD_BAD = -0.5;
constexpr double REWARD_MISS = -0.5;
constexpr double REWARD_BOMB = -1.5;
__device__ constexpr double EFFORT_COST[N_JOINTS] = {2e-4, 2e-4, 2e-4,
                                                     1.3e-4, 0.7e-4, 0.7e-4};
constexpr double GOOD_HALF_X = 0.4;
constexpr double GOOD_HALF_Z = 0.5;
constexpr double GOOD_Z_SHIFT = -0.25;
constexpr double BAD_HALF = 0.175;
constexpr double BOMB_RADIUS = 0.18;
constexpr double SABER_LEN = 1.0;
__device__ constexpr double SHOULDER[2][3] = {{-0.2, 1.47, -0.65},
                                              {0.2, 1.47, -0.65}};
constexpr double L_UPPER = 0.35;
constexpr double L_FORE = 0.30;
constexpr double PITCH_LIM = 1.5707963267948966;
constexpr double PITCH_LO = -1.5707963267948966;
constexpr double PITCH_HI = 1.1344640137963142;
__device__ constexpr double YAW_LO[2] = {-1.7453292519943295,
                                         -0.7853981633974483};
__device__ constexpr double YAW_HI[2] = {0.7853981633974483,
                                         1.7453292519943295};
constexpr double TWIST_LIM = 1.5707963267948966;
__device__ constexpr double TWIST_LO[2] = {-1.5707963267948966,
                                           -1.2217304763960306};
__device__ constexpr double TWIST_HI[2] = {1.2217304763960306,
                                           1.5707963267948966};
constexpr double ELBOW_HI = 2.6179938779914944;
__device__ constexpr double ROLL_LO[2] = {0.0, -3.141592653589793};
__device__ constexpr double ROLL_HI[2] = {3.141592653589793, 0.0};
constexpr double BEND_LIM = 1.6580627893946132;
__device__ constexpr double JOINT_ALPHA[N_JOINTS] = {80.0, 80.0, 80.0, 80.0,
                                                     200.0, 200.0};
__device__ constexpr double JOINT_OMEGA[N_JOINTS] = {8.0, 8.0, 8.0, 8.0,
                                                     20.0, 20.0};
constexpr int N_BLADE_SAMPLES = 8;
constexpr double REACH_Z = 1.75;
constexpr double Z_MISS = 0.65;
constexpr double PRE_FULL = 1.7453292519943295;
constexpr double POST_FULL = 1.0471975511965976;
constexpr double PTS_PRE = 70.0;
constexpr double PTS_POST = 30.0;
constexpr double PTS_ACC = 15.0;
constexpr double SCORE_MAX = 115.0;
constexpr double CHAIN_LINK_PTS = 20.0;
constexpr double SWING_BREAK = 0.005;
constexpr int FOLLOW_TIMEOUT = 15;
__device__ constexpr double HEAD[3] = {0.0, 1.70, -0.65};
constexpr double LOOK_VERT = 0.8;
constexpr double LOOK_DIST = 8.0;
constexpr double X_NORM = 1.0;
constexpr double Y_NORM = 0.75;
constexpr double Z_SCALE = 8.0;
constexpr double Z_CLIP_LO = -3.0;
constexpr double Z_CLIP_HI = 12.0;

// --- small device math (mirrors lib.rs helpers) ------------------------------
template <typename T>
__device__ __forceinline__ T tmin(T a, T b) { return a < b ? a : b; }
template <typename T>
__device__ __forceinline__ T tmax(T a, T b) { return a > b ? a : b; }
template <typename T>
__device__ __forceinline__ T tclamp(T x, T lo, T hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}
// real BS scores are integers; explicit floor(x + 0.5) matches lib.rs
template <typename T>
__device__ __forceinline__ T round_pts(T x) { return floor(x + T(0.5)); }
template <typename T>
__device__ __forceinline__ T dot3(const T a[3], const T b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
template <typename T>
__device__ __forceinline__ void cross3(const T a[3], const T b[3], T o[3]) {
  o[0] = a[1] * b[2] - a[2] * b[1];
  o[1] = a[2] * b[0] - a[0] * b[2];
  o[2] = a[0] * b[1] - a[1] * b[0];
}

/// Forward kinematics (lib.rs fk): wrist + unit blade dir from the six
/// joint angles q = [pitch, yaw, twist, elbow, roll, bend].
template <typename T>
__device__ void fk(int hand, const T q[N_JOINTS], T w[3], T b[3]) {
  T sp = sin(q[0]), cp = cos(q[0]);
  T sy = sin(q[1]), cy = cos(q[1]);
  T st = sin(q[2]), ct = cos(q[2]);
  T se = sin(q[3]), ce = cos(q[3]);
  T m[3][3] = {
      {cy * ct - sy * sp * st, -cy * st - sy * sp * ct, sy * cp},
      {cp * st, cp * ct, sp},
      {-sy * ct - cy * sp * st, sy * st - cy * sp * ct, cy * cp},
  };
  T f[3];
  for (int k = 0; k < 3; k++) f[k] = m[k][1] * se + m[k][2] * ce;
  for (int k = 0; k < 3; k++)
    w[k] = T(SHOULDER[hand][k]) + T(L_UPPER) * m[k][2] + T(L_FORE) * f[k];
  T sr = sin(q[4]), cr = cos(q[4]);
  T sb = sin(q[5]), cb = cos(q[5]);
  T d0 = -sr * sb, d1 = cr * sb, d2 = cb;
  for (int k = 0; k < 3; k++)
    b[k] = m[k][0] * d0 + (m[k][1] * ce - m[k][2] * se) * d1 + f[k] * d2;
}

template <typename T>
__device__ __forceinline__ void joint_lim(int h, int j, T& lo, T& hi) {
  switch (j) {
    case 0: lo = T(PITCH_LO); hi = T(PITCH_HI); break;
    case 1: lo = T(YAW_LO[h]); hi = T(YAW_HI[h]); break;
    case 2: lo = T(TWIST_LO[h]); hi = T(TWIST_HI[h]); break;
    case 3: lo = T(0); hi = T(ELBOW_HI); break;
    case 4: lo = T(ROLL_LO[h]); hi = T(ROLL_HI[h]); break;
    default: lo = T(-BEND_LIM); hi = T(BEND_LIM); break;
  }
}

/// Triangle vs axis-aligned box, verts RELATIVE to the box center
/// (Akenine-Möller SAT; lib.rs tri_box_overlap — early returns kept).
template <typename T>
__device__ bool tri_box_overlap(const T v[3][3], const T h[3]) {
  for (int k = 0; k < 3; k++) {
    T lo = tmin(tmin(v[0][k], v[1][k]), v[2][k]);
    T hi = tmax(tmax(v[0][k], v[1][k]), v[2][k]);
    if (lo > h[k] || hi < -h[k]) return false;
  }
  T e[3][3];
  for (int k = 0; k < 3; k++) {
    e[0][k] = v[1][k] - v[0][k];
    e[1][k] = v[2][k] - v[1][k];
    e[2][k] = v[0][k] - v[2][k];
  }
  for (int j = 0; j < 3; j++) {  // 9 cross-axis tests: a = u_k x e_j
    T ax[3][3] = {{T(0), -e[j][2], e[j][1]},
                  {e[j][2], T(0), -e[j][0]},
                  {-e[j][1], e[j][0], T(0)}};
    for (int k = 0; k < 3; k++) {
      T p0 = dot3(v[0], ax[k]);
      T p1 = dot3(v[1], ax[k]);
      T p2 = dot3(v[2], ax[k]);
      T r = h[0] * fabs(ax[k][0]) + h[1] * fabs(ax[k][1])
            + h[2] * fabs(ax[k][2]);
      if (tmin(tmin(p0, p1), p2) > r || tmax(tmax(p0, p1), p2) < -r)
        return false;
    }
  }
  T n[3];
  cross3(e[0], e[1], n);
  T r = h[0] * fabs(n[0]) + h[1] * fabs(n[1]) + h[2] * fabs(n[2]);
  return fabs(dot3(n, v[0])) <= r;
}

/// Closest distance from point p to triangle (a, b, c) — Ericson's
/// 7-region cascade (lib.rs tri_point_dist), early returns kept.
template <typename T>
__device__ T tri_point_dist(const T a[3], const T b[3], const T c[3],
                            const T p[3]) {
  T ab[3], ac[3], ap[3];
  for (int k = 0; k < 3; k++) {
    ab[k] = b[k] - a[k];
    ac[k] = c[k] - a[k];
    ap[k] = p[k] - a[k];
  }
  T d1 = dot3(ab, ap);
  T d2 = dot3(ac, ap);
  T q[3];
  if (d1 <= T(0) && d2 <= T(0)) {
    for (int k = 0; k < 3; k++) q[k] = p[k] - a[k];
    return sqrt(dot3(q, q));
  }
  T bp[3];
  for (int k = 0; k < 3; k++) bp[k] = p[k] - b[k];
  T d3 = dot3(ab, bp);
  T d4 = dot3(ac, bp);
  if (d3 >= T(0) && d4 <= d3) {
    for (int k = 0; k < 3; k++) q[k] = p[k] - b[k];
    return sqrt(dot3(q, q));
  }
  T vc = d1 * d4 - d3 * d2;
  if (vc <= T(0) && d1 >= T(0) && d3 <= T(0)) {
    T denom = d1 - d3;
    T v = fabs(denom) < T(1e-15) ? T(0) : d1 / denom;
    for (int k = 0; k < 3; k++) q[k] = p[k] - (a[k] + v * ab[k]);
    return sqrt(dot3(q, q));
  }
  T cp[3];
  for (int k = 0; k < 3; k++) cp[k] = p[k] - c[k];
  T d5 = dot3(ab, cp);
  T d6 = dot3(ac, cp);
  if (d6 >= T(0) && d5 <= d6) {
    for (int k = 0; k < 3; k++) q[k] = p[k] - c[k];
    return sqrt(dot3(q, q));
  }
  T vb = d5 * d2 - d1 * d6;
  if (vb <= T(0) && d2 >= T(0) && d6 <= T(0)) {
    T denom = d2 - d6;
    T w = fabs(denom) < T(1e-15) ? T(0) : d2 / denom;
    for (int k = 0; k < 3; k++) q[k] = p[k] - (a[k] + w * ac[k]);
    return sqrt(dot3(q, q));
  }
  T va = d3 * d6 - d5 * d4;
  if (va <= T(0) && (d4 - d3) >= T(0) && (d5 - d6) >= T(0)) {
    T denom = (d4 - d3) + (d5 - d6);
    T w = fabs(denom) < T(1e-15) ? T(0) : (d4 - d3) / denom;
    for (int k = 0; k < 3; k++) q[k] = p[k] - (b[k] + w * (c[k] - b[k]));
    return sqrt(dot3(q, q));
  }
  T denom = T(1) / (va + vb + vc);
  T v = vb * denom;
  T w = vc * denom;
  for (int k = 0; k < 3; k++) q[k] = p[k] - (a[k] + ab[k] * v + ac[k] * w);
  return sqrt(dot3(q, q));
}

template <typename T>
__device__ T seg_point_dist3(const T a[3], const T b[3], const T c[3]) {
  T ab[3], ac[3];
  for (int k = 0; k < 3; k++) {
    ab[k] = b[k] - a[k];
    ac[k] = c[k] - a[k];
  }
  T denom = dot3(ab, ab);
  T t = denom < T(1e-12) ? T(0) : tclamp(dot3(ac, ab) / denom, T(0), T(1));
  T px = a[0] + t * ab[0] - c[0];
  T py = a[1] + t * ab[1] - c[1];
  T pz = a[2] + t * ab[2] - c[2];
  return sqrt(px * px + py * py + pz * pz);
}

/// Orthonormal note frame columns (lib.rs note_frame, env v7.6 look-at):
/// z' blends +z toward the (damped-y) head->note ray with weight lam; the
/// cut-direction z-roll rides as up; x' = up x z'; y' = z' x x'.
template <typename T>
__device__ void note_frame(const T p[3], T cos_r, T sin_r, T lam, T f[3][3]) {
  T up[3] = {-sin_r, cos_r, T(0)};
  T hp1 = T(HEAD[1]) + T(LOOK_VERT) * (p[1] - T(HEAD[1]));
  T d[3] = {p[0] - T(HEAD[0]), p[1] - hp1, p[2] - T(HEAD[2])};
  T dn = tmax(sqrt(dot3(d, d)), T(1e-9));
  T g[3] = {lam * d[0] / dn, lam * d[1] / dn,
            (T(1) - lam) + lam * d[2] / dn};
  T gn = tmax(sqrt(dot3(g, g)), T(1e-9));
  for (int k = 0; k < 3; k++) f[2][k] = g[k] / gn;
  T cx[3];
  cross3(up, f[2], cx);
  T cn = tmax(sqrt(dot3(cx, cx)), T(1e-9));
  for (int k = 0; k < 3; k++) f[0][k] = cx[k] / cn;
  cross3(f[2], f[0], f[1]);
}

/// One point into the note frame (translate to c, project on f's rows).
template <typename T>
__device__ void to_note_frame(const T p[3], const T c[3], const T f[3][3],
                              T out[3]) {
  T q[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
  out[0] = dot3(q, f[0]);
  out[1] = dot3(q, f[1]);
  out[2] = dot3(q, f[2]);
}

/// The swept blade surface = quad (w0->t0, w1->t1) as triangles
/// (w0, t0, t1) and (w0, t1, w1) — box test in the note frame. The two
/// triangles are unrolled explicitly: an indexed pointer array here
/// spills to local memory (48 B stack) and taxes every vertex fetch in
/// the per-object judgment loop.
template <typename T>
__device__ bool sweep_hits_box(const T w0[3], const T t0[3], const T t1[3],
                               const T w1[3], const T c[3], const T h[3],
                               const T f[3][3]) {
  T rel[3][3];
  to_note_frame(w0, c, f, rel[0]);
  to_note_frame(t0, c, f, rel[1]);
  to_note_frame(t1, c, f, rel[2]);
  if (tri_box_overlap(rel, h)) return true;
  // second triangle (w0, t1, w1): rel[0] (w0) is already in place
  to_note_frame(t1, c, f, rel[1]);
  to_note_frame(w1, c, f, rel[2]);
  return tri_box_overlap(rel, h);
}

template <typename T>
__device__ T sweep_dist(const T w0[3], const T t0[3], const T t1[3],
                        const T w1[3], const T p[3]) {
  return tmin(tri_point_dist(w0, t0, t1, p), tri_point_dist(w0, t1, w1, p));
}

// --- kernel parameters (structure-of-arrays pointers) ------------------------
template <typename T>
struct Sim {
  // per-env state (mutated in place)
  T *q, *om, *wrist, *blade, *wrist_vel, *swing_acc, *axis;
  bool* axis_valid;
  T* tr_acc;
  int *tr_age, *tr_cnt;
  bool* tr_active;
  int* tr_ptr;
  int* tick;
  uint8_t* status;
  int* first_pending;  // per-env low-water cursor over status (monotone)
  T* ep_return;
  int *good, *bad, *miss, *bombs;
  T* score;
  // score decomposition (2026-07-25): pre/acc/post point totals
  T *score_pre, *score_acc, *score_post;
  int64_t* chart_idx;
  // chart pool (P, C), read-only
  const T *p_hit, *p_speed, *p_center, *p_dvec, *p_rot, *p_good_hy, *p_bad_hy;
  const bool *p_valid, *p_is_right, *p_any_dir, *p_is_bomb, *p_is_head,
      *p_is_link, *p_show_dir, *p_free_pre, *p_free_post;
  const int64_t* p_n_notes;
  const T *p_last_hit, *p_tail;
  // io
  const int64_t *actions, *cand;
  T* ret_buf;
  int* ret_total;
  float* obs;
  T* reward;
  bool* done;
  int C, max_steps, cap, mode;  // mode: 0 = step, 1 = reset-all
  T effort_coef;
  bool auto_reset;
};

// One block per env; the whole step (integrate, judge, done, auto-reset,
// obs) in one launch. Mirrors lib.rs step_inner's phase order exactly.
template <typename T>
__global__ void step_kernel(Sim<T> P) {
  const int e = blockIdx.x;
  const int tid = threadIdx.x;
  const int C = P.C;
  extern __shared__ unsigned char smem_raw[];
  T* scratch = reinterpret_cast<T*>(smem_raw);  // per-object reward terms

  __shared__ T sh_pw[2][3], sh_pd[2][3], sh_w[2][3], sh_d[2][3];
  __shared__ T sh_t0[2][3], sh_t1[2][3];
  __shared__ T sh_delta[2], sh_swing[2];
  __shared__ bool sh_cont[2];
  __shared__ int sh_nnew[2];
  __shared__ int sh_good, sh_bad, sh_miss, sh_bomb;
  __shared__ T sh_score, sh_rew;
  __shared__ T sh_pre, sh_acc, sh_post;
  __shared__ bool sh_doreset;
  __shared__ int64_t sh_chart;
  __shared__ int sh_slots[N_NOTE_SLOTS];
  __shared__ int sh_nslots;

  const int64_t c0 = P.chart_idx[e];
  const bool stepping = (P.mode == 0);

  if (stepping) {
    if (tid == 0) {
      sh_good = sh_bad = sh_miss = sh_bomb = 0;
      sh_nnew[0] = sh_nnew[1] = 0;
      sh_score = T(0);
      sh_pre = sh_acc = sh_post = T(0);
    }
    for (int i = tid; i < C; i += blockDim.x) scratch[i] = T(0);

    // Phase A (threads 0/1, one hand each): joint integrator, FK, swept
    // quad, swing-chain update (lib.rs per-arm loop)
    if (tid < 2) {
      const int h = tid;
      const int qb = (e * 2 + h) * N_JOINTS;
      const int wb = (e * 2 + h) * 3;
      T q[N_JOINTS], om[N_JOINTS], pw[3], pd[3];
      for (int j = 0; j < N_JOINTS; j++) {
        q[j] = P.q[qb + j];
        om[j] = P.om[qb + j];
      }
      for (int k = 0; k < 3; k++) {
        pw[k] = P.wrist[wb + k];
        pd[k] = P.blade[wb + k];
        sh_pw[h][k] = pw[k];
        sh_pd[h][k] = pd[k];
      }
      for (int j = 0; j < N_JOINTS; j++) {
        const int a = (int)P.actions[e * 12 + h * N_JOINTS + j];
        const T al = T(JOINT_ALPHA[j]);
        om[j] += a == 1 ? al * T(DT) : (a == 2 ? -al * T(DT) : T(0));
        const T omax = T(JOINT_OMEGA[j]);
        om[j] = tclamp(om[j] * (T(1) - T(DRAG)), -omax, omax);
        q[j] += om[j] * T(DT);
        T lo, hi;
        joint_lim(h, j, lo, hi);
        if (q[j] < lo || q[j] > hi) {  // hitting a limit kills omega
          q[j] = tclamp(q[j], lo, hi);
          om[j] = T(0);
        }
        P.q[qb + j] = q[j];
        P.om[qb + j] = om[j];
      }
      T w[3], b[3];
      fk(h, q, w, b);
      for (int k = 0; k < 3; k++) {
        P.wrist[wb + k] = w[k];
        P.blade[wb + k] = b[k];
        P.wrist_vel[wb + k] = (w[k] - pw[k]) / T(DT);
        sh_w[h][k] = w[k];
        sh_d[h][k] = b[k];
        sh_t0[h][k] = pw[k] + T(SABER_LEN) * pd[k];
        sh_t1[h][k] = w[k] + T(SABER_LEN) * b[k];
      }
      // swing chain (feeds pre-swing + follow-through)
      const T dp = tclamp(dot3(pd, b), T(-1), T(1));
      const T delta = acos(dp);
      T sacc = P.swing_acc[e * 2 + h];
      bool cont = false;
      if (delta < T(SWING_BREAK)) {
        sacc = T(0);
        P.axis_valid[e * 2 + h] = false;
      } else {
        T cx[3];
        cross3(pd, b, cx);
        const T cn = sqrt(dot3(cx, cx));
        const T ax[3] = {cx[0] / cn, cx[1] / cn, cx[2] / cn};
        const int ab = (e * 2 + h) * 3;
        const bool rev = P.axis_valid[e * 2 + h]
            && (P.axis[ab] * ax[0] + P.axis[ab + 1] * ax[1]
                + P.axis[ab + 2] * ax[2]) < T(0);
        if (rev) {
          sacc = delta;  // reversal: a new swing starts
        } else {
          sacc += delta;
          cont = true;
        }
        for (int k = 0; k < 3; k++) P.axis[ab + k] = ax[k];
        P.axis_valid[e * 2 + h] = true;
      }
      P.swing_acc[e * 2 + h] = sacc;
      sh_swing[h] = sacc;
      sh_delta[h] = delta;
      sh_cont[h] = cont;
    }
    __syncthreads();

    // Phase B (thread 0): effort cost + existing-tracker resolution
    // (BEFORE this tick's cuts, lib.rs order)
    if (tid == 0) {
      T rew = T(0);
      T eff = T(0);
      for (int h = 0; h < 2; h++)
        for (int j = 0; j < N_JOINTS; j++)
          if (P.actions[e * 12 + h * N_JOINTS + j] != 0)
            eff += T(EFFORT_COST[j]);
      rew -= P.effort_coef * eff;
      for (int h = 0; h < 2; h++) {
        for (int s = 0; s < N_TRACKERS; s++) {
          const int b = (e * 2 + h) * N_TRACKERS + s;
          if (!P.tr_active[b]) continue;
          T acc = P.tr_acc[b];
          bool resolve;
          if (!sh_cont[h]) {
            resolve = true;  // chain broke or reversed
          } else {
            acc += sh_delta[h];
            const int age = P.tr_age[b] + 1;
            P.tr_acc[b] = acc;
            P.tr_age[b] = age;
            resolve = acc >= T(POST_FULL) || age >= FOLLOW_TIMEOUT;
          }
          if (resolve) {
            const T pts = round_pts(T(PTS_POST)
                                    * tmin(acc / T(POST_FULL), T(1)));
            const T tot = pts * T(P.tr_cnt[b]);
            rew += tot / T(SCORE_MAX);
            sh_score += tot;
            sh_post += tot;
            P.tr_active[b] = false;
          }
        }
      }
      sh_rew = rew;
    }
    __syncthreads();

    // Phase C (all threads stride the chart): per-object judgment — the
    // expensive swept-surface SAT stage. Each object is independent;
    // reward terms land in scratch[i] for the ordered sum in phase D.
    const T time_new = T(P.tick[e] + 1) * T(DT);
    // everything below the low-water cursor is resolved (or padding):
    // skip the prefix instead of loading and branching over it
    const int fp0 = P.first_pending[e];
    for (int i = fp0 + tid; i < C; i += blockDim.x) {
      const int64_t cb = c0 * C + i;
      if (!P.p_valid[cb] || P.status[e * C + i] != 0) continue;
      const T z = (P.p_hit[cb] - time_new) * P.p_speed[cb];
      const bool bomb = P.p_is_bomb[cb];
      if (z < T(-Z_MISS)) {
        if (bomb) {
          P.status[e * C + i] = 4;  // passed clean: silent
        } else {
          P.status[e * C + i] = 3;
          atomicAdd(&sh_miss, 1);
          scratch[i] = T(REWARD_MISS);
        }
        continue;
      }
      if (z > T(REACH_Z)) continue;
      const T center[3] = {P.p_center[cb * 2], P.p_center[cb * 2 + 1], z};
      if (bomb) {  // either saber's sweep within BOMB_RADIUS of center
        for (int h = 0; h < 2; h++) {
          if (sweep_dist(sh_pw[h], sh_t0[h], sh_t1[h], sh_w[h], center)
              <= T(BOMB_RADIUS)) {
            P.status[e * C + i] = 2;
            atomicAdd(&sh_bomb, 1);
            scratch[i] = T(REWARD_BOMB);
            break;
          }
        }
        continue;
      }
      const int m = P.p_is_right[cb] ? 1 : 0;  // matching saber
      const bool is_head = P.p_is_head[cb];
      const bool is_link = P.p_is_link[cb];
      // env v7.6: plain notes billboard toward the head; heads/links keep
      // the pure z-roll (lam = 0)
      const T lam = (!is_head && !is_link)
          ? tclamp(T(1) - z / T(LOOK_DIST), T(0), T(1))
          : T(0);
      T frame[3][3];
      note_frame(center, P.p_rot[cb * 2], P.p_rot[cb * 2 + 1], lam, frame);
      // the good box's 0.25 m shift rides the note's LOCAL z
      const T gc[3] = {center[0] + T(GOOD_Z_SHIFT) * frame[2][0],
                       center[1] + T(GOOD_Z_SHIFT) * frame[2][1],
                       center[2] + T(GOOD_Z_SHIFT) * frame[2][2]};
      const T good_h[3] = {T(GOOD_HALF_X), P.p_good_hy[cb], T(GOOD_HALF_Z)};
      if (sweep_hits_box(sh_pw[m], sh_t0[m], sh_t1[m], sh_w[m], gc, good_h,
                         frame)) {
        // contact point: blade sample whose displacement segment passes
        // closest to the center; first strict minimum wins (Rust's `<`)
        T d_seg = std::numeric_limits<T>::infinity();
        T s_star = T(0);
        for (int j = 1; j <= N_BLADE_SAMPLES; j++) {
          const T r = (T(j) / T(N_BLADE_SAMPLES)) * T(SABER_LEN);
          T p0[3], p1[3];
          for (int k = 0; k < 3; k++) {
            p0[k] = sh_pw[m][k] + r * sh_pd[m][k];
            p1[k] = sh_w[m][k] + r * sh_d[m][k];
          }
          const T d = seg_point_dist3(p0, p1, center);
          if (d < d_seg) {
            d_seg = d;
            s_star = r;
          }
        }
        T vc[3];
        for (int k = 0; k < 3; k++)
          vc[k] = (sh_w[m][k] + s_star * sh_d[m][k] - sh_pw[m][k]
                   - s_star * sh_pd[m][k]) / T(DT);
        const T vc_mag = sqrt(vc[0] * vc[0] + vc[1] * vc[1] + vc[2] * vc[2]);
        const T d_min = sweep_dist(sh_pw[m], sh_t0[m], sh_t1[m], sh_w[m],
                                   center);
        int code;
        if (vc_mag < T(V_MIN_CUT)) {
          code = 1;  // slow
        } else if (P.p_any_dir[cb]) {
          code = 0;  // dot / chain link: any direction
        } else {
          // env v7.6: direction check in the NOTE'S LOCAL frame —
          // required direction is local -y'
          const T vlx = dot3(vc, frame[0]);
          const T vly = dot3(vc, frame[1]);
          const T v2 = sqrt(vlx * vlx + vly * vly);
          const T cosv = -vly / tmax(v2, T(1e-9));
          code = cosv >= T(COS_TOL) ? 0 : 2;  // ok / angle
        }
        if (code == 0) {
          P.status[e * C + i] = 1;
          atomicAdd(&sh_good, 1);
          T pts;
          if (is_link) {
            pts = T(CHAIN_LINK_PTS);  // flat, regardless of swing
          } else {
            const T pre = P.p_free_pre[cb]
                ? T(PTS_PRE)
                : round_pts(T(PTS_PRE)
                            * tmin(sh_swing[m] / T(PRE_FULL), T(1)));
            const T acc = round_pts(
                T(PTS_ACC) * (T(1) - tclamp(d_min / T(ACC_NORM), T(0), T(1))));
            pts = pre + acc;
            atomicAdd(&sh_pre, pre);
            atomicAdd(&sh_acc, acc);
            if (is_head) {
              // chain heads: no follow-through (max 85)
            } else if (P.p_free_post[cb]) {
              pts += T(PTS_POST);  // arc head: granted in full
              atomicAdd(&sh_post, T(PTS_POST));
            } else {
              atomicAdd(&sh_nnew[m], 1);  // spawn a follow-through tracker
            }
          }
          scratch[i] = pts / T(SCORE_MAX);
          atomicAdd(&sh_score, pts);  // integer-valued: exact at any order
        } else {
          P.status[e * C + i] = 2;
          atomicAdd(&sh_bad, 1);
          scratch[i] = T(REWARD_BAD);
        }
      } else {
        // wrong-color saber only triggers the SMALL box
        const T bad_h[3] = {T(BAD_HALF), P.p_bad_hy[cb], T(BAD_HALF)};
        if (sweep_hits_box(sh_pw[1 - m], sh_t0[1 - m], sh_t1[1 - m],
                           sh_w[1 - m], center, bad_h, frame)) {
          P.status[e * C + i] = 2;
          atomicAdd(&sh_bad, 1);
          scratch[i] = T(REWARD_BAD);
        }
      }
    }
    __syncthreads();

    // Phase D (thread 0): tracker spawn (aggregated ring slot), done,
    // ordered reward sum, episode bookkeeping
    if (tid == 0) {
      for (int h = 0; h < 2; h++) {
        const int n = sh_nnew[h];
        if (n > 0) {
          const int ptr = P.tr_ptr[e * 2 + h];
          const int b = (e * 2 + h) * N_TRACKERS + ptr;
          P.tr_acc[b] = T(0);
          P.tr_age[b] = 0;
          P.tr_cnt[b] = n;
          P.tr_active[b] = true;
          P.tr_ptr[e * 2 + h] = (ptr + 1) % N_TRACKERS;
        }
      }
      const int tick_new = P.tick[e] + 1;
      P.tick[e] = tick_new;
      const T time_now = T(tick_new) * T(DT);
      // advance the low-water cursor: "smallest pending index" is
      // monotone within an episode (status only goes 0 -> nonzero
      // between resets); padding (p_valid false) counts as resolved.
      // Amortized O(1) per tick, and the pending test becomes free.
      int fp = P.first_pending[e];
      while (fp < C
             && !(P.p_valid[c0 * C + fp] && P.status[e * C + fp] == 0))
        fp++;
      P.first_pending[e] = fp;
      const bool pending = fp < C;
      bool no_trk = true;
      for (int b = e * 2 * N_TRACKERS; b < (e + 1) * 2 * N_TRACKERS; b++)
        if (P.tr_active[b]) {
          no_trk = false;
          break;
        }
      const int64_t nn = P.p_n_notes[c0];
      const T last_hit = P.p_last_hit[c0];
      const bool done =
          (!pending && no_trk && (nn == 0 || time_now > last_hit))
          || time_now > last_hit + P.p_tail[c0] || tick_new >= P.max_steps;
      T rew = sh_rew;
      for (int i = 0; i < C; i++) rew += scratch[i];  // Rust's note order
      T score_add = sh_score;
      T post_add = sh_post;
      if (done) {  // hard ends force-resolve pending follow-throughs
        for (int b = (e * 2) * N_TRACKERS; b < (e * 2 + 2) * N_TRACKERS;
             b++) {
          if (!P.tr_active[b]) continue;
          const T pts = round_pts(T(PTS_POST)
                                  * tmin(P.tr_acc[b] / T(POST_FULL), T(1)));
          const T tot = pts * T(P.tr_cnt[b]);
          rew += tot / T(SCORE_MAX);
          score_add += tot;
          post_add += tot;
          P.tr_active[b] = false;
        }
      }
      P.good[e] += sh_good;
      P.bad[e] += sh_bad;
      P.miss[e] += sh_miss;
      P.bombs[e] += sh_bomb;
      P.score[e] += score_add;
      P.score_pre[e] += sh_pre;
      P.score_acc[e] += sh_acc;
      P.score_post[e] += post_add;
      const T ep = P.ep_return[e] + rew;
      P.ep_return[e] = ep;
      P.reward[e] = rew;
      P.done[e] = done;
      if (done) {  // episode-return ring buffer (order logging-grade)
        const int pos = atomicAdd(P.ret_total, 1);
        P.ret_buf[pos % P.cap] = ep;
      }
      sh_doreset = done && P.auto_reset;
      sh_chart = c0;
    }
  } else {
    if (tid == 0) {
      sh_doreset = true;
      sh_chart = c0;
    }
  }
  __syncthreads();

  // Phase E: reset (done envs under auto-reset, or everyone in mode 1) —
  // start pose all-zero joints at rest, fresh chart from cand
  if (sh_doreset) {
    if (tid == 0) {
      sh_chart = P.cand[e];
      P.chart_idx[e] = sh_chart;
      P.tick[e] = 0;
      P.ep_return[e] = T(0);
      P.good[e] = 0;
      P.bad[e] = 0;
      P.miss[e] = 0;
      P.bombs[e] = 0;
      P.score[e] = T(0);
      P.score_pre[e] = T(0);
      P.score_acc[e] = T(0);
      P.score_post[e] = T(0);
      P.tr_ptr[e * 2] = 0;
      P.tr_ptr[e * 2 + 1] = 0;
      P.first_pending[e] = 0;
    }
    if (tid < 2) {
      const int h = tid;
      const int qb = (e * 2 + h) * N_JOINTS;
      const int wb = (e * 2 + h) * 3;
      T q0[N_JOINTS] = {T(0), T(0), T(0), T(0), T(0), T(0)};
      T w[3], b[3];
      fk(h, q0, w, b);
      for (int j = 0; j < N_JOINTS; j++) {
        P.q[qb + j] = T(0);
        P.om[qb + j] = T(0);
      }
      for (int k = 0; k < 3; k++) {
        P.wrist[wb + k] = w[k];
        P.blade[wb + k] = b[k];
        P.wrist_vel[wb + k] = T(0);
      }
      P.swing_acc[e * 2 + h] = T(0);
      P.axis_valid[e * 2 + h] = false;
    }
    for (int b = tid; b < 2 * N_TRACKERS; b += blockDim.x)
      P.tr_active[e * 2 * N_TRACKERS + b] = false;
    for (int i = tid; i < C; i += blockDim.x) P.status[e * C + i] = 0;
  }
  __syncthreads();

  // Phase F: observation (mirrors lib.rs build_obs), written as f32
  const int64_t cc = sh_chart;
  float* obs = P.obs + (int64_t)e * OBS_DIM;
  for (int k = tid; k < OBS_DIM; k += blockDim.x) obs[k] = 0.f;
  __syncthreads();
  if (tid < 2) {
    const int h = tid;
    const int base = h * SABER_FEAT;
    const int qb = (e * 2 + h) * N_JOINTS;
    const int wb = (e * 2 + h) * 3;
    obs[base + 0] = (float)(P.q[qb + 0] / T(PITCH_LIM));
    obs[base + 1] = (float)(P.q[qb + 1] / T(YAW_HI[1]));
    obs[base + 2] = (float)(P.q[qb + 2] / T(TWIST_LIM));
    obs[base + 3] = (float)((T(2) * P.q[qb + 3] - T(ELBOW_HI)) / T(ELBOW_HI));
    obs[base + 4] = (float)cos(P.q[qb + 4]);
    obs[base + 5] = (float)sin(P.q[qb + 4]);
    obs[base + 6] = (float)(P.q[qb + 5] / T(BEND_LIM));
    for (int j = 0; j < N_JOINTS; j++)
      obs[base + 7 + j] = (float)(P.om[qb + j] / T(JOINT_OMEGA[j]));
    obs[base + 13] = (float)(P.wrist[wb + 0] / T(X_NORM));
    obs[base + 14] = (float)((P.wrist[wb + 1] - T(Y_CENTER)) / T(Y_NORM));
    obs[base + 15] = (float)P.wrist[wb + 2];
    for (int k = 0; k < 3; k++) {
      obs[base + 16 + k] = (float)(P.wrist_vel[wb + k] / T(V_NORM));
      obs[base + 19 + k] = (float)P.blade[wb + k];
    }
    obs[base + 22] =
        (float)tmin(P.swing_acc[e * 2 + h] / T(PRE_FULL), T(2));
  }
  if (tid == 0) {  // next 16 pending objects in chart (arrival) order
    // start at the low-water cursor (fresh 0 after a phase-E reset):
    // typically terminates within ~N_NOTE_SLOTS iterations, not O(C)
    int ns = 0;
    for (int i = P.first_pending[e]; i < C && ns < N_NOTE_SLOTS; i++)
      if (P.p_valid[cc * C + i] && P.status[e * C + i] == 0)
        sh_slots[ns++] = i;
    sh_nslots = ns;
  }
  __syncthreads();
  const T time_o = T(P.tick[e]) * T(DT);
  for (int s = tid; s < sh_nslots; s += blockDim.x) {
    const int i = sh_slots[s];
    const int64_t cb = cc * C + i;
    float* nb = obs + OFF_NOTES + s * NOTE_FEAT;
    const T z = tclamp((P.p_hit[cb] - time_o) * P.p_speed[cb], T(Z_CLIP_LO),
                       T(Z_CLIP_HI));
    const bool bomb = P.p_is_bomb[cb];
    nb[0] = (float)(P.p_center[cb * 2] / T(X_NORM));
    nb[1] = (float)((P.p_center[cb * 2 + 1] - T(Y_CENTER)) / T(Y_NORM));
    nb[2] = (float)(z / T(Z_SCALE));
    nb[3] = (float)(-P.p_speed[cb] / T(Z_SCALE));
    nb[4] = (!bomb && !P.p_is_right[cb]) ? 1.f : 0.f;
    nb[5] = (!bomb && P.p_is_right[cb]) ? 1.f : 0.f;
    nb[6] = P.p_show_dir[cb] ? (float)P.p_dvec[cb * 2] : 0.f;
    nb[7] = P.p_show_dir[cb] ? (float)P.p_dvec[cb * 2 + 1] : 0.f;
    nb[8] = bomb ? 1.f : 0.f;
    nb[9] = P.p_is_head[cb] ? 1.f : 0.f;
    nb[10] = P.p_is_link[cb] ? 1.f : 0.f;
    nb[11] = P.p_free_pre[cb] ? 1.f : 0.f;
    nb[12] = P.p_free_post[cb] ? 1.f : 0.f;
    nb[13] = 1.f;  // valid
    obs[OFF_NOTE_MASK + s] = 1.f;
  }
}

// tensor-list slots (order is a contract with cuda_env.py _tensors)
enum {
  TQ = 0, TOM, TWRIST, TBLADE, TWVEL, TSWING, TAXIS, TAXV,
  TTRACC, TTRAGE, TTRCNT, TTRACT, TTRPTR, TTICK, TSTATUS, TFIRSTP, TEPRET,
  TGOOD, TBAD, TMISS, TBOMBS, TSCORE, TSPRE, TSACC, TSPOST, TCIDX,
  PHIT, PSPEED, PCENTER, PDVEC, PROT, PGHY, PBHY,
  PVALID, PISR, PANY, PBOMB, PHEAD, PLINK, PSHOW, PFPRE, PFPOST,
  PNNOTES, PLASTHIT, PTAIL,
  RBUF, RTOT, OOBS, OREW, ODONE, T_COUNT
};

void step(std::vector<at::Tensor> ts, at::Tensor actions, at::Tensor cand,
          int64_t mode, int64_t max_steps, double effort_coef,
          bool auto_reset) {
  TORCH_CHECK((int)ts.size() == T_COUNT, "expected ", (int)T_COUNT,
              " state tensors, got ", ts.size());
  const auto& q = ts[TQ];
  TORCH_CHECK(q.is_cuda(), "state must live on CUDA");
  TORCH_CHECK(actions.scalar_type() == at::kLong && actions.is_contiguous());
  TORCH_CHECK(cand.scalar_type() == at::kLong && cand.is_contiguous());
  const int64_t N = q.size(0);
  const int64_t C = ts[TSTATUS].size(1);
  const int64_t cap = ts[RBUF].size(0);
  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "beat_arena_step", [&] {
    Sim<scalar_t> P;
    P.q = ts[TQ].data_ptr<scalar_t>();
    P.om = ts[TOM].data_ptr<scalar_t>();
    P.wrist = ts[TWRIST].data_ptr<scalar_t>();
    P.blade = ts[TBLADE].data_ptr<scalar_t>();
    P.wrist_vel = ts[TWVEL].data_ptr<scalar_t>();
    P.swing_acc = ts[TSWING].data_ptr<scalar_t>();
    P.axis = ts[TAXIS].data_ptr<scalar_t>();
    P.axis_valid = ts[TAXV].data_ptr<bool>();
    P.tr_acc = ts[TTRACC].data_ptr<scalar_t>();
    P.tr_age = ts[TTRAGE].data_ptr<int>();
    P.tr_cnt = ts[TTRCNT].data_ptr<int>();
    P.tr_active = ts[TTRACT].data_ptr<bool>();
    P.tr_ptr = ts[TTRPTR].data_ptr<int>();
    P.tick = ts[TTICK].data_ptr<int>();
    P.status = ts[TSTATUS].data_ptr<uint8_t>();
    P.first_pending = ts[TFIRSTP].data_ptr<int>();
    P.ep_return = ts[TEPRET].data_ptr<scalar_t>();
    P.good = ts[TGOOD].data_ptr<int>();
    P.bad = ts[TBAD].data_ptr<int>();
    P.miss = ts[TMISS].data_ptr<int>();
    P.bombs = ts[TBOMBS].data_ptr<int>();
    P.score = ts[TSCORE].data_ptr<scalar_t>();
    P.score_pre = ts[TSPRE].data_ptr<scalar_t>();
    P.score_acc = ts[TSACC].data_ptr<scalar_t>();
    P.score_post = ts[TSPOST].data_ptr<scalar_t>();
    P.chart_idx = ts[TCIDX].data_ptr<int64_t>();
    P.p_hit = ts[PHIT].data_ptr<scalar_t>();
    P.p_speed = ts[PSPEED].data_ptr<scalar_t>();
    P.p_center = ts[PCENTER].data_ptr<scalar_t>();
    P.p_dvec = ts[PDVEC].data_ptr<scalar_t>();
    P.p_rot = ts[PROT].data_ptr<scalar_t>();
    P.p_good_hy = ts[PGHY].data_ptr<scalar_t>();
    P.p_bad_hy = ts[PBHY].data_ptr<scalar_t>();
    P.p_valid = ts[PVALID].data_ptr<bool>();
    P.p_is_right = ts[PISR].data_ptr<bool>();
    P.p_any_dir = ts[PANY].data_ptr<bool>();
    P.p_is_bomb = ts[PBOMB].data_ptr<bool>();
    P.p_is_head = ts[PHEAD].data_ptr<bool>();
    P.p_is_link = ts[PLINK].data_ptr<bool>();
    P.p_show_dir = ts[PSHOW].data_ptr<bool>();
    P.p_free_pre = ts[PFPRE].data_ptr<bool>();
    P.p_free_post = ts[PFPOST].data_ptr<bool>();
    P.p_n_notes = ts[PNNOTES].data_ptr<int64_t>();
    P.p_last_hit = ts[PLASTHIT].data_ptr<scalar_t>();
    P.p_tail = ts[PTAIL].data_ptr<scalar_t>();
    P.actions = actions.data_ptr<int64_t>();
    P.cand = cand.data_ptr<int64_t>();
    P.ret_buf = ts[RBUF].data_ptr<scalar_t>();
    P.ret_total = ts[RTOT].data_ptr<int>();
    P.obs = ts[OOBS].data_ptr<float>();
    P.reward = ts[OREW].data_ptr<scalar_t>();
    P.done = ts[ODONE].data_ptr<bool>();
    P.C = (int)C;
    P.max_steps = (int)max_steps;
    P.cap = (int)cap;
    P.mode = (int)mode;
    P.effort_coef = (scalar_t)effort_coef;
    P.auto_reset = auto_reset;
    const size_t shmem = (size_t)C * sizeof(scalar_t);
    step_kernel<scalar_t><<<(unsigned)N, THREADS, shmem,
                            at::cuda::getCurrentCUDAStream()>>>(P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("step", &step,
        "One sim step (mode 0) or reset-all (mode 1) for every env; "
        "state tensors are mutated in place, obs/reward/done written "
        "into their output slots");
}
