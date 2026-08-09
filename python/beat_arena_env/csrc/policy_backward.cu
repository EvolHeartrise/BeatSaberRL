// Fused NotePolicy backward — env v10 layout (P5): ALL parameter
// gradients in one kernel launch (+ one tiny reduce), with in-kernel
// forward RECOMPUTATION.
//
// Companion to policy_forward.cu (same packed-weight contract: obs 340,
// 19 tokens [saber_L, saber_R, body, note_0..15], body MLP, trunk
// 192 -> 128, 29 x K heads + value). The autograd boundary lives in
// policy_cuda.FusedPolicyFn: python computes the PPO loss on the tiny
// (mb, 29, K)/(mb,) outputs and hands this kernel d(loss)/d(logits) and
// d(loss)/d(value); the kernel produces d(loss)/d(packed weights). No
// grad w.r.t. obs (inputs are data).
//
// Memory design (the 2026-07-25 OOM lesson, unchanged): the ONLY
// per-row tensors are the obs minibatch and the (mb, NOUT) grads —
// everything else is recomputed per sample inside the block:
//   Phase A   forward recompute, saving in smem only each transformer
//             block's INPUT token stream + LN stats + trunk activations.
//   Phase B   head/trunk backward from (grad_logits, grad_value).
//   Phase C   blocks in reverse; re-materialized intermediates.
//   Phase D   front note/saber/body MLP backward.
// Recompute passes reuse the SAME arithmetic (shared device functions,
// saved LN stats), so re-materialized activations are bitwise identical
// to phase A's.
//
// v10 smem diet (the 19th token + body outgrew the 48 KB budget): the
// obs row is read straight from global via __ldg (L2-resident across
// the block's many passes), and df1 is computed IN PLACE over f1 (the
// ReLU mask reads each element before overwriting it; dWf2 completes
// first behind a barrier). Total 46.9 KB/block.
//
// Gradient accumulation across rows: striped atomics over `ncopies`
// partial buffers + a grid-stride reduce (fp32 atomics reassociate ->
// grads are NOT bitwise run-to-run at default ncopies; ncopies >= mb
// makes the whole backward deterministic — tests use both).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

constexpr int D = 64;           // embed
constexpr int TT = 19;          // tokens: 2 sabers + body + 16 note slots
constexpr int TRUNK = 128;
constexpr int NOTE_FEAT = 14;
constexpr int N_NOTE_SLOTS = 16;
constexpr int SABER_FEAT = 25;
constexpr int SABER_IN = SABER_FEAT + 2;
constexpr int BODY_FEAT = 54;  // env v12: + the 4 ground-contact lanes
constexpr int OFF_NOTES = 100;
constexpr int OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT;  // 324
constexpr int OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS + 4; // 344 (env v12)
constexpr int N_HEADS = 29;
constexpr int MAX_K = 7;
constexpr int MAX_NOUT = N_HEADS * MAX_K + 1;         // 204
constexpr int THREADS = 128;
constexpr int SROW = D + 1;     // padded token-row stride
constexpr int MAXB = 2;         // saved block inputs the smem can hold

__device__ constexpr int SABER_L_IDX[SABER_FEAT] = {
    18, 19, 20, 21, 22, 23, 24, 25,
    49, 50, 51, 52, 53, 54, 55,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 84};
__device__ constexpr int SABER_R_IDX[SABER_FEAT] = {
    26, 27, 28, 29, 30, 31, 32, 33,
    56, 57, 58, 59, 60, 61, 62,
    85, 86, 87, 88, 89, 90, 91, 92, 93, 94};
__device__ constexpr int BODY_IDX[BODY_FEAT] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    95, 96, 97, 98, 99,
    340, 341, 342, 343};  // env v12 ground-contact lanes

// --- packed offsets (contract with policy_cuda._pack; == forward's) --------
constexpr int O_NOTE_W1 = 0;
constexpr int O_NOTE_B1 = O_NOTE_W1 + NOTE_FEAT * D;
constexpr int O_NOTE_W2 = O_NOTE_B1 + D;
constexpr int O_NOTE_B2 = O_NOTE_W2 + D * D;
constexpr int O_SAB_W1 = O_NOTE_B2 + D;
constexpr int O_SAB_B1 = O_SAB_W1 + SABER_IN * D;
constexpr int O_SAB_W2 = O_SAB_B1 + D;
constexpr int O_SAB_B2 = O_SAB_W2 + D * D;
constexpr int O_BODY_W1 = O_SAB_B2 + D;
constexpr int O_BODY_B1 = O_BODY_W1 + BODY_FEAT * D;
constexpr int O_BODY_W2 = O_BODY_B1 + D;
constexpr int O_BODY_B2 = O_BODY_W2 + D * D;
constexpr int FRONT = O_BODY_B2 + D;                  // 18752 (v12)
constexpr int B_LN1_G = 0;
constexpr int B_LN1_B = B_LN1_G + D;
constexpr int B_WQ = B_LN1_B + D;
constexpr int B_BQ = B_WQ + D * D;
constexpr int B_WK = B_BQ + D;
constexpr int B_BK = B_WK + D * D;
constexpr int B_WV = B_BK + D;
constexpr int B_BV = B_WV + D * D;
constexpr int B_LN2_G = B_BV + D;
constexpr int B_LN2_B = B_LN2_G + D;
constexpr int B_FFW1 = B_LN2_B + D;
constexpr int B_FFB1 = B_FFW1 + D * TRUNK;
constexpr int B_FFW2 = B_FFB1 + TRUNK;
constexpr int B_FFB2 = B_FFW2 + TRUNK * D;
constexpr int BLK = B_FFB2 + D;                       // 29312
constexpr int T_W1 = 0;
constexpr int T_B1 = T_W1 + 3 * D * TRUNK;
constexpr int T_W2 = T_B1 + TRUNK;
constexpr int T_B2 = T_W2 + TRUNK * TRUNK;
constexpr int TRUNK_SZ = T_B2 + TRUNK;                // 41216
static_assert(FRONT == 18752 && BLK == 29312 && TRUNK_SZ == 41216,
              "packed layout drifted");

constexpr int expected_numel(int n_blocks, int n_gears) {
  return FRONT + n_blocks * BLK + TRUNK_SZ
         + (TRUNK + 1) * (N_HEADS * n_gears + 1);
}

struct BSmem {
  float madd[TT];                 // (key_mask - 1) * 16384
  float xin[MAXB][TT][SROW];      // saved block INPUT token streams
  float stats[MAXB][2][TT][2];    // [block][ln1/ln2][token][mean, rstd]
  float dx[TT][SROW];             // token-grad stream (backprops in place)
  float xmid[TT][SROW];           // post-attention residual (pre-LN2)
  float zin[3 * D];               // trunk input = final x[0] | x[1] | x[2]
  float t1[TRUNK], t2[TRUNK], dt1[TRUNK], dt2[TRUNK], dout[MAX_NOUT];
  union {
    struct { float h[TT][SROW], q[TT][SROW], k[TT][SROW], v[TT][SROW];
             float att[TT][TT + 1], datt[TT][TT + 1]; } a;
    struct { float h2[TT][SROW], f1[TT][TRUNK + 1]; } f;  // f1 becomes df1
    struct { float hid[TT][SROW], dh[TT][SROW]; } fr;
  } u;
};

// obs access — straight from global, L2-resident (smem diet)
#define OB(i) __ldg(orow + (i))

// ---- shared forward pieces (identical arithmetic in every pass) -----------

__device__ __forceinline__ void ln_stats(
    const float (&src)[TT][SROW], float (&st)[TT][2], int tid) {
  const int warp = tid >> 5, wl = tid & 31;
  for (int t = warp; t < TT; t += THREADS / 32) {
    const float v0 = src[t][wl], v1 = src[t][wl + 32];
    float sum = v0 + v1;
    #pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(~0u, sum, o);
    const float mean = __shfl_sync(~0u, sum, 0) * (1.f / D);
    const float d0 = v0 - mean, d1 = v1 - mean;
    float sq = d0 * d0 + d1 * d1;
    #pragma unroll
    for (int o = 16; o; o >>= 1) sq += __shfl_down_sync(~0u, sq, o);
    const float rstd = rsqrtf(__shfl_sync(~0u, sq, 0) * (1.f / D) + 1e-5f);
    if (wl == 0) { st[t][0] = mean; st[t][1] = rstd; }
  }
}

// Apply LN from saved stats: thread (half, lane) covers its token range.
__device__ __forceinline__ void ln_apply(
    const float (&src)[TT][SROW], float (&dst)[TT][SROW],
    const float* __restrict__ g, const float* __restrict__ b,
    const float (&st)[TT][2], int t_lo, int t_hi, int lane) {
  const float gg = __ldg(g + lane), bb = __ldg(b + lane);
  for (int t = t_lo; t < t_hi; ++t)
    dst[t][lane] = (src[t][lane] - st[t][0]) * st[t][1] * gg + bb;
}

// h -> q,k,v -> masked scores -> softmax(att); optionally xmid = xin+att@v.
__device__ void attn_fwd(BSmem& s, const float* __restrict__ B, int bi,
                         bool want_xmid, int tid, int t_lo, int t_hi,
                         int lane) {
  {  // qkv
    float aq[10], ak[10], av[10];
    const float bq = __ldg(B + B_BQ + lane);
    const float bk = __ldg(B + B_BK + lane);
    const float bv = __ldg(B + B_BV + lane);
    #pragma unroll
    for (int i = 0; i < 10; ++i) { aq[i] = bq; ak[i] = bk; av[i] = bv; }
    for (int k = 0; k < D; ++k) {
      const float wq = __ldg(B + B_WQ + k * D + lane);
      const float wk = __ldg(B + B_WK + k * D + lane);
      const float wv = __ldg(B + B_WV + k * D + lane);
      // compile-time-indexed accumulators (see policy_forward.cu header):
      // a runtime `t` index would push aq/ak/av into local memory
      #pragma unroll
      for (int i = 0; i < 10; ++i) {
        const int t = t_lo + i;
        if (t < t_hi) {
          const float hv = s.u.a.h[t][k];
          aq[i] += hv * wq;
          ak[i] += hv * wk;
          av[i] += hv * wv;
        }
      }
    }
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
      const int t = t_lo + i;
      if (t < t_hi) {
        s.u.a.q[t][lane] = aq[i];
        s.u.a.k[t][lane] = ak[i];
        s.u.a.v[t][lane] = av[i];
      }
    }
  }
  __syncthreads();
  for (int idx = tid; idx < TT * TT; idx += THREADS) {
    const int t = idx / TT, uu = idx % TT;
    float a = 0.f;
    for (int k = 0; k < D; ++k) a += s.u.a.q[t][k] * s.u.a.k[uu][k];
    s.u.a.att[t][uu] = a * 0.125f + s.madd[uu];
  }
  __syncthreads();
  if (tid < TT) {
    float m = -3.4e38f;
    for (int uu = 0; uu < TT; ++uu) m = fmaxf(m, s.u.a.att[tid][uu]);
    float den = 0.f;
    for (int uu = 0; uu < TT; ++uu) {
      const float e = expf(s.u.a.att[tid][uu] - m);
      s.u.a.att[tid][uu] = e;
      den += e;
    }
    const float inv = 1.f / den;
    for (int uu = 0; uu < TT; ++uu) s.u.a.att[tid][uu] *= inv;
  }
  __syncthreads();
  if (want_xmid) {
    for (int t = t_lo; t < t_hi; ++t) {
      float a = 0.f;
      #pragma unroll
      for (int uu = 0; uu < TT; ++uu)
        a += s.u.a.att[t][uu] * s.u.a.v[uu][lane];
      s.xmid[t][lane] = s.xin[bi][t][lane] + a;
    }
    __syncthreads();
  }
}

// xmid -> h2 = LN2(xmid) -> f1 = relu(h2 Wf1 + b).
__device__ void ffn_fwd(BSmem& s, const float* __restrict__ B, int bi,
                        bool compute_stats, int tid, int t_lo, int t_hi,
                        int lane) {
  if (compute_stats) { ln_stats(s.xmid, s.stats[bi][1], tid); __syncthreads(); }
  ln_apply(s.xmid, s.u.f.h2, B + B_LN2_G, B + B_LN2_B, s.stats[bi][1],
           t_lo, t_hi, lane);
  __syncthreads();
  {
    float acc[TT];
    const float b1 = __ldg(B + B_FFB1 + tid);
    #pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = b1;
    for (int k = 0; k < D; ++k) {
      const float w = __ldg(B + B_FFW1 + k * TRUNK + tid);
      #pragma unroll
      for (int t = 0; t < TT; ++t) acc[t] += s.u.f.h2[t][k] * w;
    }
    #pragma unroll
    for (int t = 0; t < TT; ++t) s.u.f.f1[t][tid] = fmaxf(acc[t], 0.f);
  }
  __syncthreads();
}

// Full block recompute from its saved input (phase A: compute_stats).
__device__ void block_fwd(BSmem& s, const float* __restrict__ B, int bi,
                          bool compute_stats, int tid, int t_lo, int t_hi,
                          int lane) {
  if (compute_stats) {
    ln_stats(s.xin[bi], s.stats[bi][0], tid);
    __syncthreads();
  }
  ln_apply(s.xin[bi], s.u.a.h, B + B_LN1_G, B + B_LN1_B, s.stats[bi][0],
           t_lo, t_hi, lane);
  __syncthreads();
  attn_fwd(s, B, bi, /*want_xmid=*/true, tid, t_lo, t_hi, lane);
  ffn_fwd(s, B, bi, compute_stats, tid, t_lo, t_hi, lane);
}

// FFN layer 2 + residual -> next block's input (or zin for the last).
__device__ void ffn_out(BSmem& s, const float* __restrict__ B, int bi,
                        int n_blocks, int half, int t_lo, int t_hi,
                        int lane) {
  if (bi + 1 < n_blocks) {
    float acc[10];
    const float b2 = __ldg(B + B_FFB2 + lane);
    #pragma unroll
    for (int i = 0; i < 10; ++i) acc[i] = b2;
    for (int k = 0; k < TRUNK; ++k) {
      const float w = __ldg(B + B_FFW2 + k * D + lane);
      #pragma unroll
      for (int i = 0; i < 10; ++i) {
        const int t = t_lo + i;
        if (t < t_hi) acc[i] += s.u.f.f1[t][k] * w;
      }
    }
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
      const int t = t_lo + i;
      if (t < t_hi) s.xin[bi + 1][t][lane] = s.xmid[t][lane] + acc[i];
    }
  } else if (half == 0) {  // only the 3 readout tokens feed the trunk
    float acc[3];
    const float b2 = __ldg(B + B_FFB2 + lane);
    acc[0] = b2; acc[1] = b2; acc[2] = b2;
    for (int k = 0; k < TRUNK; ++k) {
      const float w = __ldg(B + B_FFW2 + k * D + lane);
      acc[0] += s.u.f.f1[0][k] * w;
      acc[1] += s.u.f.f1[1][k] * w;
      acc[2] += s.u.f.f1[2][k] * w;
    }
    s.zin[lane] = s.xmid[0][lane] + acc[0];
    s.zin[D + lane] = s.xmid[1][lane] + acc[1];
    s.zin[2 * D + lane] = s.xmid[2][lane] + acc[2];
  }
  __syncthreads();
}

// Front note/saber/body MLPs (same as forward kernel) -> xin[0].
__device__ void front_fwd(BSmem& s, const float* __restrict__ W,
                          const float* __restrict__ orow, int half,
                          int lane) {
  {  // note layer 1 (14 -> 64, ReLU)
    float acc[8];
    const float b1 = __ldg(W + O_NOTE_B1 + lane);
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = b1;
    for (int k = 0; k < NOTE_FEAT; ++k) {
      const float w = __ldg(W + O_NOTE_W1 + k * D + lane);
      #pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[i] += OB(OFF_NOTES + (half * 8 + i) * NOTE_FEAT + k) * w;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i)
      s.u.fr.hid[3 + half * 8 + i][lane] = fmaxf(acc[i], 0.f);
  }
  {  // saber layer 1 (27 -> 64, ReLU); token t == half
    const int* idx = half == 0 ? SABER_L_IDX : SABER_R_IDX;
    float a = __ldg(W + O_SAB_B1 + lane);
    for (int k = 0; k < SABER_FEAT; ++k)
      a += OB(idx[k]) * __ldg(W + O_SAB_W1 + k * D + lane);
    a += __ldg(W + O_SAB_W1 + (SABER_FEAT + half) * D + lane);
    s.u.fr.hid[half][lane] = fmaxf(a, 0.f);
  }
  if (half == 0) {  // body layer 1 (50 -> 64, ReLU); token 2
    float a = __ldg(W + O_BODY_B1 + lane);
    for (int k = 0; k < BODY_FEAT; ++k)
      a += OB(BODY_IDX[k]) * __ldg(W + O_BODY_W1 + k * D + lane);
    s.u.fr.hid[2][lane] = fmaxf(a, 0.f);
  }
  __syncthreads();
  {  // note layer 2 (64 -> 64, linear)
    float acc[8];
    const float b2 = __ldg(W + O_NOTE_B2 + lane);
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = b2;
    for (int k = 0; k < D; ++k) {
      const float w = __ldg(W + O_NOTE_W2 + k * D + lane);
      #pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[i] += s.u.fr.hid[3 + half * 8 + i][k] * w;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) s.xin[0][3 + half * 8 + i][lane] = acc[i];
  }
  {  // saber layer 2
    float a = __ldg(W + O_SAB_B2 + lane);
    for (int k = 0; k < D; ++k)
      a += s.u.fr.hid[half][k] * __ldg(W + O_SAB_W2 + k * D + lane);
    s.xin[0][half][lane] = a;
  }
  if (half == 0) {  // body layer 2
    float a = __ldg(W + O_BODY_B2 + lane);
    for (int k = 0; k < D; ++k)
      a += s.u.fr.hid[2][k] * __ldg(W + O_BODY_W2 + k * D + lane);
    s.xin[0][2][lane] = a;
  }
  __syncthreads();
}

// ---- backward pieces ------------------------------------------------------

__device__ __forceinline__ void ln_bwd_param(
    const float (&dh)[TT][SROW], const float (&x)[TT][SROW],
    const float (&st)[TT][2], float* Gg, float* Gb, int t_lo, int t_hi,
    int lane) {
  float dg = 0.f, db = 0.f;
  for (int t = t_lo; t < t_hi; ++t) {
    const float xh = (x[t][lane] - st[t][0]) * st[t][1];
    dg += dh[t][lane] * xh;
    db += dh[t][lane];
  }
  atomicAdd(Gg + lane, dg);
  atomicAdd(Gb + lane, db);
}

__device__ void ln_bwd_data(
    const float (&dh)[TT][SROW], const float (&x)[TT][SROW],
    const float* __restrict__ g, const float (&st)[TT][2],
    float (&dxout)[TT][SROW], int tid) {
  const int warp = tid >> 5, wl = tid & 31;
  for (int t = warp; t < TT; t += THREADS / 32) {
    const float mean = st[t][0], r = st[t][1];
    const float xh0 = (x[t][wl] - mean) * r;
    const float xh1 = (x[t][wl + 32] - mean) * r;
    const float dxh0 = dh[t][wl] * __ldg(g + wl);
    const float dxh1 = dh[t][wl + 32] * __ldg(g + wl + 32);
    float s1 = dxh0 + dxh1;
    #pragma unroll
    for (int o = 16; o; o >>= 1) s1 += __shfl_down_sync(~0u, s1, o);
    s1 = __shfl_sync(~0u, s1, 0);
    float s2 = dxh0 * xh0 + dxh1 * xh1;
    #pragma unroll
    for (int o = 16; o; o >>= 1) s2 += __shfl_down_sync(~0u, s2, o);
    s2 = __shfl_sync(~0u, s2, 0);
    dxout[t][wl] += r * (dxh0 - (s1 + xh0 * s2) * (1.f / D));
    dxout[t][wl + 32] += r * (dxh1 - (s1 + xh1 * s2) * (1.f / D));
  }
}

// Heads + trunk backward: dout -> head/trunk grads and dx[0..2];
// zeroes dx[3..18].
__device__ void head_trunk_bwd(BSmem& s, const float* __restrict__ TW,
                               const float* __restrict__ HW, float* Gt,
                               float* Gh, int nout, int tid, int half,
                               int lane) {
  for (int idx = tid; idx < TRUNK * nout; idx += THREADS)
    atomicAdd(Gh + idx, s.t2[idx / nout] * s.dout[idx % nout]);
  for (int o = tid; o < nout; o += THREADS)
    atomicAdd(Gh + TRUNK * nout + o, s.dout[o]);
  {
    float a = 0.f;
    for (int j = 0; j < nout; ++j)
      a += __ldg(HW + tid * nout + j) * s.dout[j];
    s.dt2[tid] = a;
  }
  __syncthreads();
  {  // trunk layer 2 (linear)
    const float d = s.dt2[tid];
    for (int m = 0; m < TRUNK; ++m)
      atomicAdd(Gt + T_W2 + m * TRUNK + tid, s.t1[m] * d);
    atomicAdd(Gt + T_B2 + tid, d);
    float a = 0.f;
    for (int j = 0; j < TRUNK; ++j)
      a += __ldg(TW + T_W2 + tid * TRUNK + j) * s.dt2[j];
    s.dt1[tid] = (s.t1[tid] > 0.f) ? a : 0.f;
  }
  __syncthreads();
  {  // trunk layer 1 (ReLU) -> dz -> readout-token grads
    const float d = s.dt1[tid];
    for (int m = 0; m < 3 * D; ++m)
      atomicAdd(Gt + T_W1 + m * TRUNK + tid, s.zin[m] * d);
    atomicAdd(Gt + T_B1 + tid, d);
  }
  __syncthreads();
  for (int z = tid; z < 3 * D; z += THREADS) {
    float a = 0.f;
    for (int j = 0; j < TRUNK; ++j)
      a += __ldg(TW + T_W1 + z * TRUNK + j) * s.dt1[j];
    s.dx[z / D][z % D] = a;
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) s.dx[3 + half * 8 + i][lane] = 0.f;
  __syncthreads();
}

// FFN backward for one block. In: dx = d(block output), resident
// xmid/h2/f1. Out: dx = d(xmid); FFN + LN2 grads accumulated. f1 is
// OVERWRITTEN with df1 in place (smem diet).
__device__ void ffn_bwd(BSmem& s, const float* __restrict__ B, float* G,
                        int bi, int tid, int t_lo, int t_hi, int half,
                        int lane) {
  {  // dWf2 (128 x 64) + dbf2 — must finish before f1 -> df1 overwrite
    for (int m = 0; m < 64; ++m) {
      const int k = half * 64 + m;
      float acc = 0.f;
      #pragma unroll
      for (int t = 0; t < TT; ++t) acc += s.u.f.f1[t][k] * s.dx[t][lane];
      atomicAdd(G + B_FFW2 + k * D + lane, acc);
    }
    float ab = 0.f;
    for (int t = t_lo; t < t_hi; ++t) ab += s.dx[t][lane];
    atomicAdd(G + B_FFB2 + lane, ab);
  }
  __syncthreads();
  {  // df1 IN PLACE over f1: thread owns hidden column tid; the ReLU
     // mask reads f1[t][tid] before writing it (same thread, no race)
    float acc[TT];
    #pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.f;
    for (int j = 0; j < D; ++j) {
      const float w = __ldg(B + B_FFW2 + tid * D + j);
      #pragma unroll
      for (int t = 0; t < TT; ++t) acc[t] += w * s.dx[t][j];
    }
    #pragma unroll
    for (int t = 0; t < TT; ++t)
      s.u.f.f1[t][tid] = (s.u.f.f1[t][tid] > 0.f) ? acc[t] : 0.f;
  }
  __syncthreads();
  {  // dWf1 (64 x 128) + dbf1: thread owns column tid; f1 now holds df1
    for (int i = 0; i < D; ++i) {
      float acc = 0.f;
      #pragma unroll
      for (int t = 0; t < TT; ++t) acc += s.u.f.h2[t][i] * s.u.f.f1[t][tid];
      atomicAdd(G + B_FFW1 + i * TRUNK + tid, acc);
    }
    float ab = 0.f;
    #pragma unroll
    for (int t = 0; t < TT; ++t) ab += s.u.f.f1[t][tid];
    atomicAdd(G + B_FFB1 + tid, ab);
  }
  __syncthreads();
  {  // dh2 (overwrites h2 — its readers are done)
    float accs[10];
    #pragma unroll
    for (int i = 0; i < 10; ++i) accs[i] = 0.f;
    for (int c = 0; c < TRUNK; ++c) {
      const float w = __ldg(B + B_FFW1 + lane * TRUNK + c);
      #pragma unroll
      for (int i = 0; i < 10; ++i) {
        const int t = t_lo + i;
        if (t < t_hi) accs[i] += w * s.u.f.f1[t][c];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
      const int t = t_lo + i;
      if (t < t_hi) s.u.f.h2[t][lane] = accs[i];
    }
  }
  __syncthreads();
  ln_bwd_param(s.u.f.h2, s.xmid, s.stats[bi][1], G + B_LN2_G, G + B_LN2_B,
               t_lo, t_hi, lane);
  ln_bwd_data(s.u.f.h2, s.xmid, B + B_LN2_G, s.stats[bi][1], s.dx, tid);
  __syncthreads();
}

// Attention backward for one block. In: dx = d(xmid), resident
// h/q/k/v/att (re-materialized). Out: dx = d(block input).
__device__ void attn_bwd(BSmem& s, const float* __restrict__ B, float* G,
                         int bi, int tid, int t_lo, int t_hi, int half,
                         int lane) {
  // dP = dx @ v^T (masked keys: att column == 0 -> grads vanish exactly)
  for (int idx = tid; idx < TT * TT; idx += THREADS) {
    const int t = idx / TT, uu = idx % TT;
    float a = 0.f;
    for (int j = 0; j < D; ++j) a += s.dx[t][j] * s.u.a.v[uu][j];
    s.u.a.datt[t][uu] = a;
  }
  __syncthreads();
  if (tid < TT) {  // softmax backward per query row
    float rs = 0.f;
    for (int uu = 0; uu < TT; ++uu)
      rs += s.u.a.att[tid][uu] * s.u.a.datt[tid][uu];
    for (int uu = 0; uu < TT; ++uu)
      s.u.a.datt[tid][uu] = s.u.a.att[tid][uu] * (s.u.a.datt[tid][uu] - rs);
  }
  __syncthreads();
  // dq, dk, dv into registers, then overwrite q, k, v
  float rdq[10], rdk[10], rdv[10];
  #pragma unroll
  for (int i = 0; i < 10; ++i) {
    const int t = t_lo + i;
    if (t < t_hi) {
      float aq = 0.f, ak = 0.f, av = 0.f;
      #pragma unroll
      for (int uu = 0; uu < TT; ++uu) {
        aq += s.u.a.datt[t][uu] * s.u.a.k[uu][lane];
        ak += s.u.a.datt[uu][t] * s.u.a.q[uu][lane];
        av += s.u.a.att[uu][t] * s.dx[uu][lane];
      }
      rdq[i] = aq * 0.125f;
      rdk[i] = ak * 0.125f;
      rdv[i] = av;
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 10; ++i) {
    const int t = t_lo + i;
    if (t < t_hi) {
      s.u.a.q[t][lane] = rdq[i];
      s.u.a.k[t][lane] = rdk[i];
      s.u.a.v[t][lane] = rdv[i];
    }
  }
  __syncthreads();
  {  // dWq/dWk/dWv (64 x 64 each) + biases
    for (int m = 0; m < 32; ++m) {
      const int i = half * 32 + m;
      float aq = 0.f, ak = 0.f, av = 0.f;
      #pragma unroll
      for (int t = 0; t < TT; ++t) {
        const float hv = s.u.a.h[t][i];
        aq += hv * s.u.a.q[t][lane];
        ak += hv * s.u.a.k[t][lane];
        av += hv * s.u.a.v[t][lane];
      }
      atomicAdd(G + B_WQ + i * D + lane, aq);
      atomicAdd(G + B_WK + i * D + lane, ak);
      atomicAdd(G + B_WV + i * D + lane, av);
    }
    float bq = 0.f, bk = 0.f, bv = 0.f;
    for (int t = t_lo; t < t_hi; ++t) {
      bq += s.u.a.q[t][lane];
      bk += s.u.a.k[t][lane];
      bv += s.u.a.v[t][lane];
    }
    atomicAdd(G + B_BQ + lane, bq);
    atomicAdd(G + B_BK + lane, bk);
    atomicAdd(G + B_BV + lane, bv);
  }
  __syncthreads();
  {  // dh1 = dq Wq^T + dk Wk^T + dv Wv^T (overwrites h — readers done)
    float accs[10];
    #pragma unroll
    for (int i = 0; i < 10; ++i) accs[i] = 0.f;
    for (int j = 0; j < D; ++j) {
      const float wq = __ldg(B + B_WQ + lane * D + j);
      const float wk = __ldg(B + B_WK + lane * D + j);
      const float wv = __ldg(B + B_WV + lane * D + j);
      #pragma unroll
      for (int i = 0; i < 10; ++i) {
        const int t = t_lo + i;
        if (t < t_hi)
          accs[i] += wq * s.u.a.q[t][j] + wk * s.u.a.k[t][j]
                     + wv * s.u.a.v[t][j];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
      const int t = t_lo + i;
      if (t < t_hi) s.u.a.h[t][lane] = accs[i];
    }
  }
  __syncthreads();
  ln_bwd_param(s.u.a.h, s.xin[bi], s.stats[bi][0], G + B_LN1_G, G + B_LN1_B,
               t_lo, t_hi, lane);
  ln_bwd_data(s.u.a.h, s.xin[bi], B + B_LN1_G, s.stats[bi][0], s.dx, tid);
  __syncthreads();
}

// Front MLP backward (hiddens recomputed from obs). In: dx = d(xin[0]).
__device__ void front_bwd(BSmem& s, const float* __restrict__ W,
                          const float* __restrict__ orow, float* G,
                          int tid, int half, int lane) {
  {  // recompute post-ReLU hiddens (bitwise same as front_fwd layer 1)
    float acc[8];
    const float b1 = __ldg(W + O_NOTE_B1 + lane);
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = b1;
    for (int k = 0; k < NOTE_FEAT; ++k) {
      const float w = __ldg(W + O_NOTE_W1 + k * D + lane);
      #pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[i] += OB(OFF_NOTES + (half * 8 + i) * NOTE_FEAT + k) * w;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i)
      s.u.fr.hid[3 + half * 8 + i][lane] = fmaxf(acc[i], 0.f);
    const int* idx = half == 0 ? SABER_L_IDX : SABER_R_IDX;
    float a = __ldg(W + O_SAB_B1 + lane);
    for (int k = 0; k < SABER_FEAT; ++k)
      a += OB(idx[k]) * __ldg(W + O_SAB_W1 + k * D + lane);
    a += __ldg(W + O_SAB_W1 + (SABER_FEAT + half) * D + lane);
    s.u.fr.hid[half][lane] = fmaxf(a, 0.f);
    if (half == 0) {
      float ab = __ldg(W + O_BODY_B1 + lane);
      for (int k = 0; k < BODY_FEAT; ++k)
        ab += OB(BODY_IDX[k]) * __ldg(W + O_BODY_W1 + k * D + lane);
      s.u.fr.hid[2][lane] = fmaxf(ab, 0.f);
    }
  }
  __syncthreads();
  {  // dWn2 (64 x 64) over the 16 note tokens + dbn2
    for (int m = 0; m < 32; ++m) {
      const int i = half * 32 + m;
      float acc = 0.f;
      #pragma unroll
      for (int st = 0; st < N_NOTE_SLOTS; ++st)
        acc += s.u.fr.hid[3 + st][i] * s.dx[3 + st][lane];
      atomicAdd(G + O_NOTE_W2 + i * D + lane, acc);
    }
    float ab = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) ab += s.dx[3 + half * 8 + i][lane];
    atomicAdd(G + O_NOTE_B2 + lane, ab);
  }
  {  // dWs2 over the 2 saber tokens + dbs2; dWb2 over the body token
    for (int m = 0; m < 32; ++m) {
      const int i = half * 32 + m;
      atomicAdd(G + O_SAB_W2 + i * D + lane,
                s.u.fr.hid[0][i] * s.dx[0][lane]
                + s.u.fr.hid[1][i] * s.dx[1][lane]);
      atomicAdd(G + O_BODY_W2 + i * D + lane,
                s.u.fr.hid[2][i] * s.dx[2][lane]);
    }
    if (half == 0) {
      atomicAdd(G + O_SAB_B2 + lane, s.dx[0][lane] + s.dx[1][lane]);
      atomicAdd(G + O_BODY_B2 + lane, s.dx[2][lane]);
    }
  }
  {  // dn1 / ds1 / db1 (ReLU-masked) -> dh
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    float acs = 0.f, acb = 0.f;
    for (int j = 0; j < D; ++j) {
      const float wn = __ldg(W + O_NOTE_W2 + lane * D + j);
      #pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] += wn * s.dx[3 + half * 8 + i][j];
      acs += __ldg(W + O_SAB_W2 + lane * D + j) * s.dx[half][j];
      acb += __ldg(W + O_BODY_W2 + lane * D + j) * s.dx[2][j];
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int t = 3 + half * 8 + i;
      s.u.fr.dh[t][lane] = (s.u.fr.hid[t][lane] > 0.f) ? acc[i] : 0.f;
    }
    s.u.fr.dh[half][lane] = (s.u.fr.hid[half][lane] > 0.f) ? acs : 0.f;
    if (half == 0)
      s.u.fr.dh[2][lane] = (s.u.fr.hid[2][lane] > 0.f) ? acb : 0.f;
  }
  __syncthreads();
  // dWn1 (14 x 64) + dbn1
  for (int idx = tid; idx < NOTE_FEAT * D; idx += THREADS) {
    const int f = idx / D, c = idx % D;
    float acc = 0.f;
    #pragma unroll
    for (int st = 0; st < N_NOTE_SLOTS; ++st)
      acc += OB(OFF_NOTES + st * NOTE_FEAT + f) * s.u.fr.dh[3 + st][c];
    atomicAdd(G + O_NOTE_W1 + idx, acc);
  }
  {
    float ab = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) ab += s.u.fr.dh[3 + half * 8 + i][lane];
    atomicAdd(G + O_NOTE_B1 + lane, ab);
  }
  // dWs1 (27 x 64: 25 gathered features + hand one-hot) + dbs1
  for (int idx = tid; idx < SABER_IN * D; idx += THREADS) {
    const int f = idx / D, c = idx % D;
    const float in0 = (f < SABER_FEAT) ? OB(SABER_L_IDX[f])
                                       : (f == SABER_FEAT ? 1.f : 0.f);
    const float in1 = (f < SABER_FEAT) ? OB(SABER_R_IDX[f])
                                       : (f == SABER_FEAT + 1 ? 1.f : 0.f);
    atomicAdd(G + O_SAB_W1 + idx,
              in0 * s.u.fr.dh[0][c] + in1 * s.u.fr.dh[1][c]);
  }
  if (half == 0)
    atomicAdd(G + O_SAB_B1 + lane, s.u.fr.dh[0][lane] + s.u.fr.dh[1][lane]);
  // dWb1 (50 x 64, gathered) + dbb1
  for (int idx = tid; idx < BODY_FEAT * D; idx += THREADS) {
    const int f = idx / D, c = idx % D;
    atomicAdd(G + O_BODY_W1 + idx, OB(BODY_IDX[f]) * s.u.fr.dh[2][c]);
  }
  if (half == 0)
    atomicAdd(G + O_BODY_B1 + lane, s.u.fr.dh[2][lane]);
}

// ---- kernels --------------------------------------------------------------

__global__ void __launch_bounds__(THREADS)
policy_bwd_kernel(const float* __restrict__ W, const float* __restrict__ obs,
                  const float* __restrict__ gl, const float* __restrict__ gv,
                  float* __restrict__ partial, int n_blocks, int n_gears,
                  int ncopies, int N) {
  __shared__ BSmem s;
  const int n = blockIdx.x;
  if (n >= N) return;
  const int tid = threadIdx.x;
  const int lane = tid & (D - 1);
  const int half = tid >> 6;
  const int t_lo = half * 10;
  const int t_hi = half == 0 ? 10 : TT;
  const int nout = N_HEADS * n_gears + 1;
  const int nlogits = nout - 1;
  const size_t NP = expected_numel(n_blocks, n_gears);
  float* G = partial + (size_t)(n % ncopies) * NP;
  const float* orow = obs + (size_t)n * OBS_DIM;

  if (tid < TT)
    s.madd[tid] =
        (tid < 3) ? 0.f : (OB(OFF_NOTE_MASK + tid - 3) - 1.f) * 16384.f;
  __syncthreads();

  // ---- phase A: forward recompute, saving block inputs + LN stats --------
  front_fwd(s, W, orow, half, lane);
  for (int bi = 0; bi < n_blocks; ++bi) {
    const float* B = W + FRONT + (size_t)bi * BLK;
    block_fwd(s, B, bi, /*compute_stats=*/true, tid, t_lo, t_hi, lane);
    ffn_out(s, B, bi, n_blocks, half, t_lo, t_hi, lane);
  }
  const float* TW = W + FRONT + (size_t)n_blocks * BLK;
  const float* HW = TW + TRUNK_SZ;
  {  // trunk forward (t1 ReLU, t2 linear)
    float a = __ldg(TW + T_B1 + tid);
    for (int k = 0; k < 3 * D; ++k)
      a += s.zin[k] * __ldg(TW + T_W1 + k * TRUNK + tid);
    s.t1[tid] = fmaxf(a, 0.f);
  }
  __syncthreads();
  {
    float a = __ldg(TW + T_B2 + tid);
    for (int k = 0; k < TRUNK; ++k)
      a += s.t1[k] * __ldg(TW + T_W2 + k * TRUNK + tid);
    s.t2[tid] = a;
  }
  // stage upstream grads
  for (int o = tid; o < nout; o += THREADS)
    s.dout[o] = o < nlogits ? gl[(size_t)n * nlogits + o] : gv[n];
  __syncthreads();

  // ---- phase B: heads + trunk backward -----------------------------------
  float* Gt = G + FRONT + (size_t)n_blocks * BLK;
  head_trunk_bwd(s, TW, HW, Gt, Gt + TRUNK_SZ, nout, tid, half, lane);

  // ---- phase C: transformer blocks in reverse ----------------------------
  for (int bi = n_blocks - 1; bi >= 0; --bi) {
    const float* B = W + FRONT + (size_t)bi * BLK;
    float* GB = G + FRONT + (size_t)bi * BLK;
    if (bi != n_blocks - 1) {
      // re-materialize xmid/h2/f1 (the last block's are still resident)
      ln_apply(s.xin[bi], s.u.a.h, B + B_LN1_G, B + B_LN1_B, s.stats[bi][0],
               t_lo, t_hi, lane);
      __syncthreads();
      attn_fwd(s, B, bi, /*want_xmid=*/true, tid, t_lo, t_hi, lane);
      ffn_fwd(s, B, bi, /*compute_stats=*/false, tid, t_lo, t_hi, lane);
    }
    ffn_bwd(s, B, GB, bi, tid, t_lo, t_hi, half, lane);
    // re-materialize h/q/k/v/att for the attention half
    ln_apply(s.xin[bi], s.u.a.h, B + B_LN1_G, B + B_LN1_B, s.stats[bi][0],
             t_lo, t_hi, lane);
    __syncthreads();
    attn_fwd(s, B, bi, /*want_xmid=*/false, tid, t_lo, t_hi, lane);
    attn_bwd(s, B, GB, bi, tid, t_lo, t_hi, half, lane);
  }

  // ---- phase D: front MLPs -----------------------------------------------
  front_bwd(s, W, orow, G, tid, half, lane);
}

__global__ void reduce_partials_kernel(const float* __restrict__ partial,
                                       float* __restrict__ out, int np,
                                       int ncopies) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= np) return;
  float a = 0.f;
  for (int c = 0; c < ncopies; ++c) a += partial[(size_t)c * np + i];
  out[i] = a;
}

void backward(at::Tensor weights, at::Tensor obs, at::Tensor grad_logits,
              at::Tensor grad_value, at::Tensor partial, at::Tensor grad_out,
              int64_t n_blocks, int64_t n_gears) {
  TORCH_CHECK(n_blocks >= 1 && n_blocks <= MAXB,
              "fused backward supports 1..", MAXB, " blocks");
  TORCH_CHECK(n_gears == 3 || n_gears == 5 || n_gears == 7,
              "n_gears must be 3, 5, or 7");
  const int64_t NP = expected_numel((int)n_blocks, (int)n_gears);
  const int64_t NLOGITS = N_HEADS * n_gears;
  TORCH_CHECK(obs.is_cuda() && obs.scalar_type() == at::kFloat
              && obs.is_contiguous() && obs.dim() == 2
              && obs.size(1) == OBS_DIM,
              "obs must be a contiguous CUDA f32 (N, ", OBS_DIM, ") tensor");
  const int64_t N = obs.size(0);
  TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == at::kFloat
              && weights.is_contiguous() && weights.numel() == NP,
              "packed weights must be contiguous CUDA f32 with ", NP,
              " elements");
  TORCH_CHECK(grad_logits.is_cuda()
              && grad_logits.scalar_type() == at::kFloat
              && grad_logits.is_contiguous()
              && grad_logits.numel() == N * NLOGITS,
              "grad_logits must be contiguous CUDA f32 with N*", NLOGITS,
              " elements");
  TORCH_CHECK(grad_value.is_cuda() && grad_value.scalar_type() == at::kFloat
              && grad_value.is_contiguous() && grad_value.numel() == N,
              "grad_value must be contiguous CUDA f32 with N elements");
  TORCH_CHECK(partial.is_cuda() && partial.scalar_type() == at::kFloat
              && partial.is_contiguous() && partial.numel() % NP == 0
              && partial.numel() >= NP,
              "partial workspace must hold k>=1 copies of the ", NP,
              "-element packed grad (caller zeroes it)");
  TORCH_CHECK(grad_out.is_cuda() && grad_out.scalar_type() == at::kFloat
              && grad_out.is_contiguous() && grad_out.numel() == NP);
  const int ncopies = (int)(partial.numel() / NP);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (N > 0) {
    policy_bwd_kernel<<<(unsigned)N, THREADS, 0, stream>>>(
        weights.data_ptr<float>(), obs.data_ptr<float>(),
        grad_logits.data_ptr<float>(), grad_value.data_ptr<float>(),
        partial.data_ptr<float>(), (int)n_blocks, (int)n_gears, ncopies,
        (int)N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  reduce_partials_kernel<<<(unsigned)((NP + 255) / 256), 256, 0, stream>>>(
      partial.data_ptr<float>(), grad_out.data_ptr<float>(), (int)NP,
      ncopies);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backward", &backward,
        "Fused NotePolicy backward (env v10): recomputes the forward per "
        "sample and accumulates ALL parameter grads (packed layout) from "
        "grad_logits (N, 29*K) + grad_value (N). partial = zeroed "
        "(ncopies*NP) workspace; grad_out = (NP) packed grad.");
}
