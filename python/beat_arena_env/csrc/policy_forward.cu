// Fused NotePolicy forward (inference only) as ONE CUDA kernel launch —
// env v10 layout (P5): obs 340, 19 tokens [saber_L, saber_R, body,
// note_0..15], body token (50 gathered features), trunk over the
// 3-token readout (192 -> 128), 29 action heads x K logits (K = n_gears
// in {3, 5, 7}, runtime) + value head, per-head argmax.
//
// The whole policy.py forward — note/saber/body MLPs, n_blocks pre-LN
// transformer blocks (single-head QKV attention with key masking, FFN,
// residuals), trunk, fused head GEMM — runs in a single launch: one
// thread block (128 threads) per sample, activations in shared memory,
// weights streamed from a packed flat tensor (policy_cuda._pack — the
// offsets below are a contract with it).
//
// v10 restructure notes vs the v8 kernel:
// - Saber/body token features are GATHERED from the obs row via constant
//   index tables (the v10 obs interleaves q/qd/derived blocks; policy.py
//   _SABER_L_IDX/_SABER_R_IDX/_BODY_IDX verbatim).
// - 19 tokens split across the two thread halves as 10 + 9, kept as
//   COMPILE-TIME 10-wide unrolls with an `if (t < t_hi)` guard (half 1
//   runs 9 live lanes). The guard, not a runtime trip count: a loop
//   bounded by the runtime t_lo/t_hi indexes the per-token accumulator
//   arrays with a runtime index, which nvcc cannot register-allocate —
//   they land in local memory (measured: 200 B stack frame, 860 LDL /
//   690 STL, fused forward 0.26-0.47x eager). With `i` a compile-time
//   induction variable the accumulators live in registers, as in v8.
// - Last-block pruning now keeps 3 query tokens (sabers + body) — the
//   exact-pruning argument is unchanged: the key-dim mask means dropped
//   rows were never read again.
// - Head count/logit width are runtime (29 x K + 1 as one GEMM).
//
// Junk inertness: masked note keys score -16384, whose softmax weight
// underflows to exactly 0.0 — outputs are bitwise independent of junk
// in masked slots (pinned by tests/test_policy_cuda.py).

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
constexpr int SABER_IN = SABER_FEAT + 2;              // + hand one-hot
constexpr int BODY_FEAT = 54;  // env v12: + the 4 ground-contact lanes
constexpr int OFF_NOTES = 100;
constexpr int OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT;  // 324
constexpr int OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS + 4; // 344 (env v12)
constexpr int N_HEADS = 29;     // action heads, K logits each (runtime)
constexpr int MAX_K = 7;
constexpr int MAX_NOUT = N_HEADS * MAX_K + 1;         // 204
constexpr int THREADS = 128;
constexpr int SROW = D + 1;     // padded token-row stride

// token-feature gather tables (policy.py _SABER_*_IDX / _BODY_IDX)
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

// --- packed-weight offsets (contract with policy_cuda._pack) ---------------
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
// per transformer block (relative to its base)
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
// after the blocks — trunk input = 3 * D (saber_L | saber_R | body)
constexpr int T_W1 = 0;
constexpr int T_B1 = T_W1 + 3 * D * TRUNK;
constexpr int T_W2 = T_B1 + TRUNK;
constexpr int T_B2 = T_W2 + TRUNK * TRUNK;
constexpr int TRUNK_SZ = T_B2 + TRUNK;                // 41216
constexpr int H_W = 0;                                // TRUNK x NOUT
static_assert(FRONT == 18752 && BLK == 29312 && TRUNK_SZ == 41216,
              "packed layout drifted");

constexpr int expected_numel(int n_blocks, int n_gears) {
  return FRONT + n_blocks * BLK + TRUNK_SZ
         + (TRUNK + 1) * (N_HEADS * n_gears + 1);
}

struct Smem {
  float x[TT][SROW];     // residual stream
  float h[TT][SROW];     // post-LN scratch; q lives here between qkv and LN2
  float obs[OBS_DIM];
  float madd[TT];        // (key_mask - 1) * 16384 additive score mask
  union {
    struct { float k[TT][SROW], v[TT][SROW];
             float att[TT][TT + 1]; } a;
    struct { float f1[TT][TRUNK + 1]; } f;
    struct { float t1[TRUNK], t2[TRUNK], out[MAX_NOUT]; } t;
  } u;
};

// LayerNorm over each token row of src into dst: one warp per token,
// shuffle reductions, biased variance, eps 1e-5 (torch defaults).
__device__ __forceinline__ void ln_tokens(
    const float (&src)[TT][SROW], float (&dst)[TT][SROW],
    const float* __restrict__ g, const float* __restrict__ b, int tid) {
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
    dst[t][wl] = d0 * rstd * __ldg(g + wl) + __ldg(b + wl);
    dst[t][wl + 32] = d1 * rstd * __ldg(g + wl + 32) + __ldg(b + wl + 32);
  }
}

__global__ void __launch_bounds__(THREADS, 4)
policy_fwd_kernel(const float* __restrict__ W, const float* __restrict__ obs,
                  float* __restrict__ logits, float* __restrict__ value,
                  int64_t* __restrict__ actions, int n_blocks, int n_gears,
                  int N) {
  __shared__ Smem s;
  const int n = blockIdx.x;
  if (n >= N) return;
  const int tid = threadIdx.x;
  const int lane = tid & (D - 1);   // output index for 64-wide layers
  const int half = tid >> 6;        // token-half selector (0 or 1)
  // token ranges per half: half 0 -> [0, 10), half 1 -> [10, 19)
  const int t_lo = half * 10;
  const int t_hi = half == 0 ? 10 : TT;
  const int NOUT = N_HEADS * n_gears + 1;

  // ---- stage obs row; additive key mask -----------------------------------
  const float* orow = obs + (size_t)n * OBS_DIM;
  for (int i = tid; i < OBS_DIM; i += THREADS) s.obs[i] = orow[i];
  __syncthreads();
  if (tid < TT)
    s.madd[tid] =
        (tid < 3) ? 0.f : (s.obs[OFF_NOTE_MASK + tid - 3] - 1.f) * 16384.f;

  // ---- front MLPs: 2 saber + 1 body + 16 note tokens -> x ------------------
  {  // note layer 1 (14 -> 64, ReLU): each half owns 8 note slots
    float acc[8];
    const float b1 = __ldg(W + O_NOTE_B1 + lane);
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = b1;
    for (int k = 0; k < NOTE_FEAT; ++k) {
      const float w = __ldg(W + O_NOTE_W1 + k * D + lane);
      #pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[i] += s.obs[OFF_NOTES + (half * 8 + i) * NOTE_FEAT + k] * w;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i)
      s.h[3 + half * 8 + i][lane] = fmaxf(acc[i], 0.f);
  }
  {  // saber layer 1 (27 -> 64, ReLU): token t == half; 25 gathered
     // features + hand one-hot ([1,0] left, [0,1] right)
    const int* idx = half == 0 ? SABER_L_IDX : SABER_R_IDX;
    float a = __ldg(W + O_SAB_B1 + lane);
    for (int k = 0; k < SABER_FEAT; ++k)
      a += s.obs[idx[k]] * __ldg(W + O_SAB_W1 + k * D + lane);
    a += __ldg(W + O_SAB_W1 + (SABER_FEAT + half) * D + lane);  // one-hot
    s.h[half][lane] = fmaxf(a, 0.f);
  }
  if (half == 0) {  // body layer 1 (50 -> 64, ReLU): token 2
    float a = __ldg(W + O_BODY_B1 + lane);
    for (int k = 0; k < BODY_FEAT; ++k)
      a += s.obs[BODY_IDX[k]] * __ldg(W + O_BODY_W1 + k * D + lane);
    s.h[2][lane] = fmaxf(a, 0.f);
  }
  __syncthreads();
  {  // note layer 2 (64 -> 64, linear): each half owns 8 note tokens
    float acc[8];
    const float b2 = __ldg(W + O_NOTE_B2 + lane);
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = b2;
    for (int k = 0; k < D; ++k) {
      const float w = __ldg(W + O_NOTE_W2 + k * D + lane);
      #pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] += s.h[3 + half * 8 + i][k] * w;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) s.x[3 + half * 8 + i][lane] = acc[i];
  }
  {  // saber layer 2
    float a = __ldg(W + O_SAB_B2 + lane);
    for (int k = 0; k < D; ++k)
      a += s.h[half][k] * __ldg(W + O_SAB_W2 + k * D + lane);
    s.x[half][lane] = a;
  }
  if (half == 0) {  // body layer 2
    float a = __ldg(W + O_BODY_B2 + lane);
    for (int k = 0; k < D; ++k)
      a += s.h[2][k] * __ldg(W + O_BODY_W2 + k * D + lane);
    s.x[2][lane] = a;
  }
  __syncthreads();

  // ---- transformer blocks (all but the last full-width) -------------------
  for (int bi = 0; bi + 1 < n_blocks; ++bi) {
    const float* B = W + FRONT + (size_t)bi * BLK;
    ln_tokens(s.x, s.h, B + B_LN1_G, B + B_LN1_B, tid);
    __syncthreads();
    {  // qkv (3x 64 -> 64): each half owns its token range
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
        #pragma unroll
        for (int i = 0; i < 10; ++i) {
          const int t = t_lo + i;
          if (t < t_hi) {
            const float hv = s.h[t][k];
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
          s.u.a.k[t][lane] = ak[i];
          s.u.a.v[t][lane] = av[i];
        }
      }
      // h's LN1 content is dead after every k-loop: q reuses its storage
      __syncthreads();
      #pragma unroll
      for (int i = 0; i < 10; ++i) {
        const int t = t_lo + i;
        if (t < t_hi) s.h[t][lane] = aq[i];
      }
    }
    __syncthreads();
    // scores: q @ k^T / sqrt(64) + key mask — 361 flat dot products
    for (int idx = tid; idx < TT * TT; idx += THREADS) {
      const int t = idx / TT, u = idx % TT;
      float a = 0.f;
      for (int k = 0; k < D; ++k) a += s.h[t][k] * s.u.a.k[u][k];
      s.u.a.att[t][u] = a * 0.125f + s.madd[u];
    }
    __syncthreads();
    if (tid < TT) {  // softmax per query row
      float m = -3.4e38f;
      for (int u = 0; u < TT; ++u) m = fmaxf(m, s.u.a.att[tid][u]);
      float den = 0.f;
      for (int u = 0; u < TT; ++u) {
        const float e = expf(s.u.a.att[tid][u] - m);
        s.u.a.att[tid][u] = e;
        den += e;
      }
      const float inv = 1.f / den;
      for (int u = 0; u < TT; ++u) s.u.a.att[tid][u] *= inv;
    }
    __syncthreads();
    {  // residual add: x += att @ v (u-outer: one v load feeds all tokens)
      float a[10] = {0.f};
      #pragma unroll
      for (int u = 0; u < TT; ++u) {
        const float vu = s.u.a.v[u][lane];
        #pragma unroll
        for (int i = 0; i < 10; ++i) {
          const int t = t_lo + i;
          if (t < t_hi) a[i] += s.u.a.att[t][u] * vu;
        }
      }
      #pragma unroll
      for (int i = 0; i < 10; ++i) {
        const int t = t_lo + i;
        if (t < t_hi) s.x[t][lane] += a[i];
      }
    }
    __syncthreads();
    ln_tokens(s.x, s.h, B + B_LN2_G, B + B_LN2_B, tid);
    __syncthreads();
    {  // FFN layer 1 (64 -> 128, ReLU): thread owns hidden column tid
      float acc[TT];
      const float b1 = __ldg(B + B_FFB1 + tid);
      #pragma unroll
      for (int t = 0; t < TT; ++t) acc[t] = b1;
      for (int k = 0; k < D; ++k) {
        const float w = __ldg(B + B_FFW1 + k * TRUNK + tid);
        #pragma unroll
        for (int t = 0; t < TT; ++t) acc[t] += s.h[t][k] * w;
      }
      #pragma unroll
      for (int t = 0; t < TT; ++t) s.u.f.f1[t][tid] = fmaxf(acc[t], 0.f);
    }
    __syncthreads();
    {  // FFN layer 2 (128 -> 64) + residual
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
        if (t < t_hi) s.x[t][lane] += acc[i];
      }
    }
    __syncthreads();
  }

  // ---- last transformer block, pruned to the 3 readout tokens -------------
  if (n_blocks > 0) {
    const float* B = W + FRONT + (size_t)(n_blocks - 1) * BLK;
    ln_tokens(s.x, s.h, B + B_LN1_G, B + B_LN1_B, tid);
    __syncthreads();
    {  // k/v for all 19 tokens; q for tokens 0-2 (all live in half 0)
      float aq[3], ak[10], av[10];
      const float bk = __ldg(B + B_BK + lane);
      const float bv = __ldg(B + B_BV + lane);
      #pragma unroll
      for (int i = 0; i < 10; ++i) { ak[i] = bk; av[i] = bv; }
      if (half == 0) {
        const float bq = __ldg(B + B_BQ + lane);
        aq[0] = bq; aq[1] = bq; aq[2] = bq;
      }
      for (int k = 0; k < D; ++k) {
        const float wk = __ldg(B + B_WK + k * D + lane);
        const float wv = __ldg(B + B_WV + k * D + lane);
        #pragma unroll
        for (int i = 0; i < 10; ++i) {
          const int t = t_lo + i;
          if (t < t_hi) {
            const float hv = s.h[t][k];
            ak[i] += hv * wk;
            av[i] += hv * wv;
          }
        }
        if (half == 0) {
          const float wq = __ldg(B + B_WQ + k * D + lane);
          aq[0] += s.h[0][k] * wq;
          aq[1] += s.h[1][k] * wq;
          aq[2] += s.h[2][k] * wq;
        }
      }
      #pragma unroll
      for (int i = 0; i < 10; ++i) {
        const int t = t_lo + i;
        if (t < t_hi) {
          s.u.a.k[t][lane] = ak[i];
          s.u.a.v[t][lane] = av[i];
        }
      }
      __syncthreads();
      if (half == 0) {
        s.h[0][lane] = aq[0];
        s.h[1][lane] = aq[1];
        s.h[2][lane] = aq[2];
      }
    }
    __syncthreads();
    // scores for query rows 0-2 only (57 of the 361 dots)
    for (int idx = tid; idx < 3 * TT; idx += THREADS) {
      const int t = idx / TT, u = idx % TT;
      float a = 0.f;
      for (int k = 0; k < D; ++k) a += s.h[t][k] * s.u.a.k[u][k];
      s.u.a.att[t][u] = a * 0.125f + s.madd[u];
    }
    __syncthreads();
    if (tid < 3) {  // softmax, query rows 0-2
      float m = -3.4e38f;
      for (int u = 0; u < TT; ++u) m = fmaxf(m, s.u.a.att[tid][u]);
      float den = 0.f;
      for (int u = 0; u < TT; ++u) {
        const float e = expf(s.u.a.att[tid][u] - m);
        s.u.a.att[tid][u] = e;
        den += e;
      }
      const float inv = 1.f / den;
      for (int u = 0; u < TT; ++u) s.u.a.att[tid][u] *= inv;
    }
    __syncthreads();
    if (half == 0) {  // residual add: x += att @ v, tokens 0-2
      float a0 = 0.f, a1 = 0.f, a2 = 0.f;
      #pragma unroll
      for (int u = 0; u < TT; ++u) {
        const float vu = s.u.a.v[u][lane];
        a0 += s.u.a.att[0][u] * vu;
        a1 += s.u.a.att[1][u] * vu;
        a2 += s.u.a.att[2][u] * vu;
      }
      s.x[0][lane] += a0;
      s.x[1][lane] += a1;
      s.x[2][lane] += a2;
    }
    __syncthreads();
    {  // LN2 on tokens 0-2 only (warps 0-2 own one token each)
      const int warp = tid >> 5, wl = tid & 31;
      if (warp < 3) {
        const int t = warp;
        const float v0 = s.x[t][wl], v1 = s.x[t][wl + 32];
        float sum = v0 + v1;
        #pragma unroll
        for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(~0u, sum, o);
        const float mean = __shfl_sync(~0u, sum, 0) * (1.f / D);
        const float d0 = v0 - mean, d1 = v1 - mean;
        float sq = d0 * d0 + d1 * d1;
        #pragma unroll
        for (int o = 16; o; o >>= 1) sq += __shfl_down_sync(~0u, sq, o);
        const float rstd =
            rsqrtf(__shfl_sync(~0u, sq, 0) * (1.f / D) + 1e-5f);
        const float* g = B + B_LN2_G;
        const float* b = B + B_LN2_B;
        s.h[t][wl] = d0 * rstd * __ldg(g + wl) + __ldg(b + wl);
        s.h[t][wl + 32] = d1 * rstd * __ldg(g + wl + 32) + __ldg(b + wl + 32);
      }
    }
    __syncthreads();
    {  // FFN layer 1 (64 -> 128, ReLU), tokens 0-2
      float acc[3];
      const float b1 = __ldg(B + B_FFB1 + tid);
      acc[0] = b1; acc[1] = b1; acc[2] = b1;
      for (int k = 0; k < D; ++k) {
        const float w = __ldg(B + B_FFW1 + k * TRUNK + tid);
        acc[0] += s.h[0][k] * w;
        acc[1] += s.h[1][k] * w;
        acc[2] += s.h[2][k] * w;
      }
      s.u.f.f1[0][tid] = fmaxf(acc[0], 0.f);
      s.u.f.f1[1][tid] = fmaxf(acc[1], 0.f);
      s.u.f.f1[2][tid] = fmaxf(acc[2], 0.f);
    }
    __syncthreads();
    if (half == 0) {  // FFN layer 2 (128 -> 64) + residual, tokens 0-2
      float acc[3];
      const float b2 = __ldg(B + B_FFB2 + lane);
      acc[0] = b2; acc[1] = b2; acc[2] = b2;
      for (int k = 0; k < TRUNK; ++k) {
        const float w = __ldg(B + B_FFW2 + k * D + lane);
        acc[0] += s.u.f.f1[0][k] * w;
        acc[1] += s.u.f.f1[1][k] * w;
        acc[2] += s.u.f.f1[2][k] * w;
      }
      s.x[0][lane] += acc[0];
      s.x[1][lane] += acc[1];
      s.x[2][lane] += acc[2];
    }
    __syncthreads();
  }

  // ---- readout: trunk(concat 3 readout tokens) -> heads --------------------
  const float* TW = W + FRONT + (size_t)n_blocks * BLK;
  {  // trunk layer 1 (192 -> 128, ReLU); input z = x[0] | x[1] | x[2]
    float a = __ldg(TW + T_B1 + tid);
    for (int k = 0; k < 3 * D; ++k) {
      const float z = s.x[k / D][k % D];
      a += z * __ldg(TW + T_W1 + k * TRUNK + tid);
    }
    s.u.t.t1[tid] = fmaxf(a, 0.f);
  }
  __syncthreads();
  {  // trunk layer 2 (128 -> 128, linear — _mlp has no final act)
    float a = __ldg(TW + T_B2 + tid);
    for (int k = 0; k < TRUNK; ++k)
      a += s.u.t.t1[k] * __ldg(TW + T_W2 + k * TRUNK + tid);
    s.u.t.t2[tid] = a;
  }
  __syncthreads();
  const float* HW = TW + TRUNK_SZ;
  const int NLOGITS = NOUT - 1;
  for (int o = tid; o < NOUT; o += THREADS) {
    // 29 heads x K logits + value as one 128 -> NOUT matmul
    float a = __ldg(HW + H_W + TRUNK * NOUT + o);  // bias after the matrix
    for (int k = 0; k < TRUNK; ++k)
      a += s.u.t.t2[k] * __ldg(HW + H_W + k * NOUT + o);
    s.u.t.out[o] = a;
    if (o < NLOGITS) logits[(size_t)n * NLOGITS + o] = a;
    else value[n] = a;
  }
  __syncthreads();
  if (tid < N_HEADS) {  // strict-greater scan == torch.argmax (first max)
    const float* o = s.u.t.out + tid * n_gears;
    int best = 0;
    for (int k = 1; k < n_gears; ++k)
      if (o[k] > o[best]) best = k;
    actions[(size_t)n * N_HEADS + tid] = best;
  }
}

void forward(at::Tensor weights, at::Tensor obs, at::Tensor logits,
             at::Tensor value, at::Tensor actions, int64_t n_blocks,
             int64_t n_gears) {
  TORCH_CHECK(n_gears == 3 || n_gears == 5 || n_gears == 7,
              "n_gears must be 3, 5, or 7");
  const int64_t NLOGITS = N_HEADS * n_gears;
  TORCH_CHECK(obs.is_cuda() && obs.scalar_type() == at::kFloat
              && obs.is_contiguous() && obs.dim() == 2
              && obs.size(1) == OBS_DIM,
              "obs must be a contiguous CUDA f32 (N, ", OBS_DIM, ") tensor");
  const int64_t N = obs.size(0);
  TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == at::kFloat
              && weights.is_contiguous()
              && weights.numel() == expected_numel((int)n_blocks,
                                                   (int)n_gears),
              "packed weights must be contiguous CUDA f32 with ",
              expected_numel((int)n_blocks, (int)n_gears), " elements");
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat
              && logits.is_contiguous() && logits.numel() == N * NLOGITS);
  TORCH_CHECK(value.is_cuda() && value.scalar_type() == at::kFloat
              && value.is_contiguous() && value.numel() == N);
  TORCH_CHECK(actions.is_cuda() && actions.scalar_type() == at::kLong
              && actions.is_contiguous() && actions.numel() == N * N_HEADS);
  if (N == 0) return;
  policy_fwd_kernel<<<(unsigned)N, THREADS, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      weights.data_ptr<float>(), obs.data_ptr<float>(),
      logits.data_ptr<float>(), value.data_ptr<float>(),
      actions.data_ptr<int64_t>(), (int)n_blocks, (int)n_gears, (int)N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward,
        "Fused NotePolicy forward (env v10): writes logits (N, 29*K), "
        "value (N) and per-head argmax actions (N, 29) in one launch");
}
