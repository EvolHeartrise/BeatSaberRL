// Fused NotePolicy forward (inference only) as ONE CUDA kernel launch.
//
// The whole policy.py forward — note/saber MLPs, 2 pre-LN transformer
// blocks (single-head QKV attention with key masking, FFN, residuals),
// trunk, 12 action heads + value head, and per-head argmax — runs in a
// single launch: one thread block (128 threads) per sample, activations
// in shared memory, weights streamed from a packed flat tensor.
//
// Why this wins at small batch: the eager forward is ~100 tiny kernels,
// each round-tripping (B, 18, 64) activations through global memory and
// paying a launch; here activations never leave the SM and the launch
// count is 1. The GTX 1080 has no tensor cores, so fp32 SIMT FMAs are
// the peak path anyway.
//
// Layout / fusion choices:
// - Weights are packed by policy_cuda.py into ONE fp32 tensor, every
//   matrix TRANSPOSED to [in][out] so the thread that owns output j
//   reads W[k*OUT + j] — coalesced across the warp, broadcast across
//   token accumulators. Offsets below are a contract with _pack().
//   Total = 107,365 floats (429 KB): L2-resident, so all N blocks after
//   the first read it at L2 bandwidth.
// - Shared memory (~21 KB/block, 4 blocks/SM): obs row (286), residual stream
//   x[18][65] (row stride 65 kills bank conflicts in the score phase,
//   where threads read different token rows at the same column),
//   post-LN scratch h[18][65], and a phase union overlaying
//   {q,k,v,scores} / ffn hidden [18][129] / trunk scratch.
// - Matmuls are output-stationary over tokens: thread tid owns output
//   lane = tid%64 and half = tid/64 of the 18 tokens (9 each), so each
//   weight element is loaded once (twice for split-token layers) and
//   feeds 9-18 FMAs off shared-memory broadcasts.
// - Attention: scores (18x18) computed by 324 flat dot products over
//   all threads; key masking adds (mask-1)*16384 exactly as policy.py;
//   softmax per row by one thread (18 rows, 18 wide — trivial);
//   att @ v accumulates straight into the residual x rows.
// - LayerNorm: one warp per token, shuffle reductions, biased variance,
//   eps 1e-5 (torch defaults).
// - Argmax: strict-greater scan per head (torch.argmax returns the
//   first maximal index) — written as int64 actions alongside logits.
//
// - Last-block pruning (2026-07-27, mirrors policy.py's exact pruning):
//   only the two saber tokens feed the trunk, so the final block computes
//   q/scores/softmax/att@v/LN2/FFN for tokens 0-1 only (k/v stay
//   full-width). Bitwise identical to the unpruned kernel — surviving
//   tokens keep the same FP ops in the same order.
//
// No backward, fp32 only, arch fixed at embed=64 / trunk=128 / 1 head
// (both production checkpoints); n_blocks is a runtime loop.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

constexpr int D = 64;           // embed
constexpr int TT = 18;          // tokens: 2 sabers + 16 note slots
constexpr int TRUNK = 128;
constexpr int NOTE_FEAT = 14;
constexpr int N_NOTE_SLOTS = 16;
constexpr int SABER_FEAT = 23;
constexpr int SABER_IN = SABER_FEAT + 2;              // + hand one-hot
constexpr int OFF_NOTES = 2 * SABER_FEAT;             // 46
constexpr int OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT;  // 270
constexpr int OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS; // 286
constexpr int N_HEADS = 12;     // action heads, 3 logits each
constexpr int NLOGITS = 3 * N_HEADS;                  // 36
constexpr int NOUT = NLOGITS + 1;                     // + value
constexpr int THREADS = 128;
constexpr int SROW = D + 1;     // padded token-row stride (bank conflicts)

// --- packed-weight offsets (contract with policy_cuda._pack) ---------------
constexpr int O_NOTE_W1 = 0;
constexpr int O_NOTE_B1 = O_NOTE_W1 + NOTE_FEAT * D;
constexpr int O_NOTE_W2 = O_NOTE_B1 + D;
constexpr int O_NOTE_B2 = O_NOTE_W2 + D * D;
constexpr int O_SAB_W1 = O_NOTE_B2 + D;
constexpr int O_SAB_B1 = O_SAB_W1 + SABER_IN * D;
constexpr int O_SAB_W2 = O_SAB_B1 + D;
constexpr int O_SAB_B2 = O_SAB_W2 + D * D;
constexpr int FRONT = O_SAB_B2 + D;                   // 10944
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
// after the blocks
constexpr int T_W1 = 0;
constexpr int T_B1 = T_W1 + TRUNK * TRUNK;
constexpr int T_W2 = T_B1 + TRUNK;
constexpr int T_B2 = T_W2 + TRUNK * TRUNK;
constexpr int TRUNK_SZ = T_B2 + TRUNK;                // 33024
constexpr int H_W = 0;
constexpr int H_B = H_W + TRUNK * NOUT;
constexpr int HEAD_SZ = H_B + NOUT;                   // 4773
static_assert(FRONT == 10944 && BLK == 29312 && TRUNK_SZ == 33024
              && HEAD_SZ == 4773, "packed layout drifted");

constexpr int expected_numel(int n_blocks) {
  return FRONT + n_blocks * BLK + TRUNK_SZ + HEAD_SZ;
}

// Shared-memory diet (2026-07-27): q has no union slot — after the qkv
// phase's register accumulation, LN1's content in `h` is dead, so q is
// stored INTO `h` (same SROW padding, so the score phase's addressing is
// unchanged). 26,056 -> 21,376 B/block lifts residency from 3 to 4
// blocks/SM on GP104 — the kernel measured latency-bound, so warps are
// the scarce resource. Bit-identical: pure storage relocation.
struct Smem {
  float x[TT][SROW];     // residual stream
  float h[TT][SROW];     // post-LN / MLP hidden scratch; q lives here
                         // between the qkv store and LN2's overwrite
  float obs[OBS_DIM];
  float madd[TT];        // (key_mask - 1) * 16384 additive score mask
  union {
    struct { float k[TT][SROW], v[TT][SROW];
             float att[TT][TT + 2]; } a;
    struct { float f1[TT][TRUNK + 1]; } f;
    struct { float t1[TRUNK], t2[TRUNK], out[NOUT]; } t;
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
                  int64_t* __restrict__ actions, int n_blocks, int N) {
  __shared__ Smem s;
  const int n = blockIdx.x;
  if (n >= N) return;
  const int tid = threadIdx.x;
  const int lane = tid & (D - 1);   // output index for 64-wide layers
  const int half = tid >> 6;        // token-half selector (0 or 1)

  // ---- stage obs row; additive key mask -----------------------------------
  const float* orow = obs + (size_t)n * OBS_DIM;
  for (int i = tid; i < OBS_DIM; i += THREADS) s.obs[i] = orow[i];
  __syncthreads();
  if (tid < TT)
    s.madd[tid] =
        (tid < 2) ? 0.f : (s.obs[OFF_NOTE_MASK + tid - 2] - 1.f) * 16384.f;

  // ---- front MLPs: 16 note tokens + 2 saber tokens -> x -------------------
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
      s.h[2 + half * 8 + i][lane] = fmaxf(acc[i], 0.f);
  }
  {  // saber layer 1 (25 -> 64, ReLU): token t == half; features are the
     // 23 obs saber features + hand one-hot ([1,0] left, [0,1] right)
    float a = __ldg(W + O_SAB_B1 + lane);
    for (int k = 0; k < SABER_FEAT; ++k)
      a += s.obs[half * SABER_FEAT + k] * __ldg(W + O_SAB_W1 + k * D + lane);
    a += __ldg(W + O_SAB_W1 + (SABER_FEAT + half) * D + lane);  // one-hot
    s.h[half][lane] = fmaxf(a, 0.f);
  }
  __syncthreads();
  {  // note layer 2 (64 -> 64, linear): 2 outputs x 4 tokens per thread
     // (2026-07-27 FMA:load blocking — one h[t][k] smem load feeds two
     // FMAs; each (t, out) accumulator keeps its own ascending-k chain,
     // so results are bit-identical to the 1-output x 8-token layout)
    float acc[4], acc2[4];
    const int g = tid >> 5;             // token group: 4 tokens each
    const int o = tid & 31;             // output pair (o, o+32)
    const float b2a = __ldg(W + O_NOTE_B2 + o);
    const float b2b = __ldg(W + O_NOTE_B2 + o + 32);
    #pragma unroll
    for (int i = 0; i < 4; ++i) { acc[i] = b2a; acc2[i] = b2b; }
    for (int k = 0; k < D; ++k) {
      const float wa = __ldg(W + O_NOTE_W2 + k * D + o);
      const float wb = __ldg(W + O_NOTE_W2 + k * D + o + 32);
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float hv = s.h[2 + g * 4 + i][k];
        acc[i] += hv * wa; acc2[i] += hv * wb;
      }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      s.x[2 + g * 4 + i][o] = acc[i];
      s.x[2 + g * 4 + i][o + 32] = acc2[i];
    }
  }
  {  // saber layer 2
    float a = __ldg(W + O_SAB_B2 + lane);
    for (int k = 0; k < D; ++k)
      a += s.h[half][k] * __ldg(W + O_SAB_W2 + k * D + lane);
    s.x[half][lane] = a;
  }
  __syncthreads();

  // ---- transformer blocks -------------------------------------------------
  // All but the last run full-width; the last block is pruned to the two
  // saber tokens (see below).
  for (int bi = 0; bi + 1 < n_blocks; ++bi) {
    const float* B = W + FRONT + (size_t)bi * BLK;
    ln_tokens(s.x, s.h, B + B_LN1_G, B + B_LN1_B, tid);
    __syncthreads();
    {  // qkv (3x 64 -> 64): each half owns 9 tokens, 27 accumulators
      float aq[9], ak[9], av[9];
      const float bq = __ldg(B + B_BQ + lane);
      const float bk = __ldg(B + B_BK + lane);
      const float bv = __ldg(B + B_BV + lane);
      #pragma unroll
      for (int i = 0; i < 9; ++i) { aq[i] = bq; ak[i] = bk; av[i] = bv; }
      for (int k = 0; k < D; ++k) {
        const float wq = __ldg(B + B_WQ + k * D + lane);
        const float wk = __ldg(B + B_WK + k * D + lane);
        const float wv = __ldg(B + B_WV + k * D + lane);
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
          const float hv = s.h[half * 9 + i][k];
          aq[i] += hv * wq; ak[i] += hv * wk; av[i] += hv * wv;
        }
      }
      #pragma unroll
      for (int i = 0; i < 9; ++i) {
        const int t = half * 9 + i;
        s.u.a.k[t][lane] = ak[i];
        s.u.a.v[t][lane] = av[i];
      }
      // h's LN1 content is dead once every thread's k-loop is done; the
      // barrier makes that true, then q reuses h's storage (smem diet)
      __syncthreads();
      #pragma unroll
      for (int i = 0; i < 9; ++i) s.h[half * 9 + i][lane] = aq[i];
    }
    __syncthreads();
    // scores: q @ k^T / sqrt(64) + key mask — 324 flat dot products.
    // Each thread's 2-3 dots interleave their k-loops (2026-07-27):
    // independent accumulators in the same k order are bit-identical to
    // the back-to-back version but give the scheduler 2-3 FMA chains to
    // hide latency with instead of one 64-long dependent chain.
    {
      const int i1 = tid + THREADS, i2 = tid + 2 * THREADS;
      const bool has2 = i2 < TT * TT;   // 324 dots: threads 0-67 own 3
      const int t0 = tid / TT, u0 = tid % TT;
      const int t1 = i1 / TT, u1 = i1 % TT;
      const int t2 = has2 ? i2 / TT : 0, u2 = has2 ? i2 % TT : 0;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f;
      for (int k = 0; k < D; ++k) {   // q lives in h (smem diet)
        a0 += s.h[t0][k] * s.u.a.k[u0][k];
        a1 += s.h[t1][k] * s.u.a.k[u1][k];
        if (has2) a2 += s.h[t2][k] * s.u.a.k[u2][k];
      }
      s.u.a.att[t0][u0] = a0 * 0.125f + s.madd[u0];
      s.u.a.att[t1][u1] = a1 * 0.125f + s.madd[u1];
      if (has2) s.u.a.att[t2][u2] = a2 * 0.125f + s.madd[u2];
    }
    __syncthreads();
    if (tid < TT) {  // softmax per query row (18-wide: one thread each)
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
    {  // residual add: x += att @ v (each half owns 9 tokens). u-outer
       // (2026-07-27): one v[u][lane] load feeds 9 FMAs instead of 1;
       // each token's accumulator still sums u ascending -> bit-identical.
      float a[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int u = 0; u < TT; ++u) {
        const float vu = s.u.a.v[u][lane];
        #pragma unroll
        for (int i = 0; i < 9; ++i)
          a[i] += s.u.a.att[half * 9 + i][u] * vu;
      }
      #pragma unroll
      for (int i = 0; i < 9; ++i) s.x[half * 9 + i][lane] += a[i];
    }
    __syncthreads();
    ln_tokens(s.x, s.h, B + B_LN2_G, B + B_LN2_B, tid);
    __syncthreads();
    {  // FFN layer 1 (64 -> 128, ReLU): 2 outputs x 9 tokens per thread
       // (2026-07-27 FMA:load blocking, was 1 output x 18 tokens): same
       // 18 accumulators, but one h[t][k] smem load now feeds two FMAs
       // (11 LSU ops per 18 FMAs vs 19). Each (t, out) accumulator chain
       // keeps the same ascending-k order -> bit-identical.
      float acc[9], acc2[9];
      const float b1a = __ldg(B + B_FFB1 + lane);
      const float b1b = __ldg(B + B_FFB1 + lane + D);
      #pragma unroll
      for (int i = 0; i < 9; ++i) { acc[i] = b1a; acc2[i] = b1b; }
      for (int k = 0; k < D; ++k) {
        const float wa = __ldg(B + B_FFW1 + k * TRUNK + lane);
        const float wb = __ldg(B + B_FFW1 + k * TRUNK + lane + D);
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
          const float hv = s.h[half * 9 + i][k];
          acc[i] += hv * wa; acc2[i] += hv * wb;
        }
      }
      #pragma unroll
      for (int i = 0; i < 9; ++i) {
        const int t = half * 9 + i;
        s.u.f.f1[t][lane] = fmaxf(acc[i], 0.f);
        s.u.f.f1[t][lane + D] = fmaxf(acc2[i], 0.f);
      }
    }
    __syncthreads();
    {  // FFN layer 2 (128 -> 64) + residual
      float acc[9];
      const float b2 = __ldg(B + B_FFB2 + lane);
      #pragma unroll
      for (int i = 0; i < 9; ++i) acc[i] = b2;
      for (int k = 0; k < TRUNK; ++k) {
        const float w = __ldg(B + B_FFW2 + k * D + lane);
        #pragma unroll
        for (int i = 0; i < 9; ++i) acc[i] += s.u.f.f1[half * 9 + i][k] * w;
      }
      #pragma unroll
      for (int i = 0; i < 9; ++i) s.x[half * 9 + i][lane] += acc[i];
    }
    __syncthreads();
  }

  // ---- last transformer block, pruned (2026-07-27) ------------------------
  // Only x[0]|x[1] feed the trunk readout, so tokens 2-17 are dead after
  // this block. Mirror of policy.py's exact last-block pruning: k/v (and
  // LN1 feeding them) stay full-width — every kept token attends over all
  // 18 keys — while q, score rows, softmax, att@v, LN2 and both FFN layers
  // run for tokens 0-1 only. Bitwise identical to the full block for the
  // surviving tokens: per-token accumulator chains keep the same FP ops in
  // the same order; dropped work wrote only rows never read again.
  if (n_blocks > 0) {
    const float* B = W + FRONT + (size_t)(n_blocks - 1) * BLK;
    ln_tokens(s.x, s.h, B + B_LN1_G, B + B_LN1_B, tid);
    __syncthreads();
    {  // k/v for all 18 tokens; q for tokens 0-1 (both live in half 0)
      float aq[2], ak[9], av[9];
      const float bk = __ldg(B + B_BK + lane);
      const float bv = __ldg(B + B_BV + lane);
      #pragma unroll
      for (int i = 0; i < 9; ++i) { ak[i] = bk; av[i] = bv; }
      if (half == 0) {
        const float bq = __ldg(B + B_BQ + lane);
        aq[0] = bq; aq[1] = bq;
      }
      for (int k = 0; k < D; ++k) {
        const float wk = __ldg(B + B_WK + k * D + lane);
        const float wv = __ldg(B + B_WV + k * D + lane);
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
          const float hv = s.h[half * 9 + i][k];
          ak[i] += hv * wk; av[i] += hv * wv;
        }
        if (half == 0) {
          const float wq = __ldg(B + B_WQ + k * D + lane);
          aq[0] += s.h[0][k] * wq;
          aq[1] += s.h[1][k] * wq;
        }
      }
      #pragma unroll
      for (int i = 0; i < 9; ++i) {
        const int t = half * 9 + i;
        s.u.a.k[t][lane] = ak[i];
        s.u.a.v[t][lane] = av[i];
      }
      // barrier, then q rows 0-1 reuse h's dead storage (smem diet)
      __syncthreads();
      if (half == 0) { s.h[0][lane] = aq[0]; s.h[1][lane] = aq[1]; }
    }
    __syncthreads();
    // scores for query rows 0-1 only (36 of the 324 dots); q lives in h
    for (int idx = tid; idx < 2 * TT; idx += THREADS) {
      const int t = idx / TT, u = idx % TT;
      float a = 0.f;
      for (int k = 0; k < D; ++k) a += s.h[t][k] * s.u.a.k[u][k];
      s.u.a.att[t][u] = a * 0.125f + s.madd[u];
    }
    __syncthreads();
    if (tid < 2) {  // softmax, query rows 0-1
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
    if (half == 0) {  // residual add: x += att @ v, tokens 0-1 (u-outer,
                      // shared v load; ascending-u chains -> bit-identical)
      float a0 = 0.f, a1 = 0.f;
      #pragma unroll
      for (int u = 0; u < TT; ++u) {
        const float vu = s.u.a.v[u][lane];
        a0 += s.u.a.att[0][u] * vu;
        a1 += s.u.a.att[1][u] * vu;
      }
      s.x[0][lane] += a0;
      s.x[1][lane] += a1;
    }
    __syncthreads();
    {  // LN2 on tokens 0-1 only: same math as ln_tokens rows 0-1 (there
       // warp 0 owns t=0 and warp 1 owns t=1 on their first iteration)
      const int warp = tid >> 5, wl = tid & 31;
      if (warp < 2) {
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
    {  // FFN layer 1 (64 -> 128, ReLU), tokens 0-1
      float acc[2];
      const float b1 = __ldg(B + B_FFB1 + tid);
      acc[0] = b1; acc[1] = b1;
      for (int k = 0; k < D; ++k) {
        const float w = __ldg(B + B_FFW1 + k * TRUNK + tid);
        acc[0] += s.h[0][k] * w;
        acc[1] += s.h[1][k] * w;
      }
      s.u.f.f1[0][tid] = fmaxf(acc[0], 0.f);
      s.u.f.f1[1][tid] = fmaxf(acc[1], 0.f);
    }
    __syncthreads();
    if (half == 0) {  // FFN layer 2 (128 -> 64) + residual, tokens 0-1
      float acc[2];
      const float b2 = __ldg(B + B_FFB2 + lane);
      acc[0] = b2; acc[1] = b2;
      for (int k = 0; k < TRUNK; ++k) {
        const float w = __ldg(B + B_FFW2 + k * D + lane);
        acc[0] += s.u.f.f1[0][k] * w;
        acc[1] += s.u.f.f1[1][k] * w;
      }
      s.x[0][lane] += acc[0];
      s.x[1][lane] += acc[1];
    }
    __syncthreads();
  }

  // ---- readout: trunk(concat saber tokens) -> heads -----------------------
  const float* TW = W + FRONT + (size_t)n_blocks * BLK;
  {  // trunk layer 1 (128 -> 128, ReLU); input z = x[0] | x[1]
    float a = __ldg(TW + T_B1 + tid);
    for (int k = 0; k < TRUNK; ++k) {
      const float z = (k < D) ? s.x[0][k] : s.x[1][k - D];
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
  if (tid < NOUT) {  // 12 heads x 3 logits + value as one 128 -> 37 matmul
    float a = __ldg(HW + H_B + tid);
    for (int k = 0; k < TRUNK; ++k)
      a += s.u.t.t2[k] * __ldg(HW + H_W + k * NOUT + tid);
    s.u.t.out[tid] = a;
    if (tid < NLOGITS) logits[(size_t)n * NLOGITS + tid] = a;
    else value[n] = a;
  }
  __syncthreads();
  if (tid < N_HEADS) {  // strict-greater scan == torch.argmax (first max)
    const float* o = s.u.t.out + tid * 3;
    int best = 0;
    if (o[1] > o[0]) best = 1;
    if (o[2] > o[best]) best = 2;
    actions[(size_t)n * N_HEADS + tid] = best;
  }
}

void forward(at::Tensor weights, at::Tensor obs, at::Tensor logits,
             at::Tensor value, at::Tensor actions, int64_t n_blocks) {
  TORCH_CHECK(obs.is_cuda() && obs.scalar_type() == at::kFloat
              && obs.is_contiguous() && obs.dim() == 2
              && obs.size(1) == OBS_DIM,
              "obs must be a contiguous CUDA f32 (N, ", OBS_DIM, ") tensor");
  const int64_t N = obs.size(0);
  TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == at::kFloat
              && weights.is_contiguous()
              && weights.numel() == expected_numel((int)n_blocks),
              "packed weights must be contiguous CUDA f32 with ",
              expected_numel((int)n_blocks), " elements");
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
      actions.data_ptr<int64_t>(), (int)n_blocks, (int)N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward,
        "Fused NotePolicy forward: writes logits (N,36), value (N) and "
        "per-head argmax actions (N,12) in one kernel launch");
}
