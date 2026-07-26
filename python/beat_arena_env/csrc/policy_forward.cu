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
// - Shared memory (~26 KB/block): obs row (286), residual stream
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

struct Smem {
  float x[TT][SROW];     // residual stream
  float h[TT][SROW];     // post-LN / MLP hidden scratch
  float obs[OBS_DIM];
  float madd[TT];        // (key_mask - 1) * 16384 additive score mask
  union {
    struct { float q[TT][SROW], k[TT][SROW], v[TT][SROW];
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

__global__ void __launch_bounds__(THREADS)
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
  {  // note layer 2 (64 -> 64, linear)
    float acc[8];
    const float b2 = __ldg(W + O_NOTE_B2 + lane);
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = b2;
    for (int k = 0; k < D; ++k) {
      const float w = __ldg(W + O_NOTE_W2 + k * D + lane);
      #pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] += s.h[2 + half * 8 + i][k] * w;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) s.x[2 + half * 8 + i][lane] = acc[i];
  }
  {  // saber layer 2
    float a = __ldg(W + O_SAB_B2 + lane);
    for (int k = 0; k < D; ++k)
      a += s.h[half][k] * __ldg(W + O_SAB_W2 + k * D + lane);
    s.x[half][lane] = a;
  }
  __syncthreads();

  // ---- transformer blocks -------------------------------------------------
  for (int bi = 0; bi < n_blocks; ++bi) {
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
        s.u.a.q[t][lane] = aq[i];
        s.u.a.k[t][lane] = ak[i];
        s.u.a.v[t][lane] = av[i];
      }
    }
    __syncthreads();
    // scores: q @ k^T / sqrt(64) + key mask — 324 flat dot products
    for (int idx = tid; idx < TT * TT; idx += THREADS) {
      const int t = idx / TT, u = idx % TT;
      float a = 0.f;
      for (int k = 0; k < D; ++k) a += s.u.a.q[t][k] * s.u.a.k[u][k];
      s.u.a.att[t][u] = a * 0.125f + s.madd[u];
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
    {  // residual add: x += att @ v (each half owns 9 tokens)
      #pragma unroll
      for (int i = 0; i < 9; ++i) {
        const int t = half * 9 + i;
        float a = 0.f;
        #pragma unroll
        for (int u = 0; u < TT; ++u) a += s.u.a.att[t][u] * s.u.a.v[u][lane];
        s.x[t][lane] += a;
      }
    }
    __syncthreads();
    ln_tokens(s.x, s.h, B + B_LN2_G, B + B_LN2_B, tid);
    __syncthreads();
    {  // FFN layer 1 (64 -> 128, ReLU): thread tid owns output tid for
       // ALL 18 tokens — each weight element loaded once
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
