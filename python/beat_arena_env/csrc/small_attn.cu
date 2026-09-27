// Fused multi-head attention for the policy's T = 29 token set
// (2026-09-27, audit #10). fp16 in/out, tensor-core mma.sync (fp16
// operands, fp32 accumulate), fp32 softmax, additive fp16 key bias.
//
// Why: torch's mem-efficient SDPA (cutlass fmha, 64x64 tiles) runs 29
// queries x 29 keys at ~45% useful work per tile across B x H CTAs and
// its backward re-reads K/V per query tile — 15.5 ms of an 83.5 ms PPO
// minibatch and 8.8 ms of a 53.5 ms rollout forward (N = 32768) on the
// RTX 5090 (2026-09-27 profile). Here ONE CTA owns one sample: k/v (and
// in the backward q and dO) are staged in shared memory once (the
// forward's q A-fragments come straight from global), and each of the 4 warps
// computes one head with m16n8k16 MMAs over the whole 32-padded score
// tile, the softmax row statistics reduced within a lane quad.
//
// Numerics: scores and every accumulation fp32; P (forward) and P / dS
// (backward) enter their second GEMM as an fp16 hi + lo pair (x = hi +
// lo, two MMAs), i.e. ~fp32-accurate — torch's cutlass kernels round P
// and dS to fp16 once. The backward recomputes P from q/k and the saved
// nothing: no logsumexp or output tensor is written by the forward. Its
// key-major half (dK, dV) recomputes S^T = K Q^T with the row statistics
// of the query-major half, so P^T and P agree to fp32 rounding.
//
// Shapes: heads H = 4, head dim DH = 48 (d = 192), keys T = 29, queries
// TQ in {29 (self-attention), 5 (the final block's lead-token queries),
// 1 (the critic's pooling query)}. q/k/v are strided views with unit
// last stride (column slices of the packed qkv / kv projection); the
// output is a contiguous (B, TQ, d) tensor — the out-projection's input
// layout. Grads are written straight into caller-provided strided views
// (for the packed projections: ONE packed (B, T, 3d) / (B, T, 2d) grad).
//
// Determinism: each sample is computed by the same instruction sequence
// whatever B is, so the rollout (grad-free) and update forwards agree
// bitwise at identical weights and obs (the PPO ratio's r1 contract).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace {

constexpr int H = 4, DH = 48, D = H * DH, T = 29, NT = 32 * H;
constexpr int LDS = D + 8;          // smem row stride (halves): 400 B, so
                                    // 8 ldmatrix rows hit distinct banks
constexpr int KT = 4;               // key n-tiles (32 padded keys)

struct AttnP {
  const __half* q;
  const __half* k;
  const __half* v;
  const __half* bias;
  int64_t q_bs, q_rs, k_bs, k_rs, v_bs, v_rs, bias_bs;
  float scale;
};

struct GradP {
  const __half* dout;  // (B, TQ, D) contiguous
  __half* dq;
  __half* dk;
  __half* dv;
  int64_t dq_bs, dq_rs, dk_bs, dk_rs, dv_bs, dv_rs;
};

// ---- PTX wrappers ----------------------------------------------------------
__device__ __forceinline__ void ldsm4(uint32_t* r, const __half* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void ldsm4t(uint32_t* r, const __half* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

// c += a (16x16, row) * b (16x8, col), fp16 in, fp32 accumulate
__device__ __forceinline__ void mma(float* c, const uint32_t* a,
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t h2u(__half2 h) {
  return *reinterpret_cast<uint32_t*>(&h);
}

// fp32 pair -> fp16 hi pair + fp16 lo pair (x ~= hi + lo to ~2^-22)
__device__ __forceinline__ void split2(float x0, float x1, uint32_t& hi,
                                       uint32_t& lo) {
  const __half2 h = __floats2half2_rn(x0, x1);
  const float2 hf = __half22float2(h);
  hi = h2u(h);
  lo = h2u(__floats2half2_rn(x0 - hf.x, x1 - hf.y));
}

// A fragments (hi, lo) of a 16x16 block formed by two adjacent 16x8
// accumulator tiles c0 (k 0..7) and c1 (k 8..15)
__device__ __forceinline__ void acc_to_a(const float* c0, const float* c1,
                                         uint32_t* ahi, uint32_t* alo) {
  split2(c0[0], c0[1], ahi[0], alo[0]);
  split2(c0[2], c0[3], ahi[1], alo[1]);
  split2(c1[0], c1[1], ahi[2], alo[2]);
  split2(c1[2], c1[3], ahi[3], alo[3]);
}

// ---- smem staging ----------------------------------------------------------
// rows x D halves from a strided global source into [rows][LDS] smem
__device__ __forceinline__ void stage(__half* dst, const __half* src,
                                      int64_t rs, int rows, int tid) {
  for (int c = tid; c < rows * (D / 8); c += NT) {
    const int j = c / (D / 8), u = c - j * (D / 8);
    *reinterpret_cast<uint4*>(dst + j * LDS + u * 8) =
        *reinterpret_cast<const uint4*>(src + j * rs + u * 8);
  }
}

// smem row r of a tile with `rows` valid rows; rows past the end read the
// shared zero row (ldmatrix needs 16/32-row tiles, the data has 29 / TQ)
__device__ __forceinline__ const __half* rowp(const __half* base, int r,
                                              int rows, const __half* zero) {
  return r < rows ? base + r * LDS : zero;
}

// S (MQ m-tiles x KT key n-tiles) = A_rows (MQ*16 x DH) * B_rows^T for
// head h: A and B both [row][dh] in smem (Q K^T, dO V^T, K Q^T, V dO^T).
// NB = number of B n-tiles (even).
template <int MA, int NB>
__device__ __forceinline__ void gemm_nt(float (*c)[NB][4], const __half* A,
                                        int arows, const __half* B,
                                        int brows, const __half* zero,
                                        int h, int lane) {
#pragma unroll
  for (int m = 0; m < MA; m++)
#pragma unroll
    for (int n = 0; n < NB; n++)
#pragma unroll
      for (int e = 0; e < 4; e++) c[m][n][e] = 0.f;
#pragma unroll
  for (int ks = 0; ks < DH / 16; ks++) {
    const int col = h * DH + ks * 16;
    uint32_t a[MA][4];
#pragma unroll
    for (int m = 0; m < MA; m++)
      ldsm4(a[m], rowp(A, m * 16 + (lane & 15), arows, zero) + col
                      + (lane >> 4) * 8);
#pragma unroll
    for (int np = 0; np < NB / 2; np++) {
      uint32_t b[4];
      ldsm4(b, rowp(B, np * 16 + (lane & 7) + (lane >> 4) * 8, brows, zero)
                   + col + ((lane >> 3) & 1) * 8);
#pragma unroll
      for (int m = 0; m < MA; m++) {
        mma(c[m][2 * np], a[m], b[0], b[1]);
        mma(c[m][2 * np + 1], a[m], b[2], b[3]);
      }
    }
  }
}

// O (MA m-tiles x 6 dh n-tiles) += P (MA*16 x KB*16, accumulator tiles
// p[m][kt]) * B (KB*16 x DH) with B [k-row][dh] in smem (P V, dS K,
// P^T dO, dS^T Q): hi + lo MMAs.
template <int MA, int KB, int NP>
__device__ __forceinline__ void gemm_pn(float (*o)[DH / 8][4],
                                        float (*p)[NP][4],
                                        const __half* B, int brows,
                                        const __half* zero, int h,
                                        int lane) {
#pragma unroll
  for (int m = 0; m < MA; m++)
#pragma unroll
    for (int n = 0; n < DH / 8; n++)
#pragma unroll
      for (int e = 0; e < 4; e++) o[m][n][e] = 0.f;
#pragma unroll
  for (int kb = 0; kb < KB; kb++) {
    uint32_t ahi[MA][4], alo[MA][4];
#pragma unroll
    for (int m = 0; m < MA; m++)
      acc_to_a(p[m][2 * kb], p[m][2 * kb + 1], ahi[m], alo[m]);
#pragma unroll
    for (int np = 0; np < DH / 16; np++) {
      uint32_t b[4];
      ldsm4t(b, rowp(B, kb * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, brows,
                     zero) + h * DH + np * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int m = 0; m < MA; m++) {
        mma(o[m][2 * np], ahi[m], b[0], b[1]);
        mma(o[m][2 * np], alo[m], b[0], b[1]);
        mma(o[m][2 * np + 1], ahi[m], b[2], b[3]);
        mma(o[m][2 * np + 1], alo[m], b[2], b[3]);
      }
    }
  }
}

// store an (MA*16 x DH) accumulator block, rows < rows_ok, times `mul`
template <int MA>
__device__ __forceinline__ void store_rows(float (*o)[DH / 8][4],
                                           __half* base, int64_t rs,
                                           int rows_ok, float mul, int h,
                                           int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int m = 0; m < MA; m++)
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
      const int r = m * 16 + g + hf * 8;
      if (r < rows_ok) {
        __half* rp = base + r * rs + h * DH + t * 2;
#pragma unroll
        for (int n = 0; n < DH / 8; n++)
          *reinterpret_cast<__half2*>(rp + n * 8) = __floats2half2_rn(
              o[m][n][hf * 2] * mul, o[m][n][hf * 2 + 1] * mul);
      }
    }
}

// ---- forward -----------------------------------------------------------------
template <int TQ>
__global__ void __launch_bounds__(NT)
attn29_fwd_kernel(AttnP a, __half* __restrict__ out) {
  constexpr int MQ = (TQ + 15) / 16;
  // Q is not staged: each thread loads its own m16n8k16 A fragments
  // straight from global (issued before the K/V staging wait; the same
  // values ldmatrix would deliver, so the MMAs are unchanged). 24 KB of
  // smem per CTA -> 4 CTAs / SM instead of 2.
  __shared__ __align__(16) __half sK[T * LDS];
  __shared__ __align__(16) __half sV[T * LDS];
  __shared__ __align__(16) __half sZ[LDS];
  __shared__ float sBias[32];
  const int b = blockIdx.x, tid = threadIdx.x;
  const int h = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  uint32_t qa[MQ][DH / 16][4];
  {
    const __half* qg = a.q + b * a.q_bs + h * DH + t * 2;
#pragma unroll
    for (int m = 0; m < MQ; m++)
#pragma unroll
      for (int ks = 0; ks < DH / 16; ks++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = m * 16 + g + (e & 1) * 8;
          qa[m][ks][e] = r < TQ
              ? *reinterpret_cast<const uint32_t*>(
                    qg + r * a.q_rs + ks * 16 + (e >> 1) * 8)
              : 0u;
        }
  }
  stage(sK, a.k + b * a.k_bs, a.k_rs, T, tid);
  stage(sV, a.v + b * a.v_bs, a.v_rs, T, tid);
  for (int c = tid; c < LDS / 8; c += NT)
    reinterpret_cast<uint4*>(sZ)[c] = make_uint4(0, 0, 0, 0);
  if (tid < 32)
    sBias[tid] = tid < T ? __half2float(a.bias[b * a.bias_bs + tid])
                         : -INFINITY;
  __syncthreads();
  float s[MQ][KT][4];
#pragma unroll
  for (int m = 0; m < MQ; m++)
#pragma unroll
    for (int n = 0; n < KT; n++)
#pragma unroll
      for (int e = 0; e < 4; e++) s[m][n][e] = 0.f;
#pragma unroll
  for (int ks = 0; ks < DH / 16; ks++) {
#pragma unroll
    for (int np = 0; np < KT / 2; np++) {
      uint32_t bf[4];
      ldsm4(bf, rowp(sK, np * 16 + (lane & 7) + (lane >> 4) * 8, T, sZ)
                    + h * DH + ks * 16 + ((lane >> 3) & 1) * 8);
#pragma unroll
      for (int m = 0; m < MQ; m++) {
        mma(s[m][2 * np], qa[m][ks], bf[0], bf[1]);
        mma(s[m][2 * np + 1], qa[m][ks], bf[2], bf[3]);
      }
    }
  }
  float inv[MQ][2];
#pragma unroll
  for (int m = 0; m < MQ; m++)
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
      float mx = -INFINITY;
#pragma unroll
      for (int n = 0; n < KT; n++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          float& x = s[m][n][hf * 2 + e];
          x = x * a.scale + sBias[n * 8 + t * 2 + e];
          mx = fmaxf(mx, x);
        }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      float l = 0.f;
#pragma unroll
      for (int n = 0; n < KT; n++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          float& x = s[m][n][hf * 2 + e];
          x = __expf(x - mx);
          l += x;
        }
      l += __shfl_xor_sync(0xffffffffu, l, 1);
      l += __shfl_xor_sync(0xffffffffu, l, 2);
      inv[m][hf] = 1.f / l;
    }
  float o[MQ][DH / 8][4];
  gemm_pn<MQ, 2, KT>(o, s, sV, T, sZ, h, lane);
  __half* ob = out + (int64_t)b * TQ * D;
#pragma unroll
  for (int m = 0; m < MQ; m++)
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
      const int r = m * 16 + g + hf * 8;
      if (r < TQ) {
        __half* rp = ob + r * D + h * DH + t * 2;
#pragma unroll
        for (int n = 0; n < DH / 8; n++)
          *reinterpret_cast<__half2*>(rp + n * 8) = __floats2half2_rn(
              o[m][n][hf * 2] * inv[m][hf], o[m][n][hf * 2 + 1] * inv[m][hf]);
      }
    }
}

// ---- backward ----------------------------------------------------------------
// Query-major half (lane quads own query rows): P, dP = dO V^T,
// D_i = sum_j P dP, dS = P (dP - D), dQ = scale dS K; the row max / 1/l /
// D go to smem. Key-major half (quads own key rows): S^T = K Q^T ->
// P^T with those stats, dP^T = V dO^T, dS^T, dV = P^T dO, dK = scale
// dS^T Q.
template <int TQ>
constexpr int bwd_smem_bytes() {
  return (2 * TQ + 2 * T + 1) * LDS * 2 + (32 + 3 * H * 32) * 4;
}

template <int TQ>
__global__ void __launch_bounds__(NT)
attn29_bwd_kernel(AttnP a, GradP gp) {
  constexpr int MQ = (TQ + 15) / 16;   // query m-tiles
  constexpr int NQ = 2 * MQ;           // query n-tiles (key-major half)
  extern __shared__ __align__(16) unsigned char smem[];
  __half* sQ = reinterpret_cast<__half*>(smem);
  __half* sO = sQ + TQ * LDS;          // dO
  __half* sK = sO + TQ * LDS;
  __half* sV = sK + T * LDS;
  __half* sZ = sV + T * LDS;
  float* sBias = reinterpret_cast<float*>(sZ + LDS);
  float* sM = sBias + 32;              // [H][32] row max
  float* sL = sM + H * 32;             // [H][32] 1 / row sum
  float* sD = sL + H * 32;             // [H][32] rowsum(P dP)
  const int b = blockIdx.x, tid = threadIdx.x;
  const __half* dob = gp.dout + (int64_t)b * TQ * D;
  stage(sQ, a.q + b * a.q_bs, a.q_rs, TQ, tid);
  stage(sO, dob, D, TQ, tid);
  stage(sK, a.k + b * a.k_bs, a.k_rs, T, tid);
  stage(sV, a.v + b * a.v_bs, a.v_rs, T, tid);
  for (int c = tid; c < LDS / 8; c += NT)
    reinterpret_cast<uint4*>(sZ)[c] = make_uint4(0, 0, 0, 0);
  if (tid < 32)
    sBias[tid] = tid < T ? __half2float(a.bias[b * a.bias_bs + tid])
                         : -INFINITY;
  __syncthreads();
  const int h = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  {  // ---- query-major half
    float p[MQ][KT][4], dp[MQ][KT][4];
    gemm_nt<MQ, KT>(p, sQ, TQ, sK, T, sZ, h, lane);
    gemm_nt<MQ, KT>(dp, sO, TQ, sV, T, sZ, h, lane);
#pragma unroll
    for (int m = 0; m < MQ; m++)
#pragma unroll
      for (int hf = 0; hf < 2; hf++) {
        float mx = -INFINITY;
#pragma unroll
        for (int n = 0; n < KT; n++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            float& x = p[m][n][hf * 2 + e];
            x = x * a.scale + sBias[n * 8 + t * 2 + e];
            mx = fmaxf(mx, x);
          }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        float l = 0.f;
#pragma unroll
        for (int n = 0; n < KT; n++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            float& x = p[m][n][hf * 2 + e];
            x = __expf(x - mx);
            l += x;
          }
        l += __shfl_xor_sync(0xffffffffu, l, 1);
        l += __shfl_xor_sync(0xffffffffu, l, 2);
        const float il = 1.f / l;
        float dsum = 0.f;
#pragma unroll
        for (int n = 0; n < KT; n++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            float& x = p[m][n][hf * 2 + e];
            x *= il;
            dsum = fmaf(x, dp[m][n][hf * 2 + e], dsum);
          }
        dsum += __shfl_xor_sync(0xffffffffu, dsum, 1);
        dsum += __shfl_xor_sync(0xffffffffu, dsum, 2);
#pragma unroll
        for (int n = 0; n < KT; n++)
#pragma unroll
          for (int e = 0; e < 2; e++)
            dp[m][n][hf * 2 + e] =
                p[m][n][hf * 2 + e] * (dp[m][n][hf * 2 + e] - dsum);
        if (t == 0) {
          const int r = m * 16 + g + hf * 8;
          sM[h * 32 + r] = mx;
          sL[h * 32 + r] = il;
          sD[h * 32 + r] = dsum;
        }
      }
    float dq[MQ][DH / 8][4];
    gemm_pn<MQ, 2, KT>(dq, dp, sK, T, sZ, h, lane);
    store_rows<MQ>(dq, gp.dq + b * gp.dq_bs, gp.dq_rs, TQ, a.scale, h, lane);
  }
  __syncwarp();
  {  // ---- key-major half
    float pt[2][NQ][4], dpt[2][NQ][4];
    gemm_nt<2, NQ>(pt, sK, T, sQ, TQ, sZ, h, lane);
    gemm_nt<2, NQ>(dpt, sV, T, sO, TQ, sZ, h, lane);
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
      for (int hf = 0; hf < 2; hf++) {
        const int j = m * 16 + g + hf * 8;          // key row
        const float bj = sBias[j];
#pragma unroll
        for (int n = 0; n < NQ; n++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            const int i = n * 8 + t * 2 + e;         // query column
            float x = 0.f;
            if (i < TQ && j < T)
              x = __expf(pt[m][n][hf * 2 + e] * a.scale + bj
                         - sM[h * 32 + i]) * sL[h * 32 + i];
            pt[m][n][hf * 2 + e] = x;
            dpt[m][n][hf * 2 + e] =
                x * (dpt[m][n][hf * 2 + e] - (i < TQ ? sD[h * 32 + i] : 0.f));
          }
      }
    float acc[2][DH / 8][4];
    gemm_pn<2, MQ, NQ>(acc, pt, sO, TQ, sZ, h, lane);
    store_rows<2>(acc, gp.dv + b * gp.dv_bs, gp.dv_rs, T, 1.f, h, lane);
    gemm_pn<2, MQ, NQ>(acc, dpt, sQ, TQ, sZ, h, lane);
    store_rows<2>(acc, gp.dk + b * gp.dk_bs, gp.dk_rs, T, a.scale, h, lane);
  }
}

void check_view(const at::Tensor& t, const char* n, int64_t B, int64_t rows) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kHalf && t.dim() == 3,
              n, ": (B, rows, 192) CUDA fp16");
  TORCH_CHECK(t.size(0) == B && t.size(1) == rows && t.size(2) == D, n,
              ": shape (", B, ", ", rows, ", ", D, ") expected");
  TORCH_CHECK(t.stride(2) == 1 && t.stride(1) % 8 == 0 && t.stride(0) % 8 == 0
              && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
              n, ": unit last stride, 16-byte aligned rows");
}

AttnP make_args(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                const at::Tensor& bias) {
  const int64_t B = q.size(0);
  check_view(q, "q", B, q.size(1));
  check_view(k, "k", B, T);
  check_view(v, "v", B, T);
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kHalf
              && bias.dim() == 2 && bias.size(0) == B && bias.size(1) == T
              && bias.stride(1) == 1, "bias: (B, 29) fp16, unit last stride");
  AttnP a;
  a.q = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
  a.k = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
  a.v = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
  a.bias = reinterpret_cast<const __half*>(bias.data_ptr<at::Half>());
  a.q_bs = q.stride(0); a.q_rs = q.stride(1);
  a.k_bs = k.stride(0); a.k_rs = k.stride(1);
  a.v_bs = v.stride(0); a.v_rs = v.stride(1);
  a.bias_bs = bias.stride(0);
  // torch's SDPA default scale: 1/sqrt(head_dim) computed in double
  a.scale = (float)(1.0 / std::sqrt((double)DH));
  return a;
}

template <int TQ>
void launch_bwd(const AttnP& a, const GradP& g, int64_t B,
                cudaStream_t stream) {
  constexpr int smem = bwd_smem_bytes<TQ>();
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        attn29_bwd_kernel<TQ>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));
    attr = true;
  }
  attn29_bwd_kernel<TQ><<<B, NT, smem, stream>>>(a, g);
}

}  // namespace

// out (B, TQ, 192) = MHA(q, k, v; additive key bias), 4 heads x 48.
at::Tensor attn29_fwd(at::Tensor q, at::Tensor k, at::Tensor v,
                      at::Tensor bias) {
  const c10::cuda::CUDAGuard guard(q.device());
  AttnP a = make_args(q, k, v, bias);
  const int64_t B = q.size(0), TQ = q.size(1);
  auto out = at::empty({B, TQ, D}, q.options().memory_format(
      at::MemoryFormat::Contiguous));
  if (B == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  __half* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  if (TQ == T) {
    attn29_fwd_kernel<T><<<B, NT, 0, stream>>>(a, op);
  } else if (TQ == 5) {
    attn29_fwd_kernel<5><<<B, NT, 0, stream>>>(a, op);
  } else if (TQ == 1) {
    attn29_fwd_kernel<1><<<B, NT, 0, stream>>>(a, op);
  } else {
    TORCH_CHECK(false, "attn29_fwd: TQ must be 29, 5 or 1");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Writes dq / dk / dv into the given strided views.
void attn29_bwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor bias,
                at::Tensor dout, at::Tensor dq, at::Tensor dk,
                at::Tensor dv) {
  const c10::cuda::CUDAGuard guard(q.device());
  AttnP a = make_args(q, k, v, bias);
  const int64_t B = q.size(0), TQ = q.size(1);
  TORCH_CHECK(dout.is_contiguous() && dout.scalar_type() == at::kHalf
              && dout.dim() == 3 && dout.size(0) == B && dout.size(1) == TQ
              && dout.size(2) == D, "dout: contiguous (B, TQ, 192) fp16");
  check_view(dq, "dq", B, TQ);
  check_view(dk, "dk", B, T);
  check_view(dv, "dv", B, T);
  if (B == 0) return;
  GradP g;
  g.dout = reinterpret_cast<const __half*>(dout.data_ptr<at::Half>());
  g.dq = reinterpret_cast<__half*>(dq.data_ptr<at::Half>());
  g.dk = reinterpret_cast<__half*>(dk.data_ptr<at::Half>());
  g.dv = reinterpret_cast<__half*>(dv.data_ptr<at::Half>());
  g.dq_bs = dq.stride(0); g.dq_rs = dq.stride(1);
  g.dk_bs = dk.stride(0); g.dk_rs = dk.stride(1);
  g.dv_bs = dv.stride(0); g.dv_rs = dv.stride(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (TQ == T) {
    launch_bwd<T>(a, g, B, stream);
  } else if (TQ == 5) {
    launch_bwd<5>(a, g, B, stream);
  } else if (TQ == 1) {
    launch_bwd<1>(a, g, B, stream);
  } else {
    TORCH_CHECK(false, "attn29_bwd: TQ must be 29, 5 or 1");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
