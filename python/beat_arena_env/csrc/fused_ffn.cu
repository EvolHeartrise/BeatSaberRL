// Fused transformer-block FFN for the policy's d = 192 / hidden 768 blocks
// (2026-09-30, review item I13). fp16 in/out, tensor-core mma.sync
// (m16n8k16, fp16 operands), same PTX idiom as small_attn.cu.
//
//   out  = x + (relu(h2 W0^T + b0) W2^T + b2)          residual stream
//   hn   = LayerNorm_next(out)                          (LN variant)
//
// Why: the unfused block FFN (fp16_linear.fp16_relu_chain + the NEXT
// block's ln_attn.forward_add) writes the 768-wide hidden activation AND a
// ReLU bitmask (~1.6 KB / token), reads the hidden back for ffn.2, writes
// ffn.2's output, and reads that again in the next LayerNorm pass — at
// N = 32768 x 29 tokens x 5 non-final blocks, ~15 GB of DRAM traffic per
// rollout forward that exists only to shuttle intermediates. Here the
// hidden layer never leaves the SM: each CTA owns a 128-row tile, streams
// W0 / W2 through shared memory in 32-column hidden chunks, keeps the
// 128 x 192 output accumulator in registers and, in the epilogue, adds the
// residual and runs the next block's LayerNorm, writing only the residual
// stream and the LN output.
//
// Tiling (BM = 128 rows, 8 warps, warp w owns rows [16w, 16w + 16)):
//   registers / thread
//     A1[12][4]   h2 A-fragments of the warp's 16 rows x 192 (loaded once
//                 per tile straight from global, like attn29_fwd's Q)  48
//     O[24]       16 x 192 output accumulator: fp32 (4 regs / n-tile) 96
//                 or fp16 (2 regs / n-tile, DN16)                      48
//     S[4]        16 x 32 hidden chunk accumulator (fp32)              16
//     A2[2][4]    the chunk as fp16 GEMM2 A-fragments                   8
//     B ring      2 ldmatrix.x4 fragments                               8
//   -> ~190-210 regs at fp32 accumulate: __launch_bounds__(256, 1), one CTA
//      per SM (8 warps), persistent over row tiles.
//   shared memory (dynamic, 89,856 B)
//     3 stages x [W0 chunk 32 x (192+8) | W2 chunk 192 x (32+8)] fp16
//       = 3 x 28,160 B (row pads keep every ldmatrix 8-row phase on
//         distinct banks: 400 B and 80 B strides)
//     b0 (768) + b2 / ln_w / ln_b (192 each) fp32 = 5,376 B
//   per chunk and warp: GEMM1 48 MMAs (S = A1 W0c^T, K = 192), bias+ReLU
//   in registers -> fp16 A-fragments, GEMM2 48 MMAs (O += S W2c^T, K = 32);
//   one __syncthreads per chunk, cp.async weight prefetch 2 chunks ahead
//   (the chunk sequence runs on across the CTA's tiles, so the next tile's
//   first weights load under the previous tile's epilogue).
//
// Numerics:
//   GEMM1 (up-projection) accumulates fp32 (UP16 = false), matching the
//     unfused path's cuBLASLt RELU_AUX_BIAS GEMM (never fp16-accumulated,
//     fp16_linear.fp16_accum); UP16 = true is an experimental fp16-
//     accumulate variant (fused_ffn mode "up16").
//   The hidden value is relu(S + b0) (b0 = the fp32 MASTER bias) rounded
//     ONCE to fp16 — the same rounding point as the unfused path's stored
//     hidden; it enters GEMM2 as that fp16 value (and, in grad mode, is the
//     exact tensor written for the backward).
//   GEMM2 (down-projection) accumulates fp32, or fp16 when DN16 (= the
//     policy's fp16_accum mode "fwd"/"all", which flips the unfused ffn.2
//     torch mm the same way).
//   Epilogue: out = round_h(x + (O + b2)) in fp32 (the unfused path
//     rounded ffn.2's output to fp16 first: one rounding fewer here); LN
//     statistics on the ROUNDED out, exactly like ln_fwd_add_h_kernel
//     (E[x^2] - mu^2, clamped), hn rounded once.
//   Every float op outside the MMAs is an explicit __f*_rn intrinsic: no
//   contraction freedom for nvcc.
//
// Determinism / r1: a row's result depends only on that row's inputs and
// the weights — every MMA chain runs in a fixed k order and the quad
// reductions are commutative pairs — so it is identical whatever R is and
// wherever the row falls in a tile. The grad-mode call (hid / mask
// non-null) is the SAME kernel binary: the extra stores sit behind
// warp-uniform runtime branches, the arithmetic is untouched. Hence the
// rollout (grad-free, N = 32768) and the update (grad mode, micro-batch)
// forwards agree bitwise at identical weights and obs.
//
// Grad mode also writes the hidden (R, 768) fp16 (ffn.2's wgrad input) and
// the ReLU bitmask in cuBLASLt's RELU_AUX layout — (R, 96) bytes, element
// i of row j at byte j * 96 + i / 8, bit i % 8 (ld = 768 bits, see
// fp16_gemm.cu relu_mask_ld) — so the backward's ffn.2 dgrad runs the same
// DRELU-epilogue GEMM (fp16_gemm.cu dgrad_drelu) as the unfused chain. The
// bit is (relu input > 0), as cuBLASLt's.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int D = 192, HID = 768;
constexpr int BM = 128;              // rows per tile
constexpr int NW = BM / 16;          // warps: each owns 16 rows
constexpr int NT = 32 * NW;          // 256 threads
constexpr int HC = 32;               // hidden columns per chunk
constexpr int NCH = HID / HC;        // 24 chunks
constexpr int KS1 = D / 16;          // GEMM1 k-steps (K = 192)
constexpr int NT1 = HC / 8;          // GEMM1 n-tiles per chunk
constexpr int KS2 = HC / 16;         // GEMM2 k-steps per chunk
constexpr int NT2 = D / 8;           // GEMM2 n-tiles (192 outputs)
constexpr int LDS0 = D + 8;          // W0 chunk row stride (halves): 400 B
constexpr int LDS2 = HC + 8;         // W2 chunk row stride (halves): 80 B
constexpr int W0C = HC * LDS0;       // halves
constexpr int W2C = D * LDS2;        // halves
constexpr int STAGE = W0C + W2C;     // halves per pipeline stage
constexpr int NSTAGE = 3;
constexpr int MASK_B = HID / 8;      // bitmask bytes per row (Lt ld = 768)
constexpr int SMEM = NSTAGE * STAGE * 2 + (HID + 3 * D) * 4;

static_assert(NT1 % 2 == 0 && NT2 % 2 == 0, "ldmatrix.x4 pairs n-tiles");
static_assert((LDS0 * 2) % 16 == 0 && (LDS2 * 2) % 16 == 0,
              "ldmatrix rows must stay 16-byte aligned");
static_assert((W0C * 2) % 16 == 0 && (STAGE * 2) % 16 == 0,
              "stage bases must stay 16-byte aligned");
static_assert(HC * (D / 8) % NT == 0 && D * (HC / 8) % NT == 0,
              "chunk copy splits evenly over the CTA");
static_assert(HC == 32, "the per-chunk mask word is 4 bytes / row");

struct FfnP {
  const __half* x;     // (R, D) residual stream in
  const __half* h2;    // (R, D) LN'd FFN input
  const __half* w0;    // (HID, D) fp16 replica
  const float* b0;     // (HID) fp32 master
  const __half* w2;    // (D, HID) fp16 replica
  const float* b2;     // (D) fp32 master
  const float* lnw;    // (D) next LN weight (LN variant)
  const float* lnb;    // (D) next LN bias
  __half* xo;          // (R, D) residual stream out
  __half* ho;          // (R, D) LN output (LN variant)
  float* mean;         // (R) nullable: grad mode, LN variant
  float* rstd;         // (R) nullable
  __half* hid;         // (R, HID) nullable: grad mode
  uint8_t* mask;       // (R, MASK_B) nullable: grad mode
  int64_t rows;
  float eps;
};

// ---- PTX wrappers (the small_attn.cu idiom) --------------------------------
__device__ __forceinline__ void ldsm4(uint32_t* r, const __half* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

// c += a (16x16, row) * b (16x8, col), fp16 in, fp32 accumulate
__device__ __forceinline__ void mma_f32(float* c, const uint32_t* a,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// c += a * b, fp16 in, fp16 accumulate (c[0] = row g, c[1] = row g + 8,
// each a half2 of columns 2t, 2t + 1)
__device__ __forceinline__ void mma_f16(uint32_t* c, const uint32_t* a,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(c[0]), "+r"(c[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t h2u(__half2 h) {
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ __half2 u2h(uint32_t u) {
  return *reinterpret_cast<__half2*>(&u);
}

__device__ __forceinline__ void cp16(void* s, const void* g) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(s));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(a), "l"(g) : "memory");
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p));
}

// One m16n8 accumulator tile: fp32 (4 regs) or fp16 (2 half2 regs).
// get(hf) = (col 2t, col 2t + 1) of row g + 8 hf as fp32.
template <bool H16>
struct Acc;

template <>
struct Acc<false> {
  float c[4];
  __device__ __forceinline__ void zero() { c[0] = c[1] = c[2] = c[3] = 0.f; }
  __device__ __forceinline__ void mma(const uint32_t* a, uint32_t b0,
                                      uint32_t b1) {
    mma_f32(c, a, b0, b1);
  }
  __device__ __forceinline__ float2 get(int hf) const {
    return make_float2(c[2 * hf], c[2 * hf + 1]);
  }
};

template <>
struct Acc<true> {
  uint32_t c[2];
  __device__ __forceinline__ void zero() { c[0] = c[1] = 0u; }  // +0.0 x2
  __device__ __forceinline__ void mma(const uint32_t* a, uint32_t b0,
                                      uint32_t b1) {
    mma_f16(c, a, b0, b1);
  }
  __device__ __forceinline__ float2 get(int hf) const {
    return __half22float2(u2h(c[hf]));
  }
};

// W0 rows [c*HC, c*HC + HC) and W2 columns [c*HC, c*HC + HC) -> one stage
__device__ __forceinline__ void load_chunk(__half* st, const __half* w0,
                                           const __half* w2, int c,
                                           int tid) {
  __half* s0 = st;
  __half* s2 = st + W0C;
  const __half* g0 = w0 + (int64_t)c * HC * D;     // contiguous HC x D
#pragma unroll
  for (int u = tid; u < HC * (D / 8); u += NT) {
    const int r = u / (D / 8), k = u - r * (D / 8);
    cp16(s0 + r * LDS0 + k * 8, g0 + r * D + k * 8);
  }
  const __half* g2 = w2 + c * HC;                  // D rows, stride HID
#pragma unroll
  for (int u = tid; u < D * (HC / 8); u += NT) {
    const int r = u / (HC / 8), k = u - r * (HC / 8);
    cp16(s2 + r * LDS2 + k * 8, g2 + (int64_t)r * HID + k * 8);
  }
}

template <bool UP16, bool DN16, bool LN>
__global__ void __launch_bounds__(NT, 1)
ffn_fused_kernel(FfnP p, int64_t ntiles) {
  extern __shared__ __align__(16) unsigned char smem[];
  __half* sw = reinterpret_cast<__half*>(smem);
  float* sB0 = reinterpret_cast<float*>(smem + (size_t)NSTAGE * STAGE * 2);
  float* sB2 = sB0 + HID;
  float* sLW = sB2 + D;
  float* sLB = sLW + D;
  const int tid = threadIdx.x;
  const int w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  // this CTA's tiles: blockIdx.x, blockIdx.x + gridDim.x, ... (grid <= ntiles)
  // (the host bounds ntiles so the chunk counters below fit an int)
  const int my_tiles =
      (int)((ntiles - blockIdx.x + gridDim.x - 1) / gridDim.x);
  const int q_total = my_tiles * NCH;              // weight chunks to stream

  // prologue: two chunks in flight before anything else (my_tiles >= 1,
  // so q_total >= NCH > 2)
  load_chunk(sw, p.w0, p.w2, 0, tid);
  cp_commit();
  load_chunk(sw + STAGE, p.w0, p.w2, 1, tid);
  cp_commit();
  for (int i = tid; i < HID; i += NT) sB0[i] = p.b0[i];
  for (int i = tid; i < D; i += NT) {
    sB2[i] = p.b2[i];
    if constexpr (LN) {
      sLW[i] = p.lnw[i];
      sLB[i] = p.lnb[i];
    }
  }
  // (visibility of the sB* stores: the first chunk's __syncthreads below)

#pragma unroll 1
  for (int j = 0; j < my_tiles; j++) {
    const int64_t tile = blockIdx.x + (int64_t)j * gridDim.x;
    const int64_t rbase = tile * BM + w * 16 + g;  // rows rbase, rbase + 8
    const bool ok0 = rbase < p.rows, ok1 = rbase + 8 < p.rows;
    // next tile's activations toward L2 while this one computes (its rows
    // are one contiguous block of 384-B rows = 3 x 128-B lines each)
    {
      const int64_t nt = tile + gridDim.x;
      if (nt < ntiles) {
        const int lines = 3 * (int)min((int64_t)BM, p.rows - nt * BM);
        const char* hb = reinterpret_cast<const char*>(p.h2 + nt * BM * D);
        const char* xb = reinterpret_cast<const char*>(p.x + nt * BM * D);
        for (int i = tid; i < lines; i += NT) {
          prefetch_l2(hb + i * 128);
          prefetch_l2(xb + i * 128);
        }
      }
    }
    // GEMM1 A fragments: the m16n8k16 A layout, straight from global
    // (rows past the end are zero — their results are never stored)
    uint32_t a1[KS1][4];
    {
      const __half* q0 = p.h2 + (ok0 ? rbase : 0) * D + t * 2;
      const __half* q1 = p.h2 + (ok1 ? rbase + 8 : 0) * D + t * 2;
#pragma unroll
      for (int ks = 0; ks < KS1; ks++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const bool ok = (e & 1) ? ok1 : ok0;
          const __half* q = (e & 1) ? q1 : q0;
          a1[ks][e] = ok ? *reinterpret_cast<const uint32_t*>(
                               q + ks * 16 + (e >> 1) * 8)
                         : 0u;
        }
    }
    Acc<DN16> o[NT2];
#pragma unroll
    for (int n = 0; n < NT2; n++) o[n].zero();

#pragma unroll 1
    for (int c = 0; c < NCH; c++) {
      const int q = j * NCH + c;
      cp_wait<1>();        // chunk q landed (only q + 1 may be pending)
      __syncthreads();     // ... for every thread; and chunk q - 1's stage
                           // is free: every warp finished computing it
      if (q + 2 < q_total)
        load_chunk(sw + ((q + 2) % NSTAGE) * STAGE, p.w0, p.w2,
                   (q + 2) % NCH, tid);
      cp_commit();         // (possibly empty: keeps the group count uniform)
      const __half* s0 = sw + (q % NSTAGE) * STAGE;
      const __half* s2 = s0 + W0C;

      // ---- GEMM1: S (16 x HC) = A1 (16 x D) . W0c^T, B ring depth 1
      Acc<UP16> s[NT1];
#pragma unroll
      for (int n = 0; n < NT1; n++) s[n].zero();
      {
        constexpr int NB = KS1 * (NT1 / 2);
        uint32_t bq[2][4];
        auto baddr = [&](int i) {
          const int ks = i / (NT1 / 2), np = i % (NT1 / 2);
          return s0 + (np * 16 + (lane & 7) + (lane >> 4) * 8) * LDS0 +
                 ks * 16 + ((lane >> 3) & 1) * 8;
        };
        ldsm4(bq[0], baddr(0));
#pragma unroll
        for (int i = 0; i < NB; i++) {
          if (i + 1 < NB) ldsm4(bq[(i + 1) & 1], baddr(i + 1));
          const int ks = i / (NT1 / 2), np = i % (NT1 / 2);
          s[2 * np].mma(a1[ks], bq[i & 1][0], bq[i & 1][1]);
          s[2 * np + 1].mma(a1[ks], bq[i & 1][2], bq[i & 1][3]);
        }
      }
      // first GEMM2 B fragment ahead of the ReLU epilogue's ALU work
      auto baddr2 = [&](int i) {
        const int kb = i / (NT2 / 2), np = i % (NT2 / 2);
        return s2 + (np * 16 + (lane & 7) + (lane >> 4) * 8) * LDS2 +
               kb * 16 + ((lane >> 3) & 1) * 8;
      };
      uint32_t bq2[2][4];
      ldsm4(bq2[0], baddr2(0));

      // ---- bias + ReLU -> fp16 hidden = GEMM2 A fragments (acc_to_a
      // layout: n-tile n is k-block n / 2, half (n & 1); row half hf)
      uint32_t a2[KS2][4];
      uint32_t mw[2] = {0u, 0u};            // mask word, rows g / g + 8
#pragma unroll
      for (int n = 0; n < NT1; n++) {
        const int col = c * HC + n * 8 + t * 2;
        const float2 bb = *reinterpret_cast<const float2*>(sB0 + col);
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
          const float2 v = s[n].get(hf);
          const float y0 = __fadd_rn(v.x, bb.x);
          const float y1 = __fadd_rn(v.y, bb.y);
          const __half2 hv =
              __floats2half2_rn(fmaxf(y0, 0.f), fmaxf(y1, 0.f));
          a2[n >> 1][(n & 1) * 2 + hf] = h2u(hv);
          mw[hf] |= ((y0 > 0.f ? 1u : 0u) | (y1 > 0.f ? 2u : 0u))
                    << (n * 8 + t * 2);
        }
      }
      if (p.hid) {                          // grad mode (warp-uniform)
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
          const int64_t row = rbase + 8 * hf;
          if (hf ? ok1 : ok0) {
            __half* hr = p.hid + row * HID + c * HC + t * 2;
#pragma unroll
            for (int n = 0; n < NT1; n++)
              *reinterpret_cast<uint32_t*>(hr + n * 8) =
                  a2[n >> 1][(n & 1) * 2 + hf];
          }
        }
      }
      if (p.mask) {                         // grad mode (warp-uniform)
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
          uint32_t m = mw[hf];
          m |= __shfl_xor_sync(0xffffffffu, m, 1);
          m |= __shfl_xor_sync(0xffffffffu, m, 2);
          const int64_t row = rbase + 8 * hf;
          if (t == 0 && (hf ? ok1 : ok0))
            *reinterpret_cast<uint32_t*>(p.mask + row * MASK_B + c * 4) = m;
        }
      }

      // ---- GEMM2: O (16 x D) += Hc (16 x HC) . W2c^T, B ring depth 1
      {
        constexpr int NB = KS2 * (NT2 / 2);
#pragma unroll
        for (int i = 0; i < NB; i++) {
          if (i + 1 < NB) ldsm4(bq2[(i + 1) & 1], baddr2(i + 1));
          const int kb = i / (NT2 / 2), np = i % (NT2 / 2);
          o[2 * np].mma(a2[kb], bq2[i & 1][0], bq2[i & 1][1]);
          o[2 * np + 1].mma(a2[kb], bq2[i & 1][2], bq2[i & 1][3]);
        }
      }
    }

    // ---- epilogue: residual add (+ next LayerNorm), rows g / g + 8
    constexpr float inv_d = 1.f / D;
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
      // compiler-only fence: without it nvcc hoists both row halves'
      // loads (x, b2, ln_w / ln_b) to the top and the LN variants spill
      // (NVRTC 12.8 / ptxas sm_120: 254 regs + 32 B spill -> 226, none)
      asm volatile("" ::: "memory");
      const bool ok = hf ? ok1 : ok0;
      const int64_t row = rbase + 8 * hf;
      const int64_t rr = ok ? row : 0;
      const __half* xr = p.x + rr * D + t * 2;
      __half* xw = p.xo + rr * D + t * 2;
      // the ROUNDED sums, kept as half2 (24 regs, not 48 — the fp32 value
      // is recovered exactly by __half22float2)
      uint32_t vh[NT2];
      float sm = 0.f, ss = 0.f;
#pragma unroll
      for (int n = 0; n < NT2; n++) {
        const float2 xv =
            ok ? __half22float2(*reinterpret_cast<const __half2*>(xr + n * 8))
               : make_float2(0.f, 0.f);
        const float2 ov = o[n].get(hf);
        const float2 bb =
            *reinterpret_cast<const float2*>(sB2 + n * 8 + t * 2);
        float2 u;
        u.x = __fadd_rn(xv.x, __fadd_rn(ov.x, bb.x));
        u.y = __fadd_rn(xv.y, __fadd_rn(ov.y, bb.y));
        const __half2 hv = __float22half2_rn(u);
        if (ok) *reinterpret_cast<__half2*>(xw + n * 8) = hv;
        vh[n] = h2u(hv);
        if constexpr (LN) {
          const float2 vr = __half22float2(hv);  // the value everyone sees
          sm = __fadd_rn(sm, __fadd_rn(vr.x, vr.y));
          ss = __fadd_rn(ss, __fmaf_rn(vr.x, vr.x, __fmul_rn(vr.y, vr.y)));
        }
      }
      if constexpr (LN) {
        // the quad (t = 0..3) holds the row: commutative pair sums, so all
        // four lanes end with identical bits
        sm = __fadd_rn(sm, __shfl_xor_sync(0xffffffffu, sm, 1));
        sm = __fadd_rn(sm, __shfl_xor_sync(0xffffffffu, sm, 2));
        ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 1));
        ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 2));
        const float mu = __fmul_rn(sm, inv_d);
        const float var =
            fmaxf(__fsub_rn(__fmul_rn(ss, inv_d), __fmul_rn(mu, mu)), 0.f);
        const float r = rsqrtf(__fadd_rn(var, p.eps));
        if (p.mean && t == 0 && ok) {
          p.mean[row] = mu;
          p.rstd[row] = r;
        }
        __half* hw = p.ho + rr * D + t * 2;
#pragma unroll
        for (int n = 0; n < NT2; n++) {
          const int col = n * 8 + t * 2;
          const float2 wv = *reinterpret_cast<const float2*>(sLW + col);
          const float2 bv = *reinterpret_cast<const float2*>(sLB + col);
          const float2 vr = __half22float2(u2h(vh[n]));
          float2 y;
          y.x = __fmaf_rn(__fmul_rn(__fsub_rn(vr.x, mu), r), wv.x, bv.x);
          y.y = __fmaf_rn(__fmul_rn(__fsub_rn(vr.y, mu), r), wv.y, bv.y);
          if (ok) *reinterpret_cast<__half2*>(hw + n * 8) =
              __float22half2_rn(y);
        }
      }
    }
  }
  cp_wait<0>();
}

void check_act(const at::Tensor& t, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kHalf && t.dim() == 2
              && t.size(1) == D && t.is_contiguous()
              && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
              n, ": contiguous 16-byte-aligned (R, 192) CUDA fp16");
}

void check_vec(const at::Tensor& t, const char* n, int64_t len) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat
              && t.is_contiguous() && t.numel() == len
              && reinterpret_cast<uintptr_t>(t.data_ptr()) % 8 == 0,
              n, ": contiguous fp32 (", len, ",) CUDA, 8-byte aligned");
}

void check_w(const at::Tensor& t, const char* n, int64_t r, int64_t c) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kHalf && t.dim() == 2
              && t.size(0) == r && t.size(1) == c && t.is_contiguous()
              && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
              n, ": contiguous 16-byte-aligned (", r, ", ", c,
              ") CUDA fp16");
}

template <bool U, bool DN, bool L>
void launch(const FfnP& p, int64_t ntiles, cudaStream_t stream) {
  static int grid_cap = 0;           // per instantiation (one GPU/process)
  if (grid_cap == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ffn_fused_kernel<U, DN, L>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    int per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, ffn_fused_kernel<U, DN, L>, NT, SMEM));
    TORCH_CHECK(per_sm >= 1, "ffn_fused: kernel does not fit on an SM");
    grid_cap = per_sm * at::cuda::getCurrentDeviceProperties()
                            ->multiProcessorCount;
  }
  const int grid = (int)std::min<int64_t>(ntiles, grid_cap);
  ffn_fused_kernel<U, DN, L><<<grid, NT, SMEM, stream>>>(p, ntiles);
}

}  // namespace

// True when this device can run the kernel (opt-in shared memory >= the
// 89,856 B the pipeline needs, mma.sync m16n8k16 / cp.async: sm_80+).
bool ffn_fused_supported() {
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  int optin = 0, major = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, dev));
  return major >= 8 && optin >= SMEM;
}

// [xo, ho, mean, rstd, hid, mask]:
//   xo   = x + relu(h2 w0^T + b0) w2^T + b2          (R, 192) fp16
//   ho   = LayerNorm(xo; lnw, lnb, eps)               (R, 192) fp16, or
//          None when lnw is None (residual-only variant)
//   mean, rstd (R) fp32   save and LN only, else None
//   hid (R, 768) fp16, mask (R, 96) uint8 (cuBLASLt RELU_AUX layout)
//                         save only, else None
// acc: 0 = fp32 accumulate both GEMMs, 1 = fp16-accumulate the down-
// projection, 2 = fp16-accumulate both (experimental).
std::vector<at::Tensor> ffn_fused(at::Tensor x, at::Tensor h2, at::Tensor w0,
                                  at::Tensor b0, at::Tensor w2, at::Tensor b2,
                                  c10::optional<at::Tensor> lnw_opt,
                                  c10::optional<at::Tensor> lnb_opt,
                                  double eps, int64_t acc, bool save) {
  const c10::cuda::CUDAGuard guard(x.device());
  check_act(x, "x");
  check_act(h2, "h2");
  TORCH_CHECK(x.size(0) == h2.size(0), "x / h2 row mismatch");
  check_w(w0, "w0", HID, D);
  check_w(w2, "w2", D, HID);
  check_vec(b0, "b0", HID);
  check_vec(b2, "b2", D);
  const at::Tensor lnw = lnw_opt.value_or(at::Tensor());
  const at::Tensor lnb = lnb_opt.value_or(at::Tensor());
  const bool ln = lnw.defined();
  TORCH_CHECK(ln == lnb.defined(), "lnw / lnb: both or neither");
  if (ln) {
    check_vec(lnw, "lnw", D);
    check_vec(lnb, "lnb", D);
  }
  TORCH_CHECK(acc >= 0 && acc <= 2, "acc must be 0, 1 or 2");
  const int64_t R = x.size(0);
  auto xo = at::empty_like(x);
  at::Tensor ho, mean, rstd, hid, mask;
  if (ln) ho = at::empty_like(x);
  if (save) {
    hid = at::empty({R, HID}, x.options());
    mask = at::empty({R, MASK_B}, x.options().dtype(at::kByte));
    if (ln) {
      mean = at::empty({R}, x.options().dtype(at::kFloat));
      rstd = at::empty({R}, x.options().dtype(at::kFloat));
    }
  }
  if (R == 0) return {xo, ho, mean, rstd, hid, mask};
  FfnP p;
  p.x = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  p.h2 = reinterpret_cast<const __half*>(h2.data_ptr<at::Half>());
  p.w0 = reinterpret_cast<const __half*>(w0.data_ptr<at::Half>());
  p.b0 = b0.data_ptr<float>();
  p.w2 = reinterpret_cast<const __half*>(w2.data_ptr<at::Half>());
  p.b2 = b2.data_ptr<float>();
  p.lnw = ln ? lnw.data_ptr<float>() : nullptr;
  p.lnb = ln ? lnb.data_ptr<float>() : nullptr;
  p.xo = reinterpret_cast<__half*>(xo.data_ptr<at::Half>());
  p.ho = ln ? reinterpret_cast<__half*>(ho.data_ptr<at::Half>()) : nullptr;
  p.mean = mean.defined() ? mean.data_ptr<float>() : nullptr;
  p.rstd = rstd.defined() ? rstd.data_ptr<float>() : nullptr;
  p.hid = save ? reinterpret_cast<__half*>(hid.data_ptr<at::Half>())
               : nullptr;
  p.mask = save ? mask.data_ptr<uint8_t>() : nullptr;
  p.rows = R;
  p.eps = (float)eps;
  const int64_t ntiles = (R + BM - 1) / BM;
  TORCH_CHECK(ntiles <= INT32_MAX / NCH, "ffn_fused: too many rows");
  auto stream = at::cuda::getCurrentCUDAStream();
  if (acc == 0) {
    if (ln) launch<false, false, true>(p, ntiles, stream);
    else launch<false, false, false>(p, ntiles, stream);
  } else if (acc == 1) {
    if (ln) launch<false, true, true>(p, ntiles, stream);
    else launch<false, true, false>(p, ntiles, stream);
  } else {
    if (ln) launch<true, true, true>(p, ntiles, stream);
    else launch<true, true, false>(p, ntiles, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {xo, ho, mean, rstd, hid, mask};
}
