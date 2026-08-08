// Fused LayerNorm for the policy's small-row shapes (d = embed, 64/128;
// rows = B*T). torch's LayerNorm at d=64 was measured at 37 GB/s vs 218
// achievable on the GTX 1080 (2026-07-25 survey) — its block-per-row
// launch leaves >80% of the machine idle at tiny d. Here one WARP owns a
// row (d <= 256): shuffle reductions, no shared memory on the forward,
// fully coalesced row traffic.
//
// Backward is a single pass: dx plus per-block dw/db partials staged in
// shared memory, reduced to a fixed-size (blocks, d) buffer summed with
// a deterministic torch reduction — NO cross-block float atomics, so
// gradients are bit-reproducible run to run (unlike the fused policy
// trainer's striped-atomic path, whose jitter is documented).
//
// Math notes (fp32 end to end):
//   fwd: mu = E[x], var = E[x^2] - mu^2 (clamped >= 0), r = rsqrt(var+eps)
//        y = (x - mu) * r * w + b            (saves mu, r per row)
//   bwd: xh = (x - mu) * r, gw = dy * w
//        dx = r * (gw - mean(gw) - xh * mean(gw * xh))
//        dw = sum_rows dy * xh,  db = sum_rows dy
// The E[x^2]-mu^2 variance differs from torch's Welford at ~1e-7 —
// within every parity tolerance in the test suite.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <vector>

namespace {

constexpr int WARP = 32;
constexpr int THREADS = 256;
constexpr int WPB = THREADS / WARP;  // warps (rows in flight) per block
constexpr int MAX_D = 256;           // smem budget: 2 * WPB * MAX_D floats
constexpr int MAX_COLS = MAX_D / WARP;

__inline__ __device__ float warp_allsum(float v) {
  for (int o = WARP / 2; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// NOT column-cached / NOT template-specialized, deliberately: the same
// register-cache rewrite the fp16 kernels below carry was tried on these
// two on 2026-08-02 and SHIFTED THE fp32 BITS (rstd differed on ~0.4% of
// rows, y up to 9.5e-7) — pinning d at compile time fully unrolls the
// stats loop, which lets nvcc contract `ss += v * v` into an FMA. The
// fp32 path's seed-0 bit-exactness is the record protocol's regression
// detector, and the measured prize was ln_fwd 754 -> 692 us on a shape
// the fp32 path barely uses. Left alone on purpose.
__global__ void ln_fwd_kernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              const float* __restrict__ b,
                              float* __restrict__ y,
                              float* __restrict__ mean,
                              float* __restrict__ rstd,
                              int64_t rows, int d, float eps) {
  const int lane = threadIdx.x % WARP;
  const int wid = threadIdx.x / WARP;
  int64_t row = (int64_t)blockIdx.x * WPB + wid;
  const int64_t stride = (int64_t)gridDim.x * WPB;
  for (; row < rows; row += stride) {
    const float* xr = x + row * d;
    float s = 0.f, ss = 0.f;
    for (int c = lane; c < d; c += WARP) {
      const float v = xr[c];
      s += v;
      ss += v * v;
    }
    s = warp_allsum(s);
    ss = warp_allsum(ss);
    const float mu = s / d;
    const float r = rsqrtf(fmaxf(ss / d - mu * mu, 0.f) + eps);
    if (lane == 0) {
      mean[row] = mu;
      rstd[row] = r;
    }
    float* yr = y + row * d;
    for (int c = lane; c < d; c += WARP)
      yr[c] = (xr[c] - mu) * r * w[c] + b[c];
  }
}

// Fused residual-add + LayerNorm: x = a + b (written out — it stays the
// residual stream AND the saved backward input), y = LN(x). Saves the
// separate add kernel's full read+read+write pass at the policy's three
// residual->LN sites. The second column pass re-reads xr[c] the same
// thread just wrote (L1/L2-hot) — bit-identical to re-adding a+b.
// (Register-caching it, as the fp16 twin does, needs the compile-time d
// that shifts this kernel's fp32 bits — see ln_fwd_kernel above.)
__global__ void ln_fwd_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  const float* __restrict__ w,
                                  const float* __restrict__ bias,
                                  float* __restrict__ x,
                                  float* __restrict__ y,
                                  float* __restrict__ mean,
                                  float* __restrict__ rstd,
                                  int64_t rows, int d, float eps) {
  const int lane = threadIdx.x % WARP;
  const int wid = threadIdx.x / WARP;
  int64_t row = (int64_t)blockIdx.x * WPB + wid;
  const int64_t stride = (int64_t)gridDim.x * WPB;
  for (; row < rows; row += stride) {
    const float* ar = a + row * d;
    const float* br = b + row * d;
    float* xr = x + row * d;
    float s = 0.f, ss = 0.f;
    for (int c = lane; c < d; c += WARP) {
      const float v = ar[c] + br[c];
      xr[c] = v;
      s += v;
      ss += v * v;
    }
    s = warp_allsum(s);
    ss = warp_allsum(ss);
    const float mu = s / d;
    const float r = rsqrtf(fmaxf(ss / d - mu * mu, 0.f) + eps);
    if (lane == 0) {
      mean[row] = mu;
      rstd[row] = r;
    }
    float* yr = y + row * d;
    for (int c = lane; c < d; c += WARP)
      yr[c] = (xr[c] - mu) * r * w[c] + bias[c];
  }
}

// ---- fp16-storage variants (2026-07-25 fp16 project, stage 1) --------------
// __half2 I/O, ALL arithmetic in fp32 (GP104 native half math is 1/64
// rate and never used): stats, affine, and dw/db partials accumulate in
// fp32; activations round exactly once on store. Lane-owns-one-half2
// keeps warp transactions at a full 128 B sector (scalar half loads
// would halve effective bandwidth). COLS2_C counts half2 columns per
// lane: 1 -> d=64, 2 -> d=128, 0 -> runtime d.
// The fp32 kernels above are UNTOUCHED — the seed-0 bit-exactness of
// the fp32 path is a regression detector this file must not disturb.

template <int COLS2_C>
__global__ void ln_fwd_h_kernel(const __half2* __restrict__ x,
                                const float2* __restrict__ w,
                                const float2* __restrict__ b,
                                __half2* __restrict__ y,
                                float* __restrict__ mean,
                                float* __restrict__ rstd,
                                int64_t rows, int d2, float eps) {
  if (COLS2_C > 0) d2 = COLS2_C * WARP;
  const float inv_d = 1.f / (2 * d2);
  const int lane = threadIdx.x % WARP;
  const int wid = threadIdx.x / WARP;
  int64_t row = (int64_t)blockIdx.x * WPB + wid;
  const int64_t stride = (int64_t)gridDim.x * WPB;
  for (; row < rows; row += stride) {
    const __half2* xr = x + row * d2;
    float s = 0.f, ss = 0.f;
    // Per-ROW register cache of this lane's activation columns (declared
    // INSIDE the grid-stride loop — each row refills it). Compile-time
    // indexed at COLS2_C > 0, so it stays in registers (build log:
    // 0 bytes stack frame); the affine pass below then never re-reads
    // the row. Values are identical (__half22float2 is exact), so this
    // is bitwise. COLS2_C == 0 is the generic runtime-d fallback: no
    // compile-time bound, so it keeps re-reading rather than spilling a
    // runtime-indexed array to local memory.
    float2 cache[COLS2_C > 0 ? COLS2_C : 1];
#pragma unroll
    for (int c = lane, k = 0; c < d2; c += WARP, k++) {
      const float2 v = __half22float2(xr[c]);
      if constexpr (COLS2_C > 0) cache[k] = v;
      s += v.x + v.y;
      ss += v.x * v.x + v.y * v.y;
    }
    s = warp_allsum(s);
    ss = warp_allsum(ss);
    const float mu = s * inv_d;
    const float r = rsqrtf(fmaxf(ss * inv_d - mu * mu, 0.f) + eps);
    if (lane == 0) {
      mean[row] = mu;
      rstd[row] = r;
    }
    __half2* yr = y + row * d2;
#pragma unroll
    for (int c = lane, k = 0; c < d2; c += WARP, k++) {
      float2 v;
      if constexpr (COLS2_C > 0) v = cache[k];
      else v = __half22float2(xr[c]);
      const float2 wv = w[c], bv = b[c];
      float2 o;
      o.x = (v.x - mu) * r * wv.x + bv.x;
      o.y = (v.y - mu) * r * wv.y + bv.y;
      yr[c] = __float22half2_rn(o);
    }
  }
}

// x = round_h(a + b) written out; stats computed on the ROUNDED value so
// forward normalization and the backward's xh reconstruction (from the
// saved fp16 x) see the identical tensor.
//
// bb (nullable, fp32 (d,)): a per-channel bias for the b operand, added
// in this pass's fp32 arithmetic as a + (b + bb) — the fp16 policy's
// ffn.2 then runs bias-FREE (the sm_61 addmm bias epilogue costs 33-59%
// of the GEMM, round-4 #2) and its bias rides here instead. This MOVES
// the bias's rounding position (fp32 sum rounded once, instead of the
// GEMM epilogue rounding mm+bias to fp16 first), i.e. near-tie class —
// the host rejects bb on the fp32 path, whose bits are the record
// protocol's regression detector. bb == nullptr is bitwise unchanged
// (the add is branched around, not multiplied by zero).
template <int COLS2_C>
__global__ void ln_fwd_add_h_kernel(const __half2* __restrict__ a,
                                    const __half2* __restrict__ b_in,
                                    const float2* __restrict__ w,
                                    const float2* __restrict__ b,
                                    const float2* __restrict__ bb,
                                    __half2* __restrict__ x,
                                    __half2* __restrict__ y,
                                    float* __restrict__ mean,
                                    float* __restrict__ rstd,
                                    int64_t rows, int d2, float eps) {
  if (COLS2_C > 0) d2 = COLS2_C * WARP;
  const float inv_d = 1.f / (2 * d2);
  const int lane = threadIdx.x % WARP;
  const int wid = threadIdx.x / WARP;
  int64_t row = (int64_t)blockIdx.x * WPB + wid;
  const int64_t stride = (int64_t)gridDim.x * WPB;
  for (; row < rows; row += stride) {
    const __half2* ar = a + row * d2;
    const __half2* br = b_in + row * d2;
    __half2* xr = x + row * d2;
    float s = 0.f, ss = 0.f;
    // per-row register cache of the ROUNDED sum (see ln_fwd_h_kernel):
    // the affine pass used to re-read the hv it had just written
    float2 cache[COLS2_C > 0 ? COLS2_C : 1];
#pragma unroll
    for (int c = lane, k = 0; c < d2; c += WARP, k++) {
      const float2 va = __half22float2(ar[c]);
      const float2 vb = __half22float2(br[c]);
      float2 v;
      if (bb) {
        const float2 vbb = bb[c];
        v.x = va.x + (vb.x + vbb.x);
        v.y = va.y + (vb.y + vbb.y);
      } else {
        v.x = va.x + vb.x;
        v.y = va.y + vb.y;
      }
      const __half2 hv = __float22half2_rn(v);
      xr[c] = hv;
      const float2 vr = __half22float2(hv);  // the value everyone else sees
      if constexpr (COLS2_C > 0) cache[k] = vr;
      s += vr.x + vr.y;
      ss += vr.x * vr.x + vr.y * vr.y;
    }
    s = warp_allsum(s);
    ss = warp_allsum(ss);
    const float mu = s * inv_d;
    const float r = rsqrtf(fmaxf(ss * inv_d - mu * mu, 0.f) + eps);
    if (lane == 0) {
      mean[row] = mu;
      rstd[row] = r;
    }
    __half2* yr = y + row * d2;
#pragma unroll
    for (int c = lane, k = 0; c < d2; c += WARP, k++) {
      float2 v;
      if constexpr (COLS2_C > 0) v = cache[k];
      else v = __half22float2(xr[c]);
      const float2 wv = w[c], bv = b[c];
      float2 o;
      o.x = (v.x - mu) * r * wv.x + bv.x;
      o.y = (v.y - mu) * r * wv.y + bv.y;
      yr[c] = __float22half2_rn(o);
    }
  }
}

// gx (nullable): residual gradient added into dx in the store loop —
// the fused form of _LnAddFn.backward's `dx = dx + gx`. fp32: bitwise
// identical to the unfused add (same two operands, same order). fp16:
// the sum is formed in fp32 and rounded ONCE instead of round(dx) then
// a second rounded add — strictly more accurate on the residual path.
template <int COLS2_C>
__global__ void ln_bwd_h_kernel(const __half2* __restrict__ x,
                                const __half2* __restrict__ dy,
                                const float2* __restrict__ w,
                                const float* __restrict__ mean,
                                const float* __restrict__ rstd,
                                const __half2* __restrict__ gx,
                                __half2* __restrict__ dx,
                                float* __restrict__ pdw,  // (gridDim.x, d)
                                float* __restrict__ pdb,  // (gridDim.x, d)
                                int64_t rows, int d2) {
  extern __shared__ float sm[];  // [2][WPB][d]
  if (COLS2_C > 0) d2 = COLS2_C * WARP;
  const int d = 2 * d2;
  const float inv_d = 1.f / d;
  const int lane = threadIdx.x % WARP;
  const int wid = threadIdx.x / WARP;
  float acc_dw[COLS2_C > 0 ? 2 * COLS2_C : MAX_COLS] = {};
  float acc_db[COLS2_C > 0 ? 2 * COLS2_C : MAX_COLS] = {};
  int64_t row = (int64_t)blockIdx.x * WPB + wid;
  const int64_t stride = (int64_t)gridDim.x * WPB;
  for (; row < rows; row += stride) {
    const __half2* xr = x + row * d2;
    const __half2* gr = dy + row * d2;
    const float mu = mean[row];
    const float r = rstd[row];
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int c = lane, k = 0; c < d2; c += WARP, k++) {
      const float2 xv = __half22float2(xr[c]);
      const float2 gv = __half22float2(gr[c]);
      const float2 wv = w[c];
      const float xh0 = (xv.x - mu) * r, xh1 = (xv.y - mu) * r;
      const float gw0 = gv.x * wv.x, gw1 = gv.y * wv.y;
      s1 += gw0 + gw1;
      s2 += gw0 * xh0 + gw1 * xh1;
      acc_dw[2 * k] += gv.x * xh0;
      acc_dw[2 * k + 1] += gv.y * xh1;
      acc_db[2 * k] += gv.x;
      acc_db[2 * k + 1] += gv.y;
    }
    s1 = warp_allsum(s1) * inv_d;
    s2 = warp_allsum(s2) * inv_d;
    __half2* dxr = dx + row * d2;
    const __half2* gxr = gx ? gx + row * d2 : nullptr;
#pragma unroll
    for (int c = lane; c < d2; c += WARP) {
      const float2 xv = __half22float2(xr[c]);
      const float2 gv = __half22float2(gr[c]);
      const float2 wv = w[c];
      float2 o;
      o.x = (gv.x * wv.x - s1 - (xv.x - mu) * r * s2) * r;
      o.y = (gv.y * wv.y - s1 - (xv.y - mu) * r * s2) * r;
      if (gxr) {
        const float2 gxv = __half22float2(gxr[c]);
        o.x += gxv.x;
        o.y += gxv.y;
      }
      dxr[c] = __float22half2_rn(o);
    }
  }
#pragma unroll
  for (int c = lane, k = 0; c < d2; c += WARP, k++) {
    sm[wid * d + 2 * c] = acc_dw[2 * k];
    sm[wid * d + 2 * c + 1] = acc_dw[2 * k + 1];
    sm[(WPB + wid) * d + 2 * c] = acc_db[2 * k];
    sm[(WPB + wid) * d + 2 * c + 1] = acc_db[2 * k + 1];
  }
  __syncthreads();
  for (int c = threadIdx.x; c < d; c += blockDim.x) {
    float tdw = 0.f, tdb = 0.f;
    for (int i = 0; i < WPB; i++) {
      tdw += sm[i * d + c];
      tdb += sm[(WPB + i) * d + c];
    }
    pdw[(int64_t)blockIdx.x * d + c] = tdw;
    pdb[(int64_t)blockIdx.x * d + c] = tdb;
  }
}

// COLS_C > 0 pins d = COLS_C * WARP at compile time: the column loops
// fully unroll and acc_dw/acc_db live in REGISTERS. With runtime d
// (COLS_C == 0, the generic fallback) the runtime-indexed accumulators
// are forced to local memory — read-modify-written twice per column per
// row (~576 local accesses/thread at production shapes; the 2026-07-25
// profile put ln_bwd at 4.3% of the update).
template <int COLS_C>
__global__ void ln_bwd_kernel(const float* __restrict__ x,
                              const float* __restrict__ dy,
                              const float* __restrict__ w,
                              const float* __restrict__ mean,
                              const float* __restrict__ rstd,
                              const float* __restrict__ gx,
                              float* __restrict__ dx,
                              float* __restrict__ pdw,  // (gridDim.x, d)
                              float* __restrict__ pdb,  // (gridDim.x, d)
                              int64_t rows, int d) {
  extern __shared__ float sm[];  // [2][WPB][d]
  if (COLS_C > 0) d = COLS_C * WARP;  // constant-folds every loop below
  const int lane = threadIdx.x % WARP;
  const int wid = threadIdx.x / WARP;
  float acc_dw[COLS_C > 0 ? COLS_C : MAX_COLS] = {};
  float acc_db[COLS_C > 0 ? COLS_C : MAX_COLS] = {};
  int64_t row = (int64_t)blockIdx.x * WPB + wid;
  const int64_t stride = (int64_t)gridDim.x * WPB;
  for (; row < rows; row += stride) {
    const float* xr = x + row * d;
    const float* gr = dy + row * d;
    const float mu = mean[row];
    const float r = rstd[row];
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int c = lane, k = 0; c < d; c += WARP, k++) {
      const float xh = (xr[c] - mu) * r;
      const float g = gr[c];
      const float gw = g * w[c];
      s1 += gw;
      s2 += gw * xh;
      acc_dw[k] += g * xh;
      acc_db[k] += g;
    }
    s1 = warp_allsum(s1) / d;
    s2 = warp_allsum(s2) / d;
    float* dxr = dx + row * d;
    const float* gxr = gx ? gx + row * d : nullptr;
    for (int c = lane; c < d; c += WARP) {
      const float xh = (xr[c] - mu) * r;
      const float o = (gr[c] * w[c] - s1 - xh * s2) * r;
      dxr[c] = gxr ? o + gxr[c] : o;
    }
  }
  // block reduction of the weight/bias partials: each warp stages its
  // lane-owned columns, then the block sums across warps into this
  // block's row of the (blocks, d) partial buffers
#pragma unroll
  for (int c = lane, k = 0; c < d; c += WARP, k++) {
    sm[wid * d + c] = acc_dw[k];
    sm[(WPB + wid) * d + c] = acc_db[k];
  }
  __syncthreads();
  for (int c = threadIdx.x; c < d; c += blockDim.x) {
    float tdw = 0.f, tdb = 0.f;
    for (int i = 0; i < WPB; i++) {
      tdw += sm[i * d + c];
      tdb += sm[(WPB + i) * d + c];
    }
    pdw[(int64_t)blockIdx.x * d + c] = tdw;
    pdb[(int64_t)blockIdx.x * d + c] = tdb;
  }
}

int grid_for(int64_t rows, int cap) {
  return (int)std::min<int64_t>((rows + WPB - 1) / WPB, cap);
}

bool half_ok(const at::Tensor& x, int d) {
  return x.scalar_type() == at::kHalf && d % 2 == 0
      && (reinterpret_cast<uintptr_t>(x.data_ptr()) % 4) == 0;
}

std::vector<at::Tensor> ln_fwd(at::Tensor x, at::Tensor w, at::Tensor b,
                               double eps) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && x.is_contiguous()
              && (x.scalar_type() == at::kFloat
                  || x.scalar_type() == at::kHalf),
              "x: contiguous CUDA f32/f16");
  const int d = (int)x.size(-1);
  TORCH_CHECK(d <= MAX_D, "d > ", MAX_D, " unsupported");
  TORCH_CHECK(w.is_contiguous() && b.is_contiguous()
              && w.numel() == d && b.numel() == d
              && w.scalar_type() == at::kFloat
              && b.scalar_type() == at::kFloat, "w/b: fp32 contiguous");
  const int64_t rows = x.numel() / d;
  // stats are ALWAYS fp32, regardless of x dtype
  auto y = at::empty_like(x);
  auto mean = at::empty({rows}, x.options().dtype(at::kFloat));
  auto rstd = at::empty({rows}, x.options().dtype(at::kFloat));
  if (rows == 0) return {y, mean, rstd};
  if (x.scalar_type() == at::kFloat) {
    ln_fwd_kernel<<<grid_for(rows, 1024), THREADS, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        rows, d, (float)eps);
  } else {
    TORCH_CHECK(half_ok(x, d), "half path: even d, 4B-aligned");
    const int d2 = d / 2;
    auto launch = [&](auto kernel) {
      kernel<<<grid_for(rows, 1024), THREADS, 0,
               at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __half2*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const float2*>(w.data_ptr<float>()),
          reinterpret_cast<const float2*>(b.data_ptr<float>()),
          reinterpret_cast<__half2*>(y.data_ptr<at::Half>()),
          mean.data_ptr<float>(), rstd.data_ptr<float>(), rows, d2,
          (float)eps);
    };
    if (d == 64) launch(ln_fwd_h_kernel<1>);
    else if (d == 128) launch(ln_fwd_h_kernel<2>);
    else launch(ln_fwd_h_kernel<0>);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}

std::vector<at::Tensor> ln_fwd_add(at::Tensor a, at::Tensor b, at::Tensor w,
                                   at::Tensor bias, double eps,
                                   c10::optional<at::Tensor> b_bias_opt) {
  const at::Tensor bb = b_bias_opt.value_or(at::Tensor());
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(a.is_cuda() && a.is_contiguous() && b.is_contiguous()
              && (a.scalar_type() == at::kFloat
                  || a.scalar_type() == at::kHalf)
              && b.scalar_type() == a.scalar_type()
              && a.sizes() == b.sizes(),
              "a/b: same-shape same-dtype contiguous CUDA f32/f16");
  const int d = (int)a.size(-1);
  TORCH_CHECK(d <= MAX_D, "d > ", MAX_D, " unsupported");
  TORCH_CHECK(w.is_contiguous() && bias.is_contiguous()
              && w.numel() == d && bias.numel() == d
              && w.scalar_type() == at::kFloat
              && bias.scalar_type() == at::kFloat, "w/b: fp32 contiguous");
  if (bb.defined()) {
    // near-tie rounding change: fp16 stream only (see the kernel note)
    TORCH_CHECK(a.scalar_type() == at::kHalf,
                "b_bias: fp16 path only — the fp32 path must keep the "
                "bias in its GEMM epilogue (record-run bit-exactness)");
    TORCH_CHECK(bb.is_contiguous() && bb.scalar_type() == at::kFloat
                && bb.numel() == d, "b_bias: fp32 contiguous (d,)");
  }
  const int64_t rows = a.numel() / d;
  auto x = at::empty_like(a);
  auto y = at::empty_like(a);
  auto mean = at::empty({rows}, a.options().dtype(at::kFloat));
  auto rstd = at::empty({rows}, a.options().dtype(at::kFloat));
  if (rows == 0) return {x, y, mean, rstd};
  if (a.scalar_type() == at::kFloat) {
    ln_fwd_add_kernel<<<grid_for(rows, 1024), THREADS, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), w.data_ptr<float>(),
        bias.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(), rows, d,
        (float)eps);
  } else {
    TORCH_CHECK(half_ok(a, d) && half_ok(b, d),
                "half path: even d, 4B-aligned");
    const int d2 = d / 2;
    auto launch = [&](auto kernel) {
      kernel<<<grid_for(rows, 1024), THREADS, 0,
               at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __half2*>(a.data_ptr<at::Half>()),
          reinterpret_cast<const __half2*>(b.data_ptr<at::Half>()),
          reinterpret_cast<const float2*>(w.data_ptr<float>()),
          reinterpret_cast<const float2*>(bias.data_ptr<float>()),
          bb.defined()
              ? reinterpret_cast<const float2*>(bb.data_ptr<float>())
              : nullptr,
          reinterpret_cast<__half2*>(x.data_ptr<at::Half>()),
          reinterpret_cast<__half2*>(y.data_ptr<at::Half>()),
          mean.data_ptr<float>(), rstd.data_ptr<float>(), rows, d2,
          (float)eps);
    };
    if (d == 64) launch(ln_fwd_add_h_kernel<1>);
    else if (d == 128) launch(ln_fwd_add_h_kernel<2>);
    else launch(ln_fwd_add_h_kernel<0>);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x, y, mean, rstd};
}

std::vector<at::Tensor> ln_bwd(at::Tensor x, at::Tensor dy, at::Tensor w,
                               at::Tensor mean, at::Tensor rstd,
                               c10::optional<at::Tensor> gx_opt) {
  const at::Tensor gx = gx_opt.value_or(at::Tensor());
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && dy.is_contiguous()
              && (x.scalar_type() == at::kFloat
                  || x.scalar_type() == at::kHalf)
              && dy.scalar_type() == x.scalar_type());
  TORCH_CHECK(mean.scalar_type() == at::kFloat
              && rstd.scalar_type() == at::kFloat);
  const int d = (int)x.size(-1);
  TORCH_CHECK(d <= MAX_D, "d > ", MAX_D, " unsupported");
  if (gx.defined()) {
    TORCH_CHECK(gx.is_contiguous() && gx.scalar_type() == x.scalar_type()
                && gx.sizes() == x.sizes(),
                "gx: contiguous, same shape/dtype as x");
  }
  const int64_t rows = x.numel() / d;
  auto dx = at::empty_like(x);
  // fixed-size fp32 partials + deterministic torch sum: bit-reproducible
  // gradients (no cross-block float atomics); dw/db land in fp32 and
  // feed the fp32 LN leaves with no cast
  // cap 1024 probed at 2048/4096 on the RTX 4090 (2026-08-08, 3 ABAB
  // pairs, bench_policy_update --fp16 mb=16384): NOISE — 1/3 paired
  // wins, ranges overlap. The grid-stride loop hides the block count on
  // 128 SMs just as it did on 20. Left at 1024.
  const int blocks = std::max(grid_for(rows, 1024), 1);
  auto pdw = at::empty({blocks, d}, x.options().dtype(at::kFloat));
  auto pdb = at::empty({blocks, d}, x.options().dtype(at::kFloat));
  if (rows > 0) {
    const size_t smem = 2 * WPB * d * sizeof(float);
    if (x.scalar_type() == at::kFloat) {
      auto launch = [&](auto kernel) {
        kernel<<<blocks, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(), dy.data_ptr<float>(), w.data_ptr<float>(),
            mean.data_ptr<float>(), rstd.data_ptr<float>(),
            gx.defined() ? gx.data_ptr<float>() : nullptr,
            dx.data_ptr<float>(), pdw.data_ptr<float>(),
            pdb.data_ptr<float>(), rows, d);
      };
      // compile-time-cols variants for the policy's real widths
      if (d == 2 * WARP) launch(ln_bwd_kernel<2>);        // d = 64
      else if (d == 4 * WARP) launch(ln_bwd_kernel<4>);   // d = 128
      else launch(ln_bwd_kernel<0>);                      // runtime d
    } else {
      TORCH_CHECK(half_ok(x, d) && half_ok(dy, d)
                  && (!gx.defined() || half_ok(gx, d)),
                  "half path: even d, 4B-aligned");
      const int d2 = d / 2;
      auto launch = [&](auto kernel) {
        kernel<<<blocks, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half2*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const __half2*>(dy.data_ptr<at::Half>()),
            reinterpret_cast<const float2*>(w.data_ptr<float>()),
            mean.data_ptr<float>(), rstd.data_ptr<float>(),
            gx.defined()
                ? reinterpret_cast<const __half2*>(gx.data_ptr<at::Half>())
                : nullptr,
            reinterpret_cast<__half2*>(dx.data_ptr<at::Half>()),
            pdw.data_ptr<float>(), pdb.data_ptr<float>(), rows, d2);
      };
      if (d == 64) launch(ln_bwd_h_kernel<1>);
      else if (d == 128) launch(ln_bwd_h_kernel<2>);
      else launch(ln_bwd_h_kernel<0>);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    pdw.zero_();
    pdb.zero_();
  }
  return {dx, pdw.sum(0), pdb.sum(0)};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_fwd", &ln_fwd, "LayerNorm forward -> (y, mean, rstd)");
  m.def("ln_fwd_add", &ln_fwd_add,
        "Residual add + LayerNorm -> (x = a + b, y = LN(x), mean, rstd); "
        "optional b_bias (fp32 (d,), fp16 path only) is added to b in "
        "this pass — the producer GEMM then runs bias-free",
        py::arg("a"), py::arg("b"), py::arg("w"), py::arg("bias"),
        py::arg("eps"), py::arg("b_bias") = py::none());
  m.def("ln_bwd", &ln_bwd,
        "LayerNorm backward -> (dx, dw, db); optional gx is a residual "
        "gradient fused into the dx store (dx += gx)",
        py::arg("x"), py::arg("dy"), py::arg("w"), py::arg("mean"),
        py::arg("rstd"), py::arg("gx") = py::none());
}
