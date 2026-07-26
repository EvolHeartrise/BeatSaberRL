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

__global__ void ln_bwd_kernel(const float* __restrict__ x,
                              const float* __restrict__ dy,
                              const float* __restrict__ w,
                              const float* __restrict__ mean,
                              const float* __restrict__ rstd,
                              float* __restrict__ dx,
                              float* __restrict__ pdw,  // (gridDim.x, d)
                              float* __restrict__ pdb,  // (gridDim.x, d)
                              int64_t rows, int d) {
  extern __shared__ float sm[];  // [2][WPB][d]
  const int lane = threadIdx.x % WARP;
  const int wid = threadIdx.x / WARP;
  float acc_dw[MAX_COLS] = {};
  float acc_db[MAX_COLS] = {};
  int64_t row = (int64_t)blockIdx.x * WPB + wid;
  const int64_t stride = (int64_t)gridDim.x * WPB;
  for (; row < rows; row += stride) {
    const float* xr = x + row * d;
    const float* gr = dy + row * d;
    const float mu = mean[row];
    const float r = rstd[row];
    float s1 = 0.f, s2 = 0.f;
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
    for (int c = lane; c < d; c += WARP) {
      const float xh = (xr[c] - mu) * r;
      dxr[c] = (gr[c] * w[c] - s1 - xh * s2) * r;
    }
  }
  // block reduction of the weight/bias partials: each warp stages its
  // lane-owned columns, then the block sums across warps into this
  // block's row of the (blocks, d) partial buffers
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

std::vector<at::Tensor> ln_fwd(at::Tensor x, at::Tensor w, at::Tensor b,
                               double eps) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && x.is_contiguous()
              && x.scalar_type() == at::kFloat, "x: contiguous CUDA f32");
  const int d = (int)x.size(-1);
  TORCH_CHECK(d <= MAX_D, "d > ", MAX_D, " unsupported");
  TORCH_CHECK(w.is_contiguous() && b.is_contiguous()
              && w.numel() == d && b.numel() == d);
  const int64_t rows = x.numel() / d;
  auto y = at::empty_like(x);
  auto mean = at::empty({rows}, x.options());
  auto rstd = at::empty({rows}, x.options());
  if (rows == 0) return {y, mean, rstd};
  ln_fwd_kernel<<<grid_for(rows, 4096), THREADS, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
      y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
      rows, d, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}

std::vector<at::Tensor> ln_bwd(at::Tensor x, at::Tensor dy, at::Tensor w,
                               at::Tensor mean, at::Tensor rstd) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_contiguous() && dy.is_contiguous()
              && x.scalar_type() == at::kFloat
              && dy.scalar_type() == at::kFloat);
  const int d = (int)x.size(-1);
  const int64_t rows = x.numel() / d;
  auto dx = at::empty_like(x);
  // fixed-size partials + deterministic torch sum: bit-reproducible
  // gradients (no cross-block float atomics)
  const int blocks = std::max(grid_for(rows, 1024), 1);
  auto pdw = at::empty({blocks, d}, x.options());
  auto pdb = at::empty({blocks, d}, x.options());
  if (rows > 0) {
    const size_t smem = 2 * WPB * d * sizeof(float);
    ln_bwd_kernel<<<blocks, THREADS, smem,
                    at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), dy.data_ptr<float>(), w.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(),
        dx.data_ptr<float>(), pdw.data_ptr<float>(), pdb.data_ptr<float>(),
        rows, d);
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
  m.def("ln_bwd", &ln_bwd, "LayerNorm backward -> (dx, dw, db)");
}
