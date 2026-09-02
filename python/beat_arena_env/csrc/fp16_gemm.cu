// fp16-storage weight-grad GEMM: dW = dY^T X with fp16 inputs, fp32
// COMPUTE and fp32 output, via explicit cublasGemmEx.
//
// PROVENANCE WARNING (2026-08-16): every measurement below is sm_61
// (GTX 1080 / cu126) — none re-measured on the 4090/861k config; see
// PERF_AUDIT_2026-08-16.md. (The audit's Tier 1.1/1.2 width-gate
// misses — ffn 512 / trunk 256 falling to the slow paths — were fixed
// 2026-08-15; BEAT_GEMM_LEGACY=1 restores the old bias_relu dispatch
// for A/B.)
//
// Why this exists (2026-07-25 gate benchmarks, GTX 1080 / cu126):
// torch's half matmul dispatch routes the TN large-K shape (K = mb*18 =
// 589,824) to a kernel that is BOTH 4.5x slower than fp32 AND sloppily
// accumulated (~3.4e-4 accumulation error vs 2e-6 here) — and
// `allow_fp16_reduced_precision_reduction = False` fixes neither.
// Explicit GemmEx with CUBLAS_COMPUTE_32F picks a fast kernel: measured
// 1.32x faster than torch fp32 at (64,64,K) and 3.4x at (128,64,K),
// with true fp32 accumulation. Forward/dgrad shapes (K = 64/128) stay
// on torch's half matmul, which is fine there.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>

namespace {

// In-place bias + ReLU epilogue: y = relu(y + b), fp32 math, one
// rounding on store (round-4 #2: the cuBLAS bias epilogue costs 33-59%
// of each addmm on sm_61, so Linear+ReLU pairs run a bias-FREE mm and
// this replaces the standalone ReLU pass at identical traffic). b is
// the fp32 MASTER bias — no fp16 replica, stable storage for graphs.
__global__ void bias_relu_kernel(__half2* __restrict__ y,
                                 const float2* __restrict__ b,
                                 int64_t n2, int d2) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) return;
  const float2 v = __half22float2(y[i]);
  const float2 bb = b[i % d2];
  float2 o;
  o.x = fmaxf(v.x + bb.x, 0.f);
  o.y = fmaxf(v.y + bb.y, 0.f);
  y[i] = __float22half2_rn(o);
}

// Power-of-two d2 specialization. Originally instantiated for the
// 138k arch's widths only (64/128 -> d2 32/64); the v17 scale-up's
// ffn hidden (512) and trunk (256) fell through to the generic
// s64-modulo kernel for three days (PERF_AUDIT_2026-08-16.md Tier
// 1.1) — d2 256/128 instantiated 2026-08-15. The generic kernel's
// `i % d2` is a SIGNED 64-BIT remainder — nvcc emits tens of
// instructions per element in a kernel whose real work is 2 loads +
// 2 FMA + 1 store. Here the index is int32 and the remainder is a
// compile-time mask. Bit-identical: index math only.
template <int D2C>
__global__ void bias_relu_kernel_p2(__half2* __restrict__ y,
                                    const float2* __restrict__ b,
                                    int n2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) return;
  const float2 v = __half22float2(y[i]);
  const float2 bb = b[i & (D2C - 1)];
  float2 o;
  o.x = fmaxf(v.x + bb.x, 0.f);
  o.y = fmaxf(v.y + bb.y, 0.f);
  y[i] = __float22half2_rn(o);
}

// BEAT_GEMM_LEGACY=1 pins the pre-2026-08-15 dispatch (generic kernel
// for d2 128/256) so the measurement ladder can A/B the width fix
// without a rebuild. Read once — dispatch stays branch-free.
bool legacy_dispatch() {
  static const bool v = []() {
    const char* e = getenv("BEAT_GEMM_LEGACY");
    return e && e[0] == '1';
  }();
  return v;
}

void bias_relu_(at::Tensor y, at::Tensor b) {
  const c10::cuda::CUDAGuard guard(y.device());
  const int d = (int)y.size(-1);
  TORCH_CHECK(y.is_cuda() && y.is_contiguous()
              && y.scalar_type() == at::kHalf && d % 2 == 0,
              "y: contiguous CUDA fp16, even last dim");
  TORCH_CHECK(b.is_cuda() && b.is_contiguous()
              && b.scalar_type() == at::kFloat && b.numel() == d,
              "b: contiguous fp32 (", d, ",)");
  const int64_t n2 = y.numel() / 2;
  if (n2 == 0) return;
  const int threads = 256;
  const int64_t blocks = (n2 + threads - 1) / threads;
  const int d2 = d / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* yp = reinterpret_cast<__half2*>(y.data_ptr<at::Half>());
  auto* bp = reinterpret_cast<const float2*>(b.data_ptr<float>());
  if (n2 <= INT32_MAX && d2 == 64) {
    bias_relu_kernel_p2<64><<<(unsigned)blocks, threads, 0, stream>>>(
        yp, bp, (int)n2);
  } else if (n2 <= INT32_MAX && d2 == 32) {
    bias_relu_kernel_p2<32><<<(unsigned)blocks, threads, 0, stream>>>(
        yp, bp, (int)n2);
  } else if (n2 <= INT32_MAX && d2 == 256 && !legacy_dispatch()) {
    bias_relu_kernel_p2<256><<<(unsigned)blocks, threads, 0, stream>>>(
        yp, bp, (int)n2);
  } else if (n2 <= INT32_MAX && d2 == 128 && !legacy_dispatch()) {
    bias_relu_kernel_p2<128><<<(unsigned)blocks, threads, 0, stream>>>(
        yp, bp, (int)n2);
  } else {
    bias_relu_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        yp, bp, n2, d2);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fused ReLU backward + bias-grad partials (2026-07-28): gm = gy
// masked by (y > 0) in ONE pass that also accumulates per-column fp32
// sums of gm — replacing aten::threshold_backward + a separate
// full-tensor gm.sum(0) re-read (~75 MB per ffn site per minibatch).
// Deterministic: fixed grid, each thread owns one column pair (the
// grid stride is a multiple of D2C) and walks its elements ascending;
// block groups combine in fixed order; the (grid, d) partials finish
// with a torch sum(0).
template <int D2C>
__global__ void relu_bwd_kernel(const __half2* __restrict__ gy,
                                const __half2* __restrict__ y,
                                __half2* __restrict__ gm,
                                float* __restrict__ part, int n2) {
  const int stride = gridDim.x * blockDim.x;
  float ax = 0.f, ay = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += stride) {
    const float2 gf = __half22float2(gy[i]);
    const float2 yf = __half22float2(y[i]);
    float2 o;
    o.x = yf.x > 0.f ? gf.x : 0.f;
    o.y = yf.y > 0.f ? gf.y : 0.f;
    gm[i] = __float22half2_rn(o);   // exact: each value is gy or 0
    ax += o.x;
    ay += o.y;
  }
  __shared__ float sm[512];         // (blockDim/D2C) groups x 2*D2C
  const int grp = threadIdx.x / D2C, col = threadIdx.x % D2C;
  sm[grp * 2 * D2C + 2 * col] = ax;
  sm[grp * 2 * D2C + 2 * col + 1] = ay;
  __syncthreads();
  if (threadIdx.x < D2C) {
    float bx = 0.f, by = 0.f;
    for (int g = 0; g < (int)blockDim.x / D2C; g++) {
      bx += sm[g * 2 * D2C + 2 * threadIdx.x];
      by += sm[g * 2 * D2C + 2 * threadIdx.x + 1];
    }
    part[blockIdx.x * 2 * D2C + 2 * threadIdx.x] = bx;
    part[blockIdx.x * 2 * D2C + 2 * threadIdx.x + 1] = by;
  }
}

std::vector<at::Tensor> relu_bwd(at::Tensor gy, at::Tensor y) {
  const c10::cuda::CUDAGuard guard(gy.device());
  const int d = (int)gy.size(-1);
  TORCH_CHECK(gy.is_cuda() && gy.is_contiguous()
              && gy.scalar_type() == at::kHalf, "gy: contiguous CUDA fp16");
  TORCH_CHECK(y.is_cuda() && y.is_contiguous()
              && y.scalar_type() == at::kHalf && y.sizes() == gy.sizes(),
              "y: contiguous CUDA fp16, same shape as gy");
  TORCH_CHECK(d == 64 || d == 128 || d == 256 || d == 512,
              "relu_bwd: last dim must be 64/128/256/512");
  const int64_t n2 = gy.numel() / 2;
  TORCH_CHECK(n2 <= INT32_MAX, "relu_bwd: tensor too large");
  auto gm = at::empty_like(gy);
  const int threads = 256;
  const int blocks = (int)std::min<int64_t>(
      512, (n2 + threads - 1) / threads);
  // at::empty, not zeros (2026-09-02, Tier 1.5 of the 09-02 audit
  // A §8): the kernel's tail writes ALL d columns of EVERY launched
  // block's row (threads 0..D2C-1 each store 2 floats, 2*D2C == d for
  // every dispatch below, and a thread whose grid-stride loop never
  // ran still stores its 0.f partials), so the memset was dead — ~900
  // launches per production update across the 7 relu sites. n2 == 0
  // returns before the launch with blocks == 0, i.e. an empty part.
  auto part = at::empty({blocks, d}, gy.options().dtype(at::kFloat));
  if (n2 == 0) return {gm, part};
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* gp = reinterpret_cast<const __half2*>(gy.data_ptr<at::Half>());
  auto* yp = reinterpret_cast<const __half2*>(y.data_ptr<at::Half>());
  auto* mp = reinterpret_cast<__half2*>(gm.data_ptr<at::Half>());
  auto* pp = part.data_ptr<float>();
  // v17 widths (2026-08-15, audit Tier 1.2): d 512/256 -> D2C 256/128.
  // The shared buffer stays 512 floats for every D2C: usage is
  // (blockDim/D2C) groups x 2*D2C = 2*blockDim = 512, D2C-independent;
  // the column-ownership invariant (grid stride % D2C == 0) holds
  // because blockDim = 256 is a multiple of every D2C here.
  if (d == 512) {
    relu_bwd_kernel<256><<<blocks, threads, 0, stream>>>(gp, yp, mp, pp,
                                                         (int)n2);
  } else if (d == 256) {
    relu_bwd_kernel<128><<<blocks, threads, 0, stream>>>(gp, yp, mp, pp,
                                                         (int)n2);
  } else if (d == 128) {
    relu_bwd_kernel<64><<<blocks, threads, 0, stream>>>(gp, yp, mp, pp,
                                                        (int)n2);
  } else {
    relu_bwd_kernel<32><<<blocks, threads, 0, stream>>>(gp, yp, mp, pp,
                                                        (int)n2);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gm, part};
}

at::Tensor wgrad(at::Tensor dy, at::Tensor x) {
  // dy: (R, M) fp16, x: (R, K) fp16  ->  dW = dy^T x : (M, K) fp32
  TORCH_CHECK(dy.is_cuda() && dy.scalar_type() == at::kHalf
              && dy.is_contiguous(), "dy: contiguous CUDA fp16");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf
              && x.is_contiguous(), "x: contiguous CUDA fp16");
  TORCH_CHECK(dy.size(0) == x.size(0), "row mismatch");
  const int64_t R = dy.size(0), M = dy.size(1), K = x.size(1);
  auto c = at::empty({M, K}, x.options().dtype(at::kFloat));
  const float alpha = 1.f, beta = 0.f;
  auto h = at::cuda::getCurrentCUDABlasHandle();
  // row-major C(M,K) = dy^T x  <=>  col-major C^T(K,M) = x^T dy:
  // x row-major (R,K) is col-major (K,R) -> op N gives (K,R)?? No:
  // col-major x view is (K,R); we need x^T(K,R) row-major = x col view
  // as-is with op N -> (K, R)... multiply (K,R)x(R,M): op N on x view
  // (K,R), op T on dy col view (M,R)^T -> (R,M). cuBLAS: C'(K,M) =
  // A(K,R) B(R,M) with A = x col view op N (lda=K), B = dy col view
  // (M,R) op T (ldb=M), ldc = K.
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      h, CUBLAS_OP_N, CUBLAS_OP_T, (int)K, (int)M, (int)R,
      &alpha,
      x.data_ptr(), CUDA_R_16F, (int)K,
      dy.data_ptr(), CUDA_R_16F, (int)M,
      &beta,
      c.data_ptr(), CUDA_R_32F, (int)K,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  return c;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wgrad", &wgrad,
        "dW = dY^T X, fp16 inputs, fp32 compute + output (cublasGemmEx)");
  m.def("bias_relu_", &bias_relu_,
        "in-place y = relu(y + b): fp16 y, fp32 master bias");
  m.def("relu_bwd", &relu_bwd,
        "fused ReLU backward -> (gm, part): gm = gy * (y > 0), part = "
        "(grid, d) fp32 column-sum partials of gm (db = part.sum(0))");
}
