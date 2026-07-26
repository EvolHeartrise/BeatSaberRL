// fp16-storage weight-grad GEMM: dW = dY^T X with fp16 inputs, fp32
// COMPUTE and fp32 output, via explicit cublasGemmEx.
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
  bias_relu_kernel<<<(unsigned)blocks, threads, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__half2*>(y.data_ptr<at::Half>()),
      reinterpret_cast<const float2*>(b.data_ptr<float>()),
      n2, d / 2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
}
