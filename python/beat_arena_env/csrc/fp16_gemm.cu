// fp16-storage weight-grad GEMM: dW = dY^T X with fp16 inputs, fp32
// COMPUTE and fp32 output, via explicit cublasGemmEx.
//
// PROVENANCE WARNING (2026-08-16): every measurement below is sm_61
// (GTX 1080 / cu126) — none re-measured on the 4090/861k config; see
// PERF_AUDIT_2026-08-16.md. (The audit's Tier 1.1/1.2 width-gate
// misses — ffn 512 / trunk 256 falling to the slow paths — were fixed
// 2026-08-15.)
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
#include <cublasLt.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>
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
  } else if (n2 <= INT32_MAX && d2 == 256) {
    bias_relu_kernel_p2<256><<<(unsigned)blocks, threads, 0, stream>>>(
        yp, bp, (int)n2);
  } else if (n2 <= INT32_MAX && d2 == 128) {
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
//
// NT (block size) must be a multiple of D2C so the grid stride keeps
// each thread on one column pair. The power-of-two widths run NT = 256
// (unchanged since 2026-07-28); the 3·2^k widths of the 192/384/768
// recipe (2026-09-26 — the ffn hidden 768 and the SimBa/critic 384 had
// been falling back to threshold_backward + a separate gm.sum(0) since
// the 09-13 widening) run NT = 384. The shared buffer holds 2·NT floats
// for every D2C: (NT/D2C) groups x 2·D2C.
template <int D2C, int NT = 256>
__global__ void __launch_bounds__(NT)
relu_bwd_kernel(const __half2* __restrict__ gy,
                const __half2* __restrict__ y,
                __half2* __restrict__ gm,
                float* __restrict__ part, int n2) {
  static_assert(NT % D2C == 0, "block size must be a multiple of D2C");
  const int stride = gridDim.x * blockDim.x;
  float ax = 0.f, ay = 0.f;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // 4-deep manual unroll (2026-09-26): the four iterations' loads are
  // issued before any use (memory-level parallelism — the rolled loop
  // ran one dependent 4-B load pair per trip, ~0.7 TB/s at d=768), and
  // the accumulation order is exactly the rolled loop's (element i,
  // i+stride, i+2·stride, i+3·stride, ascending), so bitwise-neutral.
  for (; (int64_t)i + 3 * (int64_t)stride < n2; i += 4 * stride) {
    __half2 g4[4], y4[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      g4[u] = gy[i + u * stride];
      y4[u] = y[i + u * stride];
    }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const float2 gf = __half22float2(g4[u]);
      const float2 yf = __half22float2(y4[u]);
      float2 o;
      o.x = yf.x > 0.f ? gf.x : 0.f;
      o.y = yf.y > 0.f ? gf.y : 0.f;
      gm[i + u * stride] = __float22half2_rn(o);
      ax += o.x;
      ay += o.y;
    }
  }
  for (; i < n2; i += stride) {
    const float2 gf = __half22float2(gy[i]);
    const float2 yf = __half22float2(y[i]);
    float2 o;
    o.x = yf.x > 0.f ? gf.x : 0.f;
    o.y = yf.y > 0.f ? gf.y : 0.f;
    gm[i] = __float22half2_rn(o);   // exact: each value is gy or 0
    ax += o.x;
    ay += o.y;
  }
  __shared__ float sm[2 * NT];      // (NT/D2C) groups x 2*D2C
  const int grp = threadIdx.x / D2C, col = threadIdx.x % D2C;
  sm[grp * 2 * D2C + 2 * col] = ax;
  sm[grp * 2 * D2C + 2 * col + 1] = ay;
  __syncthreads();
  if (threadIdx.x < D2C) {
    float bx = 0.f, by = 0.f;
    for (int g = 0; g < NT / D2C; g++) {
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
  const bool p2 = d == 64 || d == 128 || d == 256 || d == 512;
  TORCH_CHECK(p2 || d == 192 || d == 384 || d == 768,
              "relu_bwd: last dim must be 64/128/256/512/192/384/768");
  const int64_t n2 = gy.numel() / 2;
  TORCH_CHECK(n2 <= INT32_MAX, "relu_bwd: tensor too large");
  auto gm = at::empty_like(gy);
  const int threads = p2 ? 256 : 384;
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
  // The shared buffer is 2*NT floats for every D2C: usage is
  // (NT/D2C) groups x 2*D2C = 2*NT, D2C-independent; the column-
  // ownership invariant (grid stride % D2C == 0) holds because NT is a
  // multiple of D2C (static_assert): NT = 256 for the power-of-two
  // widths, 384 for 192/384/768 (2026-09-26).
  if (d == 768) {
    relu_bwd_kernel<384, 384><<<blocks, threads, 0, stream>>>(
        gp, yp, mp, pp, (int)n2);
  } else if (d == 384) {
    relu_bwd_kernel<192, 384><<<blocks, threads, 0, stream>>>(
        gp, yp, mp, pp, (int)n2);
  } else if (d == 192) {
    relu_bwd_kernel<96, 384><<<blocks, threads, 0, stream>>>(
        gp, yp, mp, pp, (int)n2);
  } else if (d == 512) {
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

// ---- wgrad + bias grad in ONE cuBLASLt GEMM (BGRADB epilogue) ----------
// dW = dy^T x (fp32 out) AND db = sum_rows dy (fp32) — the epilogue
// reduces B (= dy) over the GEMM's k (= row) dimension while the GEMM
// streams it anyway, replacing the separate full re-read of dy by a
// torch column sum (the bias grads were ~5 ms of a production
// minibatch, 2026-09-26 profile). One plan per (R, M, K), algo = the
// heuristic's first pick (deterministic for a given GPU/driver/cuBLAS —
// no timing-based choice, so a process-to-process flip of the fp16
// path's bits cannot happen).
struct LtPlan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulAlgo_t algo;
  size_t ws = 0;
  bool ok = false;
};

constexpr size_t kLtWorkspace = 32u << 20;

LtPlan& lt_plan(int64_t R, int64_t M, int64_t K) {
  static std::map<std::tuple<int64_t, int64_t, int64_t>, LtPlan> plans;
  auto key = std::make_tuple(R, M, K);
  auto it = plans.find(key);
  if (it != plans.end()) return it->second;
  LtPlan p;
  auto chk = [](cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS; };
  bool ok = chk(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F,
                                         CUDA_R_32F));
  const cublasOperation_t tn = CUBLAS_OP_N, tt = CUBLAS_OP_T;
  const cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BGRADB;
  const cudaDataType_t bt = CUDA_R_32F;
  ok = ok && chk(cublasLtMatmulDescSetAttribute(
      p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tn, sizeof(tn)));
  ok = ok && chk(cublasLtMatmulDescSetAttribute(
      p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tt, sizeof(tt)));
  ok = ok && chk(cublasLtMatmulDescSetAttribute(
      p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
  ok = ok && chk(cublasLtMatmulDescSetAttribute(
      p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
  // same col-major views as wgrad's GemmEx: C'(K,M) = x(K,R) · dy(M,R)^T
  ok = ok && chk(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16F, K, R, K));
  ok = ok && chk(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16F, M, R, M));
  ok = ok && chk(cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_32F, K, M, K));
  if (ok) {
    cublasLtMatmulPreference_t pref = nullptr;
    ok = chk(cublasLtMatmulPreferenceCreate(&pref));
    size_t wsmax = kLtWorkspace;
    ok = ok && chk(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsmax,
        sizeof(wsmax)));
    cublasLtMatmulHeuristicResult_t res{};
    int n = 0;
    ok = ok && chk(cublasLtMatmulAlgoGetHeuristic(
        at::cuda::getCurrentCUDABlasLtHandle(), p.op, p.la, p.lb, p.lc,
        p.lc, pref, 1, &res, &n)) && n > 0;
    if (ok) {
      p.algo = res.algo;
      p.ws = res.workspaceSize;
    }
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
  }
  p.ok = ok;
  return plans.emplace(key, p).first->second;
}

// Returns {dW (M,K) fp32, db (M,) fp32}, or {} when no Lt algo supports
// the shape (caller falls back to wgrad + a torch column sum).
std::vector<at::Tensor> wgrad_bgrad(at::Tensor dy, at::Tensor x) {
  TORCH_CHECK(dy.is_cuda() && dy.scalar_type() == at::kHalf
              && dy.is_contiguous(), "dy: contiguous CUDA fp16");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf
              && x.is_contiguous(), "x: contiguous CUDA fp16");
  TORCH_CHECK(dy.size(0) == x.size(0), "row mismatch");
  const c10::cuda::CUDAGuard guard(dy.device());
  const int64_t R = dy.size(0), M = dy.size(1), K = x.size(1);
  LtPlan& p = lt_plan(R, M, K);
  if (!p.ok) return {};
  auto c = at::empty({M, K}, x.options().dtype(at::kFloat));
  auto db = at::empty({M}, x.options().dtype(at::kFloat));
  void* bp = db.data_ptr();
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp)));
  at::Tensor ws;
  if (p.ws) ws = at::empty({(int64_t)p.ws}, x.options().dtype(at::kByte));
  const float alpha = 1.f, beta = 0.f;
  TORCH_CUDABLAS_CHECK(cublasLtMatmul(
      at::cuda::getCurrentCUDABlasLtHandle(), p.op, &alpha,
      x.data_ptr(), p.la, dy.data_ptr(), p.lb, &beta,
      c.data_ptr(), p.lc, c.data_ptr(), p.lc, &p.algo,
      p.ws ? ws.data_ptr() : nullptr, p.ws,
      at::cuda::getCurrentCUDAStream()));
  return {c, db};
}

// ---- ReLU bitmask epilogues: RELU_AUX_BIAS forward, DRELU dgrad ---------
// A Linear+ReLU whose consumer is another Linear (block FFN, note/saber/
// torso/leg encoders, SimBa trunk fc1, critic MLP) runs its forward as ONE
// cuBLASLt GEMM that also writes the ReLU bitmask (1 bit per element), and
// the consumer's dgrad GEMM (g @ W_next) applies that mask in its epilogue
// -- replacing the standalone relu_bwd pass (a full read of g and y plus a
// write of gm, ~7.5 ms of a production minibatch at the ffn hidden width).
// The ReLU layer's bias grad then comes from its own wgrad GEMM (BGRADB
// epilogue at >= 64k rows, see wgrad_bgrad), so no pass re-reads gm there.
// (DRELU_BGRAD would reduce db in the dgrad epilogue too, but cuBLASLt
// offers it here only with a bias of the output type -- an fp16 db, which
// overflows at the production loss scale over ~475k rows -- and no algo
// for an fp32 one; probed 2026-09-27 on the RTX 5090, cu128.)
// Layouts (col-major, m = H output features, n = R rows): the aux mask is
// (H, R) with ld = H rounded up to 128 bits (cuBLASLt requirement), i.e.
// R * ld/8 bytes. Algo = the heuristic's first pick per shape
// (deterministic, like lt_plan).
struct LtEpiPlan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulAlgo_t algo;
  size_t ws = 0;
  bool ok = false;
};

inline int64_t relu_mask_ld(int64_t H) { return (H + 127) / 128 * 128; }

// kind 0: forward  C(H,R) = W(H,K) x^T + b, relu, aux mask   (A=W op T)
// kind 1: dgrad    C(H,R) = W(M,H)^T g^T, drelu(mask)        (A=W op N)
LtEpiPlan& lt_epi_plan(int kind, int64_t R, int64_t H, int64_t K) {
  static std::map<std::tuple<int, int64_t, int64_t, int64_t>, LtEpiPlan>
      plans;
  auto key = std::make_tuple(kind, R, H, K);
  auto it = plans.find(key);
  if (it != plans.end()) return it->second;
  LtEpiPlan p;
  auto chk = [](cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS; };
  bool ok = chk(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F,
                                         CUDA_R_32F));
  const cublasOperation_t tn = CUBLAS_OP_N, tt = CUBLAS_OP_T;
  const cublasLtEpilogue_t epi = kind == 0 ? CUBLASLT_EPILOGUE_RELU_AUX_BIAS
                                           : CUBLASLT_EPILOGUE_DRELU;
  const int64_t ld = relu_mask_ld(H);
  const bool ta = kind == 0;
  ok = ok && chk(cublasLtMatmulDescSetAttribute(
      p.op, CUBLASLT_MATMUL_DESC_TRANSA, ta ? &tt : &tn,
      sizeof(tn)));
  ok = ok && chk(cublasLtMatmulDescSetAttribute(
      p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof(tn)));
  ok = ok && chk(cublasLtMatmulDescSetAttribute(
      p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
  ok = ok && chk(cublasLtMatmulDescSetAttribute(
      p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld, sizeof(ld)));
  if (ta) {
    // A = W row-major (H,K) = col-major (K,H), op T; B = x (K,R), op N
    ok = ok && chk(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16F, K, H, K));
    ok = ok && chk(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16F, K, R, K));
  } else {
    // A = W row-major (K=M_next,H) = col-major (H,K), op N; B = g (K,R)
    ok = ok && chk(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16F, H, K, H));
    ok = ok && chk(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16F, K, R, K));
  }
  ok = ok && chk(cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16F, H, R, H));
  if (ok) {
    cublasLtMatmulPreference_t pref = nullptr;
    ok = chk(cublasLtMatmulPreferenceCreate(&pref));
    size_t wsmax = kLtWorkspace;
    ok = ok && chk(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsmax,
        sizeof(wsmax)));
    cublasLtMatmulHeuristicResult_t res{};
    int n = 0;
    ok = ok && chk(cublasLtMatmulAlgoGetHeuristic(
        at::cuda::getCurrentCUDABlasLtHandle(), p.op, p.la, p.lb, p.lc,
        p.lc, pref, 1, &res, &n)) && n > 0;
    if (ok) {
      p.algo = res.algo;
      p.ws = res.workspaceSize;
    }
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
  }
  p.ok = ok;
  return plans.emplace(key, p).first->second;
}

void lt_epi_run(LtEpiPlan& p, const void* a, const void* b, void* c,
                void* bias, void* aux, const at::Tensor& like) {
  if (bias)
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
        p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux, sizeof(aux)));
  at::Tensor ws;
  if (p.ws) ws = at::empty({(int64_t)p.ws}, like.options().dtype(at::kByte));
  const float alpha = 1.f, beta = 0.f;
  TORCH_CUDABLAS_CHECK(cublasLtMatmul(
      at::cuda::getCurrentCUDABlasLtHandle(), p.op, &alpha, a, p.la, b,
      p.lb, &beta, c, p.lc, c, p.lc, &p.algo,
      p.ws ? ws.data_ptr() : nullptr, p.ws,
      at::cuda::getCurrentCUDAStream()));
}

// y = relu(x @ w^T + b) (R,H) fp16 and its bitmask (R, ld/8) uint8;
// x (R,K), w (H,K), b (H) fp16 (the replicas). {} if no Lt algo.
std::vector<at::Tensor> linear_relu_aux(at::Tensor x, at::Tensor w,
                                        at::Tensor b) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf
              && x.is_contiguous() && x.dim() == 2, "x: (R,K) fp16");
  TORCH_CHECK(w.scalar_type() == at::kHalf && w.is_contiguous()
              && w.dim() == 2 && w.size(1) == x.size(1), "w: (H,K) fp16");
  TORCH_CHECK(b.scalar_type() == at::kHalf && b.is_contiguous()
              && b.numel() == w.size(0), "b: (H) fp16");
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t R = x.size(0), K = x.size(1), H = w.size(0);
  LtEpiPlan& p = lt_epi_plan(0, R, H, K);
  if (!p.ok) return {};
  auto y = at::empty({R, H}, x.options());
  auto mask = at::empty({R, relu_mask_ld(H) / 8}, x.options().dtype(at::kByte));
  if (R == 0) return {y, mask};
  lt_epi_run(p, w.data_ptr(), x.data_ptr(), y.data_ptr(), b.data_ptr(),
             mask.data_ptr(), x);
  return {y, mask};
}

// gm = (g @ w) * mask (R,H) fp16; g (R,M), w (M,H) fp16, mask from
// linear_relu_aux. Undefined tensor (None) if no Lt algo.
at::Tensor dgrad_drelu(at::Tensor g, at::Tensor w, at::Tensor mask) {
  TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kHalf
              && g.is_contiguous() && g.dim() == 2, "g: (R,M) fp16");
  TORCH_CHECK(w.scalar_type() == at::kHalf && w.is_contiguous()
              && w.dim() == 2 && w.size(0) == g.size(1), "w: (M,H) fp16");
  const int64_t R = g.size(0), K = g.size(1), H = w.size(1);
  TORCH_CHECK(mask.scalar_type() == at::kByte && mask.is_contiguous()
              && mask.size(0) == R && mask.size(1) == relu_mask_ld(H) / 8,
              "mask: (R, ld/8) uint8 from linear_relu_aux");
  const c10::cuda::CUDAGuard guard(g.device());
  LtEpiPlan& p = lt_epi_plan(1, R, H, K);
  if (!p.ok) return at::Tensor();
  auto gm = at::empty({R, H}, g.options());
  if (R == 0) return gm;
  lt_epi_run(p, w.data_ptr(), g.data_ptr(), gm.data_ptr(), nullptr,
             mask.data_ptr(), g);
  return gm;
}

}  // namespace

// small_attn.cu (T = 29 fused attention, 2026-09-27)
at::Tensor attn29_fwd(at::Tensor q, at::Tensor k, at::Tensor v,
                      at::Tensor bias);
void attn29_bwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor bias,
                at::Tensor dout, at::Tensor dq, at::Tensor dk, at::Tensor dv);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn29_fwd", &attn29_fwd,
        "fused 4-head x 48 attention over 29 keys (TQ = 29/5/1 queries), "
        "additive fp16 key bias -> contiguous (B, TQ, 192) fp16");
  m.def("attn29_bwd", &attn29_bwd,
        "attn29 backward: writes dq/dk/dv into the given strided views");
  m.def("linear_relu_aux", &linear_relu_aux,
        "(y, mask): y = relu(x @ w^T + b) fp16 + its ReLU bitmask, one "
        "cublasLt GEMM (RELU_AUX_BIAS); [] if unsupported");
  m.def("dgrad_drelu", &dgrad_drelu,
        "gm = (g @ w) * mask fp16, one cublasLt GEMM (DRELU epilogue); "
        "None if unsupported");
  m.def("wgrad_bgrad", &wgrad_bgrad,
        "(dW = dY^T X, db = sum_rows dY), fp16 in / fp32 out, one "
        "cublasLt GEMM with the BGRADB epilogue; [] if unsupported");
  m.def("wgrad", &wgrad,
        "dW = dY^T X, fp16 inputs, fp32 compute + output (cublasGemmEx)");
  m.def("bias_relu_", &bias_relu_,
        "in-place y = relu(y + b): fp16 y, fp32 master bias");
  m.def("relu_bwd", &relu_bwd,
        "fused ReLU backward -> (gm, part): gm = gy * (y > 0), part = "
        "(grid, d) fp32 column-sum partials of gm (db = part.sum(0))");
}
