// Fused single-head attention for the policy's tiny-sequence shapes
// (2026-07-25 opt-survey round 4, item #15): T <= 32 tokens, d = 64,
// batch = minibatch rows (32k in production). The eager path
// (baddbmm -> softmax -> bmm) materializes the scores tensor and
// re-reads it three times per direction; at these shapes cuBLAS
// strided-batched GEMMs are latency-bound (~3% of FLOP roofline, the
// round-4 SDPA probe measured torch's mem-efficient backend SLOWER than
// eager on sm_61). One block per sample keeps q/k/v/p in shared memory
// and touches global exactly once per operand.
//
// Precision: fp16 I/O, ALL arithmetic fp32 (GP104 half math is 1/64
// rate). softmax is max-subtracted expf in fp32; the additive mask
// (-16384) lands in expf's exact-zero regime, so masked weights are
// bitwise 0 like the eager path. p is saved fp16 for the backward —
// same storage precision as eager's fp16 softmax output. NOT bitwise
// vs eager (different accumulation order) — the documented near-tie
// tolerance class; the fp32 update path never routes here.
//
// Backward emits CONTIGUOUS dq/dk/dv, which also removes the hidden
// .contiguous() copies eager's bmm-backward transposed views forced in
// _Fp16Linear.backward (round-4 finding #8).

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <vector>

namespace {

constexpr int D = 64;        // head dim (embed); d != 64 falls back in python
constexpr int D2 = D / 2;
constexpr int THREADS = 128;
constexpr int MAX_T = 32;

// forward: one block per sample.
//   smem: sq (Tq*D) | sk (T*D) | sv (T*D) | sp (Tq*T) | sbias (T)
// phases: load -> scores (pair per thread) -> softmax (row per thread)
//         -> out (output element per thread)
// bq/bk/bv (nullable, fp32 (D,)): the q/k/v projection BIASES, folded
// into the smem load (round-4 #2: the cuBLAS bias epilogue on sm_61
// costs 33-59%% of each addmm — the projections run bias-free mm and
// the bias rides these loads for free). fp32 masters are read directly:
// no fp16 replica cache, and the storages are stable so a captured
// CUDA graph keeps reading fresh values after every opt.step.
__global__ void attn_fwd_kernel(const __half2* __restrict__ q,
                                const __half2* __restrict__ k,
                                const __half2* __restrict__ v,
                                const float2* __restrict__ bq,
                                const float2* __restrict__ bk,
                                const float2* __restrict__ bv,
                                const __half2* __restrict__ bias,  // (B, T/2)
                                __half* __restrict__ p_out,        // (B,Tq,T)
                                __half2* __restrict__ out,         // (B,Tq,D2)
                                int Tq, int T, float alpha) {
  extern __shared__ float sm[];
  float* sq = sm;                    // Tq * D
  float* sk = sq + Tq * D;           // T * D
  float* sv = sk + T * D;            // T * D
  float* sp = sv + T * D;            // Tq * T
  float* sbias = sp + Tq * T;        // T
  const int64_t b = blockIdx.x;
  const int tid = threadIdx.x;

  // load q/k/v (half2 -> two fp32 smem slots), bias
  const __half2* qb = q + b * (Tq * D2);
  const __half2* kb = k + b * (T * D2);
  const __half2* vb = v + b * (T * D2);
  for (int i = tid; i < Tq * D2; i += THREADS) {
    const float2 f = __half22float2(qb[i]);
    const float2 bb2 = bq ? bq[i % D2] : make_float2(0.f, 0.f);
    sq[2 * i] = f.x + bb2.x;
    sq[2 * i + 1] = f.y + bb2.y;
  }
  for (int i = tid; i < T * D2; i += THREADS) {
    const int c2 = i % D2;
    const float2 fk = __half22float2(kb[i]);
    const float2 bk2 = bk ? bk[c2] : make_float2(0.f, 0.f);
    sk[2 * i] = fk.x + bk2.x;
    sk[2 * i + 1] = fk.y + bk2.y;
    const float2 fv = __half22float2(vb[i]);
    const float2 bv2 = bv ? bv[c2] : make_float2(0.f, 0.f);
    sv[2 * i] = fv.x + bv2.x;
    sv[2 * i + 1] = fv.y + bv2.y;
  }
  const __half2* bb = bias + b * (T / 2);
  for (int i = tid; i < T / 2; i += THREADS) {
    const float2 f = __half22float2(bb[i]);
    sbias[2 * i] = f.x;
    sbias[2 * i + 1] = f.y;
  }
  __syncthreads();

  // scores: pair (i, j) per thread
  for (int p = tid; p < Tq * T; p += THREADS) {
    const int i = p / T, j = p % T;
    const float* qi = sq + i * D;
    const float* kj = sk + j * D;
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < D; c++) acc += qi[c] * kj[c];
    sp[p] = acc * alpha + sbias[j];
  }
  __syncthreads();

  // softmax: row per thread (Tq <= 32 rows; serial over T in registers)
  if (tid < Tq) {
    float* row = sp + tid * T;
    float mx = row[0];
    for (int j = 1; j < T; j++) mx = fmaxf(mx, row[j]);
    float s = 0.f;
    for (int j = 0; j < T; j++) {
      const float e = expf(row[j] - mx);
      row[j] = e;
      s += e;
    }
    const float inv = 1.f / s;
    __half* pr = p_out + (b * Tq + tid) * T;
    for (int j = 0; j < T; j++) {
      row[j] *= inv;
      pr[j] = __float2half_rn(row[j]);
    }
  }
  __syncthreads();

  // out[i][c] = sum_j p[i][j] * v[j][c]; element per thread, half2 store
  __half2* ob = out + b * (Tq * D2);
  for (int p = tid; p < Tq * D2; p += THREADS) {
    const int i = p / D2, c2 = p % D2;
    const float* pi = sp + i * T;
    float ax = 0.f, ay = 0.f;
    for (int j = 0; j < T; j++) {
      const float pj = pi[j];
      ax += pj * sv[j * D + 2 * c2];
      ay += pj * sv[j * D + 2 * c2 + 1];
    }
    ob[p] = __float22half2_rn(make_float2(ax, ay));
  }
}

// backward: one block per sample.
//   smem: sA (max(Tq,T)*D: dout -> q) | sB (T*D: v -> k)
//       | sp (Tq*T) | sds (Tq*T)
// phases: load p,dout,v -> dp -> dv -> ds(=alpha*softmax_bwd)
//         -> load k -> dq -> load q -> dk
// bq/bk/bv reconstruct the BIASED q/k/v the forward attended with (the
// saved inputs are the bias-free mm outputs). dbq/dbk/dbv are computed
// python-side as deterministic torch channel-sums of dq/dk/dv.
__global__ void attn_bwd_kernel(const __half2* __restrict__ dout,
                                const __half* __restrict__ p,   // (B,Tq,T)
                                const __half2* __restrict__ q,
                                const __half2* __restrict__ k,
                                const __half2* __restrict__ v,
                                const float2* __restrict__ bq,
                                const float2* __restrict__ bk,
                                const float2* __restrict__ bv,
                                __half2* __restrict__ dq,
                                __half2* __restrict__ dk,
                                __half2* __restrict__ dv,
                                int Tq, int T, float alpha) {
  extern __shared__ float sm[];
  const int A_rows = Tq > T ? Tq : T;
  float* sA = sm;                    // A_rows * D (dout, later q)
  float* sB = sA + A_rows * D;       // T * D     (v, later k)
  float* sp = sB + T * D;            // Tq * T
  float* sds = sp + Tq * T;          // Tq * T
  const int64_t b = blockIdx.x;
  const int tid = threadIdx.x;

  const __half2* db = dout + b * (Tq * D2);
  const __half2* vb = v + b * (T * D2);
  const __half* pb = p + b * (Tq * T);
  for (int i = tid; i < Tq * D2; i += THREADS) {
    const float2 f = __half22float2(db[i]);
    sA[2 * i] = f.x;
    sA[2 * i + 1] = f.y;
  }
  for (int i = tid; i < T * D2; i += THREADS) {
    const float2 f = __half22float2(vb[i]);
    const float2 b2 = bv ? bv[i % D2] : make_float2(0.f, 0.f);
    sB[2 * i] = f.x + b2.x;
    sB[2 * i + 1] = f.y + b2.y;
  }
  for (int i = tid; i < Tq * T; i += THREADS) sp[i] = __half2float(pb[i]);
  __syncthreads();

  // dp[i][j] = dot(dout_i, v_j)  (stored into sds)
  for (int e = tid; e < Tq * T; e += THREADS) {
    const int i = e / T, j = e % T;
    const float* di = sA + i * D;
    const float* vj = sB + j * D;
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < D; c++) acc += di[c] * vj[c];
    sds[e] = acc;
  }
  __syncthreads();

  // dv[j][c] = sum_i p[i][j] * dout[i][c]
  __half2* dvb = dv + b * (T * D2);
  for (int e = tid; e < T * D2; e += THREADS) {
    const int j = e / D2, c2 = e % D2;
    float ax = 0.f, ay = 0.f;
    for (int i = 0; i < Tq; i++) {
      const float pij = sp[i * T + j];
      ax += pij * sA[i * D + 2 * c2];
      ay += pij * sA[i * D + 2 * c2 + 1];
    }
    dvb[e] = __float22half2_rn(make_float2(ax, ay));
  }

  // ds[i][j] = alpha * p[i][j] * (dp[i][j] - sum_j dp*p); row per thread
  if (tid < Tq) {
    const float* prow = sp + tid * T;
    float* drow = sds + tid * T;
    float s = 0.f;
    for (int j = 0; j < T; j++) s += drow[j] * prow[j];
    for (int j = 0; j < T; j++)
      drow[j] = alpha * prow[j] * (drow[j] - s);
  }
  __syncthreads();

  // load k over v (v dead), then dq[i][c] = sum_j ds[i][j] * k[j][c]
  const __half2* kb = k + b * (T * D2);
  for (int i = tid; i < T * D2; i += THREADS) {
    const float2 f = __half22float2(kb[i]);
    const float2 b2 = bk ? bk[i % D2] : make_float2(0.f, 0.f);
    sB[2 * i] = f.x + b2.x;
    sB[2 * i + 1] = f.y + b2.y;
  }
  __syncthreads();
  __half2* dqb = dq + b * (Tq * D2);
  for (int e = tid; e < Tq * D2; e += THREADS) {
    const int i = e / D2, c2 = e % D2;
    const float* dsi = sds + i * T;
    float ax = 0.f, ay = 0.f;
    for (int j = 0; j < T; j++) {
      const float d = dsi[j];
      ax += d * sB[j * D + 2 * c2];
      ay += d * sB[j * D + 2 * c2 + 1];
    }
    dqb[e] = __float22half2_rn(make_float2(ax, ay));
  }
  __syncthreads();

  // load q over dout (dout dead), then dk[j][c] = sum_i ds[i][j] * q[i][c]
  const __half2* qb = q + b * (Tq * D2);
  for (int i = tid; i < Tq * D2; i += THREADS) {
    const float2 f = __half22float2(qb[i]);
    const float2 b2 = bq ? bq[i % D2] : make_float2(0.f, 0.f);
    sA[2 * i] = f.x + b2.x;
    sA[2 * i + 1] = f.y + b2.y;
  }
  __syncthreads();
  __half2* dkb = dk + b * (T * D2);
  for (int e = tid; e < T * D2; e += THREADS) {
    const int j = e / D2, c2 = e % D2;
    float ax = 0.f, ay = 0.f;
    for (int i = 0; i < Tq; i++) {
      const float d = sds[i * T + j];
      ax += d * sA[i * D + 2 * c2];
      ay += d * sA[i * D + 2 * c2 + 1];
    }
    dkb[e] = __float22half2_rn(make_float2(ax, ay));
  }
}

void check_in(const at::Tensor& t, const char* name, int64_t B, int rows) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous()
              && t.scalar_type() == at::kHalf, name,
              ": contiguous CUDA fp16 required");
  TORCH_CHECK(t.dim() == 3 && t.size(0) == B && t.size(1) == rows
              && t.size(2) == D, name, ": expected (", B, ",", rows, ",",
              D, "), got ", t.sizes());
}

const float2* bias_ptr(const c10::optional<at::Tensor>& b) {
  if (!b.has_value() || !b->defined()) return nullptr;
  TORCH_CHECK(b->is_cuda() && b->is_contiguous()
              && b->scalar_type() == at::kFloat && b->numel() == D,
              "proj bias: contiguous fp32 (", D, ",)");
  return reinterpret_cast<const float2*>(b->data_ptr<float>());
}

std::vector<at::Tensor> attn_fwd(at::Tensor q, at::Tensor k, at::Tensor v,
                                 at::Tensor bias, double alpha,
                                 c10::optional<at::Tensor> bq,
                                 c10::optional<at::Tensor> bk,
                                 c10::optional<at::Tensor> bv) {
  const c10::cuda::CUDAGuard guard(q.device());
  const int64_t B = q.size(0);
  const int Tq = (int)q.size(1);
  const int T = (int)k.size(1);
  check_in(q, "q", B, Tq);
  check_in(k, "k", B, T);
  check_in(v, "v", B, T);
  TORCH_CHECK(T % 2 == 0 && T <= MAX_T && Tq <= T,
              "T must be even, <= ", MAX_T, ", Tq <= T");
  TORCH_CHECK(bias.is_cuda() && bias.is_contiguous()
              && bias.scalar_type() == at::kHalf && bias.numel() == B * T,
              "bias: contiguous fp16 (B,1,T)");
  auto out = at::empty_like(q);
  auto p = at::empty({B, Tq, T}, q.options());
  if (B == 0) return {out, p};
  const size_t smem = (size_t)(Tq * D + 2 * T * D + Tq * T + T)
                      * sizeof(float);
  attn_fwd_kernel<<<(unsigned)B, THREADS, smem,
                    at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half2*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const __half2*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const __half2*>(v.data_ptr<at::Half>()),
      bias_ptr(bq), bias_ptr(bk), bias_ptr(bv),
      reinterpret_cast<const __half2*>(bias.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(p.data_ptr<at::Half>()),
      reinterpret_cast<__half2*>(out.data_ptr<at::Half>()),
      Tq, T, (float)alpha);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, p};
}

std::vector<at::Tensor> attn_bwd(at::Tensor dout, at::Tensor p, at::Tensor q,
                                 at::Tensor k, at::Tensor v, double alpha,
                                 c10::optional<at::Tensor> bq,
                                 c10::optional<at::Tensor> bk,
                                 c10::optional<at::Tensor> bv) {
  const c10::cuda::CUDAGuard guard(q.device());
  const int64_t B = q.size(0);
  const int Tq = (int)q.size(1);
  const int T = (int)k.size(1);
  check_in(dout, "dout", B, Tq);
  check_in(q, "q", B, Tq);
  check_in(k, "k", B, T);
  check_in(v, "v", B, T);
  TORCH_CHECK(p.is_cuda() && p.is_contiguous()
              && p.scalar_type() == at::kHalf
              && p.sizes() == at::IntArrayRef({B, Tq, T}), "p: (B,Tq,T) fp16");
  auto dq = at::empty_like(q);
  auto dk = at::empty_like(k);
  auto dv = at::empty_like(v);
  if (B == 0) return {dq, dk, dv};
  const int A_rows = Tq > T ? Tq : T;
  const size_t smem = (size_t)(A_rows * D + T * D + 2 * Tq * T)
                      * sizeof(float);
  attn_bwd_kernel<<<(unsigned)B, THREADS, smem,
                    at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half2*>(dout.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(p.data_ptr<at::Half>()),
      reinterpret_cast<const __half2*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const __half2*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const __half2*>(v.data_ptr<at::Half>()),
      bias_ptr(bq), bias_ptr(bk), bias_ptr(bv),
      reinterpret_cast<__half2*>(dq.data_ptr<at::Half>()),
      reinterpret_cast<__half2*>(dk.data_ptr<at::Half>()),
      reinterpret_cast<__half2*>(dv.data_ptr<at::Half>()),
      Tq, T, (float)alpha);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_fwd", &attn_fwd,
        "fused single-head attention forward -> (out, p); optional "
        "bq/bk/bv fold the projection biases into the load",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("bias"),
        py::arg("alpha"), py::arg("bq") = py::none(),
        py::arg("bk") = py::none(), py::arg("bv") = py::none());
  m.def("attn_bwd", &attn_bwd,
        "fused single-head attention backward -> (dq, dk, dv)",
        py::arg("dout"), py::arg("p"), py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("alpha"), py::arg("bq") = py::none(),
        py::arg("bk") = py::none(), py::arg("bv") = py::none());
}
