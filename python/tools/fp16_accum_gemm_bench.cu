// cuBLASLt microbenchmark kernel for tools/fp16_accum_gemm_bench.py
// (issue #3): the policy's GEMM shapes with our epilogues, COMPUTE_32F vs
// COMPUTE_16F, any of the heuristic's top-16 algos.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <map>
#include <tuple>
#include <vector>

// kind: 0 fwd plain, 1 fwd bias, 2 fwd relu_aux_bias, 3 dgrad plain, 4 dgrad drelu
// c16: 0 -> COMPUTE_32F/scale 32F ; 1 -> COMPUTE_16F/scale 16F
struct P { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc; cublasLtMatmulAlgo_t algo; size_t ws; bool ok; int nalgo; std::vector<cublasLtMatmulHeuristicResult_t> all; };
static std::map<std::tuple<int,int,int64_t,int64_t,int64_t>, P> plans;
static inline int64_t mld(int64_t H){return (H+127)/128*128;}

P& plan(int kind, int c16, int64_t R, int64_t N, int64_t K) {
  auto key = std::make_tuple(kind, c16, R, N, K);
  auto it = plans.find(key); if (it != plans.end()) return it->second;
  P p{}; auto ck=[](cublasStatus_t s){return s==CUBLAS_STATUS_SUCCESS;};
  bool ok = ck(cublasLtMatmulDescCreate(&p.op, c16?CUBLAS_COMPUTE_16F:CUBLAS_COMPUTE_32F, c16?CUDA_R_16F:CUDA_R_32F));
  cublasOperation_t tn=CUBLAS_OP_N, tt=CUBLAS_OP_T;
  bool fwd = kind<3;
  cublasLtEpilogue_t epi = kind==1?CUBLASLT_EPILOGUE_BIAS: kind==2?CUBLASLT_EPILOGUE_RELU_AUX_BIAS: kind==4?CUBLASLT_EPILOGUE_DRELU: CUBLASLT_EPILOGUE_DEFAULT;
  // fwd: C(N,R) = W(K,N)^T x(K,R) ; dgrad: C(K,R) = W(K,N) g(N,R)
  int64_t m = fwd?N:K, kk = fwd?K:N;
  ok = ok && ck(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, fwd?&tt:&tn, sizeof(tn)));
  ok = ok && ck(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof(tn)));
  ok = ok && ck(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
  if (kind==2||kind==4){ int64_t ld=mld(m); ok = ok && ck(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld, sizeof(ld))); }
  ok = ok && ck(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16F, K, N, K)); // W row-major (N,K) = col (K,N)
  ok = ok && ck(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16F, kk, R, kk));
  ok = ok && ck(cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16F, m, R, m));
  if (ok) {
    cublasLtMatmulPreference_t pref; ck(cublasLtMatmulPreferenceCreate(&pref));
    size_t w = 32u<<20; cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&w,sizeof(w));
    cublasLtMatmulHeuristicResult_t res[16]{}; int n=0;
    ok = ck(cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(), p.op, p.la,p.lb,p.lc,p.lc,pref,16,res,&n)) && n>0;
    if (ok){p.algo=res[0].algo; p.ws=res[0].workspaceSize; for(int i=0;i<n;i++) p.all.push_back(res[i]);}
    p.nalgo = n;
    cublasLtMatmulPreferenceDestroy(pref);
  }
  p.ok = ok; return plans.emplace(key,p).first->second;
}

// x: fwd input (R,K) / dgrad g (R,N); w (N,K); bias (N) half; returns out
std::vector<at::Tensor> run(int kind, int c16, at::Tensor x, at::Tensor w, at::Tensor bias, at::Tensor aux, int iters, int ai) {
  int64_t R=x.size(0), N=w.size(0), K=w.size(1);
  bool fwd = kind<3;
  P& p = plan(kind,c16,R,N,K);
  if (!p.ok || ai >= (int)p.all.size()) return {};
  int64_t m = fwd?N:K;
  cublasLtMatmulAlgo_t algo = p.all[ai].algo; size_t wsz = p.all[ai].workspaceSize;
  auto y = at::empty({R,m}, x.options());
  if (kind==2 && (!aux.defined() || aux.numel()==0)) aux = at::empty({R, mld(m)/8}, x.options().dtype(at::kByte));
  if (kind==1||kind==2){ void* bp=bias.data_ptr(); cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bp,sizeof(bp)); }
  if (kind==2||kind==4){ void* ap=aux.data_ptr(); cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,&ap,sizeof(ap)); }
  at::Tensor ws; if (wsz) ws = at::empty({(int64_t)wsz}, x.options().dtype(at::kByte));
  float a32=1.f,b32=0.f; __half a16=__float2half(1.f), b16=__float2half(0.f);
  const void* al = c16?(const void*)&a16:(const void*)&a32; const void* be = c16?(const void*)&b16:(const void*)&b32;
  auto st = at::cuda::getCurrentCUDAStream();
  for (int i=0;i<iters;i++)
    TORCH_CUDABLAS_CHECK(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), p.op, al, w.data_ptr(), p.la, x.data_ptr(), p.lb, be, y.data_ptr(), p.lc, y.data_ptr(), p.lc, &algo, wsz?ws.data_ptr():nullptr, wsz, st));
  return {y, aux.defined()?aux:at::Tensor()};
}
int nalgo(int kind, int c16, int64_t R, int64_t N, int64_t K){ return (int)plan(kind,c16,R,N,K).all.size(); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){ m.def("run",&run); m.def("nalgo",&nalgo); }
