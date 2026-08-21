// Warp-per-env go/no-go micro-benchmark (2026-08-21).
//
// Question: on the real Beat Arena shapes (NV=35 dense SPD factor +
// triangular solves, f32, 8 substeps/tick), does 32-lane cooperative
// linear algebra beat the serial thread-per-env transcription ONCE THE
// RESIDENCY DIFFERENCE IS PRICED IN?
//
// Three kernels, same total work (N envs x S substeps of: chol factor
// 35x35 + 1 fwd/back solve + a 12-wide multi-RHS fwd solve — the
// production per-substep chol menu, shapes from beat_arena_cuda.cu):
//   A: thread-per-env, serial right-looking factor (BEAT_CHOL_RL form),
//      M in local memory — the stock architecture.
//   B: warp-per-env, M in shared memory, cooperative factor/solves
//      (lane-parallel scale + rank-1 update, __syncwarp barriers).
//   C: warp-per-env like B, but the serial fraction is emulated: lane 0
//      alone re-runs the factor's j-loop scale step (stand-in for the
//      FK/RNEA/PGS code that a partial rewrite leaves serial).
// Timed with CUDA events over the whole grid; prints ms and env-steps/s.
#include <cstdio>
#include <cuda_runtime.h>

constexpr int NV = 35;
constexpr int NW = 12;      // multi-RHS width (weld/contact solve tier)
constexpr float PD_FLOOR = 1e-8f;

__device__ __forceinline__ float warp_bcast(float v, int src) {
  return __shfl_sync(0xffffffffu, v, src);
}

// ---------- serial reference pieces (transcribed shapes) ----------
__device__ void chol_factor_serial(float* a, bool* bad) {
  for (int j = 0; j < NV; j++) {
    const int cj = j * NV;
    float dpiv = a[cj + j];
    if (dpiv <= 0.f) *bad = true;
    dpiv = fmaxf(dpiv, PD_FLOOR);
    const float dj = sqrtf(dpiv);
    a[cj + j] = dj;
    for (int i = j + 1; i < NV; i++) a[cj + i] /= dj;
    for (int c = j + 1; c < NV; c++) {
      const int cc = c * NV;
      const float ljc = a[cj + c];
      for (int i = c; i < NV; i++) a[cc + i] -= a[cj + i] * ljc;
    }
  }
}
__device__ void solve_serial(const float* a, float* x) {
  for (int i = 0; i < NV; i++) {           // forward
    float s = x[i];
    for (int k = 0; k < i; k++) s -= a[k * NV + i] * x[k];
    x[i] = s / a[i * NV + i];
  }
  for (int i = NV - 1; i >= 0; i--) {      // backward
    float s = x[i];
    for (int k = i + 1; k < NV; k++) s -= a[i * NV + k] * x[k];
    x[i] = s / a[i * NV + i];
  }
}
__device__ void solve_multi_serial(const float* a, float* xw /*NV x NW*/) {
  for (int i = 0; i < NV; i++) {
    const float d = a[i * NV + i];
    for (int w = 0; w < NW; w++) {
      float s = xw[i * NW + w];
      for (int k = 0; k < i; k++) s -= a[k * NV + i] * xw[k * NW + w];
      xw[i * NW + w] = s / d;
    }
  }
}

// ---------- A: thread-per-env, local memory ----------
__global__ void __launch_bounds__(32, 1) kA(int n_env, int substeps,
                                            const float* m0, float* out) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_env) return;
  float a[NV * NV], x[NV], xw[NV * NW];
  bool bad = false;
  float acc = 0.f;
  for (int s = 0; s < substeps; s++) {
    for (int i = 0; i < NV * NV; i++) a[i] = m0[i] + (s == 0 ? 0.f : acc * 1e-20f);
    for (int i = 0; i < NV; i++) x[i] = 1.f + i;
    for (int i = 0; i < NV * NW; i++) xw[i] = 1.f + (i % 7);
    chol_factor_serial(a, &bad);
    solve_serial(a, x);
    solve_multi_serial(a, xw);
    acc += x[0] + xw[5];
  }
  out[e] = acc + (bad ? 1.f : 0.f);
}

// ---------- B: warp-per-env, cooperative, shared memory ----------
__device__ void chol_factor_coop(float* a /*smem*/, bool* bad, int lane) {
  for (int j = 0; j < NV; j++) {
    const int cj = j * NV;
    float dj;
    if (lane == 0) {
      float dpiv = a[cj + j];
      if (dpiv <= 0.f) *bad = true;
      dpiv = fmaxf(dpiv, PD_FLOOR);
      dj = sqrtf(dpiv);
      a[cj + j] = dj;
    }
    dj = warp_bcast(dj, 0);
    for (int i = j + 1 + lane; i < NV; i += 32) a[cj + i] /= dj;
    __syncwarp();
    // rank-1 update: COLUMN-PER-LANE — lane owns cols c = j+1+lane,
    // +32, ...; rows of one column run serially in-lane (contiguous
    // smem, no index arithmetic, no cross-lane dependency)
    for (int c = j + 1 + lane; c < NV; c += 32) {
      const int cc = c * NV;
      const float ljc = a[cj + c];
      for (int i = c; i < NV; i++) a[cc + i] -= a[cj + i] * ljc;
    }
    __syncwarp();
  }
}
__device__ void solve_coop(const float* a, float* x, int lane) {
  // forward: sequential over i, lanes split the dot product
  for (int i = 0; i < NV; i++) {
    float s = 0.f;
    for (int k = lane; k < i; k += 32) s += a[k * NV + i] * x[k];
    for (int off = 16; off; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    if (lane == 0) x[i] = (x[i] - s) / a[i * NV + i];
    __syncwarp();
  }
  for (int i = NV - 1; i >= 0; i--) {
    float s = 0.f;
    for (int k = i + 1 + lane; k < NV; k += 32) s += a[i * NV + k] * x[k];
    for (int off = 16; off; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    if (lane == 0) x[i] = (x[i] - s) / a[i * NV + i];
    __syncwarp();
  }
}
__device__ void solve_multi_coop(const float* a, float* xw, int lane) {
  // lanes own RHS columns (12 of 32 active) — the natural map
  for (int i = 0; i < NV; i++) {
    const float d = a[i * NV + i];
    if (lane < NW) {
      float s = xw[i * NW + lane];
      for (int k = 0; k < i; k++) s -= a[k * NV + i] * xw[k * NW + lane];
      xw[i * NW + lane] = s / d;
    }
    __syncwarp();
  }
}

template <bool EMULATE_SERIAL>
__global__ void __launch_bounds__(32, 1) kB(int n_env, int substeps,
                                            const float* m0, float* out) {
  const int e = blockIdx.x;
  if (e >= n_env) return;
  const int lane = threadIdx.x;
  __shared__ float a[NV * NV];
  __shared__ float x[NV];
  __shared__ float xw[NV * NW];
  __shared__ bool bad;
  if (lane == 0) bad = false;
  float acc = 0.f;
  for (int s = 0; s < substeps; s++) {
    for (int i = lane; i < NV * NV; i += 32) a[i] = m0[i] + (s == 0 ? 0.f : acc * 1e-20f);
    for (int i = lane; i < NV; i += 32) x[i] = 1.f + i;
    for (int i = lane; i < NV * NW; i += 32) xw[i] = 1.f + (i % 7);
    __syncwarp();
    if (EMULATE_SERIAL && lane == 0) {
      // stand-in for the code a partial rewrite leaves serial (FK/RNEA/
      // contacts/PGS/integrate/obs): ~the same op count again, serial
      float t = 0.f;
      for (int i = 0; i < NV * NV; i++) t += a[i] * 1e-9f;
      for (int j = 0; j < NV; j++)
        for (int i = 0; i < NV; i++) t += a[j * NV + i] * a[i];
      acc += t * 1e-20f;
    }
    __syncwarp();
    chol_factor_coop(a, &bad, lane);
    solve_coop(a, x, lane);
    solve_multi_coop(a, xw, lane);
    __syncwarp();
    acc += x[0] + xw[5];
  }
  if (lane == 0) out[e] = acc + (bad ? 1.f : 0.f);
}

static void fill_spd(float* m) {
  // SPD: A = B^T B + NV*I with fixed pseudo-random B
  float b[NV * NV];
  unsigned st = 12345;
  for (int i = 0; i < NV * NV; i++) {
    st = st * 1664525u + 1013904223u;
    b[i] = ((st >> 8) & 0xffff) / 65535.0f - 0.5f;
  }
  for (int c = 0; c < NV; c++)
    for (int r = 0; r < NV; r++) {
      float s = (r == c) ? float(NV) : 0.f;
      for (int k = 0; k < NV; k++) s += b[k * NV + r] * b[k * NV + c];
      m[c * NV + r] = s;
    }
}

int main(int argc, char** argv) {
  const int n_env = argc > 1 ? atoi(argv[1]) : 32768;
  const int substeps = 8, reps = 30;
  float hm[NV * NV];
  fill_spd(hm);
  float *dm, *dout;
  cudaMalloc(&dm, sizeof(hm));
  cudaMalloc(&dout, n_env * sizeof(float));
  cudaMemcpy(dm, hm, sizeof(hm), cudaMemcpyHostToDevice);
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  auto run = [&](const char* name, auto launch) {
    launch();  // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int r = 0; r < reps; r++) launch();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    ms /= reps;
    printf("%-28s %8.3f ms  %10.1f k env-ticks/s\n", name, ms,
           n_env / ms);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("  ERROR: %s\n", cudaGetErrorString(err));
  };
  run("A thread-per-env (local)", [&]{ kA<<<(n_env + 31) / 32, 32>>>(n_env, substeps, dm, dout); });
  run("B warp-per-env coop", [&]{ kB<false><<<n_env, 32>>>(n_env, substeps, dm, dout); });
  run("C coop + serial-emu", [&]{ kB<true><<<n_env, 32>>>(n_env, substeps, dm, dout); });
  // correctness spot check: A vs B first env
  float a0, b0;
  kA<<<1, 32>>>(1, substeps, dm, dout);
  cudaMemcpy(&a0, dout, 4, cudaMemcpyDeviceToHost);
  kB<false><<<1, 32>>>(1, substeps, dm, dout);
  cudaMemcpy(&b0, dout, 4, cudaMemcpyDeviceToHost);
  printf("check A=%g B=%g (tolerance-tier: reassociated)\n", a0, b0);
  return 0;
}
