// Stage R (GPU half): cudaEvent timing harness and the three microbenchmarks that calibrate
// the roofline efficiencies (copy -> eta_m, GEMV -> decode-like, WMMA GEMM -> eta_c).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hs {

#define HS_CUDA(expr)                                                                              \
    do {                                                                                           \
        cudaError_t err_ = (expr);                                                                 \
        if (err_ != cudaSuccess)                                                                   \
            throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(err_));        \
    } while (0)

struct TimingStats {
    double median_ms = 0, q25_ms = 0, q75_ms = 0, min_ms = 0, max_ms = 0;
    int reps = 0;
};

// ============================================================================ R1
// time_cuda: GPU time of `launch()` (which enqueues work on the default stream).
//   1. call launch() `warmup` times, then check cudaGetLastError and synchronize;
//   2. for each of `reps` repetitions: record a start event, launch(), record a stop event,
//      cudaEventSynchronize(stop), cudaEventElapsedTime -> one sample in ms;
//   3. check every CUDA call (wrap it in HS_CUDA so errors throw), destroy the events;
//   4. statistics over the `reps` samples only (never the warmup): min, max, and the 25/50/75
//      percentiles with linear interpolation: sorted v, h = (n - 1) p,
//      q(p) = v[floor h] + (h - floor h) (v[floor h + 1] - v[floor h]).
// warmup < 0 or reps < 1 -> std::invalid_argument.
// Oracles (profile.cu): a globaltimer spin kernel of known length; a launch whose warmup calls
//   are 20x slower; a skewed duration pattern whose median, q25 and q75 are known.
template <class Launch> TimingStats time_cuda(Launch &&launch, int warmup, int reps) {
    // BEGIN TODO R1
    throw std::logic_error("Complete R1");
    // END TODO R1
}

// ============================================================================ R2
// copy_vec4_kernel: out[i] = in[i] for i < n with 16-byte (float4) loads/stores for the first
//   4 * floor(n / 4) elements and scalar accesses for the tail; grid-stride loops (the
//   launcher caps the grid, so one pass per thread is NOT enough). in/out are cudaMalloc'ed
//   (256-byte aligned). Effective bandwidth = 2 * 4n bytes / time.
__global__ void copy_vec4_kernel(const float *__restrict__ in, float *__restrict__ out, size_t n) {
    // BEGIN TODO R2
    asm("trap;");
    // END TODO R2
}

// ============================================================================ R3
// gemv_fp16_kernel: y[row] = sum_k W[row, k] x[k], W row-major M x K (FP16), x FP16, y FP32
//   with FP32 accumulation. Launch (provided): 256 threads = 8 warps per block, one warp per
//   row, row = blockIdx.x * 8 + warp. K is a multiple of 8: each lane reads 8 halves (16 bytes)
//   at a time, then the warp reduces with __shfl_down_sync (offsets 16, 8, 4, 2, 1) and lane 0
//   writes y[row]. Rows >= M do nothing.
__global__ void gemv_fp16_kernel(const __half *__restrict__ W, const __half *__restrict__ x,
                                 float *__restrict__ y, int M, int K) {
    // BEGIN TODO R3
    asm("trap;");
    // END TODO R3
}

// ============================================================================ R4
// gemm_wmma_kernel: C = A B with A (M x K), B (K x N) row-major FP16, C (M x N) row-major FP32,
//   FP32 accumulation on tensor cores (nvcuda::wmma 16x16x16). Launch (provided): grid
//   (N/64, M/64), 128 threads = 4 warps; block tile 64 x 64, K step 32. Each warp owns a 32 x 32
//   quadrant (warp_m = warp / 2, warp_n = warp % 2) = 2 x 2 accumulator fragments.
//   Per K step: all threads copy the 64 x 32 A tile and the 32 x 64 B tile into shared memory
//   with 16-byte loads (padded rows: As[64][40], Bs[32][72]), __syncthreads, each warp does two
//   16-deep MMA steps, __syncthreads. Finally store fragments straight to C (ldm = N).
//   Preconditions (checked by the launcher): M, N multiples of 64, K multiple of 32.
constexpr int kGemmBM = 64, kGemmBN = 64, kGemmBK = 32;
__global__ void gemm_wmma_kernel(const __half *__restrict__ A, const __half *__restrict__ B,
                                 float *__restrict__ C, int M, int N, int K) {
    // BEGIN TODO R4
    asm("trap;");
    // END TODO R4
}

// ---------------------------------------------------------------- provided launchers
inline int sm_count() {
    static int count = 0;
    if (!count) {
        int dev = 0;
        HS_CUDA(cudaGetDevice(&dev));
        HS_CUDA(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, dev));
    }
    return count;
}

inline void launch_copy(const float *in, float *out, size_t n) {
    size_t want = (n / 4 + 255) / 256;
    int blocks =
        static_cast<int>(std::max<size_t>(1, std::min<size_t>(want, size_t(sm_count()) * 8)));
    copy_vec4_kernel<<<blocks, 256>>>(in, out, n);
}

inline void launch_gemv(const __half *W, const __half *x, float *y, int M, int K) {
    if (M < 1 || K < 8 || K % 8)
        throw std::invalid_argument("gemv: K must be a positive multiple of 8");
    gemv_fp16_kernel<<<(M + 7) / 8, 256>>>(W, x, y, M, K);
}

inline void launch_gemm(const __half *A, const __half *B, float *C, int M, int N, int K) {
    if (M < 64 || N < 64 || K < 32 || M % 64 || N % 64 || K % 32)
        throw std::invalid_argument("gemm: need M, N multiples of 64 and K multiple of 32");
    dim3 grid(N / kGemmBN, M / kGemmBM);
    gemm_wmma_kernel<<<grid, 128>>>(A, B, C, M, N, K);
}

} // namespace hs
