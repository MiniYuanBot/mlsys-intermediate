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
    if (warmup < 0 || reps < 1) throw std::invalid_argument("time_cuda: bad counts");
    for (int i = 0; i < warmup; ++i) launch();
    HS_CUDA(cudaGetLastError());
    HS_CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    HS_CUDA(cudaEventCreate(&start));
    HS_CUDA(cudaEventCreate(&stop));
    std::vector<double> ms;
    for (int r = 0; r < reps; ++r) {
        HS_CUDA(cudaEventRecord(start));
        launch();
        HS_CUDA(cudaEventRecord(stop));
        HS_CUDA(cudaEventSynchronize(stop));
        float x = 0;
        HS_CUDA(cudaEventElapsedTime(&x, start, stop));
        ms.push_back(x);
    }
    HS_CUDA(cudaGetLastError());
    HS_CUDA(cudaEventDestroy(start));
    HS_CUDA(cudaEventDestroy(stop));
    std::sort(ms.begin(), ms.end());
    auto q = [&](double p) {
        double h = (ms.size() - 1) * p;
        size_t lo = static_cast<size_t>(std::floor(h));
        size_t hi = std::min(lo + 1, ms.size() - 1);
        return ms[lo] + (h - lo) * (ms[hi] - ms[lo]);
    };
    TimingStats s;
    s.reps = reps;
    s.min_ms = ms.front(), s.max_ms = ms.back();
    s.q25_ms = q(0.25), s.median_ms = q(0.5), s.q75_ms = q(0.75);
    return s;
    // END TODO R1
}

// ============================================================================ R2
// copy_vec4_kernel: out[i] = in[i] for i < n with 16-byte (float4) loads/stores for the first
//   4 * floor(n / 4) elements and scalar accesses for the tail; grid-stride loops (the
//   launcher caps the grid, so one pass per thread is NOT enough). in/out are cudaMalloc'ed
//   (256-byte aligned). Effective bandwidth = 2 * 4n bytes / time.
__global__ void copy_vec4_kernel(const float *__restrict__ in, float *__restrict__ out, size_t n) {
    // BEGIN TODO R2
    size_t tid = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    size_t n4 = n / 4;
    const float4 *in4 = reinterpret_cast<const float4 *>(in);
    float4 *out4 = reinterpret_cast<float4 *>(out);
    for (size_t i = tid; i < n4; i += stride) out4[i] = in4[i];
    for (size_t i = 4 * n4 + tid; i < n; i += stride) out[i] = in[i];
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
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x / 32) + warp;
    if (row >= M) return;
    const uint4 *w8 = reinterpret_cast<const uint4 *>(W + size_t(row) * K);
    const uint4 *x8 = reinterpret_cast<const uint4 *>(x);
    float acc = 0.f;
    for (int i = lane; i < K / 8; i += 32) {
        uint4 wv = w8[i], xv = x8[i];
        const __half2 *wh = reinterpret_cast<const __half2 *>(&wv);
        const __half2 *xh = reinterpret_cast<const __half2 *>(&xv);
        for (int j = 0; j < 4; ++j) {
            float2 a = __half22float2(wh[j]), b = __half22float2(xh[j]);
            acc += a.x * b.x + a.y * b.y;
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
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
    using namespace nvcuda;
    __shared__ __align__(32) __half As[kGemmBM][kGemmBK + 8];
    __shared__ __align__(32) __half Bs[kGemmBK][kGemmBN + 8];
    int tid = threadIdx.x, warp = tid / 32;
    int warp_m = warp / 2, warp_n = warp % 2;
    int bm = blockIdx.y * kGemmBM, bn = blockIdx.x * kGemmBN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    for (int k0 = 0; k0 < K; k0 += kGemmBK) {
        for (int r = 0; r < 2; ++r) {
            int idx = tid + r * 128;
            int arow = idx / 4, acol = (idx % 4) * 8;
            *reinterpret_cast<uint4 *>(&As[arow][acol]) =
                *reinterpret_cast<const uint4 *>(&A[size_t(bm + arow) * K + k0 + acol]);
            int brow = idx / 8, bcol = (idx % 8) * 8;
            *reinterpret_cast<uint4 *>(&Bs[brow][bcol]) =
                *reinterpret_cast<const uint4 *>(&B[size_t(k0 + brow) * N + bn + bcol]);
        }
        __syncthreads();
        for (int kk = 0; kk < kGemmBK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb[2];
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(fa[i], &As[warp_m * 32 + i * 16][kk], kGemmBK + 8);
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(fb[j], &Bs[kk][warp_n * 32 + j * 16], kGemmBN + 8);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            float *dst = C + size_t(bm + warp_m * 32 + i * 16) * N + bn + warp_n * 32 + j * 16;
            wmma::store_matrix_sync(dst, acc[i][j], N, wmma::mem_row_major);
        }
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
