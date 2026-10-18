#pragma once
// Fused residual-add + RMSNorm, the pattern every pre-norm decoder layer runs twice:
//   z = x + residual;  residual <- z (write-back);  y = z * rsqrt(mean(z^2) + eps) * gamma
// One block of kThreads threads per row. No PyTorch: test.cu checks against FP64 on the host.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

constexpr float kEps = 1e-5f;
constexpr int kThreads = 256; // launch width; must be a multiple of 32 and <= 1024

// C5: sum `v` over every thread of the block and return the total to EVERY thread.
// Reduce inside each warp with __shfl_xor_sync over the full 32-lane mask (offsets 16..1),
// write one partial per warp to __shared__ memory, then let each warp reduce the
// blockDim.x / 32 partials again. Must be callable several times in a row from the same
// kernel (protect the shared buffer before it is overwritten). Every thread of the block
// must call it (it contains __syncthreads). Test: test.cu --only reduce (blockDim 32..1024,
// two consecutive calls, exact integer sums, all threads see the total).
__device__ __forceinline__ float block_reduce_sum(float v) {
    // BEGIN TODO C5
    asm("trap;");
    return 0.0f;
    // END TODO C5
}

// C6: FP32 fused residual-add + RMSNorm for row blockIdx.x.
// Pass 1: z = x + residual, store z back into residual, accumulate z*z. Pass 2 (after
// block_reduce_sum): read residual back and write y = z * inv * gamma with
// inv = rsqrtf(sum / width + kEps); divide by the TRUE width.
// When width % 4 == 0 use float4 loads/stores (rows are 16-byte aligned because
// cudaMalloc returns 256-byte aligned pointers); otherwise use scalar accesses.
// Each thread must touch the same element indices in both passes. Every thread calls
// block_reduce_sum even if it owns no element. Test: test.cu --only f32 (odd widths,
// FP64 oracle, exact residual write-back).
__global__ void fused_add_rmsnorm_f32(const float *__restrict__ x, float *__restrict__ residual,
                                      const float *__restrict__ gamma, float *__restrict__ y,
                                      int rows, int width) {
    // BEGIN TODO C6
    asm("trap;");
    // END TODO C6
}

// C7: FP16 storage, FP32 math. z = float(x) + float(residual); residual <- __half(z)
// (round to nearest); accumulate z*z in FP32 (an FP16 sum overflows above 65504);
// pass 2 reads the rounded residual back and writes y = __half(z * inv * gamma).
// When width % 2 == 0 use __half2 loads/stores (__half22float2 / __floats2half2_rn);
// otherwise scalar __half. Test: test.cu --only f16 (odd widths, large-magnitude rows).
__global__ void fused_add_rmsnorm_f16(const __half *__restrict__ x, __half *__restrict__ residual,
                                      const __half *__restrict__ gamma, __half *__restrict__ y,
                                      int rows, int width) {
    // BEGIN TODO C7
    asm("trap;");
    // END TODO C7
}

inline void launch_fused_f32(const float *x, float *residual, const float *gamma, float *y,
                             int rows, int width, cudaStream_t stream = 0) {
    if (rows <= 0 || width <= 0) throw std::invalid_argument("rows and width must be positive");
    fused_add_rmsnorm_f32<<<rows, kThreads, 0, stream>>>(x, residual, gamma, y, rows, width);
}

inline void launch_fused_f16(const __half *x, __half *residual, const __half *gamma, __half *y,
                             int rows, int width, cudaStream_t stream = 0) {
    if (rows <= 0 || width <= 0) throw std::invalid_argument("rows and width must be positive");
    fused_add_rmsnorm_f16<<<rows, kThreads, 0, stream>>>(x, residual, gamma, y, rows, width);
}
