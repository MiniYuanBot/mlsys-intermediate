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
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    __syncthreads(); // a previous call may still be reading warp_sums
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    const int warps = blockDim.x >> 5;
    v = lane < warps ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
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
    const int row = blockIdx.x;
    if (row >= rows) return; // whole block exits together: no barrier is skipped
    const size_t base = size_t(row) * width;
    float sum = 0.0f;
    if (width % 4 == 0) {
        const int w4 = width / 4;
        const float4 *x4 = reinterpret_cast<const float4 *>(x + base);
        float4 *r4 = reinterpret_cast<float4 *>(residual + base);
        for (int j = threadIdx.x; j < w4; j += blockDim.x) {
            float4 a = x4[j], b = r4[j];
            float4 z = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            r4[j] = z;
            sum += z.x * z.x + z.y * z.y + z.z * z.z + z.w * z.w;
        }
        const float inv = rsqrtf(block_reduce_sum(sum) / width + kEps);
        const float4 *g4 = reinterpret_cast<const float4 *>(gamma);
        float4 *y4 = reinterpret_cast<float4 *>(y + base);
        for (int j = threadIdx.x; j < w4; j += blockDim.x) {
            float4 z = r4[j], g = g4[j];
            y4[j] = make_float4(z.x * inv * g.x, z.y * inv * g.y, z.z * inv * g.z, z.w * inv * g.w);
        }
    } else {
        for (int j = threadIdx.x; j < width; j += blockDim.x) {
            float z = x[base + j] + residual[base + j];
            residual[base + j] = z;
            sum += z * z;
        }
        const float inv = rsqrtf(block_reduce_sum(sum) / width + kEps);
        for (int j = threadIdx.x; j < width; j += blockDim.x)
            y[base + j] = residual[base + j] * inv * gamma[j];
    }
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
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = size_t(row) * width;
    float sum = 0.0f;
    if (width % 2 == 0) {
        const int w2 = width / 2;
        const __half2 *x2 = reinterpret_cast<const __half2 *>(x + base);
        __half2 *r2 = reinterpret_cast<__half2 *>(residual + base);
        for (int j = threadIdx.x; j < w2; j += blockDim.x) {
            float2 a = __half22float2(x2[j]), b = __half22float2(r2[j]);
            float2 z = make_float2(a.x + b.x, a.y + b.y);
            r2[j] = __floats2half2_rn(z.x, z.y);
            sum += z.x * z.x + z.y * z.y;
        }
        const float inv = rsqrtf(block_reduce_sum(sum) / width + kEps);
        const __half2 *g2 = reinterpret_cast<const __half2 *>(gamma);
        __half2 *y2 = reinterpret_cast<__half2 *>(y + base);
        for (int j = threadIdx.x; j < w2; j += blockDim.x) {
            float2 z = __half22float2(r2[j]), g = __half22float2(g2[j]);
            y2[j] = __floats2half2_rn(z.x * inv * g.x, z.y * inv * g.y);
        }
    } else {
        for (int j = threadIdx.x; j < width; j += blockDim.x) {
            float z = __half2float(x[base + j]) + __half2float(residual[base + j]);
            residual[base + j] = __float2half_rn(z);
            sum += z * z;
        }
        const float inv = rsqrtf(block_reduce_sum(sum) / width + kEps);
        for (int j = threadIdx.x; j < width; j += blockDim.x)
            y[base + j] =
                __float2half_rn(__half2float(residual[base + j]) * inv * __half2float(gamma[j]));
    }
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
