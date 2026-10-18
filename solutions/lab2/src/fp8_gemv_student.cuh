// Stage R device side: weight-only FP8 (E4M3) GEMV and small-batch GEMM, W8A32.
// Layout (R2): codes [rows, stride] with stride = cols rounded up to 16, so every row
// starts 16-byte aligned (cudaMalloc returns >= 256-byte aligned pointers); scales
// [rows, ceil(cols / group)] FP32 with group a multiple of 16. X is FP32 [batch, cols]
// and Y is FP32 [batch, rows]. sm_89 converts two E4M3 values to a half2 in one
// instruction (cvt.rn.f16x2.e4m3x2); every E4M3 value is exact in FP16.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace fp8 {
constexpr int kTile = 8; // tokens per R4 tile

// R3: one warp per output row; lane l reads the 16-byte chunks l, l+32, ... of its
// row with ONE 128-bit load each (uint4), converts the 16 codes in registers
// (__nv_cvt_fp8x2_to_halfraw2(.., __NV_E4M3) or cvt PTX), multiplies by x and by the
// chunk's group scale, then the warp reduces with shuffles and lane 0 writes y[row].
// Columns >= cols are padding: never read x beyond cols and never use padding codes
// (the test fills them with 0x7F = NaN and puts NaN right after the end of x).
// Rows past `rows` return as a whole warp. x points to a cudaMalloc'ed row, so when
// cols % 4 == 0 every x row is 16-byte aligned and float4 loads of x are allowed.
// Test: R3 vs FP64 decode of the same codes.
__global__ void fp8_gemv(const uint8_t *codes, const float *scales, const float *x, float *y,
                         int rows, int cols, int group) {
    // BEGIN TODO R3
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= rows) return;
    const int stride = (cols + 15) / 16 * 16, chunks = stride / 16;
    const int ng = (cols + group - 1) / group;
    const uint4 *w = reinterpret_cast<const uint4 *>(codes + size_t(row) * stride);
    float sum = 0;
    const bool aligned = (cols % 4) == 0; // x rows are 16-byte aligned, float4 loads ok
    for (int ch = lane; ch < chunks; ch += 32) {
        const uint4 v = w[ch]; // one 128-bit load = 16 weights
        const unsigned words[4] = {v.x, v.y, v.z, v.w};
        const int c0 = ch * 16;
        float wv[16];
#pragma unroll
        for (int j = 0; j < 4; ++j) { // cvt.rn.f16x2.e4m3x2: two codes per instruction
            const float2 a = __half22float2(__half2(
                __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(words[j] & 0xffffu), __NV_E4M3)));
            const float2 b = __half22float2(__half2(
                __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(words[j] >> 16), __NV_E4M3)));
            wv[4 * j] = a.x, wv[4 * j + 1] = a.y, wv[4 * j + 2] = b.x, wv[4 * j + 3] = b.y;
        }
        float part = 0;
        if (aligned && c0 + 16 <= cols) {
            const float4 *xv = reinterpret_cast<const float4 *>(x + c0);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 q = xv[j];
                part = fmaf(wv[4 * j], q.x, part), part = fmaf(wv[4 * j + 1], q.y, part);
                part = fmaf(wv[4 * j + 2], q.z, part), part = fmaf(wv[4 * j + 3], q.w, part);
            }
        } else {
            for (int i = 0; i < 16 && c0 + i < cols; ++i) part = fmaf(wv[i], x[c0 + i], part);
        }
        sum = fmaf(scales[size_t(row) * ng + c0 / group], part, sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) y[row] = sum;
    // END TODO R3
}

// R4: like R3, but grid.y selects a tile of kTile tokens and each converted weight
// is reused for all tokens of the tile (kTile FP32 accumulators per lane). The last
// tile may be partial: tokens >= batch are neither read nor written, yet every lane
// still takes part in all shuffles. Test: R4 batches 1/3/8/9/17 vs FP64.
__global__ void fp8_tiled_batch(const uint8_t *codes, const float *scales, const float *x, float *y,
                                int rows, int cols, int group, int batch) {
    // BEGIN TODO R4
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= rows) return;
    const int t0 = blockIdx.y * kTile;
    const int tokens = min(kTile, batch - t0);
    const int stride = (cols + 15) / 16 * 16, chunks = stride / 16;
    const int ng = (cols + group - 1) / group;
    const uint4 *w = reinterpret_cast<const uint4 *>(codes + size_t(row) * stride);
    const bool aligned = (cols % 4) == 0;
    float sums[kTile];
#pragma unroll
    for (int t = 0; t < kTile; ++t) sums[t] = 0;
    for (int ch = lane; ch < chunks; ch += 32) {
        const uint4 v = w[ch];
        const unsigned words[4] = {v.x, v.y, v.z, v.w};
        const int c0 = ch * 16;
        const float s = scales[size_t(row) * ng + c0 / group];
        float wv[16];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = __half22float2(__half2(
                __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(words[j] & 0xffffu), __NV_E4M3)));
            const float2 b = __half22float2(__half2(
                __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(words[j] >> 16), __NV_E4M3)));
            wv[4 * j] = a.x * s, wv[4 * j + 1] = a.y * s;
            wv[4 * j + 2] = b.x * s, wv[4 * j + 3] = b.y * s;
        }
        if (aligned && c0 + 16 <= cols) {
#pragma unroll
            for (int t = 0; t < kTile; ++t) {
                if (t < tokens) {
                    const float4 *xv =
                        reinterpret_cast<const float4 *>(x + size_t(t0 + t) * cols + c0);
                    float acc = sums[t];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float4 q = xv[j];
                        acc = fmaf(wv[4 * j], q.x, acc), acc = fmaf(wv[4 * j + 1], q.y, acc);
                        acc = fmaf(wv[4 * j + 2], q.z, acc), acc = fmaf(wv[4 * j + 3], q.w, acc);
                    }
                    sums[t] = acc;
                }
            }
        } else {
            for (int t = 0; t < tokens; ++t)
                for (int i = 0; i < 16 && c0 + i < cols; ++i)
                    sums[t] = fmaf(wv[i], x[size_t(t0 + t) * cols + c0 + i], sums[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < kTile; ++t) {
        for (int offset = 16; offset > 0; offset >>= 1)
            sums[t] += __shfl_down_sync(0xffffffffu, sums[t], offset);
        if (lane == 0 && t < tokens) y[size_t(t0 + t) * rows + row] = sums[t];
    }
    // END TODO R4
}

inline void launch_fp8_gemv(const uint8_t *codes, const float *scales, const float *x, float *y,
                            int rows, int cols, int group) {
    fp8_gemv<<<(rows + 3) / 4, 128>>>(codes, scales, x, y, rows, cols, group);
}
inline void launch_fp8_batch(const uint8_t *codes, const float *scales, const float *x, float *y,
                             int rows, int cols, int group, int batch) {
    fp8_tiled_batch<<<dim3((rows + 3) / 4, (batch + kTile - 1) / kTile), 128>>>(
        codes, scales, x, y, rows, cols, group, batch);
}
} // namespace fp8
