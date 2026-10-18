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
    asm("trap;"); // Complete R3
    // END TODO R3
}

// R4: like R3, but grid.y selects a tile of kTile tokens and each converted weight
// is reused for all tokens of the tile (kTile FP32 accumulators per lane). The last
// tile may be partial: tokens >= batch are neither read nor written, yet every lane
// still takes part in all shuffles. Test: R4 batches 1/3/8/9/17 vs FP64.
__global__ void fp8_tiled_batch(const uint8_t *codes, const float *scales, const float *x, float *y,
                                int rows, int cols, int group, int batch) {
    // BEGIN TODO R4
    asm("trap;"); // Complete R4
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
