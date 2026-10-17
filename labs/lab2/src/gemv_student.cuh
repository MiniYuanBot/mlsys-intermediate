#pragma once
#include <cstdint>
#include <cuda_runtime.h>
// B1: extract four unsigned bits; shift=0 or 4. bfe.u32 is PTX, not CPU asm.
// Test: kernels_against_double_oracle exercises all signed codes via inputs.
__device__ inline unsigned nibble(unsigned byte, unsigned shift) {
    // BEGIN TODO B1
    asm("trap;");
    return 0;
    // END TODO B1
}
// B2: one warp per output row, lane-stride K then warp sum. q+8 encoding.
// Row tails may return (whole warp); K tails must NOT return individual lanes.
// Test: warp_kernel covers rows 1/7/33, K 1/31/65/257 and groups 3/32/128.
__global__ void fused_gemv(const uint8_t *packed, const float *scales, const float *x, float *y,
                           int m, int n, int group) {
    // BEGIN TODO B2
    asm("trap;");
    // END TODO B2
}
// B3: each lane owns one packed BYTE and processes up to two neighboring K.
// The pair may cross a quantization group boundary. Separate both scales.
// Test: pair_group_boundary includes group=3 and odd K.
__global__ void pair_gemv(const uint8_t *p, const float *s, const float *x, float *y, int rows,
                          int cols, int group) {
    // BEGIN TODO B3
    asm("trap;");
    // END TODO B3
}
// B4: reuse one unpacked weight across a tile of FOUR tokens in registers.
// Grid.y selects token tile; X [batch,cols], output [batch,rows]. Every lane
// participates in all four reductions, including a partial final token tile.
// Test: tiled_batch uses batches 1/3/4/7 and non-divisible dimensions.
__global__ void tiled_batch(const uint8_t *p, const float *s, const float *x, float *y, int rows,
                            int cols, int group, int batch) {
    // BEGIN TODO B4
    asm("trap;");
    // END TODO B4
}
inline void launch_fused(const uint8_t *p, const float *s, const float *x, float *y, int m, int n,
                         int g) {
    fused_gemv<<<(m + 3) / 4, 128>>>(p, s, x, y, m, n, g);
}
inline void launch_pair(const uint8_t *p, const float *s, const float *x, float *y, int m, int n,
                        int g) {
    pair_gemv<<<(m + 3) / 4, 128>>>(p, s, x, y, m, n, g);
}
inline void launch_batch(const uint8_t *p, const float *s, const float *x, float *y, int m, int n,
                         int g, int batch) {
    tiled_batch<<<dim3((m + 3) / 4, (batch + 3) / 4), 128>>>(p, s, x, y, m, n, g, batch);
}

// B5: parallelize the reduction dimension into ceil(cols/chunk) partitions.
// One warp per (row,partition), partial layout [partition,rows]. No atomics.
// Test: split_k_tail verifies chunk=17, including cols<chunk.
__global__ void split_partial(const uint8_t *p, const float *s, const float *x, float *partial,
                              int rows, int cols, int group, int chunk) {
    // BEGIN TODO B5
    asm("trap;");
    // END TODO B5
}
// B6: each thread sums partials of one row, grid-stride over rows. Fixed
// partition order makes this deterministic, though its FP32 rounding differs
// from B2. Test: split_k_tail compares against the independent double oracle.
__global__ void merge_partial(const float *partial, float *y, int rows, int parts) {
    // BEGIN TODO B6
    asm("trap;");
    // END TODO B6
}
inline void launch_split(const uint8_t *p, const float *s, const float *x, float *y, float *partial,
                         int rows, int cols, int group, int chunk) {
    int parts = (cols + chunk - 1) / chunk;
    split_partial<<<dim3((rows + 3) / 4, parts), 128>>>(p, s, x, partial, rows, cols, group, chunk);
    merge_partial<<<(rows + 127) / 128, 128>>>(partial, y, rows, parts);
}
