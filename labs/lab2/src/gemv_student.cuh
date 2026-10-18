#pragma once
#include <cstdint>
#include <cuda_runtime.h>
// Stage B: W4A32 SIMT kernels over the lab layout (codes [rows, ceil(cols/2)], even
// column in the low nibble, u = q + 8; FP32 scales [rows, ceil(cols/group)]).
// X is FP32 [batch, cols], Y is FP32 [batch, rows]. Kernel inputs in the tests come
// from the test's own packer, outputs are compared with an FP64 decode.

// B1: return the 4-bit field of `byte` starting at bit `shift` (0 or 4) as an
// UNSIGNED value 0..15, using inline PTX bfe.u32 (the offset 8 is removed by the
// caller; a sign-extending extract would be wrong for this offset-binary format).
// Test: B1 probe over all 256 bytes and both shifts.
__device__ inline unsigned nibble(unsigned byte, unsigned shift) {
    // BEGIN TODO B1
    asm("trap;"); // Complete B1
    return 0;
    // END TODO B1
}
// B2: one warp per output row; lane l handles columns l, l+32, ...; partial sums
// reduced with __shfl_down_sync over the full warp; lane 0 writes y[row]. A warp whose
// row >= m returns as a whole; lanes without columns must still join the shuffles.
// Test: B2 rows 1/7/33 x cols 1/31/65/257 x groups 3/32/128.
__global__ void fused_gemv(const uint8_t *packed, const float *scales, const float *x, float *y,
                           int m, int n, int group) {
    // BEGIN TODO B2
    asm("trap;"); // Complete B2
    // END TODO B2
}
// B3: lane l handles bytes l, l+32, ... of its row, i.e. columns 2b and 2b+1. The
// two columns may belong to different groups (each uses its own scale) and the odd
// tail has no second column. Test: B3 with group 3 and odd widths.
__global__ void pair_gemv(const uint8_t *p, const float *s, const float *x, float *y, int rows,
                          int cols, int group) {
    // BEGIN TODO B3
    asm("trap;"); // Complete B3
    // END TODO B3
}
// B4: grid.y selects a tile of FOUR tokens; each decoded weight is reused for all
// tokens of the tile from registers. Tokens >= batch are neither read nor written,
// but every lane still runs all four reductions. Test: B4 batches 1/3/4/7/8.
__global__ void tiled_batch(const uint8_t *p, const float *s, const float *x, float *y, int rows,
                            int cols, int group, int batch) {
    // BEGIN TODO B4
    asm("trap;"); // Complete B4
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

// B5: split K into parts = ceil(cols/chunk); blockIdx.y = part, one warp per
// (row, part) sums columns [part*chunk, min(cols, (part+1)*chunk)) and lane 0 writes
// partial[part * rows + row]. No atomics. Test: B5/B6 chunk 17 incl. cols < chunk,
// and a direct read-back of the partial buffer.
__global__ void split_partial(const uint8_t *p, const float *s, const float *x, float *partial,
                              int rows, int cols, int group, int chunk) {
    // BEGIN TODO B5
    asm("trap;"); // Complete B5
    // END TODO B5
}
// B6: y[row] = sum over ALL parts of partial[part * rows + row], in part order (so
// the result is deterministic); one thread per row, grid-stride. Test: B5/B6.
__global__ void merge_partial(const float *partial, float *y, int rows, int parts) {
    // BEGIN TODO B6
    asm("trap;"); // Complete B6
    // END TODO B6
}
inline void launch_split(const uint8_t *p, const float *s, const float *x, float *y, float *partial,
                         int rows, int cols, int group, int chunk) {
    int parts = (cols + chunk - 1) / chunk;
    split_partial<<<dim3((rows + 3) / 4, parts), 128>>>(p, s, x, partial, rows, cols, group, chunk);
    merge_partial<<<(rows + 127) / 128, 128>>>(partial, y, rows, parts);
}
