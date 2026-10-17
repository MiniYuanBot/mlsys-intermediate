#pragma once
#include <cstdint>
#include <cuda_runtime.h>
// B1: extract four unsigned bits; shift=0 or 4. bfe.u32 is PTX, not CPU asm.
// Test: kernels_against_double_oracle exercises all signed codes via inputs.
__device__ inline unsigned nibble(unsigned byte, unsigned shift) {
    // BEGIN TODO B1
    unsigned out;
    asm("bfe.u32 %0, %1, %2, 4;" : "=r"(out) : "r"(byte), "r"(shift));
    return out;
    // END TODO B1
}
// B2: one warp per output row, lane-stride K then warp sum. q+8 encoding.
// Row tails may return (whole warp); K tails must NOT return individual lanes.
// Test: warp_kernel covers rows 1/7/33, K 1/31/65/257 and groups 3/32/128.
__global__ void fused_gemv(const uint8_t *packed, const float *scales, const float *x, float *y,
                           int m, int n, int group) {
    // BEGIN TODO B2
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= m) return;
    int stride = (n + 1) / 2;
    int ng = (n + group - 1) / group;
    float sum = 0;
    for (int col = lane; col < n; col += 32) {
        unsigned byte = packed[row * stride + col / 2];
        int q = int(nibble(byte, 4 * (col & 1))) - 8;
        float weight = float(q) * scales[row * ng + col / group];
        sum = fmaf(weight, x[col], sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) y[row] = sum;
    // END TODO B2
}
// B3: each lane owns one packed BYTE and processes up to two neighboring K.
// The pair may cross a quantization group boundary. Separate both scales.
// Test: pair_group_boundary includes group=3 and odd K.
__global__ void pair_gemv(const uint8_t *p, const float *s, const float *x, float *y, int rows,
                          int cols, int group) {
    // BEGIN TODO B3
    const int lane = threadIdx.x % 32;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= rows) return;
    const int bytes = (cols + 1) / 2;
    const int ng = (cols + group - 1) / group;
    float sum = 0;
    for (int pair = lane; pair < bytes; pair += 32) {
        const unsigned code = p[row * bytes + pair];
        const int col = 2 * pair;
        const float lo = (int(nibble(code, 0)) - 8) * s[row * ng + col / group];
        sum = fmaf(lo, x[col], sum);
        if (col + 1 < cols) {
            const float hi = (int(nibble(code, 4)) - 8) * s[row * ng + (col + 1) / group];
            sum = fmaf(hi, x[col + 1], sum);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) y[row] = sum;
    // END TODO B3
}
// B4: reuse one unpacked weight across a tile of FOUR tokens in registers.
// Grid.y selects token tile; X [batch,cols], output [batch,rows]. Every lane
// participates in all four reductions, including a partial final token tile.
// Test: tiled_batch uses batches 1/3/4/7 and non-divisible dimensions.
__global__ void tiled_batch(const uint8_t *p, const float *s, const float *x, float *y, int rows,
                            int cols, int group, int batch) {
    // BEGIN TODO B4
    const int lane = threadIdx.x % 32;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= rows) return;
    const int token = blockIdx.y * 4;
    const int stride = (cols + 1) / 2;
    const int ng = (cols + group - 1) / group;
    float sums[4] = {0, 0, 0, 0};
    for (int col = lane; col < cols; col += 32) {
        const unsigned byte = p[row * stride + col / 2];
        const int quant = int(nibble(byte, 4 * (col & 1))) - 8;
        const float weight = quant * s[row * ng + col / group];
        for (int t = 0; t < 4; ++t) {
            if (token + t < batch) {
                float value = x[(token + t) * cols + col];
                sums[t] = fmaf(weight, value, sums[t]);
            }
        }
    }
    for (int t = 0; t < 4; ++t) {
        for (int offset = 16; offset > 0; offset >>= 1)
            sums[t] += __shfl_down_sync(0xffffffff, sums[t], offset);
        if (lane == 0 && token + t < batch) y[(token + t) * rows + row] = sums[t];
    }
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
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) return;
    const int part = blockIdx.y;
    const int begin = part * chunk;
    const int end = min(cols, begin + chunk);
    const int stride = (cols + 1) / 2;
    const int ng = (cols + group - 1) / group;
    float value = 0;
    for (int col = begin + lane; col < end; col += 32) {
        const int quant = int(nibble(p[row * stride + col / 2], 4 * (col % 2))) - 8;
        const float weight = quant * s[row * ng + col / group];
        value = fmaf(weight, x[col], value);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) partial[part * rows + row] = value;
    // END TODO B5
}
// B6: each thread sums partials of one row, grid-stride over rows. Fixed
// partition order makes this deterministic, though its FP32 rounding differs
// from B2. Test: split_k_tail compares against the independent double oracle.
__global__ void merge_partial(const float *partial, float *y, int rows, int parts) {
    // BEGIN TODO B6
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int row = thread; row < rows; row += stride) {
        float value = 0;
        for (int part = 0; part < parts; ++part) {
            const int index = part * rows + row;
            value += partial[index];
        }
        y[row] = value;
    }
    // END TODO B6
}
inline void launch_split(const uint8_t *p, const float *s, const float *x, float *y, float *partial,
                         int rows, int cols, int group, int chunk) {
    int parts = (cols + chunk - 1) / chunk;
    split_partial<<<dim3((rows + 3) / 4, parts), 128>>>(p, s, x, partial, rows, cols, group, chunk);
    merge_partial<<<(rows + 127) / 128, 128>>>(partial, y, rows, parts);
}
