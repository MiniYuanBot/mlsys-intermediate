#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
// C5: one block per row. Reduce squared x+residual in shared memory, divide
// by TRUE width, then apply rsqrt and gamma. All threads reach every barrier;
// padded reduction lanes contribute zero. Test: test.cu double CPU oracle.
__global__ void fused_residual_rms(const float *x, const float *residual, const float *gamma,
                                   float *y, int rows, int width) {
    // BEGIN TODO C5
    int r = blockIdx.x, t = threadIdx.x;
    if (r >= rows) return;
    __shared__ float partial[256];
    float sum = 0;
    for (int j = t; j < width; j += blockDim.x) {
        float v = x[r * width + j] + residual[r * width + j];
        sum += v * v;
    }
    partial[t] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (t < stride) partial[t] += partial[t + stride];
        __syncthreads();
    }
    float inv = rsqrtf(partial[0] / width + 1e-5f);
    for (int j = t; j < width; j += blockDim.x)
        y[r * width + j] = (x[r * width + j] + residual[r * width + j]) * inv * gamma[j];
    // END TODO C5
}
inline void launch_fusion(const float *x, const float *r, const float *g, float *y, int rows,
                          int width) {
    fused_residual_rms<<<rows, 256>>>(x, r, g, y, rows, width);
}
