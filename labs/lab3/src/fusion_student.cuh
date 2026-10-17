#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
// C5: one block per row. Reduce squared x+residual in shared memory, divide
// by TRUE width, then apply rsqrt and gamma. All threads reach every barrier;
// padded reduction lanes contribute zero. Test: test.cu double CPU oracle.
__global__ void fused_residual_rms(const float *x, const float *residual, const float *gamma,
                                   float *y, int rows, int width) {
    // BEGIN TODO C5
    asm("trap;");
    // END TODO C5
}
inline void launch_fusion(const float *x, const float *r, const float *g, float *y, int rows,
                          int width) {
    fused_residual_rms<<<rows, 256>>>(x, r, g, y, rows, width);
}
