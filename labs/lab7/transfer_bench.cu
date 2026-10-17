// Provided hardware calibration: two pinned-memory streams, no full MoE model.
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#define CUDA(x)                                                                                    \
    do {                                                                                           \
        auto e = (x);                                                                              \
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));                     \
    } while (0)
__global__ void consume(float *x, float *out, int n, int step) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n) x[j] = 2 * x[j] + 1;
    if (j == n - 1) out[step] = x[j];
}
int main() {
    const int n = 1 << 20, steps = 32;
    float *host;
    CUDA(cudaMallocHost(&host, n * 4));
    for (int i = 0; i < n; ++i) host[i] = float(i % 17);
    float *d[2], *out;
    cudaStream_t streams[2];
    for (int i = 0; i < 2; ++i) {
        CUDA(cudaMalloc(&d[i], n * 4));
        CUDA(cudaStreamCreate(&streams[i]));
    }
    CUDA(cudaMalloc(&out, steps * 4));
    std::cout << "kind,streams,repeat,bytes,wall_ms\n";
    for (int s : {1, 2})
        for (int rep = 0; rep < 5; ++rep) {
            CUDA(cudaDeviceSynchronize());
            auto start = std::chrono::steady_clock::now();
            for (int step = 0; step < steps; ++step) {
                int b = step % s;
                CUDA(cudaMemcpyAsync(d[b], host, n * 4, cudaMemcpyHostToDevice, streams[b]));
                consume<<<(n + 255) / 256, 256, 0, streams[b]>>>(d[b], out, n, step);
                CUDA(cudaGetLastError());
            }
            CUDA(cudaDeviceSynchronize());
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            std::vector<float> check(steps);
            CUDA(cudaMemcpy(check.data(), out, steps * 4, cudaMemcpyDeviceToHost));
            float expected = 2 * host[n - 1] + 1;
            for (float x : check)
                if (std::abs(x - expected) > 1e-5)
                    throw std::runtime_error("buffer reuse ordering failure");
            std::cout << "transfer_calibration," << s << ',' << rep << ',' << size_t(n) * 4 * steps
                      << ',' << ms << '\n';
        }
    for (int i = 0; i < 2; ++i) {
        CUDA(cudaFree(d[i]));
        CUDA(cudaStreamDestroy(streams[i]));
    }
    CUDA(cudaFree(out));
    CUDA(cudaFreeHost(host));
}
