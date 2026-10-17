#pragma once
#include "gemv_student.cuh"
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#define CUDA(x)                                                                                    \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));                     \
    } while (0)
#define BLAS(x)                                                                                    \
    do {                                                                                           \
        if ((x) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failure");              \
    } while (0)
__global__ void unpack(const uint8_t *p, const float *s, float *w, int m, int n, int g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * n) return;
    int r = i / n, c = i % n;
    int q = ((p[r * ((n + 1) / 2) + c / 2] >> (4 * (c & 1))) & 15) - 8;
    w[i] = q * s[r * ((n + g - 1) / g) + c / g];
}
struct Fixture {
    int m, n, g;
    std::vector<float> truth, original_truth;
    uint8_t *p = nullptr;
    float *s = nullptr, *x = nullptr, *y = nullptr, *w = nullptr;
    cublasHandle_t handle;
    Fixture(int rows, int cols, int group)
        : m(rows), n(cols), g(group), truth(m, 0), original_truth(m, 0) {
        std::mt19937 rng(17);
        std::uniform_real_distribution<float> d(-1, 1);
        std::vector<float> hw(m * n), hx(n), hs(m * ((n + g - 1) / g));
        std::vector<uint8_t> hp(m * ((n + 1) / 2), 0);
        for (float &v : hw) v = d(rng);
        for (float &v : hx) v = d(rng);
        int ng = (n + g - 1) / g;
        for (int r = 0; r < m; ++r)
            for (int j = 0; j < ng; ++j) {
                float mx = 0;
                for (int c = j * g; c < std::min(n, (j + 1) * g); ++c)
                    mx = std::max(mx, std::abs(hw[r * n + c]));
                hs[r * ng + j] = mx == 0 ? 1 : mx / 7;
            }
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < n; ++c) {
                int q = std::clamp(int(std::lround(hw[r * n + c] / hs[r * ng + c / g])), -7, 7);
                hp[r * ((n + 1) / 2) + c / 2] |= uint8_t((q + 8) << (4 * (c & 1)));
                truth[r] += q * hs[r * ng + c / g] * hx[c];
                original_truth[r] += hw[r * n + c] * hx[c];
            }
        CUDA(cudaMalloc(&p, hp.size()));
        CUDA(cudaMalloc(&s, hs.size() * 4));
        CUDA(cudaMalloc(&x, n * 4));
        CUDA(cudaMalloc(&y, m * 4));
        CUDA(cudaMalloc(&w, m * n * 4));
        CUDA(cudaMemcpy(p, hp.data(), hp.size(), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(s, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(x, hx.data(), n * 4, cudaMemcpyHostToDevice));
        BLAS(cublasCreate(&handle));
        unpack<<<(m * n + 255) / 256, 256>>>(p, s, w, m, n, g);
        CUDA(cudaDeviceSynchronize());
    }
    ~Fixture() {
        cublasDestroy(handle);
        cudaFree(p);
        cudaFree(s);
        cudaFree(x);
        cudaFree(y);
        cudaFree(w);
    }
    void fused() {
        launch_fused(p, s, x, y, m, n, g);
        CUDA(cudaGetLastError());
    }
    void blas() {
        float a = 1, b = 0;
        BLAS(cublasSgemv(handle, CUBLAS_OP_T, n, m, &a, w, n, x, 1, &b, y, 1));
    }
    void decomposed() {
        unpack<<<(m * n + 255) / 256, 256>>>(p, s, w, m, n, g);
        blas();
    }
    double error() {
        std::vector<float> hy(m);
        CUDA(cudaMemcpy(hy.data(), y, m * 4, cudaMemcpyDeviceToHost));
        double mx = 0;
        for (int r = 0; r < m; ++r) {
            if (!std::isfinite(hy[r])) throw std::runtime_error("nonfinite output");
            mx = std::max(mx, double(std::abs(hy[r] - truth[r])));
            if (std::abs(hy[r] - truth[r]) > 2e-3 + 2e-4 * std::abs(truth[r]))
                throw std::runtime_error("numerical mismatch");
        }
        return mx;
    }
    double quant_error() {
        double err = 0, norm = 0;
        for (int r = 0; r < m; ++r) {
            err += std::pow(truth[r] - original_truth[r], 2);
            norm += std::pow(original_truth[r], 2);
        }
        return std::sqrt(err / std::max(norm, 1e-12));
    }
    template <class F> float time(F fn) {
        for (int i = 0; i < 20; ++i) fn();
        CUDA(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        CUDA(cudaEventCreate(&a));
        CUDA(cudaEventCreate(&b));
        CUDA(cudaEventRecord(a));
        for (int i = 0; i < 100; ++i) fn();
        CUDA(cudaEventRecord(b));
        CUDA(cudaEventSynchronize(b));
        float ms;
        CUDA(cudaEventElapsedTime(&ms, a, b));
        CUDA(cudaEventDestroy(a));
        CUDA(cudaEventDestroy(b));
        return ms * 10;
    }
};
