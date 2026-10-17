#pragma once
#include "fusion_student.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#define CUDA(x)                                                                                    \
    do {                                                                                           \
        auto e = (x);                                                                              \
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));                     \
    } while (0)
__global__ void add(const float *x, const float *r, float *z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] + r[i];
}
// Same reduction algorithm as fused, but materializes the residual sum.
__global__ void norm(const float *z, const float *g, float *y, int width) {
    int row = blockIdx.x, t = threadIdx.x;
    __shared__ float s[256];
    float v = 0;
    for (int j = t; j < width; j += 256) v += z[row * width + j] * z[row * width + j];
    s[t] = v;
    __syncthreads();
    for (int k = 128; k; k /= 2) {
        if (t < k) s[t] += s[t + k];
        __syncthreads();
    }
    float inv = rsqrtf(s[0] / width + 1e-5f);
    for (int j = t; j < width; j += 256) y[row * width + j] = z[row * width + j] * inv * g[j];
}
struct Fixture {
    int rows, width;
    float *x, *r, *g, *y, *z;
    std::vector<float> truth;
    Fixture(int a, int b) : rows(a), width(b), truth(a * b) {
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> d(-1, 1);
        std::vector<float> hx(a * b), hr(a * b), hg(b);
        for (auto &v : hx) v = d(rng);
        for (auto &v : hr) v = d(rng);
        for (auto &v : hg) v = d(rng);
        for (int i = 0; i < a; ++i) {
            double sum = 0;
            for (int j = 0; j < b; ++j) sum += std::pow(double(hx[i * b + j] + hr[i * b + j]), 2);
            for (int j = 0; j < b; ++j)
                truth[i * b + j] =
                    (hx[i * b + j] + hr[i * b + j]) / std::sqrt(sum / b + 1e-5) * hg[j];
        }
        CUDA(cudaMalloc(&x, a * b * 4));
        CUDA(cudaMalloc(&r, a * b * 4));
        CUDA(cudaMalloc(&g, b * 4));
        CUDA(cudaMalloc(&y, a * b * 4));
        CUDA(cudaMalloc(&z, a * b * 4));
        CUDA(cudaMemcpy(x, hx.data(), a * b * 4, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(r, hr.data(), a * b * 4, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(g, hg.data(), b * 4, cudaMemcpyHostToDevice));
    }
    ~Fixture() {
        cudaFree(x);
        cudaFree(r);
        cudaFree(g);
        cudaFree(y);
        cudaFree(z);
    }
    void fused() {
        launch_fusion(x, r, g, y, rows, width);
        CUDA(cudaGetLastError());
    }
    void baseline() {
        add<<<(rows * width + 255) / 256, 256>>>(x, r, z, rows * width);
        norm<<<rows, 256>>>(z, g, y, width);
        CUDA(cudaGetLastError());
    }
    void check() {
        std::vector<float> hy(rows * width);
        CUDA(cudaMemcpy(hy.data(), y, rows * width * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < hy.size(); ++i)
            if (!std::isfinite(hy[i]) ||
                std::abs(hy[i] - truth[i]) > 2e-5 + 2e-5 * std::abs(truth[i]))
                throw std::runtime_error("fusion mismatch");
    }
    template <class F> float time(F fn) {
        for (int i = 0; i < 20; ++i) fn();
        CUDA(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        CUDA(cudaEventCreate(&a));
        CUDA(cudaEventCreate(&b));
        CUDA(cudaEventRecord(a));
        for (int i = 0; i < 200; ++i) fn();
        CUDA(cudaEventRecord(b));
        CUDA(cudaEventSynchronize(b));
        float ms;
        CUDA(cudaEventElapsedTime(&ms, a, b));
        CUDA(cudaEventDestroy(a));
        CUDA(cudaEventDestroy(b));
        return ms * 5;
    }
};
