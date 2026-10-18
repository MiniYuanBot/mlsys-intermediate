#pragma once
// Provided: error checks, the two-kernel baseline, host fixtures and the FP64 oracle.
// The baseline uses its own shared-memory tree reduction so it never depends on C5.
#include "fusion_student.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_OK(call)                                                                              \
    do {                                                                                           \
        cudaError_t e_ = (call);                                                                   \
        if (e_ != cudaSuccess)                                                                     \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e_));          \
    } while (0)

__device__ __forceinline__ float to_f(float v) {
    return v;
}
__device__ __forceinline__ float to_f(__half v) {
    return __half2float(v);
}
template <class T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) {
    return v;
}
template <> __device__ __forceinline__ __half from_f<__half>(float v) {
    return __float2half_rn(v);
}

// Baseline kernel 1: residual <- x + residual (elementwise, grid-stride).
template <class T> __global__ void add_inplace(const T *x, T *residual, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x)
        residual[i] = from_f<T>(to_f(x[i]) + to_f(residual[i]));
}

// Baseline kernel 2: y = residual * rsqrt(mean(residual^2) + eps) * gamma, one block per row.
template <class T> __global__ void rmsnorm_rows(const T *z, const T *gamma, T *y, int width) {
    __shared__ float s[kThreads];
    const size_t base = size_t(blockIdx.x) * width;
    float v = 0;
    for (int j = threadIdx.x; j < width; j += kThreads) {
        float a = to_f(z[base + j]);
        v += a * a;
    }
    s[threadIdx.x] = v;
    __syncthreads();
    for (int k = kThreads / 2; k; k /= 2) {
        if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
        __syncthreads();
    }
    const float inv = rsqrtf(s[0] / width + kEps);
    for (int j = threadIdx.x; j < width; j += kThreads)
        y[base + j] = from_f<T>(to_f(z[base + j]) * inv * to_f(gamma[j]));
}

template <class T>
void launch_two_kernel(const T *x, T *residual, const T *gamma, T *y, int rows, int width,
                       cudaStream_t stream = 0) {
    size_t n = size_t(rows) * width;
    int blocks = int(std::min<size_t>((n + 255) / 256, 4096));
    add_inplace<T><<<blocks, 256, 0, stream>>>(x, residual, n);
    rmsnorm_rows<T><<<rows, kThreads, 0, stream>>>(residual, gamma, y, width);
}

inline void launch_fused(const float *x, float *r, const float *g, float *y, int rows, int width,
                         cudaStream_t s = 0) {
    launch_fused_f32(x, r, g, y, rows, width, s);
}
inline void launch_fused(const __half *x, __half *r, const __half *g, __half *y, int rows,
                         int width, cudaStream_t s = 0) {
    launch_fused_f16(x, r, g, y, rows, width, s);
}

inline float host_f(float v) {
    return v;
}
inline float host_f(__half v) {
    return __half2float(v);
}
template <class T> T host_from(float v);
template <> inline float host_from<float>(float v) {
    return v;
}
template <> inline __half host_from<__half>(float v) {
    return __float2half_rn(v);
}

// Device buffers plus host copies of the inputs. `scale` widens the value range
// (scale 100 makes per-thread FP16 sums of squares exceed 65504 at width 4096).
template <class T> struct Fixture {
    int rows, width;
    std::vector<T> hx, hr, hg;
    T *x = nullptr, *r = nullptr, *g = nullptr, *y = nullptr;
    Fixture(int rows_, int width_, unsigned seed, float scale = 1.0f, bool zero_x = false)
        : rows(rows_), width(width_), hx(size_t(rows_) * width_), hr(hx.size()), hg(width_) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> d(-scale, scale), gd(0.25f, 2.0f);
        for (auto &v : hx) v = host_from<T>(zero_x ? 0.0f : d(rng));
        for (auto &v : hr) v = host_from<T>(d(rng));
        for (auto &v : hg) v = host_from<T>(gd(rng));
        size_t bytes = hx.size() * sizeof(T);
        CUDA_OK(cudaMalloc(&x, bytes));
        CUDA_OK(cudaMalloc(&r, bytes));
        CUDA_OK(cudaMalloc(&g, hg.size() * sizeof(T)));
        CUDA_OK(cudaMalloc(&y, bytes));
        reset();
        CUDA_OK(cudaMemcpy(x, hx.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(g, hg.data(), hg.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    Fixture(const Fixture &) = delete;
    Fixture &operator=(const Fixture &) = delete;
    void reset() { // residual is updated in place, so restore it before every check
        CUDA_OK(cudaMemcpy(r, hr.data(), hr.size() * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(y, 0xff, hr.size() * sizeof(T))); // NaN pattern: unwritten y fails
    }
    ~Fixture() {
        cudaFree(x);
        cudaFree(r);
        cudaFree(g);
        cudaFree(y);
    }
};

// FP64 oracle. Expected residual = T(float(x) + float(r)) exactly; y uses the FP64 sum of
// squares of the unrounded z and the stored (rounded) residual, as the kernels do.
template <class T> struct Expected {
    std::vector<T> residual;
    std::vector<double> y;
};
template <class T> Expected<T> oracle(const Fixture<T> &f) {
    Expected<T> e{std::vector<T>(f.hx.size()), std::vector<double>(f.hx.size())};
    for (int i = 0; i < f.rows; ++i) {
        double sum = 0;
        size_t base = size_t(i) * f.width;
        for (int j = 0; j < f.width; ++j) {
            double z = double(host_f(f.hx[base + j])) + double(host_f(f.hr[base + j]));
            sum += z * z;
            e.residual[base + j] = host_from<T>(host_f(f.hx[base + j]) + host_f(f.hr[base + j]));
        }
        double inv = 1.0 / std::sqrt(sum / f.width + double(kEps));
        for (int j = 0; j < f.width; ++j)
            e.y[base + j] = double(host_f(e.residual[base + j])) * inv * double(host_f(f.hg[j]));
    }
    return e;
}

// Compares device results with the oracle; returns "" or a description of the first miss.
template <class T> std::string compare(const Fixture<T> &f, const Expected<T> &e, double tol) {
    std::vector<T> hy(f.hx.size()), hres(f.hx.size());
    CUDA_OK(cudaMemcpy(hy.data(), f.y, hy.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(hres.data(), f.r, hres.size() * sizeof(T), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < hy.size(); ++i) {
        float got = host_f(hres[i]), want = host_f(e.residual[i]);
        if (!(got == want)) {
            char msg[160];
            std::snprintf(msg, sizeof msg, "residual[%zu] expected %.7g vs actual %.7g", i, want,
                          got);
            return msg;
        }
    }
    for (size_t i = 0; i < hy.size(); ++i) {
        double got = host_f(hy[i]), want = e.y[i];
        if (!std::isfinite(got) || std::abs(got - want) > tol + tol * std::abs(want)) {
            char msg[160];
            std::snprintf(msg, sizeof msg, "y[%zu] (row %zu col %zu) expected %.7g vs actual %.7g",
                          i, i / f.width, i % f.width, want, got);
            return msg;
        }
    }
    return "";
}
