// Stage R fixture (provided). Expert pool pattern, the verification kernel, device
// reporting and the timing helpers. Students implement src/r_pipeline.cuh only.
//
// The kernel is given so that the race test is deterministic: it reads the slot once,
// optionally spins, then re-reads it. If a copy lands in the slot while the kernel is
// still reading, the two sums differ and out_bad is incremented on the device. A missing
// "copy waits for the previous compute on this slot" dependency is therefore detected by
// the data, not by a timing heuristic.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CUDA_OK
#define CUDA_OK(call)                                                                              \
    do {                                                                                           \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess)                                                                   \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(err_));        \
    } while (0)
#endif

namespace rfix {

// Expert e byte i. Deterministic, zero-mean-ish, and different for every expert, so a wrong
// slot index or a wrong expert offset changes the checksum.
__host__ __device__ inline int8_t pattern(int expert, size_t i) {
    unsigned long long h = (unsigned long long)expert * 1315423911ull + i * 2654435761ull;
    h ^= h >> 13;
    return (int8_t)(int)((h % 201ull) - 100ull); // [-100, 100]
}

inline void fill_pool(int8_t *host, int experts, size_t bytes) {
    for (int e = 0; e < experts; ++e)
        for (size_t i = 0; i < bytes; ++i) host[size_t(e) * bytes + i] = pattern(e, i);
}

// Independent host-side expected checksum (the oracle for one expert's payload).
inline long long expected_sum(int expert, size_t bytes) {
    long long total = 0;
    for (size_t i = 0; i < bytes; ++i) total += pattern(expert, i);
    return total;
}

// reads >= 2 passes over the slot: pass 0, an optional spin, then (reads-1) verification
// passes. A pass that disagrees with pass 0 means the buffer changed under the kernel.
// out_sum[step] accumulates the LAST pass, so a late overwrite also changes the checksum.
__global__ void expert_consume(const int8_t *__restrict__ slot, size_t bytes, int reads,
                               long long spin_cycles, long long *out_sum, int *out_bad, int step) {
    size_t stride = size_t(gridDim.x) * blockDim.x;
    size_t start = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    long long first = 0;
    for (size_t i = start; i < bytes; i += stride) first += slot[i];
    if (spin_cycles > 0) {
        long long t0 = clock64();
        while (clock64() - t0 < spin_cycles) {
        }
    }
    long long last = first;
    for (int r = 1; r < reads; ++r) {
        long long s = 0;
        for (size_t i = start; i < bytes; i += stride) s += slot[i];
        if (s != first) atomicAdd(out_bad + step, 1);
        last = s;
    }
    atomicAdd((unsigned long long *)(out_sum + step), (unsigned long long)last);
}

// The CPU expert: the same reduction the device kernel performs, so that a CPU step and a
// GPU step compute the same number for the same payload. This is what the R5 cost model
// calls cpu_ms; it is a plain scalar loop, not an AVX2/AMX kernel like kTransformers uses.
inline long long cpu_consume(const int8_t *payload, size_t bytes, int reads) {
    long long last = 0;
    for (int r = 0; r < std::max(1, reads); ++r) {
        long long sum = 0;
        for (size_t i = 0; i < bytes; ++i) sum += payload[i];
        last = sum;
    }
    return last;
}

// ---------- timing ----------

struct Stats {
    double median, p25, p75;
    bool stable; // p75 <= 1.5 * p25: clocks held still across the repetitions
};
inline Stats summarize(std::vector<double> ms) {
    std::sort(ms.begin(), ms.end());
    auto at = [&](double f) { return ms[size_t(f * (ms.size() - 1) + 0.5)]; };
    Stats s{at(0.5), at(0.25), at(0.75), false};
    s.stable = s.p75 <= 1.5 * s.p25;
    return s;
}

// Laptop GPUs idle down to ~810 MHz memory clock between configurations, which made
// bandwidth-bound work up to 8x slower in lab1. Keep the GPU busy with the SAME work for at
// least 300 ms before timing anything (labs/docs/authoring.md section 5).
constexpr double kSpinMs = 300;
template <class F> void spin_warmup(F launch) {
    cudaEvent_t a, b;
    CUDA_OK(cudaEventCreate(&a));
    CUDA_OK(cudaEventCreate(&b));
    CUDA_OK(cudaEventRecord(a));
    for (float busy = 0; busy < kSpinMs;) {
        for (int i = 0; i < 4; ++i) launch();
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        CUDA_OK(cudaEventElapsedTime(&busy, a, b));
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
}

// ---------- device report ----------

struct DeviceInfo {
    char name[256];
    int sms, async_engines, major, minor;
    size_t free_bytes, total_bytes;
};
inline DeviceInfo device_info() {
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    DeviceInfo info{};
    snprintf(info.name, sizeof info.name, "%s", prop.name);
    info.sms = prop.multiProcessorCount;
    info.async_engines = prop.asyncEngineCount;
    info.major = prop.major;
    info.minor = prop.minor;
    CUDA_OK(cudaMemGetInfo(&info.free_bytes, &info.total_bytes));
    return info;
}
inline void print_device(const DeviceInfo &info) {
    // asyncEngineCount 1 (the usual answer under WDDM/WSL) means H2D and D2H serialise with
    // each other, while a copy can still overlap a kernel. Design the pipeline for 1.
    printf("# device %s sm_%d%d, %d SMs, asyncEngineCount %d, VRAM %.2f/%.2f GiB free\n", info.name,
           info.major, info.minor, info.sms, info.async_engines,
           double(info.free_bytes) / (1 << 30), double(info.total_bytes) / (1 << 30));
}

} // namespace rfix
