#pragma once
#include "moe_student.hpp"
#include <array>
#include <cuda_runtime.h>
#include <string>
#define CUDA_OK(call)                                                                              \
    do {                                                                                           \
        auto err = (call);                                                                         \
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));                 \
    } while (0)
namespace moe {
// C3: one thread computes one output channel of one dispatched token.
// Matrices are INT8 [rows,cols]; scales:[rows]. gather=true reads x[ids[token],:].
// Both layers compute actual MLP values; the second layer disables ReLU/gather.
// Test: c_cuda_forward checks all outputs against CPU forward, for odd D/H and empty bins.
__global__ void expert_matvec(const int8_t *q, const float *scales, const float *x, const int *ids,
                              float *out, int rows, int cols, int n, bool gather, bool relu) {
    // BEGIN TODO C3
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * rows) return;
    int token = idx / rows;
    int row = idx % rows;
    int input_token = gather ? ids[token] : token;
    float sum = 0;
    for (int col = 0; col < cols; ++col) {
        sum += float(q[row * cols + col]) * x[input_token * cols + col];
    }
    sum *= scales[row];
    out[idx] = relu ? fmaxf(0.f, sum) : sum;
    // END TODO C3
}
// C4: at most slots=1 or 2 experts reside in GPU weight buffers. H2D weights come
// from cudaMallocHost memory. Copy must wait for previous compute on the same slot;
// compute waits for THIS expert's ready event. Inputs stay resident; outputs are per-route.
// There is no CPU cudaDeviceSynchronize inside the loop. Test: c_cuda_forward.
inline std::vector<float> offload_forward(const std::vector<Expert> &experts, const Dispatch &d,
                                          const std::vector<float> &x, int tokens, int width,
                                          int slots) {
    if (slots < 1 || slots > 2 || experts.empty()) throw std::invalid_argument("slots");
    int hidden = experts[0].up.rows;
    size_t qn = 2 * size_t(hidden) * width, sn = hidden + width, routes = d.ordered.size();
    if (routes == 0) return std::vector<float>(tokens * width, 0);
    int8_t *host_q;
    float *host_s;
    CUDA_OK(cudaMallocHost(&host_q, qn * experts.size()));
    CUDA_OK(cudaMallocHost(&host_s, sn * experts.size() * sizeof(float)));
    // C6: serialize two row-major matrices and their row scales into each pinned
    // expert record. These exact offsets are consumed by C3/C4 on GPU.
    // qn counts INT8 bytes, sn counts FLOAT ELEMENTS; confusing units corrupts results.
    // Test: c_cuda_forward validates odd-width records over repeated slot reuse.
    // BEGIN TODO C6
    for (size_t e = 0; e < experts.size(); ++e) {
        std::copy(experts[e].up.q.begin(), experts[e].up.q.end(), host_q + e * qn);
        std::copy(experts[e].down.q.begin(), experts[e].down.q.end(),
                  host_q + e * qn + hidden * width);
        std::copy(experts[e].up.scale.begin(), experts[e].up.scale.end(), host_s + e * sn);
        std::copy(experts[e].down.scale.begin(), experts[e].down.scale.end(),
                  host_s + e * sn + hidden);
    }
    // END TODO C6
    float *dx, *dy;
    int *ids;
    CUDA_OK(cudaMalloc(&dx, x.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&dy, routes * width * sizeof(float)));
    CUDA_OK(cudaMalloc(&ids, routes * sizeof(int)));
    std::vector<int> host_ids;
    for (auto route : d.ordered) host_ids.push_back(route.token);
    CUDA_OK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(ids, host_ids.data(), routes * sizeof(int), cudaMemcpyHostToDevice));
    std::array<int8_t *, 2> dq{};
    std::array<float *, 2> ds{}, dh{};
    std::array<cudaStream_t, 2> copies{}, computes{};
    std::array<cudaEvent_t, 2> ready{}, done{};
    for (int s = 0; s < slots; ++s) {
        CUDA_OK(cudaMalloc(&dq[s], qn));
        CUDA_OK(cudaMalloc(&ds[s], sn * sizeof(float)));
        CUDA_OK(cudaMalloc(&dh[s], routes * hidden * sizeof(float)));
        CUDA_OK(cudaStreamCreateWithFlags(&copies[s], cudaStreamNonBlocking));
        CUDA_OK(cudaStreamCreateWithFlags(&computes[s], cudaStreamNonBlocking));
        CUDA_OK(cudaEventCreateWithFlags(&ready[s], cudaEventDisableTiming));
        CUDA_OK(cudaEventCreateWithFlags(&done[s], cudaEventDisableTiming));
    }
    // BEGIN TODO C4
    std::array<bool, 2> busy{false, false};
    int launched = 0;
    for (size_t e = 0; e < experts.size(); ++e) {
        int begin = d.offsets[e];
        int n = d.offsets[e + 1] - begin;
        if (n == 0) continue;
        int s = launched++ % slots;
        if (busy[s]) CUDA_OK(cudaStreamWaitEvent(copies[s], done[s], 0));
        CUDA_OK(cudaMemcpyAsync(dq[s], host_q + e * qn, qn, cudaMemcpyHostToDevice, copies[s]));
        CUDA_OK(cudaMemcpyAsync(ds[s], host_s + e * sn, sn * sizeof(float), cudaMemcpyHostToDevice,
                                copies[s]));
        CUDA_OK(cudaEventRecord(ready[s], copies[s]));
        CUDA_OK(cudaStreamWaitEvent(computes[s], ready[s], 0));
        expert_matvec<<<(n * hidden + 127) / 128, 128, 0, computes[s]>>>(
            dq[s], ds[s], dx, ids + begin, dh[s], hidden, width, n, true, true);
        CUDA_OK(cudaGetLastError());
        expert_matvec<<<(n * width + 127) / 128, 128, 0, computes[s]>>>(
            dq[s] + hidden * width, ds[s] + hidden, dh[s], nullptr, dy + begin * width, width,
            hidden, n, false, false);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaEventRecord(done[s], computes[s]));
        busy[s] = true;
    }
    for (int s = 0; s < slots; ++s) CUDA_OK(cudaStreamSynchronize(computes[s]));
    // END TODO C4
    std::vector<float> ys(routes * width);
    CUDA_OK(cudaMemcpy(ys.data(), dy, ys.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int s = 0; s < slots; ++s) {
        CUDA_OK(cudaFree(dq[s]));
        CUDA_OK(cudaFree(ds[s]));
        CUDA_OK(cudaFree(dh[s]));
        CUDA_OK(cudaEventDestroy(ready[s]));
        CUDA_OK(cudaEventDestroy(done[s]));
        CUDA_OK(cudaStreamDestroy(copies[s]));
        CUDA_OK(cudaStreamDestroy(computes[s]));
    }
    CUDA_OK(cudaFree(dx));
    CUDA_OK(cudaFree(dy));
    CUDA_OK(cudaFree(ids));
    CUDA_OK(cudaFreeHost(host_q));
    CUDA_OK(cudaFreeHost(host_s));
    return combine(d, ys, tokens, width);
}
} // namespace moe
