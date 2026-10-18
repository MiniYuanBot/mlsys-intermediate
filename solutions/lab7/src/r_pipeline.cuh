// Stage R: a real expert-offload pipeline. MB-scale experts live in a pinned host pool and
// are streamed into a fixed number of pre-allocated device slots; nothing is allocated
// inside a timed region. The fixture (r_fixture.cuh) provides the payload pattern, the
// verification kernel and the timing helpers; this file is the part you implement.
#pragma once
#include "r_fixture.cuh"
#include <map>
#include <numeric>

namespace moer {

using rfix::Stats;

// ---------- R1: pinned pool and pre-allocated device slots ----------

struct Pool {
    int experts = 0, slots = 0;
    size_t bytes = 0;                         // bytes per expert
    int8_t *host = nullptr;                   // pinned, experts * bytes
    bool pinned = false;                      // true only if the pool came from cudaMallocHost
    std::vector<int8_t *> slot;               // device buffers, bytes each
    cudaStream_t copy_stream{};               // one copy stream: the GPU has one H2D copy engine
    std::vector<cudaStream_t> compute_stream; // one per slot, so computes can overlap
    std::vector<cudaEvent_t> copy_start, copy_end, comp_start, comp_end; // one per step
    cudaEvent_t wall_a{}, wall_b{}; // around the whole pipeline
    long long *device_sum = nullptr;
    int *device_bad = nullptr;
    int steps = 0;
};

// R1: build the pool for `experts` experts of `bytes` each, `slots` device slots and
// `steps` pipeline steps. Everything a timed run touches must be created here:
//   - the host pool with cudaMallocHost (pinned; async copies silently fall back to
//     synchronous staging on pageable memory), filled with rfix::fill_pool
//   - one device buffer per slot, memset to 0 so that a compute which runs before its copy
//     reads zeros instead of the right data
//   - one copy stream plus one compute stream per slot (cudaStreamNonBlocking)
//   - 4 timing events per step plus wall_a/wall_b (plain cudaEventCreate:
//     cudaEventDisableTiming events cannot be passed to cudaEventElapsedTime)
//   - device_sum[steps] and device_bad[steps], both zeroed
// Throw std::runtime_error mentioning "pinned" if cudaMallocHost fails: WSL2 limits pinned
// system memory without documenting a number, so a 2 GiB pool can legitimately fail.
// Test: r_pool (pointer attributes say the pool is pinned, slots start at zero).
inline Pool build_pool(int experts, size_t bytes, int slots, int steps) {
    // BEGIN TODO R1
    if (experts < 1 || slots < 1 || steps < 1 || bytes == 0)
        throw std::invalid_argument("pool shape");
    Pool p;
    p.experts = experts;
    p.bytes = bytes;
    p.slots = slots;
    p.steps = steps;
    cudaError_t host_status = cudaMallocHost(&p.host, size_t(experts) * bytes);
    if (host_status != cudaSuccess)
        throw std::runtime_error(std::string("cudaMallocHost for the pinned expert pool (") +
                                 std::to_string(double(experts) * double(bytes) / (1 << 20)) +
                                 " MiB) failed: " + cudaGetErrorString(host_status));
    p.pinned = true;
    rfix::fill_pool(p.host, experts, bytes);
    p.slot.resize(slots);
    p.compute_stream.resize(slots);
    for (int s = 0; s < slots; ++s) {
        CUDA_OK(cudaMalloc(&p.slot[s], bytes));
        CUDA_OK(cudaMemset(p.slot[s], 0, bytes));
        CUDA_OK(cudaStreamCreateWithFlags(&p.compute_stream[s], cudaStreamNonBlocking));
    }
    CUDA_OK(cudaStreamCreateWithFlags(&p.copy_stream, cudaStreamNonBlocking));
    for (auto *events : {&p.copy_start, &p.copy_end, &p.comp_start, &p.comp_end}) {
        events->resize(steps);
        for (int i = 0; i < steps; ++i) CUDA_OK(cudaEventCreate(&(*events)[i]));
    }
    CUDA_OK(cudaEventCreate(&p.wall_a));
    CUDA_OK(cudaEventCreate(&p.wall_b));
    CUDA_OK(cudaMalloc(&p.device_sum, size_t(steps) * sizeof(long long)));
    CUDA_OK(cudaMalloc(&p.device_bad, size_t(steps) * sizeof(int)));
    CUDA_OK(cudaMemset(p.device_sum, 0, size_t(steps) * sizeof(long long)));
    CUDA_OK(cudaMemset(p.device_bad, 0, size_t(steps) * sizeof(int)));
    CUDA_OK(cudaDeviceSynchronize());
    return p;
    // END TODO R1
}

inline void destroy_pool(Pool &p) {
    for (auto *b : p.slot) cudaFree(b);
    for (auto s : p.compute_stream) cudaStreamDestroy(s);
    if (p.copy_stream) cudaStreamDestroy(p.copy_stream);
    for (auto *events : {&p.copy_start, &p.copy_end, &p.comp_start, &p.comp_end})
        for (auto e : *events) cudaEventDestroy(e);
    if (p.wall_a) cudaEventDestroy(p.wall_a);
    if (p.wall_b) cudaEventDestroy(p.wall_b);
    cudaFree(p.device_sum);
    cudaFree(p.device_bad);
    if (p.host) cudaFreeHost(p.host);
    p = Pool{};
}

// ---------- R2: the copy/compute pipeline ----------

struct PipelineConfig {
    int reads = 2;             // passes over the slot; pass >= 1 re-checks pass 0 (compute knob)
    long long spin_cycles = 0; // extra device cycles inside the kernel, before the re-reads
    int blocks = 0;            // 0: 8 blocks per SM
    int threads = 256;
};
struct Segment {
    int step, expert, slot;
    double copy_ms, compute_ms;             // per-segment, from this step's own events
    double copy_begin_ms, compute_begin_ms; // offsets from the first copy, for a timeline plot
};
struct RunResult {
    std::vector<Segment> segment;
    double wall_ms = 0; // cudaEvent around the whole pipeline
    double copy_sum_ms = 0, compute_sum_ms = 0;
    std::vector<long long> sum; // per-step checksum the kernel accumulated
    int bad = 0;                // slot contents changed under a running kernel
};

// R2: run `schedule` (expert IDs) through the slots round-robin, step i using
// slot i % slots, with exactly these dependencies:
//   1. the copy of step i waits for the compute of the PREVIOUS step that used this slot
//      (cudaStreamWaitEvent on comp_end of step i - slots); without it a copy overwrites a
//      slot that a kernel is still reading
//   2. the compute of step i waits for its OWN copy (cudaStreamWaitEvent on copy_end[i]);
//      without it the kernel reads a half-written slot
//   3. after the loop, the copy stream waits for the last compute of every slot, so that
//      wall_b (recorded on the copy stream right after your code) covers the trailing
//      computes. The compute streams are cudaStreamNonBlocking, so they do NOT join the
//      default stream on their own: without this the events are read before the work is
//      done and cudaEventElapsedTime returns cudaErrorNotReady.
// Copies all go on pool.copy_stream (one copy engine); step i computes on the stream of its
// slot, so the copy of step i+1 overlaps the compute of step i.
// Record copy_start/copy_end around the memcpy and comp_start/comp_end around the kernel.
// Allocate nothing, and call no cudaDeviceSynchronize/cudaMalloc/cudaFree inside the loop:
// one synchronising call drains the pipeline and removes the overlap you are measuring.
// Returns per-segment times plus the checksums. Test: r_pipeline, r_race.
inline RunResult run_pipeline(Pool &pool, const std::vector<int> &schedule,
                              const PipelineConfig &cfg) {
    if (int(schedule.size()) > pool.steps) throw std::invalid_argument("schedule longer than pool");
    int steps = int(schedule.size());
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    int blocks = cfg.blocks > 0 ? cfg.blocks : prop.multiProcessorCount * 8;
    CUDA_OK(cudaMemset(pool.device_sum, 0, size_t(pool.steps) * sizeof(long long)));
    CUDA_OK(cudaMemset(pool.device_bad, 0, size_t(pool.steps) * sizeof(int)));
    CUDA_OK(cudaDeviceSynchronize());
    RunResult out;
    CUDA_OK(cudaEventRecord(pool.wall_a, pool.copy_stream));
    // BEGIN TODO R2
    for (int i = 0; i < steps; ++i) {
        int s = i % pool.slots;
        if (i >= pool.slots)
            CUDA_OK(cudaStreamWaitEvent(pool.copy_stream, pool.comp_end[i - pool.slots], 0));
        CUDA_OK(cudaEventRecord(pool.copy_start[i], pool.copy_stream));
        CUDA_OK(cudaMemcpyAsync(pool.slot[s], pool.host + size_t(schedule[i]) * pool.bytes,
                                pool.bytes, cudaMemcpyHostToDevice, pool.copy_stream));
        CUDA_OK(cudaEventRecord(pool.copy_end[i], pool.copy_stream));
        CUDA_OK(cudaStreamWaitEvent(pool.compute_stream[s], pool.copy_end[i], 0));
        CUDA_OK(cudaEventRecord(pool.comp_start[i], pool.compute_stream[s]));
        rfix::expert_consume<<<blocks, cfg.threads, 0, pool.compute_stream[s]>>>(
            pool.slot[s], pool.bytes, cfg.reads, cfg.spin_cycles, pool.device_sum, pool.device_bad,
            i);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaEventRecord(pool.comp_end[i], pool.compute_stream[s]));
    }
    for (int s = 0; s < pool.slots && s < steps; ++s) {
        int last = steps - 1;
        while (last % pool.slots != s) --last;
        CUDA_OK(cudaStreamWaitEvent(pool.copy_stream, pool.comp_end[last], 0));
    }
    // END TODO R2
    CUDA_OK(cudaEventRecord(pool.wall_b, pool.copy_stream));
    CUDA_OK(cudaEventSynchronize(pool.wall_b));
    CUDA_OK(cudaDeviceSynchronize());
    float wall = 0;
    CUDA_OK(cudaEventElapsedTime(&wall, pool.wall_a, pool.wall_b));
    out.wall_ms = wall;
    for (int i = 0; i < steps; ++i) {
        float copy = 0, compute = 0, copy_begin = 0, compute_begin = 0;
        CUDA_OK(cudaEventElapsedTime(&copy, pool.copy_start[i], pool.copy_end[i]));
        CUDA_OK(cudaEventElapsedTime(&compute, pool.comp_start[i], pool.comp_end[i]));
        CUDA_OK(cudaEventElapsedTime(&copy_begin, pool.copy_start[0], pool.copy_start[i]));
        CUDA_OK(cudaEventElapsedTime(&compute_begin, pool.copy_start[0], pool.comp_start[i]));
        out.segment.push_back(
            {i, schedule[i], i % pool.slots, copy, compute, copy_begin, compute_begin});
        out.copy_sum_ms += copy;
        out.compute_sum_ms += compute;
    }
    out.sum.resize(steps);
    std::vector<int> bad(steps);
    CUDA_OK(cudaMemcpy(out.sum.data(), pool.device_sum, size_t(steps) * sizeof(long long),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(bad.data(), pool.device_bad, size_t(steps) * sizeof(int),
                       cudaMemcpyDeviceToHost));
    out.bad = std::accumulate(bad.begin(), bad.end(), 0);
    return out;
}

// ---------- R3: overlap ratio and measured H2D bandwidth ----------

// R3a: how much of the shorter side was actually hidden,
//   ratio = (sum(copy) + sum(compute) - wall) / min(sum(copy), sum(compute)),
// clamped to [0, 1]. 0 means fully serial, 1 means the shorter side is completely hidden.
// Return 0 when either sum is 0. Test: r_overlap (hand numbers and the serial baseline).
inline double overlap_ratio(const RunResult &r) {
    // BEGIN TODO R3
    double shorter = std::min(r.copy_sum_ms, r.compute_sum_ms);
    if (shorter <= 0) return 0;
    double hidden = r.copy_sum_ms + r.compute_sum_ms - r.wall_ms;
    return std::max(0.0, std::min(1.0, hidden / shorter));
    // END TODO R3
}

struct Bandwidth {
    double gbps, median_ms;
    bool stable;    // p75 <= 1.5 * p25
    bool is_pinned; // what cudaPointerGetAttributes says about the host buffer actually used
};

// R3b: measure one-way H2D bandwidth for a `bytes` transfer, from `pinned` or pageable host
// memory. Allocate with cudaMallocHost when pinned and plain new/malloc otherwise, spin-warm
// the GPU with rfix::spin_warmup on the SAME copy for >= 300 ms (a laptop GPU that has been
// idle runs at a low P-state), then time >= 9 repetitions with events, take the median and
// report GB/s = bytes / (median_ms * 1e-3) / 1e9 as decimal GB. Fill is_pinned from
// cudaPointerGetAttributes(host) == cudaMemoryTypeHost, not from the flag you were passed.
// Free everything before returning. Test: r_bandwidth (independent timing of the same copy).
inline Bandwidth measure_h2d(size_t bytes, bool pinned) {
    // BEGIN TODO R4
    int8_t *host = nullptr, *device = nullptr;
    if (pinned) {
        cudaError_t status = cudaMallocHost(&host, bytes);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("cudaMallocHost (pinned) failed: ") +
                                     cudaGetErrorString(status));
    } else {
        host = new int8_t[bytes];
    }
    std::fill(host, host + bytes, int8_t(7));
    CUDA_OK(cudaMalloc(&device, bytes));
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto copy = [&] { cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream); };
    rfix::spin_warmup(copy);
    CUDA_OK(cudaStreamSynchronize(stream));
    cudaEvent_t a, b;
    CUDA_OK(cudaEventCreate(&a));
    CUDA_OK(cudaEventCreate(&b));
    std::vector<double> ms;
    for (int rep = 0; rep < 12; ++rep) {
        CUDA_OK(cudaEventRecord(a, stream));
        copy();
        CUDA_OK(cudaEventRecord(b, stream));
        CUDA_OK(cudaEventSynchronize(b));
        float t = 0;
        CUDA_OK(cudaEventElapsedTime(&t, a, b));
        if (rep >= 3) ms.push_back(t);
    }
    auto stats = rfix::summarize(ms);
    cudaPointerAttributes attributes{};
    bool is_pinned = cudaPointerGetAttributes(&attributes, host) == cudaSuccess &&
                     attributes.type == cudaMemoryTypeHost;
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaStreamDestroy(stream);
    cudaFree(device);
    if (pinned)
        cudaFreeHost(host);
    else
        delete[] host;
    cudaGetLastError();
    return {double(bytes) / (stats.median * 1e-3) / 1e9, stats.median, stats.stable, is_pinned};
    // END TODO R4
}

// ---------- R4: CPU/GPU hybrid policy ----------

struct CostModel {
    size_t bytes = 0;    // per expert
    double h2d_gbps = 0; // measured, from measure_h2d
    double gpu_ms = 0;   // measured GPU compute for one expert
    double cpu_ms = 0;   // measured CPU compute for one expert (same payload)
};
struct HybridStep {
    int expert;
    bool on_cpu; // true: compute on the host, no copy, cache unchanged
    bool was_resident;
    double ms; // modeled cost of this step
};
struct HybridPlan {
    std::vector<HybridStep> step;
    double total_ms = 0;
    size_t copied_bytes = 0;
    int cpu_steps = 0, gpu_steps = 0, resident_hits = 0;
};

// R5: Fiddler-style per-expert decision over a routing trace, with `slots` GPU slots under
// LRU. For each expert in `schedule`:
//   - already resident: GPU, cost gpu_ms, refresh its recency, no bytes moved
//   - otherwise compare fetch + compute, bytes / (h2d_gbps * 1e9) * 1e3 + gpu_ms, against
//     cpu_ms and take the cheaper one; a tie goes to the GPU (the copy also warms the cache)
//   - a CPU step moves NO weights, so it must not admit the expert or evict anything: the
//     residency set is unchanged
//   - a GPU fetch admits the expert, evicting the least recently used one when full
// Accumulate total_ms, copied_bytes and the counters. Test: r_hybrid (hand cases at both
// extremes, an independent recomputation, and the "CPU steps do not touch the cache" rule).
inline HybridPlan hybrid_plan(const std::vector<int> &schedule, int slots, const CostModel &cost) {
    // BEGIN TODO R5
    if (slots < 1) throw std::invalid_argument("slots");
    double fetch_ms = double(cost.bytes) / (cost.h2d_gbps * 1e9) * 1e3;
    HybridPlan plan;
    std::vector<int> resident; // front = least recently used
    for (int expert : schedule) {
        auto it = std::find(resident.begin(), resident.end(), expert);
        if (it != resident.end()) {
            resident.erase(it);
            resident.push_back(expert);
            plan.step.push_back({expert, false, true, cost.gpu_ms});
            plan.total_ms += cost.gpu_ms;
            ++plan.gpu_steps;
            ++plan.resident_hits;
            continue;
        }
        if (cost.cpu_ms < fetch_ms + cost.gpu_ms) {
            plan.step.push_back({expert, true, false, cost.cpu_ms});
            plan.total_ms += cost.cpu_ms;
            ++plan.cpu_steps;
            continue;
        }
        if (int(resident.size()) == slots) resident.erase(resident.begin());
        resident.push_back(expert);
        plan.step.push_back({expert, false, false, fetch_ms + cost.gpu_ms});
        plan.total_ms += fetch_ms + cost.gpu_ms;
        plan.copied_bytes += cost.bytes;
        ++plan.gpu_steps;
    }
    return plan;
    // END TODO R5
}

} // namespace moer
