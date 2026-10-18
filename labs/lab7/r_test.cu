// Stage R correctness checks. Each check prints one PASS/FAIL line and the run continues;
// a group is selected by argv[1] so that a trapped kernel only fails its own group
// (exercise.py runs the groups in separate processes).
// Oracles: the expert payload checksum is recomputed on the host from rfix::pattern, the
// H2D bandwidth is re-timed here with its own events, the overlap ratio is checked against
// hand numbers, and the hybrid plan is recomputed by an independent LRU walk.
#include "src/r_pipeline.cuh"
#include <chrono>
#include <functional>
#include <iostream>
#include <sstream>

namespace {
int g_pass = 0, g_fail = 0;
void check(bool ok, const std::string &name, const std::string &detail = "") {
    if (ok) {
        ++g_pass;
        std::cout << "PASS " << name << '\n';
    } else {
        ++g_fail;
        std::cout << "FAIL " << name << ": " << detail << '\n';
    }
}
std::string fmt(double expected, double actual) {
    std::ostringstream s;
    s.precision(8);
    s << "expected " << expected << " vs actual " << actual;
    return s.str();
}
void section(const std::string &name, const std::function<void()> &body) {
    try {
        body();
    } catch (const std::exception &e) {
        check(false, name, std::string("threw: ") + e.what());
    }
}
bool host_pointer_is_pinned(const void *p) {
    cudaPointerAttributes attributes{};
    cudaError_t status = cudaPointerGetAttributes(&attributes, p);
    cudaGetLastError();
    return status == cudaSuccess && attributes.type == cudaMemoryTypeHost;
}

// R1: the pool must be pinned and pre-allocated, and the slots must start at zero so that a
// compute running ahead of its copy is visible in the checksum.
void r_pool_tests() {
    section("r_pool", [] {
        const size_t bytes = 4u << 20;
        auto pool = moer::build_pool(4, bytes, 2, 8);
        check(pool.host != nullptr && pool.pinned, "r_pool reports a pinned host pool");
        check(host_pointer_is_pinned(pool.host),
              "r_pool cudaPointerGetAttributes confirms page-locked host memory "
              "(cudaMemcpyAsync silently falls back to a synchronous staged copy otherwise)");
        bool pattern_ok = true;
        for (int e = 0; e < 4; ++e)
            for (size_t i = 0; i < bytes; i += 7919)
                pattern_ok &= pool.host[size_t(e) * bytes + i] == rfix::pattern(e, i);
        check(pattern_ok, "r_pool each expert holds its own payload");
        check(int(pool.slot.size()) == 2 && pool.slot[0] != pool.slot[1],
              "r_pool allocates one distinct device buffer per slot");
        std::vector<int8_t> probe(bytes);
        bool zeroed = true;
        for (int s = 0; s < 2; ++s) {
            CUDA_OK(cudaMemcpy(probe.data(), pool.slot[s], bytes, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < bytes; i += 4093) zeroed &= probe[i] == 0;
        }
        check(zeroed, "r_pool device slots start zeroed");
        check(int(pool.compute_stream.size()) == 2 && pool.copy_stream != nullptr &&
                  pool.compute_stream[0] != pool.compute_stream[1],
              "r_pool uses a copy stream plus one compute stream per slot");
        check(int(pool.copy_start.size()) == 8 && int(pool.comp_end.size()) == 8,
              "r_pool pre-creates the per-step timing events");
        moer::destroy_pool(pool);
        check(pool.host == nullptr, "r_pool destroy releases the pool");
    });
}

// R2: the dependency graph. The checksum oracle is computed on the host; `bad` counts slot
// contents that changed while a kernel was reading them.
void r_pipeline_tests() {
    section("r_pipeline", [] {
        const size_t bytes = 8u << 20;
        std::vector<int> schedule = {0, 1, 2, 3, 0, 2, 1, 3};
        for (int slots : {1, 2, 3}) {
            auto pool = moer::build_pool(4, bytes, slots, int(schedule.size()));
            moer::PipelineConfig cfg;
            cfg.reads = 3;
            auto r = moer::run_pipeline(pool, schedule, cfg);
            bool sums_ok = r.sum.size() == schedule.size();
            long long first_bad_expected = 0, first_bad_got = 0;
            for (size_t i = 0; sums_ok && i < schedule.size(); ++i) {
                long long want = rfix::expected_sum(schedule[i], bytes);
                if (r.sum[i] != want) {
                    sums_ok = false;
                    first_bad_expected = want;
                    first_bad_got = r.sum[i];
                }
            }
            check(sums_ok,
                  "r_pipeline slots=" + std::to_string(slots) +
                      " every step computes on its own expert's bytes",
                  sums_ok ? "" : fmt(double(first_bad_expected), double(first_bad_got)));
            check(r.bad == 0,
                  "r_pipeline slots=" + std::to_string(slots) +
                      " no slot changed under a running kernel",
                  fmt(0, r.bad));
            bool timed = true;
            for (const auto &s : r.segment) timed &= s.copy_ms > 0 && s.compute_ms > 0;
            check(timed && r.wall_ms > 0,
                  "r_pipeline slots=" + std::to_string(slots) + " per-segment events are timed");
            // Sanity bound, not an overlap claim: with one slot the wall time also contains
            // launch/sync gaps between segments, and laptop clocks drift between events
            // (observed up to ~16% over the segment sum), so allow 25% + 1 ms.
            check(r.wall_ms <= 1.25 * (r.copy_sum_ms + r.compute_sum_ms) + 1.0,
                  "r_pipeline slots=" + std::to_string(slots) +
                      " wall time never exceeds the serial sum",
                  fmt(r.copy_sum_ms + r.compute_sum_ms, r.wall_ms));
            bool slots_ok = true;
            for (const auto &s : r.segment) slots_ok &= s.slot == s.step % slots;
            check(slots_ok,
                  "r_pipeline slots=" + std::to_string(slots) + " steps use the slots round-robin");
            // Read the slots back: slot s must hold the expert of the LAST step with
            // step % slots == s. A rotation that is offset from the documented one keeps
            // the pipeline self-consistent, so only the device contents show it.
            std::vector<int8_t> probe(bytes);
            bool residue_ok = true;
            for (int s = 0; s < slots; ++s) {
                int last = int(schedule.size()) - 1;
                while (last >= 0 && last % slots != s) --last;
                if (last < 0) continue;
                CUDA_OK(cudaMemcpy(probe.data(), pool.slot[s], bytes, cudaMemcpyDeviceToHost));
                long long sum = 0;
                for (size_t i = 0; i < bytes; ++i) sum += probe[i];
                residue_ok &= sum == rfix::expected_sum(schedule[last], bytes);
            }
            check(residue_ok, "r_pipeline slots=" + std::to_string(slots) +
                                  " each slot ends holding the expert of its last step");
            moer::destroy_pool(pool);
        }
    });
}

// R2 race: a deliberately long compute makes the missing "copy waits for this slot's
// previous compute" dependency visible in the data. With the dependency the kernel reads a
// stable slot; without it the copy of step i+slots lands mid-kernel and either `bad` or the
// checksum reports it.
void r_race_tests() {
    section("r_race", [] {
        const size_t bytes = 16u << 20;
        std::vector<int> schedule = {0, 1, 2, 3, 4, 5};
        auto pool = moer::build_pool(6, bytes, 2, int(schedule.size()));
        moer::PipelineConfig cfg;
        cfg.reads = 4;
        cfg.spin_cycles = 60'000'000; // tens of ms: far longer than a 16 MiB copy
        auto r = moer::run_pipeline(pool, schedule, cfg);
        bool sums_ok = true;
        for (size_t i = 0; i < schedule.size(); ++i)
            sums_ok &= r.sum[i] == rfix::expected_sum(schedule[i], bytes);
        check(sums_ok && r.bad == 0,
              "r_race a slow compute still reads a stable slot (slot reuse is ordered)",
              "bad=" + std::to_string(r.bad));
        check(r.wall_ms > 1.0, "r_race the lengthened compute actually ran", fmt(1, r.wall_ms));
        moer::destroy_pool(pool);
    });
}

// R3: overlap ratio on hand numbers, then on a measured balanced pipeline.
void r_overlap_tests() {
    section("r_overlap", [] {
        moer::RunResult serial;
        serial.copy_sum_ms = 10;
        serial.compute_sum_ms = 10;
        serial.wall_ms = 20;
        check(std::abs(moer::overlap_ratio(serial)) < 1e-9,
              "r_overlap a fully serial run has ratio 0", fmt(0, moer::overlap_ratio(serial)));
        moer::RunResult perfect = serial;
        perfect.wall_ms = 10;
        check(std::abs(moer::overlap_ratio(perfect) - 1) < 1e-9,
              "r_overlap a fully hidden shorter side has ratio 1",
              fmt(1, moer::overlap_ratio(perfect)));
        moer::RunResult half;
        half.copy_sum_ms = 8;
        half.compute_sum_ms = 4;
        half.wall_ms = 10; // 8 + 4 - 10 = 2 hidden out of min(8,4) = 4
        check(std::abs(moer::overlap_ratio(half) - 0.5) < 1e-9,
              "r_overlap divides the hidden time by the shorter side",
              fmt(0.5, moer::overlap_ratio(half)));
        moer::RunResult empty;
        check(moer::overlap_ratio(empty) == 0, "r_overlap an empty run reports 0");
        moer::RunResult noisy = half;
        noisy.wall_ms = 1; // clock noise must not produce a ratio above 1
        check(moer::overlap_ratio(noisy) <= 1.0, "r_overlap clamps to at most 1");
    });
    // A balanced, measured pipeline: reads are tuned so that compute is close to the copy
    // time, which is where overlap can pay off. This is a performance check, so it uses a
    // tolerant threshold and the spin warmup from the fixture.
    section("r_overlap_measured", [] {
        const size_t bytes = 16u << 20;
        std::vector<int> schedule;
        for (int i = 0; i < 12; ++i) schedule.push_back(i % 6);
        auto pool = moer::build_pool(6, bytes, 2, int(schedule.size()));
        moer::PipelineConfig cfg;
        cfg.reads = 16; // about one copy time of device-side reading
        rfix::spin_warmup([&] { moer::run_pipeline(pool, schedule, cfg); });
        std::vector<double> ratios;
        for (int rep = 0; rep < 5; ++rep)
            ratios.push_back(moer::overlap_ratio(moer::run_pipeline(pool, schedule, cfg)));
        std::sort(ratios.begin(), ratios.end());
        double median = ratios[2];
        check(median > 0.2,
              "r_overlap_measured two slots hide part of the copy (one stream or a "
              "synchronising call inside the loop drops this to ~0)",
              fmt(0.2, median));
        moer::destroy_pool(pool);
    });
}

// R4: the measured bandwidth is re-timed here with independent events on the default stream.
void r_bandwidth_tests() {
    section("r_bandwidth", [] {
        const size_t bytes = 32u << 20;
        auto pinned = moer::measure_h2d(bytes, true);
        check(pinned.is_pinned,
              "r_bandwidth the pinned measurement really used page-locked memory");
        check(pinned.gbps > 0.5 && pinned.gbps < 64,
              "r_bandwidth pinned H2D is a plausible PCIe number", fmt(0, pinned.gbps));
        // Independent oracle: time the same copy here, from our own pinned buffer.
        int8_t *host = nullptr, *device = nullptr;
        CUDA_OK(cudaMallocHost(&host, bytes));
        CUDA_OK(cudaMalloc(&device, bytes));
        std::fill(host, host + bytes, int8_t(3));
        cudaEvent_t a, b;
        CUDA_OK(cudaEventCreate(&a));
        CUDA_OK(cudaEventCreate(&b));
        auto copy = [&] { cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice); };
        rfix::spin_warmup(copy);
        std::vector<double> ms;
        for (int rep = 0; rep < 9; ++rep) {
            CUDA_OK(cudaEventRecord(a));
            copy();
            CUDA_OK(cudaEventRecord(b));
            CUDA_OK(cudaEventSynchronize(b));
            float t = 0;
            CUDA_OK(cudaEventElapsedTime(&t, a, b));
            ms.push_back(t);
        }
        auto own = rfix::summarize(ms);
        double own_gbps = double(bytes) / (own.median * 1e-3) / 1e9;
        check(std::abs(pinned.gbps - own_gbps) < 0.35 * own_gbps,
              "r_bandwidth agrees with an independently timed copy within 35%",
              fmt(own_gbps, pinned.gbps));
        check(std::abs(double(bytes) / (pinned.median_ms * 1e-3) / 1e9 - pinned.gbps) <
                  1e-6 * own_gbps + 1e-6,
              "r_bandwidth GB/s is bytes / median seconds in decimal GB");
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        cudaFree(device);
        cudaFreeHost(host);
        auto pageable = moer::measure_h2d(bytes, false);
        check(!pageable.is_pinned,
              "r_bandwidth the pageable measurement used ordinary host memory");
        check(pageable.gbps > 0.1 && pageable.gbps <= pinned.gbps * 1.35,
              "r_bandwidth pageable H2D is not faster than pinned beyond noise",
              fmt(pinned.gbps, pageable.gbps));
    });
}

// R5: the hybrid policy, checked against hand-chosen extremes and an independent walk.
void r_hybrid_tests() {
    section("r_hybrid", [] {
        std::vector<int> schedule = {0, 1, 0, 2, 3, 0, 1, 2};
        moer::CostModel cost;
        cost.bytes = 16u << 20;
        cost.h2d_gbps = 12.0; // 16 MiB / 12 GB/s = 1.398 ms
        cost.gpu_ms = 0.2;
        double fetch = double(cost.bytes) / (cost.h2d_gbps * 1e9) * 1e3;
        // A CPU that is slower than fetch + GPU compute: everything goes to the GPU.
        cost.cpu_ms = 100;
        auto gpu = moer::hybrid_plan(schedule, 2, cost);
        check(gpu.cpu_steps == 0 && gpu.gpu_steps == int(schedule.size()),
              "r_hybrid a slow CPU keeps every expert on the GPU", fmt(0, gpu.cpu_steps));
        // 2 slots under LRU on 0,1,0,2,3,0,1,2: hits at step 2 only, so 7 fetches.
        check(gpu.resident_hits == 1 && gpu.copied_bytes == 7 * cost.bytes,
              "r_hybrid counts LRU hits and copied bytes",
              fmt(double(7 * cost.bytes), double(gpu.copied_bytes)));
        check(std::abs(gpu.total_ms - (7 * (fetch + cost.gpu_ms) + cost.gpu_ms)) < 1e-9,
              "r_hybrid totals the modeled per-step costs",
              fmt(7 * (fetch + cost.gpu_ms) + cost.gpu_ms, gpu.total_ms));
        // A CPU faster than a fetch: nothing is ever copied, so nothing becomes resident and
        // the repeat of expert 0 at step 2 still runs on the CPU. A policy that admitted the
        // expert anyway would report a hit here.
        cost.cpu_ms = 0.5;
        auto cpu = moer::hybrid_plan(schedule, 2, cost);
        check(cpu.gpu_steps == 0 && cpu.resident_hits == 0 && cpu.copied_bytes == 0,
              "r_hybrid a fast CPU avoids the copies entirely", fmt(0, double(cpu.copied_bytes)));
        check(cpu.step.size() == schedule.size() && cpu.cpu_steps == int(schedule.size()) &&
                  cpu.step[2].on_cpu,
              "r_hybrid CPU steps do not admit the expert, so nothing becomes resident",
              fmt(double(schedule.size()), cpu.cpu_steps));
        check(std::abs(cpu.total_ms - cost.cpu_ms * double(schedule.size())) < 1e-9,
              "r_hybrid an all-CPU plan costs cpu_ms per step",
              fmt(cost.cpu_ms * double(schedule.size()), cpu.total_ms));
        // fetch <= cpu_ms < fetch + gpu_ms: the CPU beats a fetch plus its GPU compute, so the
        // expert belongs on the CPU. Comparing against the transfer alone sends it to the GPU.
        cost.cpu_ms = fetch + cost.gpu_ms / 2;
        auto boundary = moer::hybrid_plan({6, 7}, 1, cost);
        check(boundary.cpu_steps == 2 && boundary.copied_bytes == 0,
              "r_hybrid counts the GPU compute that follows a fetch", fmt(2, boundary.cpu_steps));
        // Recency must be refreshed on a hit. On 0,1,0,2,1 with two slots, LRU evicts 1 at
        // step 3 (0 was just used) and misses it at step 4: 1 hit and 4 fetches. A FIFO that
        // ignores the hit keeps 1 resident and reports 2 hits with 3 fetches.
        cost.cpu_ms = 100;
        auto recency = moer::hybrid_plan({0, 1, 0, 2, 1}, 2, cost);
        check(recency.resident_hits == 1 && recency.copied_bytes == 4 * cost.bytes,
              "r_hybrid a hit refreshes recency (LRU, not FIFO)",
              "hits=" + std::to_string(recency.resident_hits) +
                  " copies=" + std::to_string(recency.copied_bytes / cost.bytes));
        // The tie goes to the GPU: the copy also warms the cache for later steps.
        cost.cpu_ms = fetch + cost.gpu_ms;
        auto tie = moer::hybrid_plan({5, 5}, 1, cost);
        check(tie.cpu_steps == 0 && tie.resident_hits == 1,
              "r_hybrid breaks a tie towards the GPU");
        // Independent recomputation for a middle cost, with its own LRU bookkeeping.
        cost.cpu_ms = fetch * 0.8;
        auto mid = moer::hybrid_plan(schedule, 2, cost);
        std::vector<int> resident;
        double total = 0;
        size_t copied = 0;
        int hits = 0, cpu_steps = 0;
        for (int e : schedule) {
            auto it = std::find(resident.begin(), resident.end(), e);
            if (it != resident.end()) {
                resident.erase(it);
                resident.push_back(e);
                total += cost.gpu_ms;
                ++hits;
            } else if (cost.cpu_ms < fetch + cost.gpu_ms) {
                total += cost.cpu_ms;
                ++cpu_steps;
            } else {
                if (resident.size() == 2) resident.erase(resident.begin());
                resident.push_back(e);
                total += fetch + cost.gpu_ms;
                copied += cost.bytes;
            }
        }
        check(std::abs(mid.total_ms - total) < 1e-9 && mid.copied_bytes == copied &&
                  mid.resident_hits == hits && mid.cpu_steps == cpu_steps,
              "r_hybrid matches an independent LRU walk", fmt(total, mid.total_ms));
    });
}
} // namespace

int main(int argc, char **argv) {
    std::string group = argc > 1 ? argv[1] : "all";
    rfix::print_device(rfix::device_info());
    if (group == "R1" || group == "all") r_pool_tests();
    if (group == "R2" || group == "all") r_pipeline_tests();
    if (group == "race" || group == "all") r_race_tests();
    if (group == "R3" || group == "all") r_overlap_tests();
    if (group == "R4" || group == "all") r_bandwidth_tests();
    if (group == "R5" || group == "all") r_hybrid_tests();
    std::cout << "SUMMARY R:" << group << ": " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
