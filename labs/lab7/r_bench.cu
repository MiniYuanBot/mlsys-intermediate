// Stage R benchmark: real H2D bandwidth, the copy/compute overlap of an MB-scale expert
// pipeline, and the CPU/GPU hybrid crossover on a routing trace.
// Timing: cudaEvent around each segment and around the whole pipeline; every configuration
// is spin-warmed for >= 300 ms with the same launch before the timed repetitions, and each
// row carries p25/p75 plus a `stable` flag (p75 <= 1.5 * p25). Rows marked NO were measured
// while the clocks or a co-tenant moved; rerun them before using the numbers.
// Bytes: the logical payload each step moves over PCIe, not profiler counters.
#include "src/r_pipeline.cuh"
#include <chrono>
#include <iostream>

namespace {
std::vector<int> cycle_schedule(int steps, int experts) {
    std::vector<int> s;
    for (int i = 0; i < steps; ++i) s.push_back(i % experts);
    return s;
}
// Measure the CPU expert on the same payload, so the hybrid cost model uses two measured
// numbers instead of one measured and one guessed.
double cpu_expert_ms(const int8_t *payload, size_t bytes, int reads) {
    volatile long long sink = 0;
    sink = rfix::cpu_consume(payload, bytes, reads); // warm the caches
    std::vector<double> ms;
    for (int rep = 0; rep < 5; ++rep) {
        auto begin = std::chrono::steady_clock::now();
        sink = rfix::cpu_consume(payload, bytes, reads);
        ms.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                .count());
    }
    (void)sink;
    return rfix::summarize(ms).median;
}
} // namespace

int main() {
    auto info = rfix::device_info();
    rfix::print_device(info);
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    std::cout << "# driver " << driver << ", runtime " << runtime << "; asyncEngineCount "
              << info.async_engines
              << (info.async_engines <= 1 ? " (H2D and D2H serialise; a copy can still overlap "
                                            "a kernel)"
                                          : " (H2D and D2H have separate engines)")
              << '\n';

    // ---- R4: pinned vs pageable H2D ----
    std::cout << "kind,memory,bytes,median_ms,gbps,stable\n";
    double pinned_gbps = 0;
    for (size_t mib : {1u, 4u, 16u, 32u, 64u})
        for (bool pinned : {true, false}) {
            size_t bytes = mib << 20;
            auto b = moer::measure_h2d(bytes, pinned);
            if (pinned && mib == 32) pinned_gbps = b.gbps;
            std::cout << "measured_h2d," << (b.is_pinned ? "pinned" : "pageable") << "," << bytes
                      << "," << b.median_ms << "," << b.gbps << "," << (b.stable ? "yes" : "NO")
                      << '\n';
        }

    // ---- R2/R3: overlap across expert size, slots and compute intensity ----
    // reads is the compute knob: one pass reads the whole slot from device memory, so
    // reads ~ 16 puts GPU compute near the copy time of the same expert on this machine.
    std::cout << "kind,expert_mib,slots,reads,steps,wall_ms,p25_ms,p75_ms,copy_sum_ms,"
                 "compute_sum_ms,serial_ms,overlap_ratio,h2d_gbps,stable\n";
    const int steps = 12;
    for (size_t mib : {4u, 8u, 16u, 32u}) {
        size_t bytes = mib << 20;
        const int experts = 6;
        // 6 experts of 32 MiB plus 3 slots is 288 MiB: far inside 8 GiB, and the pinned pool
        // stays small enough for WSL2's undocumented pinned-memory limit.
        for (int slots : {1, 2, 3})
            for (int reads : {2, 16}) {
                auto schedule = cycle_schedule(steps, experts);
                moer::Pool pool;
                try {
                    pool = moer::build_pool(experts, bytes, slots, steps);
                } catch (const std::exception &e) {
                    std::cout << "# skipped " << mib << " MiB x " << slots << " slots: " << e.what()
                              << '\n';
                    continue;
                }
                moer::PipelineConfig cfg;
                cfg.reads = reads;
                rfix::spin_warmup([&] { moer::run_pipeline(pool, schedule, cfg); });
                std::vector<double> wall, ratio, copy_sum, compute_sum;
                for (int rep = 0; rep < 9; ++rep) {
                    auto r = moer::run_pipeline(pool, schedule, cfg);
                    for (size_t i = 0; i < schedule.size(); ++i)
                        if (r.sum[i] != rfix::expected_sum(schedule[i], bytes) || r.bad)
                            throw std::runtime_error("stage R checksum mismatch during the bench");
                    wall.push_back(r.wall_ms);
                    ratio.push_back(moer::overlap_ratio(r));
                    copy_sum.push_back(r.copy_sum_ms);
                    compute_sum.push_back(r.compute_sum_ms);
                }
                auto w = rfix::summarize(wall);
                auto c = rfix::summarize(copy_sum);
                auto k = rfix::summarize(compute_sum);
                auto o = rfix::summarize(ratio);
                double moved = double(steps) * double(bytes);
                std::cout << "measured_pipeline," << mib << "," << slots << "," << reads << ","
                          << steps << "," << w.median << "," << w.p25 << "," << w.p75 << ","
                          << c.median << "," << k.median << "," << c.median + k.median << ","
                          << o.median << "," << moved / (c.median * 1e-3) / 1e9 << ","
                          << (w.stable ? "yes" : "NO") << '\n';
                moer::destroy_pool(pool);
            }
    }

    // ---- R5: the CPU/GPU crossover on a routing trace ----
    // Both sides are measured on this machine: h2d_gbps from R4, gpu_ms from the pipeline's
    // own compute segments, cpu_ms from the scalar host loop over the same payload.
    // cpu_speedup projects what a vectorised host kernel would buy (kTransformers uses AMX,
    // this laptop has neither AMX nor a hand-written AVX2 expert): cpu_ms / cpu_speedup is
    // modeled, so these rows are labelled modeled_from_measured and must not be read as a
    // measured CPU time.
    std::cout << "kind,expert_mib,reads,slots,cpu_speedup,cpu_ms,gpu_ms,fetch_ms,cpu_steps,"
                 "gpu_steps,resident_hits,copied_mib,plan_ms,all_gpu_ms,all_cpu_ms\n";
    for (size_t mib : {4u, 16u}) {
        size_t bytes = mib << 20;
        const int experts = 8;
        moer::Pool pool;
        try {
            pool = moer::build_pool(experts, bytes, 2, steps);
        } catch (const std::exception &e) {
            std::cout << "# skipped hybrid at " << mib << " MiB: " << e.what() << '\n';
            continue;
        }
        for (int reads : {2, 16}) {
            auto schedule = cycle_schedule(steps, experts);
            moer::PipelineConfig cfg;
            cfg.reads = reads;
            rfix::spin_warmup([&] { moer::run_pipeline(pool, schedule, cfg); });
            auto r = moer::run_pipeline(pool, schedule, cfg);
            double gpu_ms = r.compute_sum_ms / double(steps);
            double measured_cpu_ms = cpu_expert_ms(pool.host, bytes, reads);
            double fetch = double(bytes) / (pinned_gbps * 1e9) * 1e3;
            // Skewed trace: expert 0 is hot (every third access), the rest cycles through the
            // cold tail, so a good policy keeps the hot expert resident and decides per cold
            // singleton whether a fetch is worth it.
            std::vector<int> trace;
            for (int i = 0; i < 48; ++i) trace.push_back(i % 3 == 0 ? 0 : 1 + (i * 5) % 7);
            for (int slots : {1, 2, 4})
                for (double speedup : {1.0, 4.0, 16.0, 64.0}) {
                    if (slots != 2 && speedup != 1.0) continue; // sweep one axis at a time
                    moer::CostModel cost{bytes, pinned_gbps, gpu_ms, measured_cpu_ms / speedup};
                    auto plan = moer::hybrid_plan(trace, slots, cost);
                    moer::CostModel never = cost;
                    never.cpu_ms = 1e9; // force the GPU everywhere
                    auto all_gpu = moer::hybrid_plan(trace, slots, never);
                    std::cout << "modeled_from_measured," << mib << "," << reads << "," << slots
                              << "," << speedup << "," << cost.cpu_ms << "," << gpu_ms << ","
                              << fetch << "," << plan.cpu_steps << "," << plan.gpu_steps << ","
                              << plan.resident_hits << "," << double(plan.copied_bytes) / (1 << 20)
                              << "," << plan.total_ms << "," << all_gpu.total_ms << ","
                              << cost.cpu_ms * double(trace.size()) << '\n';
                }
        }
        moer::destroy_pool(pool);
    }
    std::cout << "RESULT: stage R bench completed\n";
    return 0;
}
