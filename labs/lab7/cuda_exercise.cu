// Stage C on the GPU: C3 (expert kernel), C6 (pinned expert record) and C4 (the slot
// pipeline). Every check prints one PASS/FAIL line and the run continues past failures.
// The oracle is the CPU forward from stage C2, which itself is checked against an FP64
// oracle in test.cpp, so a GPU result is never compared against another GPU result.
// Quantitative overlap belongs to stage R (r_bench.cu): the records here are 17-67 KB, far
// below what a PCIe measurement can resolve, and the wall time includes allocation.
#include "fixture.hpp"
#include "src/pipeline_student.cuh"
#include <chrono>
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
float worst_error(const std::vector<float> &got, const std::vector<float> &want) {
    if (got.size() != want.size()) return 1e30f;
    float error = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) return 1e30f;
        error = std::max(error, std::abs(got[i] - want[i]));
    }
    return error;
}
} // namespace

int main(int argc, char **argv) {
    bool bench = argc > 1 && std::string(argv[1]) == "bench";
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "# device " << prop.name << " sm_" << prop.major << prop.minor
              << ", asyncEngineCount " << prop.asyncEngineCount << '\n';
    if (bench) {
        std::cout << "kind,seed,width,hidden,slots,ms,max_abs_error,weight_slot_bytes\n";
        for (int seed : {1, 2, 3})
            for (int width : {64, 128}) {
                Fixture f(48, width, width * 2 + 1, 8, seed);
                auto expected = moe::forward(f.experts, f.dispatch, f.x, f.tokens, width);
                for (int slots : {1, 2}) {
                    moe::offload_forward(f.experts, f.dispatch, f.x, f.tokens, width, slots);
                    std::vector<double> samples;
                    std::vector<float> got;
                    for (int repeat = 0; repeat < 5; ++repeat) {
                        auto start = std::chrono::steady_clock::now();
                        got = moe::offload_forward(f.experts, f.dispatch, f.x, f.tokens, width,
                                                   slots);
                        samples.push_back(std::chrono::duration<double, std::milli>(
                                              std::chrono::steady_clock::now() - start)
                                              .count());
                    }
                    std::sort(samples.begin(), samples.end());
                    float error = worst_error(got, expected);
                    if (error > 2e-4f) {
                        std::cout << "FAIL c_cuda_forward bench seed " << seed << " width " << width
                                  << " slots " << slots << ": " << fmt(0, error) << '\n';
                        return 1;
                    }
                    size_t bytes = size_t(slots) * (2 * size_t(f.hidden) * width +
                                                    (f.hidden + width) * sizeof(float));
                    std::cout << "measured_end_to_end," << seed << "," << width << "," << f.hidden
                              << "," << slots << "," << samples[2] << "," << error << "," << bytes
                              << '\n';
                }
            }
        std::cout << "# wall time covers the whole offload_forward: pinning, cudaMalloc, stream "
                     "and event creation, H2D, kernels, D2H and the CPU combine. Steady-state "
                     "copy/compute overlap is measured in stage R, not here.\n";
        return 0;
    }

    // Correctness: odd dimensions, unused experts, repeated slot reuse, 1 and 2 slots.
    for (int seed : {1, 2, 3})
        for (int width : {7, 33}) {
            Fixture f(5, width, width * 2 + 1, 8, seed);
            auto expected = moe::forward(f.experts, f.dispatch, f.x, f.tokens, width);
            for (int slots : {1, 2}) {
                std::string label = "c_cuda_forward seed " + std::to_string(seed) + " width " +
                                    std::to_string(width) + " slots " + std::to_string(slots);
                try {
                    auto got =
                        moe::offload_forward(f.experts, f.dispatch, f.x, f.tokens, width, slots);
                    float error = worst_error(got, expected);
                    check(error <= 2e-4f, label, fmt(0, error));
                } catch (const std::exception &e) {
                    check(false, label, std::string("threw: ") + e.what());
                }
            }
        }
    // Dependency stress: 8 experts over 1 or 2 slots with many routes per expert, so each
    // slot is overwritten while earlier kernels on that slot are still running unless the
    // copy waits for that slot's previous compute. Repeated to expose an intermittent race.
    for (int slots : {1, 2}) {
        std::string label = "c_cuda_slot_reuse slots " + std::to_string(slots);
        try {
            Fixture f(512, 65, 129, 8, 17);
            auto expected = moe::forward(f.experts, f.dispatch, f.x, f.tokens, 65);
            float error = 0;
            for (int repeat = 0; repeat < 8; ++repeat) {
                auto got = moe::offload_forward(f.experts, f.dispatch, f.x, f.tokens, 65, slots);
                error = std::max(error, worst_error(got, expected));
            }
            check(error <= 2e-4f, label + " (8 repeats, 512 tokens)", fmt(0, error));
        } catch (const std::exception &e) {
            check(false, label, std::string("threw: ") + e.what());
        }
    }
    // An empty dispatch must launch no work and return zeros.
    try {
        Fixture f(4, 9, 19, 3, 5);
        auto got = moe::offload_forward(f.experts, moe::dispatch({}, 4, 3), f.x, 4, 9, 2);
        bool zeros = got.size() == 36;
        for (float v : got) zeros &= v == 0;
        check(zeros, "c_cuda_forward an empty dispatch returns zeros without launching work");
    } catch (const std::exception &e) {
        check(false, "c_cuda_forward empty dispatch", std::string("threw: ") + e.what());
    }
    std::cout << "SUMMARY C cuda: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
