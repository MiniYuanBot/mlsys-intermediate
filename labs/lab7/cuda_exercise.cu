#include "fixture.hpp"
#include "src/pipeline_student.cuh"
#include <chrono>
#include <iostream>
int main(int argc, char **argv) {
    bool bench = argc > 1 && std::string(argv[1]) == "bench";
    std::cout << "kind,seed,width,hidden,slots,ms,max_abs_error,weight_slot_bytes\n";
    for (int seed : {1, 2, 3})
        for (int width : (bench ? std::vector<int>{64, 128} : std::vector<int>{7, 33})) {
            Fixture f(bench ? 48 : 5, width, width * 2 + 1, 8, seed);
            auto expected = moe::forward(f.experts, f.dispatch, f.x, f.tokens, width);
            for (int slots : {1, 2}) {
                if (bench) moe::offload_forward(f.experts, f.dispatch, f.x, f.tokens, width, slots);
                std::vector<double> samples;
                std::vector<float> got;
                for (int repeat = 0; repeat < (bench ? 5 : 1); ++repeat) {
                    auto start = std::chrono::steady_clock::now();
                    got = moe::offload_forward(f.experts, f.dispatch, f.x, f.tokens, width, slots);
                    samples.push_back(std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - start)
                                          .count());
                }
                std::sort(samples.begin(), samples.end());
                double ms = samples[samples.size() / 2];
                if (got.size() != expected.size()) throw std::runtime_error("output size mismatch");
                for (float value : got)
                    if (!std::isfinite(value)) throw std::runtime_error("nonfinite CUDA output");
                float error = 0;
                for (size_t i = 0; i < got.size(); ++i)
                    error = std::max(error, std::abs(got[i] - expected[i]));
                if (error > 2e-4f) throw std::runtime_error("c_cuda_forward value mismatch");
                size_t bytes =
                    slots * (2 * size_t(f.hidden) * width + (f.hidden + width) * sizeof(float));
                std::cout << "measured_end_to_end," << seed << "," << width << "," << f.hidden
                          << "," << slots << "," << ms << "," << error << "," << bytes << "\n";
            }
        }
    if (!bench)
        std::cout
            << "PASS c_cuda_forward: actual INT8 MLP, odd dimensions, unused experts, slot reuse\n";
}
