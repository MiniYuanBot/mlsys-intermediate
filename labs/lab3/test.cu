// C5-C7 correctness without PyTorch: FP64 host oracle, odd widths, exact residual write-back.
// Usage: fusion_test --only reduce|f32|f16. One PASS/FAIL line per check; a device trap
// (unfinished TODO) aborts only this process, exercise.py runs the three groups separately.
#include "fusion_common.cuh"

#include <cstring>
#include <iostream>

static int passed = 0, failed = 0;
static void record(const std::string &name, const std::string &problem) {
    if (problem.empty()) {
        ++passed;
        std::cout << "PASS " << name << "\n";
    } else {
        ++failed;
        std::cout << "FAIL " << name << ": " << problem << "\n";
    }
    std::cout.flush();
}

// Two consecutive reductions per block; every thread writes what it received.
__global__ void reduce_probe(const float *in, float *out) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    float first = block_reduce_sum(in[i]);
    float second = block_reduce_sum(2.0f * in[i] + 1.0f);
    out[2 * i] = first;
    out[2 * i + 1] = second;
}

static void test_reduce() {
    for (int threads : {32, 64, 96, 256, 544, 1024}) {
        const int blocks = 3;
        const size_t n = size_t(threads) * blocks;
        std::vector<float> h(n), out(2 * n);
        std::mt19937 rng(threads);
        std::uniform_int_distribution<int> d(-8, 8);
        for (auto &v : h) v = float(d(rng)); // small integers: every float sum is exact
        float *din, *dout;
        CUDA_OK(cudaMalloc(&din, n * sizeof(float)));
        CUDA_OK(cudaMalloc(&dout, 2 * n * sizeof(float)));
        CUDA_OK(cudaMemcpy(din, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        reduce_probe<<<blocks, threads>>>(din, dout);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaMemcpy(out.data(), dout, 2 * n * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(din);
        cudaFree(dout);
        std::string problem;
        for (int b = 0; b < blocks && problem.empty(); ++b) {
            double s = 0;
            for (int t = 0; t < threads; ++t) s += h[b * size_t(threads) + t];
            double s2 = 2 * s + threads;
            for (int t = 0; t < threads && problem.empty(); ++t) {
                size_t i = b * size_t(threads) + t;
                if (out[2 * i] != s || out[2 * i + 1] != s2)
                    problem = "block " + std::to_string(b) + " thread " + std::to_string(t) +
                              ": expected (" + std::to_string(s) + ", " + std::to_string(s2) +
                              ") vs actual (" + std::to_string(out[2 * i]) + ", " +
                              std::to_string(out[2 * i + 1]) + ")";
            }
        }
        record("C5 block_reduce_sum blockDim=" + std::to_string(threads) +
                   " (two calls, all threads see the total)",
               problem);
    }
}

template <class T>
static void check_shape(const char *tag, int rows, int width, double tol, float scale = 1.0f) {
    Fixture<T> f(rows, width, unsigned(rows * 7919 + width), scale);
    Expected<T> e = oracle(f);
    std::string label = std::string(tag) + " rows=" + std::to_string(rows) +
                        " width=" + std::to_string(width) + (scale != 1.0f ? " large values" : "");
    launch_fused(f.x, f.r, f.g, f.y, rows, width);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    record(label + " fused vs FP64 oracle (y and residual write-back)", compare(f, e, tol));
    f.reset();
    launch_two_kernel(f.x, f.r, f.g, f.y, rows, width);
    CUDA_OK(cudaDeviceSynchronize());
    std::string base = compare(f, e, tol);
    if (!base.empty()) record(label + " provided two-kernel baseline", base); // fixture sanity
}

int main(int argc, char **argv) {
    const char *only = argc == 3 && !std::strcmp(argv[1], "--only") ? argv[2] : "";
    const std::string what = only;
    if (what != "reduce" && what != "f32" && what != "f16") {
        std::cerr << "usage: fusion_test --only reduce|f32|f16\n";
        return 2;
    }
    try {
        if (what == "reduce") test_reduce();
        if (what == "f32")
            for (int rows : {1, 3, 17, 64})
                for (int width : {1, 3, 31, 127, 128, 513, 1024, 2048, 4095, 4096})
                    check_shape<float>("C6 f32", rows, width, 2e-5);
        if (what == "f16") {
            for (int rows : {1, 5, 32})
                for (int width : {1, 2, 3, 31, 64, 127, 1024, 2047, 2048, 4096})
                    check_shape<__half>("C7 f16", rows, width, 4e-3);
            check_shape<__half>("C7 f16", 4, 4096, 4e-3, 100.0f); // per-thread sums > 65504
            check_shape<__half>("C7 f16", 4, 4095, 4e-3, 100.0f);
        }
    } catch (const std::exception &error) {
        record(std::string("C-") + only + " CUDA run",
               std::string(error.what()) +
                   " (a launch failure here usually means a device trap: TODO C5 "
                   "block_reduce_sum, C6 or C7 is not finished)");
    }
    std::cout << "SUMMARY " << only << ": " << passed << "/" << (passed + failed)
              << " checks passed\n";
    return failed ? 1 : 0;
}
