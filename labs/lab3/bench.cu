// C6/C7 benchmark: fused residual-add + RMSNorm vs the two-kernel baseline (FP32 and FP16).
// Measurement rules (labs/docs/authoring.md section 5):
//  * the timed unit is one CUDA graph that replays kLaunches back-to-back launches, so CPU
//    launch overhead is excluded and both methods are timed the same way as the Inductor
//    kernel in inductor_bench.py; reported time = graph time / kLaunches;
//  * before the timed reps the same graph is replayed until the GPU has been busy >= 300 ms
//    (the laptop GPU drops to a low memory clock when idle);
//  * 30 reps, the first 5 dropped; median, p25, p75 and stable = (p75 <= 1.5 * p25).
// x is all zeros here so the in-place residual stays bounded over ~1e5 launches; the kernels
// do no data-dependent work, so timing is unaffected. Correctness lives in test.cu.
#include "fusion_common.cuh"

#include <iostream>

constexpr int kLaunches = 20;
constexpr float kSpinMs = 300.0f;

struct Stats {
    double median, p25, p75;
};
static Stats summarize(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    auto q = [&](double p) { return double(v[size_t(p * (v.size() - 1) + 0.5)]); };
    return {q(0.5), q(0.25), q(0.75)};
}

template <class F> Stats time_graph(F launch_once) {
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    launch_once(stream); // warm caches/modules outside capture
    CUDA_OK(cudaStreamSynchronize(stream));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < kLaunches; ++i) launch_once(stream);
    CUDA_OK(cudaStreamEndCapture(stream, &graph));
    CUDA_OK(cudaGraphInstantiate(&exec, graph, 0));
    cudaEvent_t a, b;
    CUDA_OK(cudaEventCreate(&a));
    CUDA_OK(cudaEventCreate(&b));
    CUDA_OK(cudaEventRecord(a, stream));
    for (float busy = 0; busy < kSpinMs;) {
        for (int i = 0; i < 8; ++i) CUDA_OK(cudaGraphLaunch(exec, stream));
        CUDA_OK(cudaEventRecord(b, stream));
        CUDA_OK(cudaEventSynchronize(b));
        CUDA_OK(cudaEventElapsedTime(&busy, a, b));
    }
    std::vector<float> us;
    for (int rep = 0; rep < 30; ++rep) {
        CUDA_OK(cudaEventRecord(a, stream));
        CUDA_OK(cudaGraphLaunch(exec, stream));
        CUDA_OK(cudaEventRecord(b, stream));
        CUDA_OK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_OK(cudaEventElapsedTime(&ms, a, b));
        if (rep >= 5) us.push_back(ms * 1000.0f / kLaunches);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return summarize(us);
}

template <class T> void run(const char *dtype, int rows, int width) {
    Fixture<T> f(rows, width, 7, 1.0f, /*zero_x=*/true);
    // Ideal DRAM traffic (see lecture 3.2): read x and residual, write residual and y.
    const double ideal_bytes = 4.0 * rows * width * sizeof(T) + width * sizeof(T);
    for (int m = 0; m < 2; ++m) {
        Stats s = m == 0 ? time_graph([&](cudaStream_t st) {
            launch_two_kernel(f.x, f.r, f.g, f.y, rows, width, st);
        })
                         : time_graph([&](cudaStream_t st) {
                               launch_fused(f.x, f.r, f.g, f.y, rows, width, st);
                           });
        std::cout << dtype << ',' << rows << ',' << width << ',' << (m ? "fused" : "two_kernel")
                  << ',' << s.median << ',' << s.p25 << ',' << s.p75 << ',' << ideal_bytes << ','
                  << ideal_bytes / (s.median * 1e-6) / 1e9 << ','
                  << (s.p75 <= 1.5 * s.p25 ? "yes" : "no") << '\n';
    }
}

int main() {
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "# device " << prop.name << ", L2 " << prop.l2CacheSize / (1 << 20) << " MiB; "
              << kLaunches << " launches per CUDA graph, >= " << kSpinMs << " ms spin, 25 reps\n";
    std::cout << "dtype,rows,width,method,median_us,p25_us,p75_us,ideal_bytes,gbps_ideal,stable\n";
    for (int rows : {1, 16, 128, 1024, 4096})
        for (int width : {1024, 2048, 4096}) {
            run<float>("f32", rows, width);
            run<__half>("f16", rows, width);
        }
}
