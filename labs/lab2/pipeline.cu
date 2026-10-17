#define LOWBIT_NO_MAIN
#include "quant_test.cpp"
#include "src/gemv_student.cuh"
#include <functional>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        auto e = (x);                                                                              \
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));                     \
    } while (0)
template <class T> T *upload(const std::vector<T> &x) {
    T *p;
    CHECK(cudaMalloc(&p, x.size() * sizeof(T)));
    CHECK(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
float time_us(const std::function<void()> &fn) {
    for (int i = 0; i < 5; ++i) fn();
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < 30; ++i) fn();
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));
    return ms * 1000 / 30;
}
void gpu_case(const Packed &p, const std::vector<float> &x, int batch, bool bench,
              const std::string &tag) {
    auto expected = packed_linear(p, x, batch);
    auto dp = upload(p.bytes);
    auto ds = upload(p.scales);
    auto dx = upload(x);
    float *dy;
    float *tmp;
    CHECK(cudaMalloc(&dy, batch * p.rows * sizeof(float)));
    CHECK(cudaMalloc(&tmp, p.rows * ((p.cols + 16) / 17) * sizeof(float)));
    for (int mode = 0; mode < 4; ++mode) {
        auto launch = [&]() {
            if (mode == 2) {
                launch_batch(dp, ds, dx, dy, p.rows, p.cols, p.group, batch);
                return;
            }
            for (int b = 0; b < batch; ++b) {
                auto in = dx + b * p.cols;
                auto out = dy + b * p.rows;
                if (mode == 0) launch_fused(dp, ds, in, out, p.rows, p.cols, p.group);
                if (mode == 1) launch_pair(dp, ds, in, out, p.rows, p.cols, p.group);
                if (mode == 3) launch_split(dp, ds, in, out, tmp, p.rows, p.cols, p.group, 17);
            }
        };
        launch();
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        std::vector<float> got(batch * p.rows);
        CHECK(cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < got.size(); ++i) close(got[i], expected[i], 3e-4);
        if (bench)
            std::cout << tag << ',' << p.rows << ',' << p.cols << ',' << p.group << ',' << batch
                      << ',' << mode << ',' << time_us(launch) << '\n';
    }
    cudaFree(dp);
    cudaFree(ds);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(tmp);
}
int main(int argc, char **argv) {
    try {
        std::string part = argc > 1 ? argv[1] : "all";
        bool bench = argc > 2;
        int devices = 0;
        CHECK(cudaGetDeviceCount(&devices));
        require(devices > 0, "CUDA device required");
        if (bench) std::cout << "case,rows,cols,group,batch,mode,device_us\n";
        if (part == "B" || part == "all") {
            if (!bench) {
                for (int m : {1, 7, 33})
                    for (int n : {1, 31, 65, 257})
                        for (int g : {3, 32, 128})
                            for (int b : {1, 3, 4, 7})
                                gpu_case(pack(random_values(m * n, 17), m, n, g),
                                         random_values(b * n, 19), b, false, "kernels");
            } else {
                for (int n : {256, 1024, 2048})
                    for (int b : {1, 4, 7})
                        gpu_case(pack(random_values(n * n, 17), n, n, 128),
                                 random_values(b * n, 19), b, true, "kernels");
            }
            if (!bench)
                std::cout << "PASS B1/B2 warp_kernel; B3 pair_group_boundary; B4 tiled_batch; "
                             "B5/B6 split_k_tail\n";
        }
        if (part == "C" || part == "all") {
            int m = 31, n = 65, b = 7;
            auto w = random_values(m * n, 101);
            auto calib = random_values(13 * n, 103);
            for (int t = 0; t < 13; ++t) calib[t * n] *= 10;
            auto model = search(w, calib, m, n, 13, 16, {0, .25, .5, .75, 1});
            auto held = random_values(b * n, 107);
            std::vector<float> scaled = held;
            for (int t = 0; t < b; ++t)
                for (int k = 0; k < n; ++k) scaled[t * n + k] /= model.channels[k];
            auto cpu = execute(model, held, b), artifact = packed_linear(model.weight, scaled, b);
            for (size_t i = 0; i < cpu.size(); ++i) close(cpu[i], artifact[i]);
            gpu_case(model.weight, scaled, b, bench, "frozen_calibration");
            if (!bench)
                std::cout << "PASS C frozen artifact -> inverse scales -> CUDA packed batch -> "
                             "independent output\n";
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
