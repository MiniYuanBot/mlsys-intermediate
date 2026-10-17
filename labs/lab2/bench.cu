#include "common.cuh"
int main() {
    std::cout << "m,n,group,method,repeat,microseconds,kernel_max_abs_error,quant_relative_l2\n";
    for (int n : {512, 2048, 4096})
        for (int g : {32, 128}) {
            Fixture f(n, n, g);
            for (int rep = 0; rep < 5; ++rep) {
                // Alternate order to reduce systematic clock/thermal bias.
                for (int j = 0; j < 3; ++j) {
                    int method = (j + rep) % 3;
                    float us = method == 0   ? f.time([&] { f.fused(); })
                               : method == 1 ? f.time([&] { f.decomposed(); })
                                             : f.time([&] { f.blas(); });
                    std::cout << n << ',' << n << ',' << g << ','
                              << (method == 0   ? "fused"
                                  : method == 1 ? "unpack_then_cublas"
                                                : "resident_fp32_cublas")
                              << ',' << rep << ',' << us << ',' << f.error() << ','
                              << f.quant_error() << '\n';
                }
            }
        }
}
