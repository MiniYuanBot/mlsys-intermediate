#include "common.cuh"
int main() {
    std::cout << "rows,width,method,repeat,microseconds\n";
    for (int rows : {1, 8, 32, 128})
        for (int width : {128, 1024, 4096}) {
            Fixture f(rows, width);
            for (int rep = 0; rep < 5; ++rep)
                for (int j = 0; j < 2; ++j) {
                    int m = (j + rep) % 2;
                    float us = m ? f.time([&] { f.fused(); }) : f.time([&] { f.baseline(); });
                    f.check();
                    std::cout << rows << ',' << width << ',' << (m ? "fused" : "two_kernel") << ','
                              << rep << ',' << us << '\n';
                }
        }
}
