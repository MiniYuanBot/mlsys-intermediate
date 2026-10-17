#include "common.cuh"
int main() {
    for (int rows : {1, 17, 64})
        for (int width : {1, 31, 128, 513, 1024}) {
            Fixture f(rows, width);
            f.fused();
            f.check();
            f.baseline();
            f.check();
        }
    std::cout << "PASS fused residual RMSNorm vs double-precision CPU oracle\n";
}
