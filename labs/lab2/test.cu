#include "common.cuh"
int main() {
    for (auto shape :
         std::vector<std::pair<int, int>>{{1, 1}, {3, 31}, {37, 65}, {128, 256}, {513, 1025}}) {
        Fixture f(shape.first, shape.second, 32);
        f.fused();
        f.error();
        f.decomposed();
        f.error();
    }
    std::cout << "PASS W4A32 GEMV vs independent CPU oracle; odd tails and multiwarp rows\n";
}
