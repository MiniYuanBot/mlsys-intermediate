// What gpu_bench.cu measures. gpu_test R5 checks this plan before any number is
// taken: on the RTX 4060 Laptop the L2 is 32 MiB, larger than every Qwen3-1.7B linear
// layer in INT4/FP8 (and q/o in FP16), so back-to-back reps would read the weights
// from L2 instead of DRAM. Real decode streams every other layer through L2 between
// two uses of one layer, so a timed rep must start with this layer's weights evicted:
// either an L2 flush of >= 2x L2 runs between reps (outside the timed region), or the
// weights of the path alone are >= 2x L2.
#pragma once
#include <cstddef>
#include <string>

namespace plan {
struct Shape {
    const char *name;
    int rows, cols; // y[batch, rows] = x[batch, cols] W^T
    bool sweep;     // also run the batch sweep 1..64 (stage R)
};
// Qwen3-1.7B (hidden 2048, intermediate 6144, 16 q heads x 128) plus one square layer
// that is DRAM-sized even without a flush in FP16.
constexpr Shape kShapes[] = {
    {"qwen3-1.7b.q_proj", 2048, 2048, false},
    {"qwen3-1.7b.up_proj", 6144, 2048, true},
    {"qwen3-1.7b.down_proj", 2048, 6144, false},
    {"square-8192", 8192, 8192, true},
};
constexpr int kBatches[] = {1, 2, 4, 8, 16, 32, 64};
constexpr int kGroup = 128;      // INT4 and FP8 group size in the bench
constexpr int kSplitChunk = 512; // B5 split-K chunk in the bench
constexpr bool kFlushL2 = true;
constexpr size_t kFlushBytes = size_t(64) << 20;
constexpr float kSpinMs = 300; // keep the GPU busy this long before timing (P-state)
constexpr int kReps = 30, kDiscard = 5;

// Formats in fp8::Format order: INT4, FP8, FP16, FP32.
inline bool cold(size_t weight_bytes, size_t l2) {
    return (kFlushL2 && kFlushBytes >= 2 * l2) || weight_bytes >= 2 * l2;
}
// Returns "" when every (shape, format) is cold, else a list of the warm ones.
template <class Bytes> std::string cold_violations(size_t l2, Bytes bytes) {
    static const char *names[] = {"INT4", "FP8", "FP16", "FP32"};
    std::string out;
    for (const Shape &s : kShapes)
        for (int f = 0; f < 4; ++f)
            if (!cold(bytes(f, s.rows, s.cols, kGroup), l2))
                out += std::string(" ") + s.name + "/" + names[f];
    return out;
}
} // namespace plan
