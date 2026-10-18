// Stage R host side: FP8 E4M3 weight format and the byte model used by the bench.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace fp8 {
// E4M3 ("e4m3fn", the OCP / NVIDIA variant on sm_89): 1 sign, 4 exponent bits with
// bias 7, 3 mantissa bits. Exponent field 0 is subnormal: (m/8) * 2^-6. There is no
// infinity; S.1111.111 is NaN, so the largest finite magnitude is 1.75 * 2^8 = 448.
constexpr float kMax = 448.f;

// R1: round FP32 x to the nearest E4M3 code, ties to an even mantissa (RNE), the way
// cvt.rn.satfinite.e4m3x2.f32 does on sm_89:
//   - NaN -> 0x7F; |x| >= 448 and +-inf saturate to +-448 (0x7E / 0xFE);
//   - subnormals are kept (smallest 2^-9); |x| <= 2^-10 rounds to a signed zero;
//   - the sign bit is kept for zero results (-0.0 -> 0x80).
// Test: R1 brute-force nearest search over all codes + cross-check with the GPU cvt.
inline uint8_t encode_e4m3(float x) {
    // BEGIN TODO R1
    (void)x;
    throw std::logic_error("Complete R1");
    // END TODO R1
}

struct Fp8Packed {
    int rows, cols, group, stride; // stride = cols rounded up to a multiple of 16
    std::vector<uint8_t> codes;    // [rows, stride], padding bytes are 0
    std::vector<float> scales;     // [rows, ceil(cols / group)]
};

// R2: per-(row, group) scaling for E4M3 weights. A group is `group` consecutive
// columns of one row (group == cols gives per-output-channel scaling). Contract:
//   scale = max |w| / 448 over the group in FP32 (1 for an all-zero group);
//   code[r * stride + c] = encode_e4m3(w / scale) with FP32 division;
//   stride = cols rounded up to 16 so that each row can be read with 16-byte loads;
//   padding bytes are 0. group must be a positive multiple of 16 (a 16-byte chunk then
//   never straddles two groups); throw std::invalid_argument otherwise, for bad
//   shapes and for non-finite weights.
// Test: R2 hand fixture (two groups with different absmax), FP64 per-group oracle.
inline Fp8Packed fp8_pack(const std::vector<float> &w, int rows, int cols, int group) {
    // BEGIN TODO R2
    (void)w, (void)rows, (void)cols, (void)group;
    throw std::logic_error("Complete R2");
    // END TODO R2
}

enum class Format { INT4, FP8, FP16, FP32 };

// R5: logical bytes a GEMV must read for the weights of one [rows, cols] layer,
// metadata included (this is the b_w * rows * cols of the lecture's byte model):
//   INT4: rows * ceil(cols / 2) codes + 4 B per (row, group) scale  (lab layout)
//   FP8 : rows * stride (cols rounded up to 16) codes + 4 B per (row, group) scale
//   FP16: 2 * rows * cols;  FP32: 4 * rows * cols  (group is ignored)
// The bench adds activation and output bytes itself. Tests: R5 hand-computed
// sizes; the bench plan check (every timed rep must start with weights out of L2).
inline size_t weight_bytes(Format f, int rows, int cols, int group) {
    // BEGIN TODO R5
    (void)f, (void)rows, (void)cols, (void)group;
    throw std::logic_error("Complete R5");
    // END TODO R5
}
} // namespace fp8
