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
    if (std::isnan(x)) return 0x7F;
    const uint8_t sign = std::signbit(x) ? 0x80 : 0;
    const double a = std::fabs(double(x));
    if (!(a < kMax)) return sign | 0x7E; // saturate, also catches +-inf
    if (a == 0) return sign;
    int e = 0;
    std::frexp(a, &e);                           // a in [2^(e-1), 2^e)
    const int quantum = std::max(e - 1, -6) - 3; // spacing of the e4m3 grid around a
    const double units = std::nearbyint(std::ldexp(a, -quantum)); // RNE in grid units
    const double v = std::ldexp(units, quantum); // exact, may have moved up a binade
    if (v == 0) return sign;
    int ve = 0;
    std::frexp(v, &ve);
    const int field = ve - 1 + 7;                                         // biased exponent of v
    if (field <= 0) return sign | uint8_t(std::lround(std::ldexp(v, 9))); // subnormal
    const int mantissa = int(std::lround(std::ldexp(v, -(ve - 1)) * 8)) - 8;
    return sign | uint8_t(field << 3 | mantissa);
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
    if (rows <= 0 || cols <= 0 || group <= 0 || group % 16 != 0 || w.size() != size_t(rows) * cols)
        throw std::invalid_argument("fp8_pack shape");
    const int stride = (cols + 15) / 16 * 16;
    const int ng = (cols + group - 1) / group;
    Fp8Packed p{rows,
                cols,
                group,
                stride,
                std::vector<uint8_t>(size_t(rows) * stride, 0),
                std::vector<float>(size_t(rows) * ng)};
    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < ng; ++g) {
            const int begin = g * group, end = std::min(cols, begin + group);
            float maximum = 0;
            for (int c = begin; c < end; ++c) {
                const float v = w[size_t(r) * cols + c];
                if (!std::isfinite(v)) throw std::invalid_argument("non-finite weight");
                maximum = std::max(maximum, std::fabs(v));
            }
            const float scale = maximum > 0 ? maximum / kMax : 1.f;
            p.scales[size_t(r) * ng + g] = scale;
            for (int c = begin; c < end; ++c)
                p.codes[size_t(r) * stride + c] = encode_e4m3(w[size_t(r) * cols + c] / scale);
        }
    }
    return p;
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
    const size_t r = size_t(rows), c = size_t(cols);
    const size_t groups = r * ((c + group - 1) / group);
    switch (f) {
    case Format::INT4:
        return r * ((c + 1) / 2) + 4 * groups;
    case Format::FP8:
        return r * ((c + 15) / 16 * 16) + 4 * groups;
    case Format::FP16:
        return 2 * r * c;
    case Format::FP32:
        return 4 * r * c;
    }
    throw std::invalid_argument("format");
    // END TODO R5
}
} // namespace fp8
