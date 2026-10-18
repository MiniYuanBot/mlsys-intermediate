// Stage R: one decode step of attention over an int4 KV cache, fused (flash-decoding style).
//
// Shapes follow Qwen3-1.7B (16 query heads, 8 KV heads, head_dim 128, GQA group 2).
// One warp owns one split (a contiguous token range) of one KV head and serves every
// query head of that GQA group, so each K/V row is read from DRAM once per group.
// Each lane owns 4 consecutive channels: lane l holds channels [4l, 4l+4).
//
//   R1  Int4Cache::load4      unpack 4 nibbles + per-channel K / per-token V metadata
//                             (int4 prefix only; the FP32 residual loader is provided)
//   R2  split_partial         per-split online softmax -> (m, l, acc) partial states
//   R3  reduce_splits         log-sum-exp merge of the partial states across splits
#pragma once
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kvdec {
constexpr int kWarp = 32;
constexpr int kHeadDim = 128;               // Qwen3 head_dim
constexpr int kLaneDims = kHeadDim / kWarp; // 4 channels per lane
constexpr int kWarpsPerBlock = 4;           // one split per warp

struct DecodeShape {
    int q_heads;  // H_q (16)
    int kv_heads; // H_kv (8); query head h reads KV head h / (H_q / H_kv)
    int tokens;   // T = quantized prefix + residual window
    int split;    // tokens per split, > 0; split s covers [s*split, min(T, (s+1)*split))
    __host__ __device__ int n_splits() const { return (tokens + split - 1) / split; }
};

// One layer, all KV heads, in the lab's A/B layouts (FP32 metadata as in kv::Packed):
//   quantized prefix: n_chunks blocks of `block` tokens per head (B2 chunks, oldest first)
//     k_codes, v_codes  uint8 [kv_heads][n_chunks*block][D/2], row-major per token,
//                       byte j holds channel 2j in the low nibble and 2j+1 in the high nibble
//     K per-channel (kv::Axis::Channel with group = block):
//                       k_scale, k_offset  [kv_heads][n_chunks][D]
//     V per-token (kv::Axis::Token with group = vgroup channels):
//                       v_scale, v_offset  [kv_heads][n_chunks*block][D/vgroup]
//   residual window: res_k, res_v FP32 [kv_heads][n_res][D] for tokens n_chunks*block .. T-1
// value = offset + scale * code, code in 0..15 (unsigned). vgroup divides D and is a
// multiple of 4, so a lane's 4 channels share one V group.
struct Int4Cache {
    const uint8_t *k_codes, *v_codes;
    const float *k_scale, *k_offset, *v_scale, *v_offset;
    const float *res_k, *res_v;
    int n_chunks, block, vgroup, n_res;
    __host__ __device__ int prefix() const { return n_chunks * block; } // int4 tokens
    __host__ __device__ int tokens() const { return prefix() + n_res; }

    // R1. Write channels [4*lane, 4*lane+4) of int4 token t (0 <= t < prefix()) of KV
    // head kh, dequantized to FP32, into out[0..3]; is_v selects V, otherwise K. Read only
    // what this lane needs (one 16-bit code word plus its metadata); no shared memory.
    // Keep it branch-free: a first version that branched on prefix/residual here ran
    // about 1.8x slower at 32K tokens (likely the loads of consecutive tokens stopped
    // being issued back to back).
    __device__ void load4(int kh, bool is_v, int t, int lane, float out[4]) const {
        // BEGIN TODO R1
        asm("trap;");
        // END TODO R1
    }
    // Provided: exact FP32 residual token t (prefix() <= t < tokens()).
    __device__ void load4_tail(int kh, bool is_v, int t, int lane, float out[4]) const {
        const float *row = (is_v ? res_v : res_k) +
                           ((size_t)kh * n_res + (t - prefix())) * kHeadDim + kLaneDims * lane;
        float4 x = *reinterpret_cast<const float4 *>(row);
        out[0] = x.x, out[1] = x.y, out[2] = x.z, out[3] = x.w;
    }
};

// Dense FP16 cache [kv_heads][tokens][D]: the "KV stored in FP16" baseline and the second
// kernel of the dequantize-then-attend baseline. Provided.
struct Fp16Cache {
    const __half *k, *v;
    int n_tokens;
    __host__ __device__ int prefix() const { return n_tokens; } // no residual window
    __host__ __device__ int tokens() const { return n_tokens; }
    __device__ void load4_tail(int kh, bool is_v, int t, int lane, float out[4]) const {
        load4(kh, is_v, t, lane, out);
    }
    __device__ void load4(int kh, bool is_v, int t, int lane, float out[4]) const {
        const __half2 *p = reinterpret_cast<const __half2 *>(
            (is_v ? v : k) + ((size_t)kh * n_tokens + t) * kHeadDim + kLaneDims * lane);
        float2 a = __half22float2(p[0]), b = __half22float2(p[1]);
        out[0] = a.x, out[1] = a.y, out[2] = b.x, out[3] = b.y;
    }
};

__device__ inline float warp_sum(float x) {
    for (int offset = kWarp / 2; offset; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}

// R2. Launch: grid (ceil(n_splits / kWarpsPerBlock), kv_heads), block kWarpsPerBlock*32.
// Warp w of block x handles split s = x*kWarpsPerBlock + w of KV head kh = blockIdx.y and
// every query head h = kh*G + j (j < G) of its GQA group. q is FP32 [q_heads][D].
// For each such h, with score_t = q_h . k_t / sqrt(D) over the split's tokens:
//   part_m[h*S + s]               = max_t score_t
//   part_l[h*S + s]               = sum_t exp(score_t - m)
//   part_acc[(h*S + s)*D + c]     = sum_t exp(score_t - m) * v_t[c]      (not normalized)
// where S = n_splits. Tokens t < cache.prefix() come from cache.load4 (R1), the rest from
// cache.load4_tail; a split may straddle that boundary. Use an online (running-max)
// update and read each K/V row once.
template <int G, class Cache>
__global__ void split_partial(Cache cache, DecodeShape shape, const float *__restrict__ q,
                              float *__restrict__ part_m, float *__restrict__ part_l,
                              float *__restrict__ part_acc) {
    const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
    const int split = blockIdx.x * kWarpsPerBlock + warp, kh = blockIdx.y;
    const int S = shape.n_splits();
    if (split >= S) return;
    // BEGIN TODO R2
    (void)lane, (void)kh;
    asm("trap;");
    // END TODO R2
}

// R3. Launch: grid q_heads, block kHeadDim threads (thread c owns channel c).
// Merge the S partial states of query head h = blockIdx.x into the normalized output
//   out[h*D + c] = sum_s e^{m_s - M} acc_s[c] / sum_s e^{m_s - M} l_s,  M = max_s m_s.
// A split with m_s = -inf and l_s = 0 contributes nothing. M must be subtracted: the
// partial maxima of different splits can differ by far more than float's exp range.
__global__ void reduce_splits(int n_splits, const float *__restrict__ part_m,
                              const float *__restrict__ part_l, const float *__restrict__ part_acc,
                              float *__restrict__ out) {
    const int h = blockIdx.x, c = threadIdx.x;
    // BEGIN TODO R3
    (void)h, (void)c;
    asm("trap;");
    // END TODO R3
}

// Provided: first kernel of the dequantize-then-attend baseline. Writes the whole int4
// cache (prefix + residual) as dense FP16 [kv_heads][tokens][D] using the same R1 decoder,
// so the baseline and the fused kernel differ only in where dequantization happens.
// Launch: grid (ceil(tokens / kWarpsPerBlock), kv_heads), block kWarpsPerBlock*32.
__global__ void dequant_to_fp16(Int4Cache cache, __half *k, __half *v) {
    const int lane = threadIdx.x % kWarp;
    const int t = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarp, kh = blockIdx.y;
    if (t >= cache.tokens()) return;
    for (int is_v = 0; is_v < 2; ++is_v) {
        float x[4];
        if (t < cache.prefix())
            cache.load4(kh, is_v, t, lane, x);
        else
            cache.load4_tail(kh, is_v, t, lane, x);
        __half2 *dst = reinterpret_cast<__half2 *>(
            (is_v ? v : k) + ((size_t)kh * cache.tokens() + t) * kHeadDim + kLaneDims * lane);
        dst[0] = __floats2half2_rn(x[0], x[1]);
        dst[1] = __floats2half2_rn(x[2], x[3]);
    }
}

// Provided host launcher: R2 then R3 on the default stream. part buffers hold
// q_heads * n_splits * (D + 2) floats. G is fixed by the template (Qwen3: 2).
template <int G, class Cache>
inline void decode_attention(const Cache &cache, const DecodeShape &shape, const float *q,
                             float *part_m, float *part_l, float *part_acc, float *out,
                             cudaStream_t stream = 0) {
    const int S = shape.n_splits();
    dim3 grid((S + kWarpsPerBlock - 1) / kWarpsPerBlock, shape.kv_heads);
    split_partial<G>
        <<<grid, kWarpsPerBlock * kWarp, 0, stream>>>(cache, shape, q, part_m, part_l, part_acc);
    reduce_splits<<<shape.q_heads, kHeadDim, 0, stream>>>(S, part_m, part_l, part_acc, out);
}
} // namespace kvdec
