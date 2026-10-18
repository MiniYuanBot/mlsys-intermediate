// Strong baselines for gpu_bench.cu (provided, not a TODO). All produce FP32
// Y[batch, rows] = X[batch, cols] W^T with W stored row-major [rows, cols]. In cuBLAS's
// column-major view that W is a K x M matrix with ld = K, so every call is "TN":
// D(M x N) = op(A)=W (transposed) * B = X^T (not transposed), M = rows, N = batch, K = cols.
//   sgemm      FP32 W, FP32 X, SIMT (cublasSgemm)              resident FP32 reference
//   hgemm      FP16 W, FP16 X, FP32 out, tensor cores (GemmEx)  strong dense baseline
//   unpack     INT4 lab layout -> FP16 W (then hgemm)           "dequantize, then GEMM"
//   fp8_lt     E4M3 W, E4M3 X, per-tensor scales, FP32 out (cublasLtMatmul, sm_89+),
//              W8A8: the activations are quantized too, unlike every other path.
#pragma once
#include <cstdint>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdexcept>
#include <string>

#define CUDA_OK(x)                                                                                 \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess)                                                                     \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_));              \
    } while (0)
#define CUBLAS_OK(x)                                                                               \
    do {                                                                                           \
        cublasStatus_t s_ = (x);                                                                   \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                           \
            throw std::runtime_error(std::string("cuBLAS status ") + std::to_string(int(s_)) +     \
                                     " at " #x);                                                   \
    } while (0)

namespace base {
// INT4 lab layout -> FP16 dense W; one thread per output element.
__global__ void unpack_int4_to_half(const uint8_t *codes, const float *scales, __half *w, int rows,
                                    int cols, int group) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(rows) * cols) return;
    int r = int(i / cols), c = int(i % cols);
    int stride = (cols + 1) / 2, ng = (cols + group - 1) / group;
    int u = (codes[size_t(r) * stride + c / 2] >> (4 * (c & 1))) & 15;
    w[i] = __float2half(float(u - 8) * scales[size_t(r) * ng + c / group]);
}
inline void unpack(const uint8_t *codes, const float *scales, __half *w, int rows, int cols,
                   int group) {
    size_t n = size_t(rows) * cols;
    unpack_int4_to_half<<<unsigned((n + 255) / 256), 256>>>(codes, scales, w, rows, cols, group);
}

struct Blas {
    cublasHandle_t h{};
    cublasLtHandle_t lt{};
    void *workspace = nullptr;
    size_t workspace_bytes = size_t(32) << 20;
    Blas() {
        CUBLAS_OK(cublasCreate(&h));
        CUBLAS_OK(cublasLtCreate(&lt));
        CUDA_OK(cudaMalloc(&workspace, workspace_bytes));
    }
    ~Blas() {
        cudaFree(workspace);
        cublasLtDestroy(lt);
        cublasDestroy(h);
    }
    void sgemm(const float *w, const float *x, float *y, int rows, int cols, int batch) const {
        const float one = 1, zero = 0;
        CUBLAS_OK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, rows, batch, cols, &one, w, cols, x,
                              cols, &zero, y, rows));
    }
    void hgemm(const __half *w, const __half *x, float *y, int rows, int cols, int batch) const {
        const float one = 1, zero = 0;
        CUBLAS_OK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, rows, batch, cols, &one, w, CUDA_R_16F,
                               cols, x, CUDA_R_16F, cols, &zero, y, CUDA_R_32F, rows,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
};

// FP8 W8A8 matmul through cuBLASLt. Requirements checked by cuBLASLt on Ada (cc 8.9):
// A transposed and B not ("TN"), CUBLAS_COMPUTE_32F with CUDA_R_32F scale type,
// 16-byte aligned pointers and leading dimensions. Scales are per tensor: on sm_89
// cuBLASLt has no per-group/block scaling for FP8 (block-scaled formats need sm_100).
struct Fp8Lt {
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t a{}, b{}, c{};
    cublasLtMatmulHeuristicResult_t heuristic{};
    const Blas *blas;
    bool ok = false;
    std::string why;
    Fp8Lt(const Blas &bl, int rows, int cols, int batch, const float *a_scale, const float *b_scale)
        : blas(&bl) {
        cublasOperation_t t = CUBLAS_OP_T, n = CUBLAS_OP_N;
        CUBLAS_OK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)));
        CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &n, sizeof(n)));
        CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale,
                                                 sizeof(a_scale)));
        CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale,
                                                 sizeof(b_scale)));
        CUBLAS_OK(cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, cols, rows, cols)); // W as K x M
        CUBLAS_OK(
            cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, cols, batch, cols));   // X^T as K x N
        CUBLAS_OK(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, rows, batch, rows)); // Y^T as M x N
        cublasLtMatmulPreference_t pref{};
        CUBLAS_OK(cublasLtMatmulPreferenceCreate(&pref));
        CUBLAS_OK(
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &bl.workspace_bytes, sizeof(bl.workspace_bytes)));
        int found = 0;
        cublasStatus_t s =
            cublasLtMatmulAlgoGetHeuristic(bl.lt, op, a, b, c, c, pref, 1, &heuristic, &found);
        cublasLtMatmulPreferenceDestroy(pref);
        ok = s == CUBLAS_STATUS_SUCCESS && found > 0;
        if (!ok) why = "no cuBLASLt FP8 algorithm (status " + std::to_string(int(s)) + ")";
    }
    ~Fp8Lt() {
        cublasLtMatrixLayoutDestroy(a), cublasLtMatrixLayoutDestroy(b),
            cublasLtMatrixLayoutDestroy(c);
        cublasLtMatmulDescDestroy(op);
    }
    void run(const uint8_t *w, const uint8_t *x, float *y) const {
        const float one = 1, zero = 0;
        CUBLAS_OK(cublasLtMatmul(blas->lt, op, &one, w, a, x, b, &zero, y, c, y, c, &heuristic.algo,
                                 blas->workspace, blas->workspace_bytes, 0));
    }
};
} // namespace base
