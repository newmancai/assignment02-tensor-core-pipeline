// 问题 5.3(b) 的终验:消费端测试,不需要修改。
//
// 把你的 quant kernel 的产物(e2m1 数据 + swizzled SF)原样喂给
// cuBLASLt 的 FP4 matmul(CUDA_R_4F_E2M1 + SCALE_VEC16_UE4M3,
// 走 block-scaled tensor core),对拍 host 反量化的 double GEMM。
// 布局正确时 maxrel 应当在 4e-3 附近(bf16 输出的舍入,2^-8);
// 布局有任何错位,scale 会配错组,结果整块崩掉。
//
// 构建:make bin/m5_lowprec/test_fp4_gemm(Makefile 已配 -lcublasLt)。
#include <cublasLt.h>
#include <vector>
#include <random>
#include "../common.h"
#include "nvfp4_common.h"
#include "nvfp4_quant_kernel.h"

#define LT_CHECK(x)                                                       \
    do {                                                                  \
        cublasStatus_t s_ = (x);                                          \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cublasLt error %d at %s:%d\n", (int)s_,      \
                    __FILE__, __LINE__);                                  \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

static void dequant_host(const std::vector<uint8_t>& data,
                         const std::vector<uint8_t>& sf, int M, int K,
                         std::vector<double>& out) {
    int numKTiles = nvfp4_num_ktiles(K);
    out.assign((size_t)M * K, 0.0);
    for (int r = 0; r < M; r++)
        for (int g = 0; g < K / 16; g++) {
            uint8_t sfb = sf[sf_swizzled_offset(r, g, numKTiles)];
            float sfv = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&sfb));
            for (int i = 0; i < 8; i++) {
                uint8_t b = data[(size_t)r * K / 2 + g * 8 + i];
                out[(size_t)r * K + g * 16 + i * 2] = e2m1_decode(b & 0xF) * sfv;
                out[(size_t)r * K + g * 16 + i * 2 + 1] =
                    e2m1_decode(b >> 4) * sfv;
            }
        }
}

static int run_case(cublasLtHandle_t lt, int M, int N, int K, int sms) {
    size_t nA = (size_t)M * K, nB = (size_t)N * K;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    std::vector<__nv_bfloat16> hA(nA), hB(nB);
    for (auto& v : hA) v = __float2bfloat16(dist(rng));
    for (auto& v : hB) v = __float2bfloat16(dist(rng) * 0.3f);

    __nv_bfloat16 *dA, *dB, *dD;
    uint8_t *dAq, *dBq, *dAsf, *dBsf;
    int64_t sfA = nvfp4_sf_bytes(M, K), sfB = nvfp4_sf_bytes(N, K);
    CUDA_CHECK(cudaMalloc(&dA, nA * 2));
    CUDA_CHECK(cudaMalloc(&dB, nB * 2));
    CUDA_CHECK(cudaMalloc(&dD, (size_t)M * N * 2));
    CUDA_CHECK(cudaMalloc(&dAq, nA / 2));
    CUDA_CHECK(cudaMalloc(&dBq, nB / 2));
    CUDA_CHECK(cudaMalloc(&dAsf, sfA));
    CUDA_CHECK(cudaMalloc(&dBsf, sfB));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dAsf, 0, sfA));
    CUDA_CHECK(cudaMemset(dBsf, 0, sfB));

    launch_nvfp4_quant(dA, dAq, dAsf, M, K, sms);
    launch_nvfp4_quant(dB, dBq, dBsf, N, K, sms);
    CUDA_CHECK_KERNEL();

    cublasLtMatmulDesc_t op;
    LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &ta, sizeof(ta)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
                                            &tb, sizeof(tb)));
    cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    LT_CHECK(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dAsf, sizeof(dAsf)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dBsf, sizeof(dBsf)));

    cublasLtMatrixLayout_t la, lb, ld;
    LT_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, K, M, K));
    LT_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, K, N, K));
    LT_CHECK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, M, N, M));

    cublasLtMatmulPreference_t pref;
    LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = 64ull << 20;
    void* dws;
    CUDA_CHECK(cudaMalloc(&dws, ws));
    LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    cublasLtMatmulHeuristicResult_t heur;
    int found = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 1,
                                            &heur, &found));
    if (!found) {
        printf("M=%d N=%d K=%d: no cublasLt algo found\n", M, N, K);
        return 1;
    }
    float alpha = 1.f, beta = 0.f;
    LT_CHECK(cublasLtMatmul(lt, op, &alpha, dAq, la, dBq, lb, &beta, dD, ld,
                            dD, ld, &heur.algo, dws, ws, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> hAq(nA / 2), hBq(nB / 2), hAsf(sfA), hBsf(sfB);
    CUDA_CHECK(cudaMemcpy(hAq.data(), dAq, nA / 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hBq.data(), dBq, nB / 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hAsf.data(), dAsf, sfA, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hBsf.data(), dBsf, sfB, cudaMemcpyDeviceToHost));
    std::vector<double> dqA, dqB;
    dequant_host(hAq, hAsf, M, K, dqA);
    dequant_host(hBq, hBsf, N, K, dqB);
    std::vector<__nv_bfloat16> hD((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(hD.data(), dD, (size_t)M * N * 2,
                          cudaMemcpyDeviceToHost));

    double maxrel = 0;
    for (int nn = 0; nn < N; nn++)
        for (int m = 0; m < M; m++) {
            double ref = 0;
            for (int k = 0; k < K; k++)
                ref += dqA[(size_t)m * K + k] * dqB[(size_t)nn * K + k];
            double got = __bfloat162float(hD[(size_t)nn * M + m]);
            if (fabs(ref) > 1.0)
                maxrel = fmax(maxrel, fabs(got - ref) / fabs(ref));
        }
    bool pass = maxrel < 2e-2;
    printf("M=%-5d N=%-5d K=%-5d  maxrel=%.3e  %s\n", M, N, K, maxrel,
           pass ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dD); cudaFree(dAq); cudaFree(dBq);
    cudaFree(dAsf); cudaFree(dBsf); cudaFree(dws);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(op);
    return pass ? 0 : 1;
}

int main() {
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    cublasLtHandle_t lt;
    LT_CHECK(cublasLtCreate(&lt));
    int rc = 0;
    rc |= run_case(lt, 128, 128, 1024, sms);
    rc |= run_case(lt, 256, 512, 4096, sms);
    rc |= run_case(lt, 200, 128, 1024, sms);  // M 非 128 倍数:SF padding 路径
    cublasLtDestroy(lt);
    return rc;
}
