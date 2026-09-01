// 问题 5.3(b) 的判测与计时程序,不需要修改。实现写在
// nvfp4_quant_kernel.h。先过 03a,再做这题(host 参考用你的编码器)。
//
// 判测口径:quant 的每一步在 host 和 device 上是同样的 float 运算和
// 同样的顺序,结果逐 byte 严格相等,没有 tolerance。
// 计时输出有效带宽:读 2 B/elem,写 0.5 B/elem 数据 + 1/16 B/elem SF。
#include <vector>
#include <random>
#include "../common.h"
#include "nvfp4_common.h"
#include "e2m1_encode.h"
#include "nvfp4_quant_kernel.h"

static void host_ref(const std::vector<float>& x, int M, int K,
                     std::vector<uint8_t>& data, std::vector<uint8_t>& sf) {
    int numKTiles = nvfp4_num_ktiles(K);
    data.assign((size_t)M * K / 2, 0);
    sf.assign((size_t)nvfp4_sf_bytes(M, K), 0);
    for (int r = 0; r < M; r++)
        for (int g = 0; g < K / NVFP4_GROUP; g++) {
            float amax = 0.f;
            for (int i = 0; i < NVFP4_GROUP; i++)
                amax = fmaxf(amax, fabsf(x[(size_t)r * K + g * 16 + i]));
            __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(amax / 6.0f);
            float s = float(sf8);
            float inv = s != 0.f ? 1.0f / s : 0.f;
            sf[sf_swizzled_offset(r, g, numKTiles)] = *(uint8_t*)&sf8;
            for (int i = 0; i < NVFP4_GROUP; i += 2) {
                uint8_t lo = e2m1_encode(x[(size_t)r * K + g * 16 + i] * inv);
                uint8_t hi = e2m1_encode(x[(size_t)r * K + g * 16 + i + 1] * inv);
                data[(size_t)r * K / 2 + g * 8 + i / 2] = (uint8_t)(hi << 4 | lo);
            }
        }
}

int main() {
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    long total_bad = 0;
    for (const auto& shape :
         {std::pair{128, 1024}, {200, 4096}, {4096, 7168}}) {
        int M = shape.first;
        int K = shape.second;
        size_t n = (size_t)M * K;
        int64_t sfB = nvfp4_sf_bytes(M, K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-4.f, 4.f);
        std::vector<__nv_bfloat16> hx(n);
        std::vector<float> hxf(n);
        for (size_t i = 0; i < n; i++) {
            hx[i] = __float2bfloat16(dist(rng));
            hxf[i] = __bfloat162float(hx[i]);
        }
        __nv_bfloat16* dx;
        uint8_t *dd, *dsf;
        CUDA_CHECK(cudaMalloc(&dx, n * 2));
        CUDA_CHECK(cudaMalloc(&dd, n / 2));
        CUDA_CHECK(cudaMalloc(&dsf, sfB));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dsf, 0, sfB));

        launch_nvfp4_quant(dx, dd, dsf, M, K, sms);
        CUDA_CHECK_KERNEL();

        std::vector<uint8_t> gd(n / 2), gsf(sfB), rd, rsf;
        CUDA_CHECK(cudaMemcpy(gd.data(), dd, n / 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gsf.data(), dsf, sfB, cudaMemcpyDeviceToHost));
        host_ref(hxf, M, K, rd, rsf);
        long bad = 0;
        for (size_t i = 0; i < gd.size(); i++) bad += gd[i] != rd[i];
        for (size_t i = 0; i < gsf.size(); i++) bad += gsf[i] != rsf[i];

        float ms = time_avg_ms(
            [&] { launch_nvfp4_quant(dx, dd, dsf, M, K, sms); },
            M >= 4096 ? 50 : 200);
        double bytes = n * 2.0 + n * 0.5 + n / 16.0;
        printf("M=%-5d K=%-5d  %s(bad=%ld)  %8.2f us  %6.0f GB/s\n", M, K,
               bad ? "FAIL" : "PASS", bad, ms * 1e3,
               effective_gbps(bytes, ms));
        total_bad += bad;
        cudaFree(dx); cudaFree(dd); cudaFree(dsf);
    }
    return total_bad != 0;
}
