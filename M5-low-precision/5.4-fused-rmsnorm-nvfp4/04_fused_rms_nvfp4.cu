// 问题 5.4(全作业结束题):融合 rms_norm + NVFP4 quant。
//
// 背景见题面(vLLM issue #25179 / PR #36413):这个融合上游有两个未
// 合入的 PR,卡在"端到端收益在噪声内,没人解释清楚收益去了哪"。
// 你要做的是把 kernel 写出来,并给出那份解释。
//
// 语义:y = rms_norm(x) * w 后直接量化为 NVFP4(不落 bf16 中间值)。
//   rnorm = 1 / sqrt(mean(x_i^2) + eps),后续每组与 5.3(b) 相同
// 字节账:两步(rms 写读中间值)6.56 B/elem,融合 2.56 B/elem,
// 预言加速比 2.56x。你的任务:逐形状实测,并解释实测与预言的差距
// ——每个 M 段的限制因素是什么,证据(ncu 或推算)是什么。
//
// 本文件给出:两步基线(下面的 rms_norm_baseline_kernel + 你 5.3(b)
// 的 quant kernel)、判测、逐形状计时框架。三处要你动手:
//   1. 实现融合 kernel(结构完全自由)
//   2. 基线要公平:baseline 与融合各自把 grid 等配置调到最优再对比
//      (基线吃亏的对比没有意义,上游 PR 的教训之一)
//   3. 报告:逐形状表 + 差距归因
//
// 判测口径:sumsq 归约顺序不同会让极少数处在舍入边界的值翻转,
// 允许 1e-4 比例的 byte 不一致(host 参考的 sumsq 用 double)。
#include <vector>
#include <random>
#include "../common.h"
#include "nvfp4_common.h"
#include "e2m1_encode.h"
#include "nvfp4_quant_kernel.h"

// 给定的两步基线第一步:block-per-row 的 rms_norm,bf16 进出。
// 允许修改或另写(公平基线的一部分:它调多快,对比就有多可信)。
template <int BLOCK>
__global__ void rms_norm_baseline_kernel(const __nv_bfloat16* __restrict__ in,
                                         const __nv_bfloat16* __restrict__ w,
                                         __nv_bfloat16* __restrict__ out,
                                         int M, int K, float eps) {
    __shared__ float red[BLOCK / 32];
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const __nv_bfloat16* xr = in + (size_t)row * K;
        float ss = 0.f;
        for (int k = threadIdx.x * 8; k < K; k += BLOCK * 8) {
            float4 raw = *reinterpret_cast<const float4*>(xr + k);
            const __nv_bfloat162* h =
                reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 f = __bfloat1622float2(h[i]);
                ss += f.x * f.x + f.y * f.y;
            }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x < 32) {
            ss = threadIdx.x < BLOCK / 32 ? red[threadIdx.x] : 0.f;
#pragma unroll
            for (int o = 16; o; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
            if (threadIdx.x == 0) red[0] = ss;
        }
        __syncthreads();
        float rnorm = 1.0f / sqrtf(red[0] / K + eps);
        for (int k = threadIdx.x * 8; k < K; k += BLOCK * 8) {
            float4 raw = *reinterpret_cast<const float4*>(xr + k);
            float4 raww = *reinterpret_cast<const float4*>(w + k);
            const __nv_bfloat162* h =
                reinterpret_cast<const __nv_bfloat162*>(&raw);
            const __nv_bfloat162* hw =
                reinterpret_cast<const __nv_bfloat162*>(&raww);
            __nv_bfloat162 o2[4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 f = __bfloat1622float2(h[i]);
                float2 fw = __bfloat1622float2(hw[i]);
                o2[i] = __floats2bfloat162_rn(f.x * rnorm * fw.x,
                                              f.y * rnorm * fw.y);
            }
            *reinterpret_cast<float4*>(
                const_cast<__nv_bfloat16*>(out) + (size_t)row * K + k) =
                *reinterpret_cast<float4*>(o2);
        }
        __syncthreads();
    }
}

// TODO(核心):融合 kernel。签名自定,在 launch_fused 里接上。
static void launch_fused(const __nv_bfloat16* in, const __nv_bfloat16* w,
                         uint8_t* dataOut, uint8_t* sfOut, int M, int K,
                         float eps, int sms) {
    // TODO
    (void)in; (void)w; (void)dataOut; (void)sfOut; (void)M; (void)K;
    (void)eps; (void)sms;
}

// TODO(公平基线):两步各自的最优启动配置。默认给的是一个起点。
static void launch_two_step(const __nv_bfloat16* in, const __nv_bfloat16* w,
                            __nv_bfloat16* mid, uint8_t* dataOut,
                            uint8_t* sfOut, int M, int K, float eps,
                            int sms) {
    int grid = M < sms ? M : sms * 2;
    rms_norm_baseline_kernel<512><<<grid, 512>>>(in, w, mid, M, K, eps);
    launch_nvfp4_quant(mid, dataOut, sfOut, M, K, sms);
}

static void host_ref(const std::vector<float>& x, const std::vector<float>& w,
                     int M, int K, float eps, std::vector<uint8_t>& data) {
    int numKTiles = nvfp4_num_ktiles(K);
    (void)numKTiles;
    data.assign((size_t)M * K / 2, 0);
    for (int r = 0; r < M; r++) {
        double ss = 0;
        for (int k = 0; k < K; k++) {
            double v = x[(size_t)r * K + k];
            ss += v * v;
        }
        float rnorm = 1.0f / sqrtf((float)(ss / K) + eps);
        for (int g = 0; g < K / NVFP4_GROUP; g++) {
            float vals[NVFP4_GROUP], amax = 0.f;
            for (int i = 0; i < NVFP4_GROUP; i++) {
                int k = g * NVFP4_GROUP + i;
                vals[i] = x[(size_t)r * K + k] * rnorm * w[k];
                amax = fmaxf(amax, fabsf(vals[i]));
            }
            __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(amax / 6.0f);
            float s = float(sf8);
            float inv = s != 0.f ? 1.0f / s : 0.f;
            for (int i = 0; i < NVFP4_GROUP; i += 2)
                data[(size_t)r * K / 2 + g * 8 + i / 2] =
                    (uint8_t)(e2m1_encode(vals[i + 1] * inv) << 4 |
                              e2m1_encode(vals[i] * inv));
        }
    }
}

int main() {
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const float eps = 1e-6f;
    printf("# %-6s %-6s %10s %10s %8s\n", "M", "K", "2step_us", "fused_us",
           "speedup");
    long total_bad = 0;
    for (const auto& shape :
         {std::pair{1, 4096}, {16, 4096}, {256, 4096}, {1024, 4096},
          {4096, 4096}, {16384, 4096}, {4096, 7168}, {16384, 7168},
          {4096, 8192}, {16384, 8192}}) {
        int M = shape.first;
        int K = shape.second;
        size_t n = (size_t)M * K;
        int64_t sfB = nvfp4_sf_bytes(M, K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-2.f, 2.f);
        std::vector<__nv_bfloat16> hx(n), hw(K);
        std::vector<float> hxf(n), hwf(K);
        for (size_t i = 0; i < n; i++) {
            hx[i] = __float2bfloat16(dist(rng));
            hxf[i] = __bfloat162float(hx[i]);
        }
        for (int i = 0; i < K; i++) {
            hw[i] = __float2bfloat16(dist(rng) * 0.5f);
            hwf[i] = __bfloat162float(hw[i]);
        }
        __nv_bfloat16 *dx, *dw, *dmid;
        uint8_t *dd, *dsf;
        CUDA_CHECK(cudaMalloc(&dx, n * 2));
        CUDA_CHECK(cudaMalloc(&dw, (size_t)K * 2));
        CUDA_CHECK(cudaMalloc(&dmid, n * 2));
        CUDA_CHECK(cudaMalloc(&dd, n / 2));
        CUDA_CHECK(cudaMalloc(&dsf, sfB));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dw, hw.data(), (size_t)K * 2,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dsf, 0, sfB));

        launch_fused(dx, dw, dd, dsf, M, K, eps, sms);
        CUDA_CHECK_KERNEL();
        std::vector<uint8_t> gd(n / 2), rd;
        CUDA_CHECK(cudaMemcpy(gd.data(), dd, n / 2, cudaMemcpyDeviceToHost));
        host_ref(hxf, hwf, M, K, eps, rd);
        long bad = 0;
        for (size_t i = 0; i < gd.size(); i++) bad += gd[i] != rd[i];
        bool pass = bad <= (long)(gd.size() / 10000) + 1;
        total_bad += !pass;

        int iters = M >= 4096 ? 40 : 200;
        float t2 = time_avg_ms(
            [&] { launch_two_step(dx, dw, dmid, dd, dsf, M, K, eps, sms); },
            iters);
        float tf = time_avg_ms(
            [&] { launch_fused(dx, dw, dd, dsf, M, K, eps, sms); }, iters);
        printf("  %-6d %-6d %10.2f %10.2f %7.2fx %s(bad=%ld)\n", M, K,
               t2 * 1e3, tf * 1e3, t2 / tf, pass ? "PASS" : "FAIL", bad);
        cudaFree(dx); cudaFree(dw); cudaFree(dmid); cudaFree(dd);
        cudaFree(dsf);
    }
    return total_bad != 0;
}
