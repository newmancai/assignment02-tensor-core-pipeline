#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"
namespace cg = cooperative_groups;

constexpr int M = 256, N = 64, K = 64;

__host__ __device__ inline int swz128(int row, int colByte) {
    int atom = row >> 3, r = row & 7, chunk = colByte >> 4, in16 = colByte & 15;
    return atom * 1024 + r * 128 + ((chunk ^ r) << 4) + in16;
}

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
    return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar), "r"(phase));
}

template <int GROUP>  // 1 闂?2
__global__ void tile_kernel(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                            float* gD) {
    constexpr int BROWS = GROUP == 1 ? N : N / 2;  // B rows staged by each CTA
    __shared__ __align__(1024) uint8_t sA[128 * K * 2];
    __shared__ __align__(1024) uint8_t sB[BROWS * K * 2];
    __shared__ __align__(8) uint64_t mbar;
    __shared__ uint32_t s_taddr[1];
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    uint32_t mbar_u32 = (uint32_t)__cvta_generic_to_shared(&mbar);
    int rank = GROUP == 1 ? blockIdx.x : cg::this_cluster().block_rank();

    if (warp == 0 && lane == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_u32),
                     "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    // Group 1 allocates independently; group 2 allocates cooperatively.
    // Matching warps in both CTAs must issue the pair operation.
        uint32_t dst = (uint32_t)__cvta_generic_to_shared(s_taddr);
    if (warp == 0) {
        if constexpr (GROUP == 1) {
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
                "[%0], %1;" ::"r"(dst), "r"(64));
            asm volatile(
                "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        } else {
            asm volatile(
                "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 "
                "[%0], %1;" ::"r"(dst), "r"(64));
            asm volatile(
                "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
        }
    }

    for (int i = tid; i < 128 * K; i += blockDim.x) {
        int r = i / K, k = i % K;
        *reinterpret_cast<__nv_bfloat16*>(&sA[swz128(r, k * 2)]) =
            gA[(size_t)(rank * 128 + r) * K + k];
    }
    for (int i = tid; i < BROWS * K; i += blockDim.x) {
        int n = i / K, k = i % K;
        int gn = GROUP == 1 ? n : rank * BROWS + n;
        *reinterpret_cast<__nv_bfloat16*>(&sB[swz128(n, k * 2)]) =
            gB[(size_t)gn * K + k];
    }
    asm volatile("fence.proxy.async.shared::cta;");
    if constexpr (GROUP == 1)
        __syncthreads();
    else
        cg::this_cluster().sync();
    uint32_t taddr = s_taddr[0];

    uint32_t elected;
    asm volatile(
        "{\n.reg .pred P;\nelect.sync _|P, 0xFFFFFFFF;\nselp.b32 %0, 1, 0, "
        "P;\n}"
        : "=r"(elected));
    if (warp == 0 && elected && (GROUP == 1 || rank == 0)) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        uint32_t aBase = (uint32_t)__cvta_generic_to_shared(sA);
        uint32_t bBase = (uint32_t)__cvta_generic_to_shared(sB);
        // idesc:M= GROUP==1 ? 128 : 256
        uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | (8u << 17) |
                         ((GROUP == 1 ? 8u : 16u) << 24);
        for (int kk = 0; kk < K; kk += 16) {
            uint64_t da = make_desc_sm100(aBase + kk * 2, 0, 1024, 2);
            uint64_t db = make_desc_sm100(bBase + kk * 2, 0, 1024, 2);
            uint32_t accum = kk > 0 ? 1u : 0u;
            if constexpr (GROUP == 1) {
                asm volatile(
                    "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
                    "p;\n}\n" ::"r"(taddr),
                    "l"(da), "l"(db), "r"(idesc), "r"(accum));
            } else {
                asm volatile(
                    "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, "
                    "p;\n}\n" ::"r"(taddr),
                    "l"(da), "l"(db), "r"(idesc), "r"(accum));
            }
        }
        if constexpr (GROUP == 1) {
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                ".shared::cluster.b64 [%0];" ::"r"(mbar_u32)
                : "memory");
        } else {
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                ".shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(
                    mbar_u32),
                "h"((uint16_t)0x3)
                : "memory");
        }
    }
    mbar_wait(mbar_u32, 0);

    // Each CTA reads its 128 TMEM rows and writes them to global memory.
    asm volatile("tcgen05.fence::after_thread_sync;");
    for (int c = 0; c < N; c += 8) {
        uint32_t src = taddr + ((uint32_t)(warp * 32) << 16) + c;
        float r[8];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]),
              "=f"(r[5]), "=f"(r[6]), "=f"(r[7])
            : "r"(src));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        int row = rank * 128 + warp * 32 + lane;
#pragma unroll
        for (int i = 0; i < 8; i++) gD[(size_t)row * N + c + i] = r[i];
    }

    if constexpr (GROUP == 1) {
        __syncthreads();
        if (warp == 0)
            asm volatile(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(
                    taddr),
                "r"(64));
    } else {
        cg::this_cluster().sync();  // Both CTAs finish TMEM reads before release.
        if (warp == 0)
            asm volatile(
                "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(
                    taddr),
                "r"(64));
    }
}

int main() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(M * K), hB(N * K);
    std::vector<float> ref(M * N, 0.f);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                ref[m * N + n] += __bfloat162float(hA[m * K + k]) *
                                  __bfloat162float(hB[n * K + k]);
    __nv_bfloat16 *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, N * K * 2));
    CUDA_CHECK(cudaMalloc(&dD, M * N * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K * 2, cudaMemcpyHostToDevice));

    cudaFuncAttributes at1, at2;
    CUDA_CHECK(cudaFuncGetAttributes(&at1, tile_kernel<1>));
    CUDA_CHECK(cudaFuncGetAttributes(&at2, tile_kernel<2>));
    printf("smem/block: cta_group::1 = %zu B, cta_group::2 = %zu B\n",
           at1.sharedSizeBytes, at2.sharedSizeBytes);

    std::vector<float> got(M * N);
    tile_kernel<1><<<2, 128>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(got.data(), dD, M * N * 4, cudaMemcpyDeviceToHost));
    long bad1 = 0;
    for (int i = 0; i < M * N; i++) bad1 += got[i] != ref[i];

    // 闂備礁鎲￠悷锔炬箒缂?B:::2,濠电偞鍨堕幐鎾磻閹炬枼妲?cluster(2 CTA)
    CUDA_CHECK(cudaMemset(dD, 0, M * N * 4));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(2);
    cfg.blockDim = dim3(128);
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, tile_kernel<2>, (const __nv_bfloat16*)dA,
                                  (const __nv_bfloat16*)dB, (float*)dD));
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(got.data(), dD, M * N * 4, cudaMemcpyDeviceToHost));
    long bad2 = 0;
    for (int i = 0; i < M * N; i++) bad2 += got[i] != ref[i];

    float t1 = time_avg_ms([&] { tile_kernel<1><<<2, 128>>>(dA, dB, dD); }, 200);
    float t2 = time_avg_ms(
        [&] {
            cudaLaunchKernelEx(&cfg, tile_kernel<2>, (const __nv_bfloat16*)dA,
                               (const __nv_bfloat16*)dB, (float*)dD);
        },
        200);
    printf("::1  %s(bad=%ld)  %.2f us\n", bad1 ? "FAIL" : "PASS", bad1,
           t1 * 1e3);
    printf("::2  %s(bad=%ld)  %.2f us\n", bad2 ? "FAIL" : "PASS", bad2,
           t2 * 1e3);
    return (bad1 || bad2) != 0;
}
