// 问题 3.3:修 bug。多轮 mma + 逐轮回读的结构(模拟 M4 的 K 流式):
// 每轮一条 k16 mma(D = 本轮部分积)-> commit -> 等待 -> 各 warp 读回
// 自己的 lane 在寄存器累加。./03_bug_mbarrier <seed> <rounds>。
//
// 这个程序有 bug。提交时回答两问(先跑,后改):
// (a) 记录现象与复现条件:rounds 取 1、2、4 各什么表现?
// (b) 画出正确与错误版本的 mbarrier 状态机(phase、arrival count 的
//     变化),指出错误版本的等待放行时刻错在哪,以及为什么会是你在
//     (a) 观察到的现象(提示:tcgen05.ld 与在飞 mma 的关系)。
//
// 判测:../m3_tcgen05/judge_mbar.sh(带超时——程序可能挂死)。
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include "../common.h"

constexpr int M = 128, N = 64, K = 64;

// 128B swizzle 的物理偏移(即 2.3 的 swizzle_128B;row 是 K-major 下的
// 行 = M 或 N 维,col 是 K 维字节)。atom = 8 行 × 128B,SBO=1024。
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
    d |= (uint64_t)1 << 46;             // version = 1(SM100)
    d |= (uint64_t)layout << 61;        // 3 bit layout type
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

__global__ void tcgen05_tile(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                             float* gD, int rounds) {
    __shared__ __align__(1024) uint8_t sA[M * K * 2];   // 16 KB,swizzled
    __shared__ __align__(1024) uint8_t sB[N * K * 2];   // 8 KB,swizzled
    __shared__ __align__(8) uint64_t mbar;
    __shared__ uint32_t s_taddr[1];
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    uint32_t mbar_u32 = (uint32_t)__cvta_generic_to_shared(&mbar);

    // (1) mbarrier 初始化 + TMEM 分配
    if (warp == 0) {
        if (lane == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                             mbar_u32),
                         "r"(1));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        uint32_t dst = (uint32_t)__cvta_generic_to_shared(s_taddr);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
                "r"(dst),
            "r"(64));
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    // (2) 按 swizzled 布局写 A/B(生产环境由 TMA 完成)
    for (int i = tid; i < M * K; i += blockDim.x) {
        int r = i / K, k = i % K;
        *reinterpret_cast<__nv_bfloat16*>(&sA[swz128(r, k * 2)]) = gA[i];
    }
    for (int i = tid; i < N * K; i += blockDim.x) {
        int n = i / K, k = i % K;
        *reinterpret_cast<__nv_bfloat16*>(&sB[swz128(n, k * 2)]) = gB[i];
    }

    // (3) generic 写对 async proxy 可见
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    uint32_t taddr = s_taddr[0];

    // 每轮:一条 k16 mma(不累加,D = 本轮部分积)-> commit -> wait ->
    // 各 warp 读回自己的 lane,在寄存器里累加。ROUNDS 由 host 传入。
    uint32_t elected;
    asm volatile(
        "{\n.reg .pred P;\nelect.sync _|P, 0xFFFFFFFF;\nselp.b32 %0, 1, 0, "
        "P;\n}"
        : "=r"(elected));
    uint32_t aBase = (uint32_t)__cvta_generic_to_shared(sA);
    uint32_t bBase = (uint32_t)__cvta_generic_to_shared(sB);
    uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | (8u << 17) |
                     (8u << 24);
    float acc[8][8] = {};  // [N/8 段][8 列]
    for (int round = 0; round < rounds; round++) {
        int kk = round * 16;
        if (warp == 0 && elected) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            uint64_t da = make_desc_sm100(aBase + kk * 2, 0, 1024, 2);
            uint64_t db = make_desc_sm100(bBase + kk * 2, 0, 1024, 2);
            asm volatile(
                "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::
                    "r"(taddr),
                "l"(da), "l"(db), "r"(idesc), "r"(0));
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                ".shared::cluster.b64 [%0];" ::"r"(mbar_u32)
                : "memory");
        }
        mbar_wait(mbar_u32, 0);
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
#pragma unroll
            for (int i = 0; i < 8; i++) acc[c / 8][i] += r[i];
        }
        __syncthreads();  // 所有 warp 读完本轮才允许发下一轮
    }
    {
        int row = warp * 32 + lane;
        for (int c = 0; c < N; c += 8)
#pragma unroll
            for (int i = 0; i < 8; i++) gD[row * N + c + i] = acc[c / 8][i];
    }

    // (7) 全部读完后释放 TMEM
    __syncthreads();
    if (warp == 0)
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(
                taddr),
            "r"(64));
}

int main(int argc, char** argv) {
    unsigned seed = argc > 1 ? (unsigned)atoi(argv[1]) : 42;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(M * K), hB(N * K);
    std::vector<float> ref(M * N, 0.f);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    int ref_rounds = argc > 2 ? atoi(argv[2]) : 4;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < ref_rounds * 16; k++)
                ref[m * N + n] += __bfloat162float(hA[m * K + k]) *
                                  __bfloat162float(hB[n * K + k]);
    __nv_bfloat16 *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, N * K * 2));
    CUDA_CHECK(cudaMalloc(&dD, M * N * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K * 2, cudaMemcpyHostToDevice));
    int rounds = argc > 2 ? atoi(argv[2]) : 4;
    tcgen05_tile<<<1, 128>>>(dA, dB, dD, rounds);
    CUDA_CHECK_KERNEL();
    std::vector<float> got(M * N);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, M * N * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < M * N; i++)
        if (got[i] != ref[i]) {
            if (bad < 5)
                printf("MISMATCH D[%d][%d]: got %.1f want %.1f\n", i / N,
                       i % N, got[i], ref[i]);
            bad++;
        }
    printf(bad ? "FAIL seed=%u: %ld / %d\n" : "PASS seed=%u\n", seed,
           bad ? bad : (long)seed, M * N);
    return bad != 0;
}
