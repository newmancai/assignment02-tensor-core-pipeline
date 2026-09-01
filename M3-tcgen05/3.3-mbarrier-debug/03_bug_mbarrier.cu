#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"

constexpr int M = 128, N = 64, K = 64;

__host__ __device__ inline int swz128(int row, int col_byte) {
    int atom = row >> 3, r = row & 7, chunk = col_byte >> 4;
    return atom * 1024 + r * 128 + ((chunk ^ r) << 4) + (col_byte & 15);
}
__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3fff);
    d |= (uint64_t)((lbo >> 4) & 0x3fff) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3fff) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
    return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done) : "r"(mbar), "r"(phase));
    }
}

__global__ void tcgen05_rounds(const __nv_bfloat16* gA,
                               const __nv_bfloat16* gB,
                               float* gD, int rounds) {
    __shared__ __align__(1024) uint8_t sA[M * K * 2];
    __shared__ __align__(1024) uint8_t sB[N * K * 2];
    __shared__ __align__(8) uint64_t mbar;
    __shared__ uint32_t s_taddr;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint32_t mbar_addr =
        (uint32_t)__cvta_generic_to_shared(&mbar);

    if (warp == 0) {
        if (lane == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                         "r"(mbar_addr), "r"(1));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        uint32_t dst = (uint32_t)__cvta_generic_to_shared(&s_taddr);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
            "[%0], %1;" :: "r"(dst), "r"(64));
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    for (int i = tid; i < M * K; i += blockDim.x) {
        int row = i / K, k = i % K;
        *reinterpret_cast<__nv_bfloat16*>(&sA[swz128(row, k * 2)]) = gA[i];
    }
    for (int i = tid; i < N * K; i += blockDim.x) {
        int row = i / K, k = i % K;
        *reinterpret_cast<__nv_bfloat16*>(&sB[swz128(row, k * 2)]) = gB[i];
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    const uint32_t taddr = s_taddr;
    const uint32_t a_base = (uint32_t)__cvta_generic_to_shared(sA);
    const uint32_t b_base = (uint32_t)__cvta_generic_to_shared(sB);
    const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                           (8u << 17) | (8u << 24);
    uint32_t elected = 0;
    asm volatile(
        "{\n.reg .pred p;\nelect.sync _|p, 0xffffffff;\n"
        "selp.b32 %0, 1, 0, p;\n}" : "=r"(elected));

    float acc[8][8] = {};
    for (int round = 0; round < rounds; ++round) {
        const int kk = round * 16;
        if (warp == 0 && elected) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            uint64_t da = make_desc_sm100(a_base + kk * 2, 0, 1024, 2);
            uint64_t db = make_desc_sm100(b_base + kk * 2, 0, 1024, 2);
            asm volatile(
                "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 "
                "[%0], %1, %2, %3, p;\n}" ::
                "r"(taddr), "l"(da), "l"(db), "r"(idesc), "r"(0));
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                ".shared::cluster.b64 [%0];" :: "r"(mbar_addr) : "memory");
        }

        // Fixed bug: each completed arrival toggles the reusable barrier phase.
        // Waiting on phase 0 every time lets later loads run before their MMA.
#ifdef BUGGY_PHASE
        // Reproduction mode for the original bug: every generation waits on
        // phase zero. The normal build intentionally uses alternating parity.
        mbar_wait(mbar_addr, 0);
#else
        mbar_wait(mbar_addr, (uint32_t)(round & 1));
#endif
        asm volatile("tcgen05.fence::after_thread_sync;");
        for (int c = 0; c < N; c += 8) {
            uint32_t src = taddr + ((uint32_t)(warp * 32) << 16) + c;
            float r[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]),
                  "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7])
                : "r"(src));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[c / 8][i] += r[i];
        }
        __syncthreads();
    }

    const int row = warp * 32 + lane;
    for (int c = 0; c < N; c += 8)
        for (int i = 0; i < 8; ++i) gD[row * N + c + i] = acc[c / 8][i];

    __syncthreads();
    if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::
            "r"(taddr), "r"(64));
    }
}

int main(int argc, char** argv) {
    const unsigned seed = argc > 1 ? (unsigned)std::atoi(argv[1]) : 42;
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 4;
    if (rounds < 1 || rounds > K / 16) {
        std::fprintf(stderr, "rounds must be in [1, %d]\n", K / 16);
        return 2;
    }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(M * K), hB(N * K);
    std::vector<float> ref(M * N, 0.0f), got(M * N);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < rounds * 16; ++k)
                ref[m * N + n] += __bfloat162float(hA[m * K + k]) *
                                  __bfloat162float(hB[n * K + k]);

    __nv_bfloat16 *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(*dA)));
    CUDA_CHECK(cudaMalloc(&dB, N * K * sizeof(*dB)));
    CUDA_CHECK(cudaMalloc(&dD, M * N * sizeof(*dD)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(*dA),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K * sizeof(*dB),
                          cudaMemcpyHostToDevice));
    tcgen05_rounds<<<1, 128>>>(dA, dB, dD, rounds);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(got.data(), dD, M * N * sizeof(*dD),
                          cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < M * N; ++i) bad += got[i] != ref[i];
    if (bad) std::printf("FAIL seed=%u: %ld / %d\n", seed, bad, M * N);
    else std::printf("PASS seed=%u\n", seed);
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return bad != 0;
}
