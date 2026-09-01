// 问题 1.3：从零实现 m16n8k32 e4m3 mma，手动装载 fragment。
#include <cuda_fp8.h>
#include <cstdlib>
#include <random>
#include "../common.h"

__device__ __forceinline__ unsigned pack4(const uint8_t* x, int i0, int i1,
                                           int i2, int i3) {
    return (unsigned)x[i0] | ((unsigned)x[i1] << 8) |
           ((unsigned)x[i2] << 16) | ((unsigned)x[i3] << 24);
}

__global__ void mma_fp8(const uint8_t* A, const uint8_t* B, float* D) {
    const int lane = threadIdx.x;
    const int group = lane >> 2;
    const int tig = lane & 3;

    // A: i=4*r+j，row=group+8*(r&1)，col=4*tig+j+16*(r>>1)。
    unsigned a[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = group + 8 * (r & 1);
        const int col = 4 * tig + 16 * (r >> 1);
        a[r] = pack4(A + row * 32, col, col + 1, col + 2, col + 3);
    }

    // B: i=4*r+j，k=4*tig+j+16*r，n=group。
    unsigned b[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int k = 4 * tig + 16 * r;
        b[r] = pack4(B, (k + 0) * 8 + group, (k + 1) * 8 + group,
                     (k + 2) * 8 + group, (k + 3) * 8 + group);
    }

    float c[4] = {0.f, 0.f, 0.f, 0.f};
    float d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
          "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

    D[group * 8 + 2 * tig] = d[0];
    D[group * 8 + 2 * tig + 1] = d[1];
    D[(group + 8) * 8 + 2 * tig] = d[2];
    D[(group + 8) * 8 + 2 * tig + 1] = d[3];
}

int main(int argc, char** argv) {
    const unsigned seed = argc > 1 ? (unsigned)std::strtoul(argv[1], nullptr, 10)
                                   : 1u;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-8, 7);

    uint8_t hA[16 * 32], hB[32 * 8];
    float fA[16 * 32], fB[32 * 8], ref[16 * 8] = {};
    for (int i = 0; i < 16 * 32; ++i) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)dist(rng));
        hA[i] = *reinterpret_cast<uint8_t*>(&v);
        fA[i] = float(v);
    }
    for (int i = 0; i < 32 * 8; ++i) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)dist(rng));
        hB[i] = *reinterpret_cast<uint8_t*>(&v);
        fB[i] = float(v);
    }
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n)
            for (int k = 0; k < 32; ++k)
                ref[m * 8 + n] += fA[m * 32 + k] * fB[k * 8 + n];

    uint8_t *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, sizeof(ref)));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    mma_fp8<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();

    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < 16 * 8; ++i) {
        if (got[i] != ref[i]) {
            if (bad < 4)
                std::printf("MISMATCH D[%d][%d]: got %.0f, want %.0f\n",
                            i / 8, i % 8, got[i], ref[i]);
            ++bad;
        }
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);
    if (bad) {
        std::printf("FAIL: %ld / 128 mismatches\n", bad);
        return 1;
    }
    std::printf("PASS seed=%u\n", seed);
    return 0;
}
