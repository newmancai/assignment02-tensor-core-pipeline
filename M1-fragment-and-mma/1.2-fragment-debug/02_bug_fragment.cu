// 问题 1.2:修 bug。这个程序发一条 m16n8k16 fp16 的 mma,判测会 FAIL。
//
// 提交时回答两问(先跑,后改):
// (a) 描述错误症状:D 的哪些位置错,错成了什么样(和对的部分是什么
//     关系)。
// (b) 错的是哪个 fragment 的哪部分映射?为什么恰好产生 (a) 的症状?
//
// 运行:make run/m1_sm80/02_bug_fragment
#include <cuda_fp16.h>
#include "../common.h"

__global__ void mma_buggy(const __half* A, const __half* B, float* D) {
    int lane = threadIdx.x;
    int group = lane >> 2;
    int tig = lane & 3;

    // A fragment:8 个 fp16 逐个装载。
    __half a0 = A[group * 16 + tig * 2];
    __half a1 = A[group * 16 + tig * 2 + 1];
    __half a2 = A[(group + 8) * 16 + tig * 2];
    __half a3 = A[(group + 8) * 16 + tig * 2 + 1];
    __half a4 = A[group * 16 + tig * 2 + 8];
    __half a5 = A[group * 16 + tig * 2 + 9];
    __half a6 = A[(group + 8) * 16 + tig * 2 + 8];
    __half a7 = A[(group + 8) * 16 + tig * 2 + 9];
    unsigned ra[4];
    reinterpret_cast<__half2*>(ra)[0] = __halves2half2(a0, a1);
    reinterpret_cast<__half2*>(ra)[1] = __halves2half2(a2, a3);
    reinterpret_cast<__half2*>(ra)[2] = __halves2half2(a4, a5);
    reinterpret_cast<__half2*>(ra)[3] = __halves2half2(a6, a7);

    __half b0 = B[(tig * 2) * 8 + group];
    __half b1 = B[(tig * 2 + 1) * 8 + group];
    __half b2 = B[(tig * 2 + 8) * 8 + group];
    __half b3 = B[(tig * 2 + 9) * 8 + group];
    unsigned rb[2];
    reinterpret_cast<__half2*>(rb)[0] = __halves2half2(b0, b1);
    reinterpret_cast<__half2*>(rb)[1] = __halves2half2(b2, b3);

    float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]),
          "r"(rb[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

    D[group * 8 + tig * 2] = d[0];
    D[group * 8 + tig * 2 + 1] = d[1];
    D[(group + 8) * 8 + tig * 2] = d[2];
    D[(group + 8) * 8 + tig * 2 + 1] = d[3];
}

int main() {
    __half hA[16 * 16], hB[16 * 8];
    float ref[16 * 8] = {};
    // A 的上下半特意不同,别改这份数据。
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 16; k++)
            hA[r * 16 + k] = __float2half((float)((r * 7 + k * 3) % 9 - 4));
    for (int k = 0; k < 16; k++)
        for (int n = 0; n < 8; n++)
            hB[k * 8 + n] = __float2half((float)((k + n * 5) % 7 - 3));
    for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++)
            for (int k = 0; k < 16; k++)
                ref[r * 8 + n] += __half2float(hA[r * 16 + k]) *
                                  __half2float(hB[k * 8 + n]);

    __half *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, 16 * 8 * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    mma_buggy<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));

    long bad = 0;
    for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++)
            if (got[r * 8 + n] != ref[r * 8 + n]) {
                if (bad < 4)
                    printf("MISMATCH D[%d][%d]: got %.0f, want %.0f\n", r, n,
                           got[r * 8 + n], ref[r * 8 + n]);
                bad++;
            }
    if (bad)
        printf("FAIL: %ld / 128 mismatches\n", bad);
    else
        printf("PASS\n");
    return bad != 0;
}
