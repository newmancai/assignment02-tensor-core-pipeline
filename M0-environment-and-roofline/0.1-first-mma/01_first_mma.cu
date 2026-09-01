// 问题 0.1:最小的 tensor core 程序,不需要修改。
//
// 单个 warp 发一条 mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32:
// D[16x8] = A[16x16] × B[16x8] + C。A/B 用小整数填充(fp16 下精确,
// f32 累加也精确),host 端用 CPU 循环对拍,所以判测是严格相等。
//
// fragment 的装载按 PTX 文档的公式写成了下标计算的形式,课件 P2.2
// 推导的就是这组公式;模块 1 会让你对另一个形状把它们重新推一遍。
//
// 运行:make run/m0_env/01_first_mma
// 题面 (b) 问会用到 Makefile 的 ptx 目标和 assignment01 的
// sassonly/ptxonly 实验,见题面。
#include <cuda_fp16.h>
#include "../common.h"

__global__ void mma_demo(const __half* A, const __half* B, float* D) {
    int lane = threadIdx.x;
    int group = lane >> 2;      // 行方向的 8 个组
    int tig = lane & 3;         // 组内 4 个线程

    // A fragment:每线程 8 个 fp16,4 个 b32 寄存器。
    // 寄存器 r 的两个元素:(row, col) 见下标;k 的后半在 r=2,3。
    unsigned a[4];
    __half2* ah = reinterpret_cast<__half2*>(a);
    ah[0] = __halves2half2(A[(group)*16 + tig * 2], A[(group)*16 + tig * 2 + 1]);
    ah[1] = __halves2half2(A[(group + 8) * 16 + tig * 2],
                           A[(group + 8) * 16 + tig * 2 + 1]);
    ah[2] = __halves2half2(A[(group)*16 + tig * 2 + 8],
                           A[(group)*16 + tig * 2 + 9]);
    ah[3] = __halves2half2(A[(group + 8) * 16 + tig * 2 + 8],
                           A[(group + 8) * 16 + tig * 2 + 9]);

    // B fragment(col 布局,B 在内存里按 [k][n] 行主序存):
    unsigned b[2];
    __half2* bh = reinterpret_cast<__half2*>(b);
    bh[0] = __halves2half2(B[(tig * 2) * 8 + group], B[(tig * 2 + 1) * 8 + group]);
    bh[1] = __halves2half2(B[(tig * 2 + 8) * 8 + group],
                           B[(tig * 2 + 9) * 8 + group]);

    float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

    // D fragment:d0,d1 在 row=group,d2,d3 在 row=group+8。
    D[(group)*8 + tig * 2] = d[0];
    D[(group)*8 + tig * 2 + 1] = d[1];
    D[(group + 8) * 8 + tig * 2] = d[2];
    D[(group + 8) * 8 + tig * 2 + 1] = d[3];
}

int main() {
    __half hA[16 * 16], hB[16 * 8];
    float ref[16 * 8] = {};
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 16; k++) hA[r * 16 + k] = __float2half((r + k) % 5 - 2);
    for (int k = 0; k < 16; k++)
        for (int n = 0; n < 8; n++) hB[k * 8 + n] = __float2half((k * n) % 3 - 1);
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
    mma_demo<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));

    long bad = 0;
    for (int i = 0; i < 16 * 8; i++) bad += got[i] != ref[i];
    printf("D[0][0]=%.0f D[0][7]=%.0f D[15][0]=%.0f D[15][7]=%.0f\n", got[0],
           got[7], got[15 * 8], got[15 * 8 + 7]);
    if (bad)
        printf("FAIL: %ld mismatches\n", bad);
    else
        printf("PASS\n");
    return bad != 0;
}
