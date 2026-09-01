// 问题 1.5:行跨度对 ldmatrix 的影响,程序不需要修改。
//
// 同一个 16x16 fp16 tile 放在 smem 里,行跨度四档:32 B(紧凑)、
// 64 B、128 B(相当于 tile 嵌在 64 列 fp16 的大矩阵里)、
// 128+16 B(padding)。kernel 反复发 ldmatrix .x4,输出每次的平均
// cycle 数。
//
// 先预测再运行:按课上的 bank 模型,每档一次 ldmatrix 需要几个
// wavefront?把四档的比值写下来再跑。ncu 的 wavefront 与 conflict
// 计数用来验证预测;耗时的比值会比 wavefront 比值小,报告里解释
// 这个差别(提示:8 个 warp 的占用度下,LSU 是不是唯一瓶颈)。
// ncu 命令(两个计数器):
//   ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
//       l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
//       ./bin/m1_sm80/05_ldsm_stride
//
// 运行:make -B bin/m1_sm80/05_ldsm_stride &&
//      ./bin/m1_sm80/05_ldsm_stride
#include <cuda_fp16.h>
#include <cstdint>
#include "../common.h"

constexpr int ITERS = 4096;

// 8 个 warp 同发把 LSU 打满,吞吐由 bank 冲突决定;单 warp 的话
// 流水会把串行化掩掉大半。每 warp 用自己的 smem 区域,访问模式相同。
template <int STRIDE>
__global__ void ldsm_kernel(unsigned* out, long long* cycles) {
    constexpr int WARPS = 8;
    __shared__ uint8_t smem[WARPS * 16 * 160];
    for (int i = threadIdx.x; i < WARPS * 16 * 160; i += WARPS * 32)
        smem[i] = (uint8_t)i;
    __syncthreads();
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int r = lane & 15, h = lane >> 4;
    unsigned addr = (unsigned)__cvta_generic_to_shared(
        &smem[w * 16 * 160 + r * STRIDE + h * 16]);
    unsigned acc = 0, r0, r1, r2, r3;
    __syncthreads();
    long long t0 = clock64();
    for (int i = 0; i < ITERS; i++) {
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(addr));
        acc ^= r0 ^ r1 ^ r2 ^ r3;
    }
    long long t1 = clock64();
    __syncthreads();
    if (threadIdx.x == 0) {
        *cycles = t1 - t0;
        *out = acc;
    }
}

template <int STRIDE>
static void run_one(const char* name) {
    unsigned* dout;
    long long* dcyc;
    CUDA_CHECK(cudaMalloc(&dout, 4));
    CUDA_CHECK(cudaMalloc(&dcyc, 8));
    ldsm_kernel<STRIDE><<<1, 256>>>(dout, dcyc);  // warmup
    ldsm_kernel<STRIDE><<<1, 256>>>(dout, dcyc);
    CUDA_CHECK_KERNEL();
    long long cyc;
    CUDA_CHECK(cudaMemcpy(&cyc, dcyc, 8, cudaMemcpyDeviceToHost));
    printf("stride %-8s %6.2f cycles / ldmatrix(8 warp 均摊)\n", name,
           (double)cyc / ITERS);
    cudaFree(dout);
    cudaFree(dcyc);
}

int main() {
    run_one<32>("32B");
    run_one<64>("64B");
    run_one<128>("128B");
    run_one<144>("128B+pad");
    return 0;
}
