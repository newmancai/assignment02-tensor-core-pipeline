// 问题 5.3(c):ceiling 探针。
//
// 目标:测出"5.3(b) 这种访存形状的上限带宽"。方法是写一个和你的
// quant kernel 访存完全同形(读同样的 bf16、写同样位置的 8 byte 数据
// 与 1 byte SF)、但不做任何数学的 kernel——读进来的位 xor 一下直接
// 写出去即可。它的耗时就是这个访存模式在这块卡上的地板。
//
// 跑完把三个数放在一起:探针 GB/s、你的 quant kernel GB/s(03b 的
// 输出)、两者比值。报告里回答:你的 kernel 离自己的上限还有多远,
// 差距是访存还是计算(ncu 的 SM% / DRAM% 可以佐证)。
//
// 在下面实现探针 kernel 和 launch;main 不需要修改。
#include <vector>
#include <random>
#include "../common.h"
#include "nvfp4_common.h"

template <int BLOCK>
__global__ void probe_kernel(const __nv_bfloat16* __restrict__ in,
                             uint8_t* __restrict__ dataOut,
                             uint8_t* __restrict__ sfOut, int M, int K) {
    const int groupsPerRow = K / NVFP4_GROUP;
    const int64_t totalGroups = static_cast<int64_t>(M) * groupsPerRow;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK;
    const int numKTiles = nvfp4_num_ktiles(K);

    for (int64_t group = first; group < totalGroups; group += stride) {
        const int row = static_cast<int>(group / groupsPerRow);
        const int kGroup = static_cast<int>(group -
                                            static_cast<int64_t>(row) * groupsPerRow);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(in) +
                              group * NVFP4_GROUP;

        // Consume exactly the same 32 input bytes as the quant kernel.  Each
        // adjacent bf16 pair is folded to one output byte using only xor, so
        // every input load remains observable while numerical conversion,
        // amax reduction, division, and FP4/FP8 instructions disappear.
        uint64_t packed = 0;
        uint8_t sfByte = 0;
#pragma unroll
        for (int i = 0; i < NVFP4_GROUP; i += 2) {
            const uint16_t x = src[i];
            const uint16_t y = src[i + 1];
            const uint8_t byte = static_cast<uint8_t>(
                x ^ (x >> 8) ^ y ^ (y >> 8) ^ 0x5au);
            packed |= static_cast<uint64_t>(byte) << (i * 4);
            sfByte ^= byte;
        }

        reinterpret_cast<uint64_t*>(dataOut)[group] = packed;
        sfOut[sf_swizzled_offset(row, kGroup, numKTiles)] = sfByte;
    }
}

static void launch_probe(const __nv_bfloat16* in, uint8_t* dataOut,
                         uint8_t* sfOut, int M, int K, int sms) {
    constexpr int block = 256;
    if (M <= 0 || K < NVFP4_GROUP) return;
    const int64_t totalGroups = static_cast<int64_t>(M) * (K / NVFP4_GROUP);
    const int64_t needed = (totalGroups + block - 1) / block;
    const int maxBlocks = (sms > 0 ? sms : 1) * 8;
    const int blocks = static_cast<int>(needed < maxBlocks ? needed : maxBlocks);
    probe_kernel<block><<<blocks, block>>>(in, dataOut, sfOut, M, K);
}

int main(int argc, char** argv) {
    int only_m = 0, only_k = 0;
    if (argc == 3) {
        only_m = atoi(argv[1]);
        only_k = atoi(argv[2]);
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [M K]\n", argv[0]);
        return 2;
    }
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    for (const auto& shape :
         {std::pair{1, 4096}, {16, 4096}, {256, 4096}, {1024, 4096},
          {4096, 4096}, {16384, 4096}, {4096, 7168}, {16384, 7168},
          {4096, 8192}, {16384, 8192}}) {
        int M = shape.first;
        int K = shape.second;
        if (only_m && (M != only_m || K != only_k)) continue;
        size_t n = (size_t)M * K;
        __nv_bfloat16* dx;
        uint8_t *dd, *dsf;
        CUDA_CHECK(cudaMalloc(&dx, n * 2));
        CUDA_CHECK(cudaMalloc(&dd, n / 2));
        CUDA_CHECK(cudaMalloc(&dsf, nvfp4_sf_bytes(M, K)));
        CUDA_CHECK(cudaMemset(dx, 0x3c, n * 2));
        float ms = time_avg_ms(
            [&] { launch_probe(dx, dd, dsf, M, K, sms); }, 50);
        CUDA_CHECK_KERNEL();
        double bytes = n * 2.0 + n * 0.5 + n / 16.0;
        printf("M=%-6d K=%-5d  probe %8.2f us  %6.0f GB/s\n", M, K, ms * 1e3,
               effective_gbps(bytes, ms));
        cudaFree(dx); cudaFree(dd); cudaFree(dsf);
    }
    return 0;
}
