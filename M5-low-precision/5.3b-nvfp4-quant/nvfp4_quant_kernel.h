// 问题 5.3(b):实现 NVFP4 quant kernel。
//
// 输入 bf16 矩阵 [M, K](K 是 16 的倍数),输出:
//   dataOut:e2m1 打包数据,每行 K/2 byte,低 nibble 放偶数下标元素
//   sfOut:  e4m3 SF,swizzled 布局(偏移用 nvfp4_common.h 的
//           sf_swizzled_offset;整个 SF 张量已在调用前清零)
//
// 每组的计算顺序(判测按同一顺序生成真值,逐 byte 严格相等):
//   amax = 组内 16 个值的绝对值最大
//   sf8  = __nv_fp8_e4m3(amax / 6.0f)
//   sf   = float(sf8)
//   inv  = sf != 0 ? 1.0f / sf : 0.0f
//   nibble[i] = encode(v[i] * inv)
//
// 设备侧的 e2m1 转换直接用 cuda_fp4.h 的 __nv_fp4x2_e2m1(float2 的 .x
// 进低 nibble),它在 sm_100 家族上是单条硬件指令;你在 5.3(a) 写的
// 编码器语义与它一致,host 参考用的就是它。
//
// 组织建议:16 元素 = 32 byte,一个线程恰好负责一个组,天然免掉组内
// 线程协作;quant 没有行间依赖,grid 怎么铺完全自由。
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include "nvfp4_common.h"

template <int BLOCK>
__global__ void nvfp4_quant_kernel(const __nv_bfloat16* __restrict__ in,
                                   uint8_t* __restrict__ dataOut,
                                   uint8_t* __restrict__ sfOut, int M, int K) {
    const int groupsPerRow = K / NVFP4_GROUP;
    const int64_t totalGroups = static_cast<int64_t>(M) * groupsPerRow;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK;
    const int numKTiles = nvfp4_num_ktiles(K);

    // One thread owns one complete 16-value quantization group.  Besides
    // avoiding any synchronization, this preserves the exact sequential
    // fmaxf order used by the host reference.
    for (int64_t group = first; group < totalGroups; group += stride) {
        const int row = static_cast<int>(group / groupsPerRow);
        const int kGroup = static_cast<int>(group -
                                            static_cast<int64_t>(row) * groupsPerRow);
        const __nv_bfloat16* src = in + group * NVFP4_GROUP;

        float values[NVFP4_GROUP];
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < NVFP4_GROUP; ++i) {
            const float value = __bfloat162float(src[i]);
            values[i] = value;
            amax = fmaxf(amax, fabsf(value));
        }

        const __nv_fp8_e4m3 sf8(amax / 6.0f);
        const float sf = static_cast<float>(sf8);
        const float inv = sf != 0.0f ? 1.0f / sf : 0.0f;
        sfOut[sf_swizzled_offset(row, kGroup, numKTiles)] = sf8.__x;

        // Eight hardware E2M1 conversions form the group's eight output
        // bytes.  A single aligned 64-bit store keeps the write path fully
        // coalesced; __nv_fp4x2_e2m1 places .x in the low nibble.
        uint64_t packed = 0;
#pragma unroll
        for (int i = 0; i < NVFP4_GROUP; i += 2) {
            const __nv_fp4x2_e2m1 pair(
                make_float2(values[i] * inv, values[i + 1] * inv));
            packed |= static_cast<uint64_t>(pair.__x) << (i * 4);
        }
        reinterpret_cast<uint64_t*>(dataOut)[group] = packed;
    }
}

// 判测和 5.4 会按这个签名调用;grid 大小你自己定,写在这里。
inline void launch_nvfp4_quant(const __nv_bfloat16* in, uint8_t* dataOut,
                               uint8_t* sfOut, int M, int K, int sms) {
    constexpr int block = 256;
    if (M <= 0 || K < NVFP4_GROUP) return;
    const int64_t totalGroups = static_cast<int64_t>(M) * (K / NVFP4_GROUP);
    const int64_t needed = (totalGroups + block - 1) / block;
    // A bounded, grid-stride launch avoids scheduling tens of thousands of
    // tiny blocks on large matrices while retaining enough resident work to
    // hide memory and conversion latency.
    // With the current 39-register kernel, B300 admits six 256-thread CTAs
    // per SM.  One complete resident wave avoids the expensive partial
    // second wave that an 8*SM cap would create.
    const int maxBlocks = (sms > 0 ? sms : 1) * 6;
    const int blocks = static_cast<int>(needed < maxBlocks ? needed : maxBlocks);
    nvfp4_quant_kernel<block><<<blocks, block>>>(in, dataOut, sfOut, M, K);
}
