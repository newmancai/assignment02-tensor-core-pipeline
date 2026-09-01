// NVFP4 的格式约定。这个文件是题面材料,不需要修改。
//
// 数据格式:
//   - 量化组 = K 维连续 16 个元素,每组一个 scale factor(SF)
//   - SF = 组内 amax / 6.0(6.0 是 e2m1 的最大值),存成 e4m3;
//     反量化用的 scale 是 float(e4m3(SF)),即 SF 本身也经过了量化
//   - 数据存 e2m1,每 byte 两个,低 nibble 放偶数下标元素
//
// SF 的存放不是行主序,而是 tensor core 消费要求的 swizzled 布局
// [numMTiles, numKTiles, 32, 4, 4]:M 向按 128 行分 tile,
// numKTiles = ceil(K/64)(一个 K tile 覆盖 4 个组)。
// 字节偏移公式见 sf_swizzled_offset,这就是问题 5.3 里
// cuBLASLt / tcgen05 kind::mxf4nvf4 实际读取的布局。
#pragma once
#include <cstdint>
#include <cuda_fp8.h>

constexpr int NVFP4_GROUP = 16;

__host__ __device__ inline int nvfp4_num_ktiles(int K) {
    return (K / NVFP4_GROUP + 3) / 4;
}

// SF 张量总字节数(M 向 128 对齐产生的 padding 也要分配并清零)。
__host__ __device__ inline int64_t nvfp4_sf_bytes(int M, int K) {
    return (int64_t)((M + 127) / 128) * nvfp4_num_ktiles(K) * 512;
}

// row 行、第 kGroup 个组(= k/16)的 SF 字节偏移。
__host__ __device__ inline int64_t sf_swizzled_offset(int row, int kGroup,
                                                      int numKTiles) {
    int mTileIdx = row >> 7;         // row / 128
    int outerM   = row & 31;         // row % 32
    int innerM   = (row >> 5) & 3;   // (row / 32) % 4
    int kTileIdx = kGroup >> 2;      // kGroup / 4
    int innerK   = kGroup & 3;       // kGroup % 4
    return ((int64_t)(mTileIdx * numKTiles + kTileIdx) << 9) |
           (outerM << 4) | (innerM << 2) | innerK;
}

// e2m1 的解码(格点是确定的,给出;编码是问题 5.3(a),你自己写)。
__host__ __device__ inline float e2m1_decode(uint8_t nib) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float m = mag[nib & 7];
    return (nib & 8) ? -m : m;
}
