// 问题 5.3(a):实现 e2m1 的 round-to-nearest-even 编码。
//
// e2m1 的幅值格点是 0, 0.5, 1, 1.5, 2, 3, 4, 6(编码 0-7),bit 3 是
// 符号位。要求与硬件 cvt 指令(cvt.rn.satfinite.e2m1x2.f32)的语义
// 一致:round to nearest,恰好落在两个格点中点时取尾数为偶的那个,
// 大于 6 饱和到 6(satfinite)。输入保证是有限值。
//
// 这个函数是后面所有题目 host 参考实现的基石:5.3(b) 的判测、5.4 的
// 判测都用它生成真值,所以先用 03a_encode_check 把它和硬件逐点对齐。
//
// 提示:先把每个中点(0.25、0.75、1.25、1.75、2.5、3.5、5.0)该落到
// 哪边推清楚,再写代码。__host__ __device__ 两侧都要能编译。
#pragma once
#include <cstdint>
#include <math.h>

__host__ __device__ inline uint8_t e2m1_encode(float v) {
    // E2M1 的正数编码按数值递增，因此只需要在相邻格点的中点处分段。
    // 中点恰好可由 float 表示；偶数 magnitude code 的尾数位为 0，
    // 所以下面交替使用 <= / < 就实现了 round-to-nearest-even。
    const float a = fabsf(v);
    uint8_t magnitude;
    if (a <= 0.25f)
        magnitude = 0;  // 0.25: 0.0 (even) beats 0.5 (odd)
    else if (a < 0.75f)
        magnitude = 1;
    else if (a <= 1.25f)
        magnitude = 2;
    else if (a < 1.75f)
        magnitude = 3;
    else if (a <= 2.5f)
        magnitude = 4;
    else if (a < 3.5f)
        magnitude = 5;
    else if (a <= 5.0f)
        magnitude = 6;
    else
        magnitude = 7;  // Includes the satfinite region above 6.

    // signbit deliberately preserves negative zero, matching the CUDA FP4
    // conversion semantics as well as ordinary negative finite inputs.
    return static_cast<uint8_t>((signbit(v) ? 0x8u : 0u) | magnitude);
}
