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
    // TODO: 实现。返回 4 bit 编码(bit3 符号,bit0-2 幅值格点下标)。
    (void)v;
    return 0;
}
