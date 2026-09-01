# Prob 0.2：Tensor Core 理论峰值与机器平衡点

## 1. 统一计算口径

- 使用 **dense（稠密）** 吞吐，不使用 2:4 sparsity 的“有效吞吐”。
- RTX 5090 使用 datasheet 的 boost clock `2.407 GHz`。
- B300 的架构上限使用本次服务器实际读到的最大应用时钟 `2.032 GHz`；
  与官方系统额定值对照时，再单独列官方值。
- 一次 FMA（`a*b+c`）按一次乘法加一次加法，计 `2 FLOP`。
- BF16/FP8 均以 FP32 accumulate 为可比口径；FP4 采用 datasheet 的
  FP32 accumulate 口径。

## 2. 基本公式

```text
GPU 峰值 = 每 SM 每周期 FLOP × SM 数量 × SM 时钟
机器平衡点 = BF16 峰值 FLOP/s ÷ 显存带宽 byte/s
```

机器平衡点的含义：一个 kernel 每从显存搬运 1 byte 数据，至少需要做这么多
FLOP，才有可能先撞到计算峰值；低于它通常先受显存带宽限制。

## 3. RTX 5090 推导

RTX 5090 有 `170 SM`，boost clock 为 `2.407 GHz`。BF16 Tensor Core
在 FP32 accumulate 的 dense 口径下为 `512 FLOP/cycle/SM`：

```text
BF16 = 512 × 170 × 2.407 GHz
     = 209.505 TFLOPS ≈ 209.5 TFLOPS

FP8（仅按 16 bit → 8 bit 宽度估算）
    = 2 × 209.505
    = 419.011 TFLOPS ≈ 419.0 TFLOPS

FP4（仅按 16 bit → 4 bit 宽度估算）
    = 4 × 209.505
    = 838.021 TFLOPS ≈ 838.0 TFLOPS

机器平衡点
    = 209.505 TFLOPS ÷ 1792 GB/s
    = 116.91 FLOP/byte
```

NVIDIA 官方 dense 值为 BF16 `209.5 TFLOPS`、FP8（FP32 accumulate）
`419 TFLOPS`、FP4（FP32 accumulate）`1676 TFLOPS`。BF16 与 FP8 和上述
推导一致；官方 FP4 是简单“位宽减半”估算的 2 倍，说明 FP4 的专用数据通路
吞吐不能只靠 dtype 宽度推出。官方表中斜杠后的数是启用 sparsity 后的有效吞吐；
FP8 若改用 FP16 accumulate，官方 dense 值还会变成 `838 TFLOPS`，不能与本表
的 FP32 accumulate 直接混用。

## 4. B300 推导

CUTLASS 官方文档说明 SM100 的 `tcgen05.mma.kind::f16`（也包括 BF16）
吞吐是 Hopper FP16 Tensor Core 的 2 倍。因此 dense BF16 为：

```text
Hopper BF16 dense = 4096 FLOP/cycle/SM
B300 BF16 dense   = 2 × 4096
                  = 8192 FLOP/cycle/SM
```

本次在服务器实际读取到：`148 SM`、最大应用时钟 `2.032 GHz`、
`7680-bit` HBM 总线、内存时钟 `3996 MHz`。于是按“架构吞吐 × 最大时钟”的
硬件上限计算：

```text
BF16 = 8192 × 148 × 2.032 GHz
     = 2463.629 TFLOPS

FP8（仅按宽度估算）
    = 2 × 2463.629
    = 4927.259 TFLOPS

FP4（仅按宽度估算）
    = 4 × 2463.629
    = 9854.517 TFLOPS

实际接口理论带宽
    = 2 × 3.996 GHz × 7680 bit ÷ 8
    = 7672.32 GB/s

对应机器平衡点
    = 2463.629 TFLOPS ÷ 7672.32 GB/s
    = 321.11 FLOP/byte
```

NVIDIA 的 HGX B300 是 8 GPU 系统。官方表中 FP16/BF16 `36 PFLOPS` 和
FP8 `72 PFLOPS` 都是 sparse 值，dense 是其一半；FP4 则直接给出
`144 PFLOPS sparse / 108 PFLOPS dense`。除以 8 后，每 GPU 官方 dense 值为：

```text
BF16 = 36 ÷ 2 ÷ 8 = 2.25 PFLOPS = 2250 TFLOPS
FP8  = 72 ÷ 2 ÷ 8 = 4.50 PFLOPS = 4500 TFLOPS
FP4  = 108 ÷ 8     = 13.50 PFLOPS = 13500 TFLOPS
```

官方还给出 Blackwell Ultra 每 GPU **up to 8 TB/s**。因此采用官方额定值画
roofline 时：

```text
B300 官方 BF16 机器平衡点 = 2250 TFLOPS ÷ 8000 GB/s
                           = 281.25 FLOP/byte
```

`2463.63` 与官方 `2250 TFLOPS` 的差异主要来自口径：前者把驱动报告的最大
应用时钟当作全芯片始终可维持的频率，是架构上界；后者是 8-GPU 产品的额定、
取整规格。最大时钟不等于满载可持续时钟。FP4 官方值又高于 BF16×4，因为
Blackwell Ultra 对 NVFP4 有额外增强，不能只按位宽线性估算。

## 5. 最终结果表

| 量 | RTX 5090 | B300（实卡最大时钟推导） |
|---|---:|---:|
| BF16 FLOP/cycle/SM | 512 | 8192 |
| BF16 峰值 | 209.5 TFLOPS | 2463.63 TFLOPS |
| FP8 宽度估算 | 419.0 TFLOPS | 4927.26 TFLOPS |
| FP4 宽度估算 | 838.0 TFLOPS | 9854.52 TFLOPS |
| 显存带宽 | 1792 GB/s | 7672.32 GB/s |
| BF16 机器平衡点 | 116.91 FLOP/byte | 321.11 FLOP/byte |
| datasheet dense 对照 | BF16 209.5；FP8 419；FP4 1676 | BF16 2250；FP8 4500；FP4 13500；带宽 up to 8000 |

报告后续 roofline 建议统一采用官方 B300 参数：`2250 TFLOPS`、`8000 GB/s`、
`281.25 FLOP/byte`；实卡推导值用来解释口径差异。

## 6. 与单条 MMA 的 3.2 FLOP/byte 比较

```text
RTX 5090：116.91 ÷ 3.2 = 36.53 倍
B300 官方：281.25 ÷ 3.2 = 87.89 倍
```

单条 MMA 的计算强度远低于整卡机器平衡点，但这不表示完整 GEMM 必然只能达到
低性能。关键是把从 HBM 读入的数据在 shared memory、寄存器和 TMEM 中重复使用，
让同一份 A/B tile 服务许多条 MMA。M2 的 swizzle/descriptor 解决正确、高效供数；
M3 的 TMEM/TMA 减少搬运开销；M4 的多级 pipeline 把搬运与计算重叠。它们共同把
kernel 级计算强度和 Tensor Core 利用率提高到接近 roofline。

## 7. 数据来源

- [NVIDIA RTX Blackwell GPU Architecture，Appendix A](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [NVIDIA CUTLASS：Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/blackwell_functionality.html)
- [NVIDIA HGX Platform：HGX B300 规格](https://www.nvidia.com/en-au/data-center/hgx/)
- [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
