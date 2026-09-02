# 5.3(c) · Ceiling probe

状态：已完成，并在 NVIDIA B300 SXM6 AC 上完成 5 次复跑与 Nsight Compute
归因。

`03c_ceiling_probe.cu` 应实现与 quant kernel 同形的纯数据搬运/xor 直通，用作
带宽上界。报告需比较量化 kernel 的 GB/s 与该 ceiling 的比例。

## 实现与实验方法

probe 与 quant kernel 一样由一个线程处理一组 K 方向连续的 16 个 BF16：读取
32 byte，将相邻两个 BF16 的 4 个输入字节 xor-fold 为 1 byte，共写 8 byte packed
data，并在同一个 `sf_swizzled_offset` 写 1 byte SF。它保留相同的输入、输出地址
形状和 grid-stride 遍历，但移除了 `amax`、除法以及 FP8/FP4 转换。

计时使用程序自带的 20 次 warmup 和 50 次迭代。为与 5.4 逐行对照，main
覆盖完全相同的十个形状，并支持可选的 `M K` 过滤参数。以下为三次独立复跑
中位数（Slurm Job `15414`）：

| M | K | probe 时间 | probe 有效带宽 |
|---:|---:|---:|---:|
| 1 | 4096 | 5.18 us | 2 GB/s |
| 16 | 4096 | 5.14 us | 33 GB/s |
| 256 | 4096 | 6.21 us | 433 GB/s |
| 1024 | 4096 | 12.35 us | 870 GB/s |
| 4096 | 4096 | 35.00 us | 1228 GB/s |
| 16384 | 4096 | 123.80 us | 1389 GB/s |
| 4096 | 7168 | 57.20 us | 1315 GB/s |
| 16384 | 7168 | 212.32 us | 1417 GB/s |
| 4096 | 8192 | 64.21 us | 1339 GB/s |
| 16384 | 8192 | 241.54 us | 1424 GB/s |

共同形状 `4096×7168` 的独立配对实验（Job `15338`，5 次中位数）：

| 实现 | 时间 | 有效带宽 | quant / probe |
|---|---:|---:|---:|
| ceiling probe | 55.44 us | 1357 GB/s | — |
| NVFP4 quant | 55.73 us | 1350 GB/s | **99.5%** |

这里的有效字节数均为 `2 + 0.5 + 1/16 = 2.5625 B/elem`；比值使用未插桩
的正常计时结果，而不是 ncu replay 期间程序打印的时间。

## Nsight Compute 归因

使用 Basic set 对大形状各抓取一个 launch：

| 指标 | NVFP4 quant | ceiling probe |
|---|---:|---:|
| Duration | 59.04 us | 56.74 us |
| Memory Throughput | 85.17% | 87.40% |
| L1/TEX Throughput | 95.10% | 96.87% |
| DRAM Throughput | 13.18% | 13.72% |
| Compute (SM) Throughput | 24.04% | 19.19% |
| Registers / thread | 39 | 32 |
| Achieved occupancy | 65.57% | 77.66% |

quant 的 Memory/L1 利用率远高于 SM 利用率，且已经达到纯搬运 probe 的
99.5%，因此 kernel 整体受访存路径而不是通用 ALU 计算限制。相对 probe 的约
0.5% 残余差距来自 FP8/FP4 转换与 amax：它们让 SM 利用率增加约 4.9 个百分点、
每线程多 7 个寄存器并降低 occupancy；经过单完整 wave 的 launch 调整后，这些
额外计算大部分被访存延迟覆盖。

DRAM 百分比偏低并不与上述结论矛盾：被 profile 的 launch 位于同一输入反复访问
之后，`4096×7168` 的约 75.2 MB 有效工作集可大量命中缓存，所以瓶颈具体落在
L1/TEX 路径。这里的“GB/s”是题面定义的必要读写字节除以时间，不应误写成 ncu
测得的原始 HBM 流量。

## 复现

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p build/m5
nvcc -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_100f,code=sm_100f \
  M5-low-precision/5.3c-ceiling-probe/03c_ceiling_probe.cu \
  -o build/m5/03c_ceiling_probe
srun -G 1 --time 00:10:00 ./build/m5/03c_ceiling_probe

# 03b 前两个 shape 各产生 221 个 quant launch，跳过 442 个后抓取大 shape。
srun -G 1 --time 00:10:00 ncu --set basic \
  --kernel-name regex:nvfp4_quant_kernel --launch-skip 442 --launch-count 1 \
  ./build/m5/03b_nvfp4_quant
srun -G 1 --time 00:10:00 ncu --set basic \
  --kernel-name regex:probe_kernel --launch-count 1 \
  ./build/m5/03c_ceiling_probe 4096 7168
```

十形状三次复跑原始摘要见
[`../evidence/b300-final-regression.md`](../evidence/b300-final-regression.md)。

