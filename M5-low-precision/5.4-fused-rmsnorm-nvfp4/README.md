# 5.4 · Fused RMSNorm + NVFP4

状态：已完成。融合 kernel、调优后的公平两步基线、十个形状正确性、三次
B300 复跑与 Nsight Compute 归因均已收口。

最终结果要逐 shape 比较 fused、两步 baseline 与 5.3(c) ceiling probe，并把
收益分解为中间张量流量减少、launch 次数变化和额外归约/量化开销。

## 实现

融合 kernel 采用“一 CTA 一行 + persistent row loop”：

1. 256 个线程以连续、合并访问读取一行 BF16 `x`，同时累加 `sum(x²)`，并把
   该行保留在 8–16 KiB dynamic shared memory；
2. warp/CTA 两级归约得到 `rnorm`，不把 RMSNorm 的 BF16 中间张量写回 HBM；
3. 每线程处理一个连续 16 元素组，从 shared memory 读取 `x`、读取可复用的
   `w`，按 `x*rnorm*w` 得到输出，计算 E4M3 scale；
4. 用 8 条硬件 `__nv_fp4x2_e2m1` 转换打包成一个 `uint64_t`，并按
   `sf_swizzled_offset` 写 scale。

这种结构保留了 5.3(b) 无组内同步的一线程一组量化，同时把 RMSNorm 中间
张量的 HBM 写/读替换为片上 shared-memory 复用。输入 `x` 只从 HBM 读取一次。

## 公平基线与调优

两步基线不是故意保留的慢版本：第一步 RMSNorm 使用 8 个 BF16/线程的
`float4` 向量化读写，K=4096 用 256 线程，K>4096 用 512 线程；第二步直接
复用 5.3(b) 已调优到同形 ceiling 约 99.5% 的 quant kernel。

在 `M=4096,K=7168` 上扫描 grid cap：

| 配置 | 2×SM | 4×SM | 6×SM | 8×SM |
|---|---:|---:|---:|---:|
| fused 时间 (us) | 100.61 | 88.15 | **85.76** | 85.99 |
| RMS baseline 总两步时间 (us) | 111.21 | **98.38** | 101.10 | 98.08 |

因此默认 fused 为 `min(M,6×SM)`，RMS baseline 为 `min(M,4×SM)`；后者的
4×SM 与 8×SM 在噪声范围内，选更简单的 4×SM。编译器报告 fused 40
registers/thread、无 spill；独立 quant 为 39 registers/thread、无 spill。

## B300 三次复跑中位数

环境：NVIDIA B300 SXM6 AC，CUDA 13.0，Driver 580.126.09，Slurm Job
`15409`；ceiling 来自 Job `15414`。

| M | K | 两步 (us) | 融合 (us) | 加速比 | probe/fused 时间比 |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 11.28 | 6.16 | 1.83x | 84.1% |
| 16 | 4096 | 11.33 | 6.16 | 1.84x | 83.4% |
| 256 | 4096 | 12.30 | 8.21 | 1.50x | 75.6% |
| 1024 | 4096 | 22.84 | 18.45 | 1.24x | 66.9% |
| 4096 | 4096 | 59.52 | 53.86 | 1.11x | 65.0% |
| 16384 | 4096 | 228.35 | 205.11 | 1.11x | 60.4% |
| 4096 | 7168 | 98.92 | 86.71 | 1.14x | 66.0% |
| 16384 | 7168 | 363.19 | 341.71 | 1.06x | 62.1% |
| 4096 | 8192 | 110.13 | 98.28 | 1.12x | 65.3% |
| 16384 | 8192 | 406.89 | 382.80 | 1.06x | 63.1% |

十个形状全部 `PASS`。最大 mismatch 为 15 byte，低于题面允许的
`data_bytes/10000 + 1`；它来自 GPU FP32 树归约与 host double 顺序不同后，
极少量值跨过 FP4 舍入边界，而不是布局或打包错误。

`probe/fused` 使用同一形状的纯访存 probe 时间除以 fused 时间，表示融合实现
相对“只有同形读写、没有 RMS/scale 数学”的上限。小 M 受固定 launch floor
影响，该比例较高；大 M 稳定在约 60%–66%。

## 为什么不是理论 2.56×

`6.56/2.56≈2.56x` 只按删除 BF16 中间张量写回和重读计算，是字节数上限，
不是运行时间承诺：

- `M≤16`：两步路径约支付两次 kernel launch，融合只支付一次；两边都没有
  足够工作填满 B300，因此收益主要来自少一次 launch，实测约 1.83–1.84×；
- `M=256–1024`：并行度逐渐充足，中间流量减少开始生效，但 RMS 归约、scale、
  FP4 转换和 shared-memory round trip 仍不能消失，收益收敛到 1.24–1.50×；
- `M≥4096`：两步中的 quant 已达到同形 probe 的约 99.5%，两个独立 kernel
  各自能高效铺满 GPU。融合减少 HBM 流量，却把归约、shared 复用和量化串在
  同一 CTA 生命周期中，收益只剩 1.06–1.14×。

代表形状 `4096×7168` 的 NCU Basic（Job `15412`）显示 fused 的
Memory/L1-TEX/SM 为 77.79%/85.88%/31.91%，40 registers/thread、14.34 KiB
dynamic shared、62.89% achieved occupancy；RMS baseline 为
38.95%/47.40%/58.64%，32 registers/thread、85.30% occupancy。融合瓶颈落在
L1/TEX 与片上供数，且资源占用降低了 occupancy；低 DRAM% 还受到重复计时后
缓存命中的影响，不能把有效 GB/s 等同为实际 HBM 流量。

## 复现

从仓库根目录运行完整脚本：

```bash
sbatch M5-low-precision/run_b300.sbatch
```

单独构建/运行：

```bash
nvcc -O3 -std=c++17 -lineinfo \
  -gencode arch=compute_100f,code=sm_100f \
  M5-low-precision/5.4-fused-rmsnorm-nvfp4/04_fused_rms_nvfp4.cu \
  -o build/m5/04_fused_rms_nvfp4
srun -G 1 --time 00:15:00 ./build/m5/04_fused_rms_nvfp4
```

原始回归与 NCU 摘要见
[`../evidence/b300-final-regression.md`](../evidence/b300-final-regression.md)。

