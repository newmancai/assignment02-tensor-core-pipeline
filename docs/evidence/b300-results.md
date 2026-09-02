# B300 实验原始输出归档

日期：2026-09-01
环境：NVIDIA B300，CUDA 13.0，`compute_100f/sm_100f`
调度方式：所有 GPU 程序均通过 Slurm `srun -G 1` 执行。

本文保留报告中使用的数据及对应终端输出。性能数字是单次实验结果，可能随
GPU 时钟、温度和同卡负载波动。

当前工作环境没有可用的浏览器运行时，因此无法生成终端 PNG 截图。本文件
保留最终回归的可复制文本证据，没有用模拟图片替代实测输出。最终一致回归
对应 Slurm Job 14793。

## 3.2 正确版本

```text
PASS seed=1
PASS seed=7
PASS seed=42
PASS seed=1234
PASS seed=99999
JUDGE: PASS
```

## 3.2 移除 fence.proxy.async

使用 `-DOMIT_PROXY_FENCE` 编译，本次实验输出：

```text
PASS seed=1
PASS seed=7
PASS seed=42
PASS seed=1234
PASS seed=99999
```

本次 B300 运行没有观察到数值错误。这个结果不能证明 fence 可删除，只说明
当前访问模式和时序没有触发可见错误；PTX 内存模型中 generic proxy 的
shared store 仍需通过 proxy fence 发布给 tcgen05 使用的 async proxy。

## 3.3 修复版本

```text
rounds=1 PASS seed=42
rounds=1 PASS seed=7
rounds=2 PASS seed=42
rounds=2 PASS seed=7
rounds=4 PASS seed=42
rounds=4 PASS seed=7
JUDGE: PASS
```

## 3.3 BUGGY_PHASE 复现版本

```text
PASS seed=42
BUGGY_R2_FAILED_OR_TIMEOUT
BUGGY_R4_FAILED_OR_TIMEOUT
```

固定等待 phase 0 时，rounds=1 可以完成；从第二个 barrier generation
开始，rounds=2 和 rounds=4 在 20 秒限制内失败或超时。

## 3.4 CTA pair

正确性、容量与普通计时：

```text
smem/block: cta_group::1 = 24588 B, cta_group::2 = 20492 B
::1  PASS(bad=0)  20.51 us
::2  PASS(bad=0)  22.39 us
```

题面说明单 tile 时间差位于噪声范围，因此结论不依据上述耗时。

Nsight Compute：

```text
Metric: l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
tile_kernel<1>: 778
tile_kernel<2>: 650
```

## 4.1 Tiled GEMM

```text
[01_tiled] M=4096 N=4096 K=4096  tile=128x64x64 bf16/f32
time = 2.753 ms   49.9 TFLOPS
cuBLAS 0.141 ms 976.2 TFLOPS; exact PASS (bad=0); attainment 5.1%
```

### 4.1 Nsight Compute detailed profile

独立 profiling：Slurm Job 14904，Nsight Compute 2025.3.1，`detailed` set，
单次 `gemm_tiled_kernel`，22 passes。正常 benchmark 数据仍采用上面的 Job
14793，不能用 profiler replay 下的时间替代。

```text
Compute (SM) Throughput       46.62%
Memory Throughput             30.97%
DRAM Throughput                0.38%
Issue Slots Busy              41.40%
ALU pipeline utilization      46.6%
L2 Hit Rate                   88.47%
Theoretical Occupancy         37.50%
Achieved Occupancy            29.85%
Excessive global sectors      15728640 / 117440512 (about 13%)
```

原始证据：

- [`M4-gemm/4.1-tiled/evidence/m41-details.txt`](../../M4-gemm/4.1-tiled/evidence/m41-details.txt)
- [`M4-gemm/4.1-tiled/evidence/m41-detailed.ncu-rep`](../../M4-gemm/4.1-tiled/evidence/m41-detailed.ncu-rep)

## 4.2 TMA

```text
[02_tma] M=4096 N=4096 K=4096  tile=128x64x64 bf16/f32, TMA staging
time = 0.492 ms   279.5 TFLOPS
cuBLAS 0.140 ms 978.8 TFLOPS; exact PASS (bad=0); attainment 28.6%
```

## 4.3 Pipeline，S=3

```text
[03_pipeline] M=4096 N=4096 K=4096  tile=128x64x64  STAGES=3
STAGES=3     4096    4096    4096  time=0.478 ms  287.8 TFLOPS
cuBLAS 0.141 ms 971.9 TFLOPS; exact PASS (bad=0); attainment 29.6%
```

## 4.3 stage sweep

```text
== 形状 4096 4096 4096 ==
STAGES=2  time=0.455 ms  301.8 TFLOPS  exact PASS
STAGES=3  time=0.476 ms  288.5 TFLOPS  exact PASS
STAGES=4  time=0.543 ms  253.3 TFLOPS  exact PASS
STAGES=6  time=0.750 ms  183.3 TFLOPS  exact PASS

== 形状 256 4096 16384 ==
STAGES=2  time=0.204 ms  168.5 TFLOPS  exact PASS
STAGES=3  time=0.166 ms  207.4 TFLOPS  exact PASS
STAGES=4  time=0.181 ms  189.4 TFLOPS  exact PASS
STAGES=6  time=0.163 ms  210.5 TFLOPS  exact PASS
```

### 4.3 resident blocks/SM

Slurm Job 14903；通过
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 查询当前 kernel：

```text
STAGES=2  dynamic_smem=49152 B   max_resident_blocks_per_sm=1
STAGES=3  dynamic_smem=73728 B   max_resident_blocks_per_sm=1
STAGES=4  dynamic_smem=98304 B   max_resident_blocks_per_sm=1
STAGES=6  dynamic_smem=147456 B  max_resident_blocks_per_sm=1

num_regs/thread=74  static_smem=1024 B
smem_per_sm=233472 B  regs_per_sm=65536  hw_max_blocks_per_sm=32
```

S=2 另以 Nsight Compute 2025.3.1 `basic` set 验证（Job 14933）：achieved
occupancy 21.48%；NCU advisory 将 6.2% theoretical occupancy 的限制归因于
required shared memory。原始证据：

- [`M4-gemm/4.3-pipeline/evidence/m43-s2-details.txt`](../../M4-gemm/4.3-pipeline/evidence/m43-s2-details.txt)
- [`M4-gemm/4.3-pipeline/evidence/m43-s2-basic.ncu-rep`](../../M4-gemm/4.3-pipeline/evidence/m43-s2-basic.ncu-rep)

加入 occupancy 输出后的正常模式回归：Slurm Job 14931。两种形状、四种 stage
全部 `exact PASS`；性能分别为：

```text
4096^3:             S2 301.3, S3 287.9, S4 254.1, S6 183.5 TFLOPS
256x4096x16384:     S2 166.7, S3 207.1, S4 190.4, S6 211.7 TFLOPS
```

这些结果与 Job 14793 的趋势一致；绝对值差异属于时钟和测量波动。

## 4.5 Thin GEMM

Slurm Job `15340`；7 个 Kimi K3 投影形状 × 9 个 M，三个独立进程逐点
中位数。代表结果：

```text
M=16:    大 K 形状 20.1–95.1 TFLOPS
M=256:   大 K 形状 296.1–1005.8 TFLOPS
M=65536: 大 K 形状 1265.1–1322.6 TFLOPS（官方峰值的 56.2%–58.8%）
f_b_proj(K=128), M=65536: 413.2 TFLOPS，AI=117.9，始终 memory-bound
```

63 点原始输出、生成脚本与完整 Roofline 解释见
[`M4-gemm/4.5-thin-gemm/`](../../M4-gemm/4.5-thin-gemm/README.md)。

## M5 低精度最终回归

Slurm Job `15409`：

```text
E2M1 hardware check: PASS, 202864 values
NVFP4 byte check:    3/3 PASS, bad=0
cuBLASLt FP4 GEMM:   3/3 PASS, maxrel about 3.9e-3
Fused RMS+NVFP4:     10/10 PASS
```

5.4 三次复跑中位数：小 M 为 1.83–1.84×，过渡区为 1.24–1.50×，
大 M 为 1.06–1.14×。Job `15414` 的十形状 ceiling probe 为 2–1424 GB/s；
Job `15412` 的代表 fused profile 为 40 registers/thread、14.34 KiB dynamic
shared、62.89% achieved occupancy，Memory/L1-TEX/SM throughput 为
77.79%/85.88%/31.91%。完整逐形状表与 NCU 摘要见
[`M5-low-precision/evidence/b300-final-regression.md`](../../M5-low-precision/evidence/b300-final-regression.md)。

## assignment01 naive FP32 基线

```text
BS=16  平均 0.686 ms  3128.8 GFLOPS
PASS
```

该基线使用 assignment01 原程序的 1024³ FP32 形状，只用于比较性能量级，
不应与 4096³ BF16 Tensor Core 结果直接计算达成率。
