# 4.1 · Tiled GEMM

标准化实验元数据见 [`EXPERIMENT.md`](EXPERIMENT.md)。

状态：实现、cuBLAS 严格对拍和 B300 性能实验已完成。

实现文件：[`01_tiled.cu`](01_tiled.cu)

分析脚本：[`profile_ncu.sh`](profile_ncu.sh)，需在 Slurm GPU allocation 内运行。

## 实现

该版本把 3.2 的 m128n64k64 单 tile 扩展到完整 GEMM：

- grid 为 `(N/64, M/128)`，每个 CTA 负责一个 128×64 输出 tile；
- K 维以 64 为步长循环；
- 每轮由 128 个线程使用普通 global load/shared store staging A/B；
- shared memory 使用 K-major、128B swizzle；
- 每轮发射四条 k16 tcgen05 MMA；
- 第一条 MMA 清空 accumulator，后续 K tile 使用 `scale_c=1` 累加；
- commit 完成后才允许下一轮覆盖单缓冲 shared memory；
- epilogue 通过 `tcgen05.ld` 读取 FP32 accumulator，转为 BF16 写回。

当前实现要求 M、N、K 分别能被 128、64、64 整除。题面测试形状满足该
约束，因此没有额外的边界 tile 分支。

## 正确性

程序使用同一 A/B 输入调用 cuBLAS：

```cpp
cublasGemmEx(...,
             CUDA_R_16BF, CUDA_R_16BF,
             CUDA_R_16BF, CUBLAS_COMPUTE_32F, ...);
```

自定义 kernel 与 cuBLAS BF16 输出按 16-bit 位模式逐元素严格比较，而不是
只进行抽样或容差比较。

## B300 数据

4096³：

```text
time = 2.753 ms   49.9 TFLOPS
cuBLAS 0.141 ms 976.2 TFLOPS; exact PASS (bad=0); attainment 5.1%
```

## 瓶颈

4096³ BF16 GEMM 若只按 A/B/C 各搬运一次计算，算术强度约为：

`AI = 2×4096³ / (3×4096²×2 B) = 1365.3 FLOP/byte`

它高于 0.2 中 B300 官方机器平衡点 281.25 FLOP/byte，所以理想的充分复用
GEMM 应属于 compute-bound；但这不表示当前 4.1 实现能达到计算屋顶。

单个 K tile 的时间近似：

`T_tile = T_global_load + T_address + T_shared_store + T_mma + T_wait`

普通 staging 占用 SM 的 LSU/issue 带宽，并需要每个线程进行地址计算。
更重要的是单缓冲使 staging 与 MMA 严格串行，无法隐藏 global-memory
延迟。虽然使用了 Tensor Core，Tensor Core 得不到持续数据供给，因此该
版本的主要限制是数据供给路径和串行同步，而不是 BF16 峰值算力。

## Nsight Compute 分析

在 B300 上使用 Nsight Compute 2025.3.1 的 `detailed` set，对一次
`gemm_tiled_kernel` 采集 22 passes：

| 指标 | 数值 |
|---|---:|
| Compute (SM) Throughput | 46.62% |
| Memory Throughput | 30.97% |
| DRAM Throughput | 0.38% |
| Issue Slots Busy | 41.40% |
| ALU pipeline utilization | 46.6% |
| L2 hit rate | 88.47% |
| excessive global sectors | 15,728,640 / 117,440,512（约 13%） |
| theoretical / achieved occupancy | 37.50% / 29.85% |

DRAM 吞吐只有 0.38%，排除了 HBM 带宽饱和；ALU 是利用率最高的普通流水线，
同时 issue slots 只有 41.40%，并存在约 13% 的多余 global sectors。这与代码
结构一致：线程执行地址计算、标量 global load、swizzle 地址计算和 shared
store，且单缓冲等待造成延迟气泡。4.1 的瓶颈是普通 staging 的指令/地址
开销、访问合并与延迟，而不是 HBM 峰值或 Tensor Core 算力屋顶。

证据：[详细文本](evidence/m41-details.txt)；
[可由 Nsight Compute 重新打开的报告](evidence/m41-detailed.ncu-rep)。采集期间
profiler 会重放 kernel，不能用 NCU 下的耗时替换上面的正常 benchmark 数据。

## 与下一阶的关系

4.2 保持 tile、descriptor、MMA 和 epilogue 不变，只把普通 staging 替换
为 TMA，从而隔离搬运指令开销的影响。

完整输出见 [B300 实验归档](../../docs/evidence/b300-results.md)。

NCU 复现（必须通过 Slurm）：

```bash
srun -G 1 --time 00:15:00 bash profile_ncu.sh ./m41-detailed
```
