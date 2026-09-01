# FlashKDA on B300：从“换指令”到 recurrence 并行度重构

> 状态：可编辑初稿。成员姓名、课程信息和最终提交日期待团队补齐。

## 摘要

Kimi K3 的 Kimi Delta Attention（KDA）由 MoonshotAI 开源的 FlashKDA 提供高性能 forward kernel。官方实现即使运行在 GB200/B300 上，核心矩阵乘仍使用 SM80 世代的 `mma.sync`。本项目首先在 B300 上复现官方实现，通过 SASS 与 Nsight Compute 确认真实指令路径和性能瓶颈；随后比较机械迁移到 SM100 `tcgen05` 与并行度重构两条路线。结果表明，TP8 典型形状每卡只有 12 个 head，K2 recurrence 只发射 12 个长生命周期 CTA，SM throughput 和 achieved occupancy 极低。我们据此将 K2 沿 Value 维切分为 V16/V32/V64/V128 四个变体，并设计 B300 固定形状的资源感知 dispatcher。在两个独立 B300 运行中，优化保持 bitwise correctness，forward 高价值形状延迟降低 9.13%–26.10%，状态连续的 prefill + 64-step decode trace 降低 5.45%–5.68%。

## 1. 问题与基线

### 1.1 KDA 与 FlashKDA

KDA 是 Kimi K3 线性注意力路径。FlashKDA forward 由多个 kernel 组成，其中 K2 负责跨 chunk 的 recurrent state update，是本项目的主要优化对象。官方 benchmark 的代表形状为 `T=8192, H=96, D=128`；在 TP8 部署中，每卡独立 head 数变为 `H=12`。

### 1.2 题目要求

团队题要求三层工作：

1. 在 B300 上复现官方实现，并用 benchmark、NCU 和 SASS 确认主路径；
2. 对 CHUNK、指令形状、递推并行度、roofline、state 精度和 SM100 专版价值进行定量分析；
3. 选择一条 SM100 路线动手，和参考实现对拍并与官方 FlashKDA 比性能。

本组选择第三类路线：并行度重构，而不是只换指令。

## 2. 为什么官方仍使用 SM80 MMA

### 2.1 形状匹配

FlashKDA 使用 `CHUNK=16`。这一选择同时服务于数值稳定性、16×16 Neumann 级数求逆和 `mma.sync.m16n8k16` 的形状匹配。K2 中大量矩阵乘的 M 维是 16，SM80 MMA 可以直接以自然 tile 计算。

### 2.2 `tcgen05` 机械替换的问题

SM100 BF16 `tcgen05` 面向更大的有效 M tile。若保持 `CHUNK=16` 直接替换：

- 有效数据只占大 tile 的一小部分；
- 需要 TMEM alloc/commit/load/fence；
- 异步完成通知和 mbarrier 增加固定开销；
- 小 workload 无法摊薄这些成本。

因此“新指令峰值更高”不能推出实际 kernel 更快。真正适合评估 `tcgen05` 的位置是 M 维达到 128 的 state-update 阶段，而不是所有 M=16 路径。

## 3. B300 瓶颈诊断

### 3.1 CTA 数量

原始 K2 launcher 的 grid 为 `(N,H)`。TP8 典型形状 `N=1,H=12` 时只有 12 个 CTA，而测试 B300 有 148 个 SM。每个 CTA 又要串行遍历全部 recurrence tiles，形成长关键路径。

### 3.2 NCU 证据

官方 K2 在 H12 形状下的代表性计数器为：SM throughput 2.66%、DRAM throughput 0.36%、achieved occupancy 9.38%。V16 把 recurrence grid 从 12 个 CTA 扩展到 96 个 CTA，NCU duration 从 632.32 µs 降到 454.69 µs；但 SM/DRAM throughput 仍仅为 7.18%/1.58%。在 H74 对照中，V64 恰好发射 148 个 CTA，duration 从 645.06 µs 降到 517.31 µs，而 SM/DRAM throughput 仍仅为 26.37%/11.30%。scheduler 的 `No Eligible` 在四个配置中均为 66.63%–85.08%。

Nsight Systems 的同进程五轮对照进一步显示：Official V128 的 NVTX GPU projected span 为 3.416 ms，自动选择 V16 后为 2.505 ms（−26.7%）；最后一轮 K2 recurrence 为 625.16 µs 对 451.08 µs。浓缩图与完整解释见 `experiments/BOTTLENECK_ANALYSIS.md`。

结论是：当前边界主要由 recurrence/TMA issue critical path 与 CTA 分布决定，不是 BF16 Tensor Core 峰值或 HBM 峰值饱和。

## 4. ValueSlice 设计

### 4.1 可分解性

K2 state 的 Value 维彼此独立。把 `D=128` 切为 `v∈{16,32,64,128}` 后，每个 CTA 只更新一个 `v×D` state slice。不同 slice 之间不需要 reduction、atomic 或跨 CTA 通信。

launcher 网格从 `(N,H)` 变为 `(N,H,D/v)`：

- V16：每个 recurrence 8 个 CTA；
- V32：4 个 CTA；
- V64：2 个 CTA；
- V128：原始 1 个 CTA。

### 4.2 算术量不变

设 `C=16`、`D=128`，每个 CTA、每个 recurrence tile 的 Tensor Core 工作为：

```text
F_cta_tile(v) = 6*C*D*v + 4*C^2*v
```

slice 数为 `s=D/v`，因此总工作量：

```text
F_total(v) = x*M*s*F_cta_tile(v)
           = x*M*D*(6*C*D + 4*C^2)
```

与 v 无关。优化来自更多独立 CTA，而不是减少 FLOP。

### 4.3 代价：重复 K-only 流量

每个 Value slice 都会重新读取完整 K-only workspace。C=16、D=128 时 common bytes 为 13,888 B/tile。slice 越小，L2/TMA request 越多。T4096、BF16 state 的估算如下：

| v | slices | L2 requests / recurrence | uncached AI |
|---:|---:|---:|---:|
| 16 | 8 | 29.188 MiB | 14.25 FLOP/B |
| 32 | 4 | 15.625 MiB | 26.62 FLOP/B |
| 64 | 2 | 8.844 MiB | 47.04 FLOP/B |
| 128 | 1 | 5.453 MiB | 76.29 FLOP/B |

因此不能永远选择 V16；必须在并行度收益和重复流量之间做调度。

## 5. 资源感知 dispatcher

### 5.1 编译与资源

四个变体编译进同一个扩展。B300 ptxas 记录的 register/thread 为 V16/V32/V64/V128 = 54/58/58/73，均无 spill。动态 shared memory 使 resident block 上限分别约为 4/4/3/2。

### 5.2 决策输入

dispatcher 查询设备 SM 数、L2、shared memory、register、threads/block limit，并结合每个已编译变体的资源与离线标定的 recurrence service model。它预测每个可行 slice 的 latency，仅在相对 V128 的预测收益同时超过 3% 和 5 µs 时启用，否则保守回退 V128。

### 5.3 有效域

当前自动选择只覆盖经过标定的 B300 固定形状域。varlen、其他架构、未验证的 T/state domain 和低置信度形状全部回退 V128。这个限制是设计的一部分，不应包装成跨 GPU 的通用模型。

## 6. 正确性验证

验证覆盖：

- fixed shape BF16 public state；
- fixed shape FP32 public state；
- ragged lengths `[31,47,19]`；
- state-in/state-out/stateless；
- V16/V32/V64 与 V128 的 output 和 final state bitwise equal；
- policy 边界；
- CUDA Graph capture/replay。

bitwise equal 的原因是 ValueSlice 只分割彼此独立的 Value 行，没有改变每个输出元素内部的归约顺序。

## 7. 性能结果

### 7.1 2026-08-21 原始验证（Job 5195）

- BF16 forward：14.66%–23.30% 降时；
- FP32 forward：9.13%–21.67% 降时；
- H75 自动回退 V128，差异为 -0.23%/-0.13% 计时抖动；
- T4096 prefill + 64 次 state-carrying update：5.45% 降时。

### 7.2 2026-09-01 独立复跑（Job 14592）

- BF16 forward：14.99%–26.10% 降时；
- FP32 forward：9.37%–21.97% 降时；
- H75 仍选择 V128，差异为 +0.02%/-0.10%；
- state-carrying trace：5.68% 降时。

复跑时 GPU 时钟较低，绝对 latency 整体变大，但 dispatcher 边界、bitwise correctness 和相对收益稳定，说明结论没有依赖单次高频状态。

## 8. 六个讨论点的当前答案

### 8.1 CHUNK=16 为什么重要

CHUNK 同时影响数值范围、Neumann 级数误差、矩阵形状和 recurrence tile 数。把它增到 32/64 会减少 tile 次数，但会加大局部求逆与状态更新 tile，可能首先触碰数值误差和 shared-memory/resource 约束。最终报告仍应补一个 CHUNK=32/64 的最小 microbench 或至少给出明确的误差上界与资源表。

### 8.2 `tcgen05` 是否匹配

对主导的 M=16 路径不匹配；机械替换收益预期为负。M=128 state-update 是可测试的选择性替换点。当前答案由形状与开销分析支撑，后续可增加只测该 phase 的 microbench。

### 8.3 recurrence 并行度从哪里来

本项目已经证明 Value 维可分。其他候选包括多 head 合并、persistent kernel、2-CTA/cluster 和跨 slice multicast。多 head 合并可能减少 CTA 数；persistent kernel 不能消除单 recurrence 的串行依赖；cluster/multicast 有望减少 ValueSlice 重复的 K-only traffic，是下一阶段首选。

### 8.4 compute-bound 还是 memory-bound

不是传统意义上的 Tensor Core compute-bound 或 HBM bandwidth-bound。低 SM/DRAM throughput 与高 no-eligible-warp 指向 issue/latency/CTA-distribution-bound；L2 reuse 会改变 ValueSlice 代价，但不是唯一边界。

### 8.5 BF16 state 精度如何验证

需要同时覆盖：长序列、极端 gate/decay、不同 state dtype、state carry、多 seed、ragged lengths，并和 FP32 或 PyTorch naive reference 比较绝对/相对误差。当前 ValueSlice 与 V128 bitwise equal 证明优化没有引入新误差，但“官方 BF16 state 本身是否足够准确”仍需要单独的参考误差表。

### 8.6 是否发布 SM100 专版

建议发布“受保护的 B300/SM103 专用路径”，同时保留通用 V128 fallback。不建议以全面 tcgen05 重写为 v2 卖点；建议以 recurrence decomposition、资源感知 dispatch 与 cluster/TMA reuse 为主，选择性评估 tcgen05。这样同时保留可移植性和峰值优化空间。

## 9. 局限与下一步

1. 当前结果是 operator/trace，不是完整 Kimi serving；
2. dispatcher 标定域限于 B300 固定形状；
3. varlen distribution 尚未建模；
4. ValueSlice 重复 K-only TMA 流量，尚未用 cluster multicast 消除；
5. CHUNK=32/64、选择性 tcgen05 和 BF16 state 相对 FP32 reference 的定量表仍需补齐；
6. 正式上游前需整理 commit、CI、代码风格与 maintainer 可接受的 opt-in/autotune 接口。

## 10. 结论

本项目推翻了“B300 上仍用 SM80 MMA，所以首要优化一定是换到 tcgen05”的直觉。真实瓶颈是 K2 recurrence 的独立 CTA 太少。ValueSlice 通过架构允许的 Value 维独立性重构并行度，并用资源感知 dispatcher 控制重复流量代价，在保持 bitwise correctness 的同时稳定取得两位数的 operator 降时。该结果给出了一个更可信的 SM100 专版方向：先重构 recurrence 与数据复用，再对合适的大 tile 阶段选择性采用新指令。
