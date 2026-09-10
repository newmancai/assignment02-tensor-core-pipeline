# 答辩讲述骨架（10–12 分钟）

## Slide 1：问题

- Kimi K3 的 KDA forward 由 FlashKDA 加速。
- 在 B300 上仍看到 SM80 `mma.sync`。
- 问题：是官方落后了，还是负载根本不适合机械换指令？

## Slide 2：任务与基线

- 官方 commit `1ce47ea`。
- B300：CC 10.3、148 SM。
- TP8 后每卡 `H=12`。
- 交付要求：复现、分析、挑战、代码/报告/答辩。

## Slide 3：第一张关键图——为什么慢

- K2 grid 只有 `(N,H)=12` 个 CTA。
- 148 个 SM 大量空闲。
- NCU：SM throughput 2.66%，DRAM throughput 0.36%，occupancy 9.38%。
- 结论：不是 Tensor Core/HBM 峰值打满，而是 recurrence 关键路径和 CTA 数不足。

## Slide 4：为什么不直接 tcgen05

- 主导矩阵 M=16，贴合 `mma.sync.m16n8k16`。
- tcgen05 1-CTA BF16 的最小 M=64；保持当前朝向硬套 M=16 时行利用率至多 25%。
- TMEM/mbarrier 固定开销难以摊薄。
- 选择性 tcgen05 应首先在 M=128 phase 评估。
- V64/V128 的部分 `[16,V]` phase 可转置重排，需另测 layout/TMEM/store 成本。
- “SM80 路径”仅指 MMA atom；当前 kernel 已使用 SM90+ 的 TMA/stmatrix。

## Slide 5：我们的方案

- Value 维独立，无 reduction/atomic/跨 CTA 通信。
- grid 从 `(N,H)` 扩为 `(N,H,128/v)`。
- V16/V32/V64/V128 四个候选。
- 总 MMA FLOP 不变，只增加独立 CTA。

## Slide 6：代价与模型

- 切得越小，slice-independent inputs 被重复 TMA 读取。
- V16 并行最多，但 L2 request 最大。
- dispatcher 结合 SM/L2/smem/register/residency 与离线 service model。
- 3% + 5 µs guard band；不确定就 V128。

## Slide 7：正确性

- BF16/FP32、fixed/ragged、stateful/stateless。
- output 和 final state 均与 V128 bitwise equal。
- CUDA Graph 通过。
- 原理：没有改变每个元素内部归约顺序。

## Slide 8：性能

- Job 5195：BF16 14.66%–23.30%，FP32 9.13%–21.67%，trace 5.45%。
- Job 14592 独立复跑：BF16 14.99%–26.10%，FP32 9.37%–21.97%，trace 5.68%。
- Job 15466 第三次复跑：BF16 14.72%–25.33%，FP32 9.20%–21.90%，trace 5.42%。
- H75 fallback 的差异约 0，证明 guard band 生效。

## Slide 9：边界

- 这是 FlashKDA operator/trace，不是完整 Kimi tokens/s。
- 当前 dispatcher 是 B300 固定形状标定版，不宣称跨 GPU。
- BF16 state 相对 FP32 reference 与 varlen policy 仍需补实验。

## Slide 10：下一步与结论

- CTA Cluster + TMA multicast 复用 slice-independent inputs。
- T4096 单 sequence-head 的理想 source-request 模型可少 23.734 MiB（29.188 → 5.453 MiB）；需测 cluster occupancy。
- M=128 phase 选择性 tcgen05 microbench。
- varlen distribution-aware dispatch。
- 结论：先解决 recurrence 并行度和数据复用，再谈新指令。

## 高频追问

### Q1：为什么 bitwise equal？

ValueSlice 只把独立的 Value 行分给不同 CTA，没有改变单个输出元素的计算/归约顺序。

### Q2：为什么 V16 不是永远最快？

它把 CTA 数放大 8 倍，也把 slice-independent source request 大量复制；当原网格已经足够大时，重复流量和 CTA service layers 反而占主导。

### Q3：第二次复跑绝对 latency 为什么更大？

复跑时 SM clock 为 1095 MHz，而原日志起始时钟状态不同。我们强调同一次运行内的 A/B 相对比较；正确性、policy 边界和相对收益均稳定。

### Q4：这能说明 Kimi 端到端快 5.68% 吗？

不能。5.68% 是 state-carrying KDA trace 的延迟降低，完整模型还包含投影、其他 attention 层、通信和 serving runtime。

### Q5：为什么说这是 SM100 路线，却没有全面用 tcgen05？

SM100 优化不等于指令替换。我们针对 B300 的 SM 数、L2、shared memory 和调度行为做 decomposition 与 dispatch；下一步再在匹配的大 tile phase 选择性采用 tcgen05。

### Q6：SM103 比 SM80 好在哪里，对 Kimi KDA 有什么帮助？

SM103 增加了 tcgen05/TMEM、TMA、CTA Cluster/multicast、更大的 shared memory 和
FP8/FP6/FP4；B300 整卡也有更高 BF16 峰值、容量和带宽。对当前 KDA，已经验证的
收益不是“换新 MMA”，而是用更多独立 ValueSlice CTA 喂满更多 SM；下一步最直接的
SM103 专用优化是用 cluster multicast 去掉 slice 间的 common-input 重复读，并先在
M=128 state-update 测 tcgen05。保持当前朝向时 M=16 更适合 `mma.sync`；V64/V128
的部分 phase 可再研究转置重排，不能把全面机械替换当作默认答案。

### Q7：5.42%–5.68% trace 收益是否包含单 token decode 加速？

不包含。`T=4096` prefill 会启用 ValueSlice，但 64 次 `T=1` update 都超出当前标定域
并回退 V128；kernel 路径变化只发生在 prefill，整体 trace 收益主要来自 prefill，
不能写成 decode kernel 已加速。
