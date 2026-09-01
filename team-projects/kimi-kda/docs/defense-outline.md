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
- NCU：SM throughput ≈2.5%，DRAM throughput ≈2.2%，occupancy ≈9.4%。
- 结论：不是 Tensor Core/HBM 峰值打满，而是 recurrence 关键路径和 CTA 数不足。

## Slide 4：为什么不直接 tcgen05

- 主导矩阵 M=16，贴合 `mma.sync.m16n8k16`。
- tcgen05 BF16 有效 M tile 更大。
- TMEM/mbarrier 固定开销难以摊薄。
- 选择性 tcgen05 只值得在 M=128 phase 评估。

## Slide 5：我们的方案

- Value 维独立，无 reduction/atomic/跨 CTA 通信。
- grid 从 `(N,H)` 扩为 `(N,H,128/v)`。
- V16/V32/V64/V128 四个候选。
- 总 MMA FLOP 不变，只增加独立 CTA。

## Slide 6：代价与模型

- 切得越小，K-only workspace 被重复 TMA 读取。
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
- H75 fallback 的差异约 0，证明 guard band 生效。

## Slide 9：边界

- 这是 FlashKDA operator/trace，不是完整 Kimi tokens/s。
- 当前 dispatcher 是 B300 固定形状标定版，不宣称跨 GPU。
- BF16 state 相对 FP32 reference 与 varlen policy 仍需补实验。

## Slide 10：下一步与结论

- CTA Cluster + TMA multicast 复用 K-only workspace。
- M=128 phase 选择性 tcgen05 microbench。
- varlen distribution-aware dispatch。
- 结论：先解决 recurrence 并行度和数据复用，再谈新指令。

## 高频追问

### Q1：为什么 bitwise equal？

ValueSlice 只把独立的 Value 行分给不同 CTA，没有改变单个输出元素的计算/归约顺序。

### Q2：为什么 V16 不是永远最快？

它把 CTA 数放大 8 倍，也把 common K-only request 大量复制；当原网格已经足够大时，重复流量和 CTA service layers 反而占主导。

### Q3：第二次复跑绝对 latency 为什么更大？

复跑时 SM clock 为 1095 MHz，而原日志起始时钟状态不同。我们强调同一次运行内的 A/B 相对比较；正确性、policy 边界和相对收益均稳定。

### Q4：这能说明 Kimi 端到端快 5.68% 吗？

不能。5.68% 是 state-carrying KDA trace 的延迟降低，完整模型还包含投影、其他 attention 层、通信和 serving runtime。

### Q5：为什么说这是 SM100 路线，却没有全面用 tcgen05？

SM100 优化不等于指令替换。我们针对 B300 的 SM 数、L2、shared memory 和调度行为做 decomposition 与 dispatch；下一步再在匹配的大 tile phase 选择性采用 tcgen05。
