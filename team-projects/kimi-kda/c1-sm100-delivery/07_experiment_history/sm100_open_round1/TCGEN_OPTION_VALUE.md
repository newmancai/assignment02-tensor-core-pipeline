# tcgen05 选择权价值：主线因果问题

更新日期：2026-09-10。

## 精确问题

在保持 KDA 语义、数值契约、输入、public-full 计时边界和候选优化预算一致时：

1. 把 tensor-core backend 从 HMMA 换成 tcgen05 的直接边际收益是多少？
2. 选择 tcgen05 后，新获得或显著强化了哪些 HMMA 路径难以利用的优化自由度？
3. 这些 tcgen-specific/strongly-coupled 优化带来的附加收益是多少？
4. 最终最优 tcgen05 路径相对最优 HMMA 路径是否仍然胜出？

这里研究的不是“某条 tcgen05 指令峰值多高”，而是 **tcgen05 的选择权价值**：
直接 ISA 效果，加上它允许采用的新存储、角色和流水组织的增量价值。

## 优化分组

### 两边原则上都能做的通用优化

- 增加独立 CTA，例如 ValueSlice 将 H12 recurrence grid 从 12 扩到 96；
- phase prefetch、lookahead、双缓冲和更好的 TMA 调度；
- CHUNK/physical tile 调整；
- kernel fusion/split、shape dispatch、route cache；
- 减少冗余计算、workspace traffic 和 launch overhead。

这些优化不能计入 tcgen05 特有收益。现有 HMMA ValueSlice 约降低 27% forward
latency，已证明 HMMA 侧优化空间真实存在。

### tcgen05 特有或强耦合的优化

- 使用 TMEM 保存 accumulator，降低大 accumulator 对通用寄存器文件的压力；
- 由专门 MMA warp 异步 issue，其他 warp 同时准备、消费或执行 epilogue；
- accumulator 在多个消费阶段间保持 TMEM residency，减少 register/SMEM round-trip；
- 通过 `tcgen05.commit/wait` 与 mbarrier 构造更深的 producer/MMA/consumer pipeline；
- 在验证 lane mapping 后使用 `tcgen05.ld/st` 形成跨 warp、跨 phase carrier。

“强耦合”意味着理论上可能用 HMMA 模拟部分结构，但资源、同步和数据位置会发生
本质变化，因此更诚实的 estimand 可能是“tcgen05 + 必要协议包”的联合效应。

### HMMA 特有或更有利的优化

- 在 warp register fragments 中直接复用中间量，例如官方 P4/P6 已复用同一 U fragment；
- 避免 TMEM allocation/deallocation、commit/wait 和显式 readback 协议；
- 对小问题使用更轻量的 warp-local issue 与同步；
- 在 tcgen05 固定 tile/protocol 开销难以摊销时采用更细粒度的 shape specialization；
- 将寄存器驻留与 HMMA pipeline 联合优化，而不是强制经过 TMEM carrier。

最终比较必须允许 HMMA 和 tcgen05 各自使用自己的架构特长，并给予相同候选数、
实现工作量或自动搜索预算。只给 tcgen05 开放专属优化、却把 HMMA 固定为官方旧实现，
会高估 tcgen05 的选择权价值。

### CAKE 中存在、但尚不能认定由 tcgen05 独占的优化

- fused 与 prepare-chain 两类 route；
- 1024-thread role partition；
- compute、epilogue、beta-prefetch、mma、aux-mma、prep 的专门 warp groups；
- checkpoint/chunk 双 pipeline、23 barrier groups、97 barriers；
- shape-dependent m128/m64 dispatch。

它们与 tcgen05/TMEM 协同设计，但多数也包含通用调度思想。必须通过 matched
counterfactual 才能决定哪些收益真正依赖 tcgen05。

## 需要测量的候选矩阵

定义 public-full latency：

| ID | ISA/存储 | 允许的优化 | 用途 |
|---|---|---|---|
| `H0` | HMMA/register+SMEM | 官方组织 | 当前官方基线 |
| `H1` | HMMA/register+SMEM | 最强通用优化 + HMMA-specific 优化 | optimized-HMMA 基线 |
| `T0` | tcgen05/TMEM | 与 H1 尽量结构匹配，不启用跨 phase 特权 | 直接 ISA+必要协议 |
| `T1` | tcgen05/TMEM | H1 通用优化 + tcgen-specific 优化 | tcgen 选择权完整候选 |
| `T2` | tcgen05/TMEM | CAKE/evolution 最强 guarded route | 实际部署上界 |

分别报告：

```text
通用 HMMA 优化收益       = latency(H0) / latency(H1)
直接 ISA/必要协议收益     = latency(H1) / latency(T0)
tcgen 特殊优化附加收益    = latency(T0) / latency(T1)
tcgen 最佳相对 HMMA 最佳  = latency(H1) / latency(T1)
当前官方到部署路径总收益  = latency(H0) / latency(T2)
```

这些 ratio 不能从不同 shape 或不同 job 的旧结果相减得到。每一格必须使用相同 shape、
输入、初态、正确性 gate、public-full scope、隔离进程 CUPTI policy 和相同强度的调优
预算；H1 与 T1 都允许使用各自 ISA 特有的优化自由度。若 `T0` 无法在不破坏物理结构的情况下构造，应明确报告不可识别性，并将
`T0 -> T1` 改称联合协议包效果。

## 当前已经知道什么

- `V128 direct`: 0.919742x。说明直接 ISA replacement 不自动获益；不是 ISA 上限。
- `V16 preferred-layout L0`: 1.615021x。说明薄 N 的 tcgen compute core 有潜力。
- `V16 scalar-rematerialization L1`: 0.778523x。说明接入/布局成本可吞掉全部潜力。
- `official -> CAKE`: H12 2.4823x，H96 2.2532x。说明 tcgen05 参与的全栈组织存在
  大幅总收益，但尚未分离 direct/common/special 三部分。
- `CAKE -> guarded evolution`: 既有 H96/H64 六形状部署 geomean 1.1420x。说明即使
  已选择 CAKE，仍有额外软件/调度空间；不能自动归类为 tcgen-specific。

因此当前最强可辩护回答是：**tcgen05 的裸替换价值可能为负，但它开启的 TMEM
residency、异步 issue、warp specialization 和跨阶段流水具有正向潜力；CAKE 证明这些
能力与其他协同优化组合后总价值很大，尚缺 H1/T0/T1 结构匹配实验来量化其中“还能
做多少别的特殊优化”。**

## 论文表述边界

当前可以写：

> 选择 tcgen05 的价值主要不是 opcode substitution，而是它扩展了可行的数据驻留、
> warp-role 和异步流水设计空间；完整 SM100 共设计路径显著胜过当前官方实现。

完成 H1/T0/T1 后才可以量化写：

> 在控制通用软件优化后，tcgen05 直接/必要协议贡献 X，tcgen-specific 组织再贡献 Y，
> 最优 tcgen 路径相对最优 HMMA 路径贡献 Z。

在此之前不得把 CAKE 的 2.4823x/2.2532x 写成 X、Y 或 Z 中任何一个单项。
