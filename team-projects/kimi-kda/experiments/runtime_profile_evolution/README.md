# B300 runtime-profile 补充实验

## C1 主线 MMA 判断 receipt

`c1_b300_h12_mma_profile.json` 由 `build_c1_mma_profile.py` 从已有 SASS、
H12 targeted NCU 和 Phase-6 `tcgen05` 原始 CSV 确定性生成，
不需要人工抄数；
`c1_b300_h12_mma_assessment.json` 是确定性重放结果。它在进入挑战前就
给出主线结论：停止全面 direct swap，保留并优化现有
`mma.sync` 数据流，仅对跨 phase TMEM residency 重开测量。
前置 profile 不读取 ValueSlice/P4/Phase-1 的挑战后性能数据；它依据
官方 grid/throughput、Value 行独立语义、tile 几何、compiled residency
和 `tcgen05` 隔离 probe 选出候选。其中主导 M=16 低于
`tcgen05` 最小 M=64；V128 active blocks/SM 在 L0 从 12 降至 1，
在 L1 从 5 降至 1。这两类证据使“不能机械替换”从性能现象
收紧为可解释的物理判断。

这两个文件复用已归档实测，不声称新增 B300 运行。
Profile 同时记录 SASS、NCU、`tcgen05` CSV 与语义分析的 SHA-256，
防止证据内容变化后仍被当作同一 receipt。

## 与 C1 主线的关系

这组实验来自论文阶段，但这里只保留能回答 C1 的部分：在目标指令、数值
算法和 public ABI 不变时，B300 上仍存在由 workload 与实际驻留能力决定的
运行时调度参数机会。它进一步支持“先优化并行分解，再决定是否更换 MMA”
的作业结论。

实验对象是同一张 B300 上、每卡 H12 的 BT16 CAKE-generated prepare/chain
route，不是官方 FlashKDA ValueSlice/P4/Phase-1 补丁路径。两组收益不能相加、
相乘或冒充完整 Kimi K3 端到端收益。

## 物理规则

令 `W` 为 total chunks，`H` 为本地 head 数，`S` 为 SM 数，`R` 为该已编译
prepare kernel 在每个 SM 上的实测 resident CTA 数。选择使 H-way prepare
grid 首次装入 resident capacity 的最小正整数 `cpc`：

```text
cpc_cap(W,H,S,R) = ceil(W / floor(S*R/H))
```

本次 receipt 为 `H=12, S=148, R=5`，总 resident grid capacity 是 740 CTA。
`R=5` 来自该 kernel 的 45,056 B shared memory、128 threads 和实际 occupancy
结果；它不是 B300 或其他 kernel 的通用常数。

## 无候选筛选的前瞻结果

在首次 GPU 查询前封存 W384/W768、cpc7/cpc13、物理 route、八个独立进程
epoch、ABBA/BAAB 次序与 Bonferroni 单侧 98.75% 判据。没有进行 cpc screen
或邻域搜索。

| Profile | W | cpc9→预测值 | grid 变化 | Speedup | 单侧 98.75% lower |
|---|---:|---:|---:|---:|---:|
| Fixed 6144 | 384 | 9→7 | 516→660 | 1.0133× | 1.0129× |
| Balanced 4-way | 384 | 9→7 | 516→660 | 1.0331× | 1.0322× |
| Fixed 12288 | 768 | 9→13 | 1032→720 | 1.0494× | 1.0488× |
| Balanced 4-way | 768 | 9→13 | 1032→720 | 1.1354× | 1.1352× |

四个 profile 的 output/final state 最大绝对差均为 0，八个进程 epoch 全部同
方向。W384 需要增加 CTA、W768 需要减少 CTA，两边都加速，因此数据否定
“CTA 越少越好”或“CTA 越多越好”的单调解释，支持 resident-capacity
边界解释。

资格通过后才运行的 CUPTI 诊断将 97.98%–101.16% 的 full-span 节省归到
prepare，chain 的 95% 区间全部位于事前冻结的 ±1% 等价带。该结果只支持
已测 route 的阶段机制，不是跨 kernel、跨 GPU 的普遍定律。

## 证据与决策

- `b300_prospective_capacity_certificate.json`：前瞻资格、统计结果、正确性、
  route/resource receipt 和 claim boundary。
- `b300_prepare_mechanism_certificate.json`：资格完成后的 CUPTI prepare/chain
  归因。

当前决策为 **shadow-only**：规则可以生成 recommendation receipt，但不覆盖
现有生产 dispatcher。只要 ABI、route、kernel resource、occupancy receipt
或设备变化，就必须重新测量 `R` 并重新资格验证。
