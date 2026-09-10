# HMMA / tcgen05 双分支闭环：默认知识饱和复盘

状态：2026-09-10，B300 第 1--8 轮完成。这里的“饱和”只指冻结硬件、语义、载体和本轮机会预算下，现有默认知识能够直接提出并执行的高价值候选已经用尽；它不是全局理论最优，也不是最终 H1/T1/T2 迁移结论。

## 主线结论

主线仍是：在相同 FlashKDA 合同和可审计的新增探索机会下，最强 HMMA 与最强 tcgen05/SM100 路径谁更值得部署。多轮实验把三个原来混在一起的问题拆开了：

1. **HMMA 并非未优化。** 现有强化二进制还有 dispatch 空洞。H12、total=8192 时，V64 在 nseq=6 比 V128 快 1.17--1.22x；V32 在 nseq=3 比 V64 快 1.076x，并在偏斜输入上复现；到 nseq=4，V32 反而慢 3.82%。可复用知识是 CTA 波次相关的分段选择，不是“V32 永远更快”。
2. **tcgen05 的收益不能用一个调度数字概括。** H96 s173 对 Cake 的四块资格验证通过，速度比 1.0330x；但 virtual-152、TAIL4、MINIMAX172 和理论最小最大负载 MAX169 都失败。MAX169 比默认排程慢 6.17%，证明最小化最大 chunk load 不是延迟目标。
3. **结构成本确实存在。** V16 preferred-layout 探针去掉循环内 scalar rematerialization 后，比 scalar tcgen05 控制快 22.7%，但仍比同探针 HMMA 慢约 4.7%。这说明旧 IR 正确定位了 layout 成本，也说明仅把转换移出循环还不够；真正剩下的是生产 P3/P4 carrier、rounding 和跨 phase TMEM 生命周期。
4. **继续重复默认候选的价值已经很低。** HMMA L3 lookahead 对 L2 的速度比为 0.99998；tcgen05 的多个零构建排程目标和一个结构探针均已给出反证。下一步若继续，必须引入新的 lane-verified carrier/oracle 或完整 public-full 集成，而不是让 agent 再组合同一批参数。

## 八轮证据链

| 轮次 | HMMA 分支 | tcgen05 分支 | 写回知识 |
|---|---|---|---|
| R1 | H12 mixed6，V64/V128=1.2172x | H64 virtual152=0.5909x | HMMA 有 dispatch 空洞；跨物理一波边界危险 |
| R2 | H12 balanced6，V64/V128=1.1723x | H96 virtual152=0.5830x | V64 收益不只来自长度偏斜；virtual-SM 规则跨 H 重复失败 |
| R3 | balanced8，V64/V128=1.0114x，低于 3% gate | H96 s173/Cake=1.0330x，四块通过 | V64 的实用边界在 nseq 6 与 7/8 之间；s173 成为 exact-profile T 路线 |
| R4 | balanced7，V64/V128=1.0143x，低于 gate | grid147/grid148=1.0034x，低于 0.5% gate | raw CTA 数不是剩余高价值轴 |
| R5 | balanced3，V32/V128=1.3041x | TAIL4/default=0.9624x | 重开 V32；关键 CTA 的任务切换数不能独立优化 |
| R6 | V32/V64=1.0764x，四块通过；balanced4=0.9618x | MINIMAX172/default=0.9757x | V32/V64 crossover 被夹在 nseq 3 与 4；最大负载下降不等于变快 |
| R7 | skew3，V32/V64=1.0774x | 理论 minimax MAX169/default=0.9383x | nseq3 规则通过偏斜反证；关闭 load-minimax 调度目标 |
| R8 | L3/L2=0.99998，bitwise、148-SM B300 | preferred/scalar=1.227x，但 preferred/HMMA=0.955x | L3 精确负例；layout remat 是真成本，但一次性 staging 不足以赢 HMMA |

## 为什么现在停止默认知识搜索

当前可以签发的最强标签是：

```text
default-knowledge_high-VOI_proposal_saturation_under_frozen_B300_existing-carrier_envelope
```

它成立的理由是：两侧各完成一个冻结后的结构候选，编译、身份、正确性和两个 screen block 均留下回执；HMMA 的已知高价值离散轴已经形成边界，tcgen05 的调度子空间已有多种互相独立的反证，preferred-layout 的最小结构探针也已执行。当前没有必须继续开放 proposal 生成的高价值候选族。

它不允许写成 `theoretical_optimum`、`global dual-lane saturation` 或“tcgen05 已经没有空间”。原因是完整 T1 carrier 仍未接通，两个分支也没有在同一 public-full profile 上完成最终四块 H1/T1/T2 资格比较。当前得到的是**默认可执行知识的上限**：agent 再使用相同 source、IR 和经验，大概率只会复述已经失败的 knob 组合。

HMMA L4 lookahead 和 tcgen05 reverse-task 仍可在现有 runner 中执行，但独立审查认为它们缺少新的 stall、cache 或资源证据，属于低价值彩票。它们进入 backlog，不阻止高价值 proposal 饱和；只有新的 phase/NCU 归因显示预期收益可能超过门槛时才重开。

## 重新开启搜索的条件

只有出现下面任一新信息，才值得重开对应队列：

- **HMMA**：新的 fragment/register residency 方案、不同 profile 上的 Phase-1 瓶颈证据，或能解释 L2/L3 中性而提出可证伪的新 phase 干预；
- **tcgen05**：P3/P4 的 192-thread lane mapping oracle、BF16 rounding boundary、可编译的 preferred-layout producer，或 direct packed TMEM carrier；
- **比较层**：冻结共同 profile 与 public-call 边界后，可同时运行 H1、T1、T2 的生产级实现；
- **理论层**：形成包含指令、数据移动、同步和占用的紧致延迟下界，并证明 incumbent 距下界在预定 epsilon 内。

## 后续 agent 指导

后续 agent 不应把“无新 proposal”当性能证明。每个新提案必须先说明它引入了哪条此前不存在的信息：新 carrier、新 oracle、新 profile 反例或新下界。若没有，就直接从默认执行队列降级为 backlog。

Program IR 继续记录实际 route、source/build/binary hash、layout/storage、warp roles、barrier、TMEM 生命周期、物理 SM 与 wave；Experience IR 继续把 observation、mechanism、允许用途和 reopen key 分开。跨 HMMA/tcgen05 只迁移验证方法和物理变量，不迁移“某个值好/坏”的结论。

下一阶段的首要工程任务不是扩大搜索次数，而是实现 tcgen05 的生产 preferred-layout carrier，并在共同 profile 上做 H1/T1/T2 的 public-full 资格比较。该工作需要新 IR 和新 oracle，属于默认知识饱和后的下一研究阶段。
