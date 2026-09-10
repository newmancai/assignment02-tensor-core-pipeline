# C1 大作业完整大纲：FlashKDA 从 SM80 MMA 迁移到 SM100 是否值得

更新日期：2026-09-10。

## 0. 范围锁定与总判断

课程题目只有一条：

> **C1：FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得。**

报告严格采用课程要求的三段式主线：

```text
复现与测量 -> 分析（六个讨论点逐项给“结论 + 证据”）-> 挑战
```

外部意见中“固定 B300、公平探索 HMMA/tcgen05 两条路径”是 C1 的一个重要因果子问题，但不能覆盖已有的全部证据，也不能替换总题目。当前项目已经得到两类不同结论：

1. **工程处理效应已经有正证据。** 在一张 B300/SM103a 上，完整 CAKE/SM100 路径相对官方 H0 在九个 H12/H96 public-full profiles 上全部获益，H12 与 H96 几何平均分别为 2.4823x 和 2.2532x。这说明“为已测 B300 profile 增加完整 SM100 路径”在技术上值得。
2. **tcgen05 的独立选择权价值仍未闭合。** 八轮同机会双分支实验改进了 HMMA，也找到 tcgen05 局部正负证据，但尚无同一 public-full profile 上的生产级 H1/T0/T1 四块资格比较。因此不能把 CAKE 总收益归因于 tcgen05，也不能声称同预算最优 tcgen05 已经超过同预算最优 HMMA。
3. **产品级全面迁移仍未回答。** 当前没有真实流量权重、SM100 部署比例、端到端 serving 占比、能耗和长期维护成本数据。产品结论只能写成 guarded route 建议，不能写成停掉 HMMA 或全面替换。

答辩开场应把“值得”拆成三级，避免把不同问题混在一起：

| 决策层 | 精确问题 | 当前状态 |
|---|---|---|
| 工程可行性 | 已测 B300 profile 上，完整 SM100 路径能否胜过官方路径？ | 已回答：能，九个 profile 全部正收益 |
| 架构因果性 | 公平新增预算下，最优 tcgen05 是否胜过最优 HMMA；收益来自哪里？ | 未闭合：缺共同 profile 的 H1/T0/T1 |
| 产品部署 | 覆盖真实流量、设备、成本和风险后，是否全面迁移？ | 未研究：只能提出测量框架与 guarded 策略 |

## 1. 统一比较坐标：H0/H1/T0/T1/T2

所有图表和口头表述统一使用以下五个锚点：

```text
H0 = 课程 pin commit 的官方 HMMA 路径
H1 = 在冻结合同和新增预算下发现的最强 HMMA 路径
T0 = 与 H1 尽量结构匹配的 tcgen05 + 必要 TMEM/同步协议
T1 = 在相同新增预算下，允许 tcgen05 特有驻留、角色和流水后的最强路径
T2 = CAKE/evolution 类最强 guarded SM100 全栈部署路径
```

需要分别报告五个量，不能用一个 speedup 混写：

```text
HMMA 仍可优化多少                 = latency(H0) / latency(H1)
直接 ISA + 必要协议的联合效应       = latency(H1) / latency(T0)
tcgen05 特有组织的附加收益          = latency(T0) / latency(T1)
最优 tcgen05 相对最优 HMMA          = latency(H1) / latency(T1)
当前官方到最终部署路径的工程总收益   = latency(H0) / latency(T2)
```

其中 H0/T2 是工程处理比较；H1/T0/T1 是因果拆分。若 T0 无法在不改变物理结构的情况下构造，必须报告“不可单独识别”，并把结果称为 `tcgen05 + 必要协议包` 联合效应。

## 2. 完整报告结构与篇幅建议

下面的结构适合 8 页单栏论文，也能直接映射到 10 分钟答辩。

### 2.1 摘要

用四句话完成：

1. 题目：官方 FlashKDA 在 B300 上仍以 HMMA 执行递推矩阵乘，问题是迁移到 SM100 是否值得。
2. 方法：固定 public semantics，用 SASS/NCU/CUPTI、output/final-state correctness、隔离进程 ABBA 和多路径挑战建立证据链。
3. 结果：直接换指令和 H3 重排失败，完整 SM100 路径在九个 profile 上胜出；双分支搜索同时证明 HMMA 仍有 profile-dependent 优化空间。
4. 边界：值得增加 guarded SM100 route；纯 tcgen05 边际价值和全面产品迁移仍需 H1/T0/T1 与真实流量验证。

### 2.2 第 1 节：问题、背景与贡献

回答三件事：

- KDA 是带 recurrent state 的算子，`output` 正确不足以保证下一次调用正确，必须同时验证 final state。
- 官方 deep dive 将 CHUNK=16 归因于 BF16 数值范围、16x16 求解成本与 SM80 MMA 的自然形状；这为 C1 提供了可量化的设计假设。
- 本作业贡献不是“写了一个更快 kernel”这一句，而是建立了从官方指令身份、瓶颈分析、多条挑战路径到部署建议的完整证据链。

建议列出四项贡献：

1. 在 B300 上复现课程 pin commit，并以 SASS 证明递推路径确实使用 HMMA。
2. 逐项回答 CHUNK、tile、并行度、瓶颈、精度和 v2 发布决策六个讨论点。
3. 实现并验证 instruction-only、布局、算法重排和 full-stack 多条迁移路径，保留负结果。
4. 用证据链接的双层 IR 和同机会双分支闭环管理候选、失败作用域和结论边界。

### 2.3 第 2 节：复现与测量

#### 2.3.1 冻结对象

- FlashKDA pin commit：`1ce47ea`；CUTLASS：`5c149f5`。
- 目标设备：NVIDIA B300 SXM6 AC，physical SM count 148，编译目标 `sm_103a`。
- 语义合同：相同 q/k/v/g/beta、A_log、dt_bias、初态、packed boundary、output 与 final state。
- profile 轴：H12/H96，fixed/packed，balanced/mixed/skew/tail，总 token 与序列数量。

#### 2.3.2 三种测量分别回答什么

| 工具/范围 | 回答的问题 | 不能替代什么 |
|---|---|---|
| `cuobjdump`/SASS | 实际二进制包含并执行哪一代 tensor-core 指令 | 不能说明哪个实现更快 |
| NCU | grid、资源、occupancy、SM/DRAM 吞吐和 stall，定位瓶颈 | 不能代替 public-full 部署延迟 |
| CUPTI activity | public call 内 kernel、memcpy、memset 的 GPU span 与阶段归因 | 不自动证明数值正确或路线身份 |

#### 2.3.3 正式性能协议

- public-full scope：JIT 和一次性 allocation 在计时外，每次调用必须发生的布局、workspace、state copy-back 在计时内。
- 每次调用使用相同非零初态，采用轮转 state slots，避免递推状态在重复测量中漂移。
- one-implementation-per-worker-process，避免官方/H3 扩展的弱/global CUDA symbol interposition。
- cold-L2 CUPTI；20 warmup；每 block 100 samples；四个 balanced-order blocks。
- 保存原始 samples、block medians、bootstrap 95% interval、源码/二进制 hash、模块路径和实际 observed route。
- screen 与 qualification 分开；局部 microprobe 不得进入 public-full headline。

#### 2.3.4 正确性协议

- official-bitwise 路径：检查 output 与 final state 的 bitwise identity。
- 改变 BF16 舍入 DAG 的 algorithm-equivalent 路径：执行前冻结 FP64/FLA-style reference、误差指标和阈值，再做 CPU 反例集与 B300 GPU gate。
- packed、tail、非零初态、强/弱衰减、抵消和 two-call handoff 均要进入反例集。

#### 2.3.5 复现阶段必须呈现的结果

1. 官方 recurrence SASS：3,640 条静态 HMMA，TCGEN/UTCMMA 为 0。
2. H12/TP8 代表 profile：12 CTA 对 148 SM；历史 NCU SM/DRAM throughput 为 2.64%/1.24%。
3. 正式测量隔离修正：同进程数据降级为 diagnostic，headline 改用隔离进程 CUPTI。

### 2.4 第 3 节：六个讨论点的“结论 + 证据”

这一节不能只列问题。每个小节固定使用四行结构：

```text
结论：一句话回答。
纸面推算：公式、边界或复杂度。
实验证据：profile、实现、数值和 measurement scope。
适用边界：该结果不能外推到什么。
```

六个小节内容见 `C1_REPRODUCTION_ANALYSIS_CHALLENGE_CONTENT_20260910.md`，核心结论为：

1. CHUNK=32/64 的朴素扩大首先受当前指数恢复数值范围约束，稳定大块需要 rescale/block solve，属于算法重设。
2. CHUNK16 在 tcgen05 形状上合法，但 TMEM、descriptor、commit/wait、readback 和布局成本使直接 V128 替换变慢。
3. chunk 内外依赖不等于没有并行度；head、value slice、sequence 和 producer/consumer roles 仍可并行，H12 ValueSlice 已给出 27.0% 延迟下降。
4. 代表性 H12 不是 compute-bound 或 memory-bound，而是 parallelism/occupancy/critical-path limited；H96 必须单独判定。
5. BF16 state 必须同时验证 output、final state 和连续调用；数学等价不等于 bitwise 等价。
6. v2 应发布 guarded SM100 backend 并保留 HMMA fallback，不应无条件替换。

### 2.5 第 4 节：挑战设计

挑战不是一条单线优化，而是一组相互证伪的路径：

| 路径 | 核心问题 | 结果类型 |
|---|---|---|
| Direct V128 tcgen05 | 只换 ISA 是否有收益？ | 负：0.919742x |
| V16 thin-N/layout | tcgen core 的潜力是否被接入成本吞掉？ | 分层：L0 1.6150x，L1 0.7785x |
| H3 P/W | 代数/数据流重排能否缩短串行 K2？ | 负：H12 0.9055x，H96 0.7581x |
| CAKE full-stack | 重做驻留、roles、barriers、pipeline、layout、dispatch 是否存在大收益？ | 正：H12 2.4823x，H96 2.2532x |
| HMMA/tcgen05 双分支 | 同机会下，两条路径的局部 frontier 和失败边界是什么？ | 部分闭合，最终 H1/T1 未闭合 |

挑战路径要按“问题 -> 候选 -> gate -> 结果 -> 下一决策”叙述。不要按实验执行时间流水账叙述。

### 2.6 第 5 节：结果综合

建议将结果分为三张表，而不是一个大表：

1. **官方复现表**：版本、设备、SASS、NCU、正确性、public latency。
2. **迁移路径表**：Direct、V16、H3、CAKE，各自 scope、speedup、correctness、结论边界。
3. **八轮双分支表**：每轮 HMMA/tcgen05 候选、结果、写回经验、是否晋级。

核心叙事顺序：

```text
官方确实使用 HMMA
-> 直接 tcgen05 更慢，说明问题不只是 opcode
-> V16 证明 native compute 有潜力，但 layout/carrier 成本决定成败
-> H3 证明数学正确的重排也可能在 full path 失败
-> CAKE 证明 full-stack SM100 路径存在显著工程价值
-> HMMA 双分支结果证明公平对照不能停在 H0
-> 因此 guarded T2 值得，纯 H1/T1 选择权价值仍开放
```

### 2.7 第 6 节：multi-agent 与双层 IR

这一节只回答“我们怎样更可靠地探索和记录证据”，篇幅控制在全文约 10%。

- MARPE 复用 PIKE-B 的并行分支、repair 和候选分配。
- 复用 Atrex Kernel Agent 的隔离 GPU 执行、profiling、Git episode 和 ABBA 晋级。
- 原创边界是 HMMA/tcgen05 分支化 Program IR、带作用域 Experience IR、matched-opportunity budget 和 proof-carrying promotion。
- 当前只是窄宽度双分支闭环；没有 matched single-agent ablation，因此不声称 multi-agent 比单 agent 更快。

Program IR 回答“实际执行了什么”；Experience IR 回答“证据能在什么范围复用”。Python、头文件、CUDA 源码、`.so`、route fallback 和 device target 都属于执行身份，不能放在无关附录里。

### 2.8 第 7 节：讨论、威胁与产品边界

必须区分四类有效性：

| 威胁 | 当前处理 | 剩余缺口 |
|---|---|---|
| 内部有效性 | 隔离进程、route receipt、state reset、ABBA、source/cache hash | CAKE timing 与 correctness artifact 不是同一轮独立 oracle；per-case timing JSON 未直接记录实际加载 binary hash |
| 构念有效性 | public-full 与 microprobe 分开；H0/H1/T0/T1/T2 | T0 可能无法成为单变量反事实 |
| 外部有效性 | H12/H96、fixed/packed/tail | 仅一张 B300、无跨工具链/多节点复现；应在下一轮统一记录 driver、CUDA、clock/power/thermal 状态 |
| 产品有效性 | 提出 guarded dispatch | 无流量权重、SM100 部署比例、能耗、维护成本 |

产品决策不应只用 `性能收益 × workload 比例 × 设备比例 - 成本` 的口号。建议写成可测量的期望价值：

\[
V = \sum_{d,w} p(d,w)\,I_{\mathrm{qualified}}(d,w)\,
    [C_{H}(d,w)-C_{D}(d,w)]
    - C_{\mathrm{impl}}-C_{\mathrm{verify}}-C_{\mathrm{maint}}-R_{\mathrm{regression}}.
\]

其中 `d` 是设备，`w` 是 workload/profile，`C_H` 是 HMMA fallback 的成本，`C_D` 是 dispatcher 所选 guarded route 的成本；成本可以按 GPU 时间、费用、能耗或 SLA 罚损定义。`I_qualified` 防止把未资格验证的局部 winner 算进收益。当前没有 `p(d,w)` 和成本项，所以只给出框架，不代入虚构数字。

若 KDA 只占端到端时间比例 `f`，kernel speedup 为 `s`，系统上限还受 Amdahl 关系约束：

\[
S_{\mathrm{system}} = \frac{1}{(1-f)+f/s}.
\]

这解释了为什么 kernel 级大幅收益不能直接写成模型或服务同等倍数收益。

### 2.9 第 8 节：结论

建议最终结论：

> 在课程 pin 的官方 FlashKDA 上，B300 实际执行的递推矩阵乘仍为 HMMA。直接替换为 tcgen05 和一个正确的 H3 数据流重排均未通过 full-cost gate，说明官方停留在 HMMA 有合理的局部工程原因；但 CAKE 类完整 SM100 协同路径在九个已测 H12/H96 public-full profiles 上全部超过官方，因此为这些 profile 增加 guarded SM100 backend 值得。HMMA 的后续优化又证明 H0 不是公平因果基线；在生产级共同 profile H1/T0/T1 完成前，不能把 CAKE 总收益写成 tcgen05 的纯边际收益，也不能推出全面停用 HMMA。

## 3. 10 分钟答辩大纲

| 时间 | 页 | 内容 | 必须出现的证据 |
|---:|---:|---|---|
| 0:00--0:40 | 1 | C1 问题与三级“值得” | 工程、因果、产品三层 |
| 0:40--1:35 | 2 | FlashKDA/KDA 与官方设计 | CHUNK16、K1/K2、state |
| 1:35--2:30 | 3 | 复现与测量 | pin、B300、SASS HMMA、public-full |
| 2:30--4:20 | 4 | 六个讨论点 | 每点一行“结论 + 数字证据” |
| 4:20--5:20 | 5 | 为什么不能只换指令 | V128、V16 L0/L1 |
| 5:20--6:30 | 6 | 多路径挑战 | H3 负、CAKE 正 |
| 6:30--7:35 | 7 | 公平双分支闭环 | HMMA V32/V64、tcgen s173、minimax 反例 |
| 7:35--8:30 | 8 | 结论边界 | T2 值得；H1/T1、产品仍开放 |
| 8:30--9:20 | 9 | 部署建议 | guarded route + HMMA fallback |
| 9:20--10:00 | 10 | 贡献与复现入口 | 代码、证据、IR；框架来源边界 |

答辩中 agent 架构不宜占用一整页结果时间。最合适的位置是第 7 或第 10 页，解释为何负结果没有被误写成全局禁令，以及为何路线 fallback、Python import 和二进制 identity 会被抓住。

## 4. 推荐图表

1. **主线图**：H0/H1 与 T0/T1/T2 两条泳道，中间标出相同 semantics、budget、public-full 和 qualification gate。
2. **证据链图**：源码 pin -> build/hash -> SASS/NCU -> correctness -> CUPTI ABBA -> claim scope。
3. **多路径结果图**：Direct V128、V16 L0/L1、H3 H12/H96、CAKE H12/H96，以 1.0 为基线，并用颜色区分 microprobe 与 public-full。
4. **六问总结表**：每行只有结论、最关键数字、边界。
5. **四层迁移决策图**：instruction/kernel、operator/workload、system/maintenance、product/market；当前证据颜色只覆盖前三层的一部分。

图中不得把不同 scope 的 1.615x、1.227x、2.482x 放在同一根无注释柱状图里。至少标出 `microprobe`、`mechanism probe`、`public-full`。

## 5. 交付物清单

### 5.1 代码

- 官方 pin 与构建脚本；
- H3 CUDA 实现和 correctness runner；
- tcgen05 probes 与 Round 8 preferred-layout bridge；
- isolated-process CUPTI runner；
- execution identity、budget ledger 和 IR 工具。

### 5.2 报告

- 8 页 CAKE 单栏格式论文；
- 本大纲；
- 三段式内容稿；
- 主线交接和 evidence index。

### 5.3 答辩

- 10 页左右、10 分钟；
- 每个关键结论能指向一个 artifact；
- 准备回答“为什么 CAKE 不是纯 tcgen05”“为什么 H0 不是公平基线”“为什么一个 winner 不能代表全部 profile”“为什么只测 output 不够”。

## 6. 证据索引

- 课程任务：[`../TASK.md`](../TASK.md)
- 总交接：[`SM100_MAINLINE_DELIVERY.md`](SM100_MAINLINE_DELIVERY.md)
- 框架范围修正：[`FRAMEWORK_SCOPE_CORRECTION_20260910.md`](FRAMEWORK_SCOPE_CORRECTION_20260910.md)
- 当前主线状态：[`experiments/sm100_open_round1/CURRENT_MAINLINE_STATUS.md`](experiments/sm100_open_round1/CURRENT_MAINLINE_STATUS.md)
- 多路径复盘：[`experiments/sm100_open_round1/ROUND1_REVIEW.md`](experiments/sm100_open_round1/ROUND1_REVIEW.md)
- tcgen05 选择权价值：[`experiments/sm100_open_round1/TCGEN_OPTION_VALUE.md`](experiments/sm100_open_round1/TCGEN_OPTION_VALUE.md)
- 八轮默认知识饱和复盘：[`experiments/sm100_open_round1/dual_ir_agent_rounds/DEFAULT_KNOWLEDGE_SATURATION_REVIEW_20260910.md`](experiments/sm100_open_round1/dual_ir_agent_rounds/DEFAULT_KNOWLEDGE_SATURATION_REVIEW_20260910.md)
- CAKE/official 正式矩阵：[`experiments/sm100_open_round1/evidence/b300_stage12_cake_official_isolated_pair.json`](experiments/sm100_open_round1/evidence/b300_stage12_cake_official_isolated_pair.json)
- H3/official 正式矩阵：[`experiments/sm100_open_round1/evidence/h3_b300_isolated_process_cupti.json`](experiments/sm100_open_round1/evidence/h3_b300_isolated_process_cupti.json)
- Round 8 公平预算：[`experiments/sm100_open_round1/dual_ir_agent_rounds/round8_budget_receipt.json`](experiments/sm100_open_round1/dual_ir_agent_rounds/round8_budget_receipt.json)
- Round 8 结果：[`experiments/sm100_open_round1/dual_ir_agent_rounds/round8_writeback.json`](experiments/sm100_open_round1/dual_ir_agent_rounds/round8_writeback.json)

## 7. 公开资料边界

- [FlashKDA 官方仓库](https://github.com/MoonshotAI/FlashKDA)说明 public API、output/final-state 与 packed 输入合同；课程结论应以 pin `1ce47ea` 而非浮动 master 为准。
- [FlashKDA v1 deep dive](https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md)给出 CHUNK16、K1/K2 split、BF16 state 和 SM80-only MMA 的作者动机；在线 master 的文字可能变化，课程分析以本地 pin `1ce47ea` 的版本为准。
- [NVIDIA tcgen05 guide](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)说明 tcgen05 与 TMEM、单线程 issue 和 CTA-group 协作的编程模型；它证明能力，不证明 FlashKDA 上的收益。
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)说明 `cuobjdump --dump-sass` 的二进制反汇编用途。
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)说明 LaunchStats、Occupancy、ComputeWorkloadAnalysis 和 InstructionStats 的解释边界。
- [CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html)提供 kernel、memcpy、memset 与相关时间戳/metadata 记录。
- [CAKE](https://arxiv.org/abs/2608.12629)提供完整 compiler-agent co-design 和 FlashKDA 全栈 SM100 路径的外部背景；本作业的 B300 数字仍以本地 artifact 为准。
- [PIKE](https://github.com/pike-project/pike)与 [Atrex Kernel Agent](https://github.com/alibaba/atrex-kernel-agent)分别是 MARPE 搜索平面和隔离执行/验证能力的上游来源。
