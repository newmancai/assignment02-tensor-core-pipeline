# C1 正文素材：复现与测量 -> 分析（结论 + 证据）-> 挑战

更新日期：2026-09-10。本文档可直接改写进报告或答辩讲稿；所有数字必须保留 measurement scope。

## 一、建议开场

FlashKDA 是 Kimi Delta Attention 的高性能 forward 实现。课程 pin 的官方 kernel 可以为 B300 编译，但递推矩阵乘在实际 SASS 中仍使用 Ampere 世代 HMMA，而不是 Blackwell 原生 tcgen05。C1 要回答的不是“新指令峰值更高吗”，而是：对同一个带 recurrent state 的 KDA public contract，迁移到 SM100 是否能产生足以抵消协议、布局、验证和维护成本的实际收益。

我们的回答分三层。第一，完整 SM100 路径在已测 B300 profiles 上是否能胜过官方实现；现有 CAKE/official 隔离进程结果给出肯定的存在性证据。第二，在公平新增探索机会下，tcgen05 是否仍胜过优化后的 HMMA，以及收益来自 ISA 还是 TMEM/roles/pipeline；现有八轮闭环只部分回答。第三，是否应在产品上全面迁移；由于缺少真实流量与设备覆盖数据，本文只提出 guarded backend 决策框架。

## 二、复现与测量

### 2.1 复现对象

我们使用课程指定的 FlashKDA commit `1ce47ea` 和 CUTLASS commit `5c149f5`，在 NVIDIA B300 SXM6 AC 上编译 `sm_103a`。固定版本十分重要：当前上游 master 已能面向多个 architecture 编译，但“能为 SM100 编译”不等于“矩阵乘已经换成 tcgen05”。本文所有官方路径结论都绑定 pin commit、构建日志和二进制 hash。

比较合同固定：q/k/v/g/beta、A_log、dt_bias、初始 state、packed boundaries、dtype、output 和 final state。KDA 是递推算子，因此只验证当前 output 会漏掉跨调用状态错误；正式 gate 始终同时检查 output 与 final state。

### 2.2 如何证明“官方仍使用 SM80 MMA”

源码命名不是证据。我们对实际加载的扩展记录 module path 与 SHA-256，再用 `cuobjdump`/SASS 对递推路径计数。归档结果包含 3,640 条静态 HMMA，TCGEN/UTCMMA 为 0。因此准确表述是：

> 在课程 pin 的官方 FlashKDA 二进制中，B300 上执行的递推矩阵乘仍使用 HMMA。

不能扩写成“整个 kernel 都停留在 SM80”。官方路径仍可能使用 TMA、SM100 编译目标和其他跨代能力；被确认的是 MMA atom/递推矩阵乘指令族。

### 2.3 profile 设计

profile 必须同时覆盖：

- H12：Kimi K3 TP8 时的每卡 local heads，暴露小 grid/低并行度；
- H96：完整 head 规模，检验 H12 结论能否跨 head count；
- fixed 与 packed；
- balanced、mixed、skew 和 tail；
- 不同 nseq、总 token 与非零初始 state。

单一 `T=8192` winner 不能代表整个算子。H3 在一个 fixed case 达到 1.0219x，但 H12/H96 几何平均分别只有 0.9055x/0.7581x，正是必须使用矩阵的原因。

### 2.4 correctness 分级

我们区分两类兼容性：

1. **official-bitwise**：不改变 BF16 舍入 DAG 的 dispatch、ValueSlice 或 lookahead 候选，应在适用位置与官方 output/state bitwise 一致。
2. **algorithm-equivalent**：H3 等改变乘加/舍入顺序的路径，应在看结果前冻结共同的 FP64/FLA-style reference、指标和阈值。数学恒等式只能证明实数域等价，不能证明 BF16 bitwise 等价。

H3 的验证不是一次随机对拍：包含 80 个 single-chunk case、32 个多 chunk recurrence scenario、18 个 input-level oracle case，以及 B300 上 fixed/tail/multichunk/packed 的 8/8 output/final-state gate。CPU oracle 不复现 `tanh.approx`、`ex2.approx.ftz` 和 warp reduction 顺序，因此只授权 CUDA 实现，不能替代 GPU correctness。

### 2.5 正式计时协议

headline 使用 public-full GPU span：预分配和 JIT 在计时外，每次 forward 必需的布局转换、workspace 流量与 state copy-back 在计时内。每个 trial 从相同的非零 state slot 开始，避免递推状态随重复次数漂移。

正式比较采用：

- cold-L2 CUPTI；
- 20 次 warmup；
- 每 block 100 个 samples；
- 四个 ABBA/balanced-order blocks；
- bootstrap 95% interval；
- 同一 allocation 内完成一对实现的比较；
- 原始 samples、block medians、profile、seed 和 identity 全部留档。

我们曾发现官方与 H3 两个扩展导出同名 weak/global CUDA symbols；反转 load order 会改变绝对时间，即使 Python 使用 `RTLD_LOCAL` 也不能保证隔离。因此旧的同进程 CUDA-event/CUPTI 只保留为 development diagnostic，正式 magnitude 全部改为 one-implementation-per-worker-process。这一修正说明“代码对了、timer 也跑了”仍不足以构成可复核性能证据。

CAKE 的 source/binary manifest 冻结了测量后观察到的 SM103a cache binaries，并由 route artifact 关联到具体路径；但 per-case timing JSON 中 `binary_sha256` 仍为空，correctness 也来自既有 paired/peer artifact，而不是每个 timing worker 重新执行独立 oracle。这不推翻现有结果，但属于必须主动披露的证据链缺口。下一轮应让每个 worker 在执行后直接写回 loaded binary hash、driver/CUDA 版本及 clock、power、thermal 状态。

### 2.6 profiler 如何支持瓶颈结论

SASS 确认指令身份；NCU 的 LaunchStats、Occupancy、ComputeWorkloadAnalysis、MemoryWorkloadAnalysis、InstructionStats 和 scheduler stalls 用来解释限制因素；CUPTI activity 负责 public call 内的 GPU span 和阶段归因。三者不能互相替代。

代表性 TP8/H12 official recurrence 只有 12 CTA 面对 148 SM，历史 NCU 的 SM/DRAM throughput 为 2.64%/1.24%。两者都远未接近饱和，因此最准确的结论是 parallelism/occupancy/critical-path limited，而不是 compute-bound 或 memory-bound。该结论绑定 H12 profile；H96 必须单独 profile。

## 三、分析：六个讨论点的结论与证据

### 3.1 CHUNK=16 为什么成立，32/64 谁先破

**结论。** 在官方无 rescale 的指数恢复和当前 Neumann/forward-substitution 组织下，CHUNK=16 同时满足数值范围、求解成本和 HMMA K16 形状；机械扩大到 32/64 首先触发的是数值路径，而稳定大块仍是可研究的算法重设。

**纸面证据。** 在既定最坏门控模型中，C32 与 C64 都在第 18 个 token 首次出现 FTZ/overflow。朴素扩大密集级数/求解工作，相对 C16 的每序列模型成本约为 5.33x 和 26.67x。

**实验证据。** FLA safe/block 小形状实验中 C32/C64 仍可保持 finite，说明“大 CHUNK + rescale/block solve”没有被否定；被否定的是只改常量而保留当前数值与求解路径。

**边界。** 不能说 CHUNK>16 本质错误，也不能把 C16 的成功归因于单一原因。

### 3.2 tcgen05 最小 tile 与 CHUNK16 是否匹配，只换指令有没有收益

**结论。** CHUNK16 在 tcgen05 BF16 CTA-group 1 的合法 K 维上可以匹配，问题不在形状非法，而在 TMEM allocation、descriptor、commit/wait、readback、布局和小 grid 的联合协议成本。

**纸面证据。** tcgen05 将 accumulator 放在 TMEM，并改变 issue、同步和读回模型；它不是 HMMA 的无成本 opcode 同义替换。较高峰值只说明潜在算力，不说明小型 recurrent tile 的实际延迟。

**实验证据。** Phase-6 `m128n128k16`、V128/grid12/inner64 的 direct tcgen05 probe 只有 HMMA 的 0.919742x，即使把固定协议摊销 64 次也未胜出。V16 preferred-layout core 的 L0 为 1.615021x，但加入当时 scalar rematerialization 后 L1 为 0.778523x。

**边界。** 这些结果否定两个精确实现，不否定 thin-N、producer-ready layout、跨阶段 TMEM residency 或完整 SM100 dataflow。

### 3.3 chunk 间有递推依赖，并行度从哪里来

**结论。** 同一 sequence 的 state 顺序不能破坏，但独立 head、value-dimension slice、不同 sequence、prepare/consumer roles 和跨阶段流水仍提供并行度。

**证据。** HMMA ValueSlice 将 TP8/H12 K2 grid 从 12 CTA 扩至 96 CTA；在 fixed T8192 上，latency 从 0.7807 ms 降到 0.5698 ms，下降 27.0%。后续双分支搜索进一步发现 ValueSlice 的最佳值依赖 nseq：V32/V64 在 balanced nseq3 为 1.0764x，在 skew nseq3 为 1.0774x，到 balanced nseq4 则为 0.9618x。

**解释。** 可复用规律是 CTA wave 与工作粒度共同决定 crossover，而不是“更多 CTA 永远更快”或“V32 永远最佳”。

**边界。** persistent、多 head CTA、2-CTA 协同仍需同时核算资源、barrier、tail 和 state ownership。

### 3.4 负载是 compute-bound 还是 memory-bound

**结论。** 代表性 H12 recurrence 两者都不是；它主要受可发射独立工作量、occupancy 和串行 critical path 限制。

**证据。** 12 CTA/148 SM，历史 NCU SM throughput 2.64%、DRAM throughput 1.24%。若计算或 DRAM 已饱和，应看到相应 SOL 接近上限；这里两者都很低。

**所需指标。** grid/CTA waves、theoretical/achieved occupancy、active warps、SM 与 DRAM SOL、L2/DRAM bytes、tensor instruction counts、scheduler issue 和 stall breakdown。FLOP/byte 纸面 roofline 只能提出假设，不能替代实际小 grid 和依赖链证据。

**边界。** H96、不同 route 和不同 profile 需要独立 NCU；H12 不能代表所有 KDA。

### 3.5 BF16 state 精度怎样验证

**结论。** 同时验证 output、final state 和连续调用 state handoff，并显式区分 bitwise-compatible 与 algorithm-equivalent。

**证据。** 历史 campaign 的 200 条比较全部 finite，98 条 ValueSlice 对照 bitwise equal，独立参考关系最坏 relative RMSE 为 0.9131%。H3 又覆盖 single-chunk、多 chunk、input-derived oracle 和 B300 8/8 gate；H3 与官方通常不 BF16-equal，因此没有冒充 official-bitwise。

**反例设计。** fixed/tail/packed、强弱衰减、抵消、非零初态、长 recurrence、two-call handoff；执行前冻结阈值，并让官方与候选比较同一个高精度参考。

**边界。** 已测样例不是任意长度形式化证明；CPU 近似也不是 GPU 指令级 oracle。

### 3.6 如果是作者，v2 是否发布 SM100a 专版

**结论。** 发布 guarded SM100 backend，保留 HMMA fallback；不做全局无条件替换。

**支持发布的证据。** 隔离进程 public-full CUPTI 中，CAKE/SM100 相对官方 H0：H12 六形状几何平均 2.4823x，H96 三形状 2.2532x，九个 per-shape bootstrap intervals 全部高于 1。既有 guarded evolution 在 H96/H64 六形状中激活五个、一个回退，对 CAKE deployment geomean 再提高 1.1420x。

**支持保留 fallback 的证据。** Direct V128 为 0.919742x；V16 接入可从 1.6150x 降到 0.7785x；H3 H12/H96 几何平均为 0.9055x/0.7581x；HMMA 的最佳 ValueSlice 又随 profile 改变。

**部署条件。** 只有通过 output/state correctness、actual-route identity、public-full performance、confidence 与 regression gate 的 `(device, profile)` 才激活 SM100 route。

**边界。** 当前结论支持“增加路径”，不支持停止维护 HMMA、覆盖所有 Blackwell、宣称所有 profile 都更快或声称市场回报已闭合。

## 四、挑战：从简单替换到完整迁移

### 4.1 为什么挑战需要多路径

只测一条 tcgen05 路径会把三个问题混在一起：算术核心是否快、数据怎样送入/读出、完整 public call 是否受益。我们用多条路径逐层分解：

```text
Direct V128      -> 只换指令和必要协议
V16 L0/L1        -> 分离 native core 与布局物化
H3 P/W           -> 改变算法/数据流边界，但仍使用 matched HMMA parent
CAKE full-stack  -> 允许 fusion、roles、barriers、TMEM、pipeline、dispatch 协同
双分支 R1-R8     -> 同机会探索 HMMA 与 tcgen05 的局部 frontier
```

### 4.2 路径结果与可说结论

| 路径 | Scope | 结果 | 能说什么 | 不能说什么 |
|---|---|---:|---|---|
| Official | B300 public-full/SASS/NCU | 基线 | 课程 pin 的 recurrence 使用 HMMA | 整个 kernel 都是 SM80 |
| Direct V128 tcgen05 | Phase-6 microprobe | 0.919742x | 裸替换在此条件无收益 | tcgen05 全局无收益 |
| V16 old L0/L1 | thin-N microprobe | 1.6150x / 0.7785x | core 有潜力，接入成本可吞掉收益 | production T1 已完成 |
| H3 P/W | 9-shape isolated public-full | H12 0.9055x；H96 0.7581x | 正确的代数重排仍可能 full-path 失败 | workspace 是唯一已证实主因 |
| CAKE SM100 | 9-shape isolated public-full | H12 2.4823x；H96 2.2532x | 完整 SM100 co-design 值得 | 2.48x 全来自 tcgen05 |
| HMMA V32/V64 | exact public-full profiles | nseq3 1.0764x；nseq4 0.9618x | H1 需要 profile dispatch | H0 已代表最强 HMMA |
| tcgen H96 s173 | exact mixed public-full | 1.0330x over CAKE | 该 profile 上的 route 可资格晋级 | 与 H1 已公平决赛 |
| Load-minimax 169 | exact scheduler screen | 0.9383x | 最大负载最优不等于 latency 最优 | 所有 tcgen scheduling 已穷尽 |
| Round 8 preferred | V16 mechanism probe | 1.227x over scalar；0.955x vs HMMA | rematerialization 是实成本，一次 staging 不够 | 可与 public HMMA/CAKE headline 直接比较 |

注意两个 preferred-layout 结果属于不同历史 probe：旧 L0/L1 的 1.6150x/0.7785x 与 Round 8 的 1.227x/0.955x 不能合并成一条连续 speedup 链。

### 4.3 八轮双分支闭环给出的新知识

HMMA 分支说明旧路径并非“没有优化空间”。V64 在 nseq6 相对 V128 为 1.17--1.22x，但在 nseq7/8 只有约 1.01x；V32 在 nseq3 胜过 V64，nseq4 反而回归。结论是 profile-dependent CTA-wave dispatch。

tcgen05 分支说明调度不能用一个漂亮的理论坐标概括。virtual152 跨 H64/H96 失败，TAIL4、MINIMAX172 和构造性最优 MAX169 均回归；即使最大 load 已达到可证明下界 169，latency 仍慢 6.17%。这关闭的是现有 carrier 上的 load-minimax 目标，而不是 tcgen05 设计空间。

Round 8 给每条 lane 冻结一条结构假设，并记录 proposal、repair、compile、correctness 和 GPU block 配额。HMMA L3/L2 为 0.99998x；tcgen05 preferred/scalar 为 1.227x，但 preferred/HMMA 为 0.955x。因两边 measurement scope 不同，Round 8 的 ledger只证明新增探索机会可审计，不允许直接比较两条 lane 的绝对 latency。

### 4.4 当前停止标签

当前最强诚实标签是：

```text
default-knowledge_high-VOI_proposal_saturation_under_frozen_B300_existing-carrier_envelope
```

含义是：在冻结硬件、语义、现有 carrier 与默认知识下，高价值且可直接执行的新候选已基本用尽。它不是 theoretical optimum、global saturation 或最终 H1/T1/T2 verdict。

重新开启搜索必须带来新信息：

- HMMA：新的 fragment/register residency 或不同 profile 的 Phase-1 NCU 证据；
- tcgen05：P3/P4 lane mapping oracle、BF16 rounding boundary、可编译 preferred-layout producer 或 direct packed TMEM carrier；
- 比较：共同 public-full profiles 上可同时运行的生产级 H1/T0/T1/T2；
- 理论：同时包含指令、数据移动、同步和 occupancy 的紧致 latency lower bound。

## 五、multi-agent 与 IR 在 C1 中的正确位置

MARPE 是研究方法，不是 C1 的替代主线。它复用 PIKE-B 的并行分支、repair 和候选分配，复用 Atrex Kernel Agent 的隔离 GPU 执行、profiling、Git episodes 和 ABBA 验证。本文原创层是：

- HMMA/tcgen05 分支化 Program IR；
- 带 route/profile/phase/contract 作用域的 Experience IR；
- matched-opportunity budget；
- proof-carrying candidate promotion。

Program IR 记录实际执行对象，包括 Python executable/version、`sys.path`、module file/hash、`.so`、CUDA 源码和头文件、target、actual route 与 fallback。Experience IR 将 observation、mechanism、permitted use、invalidation key 和 reopen condition 分开。这样，一个 profile 的失败只会降低对应候选的优先级，不会被写成另一条架构路线的全局禁令。

当前实验只是窄宽度双分支闭环，没有 matched single-agent ablation。因此可说“该框架帮助我们保留并审计两条路线的证据”，不能说“multi-agent 比 single agent 搜索更快或性能更高”。

## 六、从技术证据到产品决策

### 6.1 需要补充的产品变量

完整迁移价值至少需要：

- 每种设备的请求/卡时比例 `p(device)`；
- 每种 H、序列长度、packed/tail 和并发 profile 的流量权重 `p(workload|device)`；
- KDA 在端到端 prefill/decode/serving 时间中的占比；
- kernel latency、吞吐、p95/p99、功耗/能耗与显存变化；
- 编译、验证、dispatcher、双路径维护、回归响应和工具链升级成本；
- SM100 路径可服务的产品生命周期。

当前项目没有这些分布，不能填入推测值。正确做法是把它们列为下一阶段 telemetry schema。

### 6.2 当前可执行的部署建议

1. 保留 HMMA 作为跨代兼容和未覆盖 profile 的 fallback。
2. 对已通过 public-full qualification 的 B300 profile 激活 SM100 route。
3. dispatcher 的 key 至少包含 device capability、H、sequence class、total tokens、state dtype 与 route availability。
4. 未知 profile 默认回退；收集 shadow timing 与 correctness receipt 后再晋级。
5. 随 SM100 流量权重、覆盖率和维护成本变化，周期性重算部署价值。

## 七、可直接用于答辩的结论

### 7.1 一句话

> 我们证明了课程 pin 的 FlashKDA 在 B300 上仍以 HMMA 执行递推矩阵乘；机械换 tcgen05 不值得，但完整 SM100 协同路径在九个已测 profile 上值得以 guarded backend 发布，同时保留 HMMA fallback。

### 7.2 三十秒版本

> 官方停在 HMMA 不是简单落后：直接 V128 tcgen05 只有 0.92x，正确的 H3 数据流重排跨 profile 也失败，说明协议、布局和串行路径成本足以吞掉新指令收益。但 CAKE 类 full-stack SM100 路径在隔离进程 public-full 测量中，H12/H96 几何平均达到 2.48x/2.25x，九个 profile 全部正收益。因此 v2 值得增加 SM100 guarded route。与此同时，HMMA 在公平新增探索中还能通过 profile-dependent ValueSlice 获益，所以我们尚不能把 CAKE 收益归因于 tcgen05，也不能停掉 HMMA。

### 7.3 最终书面结论

> C1 的答案不是“SM80 或 SM100 二选一”。在当前 B300 证据范围内，instruction-only 迁移失败而 full-stack SM100 co-design 成功，说明迁移价值来自新 ISA 与数据驻留、角色分工、同步、流水和 dispatch 的联合组织。工程上应新增通过资格验证的 SM100 专用路径，并保留 HMMA 作为兼容和未覆盖 profile 的 fallback。要进一步回答 tcgen05 的纯选择权价值，需要在共同 public-full profiles 上完成 H1/T0/T1；要回答全面产品迁移，还需要真实设备/流量权重、端到端收益、能耗和维护成本。

## 八、最容易被追问的边界

1. **为什么 CAKE 不等于 tcgen05 的收益？** 因为它同时改变 fusion、work partition、roles、TMEM、barriers、pipeline、layout 和 dispatch。
2. **为什么 H0 不能代表 HMMA 上限？** 因为 V64/V32 dispatch 已在多个 profile 改进 H0/旧 incumbent。
3. **为什么一个 fixed winner 不能留下 H3？** 因为预注册 gate 看跨形状 geomean 和最大 regression，H3 在 H12/H96 整体失败。
4. **为什么只测 output 不够？** 因为 final state 决定下一次 recurrent call。
5. **为什么不是 compute-bound/memory-bound？** 因为代表 H12 的 SM 与 DRAM 吞吐都很低，关键限制是 grid、occupancy 和依赖链。
6. **为什么停止继续 agent 搜索？** 停止的是冻结 carrier 下的默认高价值候选，不是理论空间；下一步需要新 carrier/oracle，而不是继续排列旧 knobs。
7. **multi-agent 是否已证明优于单 agent？** 没有；当前只证明证据组织和双 lane 审计流程可工作。
