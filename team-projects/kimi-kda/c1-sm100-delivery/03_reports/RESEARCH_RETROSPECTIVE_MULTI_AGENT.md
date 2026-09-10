# Proof-Carrying Multi-Agent Runtime Profile Evolution

## 执行摘要

前沿工程系统确实正在从单条 Agent 轨迹转向多路径、角色化和种群式搜索，但公开证据并不支持“只要增加 Agent 数量就会自然涌现性能”这一宽泛判断。真正产生稳定收益的共同结构是：并行探索、专业化上下文、可执行反馈、确定性裁决、受控共享记忆，以及与搜索过程隔离的最终验证。Google AI co-scientist 使用 Generation、Reflection、Ranking、Evolution、Proximity 和 Meta-review 等专业角色迭代假设；AlphaEvolve 使用快慢模型组合、自动 evaluator 和程序种群；Meta KernelAgent 使用 Profiler、Judge、Analyzer、Orchestrator、Optimization Worker 和 Benchmark 等角色。它们的共同点不是“聊天更多”，而是扩大搜索宽度，并把裁决交给外部事实。[^1][^2][^3]

FlashKDA Runtime Profile Evolution 已具备一条比多数通用多 Agent 系统更严格的证据链：hardware-explicit typed IR、命名 verifier 规则、真实 B300 profile、public API 同口径测量、output/final-state correctness、bootstrap 区间、held-out profile、fallback 和完整 GPU 账本。因此，多 Agent 最适合接入 typed mutation 的提案、批评、组合和假设更新层；verifier、compiler、B300 executor 和 activation gate 继续保持确定性。由此形成的系统可称为 **MARPE：Multi-Agent Runtime Profile Evolution**。

本轮已经落地第一段 MARPE 纵向切片：Agent 提案具有角色和证据来源，结构相同的 typed schedule 在验证后合并，非法候选无法因多 Agent 共识而绕过 KIR，激活仍严格要求正确性、同 scope 测量、正的置信下界和 fallback。同时新增 runtime-profile receipt，显式计算 total/max chunks、有效序列并行度、prepare CTA、SM waves 和尾波利用率。Stage 4/5 的六个受影响 B300 观测中，`effective_sequence_parallelism` 与 log-speedup 的描述性 Pearson 相关约为 0.81；这条线索随后通过 Stage 6 的固定工作量实验得到更强支持，但仍不是因果归因或新部署规则。

## 1. 从 Stage 2 到 Stage 5：已经学到什么

### 1.1 成功不是一次 kernel 生成，而是证据状态迁移

项目最初从真实冻结 FlashKDA BF16 fused kernel 恢复物理 schedule：32 个 warp、117,376 B shared memory、240 TMEM columns、5-stage ring，以及 Q/K 边上 `elect_one_per_warp` 与 `CompletionJoin` 的真实完成协议。29 个 Blackwell candidate 随后被转换成 29 个唯一 typed semantic fingerprints。这个阶段建立了一个重要边界：源码能够被 Agent 改写，不等于改写可解释、可验证或可部署。

Stage 3 打通第一个 public `recurrent_kda(..., backend="evolution")` 闭环。六个形状全部保持输出和最终状态语义，五个形状通过置信下界 gate，`backend="auto"` 保持不变并保留 CAKE fallback。由此，候选状态从 `Expressible → Verified → Calibrated` 继续推进到 `Dispatchable`。

Stage 4 将工作负载绑定到 Kimi-K3 TP8 的真实 H12/D128/BF16 合约。route counterfactual 证明已有 dispatcher 在六个形状上已经选择最佳可用 route family；优化机会位于选中 route 内部。搜索最终发现，H12 BT16 prepare-chain 且 total chunks 大于 128 时，将每个 prepare CTA 的 chunk 数从 4 调为 9，可在固定 8192 和官方 packed mixed 上分别获得 1.0369x 和 1.0807x。一次非法短 carrier 扩展造成的 memory fault 被保留为失败证据，并收缩了合法搜索空间。

Stage 5 在首次运行前冻结六个新形状。两个 64/128 chunk 边界 control 保持不变；四个受影响形状的 bootstrap 95% 下界全部超过 1，speedup 为 1.0040x 到 1.0371x。它支持 `total_chunks > 128` 这一激活阈值，却没有证明 9 是跨架构最优值，也没有解释为何 total chunks 相近时 packed profile 的收益变化明显。

### 1.2 最有价值的失败与负结果

三个负结果比继续扩大候选数量更有价值。

第一，Stage 2 的 prepared-launch speedup 不能直接当作 production API speedup；scope parity 后必须重测。第二，Stage 4 强制不同 route family 后没有发现更优路由，说明搜索应下沉到 intra-route work assignment。第三，wave quantization 在已测范围内不是主要决定因素：cpc=9 的多数 grid 低于量化阈值，而启用与关闭量化的 exploratory 中位数接近。它们共同提示 MARPE 的 Agent 应负责提出互相竞争的可证伪假设，而不是围绕最新赢家重复局部修饰。

## 2. NVIDIA 与 GPU 调度上游：为什么物理变量必须进入 Agent 接口

CUDA 的执行模型把 thread block 动态分派到能够满足资源约束的 SM；因此 grid 大小、block 资源、可驻留 block 数和最后一波的填充程度共同决定可见利用率，而不是“总 FLOPs”单独决定性能。Blackwell 调优指南进一步明确，寄存器、shared memory、每 SM 活跃 warp/block 上限以及 cluster occupancy 都会限制并发；使用 Thread Block Cluster 时应通过 occupancy API 计算实际活跃 cluster 数。对 MARPE 而言，这意味着 runtime profile 至少必须暴露 `scheduled_ctas / sm_count`、tail utilization、register/shared-memory footprint 和 cluster shape，Agent 不能只看到张量尺寸。[^11][^12]

CUTLASS 的调度谱系提供了更直接的参照：非持久调度将 grid 直接映射到逻辑 tile；静态持久调度将 launch 压缩到约一波 worker，并用轻量算术领取后续 tile；Blackwell CLC 动态持久调度再引入硬件辅助的 work queue。Stream-K 则从“按输出 tile 分解”转向“按总工作量平均分解”，专门处理 tile 数不能整齐量化到物理处理单元时的利用率损失。它们共同说明 `chunks_per_prepare_cta` 不是随意超参数，而是 work decomposition：它同时改变 worker 数、每 worker 工作量、wave 数和摊销成本。[^13][^14]

性能归因必须区分 wall-clock、GPU span 和单 kernel activity。CUPTI Activity API 提供 kernel 的提交与开始/结束时间戳；Nsight Compute 则提供 launch、occupancy、Speed-of-Light、memory workload 和 instruction 等计数器。本文的 ABBA 资格测试使用 CUPTI span 作为部署口径，Stage 7 进一步在同一次 public 调用内按两个物理 launch 做 activity attribution；NCU 只负责解释 prepare 内部的资源变化，不能替代 public-path 性能结论。[^15][^16]

## 3. 编译器与调优上游：从搜索空间到可迁移知识

TVM、Ansor、TensorIR 和 MetaSchedule 分别建立了端到端张量编译、分层搜索空间与学习成本模型、tensorized program IR、以及可组合的概率程序搜索空间。它们证明了搜索的关键不只是 optimizer，而是“哪些程序变换可表示、如何组合、如何在硬件上测量”。Triton 进一步以 tile 为中心降低了 GPU kernel 编程门槛；FlashAttention-3 展示了新硬件上 warp specialization、异步 Tensor Core/TMA 重叠和低精度路径必须共同设计。MARPE 的位置不是替代这些系统，而是把部署后观测到的 profile、物理 receipt、失败规则和阶段归因重新送回一个受类型约束的搜索循环。[^17][^18][^19][^20][^21][^22]

KDA 的算法谱系同样解释了为何 `max_chunks` 是关键变量。Gated DeltaNet 与 Kimi Linear 都包含可并行的 chunk 内计算和必须保持顺序的状态递推；总 token/chunk 数描述总工作，最长序列 chunk 数描述 recurrence critical path。因而对 segmented recurrence、linear attention、scan 和状态空间模型，可迁移的 profile 不是某个 B300 阈值，而是 `(parallel work, longest dependency chain, hardware waves, resource footprint)` 这一分解。[^23][^24]

## 4. 多 Agent 前沿：证据支持什么

| 系统 | 协作机制 | 公开结果 | 对 MARPE 的直接启示 |
|---|---|---|---|
| Agent-System Interface | DSL 搜索空间 + AutoGuide 反馈 | 9 个并行程序 benchmark，10 次迭代优于 OpenTuner 1000 次，最高超专家 mapper 1.34x | typed IR 是 Agent-System Interface 的更强形式；反馈应从原始计数器变成结构化 receipt[^4] |
| MASAI | 5 个任务专用子 Agent、独立定位/修复/测试/排序 | SWE-bench Lite 28.33%；多样候选必须依赖测试输出排序 | 多样性本身不足，必须由外部测试和验证器裁决[^5] |
| Google AI co-scientist | Supervisor + Generation/Reflection/Ranking/Evolution 等角色 | 专家小样本评审和实验验证显示研究辅助潜力 | 对应 Profile Analyst、Critic、Ranker、Synthesizer；科学假设需要 disconfirming test[^1] |
| AlphaEvolve | 快慢模型 ensemble、自动 evaluator、程序数据库与进化选择 | Google 报告 FlashAttention 最多 32.5% 提升，并有基础设施部署案例 | 候选数据库、自动 evaluator 和组合 lineage 值得借鉴；它不是经典角色社会[^2] |
| Astra | Testing/Profiling/Planning/Coding 专业 Agent | 3 个 SGLang kernel 平均 1.32x；同设置单 Agent 为 1.08x | 最直接的 multi-agent kernel 前驱；MARPE 增加 typed delta 与 deployment certificate[^6] |
| CudaForge | Coder/Judge + NCU hardware feedback | 报告 97.6% correctness、相对 PyTorch 1.68x，覆盖多种 GPU | hardware-grounded Judge 有效，但 benchmark-to-production 外推仍需谨慎[^7] |
| Meta KernelFalcon | 分层委派、确定性 Python control、隔离 worker、持久 artifact | KernelBench L1-L3 报告 100% correctness | 编排、超时和成功条件应是普通程序，不应由 LLM 自由决定[^8] |
| Meta KernelAgent | Profile → Diagnose → Prescribe → Orchestrate → Explore → Measure；top-K 与共享记忆 | H100 KernelBench L1 相对默认 `torch.compile` 几何均值 1.56x，100 题胜 65 题 | 物理 profiler 与并行探索是当前最直接工业基线；MARPE 的差异是 typed proof 和 public-ABI 激活[^3] |
| KernelArc | 策略专用 Agent、conclusions-only memory、确定性 guard、plateau drafting | H100/B200 固定候选预算消融支持更宽搜索；结果依 workload/阶段而变 | 最近的直接竞争者；共享“结论”而非整段轨迹可降低污染和重复[^9] |
| KernelBench-Verified | TF32 真实 baseline、隐藏输入分布、内存指标与 audit | 标准评测 1.43x 的最好模型在严格评测下降为 0.88x；28% kernel 增加峰值内存 | Agent 越多，winner's curse 和 reward hacking 越严重，必须使用盲测、强 baseline 和多维证据[^10] |

这些工作支持一个更准确的论断：前沿系统正在通过 coordinated populations 扩大 inference-time search，但收益来自结构化搜索与外部 evaluator，而不是不受约束的自然语言群聊。多 Agent 还会显著增加 token、候选和筛选机会；如果不匹配预算和隔离 held-out，所谓协同收益可能只是 best-of-N 或选择偏差。

## 5. MARPE：与当前系统最匹配的架构

```text
immutable runtime profile + public ABI + evidence ledger
                         │
       ┌─────────────────┼──────────────────┐
       │                 │                  │
   Wave Scout        Tail Scout         Skew Scout
       │                 │                  │
       └──────── typed semantic deltas ─────┘
                         │
               structural deduplication
                         │
          Critic → Synthesizer → KIR verifier
                         │
               compiler + physical receipt
                         │
                single-B300 FIFO executor
                         │
        correctness → paired timing → profiler
                         │
              deterministic evidence gate
                         │
       conclusions-only memory + certificate
                         │
             held-out auditor → activation
```

### 5.1 Agent 角色

`Profile Curator` 冻结 workload ABI、硬件、开发集、held-out 和统计规则，不生成优化。`Wave Scout` 围绕目标 SM wave 数反推 cpc；`Tail Scout` 关注最后一 wave 的占用和 padding；`Skew Scout` 关注 total/max chunks、CV 和短序列尾部；`History Scout` 只从已有 winner 与失败 ledger 生成邻域候选。第一轮各 Scout 互相不可见，以保留真正的探索多样性。

`Critic` 检查适用 profile、已知反例、proof obligations 和可证伪条件。`Synthesizer` 只能从两个或多个 parent typed delta 生成 child，并记录 lineage；它不能直接编辑 CUDA，也不能原地覆盖父候选。`Measurement Judge` 解释确定性测量服务产生的 receipt，但没有批准激活的权限。`Meta Reviewer` 只选择下一轮预算和停止条件。

### 5.2 非 Agent 权威

四个组件必须保持确定性：KIR verifier、compiler/receipt builder、correctness oracle 和 B300 executor。最终 activation gate 也是普通程序：缺少 output 或 final-state correctness、scope parity、正的 95% speedup 下界或 fallback 中任何一项，都不能激活。多个 Agent 对同一非法 schedule 达成一致仍然是非法；多个 Agent 喜欢一个 CI 跨 1 的候选仍然只能进入 `MEASURE/RETAIN`。

### 5.3 共享协议

Agent 间只共享版本化、可核验的信息：runtime profile digest、parent schedule digest、注册 mutation、预测的瓶颈、预期计数器变化、disconfirming test、预算和 evidence refs。相同 canonical candidate 只编译和测量一次，但保留所有 proposer 的贡献归因；同一字段不同取值保留为两个 branch；不同 typed path 可组合，但 child 必须重新通过完整 verifier。

共享记忆采用 conclusions-only：

- 保留“cpc=18 在两个开发 profile 上退化”和对应 receipt；
- 不传播 Agent 关于退化原因的长篇自我解释，除非 profiler 或 counterfactual 支持；
- 按 architecture、toolchain、route、ABI 和 profile predicate 版本化；
- 变更任一关键条件时，历史结果降级为 prior，而不是继续作为 activation evidence。

## 6. 单机 B300 并不是多 Agent 的障碍

多 Agent 的并行发生在 CPU 侧的理解、假设与 typed proposal 生成，不要求多张 GPU。B300 侧反而应该保持单写者 FIFO：一个 executor 持有 device lease，依次运行 compile、correctness smoke、低预算 screen、top-K paired qualification 和 sealed held-out。这样避免并发 kernel、cache、clock 和 telemetry 互相污染。

建议采用 successive halving：所有候选先经过零 GPU 的 schema/KIR gate；每个语义唯一候选获得相同 correctness 与 screen 配额；只有 top-K 获得完整 ABBA/paired 预算；最终策略只在搜索不可见的 held-out 上运行一次正式资格测试。JIT、失败和 superseded attempt 仍由 `account_b300_run.py` 计入 GPU-second 与能耗。

这使计算扩展方式从“更多 GPU”变成“更多独立思路共享一张可信测量仪器”。瓶颈是总 GPU candidate budget，而不是 Agent 数量。Agent 可以很多，进入 B300 队列的候选必须经过语义去重和静态筛选。

## 7. 如何严格定义“涌现”

在完成等预算消融前，应称为 multi-agent collaborative search，而不是 emergence。要宣称协同涌现，至少需要三层证据。

第一是团队优势：相同基础模型、token、typed candidate 数、B300 GPU-second、开发 profile 和随机种子下，多 Agent 最终 held-out winner 显著优于最佳单 Agent。第二是共享优势：conclusions-only shared memory 优于互不通信的 independent pooled agents，否则收益只是 best-of-N。第三是组合优势：最终 child 同时包含来自不同 Agent lineage 的 typed delta，并满足

```text
gain(A ⊕ B) > max(gain(A), gain(B)) + preregistered noise margin.
```

实验至少包含四个 arm：best solo；4 个同角色独立 Agent；Plan/Implement/Critic 角色链；策略专用 Agent + conclusions-only memory + proof gate。每个 arm 固定 24 个硬件候选，至少 3 条独立 trajectory。主要指标不只报告最好 speedup，还包括 time/token/GPU-second to first accepted winner、verifier rejection、duplicate rate、correctness failure、development-to-heldout retention、compound-delta 数和错误激活数。

## 8. 对 H12 cpc=9 的机制复盘

当前实现中 BT16 prepare grid 近似为：

```text
rectangular_prepare_ctas = ceil(total_chunks / cpc) × num_heads
prepare_waves = ceil(scheduled_prepare_ctas / SM_count).
```

对 Kimi-K3 TP8 H12、B300 148 SM，total chunks 约 512 时，cpc=4 产生约 1,536 个 CTA、11 waves；cpc=9 产生约 684 个 CTA、5 waves。减少 prepare work assignment 的调度单元是稳定事实，但端到端收益由 prepare phase 在总 critical path 中的占比决定。Stage 4/5 的六个受影响观测中，total chunks 或移除 wave 数与 log-speedup 的线性相关接近零，而 `total_chunks / max_sequence_chunks` 的描述性相关约为 0.81。样本仅 6 个且包含强混杂，不能据此更新 dispatcher；它只提高“prepare/chain critical-path mixture”假设的优先级。

Stage 6 随后固定 total work 为 512 chunks，同时改变最长 recurrence chain：固定 8192；非均匀双序列 4000/4192；非均匀四序列 1952/2016/2080/2144；skew 64/64/64/8000。六候选 screen 中 cpc=9 在 4/4 profile 上均为最快，随后同一 prepared object 的 ABBA 配对确认得到 `1.0350x/1.0583x/1.0942x/1.0360x`，四个置信下界均高于 1。绝对节省仅在 `17.20-17.89 us` 之间变化，CV 为 `1.42%`；有效序列并行度与 log-speedup 的描述性相关为 `0.99796`。

Stage 7 使用 CUPTI Activity API 在每次 public call 中直接分离 prepare 和 chain 两个物理 kernel。cpc=4 到 cpc=9 的 full-span 节省为 `18.24-18.54 us`，其中 prepare 单独节省 `17.98-18.18 us`，解释 `97.49%-98.86%` 的 full-span 差异；四个 prepare 节省的 CV 仅 `0.41%`。相反，chain 差异只有 `0.09-0.59 us`。在总 chunks 固定时，baseline chain 时间与 `max_chunks` 的 Pearson 相关为 `0.9999995`，线性描述约为 `T_chain(us)=11.99+0.824*max_chunks`；cpc=9 的斜率几乎不变。这在已测四个 profile 上确认了阶段归因：cpc 改变 prepare 工作分解，最长 recurrence chain 决定剩余 critical path。

代表性 fixed-8192 prepare 的 NCU 结果又向前推进了一层：cpc=4 到 9 时 grid 从 `1536` 降到 `684` CTA，occupancy-aware waves/SM 从 `2.08` 跨到 `0.92`；block size 仍为 128，逻辑/分配寄存器仍为 `77/80`，shared memory 仍为 `45,056 B`。executed instructions 只下降 `3.96%`，active warps 从 `29.52%` 轻微变到 `28.56%`，但 NCU 内的 duration 下降 `17.74%`，SM/DRAM/L2 active-period throughput 分别提高约 `23.5%/21.0%/22.7%`。因此最吻合的机制是 CTA work decomposition 跨过一个 resident-wave 边界，减少调度尾波与摊销；它不是由单 CTA 资源占用改变引起，也不是指令量按 grid 比例缩小。NCU 存在 counter replay，所以这是机制证据，而不是部署性能 gate。[^12][^13][^16]

Stage 8 将这个解释改写为了可证伪预测。若 `R` 是当前资源占用下每 SM 可驻留的 prepare CTA 数，则让 H-way grid 首次压入一个 resident wave 的最小值为 `cpc*=ceil(W/floor(S*R/H))`。对 H=12、S=148、R=5，它在 W=256/512/1024 上预测 cpc=5/9/17。预注册 screen 在固定与 balanced-4 profile 上 8/8 命中该 winner；Stage 8 新增四个同对象配对确认，相对当前 cpc=9 规则得到 `1.0209x-1.0859x`，四个 95% 下界全部大于 1。CUPTI 再次将至少 `96.67%` 的 full-span 收益归到 prepare，chain 变化小于 `0.61 us`。这说明可迁移的不是 cpc=9 常数，而是 `(W,H,S,R,resource footprint,max dependency length)` 的分解。[^11][^12][^13]

## 9. 已落地的工程改进

本轮新增 `RuntimeWorkloadProfile` 与 `PrepareGridGeometry`，将 workload 与 target 映射为 JSON-ready receipt。它们不预测 latency，只暴露 Agent 可讨论且 reviewer 可复算的物理几何。新增 `AgentRole`、`AgentContribution`、`CoordinatedCandidate` 和 evidence-gated activation decision。多 Agent 的共识作为 provenance 保存，KIR 验证在合并之前执行。

新增 post-hoc analysis 将 Stage 4 development 和 Stage 5 held-out 连接到统一 profile receipt，输出六个受影响观测、特征相关和三个带 falsifier 的机制假设。matched-work B300 screen、配对确认和 Stage 7 phase attribution 已经完成，并由证书固化。新增 `AutonomousRuntimeProfileAgent` 后，系统已经能够执行 canonical typed candidate 去重、verifier gate、低预算 screen、top-K paired qualification、confidence-gated activation、conclusions-only memory、预算停止和 plateau 停止；Stage 6 的 24 个候选观测已被完整 replay，4/4 profile 自动选择并激活 cpc=9。该 replay 证明闭环实现，而不是证明多 Agent 优于单 Agent。

## 10. 论文定位与下一步

最强定位不是“RPE 也使用多个 Agent”，而是：

> Existing multi-agent kernel optimizers coordinate through natural language, source code, or untyped trajectory memory. MARPE makes physical schedule proposals machine-checkable and requires a deployment-scoped empirical certificate before any runtime activation.

相对 Astra、CudaForge、KernelAgent 与 KernelArc，贡献是 typed inter-agent protocol、public ABI/final-state 语义、置信区间与 fallback；相对 CAKE，贡献是 runtime-profile-conditioned 的 post-compilation evolution，以及对多 Agent 协同的可审计、可回滚部署闭环；相对 KernelBench 系统，任务不是单 kernel 排行，而是 dispatcher-inclusive production decision。

近期路线已经从“补机制”推进到“补搜索因果”：phase attribution、代表性 NCU 机制证据、三层 work-scale 预测、canonical candidate ID、conclusions-only event log 和自主停止均已完成。Stage 9 的四 arm 等预算消融协议也已机器可读地冻结：单 Agent、单 Agent 角色扮演、同质多 Agent 和 conclusions-only 专业多 Agent 必须使用相同 token、candidate、GPU-second 和 sealed held-out。只有出现跨 lineage 的 compound winner并且 multi-role 同时胜过 roleplay 与 homogeneous control，才能把 collaborative search 升级为协同涌现结论。

## 11. Stage 10：从事后解释到前瞻预测

Stage 10 在首次 GPU 查询前封存了 W384/W768 四个精确 profile、cpc7/cpc13
预测、物理 route、8 个独立进程 epoch、ABBA/BAAB 交替顺序和 Bonferroni
单侧 98.75% 门槛。本轮不允许 screen 或相邻 cpc 搜索。四个 profile 全部通过，
最弱的 family-wise 下界仍为 1.012949x，并且 8/8 epoch 全部同方向。

这一结果有两个比“再命中一个 winner”更重要的含义。第一，W384 从 cpc9 变为
cpc7 会增加 CTA，W768 从 cpc9 变为 cpc13 会减少 CTA；两者都加速，否定了“CTA
越少越好”的单调解释。第二，两个预测的 cpc-1 都会产生 768 CTA，而候选值使
grid 回到 660/720 CTA，恰好落入 `148 SM * 5 resident CTA/SM = 740 CTA` 容量。
因此准确名称应是“resident-grid-capacity fill”，而不是普通的 one-CTA-per-SM wave。

事后 CUPTI 只做机制归因，不回写资格结论。它给出 3.89--31.57 us prepare 节省，
解释 97.98%--101.16% full-span 收益；chain 的 95% 区间均在事前冻结的 +/-1%
等价带中。所以当前可迁移的知识是：`(total chunks, local heads, SM count,
compiled-kernel resident capacity)` 决定 prepare 并行分解，`max per-sequence chunks`
决定 recurrence critical path。但不能把 R=5 或 cpc 数字本身迁移到新 kernel/新 GPU。

## 12. Multi-Agent 在本阶段的实际作用

本阶段的多 Agent 不是性能证据来源，而是研究质量控制器。独立实验审稿指出了
one-resident-wave 命名不严谨、cpc13 取整需要明确原理、应使用进程级统计单元和多重
比较校正；文献审稿给出了 Stream-K/NVIDIA occupancy 的正确对比边界；架构审稿发现
`Target.max_ctas_per_sm=8` 不能代替实测 R=5，并提出必须使用 kernel/image/resource
哈希绑定的 receipt。这些意见直接改变了实验和代码，但并不证明多 Agent 比单 Agent
更强。该问题仍由 Stage 9 等预算四 arm 消融回答。

## 13. 当前最强结论与后续边界

现在已经可以说：在同一 B300/H12/BT16 编译资源 receipt 内，容量规则对两个未测中间
工作量做出了成功的前瞻预测，并在八个独立进程中经过 family-wise 统计和 prepare
机制归因。但 policy 仍应是 `SHADOW_ONLY`：它可以生成 recommendation receipt，却继续选择
现行 cpc9 fallback。任何 ABI、target、physical variant、resource、occupancy、evidence-domain 或
hash 漂移都必须以 `RPEPxxx` 原因码回退。

对当前这张 B300 来说，继续局部 cpc 扫描的科学收益已经很低。下一个真正能推进论文的
实验是：获取第二张卡/新 resource receipt 做迁移，或执行已冻结的 Stage 9 等预算多
Agent 四 arm 消融。在此之前，不应为了扩大结论而动 production dispatcher。

## Sources

[^1]: Google Research. “[Accelerating scientific breakthroughs with an AI co-scientist](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/).” February 19, 2025.
[^2]: Google DeepMind. “[AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/).” May 14, 2025.
[^3]: PyTorch. “[KernelAgent: Hardware-Guided GPU Kernel Optimization via Multi-Agent Orchestration](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/).” March 6, 2026.
[^4]: Wei et al. “[Improving Parallel Program Performance with LLM Optimizers via Agent-System Interface](https://arxiv.org/abs/2410.15625).” arXiv:2410.15625, 2024.
[^5]: Arora et al. “[MASAI: Modular Architecture for Software-engineering AI Agents](https://arxiv.org/abs/2406.11638).” arXiv:2406.11638, 2024.
[^6]: Wei et al. “[Astra: A Multi-Agent System for GPU Kernel Performance Optimization](https://arxiv.org/abs/2509.07506).” arXiv:2509.07506, 2025.
[^7]: Zhang et al. “[CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization](https://arxiv.org/abs/2511.01884).” arXiv:2511.01884, 2025.
[^8]: PyTorch. “[KernelFalcon: Autonomous GPU Kernel Generation via Deep Agents](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/).” November 5, 2025.
[^9]: Kundu et al. “[KernelArc: A Multi-Agent Framework for GPU Kernel Optimization](https://arxiv.org/abs/2608.17071).” arXiv:2608.17071, August 17, 2026.
[^10]: Zhang et al. “[KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?](https://arxiv.org/abs/2607.16241).” arXiv:2607.16241, July 2026.
[^11]: NVIDIA. “[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/).” 2026.
[^12]: NVIDIA. “[NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/).” 2026.
[^13]: NVIDIA. “[CUTLASS Python DSL Task Scheduling: Schedules](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_schedules.html).” 2026.
[^14]: Osama et al. “[Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU](https://arxiv.org/abs/2301.03598).” PPoPP, 2023.
[^15]: NVIDIA. “[CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html).” 2026.
[^16]: NVIDIA. “[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).” 2026.
[^17]: Chen et al. “[TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799).” OSDI, 2018.
[^18]: Zheng et al. “[Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://www.usenix.org/conference/osdi20/presentation/zheng).” OSDI, 2020.
[^19]: Shao et al. “[Tensor Program Optimization with Probabilistic Programs](https://arxiv.org/abs/2205.13603).” NeurIPS, 2022.
[^20]: Feng et al. “[TensorIR: An Abstraction for Automatic Tensorized Program Optimization](https://arxiv.org/abs/2207.04296).” ASPLOS, 2023.
[^21]: Tillet, Kung, and Cox. “[Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://doi.org/10.1145/3315508.3329973).” MAPL, 2019.
[^22]: Shah et al. “[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608).” NeurIPS, 2024.
[^23]: Yang, Kautz, and Hatamizadeh. “[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).” ICLR, 2025.
[^24]: Kimi Team. “[Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692).” 2025.
[^25]: Ye et al. “[CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/abs/2608.12629).” 2026.
