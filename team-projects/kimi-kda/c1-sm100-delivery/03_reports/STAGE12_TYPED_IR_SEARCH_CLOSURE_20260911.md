# Stage 12：双分支 Typed IR 搜索扩展与理论闭包

团队：奶龙必胜（蔡雨洋、李奥、赵骋）
设备：NVIDIA B300 SXM6 AC（SM103a）
日期：2026-09-11

## 摘要

本阶段没有把“上一轮未找到更快候选”当作全局上限，而是先检查搜索表示本身。审查发现，原有 Program IR 能表达流水角色和屏障，却不能把 `MMA route / value slice / compute-warps / accumulator carrier / physical layout / materialization placement / scheduler / pipeline / TMEM lifetime / BF16 rounding` 放在同一可组合类型系统中；原 multi-agent 协调器也主要合并同构 Schedule，难以形成布局、驻留、调度和并行度的跨角色联合候选。

本阶段补上了三层能力：

1. `KernelDesign + DataflowProgram`：把 HMMA 与 tcgen05 的物理设计选择、SSA 数据流和数值边界类型化；
2. `coordinate_portfolio + portfolio_elites`：允许五类策略角色组合互不冲突的 axis patch，并按机制 niche 保留正确、同口径的最优解；
3. `canonicalize_dataflow + ClosureCertificate`：在不跨越 FP32→BF16 舍入边界的前提下消除恒等、公共和串联重排，并对有限搜索域给出“已覆盖/仍缺失”的可审计闭包。

B300 新实验同时给出了“哪里不值得继续”与“哪里确有结构性上界”。HMMA 将每个计算 warp 的 16-column block 数由 1 增至 2 后，V32、V64 相对当前基线分别只有 **0.8461x、0.8907x**；经更强验证重跑后，tcgen05 的 producer-ready 物理 B 输入相对逻辑输入在 inner=1/64 为 **0.99994x、0.99263x**，inner=64 相对 HMMA 为 **0.94606x**。但在新的 P3→BF16 舍入→P4 双 MMA 探针中，tcgen05 仅分配一次 TMEM、重用累加列并使用一次安全 shared carrier，在 inner=64 上反而达到 HMMA 的 **1.4460x（grid 12）/ 1.4303x（grid 96）**；inner=1 则只有 0.7651x/0.7089x。这说明跨阶段 tcgen 不是“不可行”，而是必须有足够长的驻留期来摊薄生命周期成本。

最终建议因此更精确：**保留 HMMA profile fallback；只对足够长的 recurrence/profile 继续移植 P3/P4 共享生命周期；停止重复搜索已闭包的 warp-density 和单独 global-B layout；完整 SM100 后端仍以 public-full 正确性和同口径计时决定是否晋级。**

## 1. 问题与“闭包”定义

这里的闭包不是“证明不存在更快 CUDA 程序”。GPU kernel 优化的全局空间没有可计算的有限边界，而且外部项目仍在持续加入新的 M64、M128、split、slab 与 persistent 路线。本文使用三个分层口径：

- **候选闭包**：声明一个有限、类型正确的候选集合；每个语义唯一候选均有正确性与性能终态证据；
- **机制闭包**：候选闭包覆盖某个明确机制及 profile，例如 V32/V64 每 warp 处理 1 或 2 个 16-column block；
- **算法开放边界**：所有不属于该有限语法或未做 public-full 测量的设计保持开放。

这比“连续若干轮没有提升便停止”更严格。闭包证书记录语义 hash、测量作用域、缺失候选和证据路径；只要加入新轴或扩大 profile，原闭包不会被误用为新空间的证明。

## 2. 联网调研带来的框架判断

### 2.1 Typed IR 应编码物理协议，而非只编码 CUDA 文本

CUTLASS Task Scheduling 将资源的 producer/consumer 协议显式区分为 `TmaUmma`、`UmmaAsync`、`AsyncUmma`、`UmmaUmma`、`ClcFetchAsync` 等类型，并检查 stage 数、transaction bytes、参与线程和 cluster shape；声明与真实硬件操作不一致会导致 hang 或 race。[^1] Schedule 还把静态 domain、动态 domain、一次性 setup、周期动作和 drain 表达为结构化事件，而不是注释约定。[^2]

这直接支持我们的设计：pipeline、carrier、layout、TMEM lifetime 必须是 IR 的一等字段，不能留到 agent 生成源码后才发现组合不可实现。MLIR Transform dialect 进一步说明了“变换 IR 与 payload IR 分离”的价值：变换可以有类型化 handle、失败传播和失效检查，payload 则继续保留领域语义。[^3]

### 2.2 扩大搜索空间不等于无约束生成

Ansor 的经验是通过层次化表示扩大组合空间，再用 evolutionary search 与 cost model 排序，而不是在源文本中盲目采样。[^4] `egg` 的 equality saturation 则保留多种等价表达并通过领域分析抽取低成本形式。[^5] 本项目采用两者的保守交集：

- 用有限枚举和 typed patches 扩大可组合轴；
- 只实现有证明条件的 layout canonicalization；
- 不把 BF16 舍入、异步 barrier 或 TMEM 生命周期当成普通代数等价式。

因此本阶段没有直接引入完整 e-graph。对异步 GPU 程序，错误地交换一次 `wait/release` 或跨越一次舍入的风险远大于当前两条重排规则的收益。

### 2.3 Multi-agent 的价值在“互补组合”，不在 agent 数量

Astra 使用生成、测试、profiling、planning 等专门角色迭代现有 CUDA，并报告平均 1.32x；[^6] KernelArc 强调 strategy-specialized agents、只共享结论的 memory、确定性 benchmark guard 与 plateau-triggered drafting，同时指出协调组件的价值依 kernel 和阶段而变化；[^7] KernelSkill 则使用长期技能与短期防回退记忆，减少重复试错。[^8]

我们据此没有简单增加并行 agent，而是把 agent 输出限制为五类 typed patch：`layout / residency / scheduler / parallelism / numerics`。协调器只组合不同角色、同一 axis 不冲突且通过 verifier 的提案；测量后按 route、驻留、调度、TMEM lifetime、blocks-per-warp 和 producer-ready contract 建立 niche。这个策略借鉴 MAP-Elites“保留不同机制下的优质解”，避免一个全局 winner 过早清空多样性。[^9]

### 2.4 外部前沿证明“全局上限”仍未关闭

FlashInfer 当前 VibeCUDA KDA PR 在 B300 十个公开 workload 上报告相对其固定 CAKE 分母的 2.3305x 几何平均，并通过 M64、M128、split、slab、persistent 进行条件分派；该 PR 仍在审查，且页面同时保留 final-state、超大 packed 输入和 dispatch 断言等未解决项，不能当作已合并生产结果。[^10] CAKE 集成 PR 在 176 个可计时 core rows 上报告 FlashKDA/CAKE 为 2.646837x、FlashKDA/export 为 2.729370x，但仍保留 22 个失败 gate。[^11]

两组外部数字的 workload、分母和代码版本不同，不能与本文数字连乘；它们只说明：真正强的前沿来自**算法族组合与 dispatch**，而不是单条 tcgen05 指令或单个 layout 旋钮。CUDA 的 Cluster Launch Control 也将动态 work stealing定位为兼顾固定工作分配和 persistent block 复用的机制，因此最值得探索的是不规则 tail 的条件路径，而非所有 profile 一律使用动态调度。[^12]

## 3. 实现：从字符串搜索到可组合设计空间

### 3.1 `KernelDesign`

新增 [`design_ir.py`](../06_agent_framework/kda_ir/design_ir.py) 对十一类物理轴建模。`verify_kernel_design()` 的关键约束包括：

- CHUNK 固定为 16；value slice 限于 16/32/64/128；
- 16-column blocks 必须均匀分给 1/2/4 个 compute warps，当前 lowering 每 warp 最多两个 block；
- HMMA 必须使用 register accumulator、HMMA B fragment、同步 pipeline 且无 TMEM；
- tcgen05 必须使用 TMEM accumulator、32B-swizzled NK shared operand 和显式异步/TMEM lifecycle；
- tcgen05 的 shared-MK A operand、A-TMEM operand 和 TMEM accumulator 是三个不同物理类型，D fragment 不能无证明地当成下一次 MMA 的 A；
- producer-ready layout 与额外 materialization 互斥；cross-phase TMEM 必须携带 producer-ready contract；
- 官方 BF16 舍入边界不可删除。

因此 HMMA↔tcgen05 不再是字符串替换，而是 carrier、layout、pipeline 和 lifetime 的联合迁移。

### 3.2 `DataflowProgram` 与冗余重排清理

数据流 IR 的每个 value 都带 `shape / dtype / layout / carrier`。对当前 tcgen Phase-6 bridge，原始表示为：

```text
u_reg[HMMA fragment]
  -> spill_u_logical[logical shared]
  -> reorder_u_preferred[tcgen05 32B-swizzled NK shared]
  -> phase6_tcgen05
```

规范化后为：

```text
u_reg[HMMA fragment]
  -> canonical direct materialization[tcgen05 32B-swizzled NK shared]
  -> phase6_tcgen05
```

在 64 次递推的 V16 模型里，显式 materialization 从 2 次降为 1 次，写入量上界由 **65,536 B/CTA** 降为 **32,768 B/CTA**。这是 IR 的结构成本，不是延迟预测。`ROUND` 被标记为 semantic boundary；规范化器不会跨过 FP32→BF16 位置，也不会在不一致 placement 间融合。

### 3.3 组合式 multi-agent portfolio

新增 [`portfolio_search.py`](../06_agent_framework/kda_ir/portfolio_search.py)：

1. 每个角色提交 `DesignPatch(updates, claim, evidence_refs)`；
2. coordinator 枚举至多三阶组合；
3. 同一轴值冲突、非法字段或物理 verifier 失败的组合被拒绝；
4. 语义相同候选按 canonical ID 去重；
5. 只有 correctness 通过且 measurement scope 相同的候选能进入 niche elite archive。

这修复了旧 coordinator 只能有效聚合同构 Schedule 的上界：例如 `paired-warps + persistent-grid` 可形成真正的复合候选，`cross-phase TMEM + producer-layout` 也能跨越旧 Program IR 的表示边界。

### 3.4 有限域闭包证书

新增 [`closure.py`](../06_agent_framework/kda_ir/closure.py)。证书拒绝无效设计、语义重复、域外观察、无证据路径、正确性失败和非正 speedup，并显式列出 `missing_design_ids`。108 项 CPU 测试全部通过，包含重排融合、舍入边界保护、跨角色组合、冲突拒绝、niche archive、闭包覆盖，以及“TMEM accumulator 不得无变换作为 A operand”的物理规则。

## 4. B300 双支路实验

### 4.1 HMMA：两列块/warp

假设：V32/V64 基线分别使用 2/4 个 compute warps，每 warp 处理一个 16-column block；若改成每 warp 两个 block，可将总线程数从 128→96、192→128，可能降低同步和调度成本。

实验使用冻结官方源与当前 ValueSlice baseline，隔离构建新扩展；每个 worker 在计时前验证 output 和 final state 均 finite 且对官方 bitwise 相等，再做 public-full、CUPTI、cold-L2、双 block 对称测量。Job 25316 的结果见 [`round9_hmma/summary.json`](../04_evidence/agent_rounds/round9_hmma/summary.json)。

| Profile | 官方 ms | 当前 baseline ms | paired ms | paired/baseline | paired/官方 |
|---|---:|---:|---:|---:|---:|
| H12 balanced nseq3, V32 | 0.322411 | 0.246658 | 0.291531 | 0.846076x | 1.105924x |
| H12 balanced nseq6, V64 | 0.199875 | 0.168882 | 0.189609 | 0.890686x | 1.054139x |

结论：压缩 compute warps 会显著回退，说明当前两 profile 的主约束不是 block-level 线程开销，而是每轮可并行发射/计算的 warp 数。paired 仍胜官方，只说明 ValueSlice 大方向有效；它不应替换当前 V32/V64 基线。此轴已在声明域内闭包，不再加采样。

### 4.2 tcgen05：producer-ready 物理 B

假设：Round 8 preferred-layout 仍由线程把逻辑 `B[K,V]` 以跨步地址搬到 tcgen05 的 `B^T[V,K]` swizzled shared layout；若 producer 直接给出物理 `B^T[V,K]`，读取可连续，可能追回与 HMMA 的约 4.7% 差距。

Job 25317 暴露出原验证数据的隐患：`A0=((5m+3k+b) mod 3)-1` 中 `3k mod 3=0`，A0 沿 K 维恒定，无法检出 16-byte sub-block 错位。我们将它改为 K-非退化输入，并根据 CUTLASS `Swizzle<1,4,3>` 的位级定义，把实现从错误的 row bit 0 修正为 address bit 7（当每行 32 B 时即 row bit 2）对 address bit 4 异或。[^13][^14] 以此为基础，Job 25380 同时编译 logical 与 physical 两个二进制，先跑全部正确性，再以 warmup=30、iters=200、repeats=7 测量。下表只使用修正后证据，Job 25317 保留为 superseded 记录。

| Phase-6 V16 probe | logical preferred µs | producer-ready µs | physical/logical | physical/HMMA |
|---|---:|---:|---:|---:|
| inner=1 | 10.255680 | 10.256320 | 0.999938x | 0.800303x |
| inner=64 | 77.051840 | 77.623839 | 0.992631x | 0.946062x |

结论：该全局输入布局改变没有收益，inner=64 约回退 0.74%，且未追回 HMMA 差距。它只闭包“Phase-6 单独换 producer-ready global B”这一机制；不能推导 P4 在完整 kernel 内免费生成 layout，也不能否定 P3/P4/P6 共用 TMEM 的跨阶段方案。

### 4.3 tcgen05：P3→BF16 round→P4 跨阶段生命周期

为了不再用单个 Phase-6 探针推测多阶段收益，我们新建了最小双 MMA 闭环：P3 计算 `A0×INV`，明确舍入到 BF16，P4 再计算 `U×MQK`。HMMA 与 tcgen05 使用相同输入和舍入点；tcgen 分支只分配一次 TMEM、在两次 MMA 间重用累加列，仅通过一个 32B-swizzled shared carrier 保存舍入后 U。它不含 P1/P2/P5/P6、beta、TMA 或生产 warp-specialized schedule，因而仍是机制探针。

Job 25377 在 inner=1/2/4 上对 P3 carrier、第二次 issue 和最终输出逐 bit 验证全部通过，32 registers、无 spill。计时结果为：

| Grid / inner | HMMA µs | tcgen shared-carrier µs | tcgen/HMMA | 每次 HMMA/tcgen µs |
|---|---:|---:|---:|---:|
| 12 / 1 | 6.671360 | 8.720000 | 0.765064x | 6.671360 / 8.720000 |
| 12 / 64 | 65.582881 | 45.353761 | **1.446030x** | 1.024733 / 0.708653 |
| 96 / 1 | 6.907840 | 9.744000 | 0.708933x | 6.907840 / 9.744000 |
| 96 / 64 | 65.593119 | 45.860481 | **1.430275x** | 1.024892 / 0.716570 |

这个结果给出了比“tcgen 裸指令更快”更有用的边界：短驻留期会被 alloc/commit/wait/carrier 固定成本吞没，长驻留期才能出现 1.43–1.45x 的机制上界。因此下一步不应给所有 profile 强行 tcgen，而应将 recurrence 长度和 phase residency 编辑进 dispatch gate。

我们还编译了两个将 P3 D-fragment 直接转为 P4 A-TMEM 的候选：一个用 `.unpack::16b`，一个将 BF16 放在独立 TMEM word 的低 16 位。两者的 24,576 个结果均全部失配，且错误模式相同；同一程序中的 shared carrier 和第二次 issue 诊断仍正确。结合 PTX 对 D 和 A 的不同 packing/layout 契约，可将问题定位为缺少显式 D-fragment→A-TMEM 物理变换，而不是数值、barrier 或第二次 commit 错误。[^13] 这条候选未进入计时，源码与失败日志作为证伪记录保留。

## 5. 与 benchmark 的同口径比较

| 路线 | 作用域 | 相对性能 | 可下结论 | 不可下结论 |
|---|---|---:|---|---|
| 官方 FlashKDA | public-full reference | 1.000x | 比较基线 | 最优实现 |
| HMMA V32 当前基线 | H12 balanced nseq3 | 1.3071x vs 官方 | 该 profile 保留 V32 | 所有 nseq 选 V32 |
| HMMA V64 当前基线 | H12 balanced nseq6 | 1.1835x vs 官方 | 该 profile 保留 V64 | 所有 nseq 选 V64 |
| HMMA paired | 同上两 profile | 0.8461x/0.8907x vs 当前基线 | 不激活 | ValueSlice 无效 |
| direct V128 tcgen05 | Phase-6, grid12, inner64 | 0.919742x vs HMMA | 裸指令替换停止 | tcgen05 全局无效 |
| Round 8 preferred（修正后） | V16 mechanism probe, grid96/inner64 | 1.2107x vs scalar；0.9564x vs HMMA | rematerialization 是实成本 | 可当 public-full headline |
| Round 9 physical B（修正后） | V16 Phase-6, inner64 | 0.992631x vs logical；0.946062x vs HMMA | 单独改输入布局停止 | 跨阶段 layout 无效 |
| Round 10 P3/P4 shared carrier | 双 MMA mechanism, inner64 | 1.4460x（grid12）/1.4303x（grid96）vs HMMA | 长驻留跨 phase 路线有明确上界 | 可当 public-full speedup |
| Round 10 P3/P4 shared carrier | 双 MMA mechanism, inner1 | 0.7651x（grid12）/0.7089x（grid96）vs HMMA | 短 profile 应回退 HMMA | tcgen 在所有 profile 都慢 |
| CAKE SM100 | 9-shape isolated public-full | H12 2.4823x；H96 2.2532x vs 官方 | 完整 co-design 值得 | 收益全由 tcgen05 产生 |
| tcgen H96 s173 | exact mixed public-full | 1.0330x vs CAKE | 仅该 profile 晋级 | scheduler 普遍有效 |
| load-minimax 169 | exact scheduler screen | 0.9383x vs incumbent | load-minimax 子空间闭包 | 所有动态调度无效 |

内部历史数字的来源与限制见 [`C1_REPRODUCTION_ANALYSIS_CHALLENGE_CONTENT_20260910.md`](C1_REPRODUCTION_ANALYSIS_CHALLENGE_CONTENT_20260910.md)。外部 VibeCUDA/CAKE PR 的分母与 profile 不同，只作为 frontier 参照，不纳入本表的排序或 speedup 连乘。

## 6. 理论闭包结论

机器可读证书见 [`round9_closure_certificate.json`](../04_evidence/agent_rounds/round9_closure_certificate.json)。截至本阶段：

### 已关闭

- H12 balanced nseq3/nseq6 上，V32/V64 每 compute warp 处理一个或两个 16-column blocks 的有限域；
- Phase-6 V16 probe 上，logical 与 producer-ready global B 两种输入契约；
- P3→BF16 round→P4 双 MMA 探针上，HMMA 与“单次 TMEM 分配 + shared carrier”在 inner=1/64、grid=12/96 的机制对照；
- 两个未经布局变换的 D-fragment→A-TMEM 直通候选（它们以正确性失败终止，不进入性能比较）；
- typed dataflow 中 `register -> logical shared -> preferred shared` 的二跳冗余 materialization。

### 仍开放

- 将已验证 P3/P4 lifecycle 嵌入含 P1–P6、beta、TMA 和 final state 的 full public-call；
- 由 CUTLASS/CuTe 布局推导出正确 D-fragment→A-TMEM 变换，并证明它比已验证 shared carrier 更快；
- M64/M128/split/slab/persistent 的条件组合及对 irregular tail 的 CLC 调度；
- producer 在上一 phase 内原位产生 consumer layout，而不是把转置成本移到 API 输入准备；
- 超出当前 typed grammar 的新算法、精度或 state ABI。

所以“理论闭包上限”的准确表述是：**当前局部语法与已测 profile 已闭包；已找到长驻留 P3/P4 tcgen05 的 1.43–1.45x 机制上界；完整 SM100 KDA 的可达上界仍取决于 full-call 移植和 profile dispatch。** 继续在已关闭旋钮上做更多 agent 轮数不会提高知识量；下一次重开必须带有 full-call lowering、可验证的 D→A TMEM 映射、新算法族或 workload 证据。

## 7. 收尾决策

1. 生产建议维持 guarded portfolio：长 recurrence 优先试用跨 phase tcgen05 route，短 recurrence 与未认证 profile 回退 HMMA。
2. HMMA 保留当前 V32/V64 compute-warp mapping，不合入 paired-warps。
3. tcgen 不合入单独 producer-ready global-B 改动；保留已通过的 P3/P4 shared-carrier lifecycle 作为 full-call lowering 的可执行规范。
4. Agent 框架以后以 typed patch、niche elite 和 closure certificate 为搜索入口，不再以“多跑几轮”代替新增机会。
5. 下一项高价值实现是将已验证 P3/P4 lifecycle 移入完整 public call；如果不能通过 output + final state 正确性和相同计时作用域，就不升级为性能结论。

## Sources

[^1]: NVIDIA, [CUTLASS Task Scheduling — Pipeline Types](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_pipelines.html), accessed 2026-09-11.
[^2]: NVIDIA, [CUTLASS Task Scheduling — Schedules](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_schedules.html), accessed 2026-09-11.
[^3]: LLVM Project, [MLIR Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/), accessed 2026-09-11.
[^4]: Zheng et al., [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://www.usenix.org/conference/osdi20/presentation/zheng), OSDI 2020.
[^5]: Willsey et al., [egg: Fast and Extensible Equality Saturation](https://arxiv.org/abs/2004.03082), POPL 2021.
[^6]: Wei et al., [Astra: A Multi-Agent System for GPU Kernel Performance Optimization](https://arxiv.org/abs/2509.07506), revised 2025-12-02.
[^7]: Kundu et al., [KernelArc: A Multi-Agent Framework for GPU Kernel Optimization](https://arxiv.org/abs/2608.17071), revised 2026-08-20.
[^8]: Sun et al., [KernelSkill: A Multi-Agent Framework for GPU Kernel Optimization](https://arxiv.org/abs/2603.10085), 2026-03-10.
[^9]: Mouret and Clune, [Illuminating search spaces by mapping elites](https://arxiv.org/abs/1504.04909), 2015.
[^10]: FlashInfer, [PR #4779: VibeCUDA SM100/SM103 recurrent KDA prefill backend](https://github.com/flashinfer-ai/flashinfer/pull/4779), accessed 2026-09-11; open PR, reported denominator and unresolved review items retained on the page.
[^11]: FlashInfer, [PR #4886: integrate CAKE-generated Blackwell prefill kernels](https://github.com/flashinfer-ai/flashinfer/pull/4886), accessed 2026-09-11; draft PR with failed gates retained in reported aggregates.
[^12]: NVIDIA, [CUDA Programming Guide — Work Stealing with Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html), accessed 2026-09-11.
[^13]: NVIDIA, [PTX ISA 9.4 — Tensor Memory load/store packing and `tcgen05.mma`](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-st), accessed 2026-09-11.
[^14]: NVIDIA CUTLASS, [`cute::Swizzle` bit-level definition](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp) and [SM100 TMEM fragment layouts](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm100_frag.hpp), accessed 2026-09-11.
