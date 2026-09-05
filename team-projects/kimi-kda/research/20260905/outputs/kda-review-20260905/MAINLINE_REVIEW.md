# FlashKDA 主线 Review 与下一阶段决策

2026-09-05 · 以 C1 题目、固定版本源码、Kimi Linear 论文和 B300 实测为依据

本报告专门回答：**是否继续坚持“不全面机械替换 `tcgen05`；发布 V128 `mma.sync` fallback + guarded ValueSlice 的 B300/SM103 路径”，以及随后最值得做什么。** 它取代此前宽泛探索报告中的后续工作排序，但不覆盖或改写历史实验。本轮新增此审查文档，没有修改 kernel、补丁、既有报告或 Git 分支，也没有发布软件。

## 1. 总判断：主线成立，论证需要更新，发布需要分层

| 决策 | 本次判断 | 准确含义 |
|---|---|---|
| C1 主线研究是否基本完成 | **是** | 官方复现、六问分析、并行度重构挑战及代码/报告/答辩材料均已形成；不需要用新的模型研究来证明作业完成 |
| 全面机械替换为 `tcgen05` | **仍不建议** | 没有完整 K2 净收益证据；当前布局、状态依赖、转换和同步成本不能被峰值规格抵消 |
| 发布 ValueSlice 研究成果/受限候选版本 | **值得** | 生产模型尺寸不变，已测高价值形状约 27% 算子降时，ValueSlice 对 V128 有严格 bitwise 对照 |
| 将当前包直接称为通用生产替代品 | **证据不足** | 调度域、默认开关、入口安全和构建复现仍需收口；未验证真实 serving、持续并发或多卡集成 |
| 继续投入 SM103 优化 | **值得，但有实验门槛** | 优先完成受保护发布；下一步围绕最佳 ValueSlice 的剩余关键路径，不重启全面指令重写 |

最合适的更新版结论是：

> **保留模型与现有数值契约，在 B300/SM103 已验证的长 prefill 域用 ValueSlice 扩展独立并行度，域外选择 V128。暂不集成机械 `tcgen05` 替换；只对能改善完整数据流、且击败当前最佳 ValueSlice 的候选重新立项。**

“专用”主要指 B300 资源与延迟标定，并不意味着 ValueSlice 的数学分解只有 SM103 才能实现。原实现已经使用 TMA；本项目不是第一次引入 TMA，也没有把 MMA atom 升级成 TCGEN。

## 2. 生产尺寸是约束，不能为了指令漂亮而修改

这里需要把四种不同的“大小”分开。

| 量 | 本主线的约束/来源 | 是否可作为无损实现调参 |
|---|---|---|
| 模型 `d_k=d_v=128`、每头完整 state 为 `128×128` | 题面与论文相符；当前 API 也固定 K=V=128 | **不能缩小模型维度或 state 容量来换性能** |
| 全模型 H96；TP8 每卡 H12 | 来自 C1 题面；93 层中 69 KDA/24 全注意力也是题面配置 | 保持模型不变，只按真实 TP/请求数映射每卡工作量 |
| `CHUNK=16` | 固定 FlashKDA v1 的算法/数值/指令联合设计 | 不是模型 head dimension，但改变它需要重新设计并验证数值路径 |
| `ValueSlice=16/32/64/128` | 每 CTA 负责的 Value 子块 | **可调**；全部 slice 合起来仍计算完整 128 维输出和完整 state |
| probe 的 `inner=512` | 同一 Phase-6 microkernel 内重复计算的次数 | 只是摊销实验变量；不是 CHUNK512、模型 loop 深度或新的生产输入尺寸 |

[C1 题面](../../C1_TASK.md) 是具体 K3/TP 形状的依据；不要把 H96、93 层配置错误归给 Kimi Linear 论文。论文 §4 明确所有实验的 key/value head dimension 均为 128，§6.3 区分 prefill chunk kernel 与生成阶段的 recurrent kernel；其 FLOP 分析使用 C64，并不要求后来的 FlashKDA v1 也采用 C64。[Kimi Linear §4、§6.3](https://arxiv.org/html/2510.26692v1)

FlashKDA v1 的 C16 与有限指数范围、小规模 Neumann 求逆、MMA 形状共同设计；片上 state 按 BF16 保存、更新使用 FP32 FMA。当前 CHUNK32/64 的负实验针对**机械延伸现有路径**，不是证明安全的大 CHUNK 算法不存在。[固定版本设计文档](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/docs/20260420-flashkda-v1-deep-dive.md#L11)

## 3. 主线证据复核

### 3.1 ValueSlice 解决的是可用并行工作不足，不是降低模型工作量

TP8 单请求每卡 H12，原 K2 是 12 CTA，面对本卡 148 SM。V16 将每个 head 的 128 个 Value 通道拆成 8 份，得到 96 CTA，无跨 slice reduction/atomic，保留每个输出元素的计算顺序。

总 Tensor FLOP 并没有减少。获得收益的交换是：**更多独立 CTA、更小的单 CTA 状态/资源占用，换取公共输入的重复请求和更多调度工作。** 因此不能把所有收益只归给 grid 大小，资源占用也随变体改变；但扩展独立并行度与 NCU/Nsys/整体计时一致，是最有证据的主解释。

Job17965 的 targeted NCU 中，V128/V16 的 SM throughput 为 2.64%/7.22%，DRAM throughput 为 1.24%/1.83%，duration 为 1.27 ms/901.22 µs。这支持“underfill + recurrence/CTA 等待链”，不支持传统的整卡 Tensor Core 或 HBM 饱和解释。V16 的 No Eligible 更高、achieved occupancy 更低，并不否定整体降时；指标分母与活跃 SM 数必须一起看。[NCU 原始汇总](../../../../experiments/final_campaign/data/raw/05_targeted_ncu_summary_17965.csv)

**目标不是把 148 个 SM 全部点亮，也不是追求最高 occupancy，而是最小化完整 forward 延迟。** 继续切得更细，可能只增加重复请求和等待。

### 3.2 约 27% 收益可信；泛化为所有 workload 则不成立

本次此前已完成的 B300 Job19845 对 `B1,T8192,H12,D128`、BF16 state 做了配对复测。每项取三轮 median 的中位数：

| 同口径完整 forward | V128 / ms | V16 / ms | 降时 |
|---|---:|---:|---:|
| eager，预热 | 0.782784 | 0.569008 | 27.31% |
| eager，调用前写 256 MiB 缓冲 | 0.783408 | 0.570272 | 27.21% |
| CUDA Graph，预热 | 0.779904 | 0.566784 | 27.33% |
| CUDA Graph，replay 前写 256 MiB 缓冲 | 0.780272 | 0.567168 | 27.31% |

这说明正收益在本次 eager/graph 和调用前缓存扰动条件下均存在。扰动不保证 K2 冷缓存，因为 K1 会先生产 workspace；也没有其他 stream 持续竞争资源。它不是生产并发测试。[Job19845 原始日志](followup_19845.log)

原主线 Job17947 已给出非常有价值的负例：同样 total T8192/H12，packed 单序列 V16 降时约 26.9%；ragged6 强制 V64 可降时约 17.4%；32×256 强制 V16 却慢约 106%。因此 total tokens 不是充分的调度特征，长度分布和自然并行度不可省略。[原始 slice/auto sweep](../../../../experiments/final_campaign/data/raw/05_dispatch_upgrade_17947.csv)

原报告表中“8×1024 因低于 3% guard 而 fallback”的说法应区分经济解释和实际控制流：**当前代码先因多序列 varlen 未标定而回退，并未走到预测收益阈值。** 1.3% 的观测收益可以解释为何保守回退合理，但不是该次 decision 的实际触发原因。

### 3.3 `tcgen05` 的停止决策保留，但关键论据必须改写

旧主线用 V128/grid12/inner64 的 L0 比值 0.920×，支持不投入机械集成。当时这是合理的阶段性工程取舍；原报告也已经注明它不能排除所有新数据流。问题在于摘要、总结和停止理由过度依赖“64 次已经充分摊销”的印象。

若 `T(n)=a+b·n`，则 `T(64)/64=a/64+b`，固定项没有消失。B300 Job19844 扩展到 inner1/4/16/64/96/128/256/512，两轮同 job 的 V128 L0 结果如下，`mma_time/tcgen_time>1` 表示 TCGEN 更快：

| grid | inner64 | 首个获胜采样点，两轮 | inner512，两轮 |
|---|---:|---:|---:|
| 12 | 0.9824 / 0.9831× | inner96：1.0236 / 1.0270× | **1.2119 / 1.2116×** |
| 148 | 0.9039 / 0.9039× | inner128：1.0129 / 1.0129× | **1.1377 / 1.1377×** |

新结果证实：**这个 L0 probe 在足够多次重复下能够出现 TCGEN 正收益。** 不能继续把 inner64 的负值当成 V128 充分摊销后的普遍结论。新旧 batch 的点也不能拼在一起拟合更精确曲线。[两轮原始 CSV](tcgen_sweep_19844_1.csv)、[第二轮](tcgen_sweep_19844_2.csv)

但它不构成发布 TCGEN 的证据，原因具体而非保守口号：

1. L0 反复使用不变 A/B，不能代表真实每 chunk 都变化的 U/state；T8192/C16 恰好有 512 个 chunk，并不使两个循环等价。
2. 包含 state/gate 和保守 scalar U 重排的 L1，inner512 仍仅约 0.261×。它暴露当前集成方式的成本，但不是所有优化实现的上界。
3. L0 两条路径的 staging、global 写出布局并不相同，不能把它称为完整集成的严格性能下界，也不能把全部固定项归因于 TMEM 分配。
4. 下一候选应与**当前最佳 ValueSlice，包括 V16**比较，不是只打败旧 V128。旧 V16 L0 已有形状相关正信号，但 L1 仍为负。
5. 原探针 L0 的逐元素检查只覆盖 inner1；新增 L1 检查覆盖 grid12 的 inner1/2/4/128。长 inner timing 不是完整正确性矩阵。

Phase-6 的 `128×16 @ 16×V_slice` 有合法 TCGEN 映射；其他以 C16 为 M 的 phase 不能在原朝向直接照搬。**真正要验证的单元应是“生产者布局—MMA—消费者—状态更新”这一段，而不是单条 atom。** TCGEN 的 TMEM 布局与异步完成协议是实际设计约束。[NVIDIA PTX：第五代 Tensor Core 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma)

### 3.4 数值审查：保护既有契约，不扩大承诺

原 Job17929 的官方扩展与 patched V128 共 10 个 tensor bitwise equal；Job17934 的 98 条 ValueSlice 比较也全部 bitwise equal。其余独立参考关系和 finite 统计应继续保留，不能把“200 行 finite”改写成“200 行统一误差阈值通过”。长序列参考误差的观测最大值为 0.9131%，不是模型质量证明。[正确性原始 CSV](../../../../experiments/final_campaign/data/raw/03_reference_correctness_17934.csv)

新探针中，整段与 16-token 对齐分段在已测小样本逐位一致；逐 token 分段在长记忆 gate 下与整段的 final state relative RMSE 为 2.246%。**这是两种执行分段的差异，不是相对 ground truth 的误差；V128/V16 一致，不是 ValueSlice 新增误差。** FP32 public-state buffer 也不代表内部 FP32 recurrence，空序列会量化初态。这些都应约束未来 chunk 调度/状态接口的声明，不能据此要求当前 ValueSlice 改变 state 精度。

对不改变舍入组织的发布路径，首要条件是 baseline 等价。若下一阶段 TMEM 数据流改变累加/舍入顺序，原 bitwise 证据不能自动继承：必须单独定义误差标准、独立参考和必要的模型级检查，不能为了通过测试临时放宽容差。

## 4. “guarded 发布”目前还差什么

### 4.1 实际策略比一句“低并发长 prefill”更具体

[dispatch.py](../../../../experiments/final_campaign/implementation/current/flash_kda/dispatch.py#L126) 的实际边界是：

- CC10.3、148 SM、L2 容量在 132,644,864 B 的 ±5%；`batch×heads` 在 1–96。
- 非 FP32 public-state 模式在 T2048/4096/8192 有标定点，中间 T 使用线性插值；FP32 public-state 模式只覆盖 T4096。
- fixed 和合法 packed 单序列进入该模型；packed 多序列直接回退 V128。单序列判断只读 `numel()` 元数据，不引入 GPU→CPU 同步。
- 候选受资源可行性及最多两层 CTA 的标定范围限制，再经过预测收益 3% 和 5 µs 门槛。

这是**离线标定的保守策略**，不是在线自适应控制器：`reuse_over_l2` 只是诊断字段，resident-block 数主要用于可行性筛选；没有实时 L2 压力、其他 stream 占用、在线测速或性能退化检测。中间 T 有插值不等于逐形状验证，3% + 5 µs 也不是置信区间。

### 4.2 按对主线发布的真实影响排列问题

| 发现与依据 | 证据等级、归属 | 主线发布处理建议 |
|---|---|---|
| 未设置环境变量时默认 `auto`；强制 slice 的优先级高于 `DISPATCH=off` | 源码确认；当前 Python policy | 统一文档与实际默认值，声明 override 优先级。不能说当前默认 opt-in，也不能把 off 当无条件总开关 |
| V128 fallback 仍调用同一 `_fwd_raw` | 源码确认；策略范围 | 它不是 FLA fallback；不能处理无兼容二进制、坏输入、错设备或地址不齐，也没有失败后自动重试 |
| contiguous beta offset view 可触发 TMA host assertion / SIGABRT | B300 clone 对照复现；上游继承 | 通用接口发布前修复或在入口明确检查/拒绝；受限预填充候选也须声明对齐前提，V128 不能兜底 |
| C++ 没有持有输入设备的 CUDA device guard | 静态确认，未双卡复现；上游继承 | 若支持任意当前设备或多 GPU 集成，应补 guard 与同设备检查；不否定正确设备上的已有单卡结果 |
| V16 编译别名的默认参数仍可能是 V128 | C++/nvcc flags 工厂检查复现；补丁新增 | 修复宏传递，或把别名排除正式接口。现有正式 wrapper 显式传 slice，主性能结论不受影响 |
| FP32 空 state 量化、非连续 offsets 未拒绝 | 前者 GPU 复现，后者静态；上游继承 | 明确 dtype/空序列/contiguous 契约与入口校验；不是 ValueSlice 引入的数值回归 |

默认与 override 的直接依据是 [wrapper 第 28–36 行](../../../../experiments/final_campaign/implementation/current/flash_kda/__init__.py#L28)。在当前实现中，强制 `FLASH_KDA_K2_VALUE_SLICE=128` 可明确选择旧变体；只设置 `DISPATCH=off` 而遗留强制 V16 并不能关闭切片。这是发布/回滚说明必须写对的细节。

对齐问题最小对照为 BF16 beta storage 的 `[1:3]` view reshape 成 `[1,1,2]`：contiguous 为真，指针余 2，子进程 −6；clone 后余 0，正常完成。它验证的是当前通用 forward 接口，不是 Kimi 真实 decode 路径已经崩溃。实际 TMA base 应统一审查；不能只再调用一次 `.contiguous()`。相关位置：[C++ beta 转置及 stream](../../implementation/upstream/csrc/flash_kda.cpp#L131)、[编译宏问题](../../../../patches/0001-k2-value-slice-and-dispatch.patch#L893)、[完整代码审查](code-review.md)。

### 4.3 复现入口与构建证据要独立于算法正确性收口

[最终提交入口](../../../../docs/c1-final/README.md) 已正确要求依次应用 0001、0002，历史审计中“缺少第二补丁”的最终入口问题已经关闭。但[项目根 README](../../../../README.md#L50) 仍只写 0001，并指向旧草稿。对明确从最终目录阅读的课程交付，这不是缺失核心补丁；对公开仓库发布，它仍是可复现性陷阱，应只保留一个醒目的权威入口。

此前已验证两个补丁顺序应用、Python 快照一致性及官方/补丁 baseline parity。本次 GPU 探针加载的是服务器现成 patched 扩展，不是重新从空构建目录生成 release 包。发布验收应再完成一次：固定上游/CUTLASS → 两个补丁 → 干净构建目录 → 打印实际 `.so` 路径/校验和 → 正式 wrapper 的 auto/强制/fallback 测试。不要用模块名或源码 hash 代替已加载二进制的身份。

**以上区分是必要的：上游通用接口问题决定能承诺多大的发布范围，不应被写成“ValueSlice 算法失败”；已有 bitwise 和性能测试也不能反过来证明通用接口安全。**

## 5. 最值得保留的五个 Insight

### A. 最优指令属于一段数据流，不属于单独的矩阵形状

官方 K2 已紧密融合多个 phase，并通过寄存器转置连接 U 的生产者与消费者。TCGEN 要胜出，需要减少整段的转换/同步/物化，而不只是提升 Phase-6 FLOP/s。L0 crossover 提供继续探索的理由，L1 的失败则指明原型必须包含什么。新研究单位应是**跨 phase 的布局和生命周期**，不是整个模型或单条 opcode。

### B. B300 专版的核心价值是匹配每卡工作量，而非宣布某代指令过时

相同 H96 模型在 TP1 和 TP8 的每卡自然并行度不同。ValueSlice 解决的是低并发长 prefill 的局部工作不足；随着请求数上升，最优 slice 会变粗。其贡献可以概括为：**把生产约束下仍可自由分配的 Value 并行度，映射到当前硬件资源。** 这比“新卡一定需要新 MMA”更准确，也更可迁移。

### C. 数据复用、可调度并行度和每 chunk 等待链必须一起优化

ValueSlice 以重复公共请求换并行度；cluster/multicast 试图拿回复用，但会增加共同驻留与同步约束。81.3% 只是旧模型的理想 source-request 减少，不是实测 HBM bytes 或延迟节省。既然 HBM 未饱和，减少 bytes 未必缩短关键路径；也可能通过减轻 TMA/缓存请求压力获益，需实测区分。[PTX 的 cluster multicast 语义](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)

因此不能预先宣布 cluster 必然是下一项最佳实现。先比较 V16 的 TMA 等待、计算依赖、输出 store 背压及真实 L2/DRAM traffic，才知道应优化哪项。源码已经有 load/store warp specialization 和多阶段 pipeline；后续是对现有流水线做定向消融，不是泛泛“加异步/双缓冲”。[K2 pipeline 与 load warp](../../implementation/upstream/csrc/smxx/fwd_kernel2.cuh#L188)

### D. Guard 的价值是界定损失，不只是捕获最漂亮的加速点

目前最可靠的正收益在 H12 长单请求；完整 policy 还包含插值及更宽 sequence-head 域。下一阶段应报告 auto 相对最佳变体的 regret，以及相对 V128 的最坏退化，而不只给平均 speedup。没有真实请求分布，就报告分形状收益面，不制造“生产平均提升”。

### E. 更改执行切块可能改变数值，ValueSlice 的无损优势应当珍惜

当前 ValueSlice 保持既有舍入组织，因而能获得 bitwise 证据，这是很强的交付优势。时间切块、跨 phase TCGEN、FP32 常驻 state 或大 CHUNK 属于不同风险层级，不能借用这一等价性证明。优化应先花在不改变模型和舍入契约的自由度上；要改变契约，则独立立项和验收。

## 6. 后续路线：一个发布里程碑，两道优化实验门槛

### 第一优先：形成“可复现、可解释、可退回”的主线候选版本

交付物是一个明确的 release candidate，不是新算法：

1. 更新最终摘要/答辩中的 TCGEN 停止依据：加入长 inner L0 crossover，保留完整 K2 尚无净收益的结论；纠正严格下界、充分摊销和 fallback 触发原因的措辞。
2. 统一根入口、最终入口、两补丁顺序、二进制身份、默认 auto/override 与强制 V128 的操作说明。选择 opt-in 发布时要让实际开关行为与声明一致。
3. 按所声明接口范围处理对齐/device guard，修复或排除编译别名；明确 public state、空序列、offsets 和 alias 的契约。不为此重写 state 算法。
4. 从干净构建目录运行正式接口验收：官方 vs patched V128、auto vs 选中变体、域外 fallback、强制路径；将已复现的异常纳入独立进程测试，避免一个 abort 吞掉全部结果。
5. 给出 policy 的已验证/插值/域外三种标签；未补齐的格点可以暂时保守回退，不需要为了扩大宣传域先外推。

建议的验证矩阵如下，属于后续实验设计，不是假称已完成：

| 维度 | 优先覆盖 | 回答的问题 |
|---|---|---|
| 生产尺寸映射 | 固定 D128；H96/48/24/12 对应题面模型 TP1/2/4/8 的每卡形状 | 模型不变时，TP 如何改变最佳 slice；单卡模拟不等于多卡实测 |
| 请求并行度 | H12，fixed B1/2/4/8；packed 单序列及多序列 | 自然并行度增加后的收益/退化；不把总 token 数当唯一特征 |
| 长度 | 标定 T2048/4096/8192；留出 T3072/6144；域外短 T 和 T16384 | 插值是否可靠、fallback 是否真的走到 V128；阈值附近另用诊断形状 |
| 数据与状态 | 随机/长记忆 gate，initial/final/None，BF16/FP32 public state，尾块及合法 packed offsets | 等价性与 API 边界；不混淆内部精度 |
| 执行环境 | eager/graph；持续并发 stream 与混合请求；多 GPU device-mismatch 单独测试 | 缓存/调度标定能否外推到目标调用环境 |

性能指标至少包括 `auto / 最快强制变体 − 1`、`auto / V128 − 1`、绝对节省 µs、各重复轮次分布。预先约定可接受退化与测量噪声，不能把已有预测 3%/5 µs 当成统计保证。取得真实流量权重后再计算加权延迟；端到端还须测 FlashKDA 在 prefill critical path 的占比，不能用 69/93 的层数比例代替。

### 第二优先：定位最佳 V16 的剩余关键路径，再选择低风险优化

在固定 H12/T8192/D128 和代表性高并发反例上，对当前最佳路径做一致口径的 timeline/NCU 分解。先验证剩余时间更接近哪一种约束：输入 producer 等待、state/计算依赖、输出 store 背压，还是资源竞争。只对被证据指向的 pipeline stage、warp 分工或布局做小范围消融，保持 C16 与舍入点。

一个可独立审查的小机会是 packed tile prefix：K1 前的 GPU prefix kernel 已生成 `ws_tile_prefix`，K2 却为每个序列重新线性扫描前序长度。按 CTA 逻辑工作量，总循环项为 `H·(128/V_slice)·N(N−1)/2`；这不是直接测得的内存事务数。可传入已有 prefix，消除重复扫描而不增加 host sync；但必须覆盖 K2-only profiling 模式的 prefix 初始化。[prefix 生成](../../implementation/upstream/csrc/smxx/fwd_launch.cu#L165)、[K2 重扫](../../implementation/upstream/csrc/smxx/fwd_kernel2.cuh#L224)

**这个机会属于多短 packed 工作量，对 N1 核心加速点几乎没有价值。** 只有目标 trace 确认该负载重要，才将它提到主优化队列；不能拿渐近复杂度代替实测收益。workspace/descriptor 复用同理，应先测 CPU wall time 和调用开销，不先假定它是长 prefill 主瓶颈。

### 第三优先：两条架构探索都只做有停止条件的原型

| 候选 | 最小有意义实验 | 继续投入条件 | 停止条件 |
|---|---|---|---|
| ValueSlice + cluster/TMA multicast | 固定生产 D128/C16，2/4/8 CTA 组；同形状对比普通 slice、共同驻留但不 multicast、实际 multicast，统计真实 traffic/等待/完整 forward | 在计入 cluster 同步与 residency 后，优于当前最佳 slice，且保留数值契约 | 只减少请求数、没有可靠净降时，或短/并发负载明显恶化且无法安全区分 |
| 选择性跨 phase TCGEN/TMEM | 先接入真实变化的 U/state、真实 BF16 舍入与生产者/消费者边界；局部通过后才测完整 K2/forward | 完整边界比最佳 `mma.sync` ValueSlice 更快；正确性、资源和维护成本都可接受 | 只有固定 operands 的 L0 更快，物化后收益消失，或必须缩小模型/修改门控才能获胜 |

第二条不是要求所有 state 永远常驻 TMEM，也不是要求一次迁移所有 phase；应以最小但完整的依赖片段验证布局复用。第一条也不是先默认 8-CTA 最大组最佳。所有新候选的基线都应是**当前最好路径**，并保留 V128 作为稳定对照。

如果只能做一件事，我会选择第一优先的发布收口；如果还能做一个新的性能实验，我会选择**最佳 V16 的关键路径与公共输入代价消融**。它决定 cluster 与 TCGEN 哪条更值得后续工程投入，比再扩大一轮孤立 MMA sweep 更接近可交付收益。

## 7. 主线外内容的位置与完成标准

Loop transformer、跨 loop 状态、低精度模型改造和新 memory architecture 暂不进入主线待办。此前探索保留为附录；只有能在**不改变当前模型和数值契约**的前提下解决 FlashKDA 的实际调度/驻留问题，才允许回到主线。完整 loop-state reference 不是本次发布的前置条件。

课程研究的完成标准已经基本满足；新增审查要求的是修正论证和交付边界，而不是否定已有工作。受限版本发布的完成标准是：**文档与实际 policy/二进制一致，承诺的输入域通过测试，guarded 收益可复现，域外与回滚行为清楚。** 通用生产部署则另需真实 serving 负载、多卡/并发、TTFT/TPOT/SLO 验收，不能在本次单卡算子结果上提前签字。

## 8. 证据、版本和本次验证边界

- 主线基线：FlashKDA `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`、CUTLASS `5c149f5`；题面 FLA snapshot `a3edffc`，原参考 campaign 使用 FLA0.5.2 并关闭 FlashKDA 回落。[原最终报告](../../../../docs/c1-final/FINAL_REPORT.md)
- 已连接 `b300-login` 完成的审查实测：Job19844 两轮 sweep 完成，后续数值测试遇 alignment abort；不能把整个 job 写成全通过。Job19845 完整结束，未对齐输入在隔离子进程按预期 −6 退出，clone 对照正常。[19844 日志](review_19844.log)、[19845 日志](followup_19845.log)
- 本卡 CC10.3/148SM，PyTorch2.10.0+cu130，driver580.126.09。服务器 final wrapper SHA-256 `c638962a3d333680e923884ba47ffcd1cc4f26b1db4b2097af9bfa01b0b4f50f`，dispatch `74e59195d1bdad5a68f3ad9793d722c8195d4f5de3266f8609526d2360ac59b8`，此前已与本地快照核对。
- 本轮重新阅读题目、最终报告、当前 policy、相关 kernel/launch、官方设计说明与论文；重新运行原始日志/CSV 的数值汇总，并进行独立的发布范围交叉审查。没有新增 GPU 作业，没有修复生产实现。
- 可复算工具：[summarize_review.py](summarize_review.py)、[编译宏只读证明](code-review-proof.py)、[GPU 探针](review_probe.py)。详细限制见[性能审查](performance-review.md)。

未完成的内容仍是未完成：全新 release 构建验收、双卡错设备复现、持续竞争条件的 policy 验收、cluster 实现、真实跨 phase TCGEN 实现、完整 Kimi serving 和模型质量测试。本报告没有把建议中的实验当成结果。
