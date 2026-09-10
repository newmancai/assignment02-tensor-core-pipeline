# C1 FlashKDA → SM100：完整研究复盘与后续指导

团队：奶龙必胜；成员：蔡雨洋、李奥、赵骋

## 1. 研究问题如何被真正回答

C1 的表面问题是“官方 kernel 使用 SM80 MMA，迁移到 SM100 是否值得”。如果把“迁移”理解为替换一条矩阵乘指令，结论会被局部实验误导。FlashKDA 是带递推 state 的完整算子，真实性能由矩阵乘、数据布局、TMEM 生命周期、warp 分工、同步、pipeline、并行工作量和 dispatch 共同决定。

本项目最终把问题分成三个层次：

1. **指令与机制层**：tcgen05 在合法 shape 下是否具有计算潜力，必要协议和布局成本有多大；
2. **算子与路径层**：在相同 public KDA 语义下，完整 SM100 路径是否稳定胜过当前官方与强化 HMMA；
3. **部署决策层**：哪些设备和 workload 应进入 SM100 route，哪些应回退 HMMA。

由此得到的主结论是：**迁移值得，但迁移对象应是经过 guard 的完整 SM100 后端，而不是机械的 instruction swap。**

## 2. 三步主线的完整闭环

### 2.1 复现与测量：先证明机器实际执行了什么

项目固定课程指定的 FlashKDA 与 CUTLASS 版本，在 NVIDIA B300 SXM6 AC 上为 `sm_103a` 构建。我们没有根据源文件名或编译目标推断指令，而是记录实际加载扩展的路径和哈希，并检查 SASS。官方 recurrence 路径中有 3,640 条静态 HMMA，TCGEN/UTCMMA 为 0。

这条证据只支持“课程 pin 的官方递推矩阵乘在 B300 上仍使用 HMMA”。它不等于“整个 kernel 都是 SM80 实现”，也不否认其使用 TMA、SM100 编译目标或其他跨代能力。

测量合同同时固定输入、非零初始 state、packed boundary、dtype、output 和 final state。KDA 是递推算子，只验证当前 output 会漏掉跨调用状态错误。因此正式正确性 gate 始终覆盖 output 与 final state，并在需要时加入连续两次调用。

正式 headline 使用 public-full GPU span：JIT 与一次性分配在计时外，每次 forward 必需的布局转换、workspace 流量和 state copy-back 在计时内。CUPTI 测量使用 cold L2、20 次 warmup、每 block 100 次采样和 4 个平衡顺序 block。

一次关键的内部有效性修正来自扩展符号互插：官方与 H3 的两个 `.so` 导出同名 weak/global CUDA symbols，改变加载顺序会改变时间。之后所有正式 magnitude 改为 one-implementation-per-worker-process；旧的同进程结果保留为开发诊断，不再承担 headline。

### 2.2 分析：把“快或慢”还原成物理原因

代表性的 TP8/H12 official recurrence 只有 12 个 CTA，而 B300 有 148 个 SM；历史 NCU 中 SM throughput 和 DRAM throughput 分别只有 2.64% 和 1.24%。因此该 profile 既没有打满 Tensor Core，也没有打满显存带宽。更准确的描述是 **parallelism / occupancy / critical-path limited**。

HMMA ValueSlice 进一步提供了干预证据：把 value dimension 切分后，K2 grid 从 12 CTA 扩到 96 CTA，fixed T8192 的 latency 从 0.7807 ms 降到 0.5698 ms，约降低 27%。这说明至少在 H12 条件下，增加独立工作比更换矩阵乘指令更直接。

CHUNK=16 的分析也从“固定常量”转成约束组合：它同时匹配指数数值范围、16×16 Neumann 求逆成本和 HMMA K16 形状。机械放大到 32/64 会先撞上当前无 rescale 指数恢复的数值边界，再放大朴素求逆成本；这不否定重新设计 rescale 或 block solve 后的大 CHUNK。

### 2.3 挑战：同时允许正结果和负结果改变结论

挑战阶段没有围绕单一路线堆调参，而是比较多种改变层次：

| 路径 | 改变层次 | 结果 | 对主线的贡献 |
|---|---|---:|---|
| Direct V128 tcgen05 | instruction-only microprobe | 0.919742× | 证明裸替换不构成迁移理由 |
| V16 preferred-layout L0 | thin-N core mechanism | 1.615021× | 证明 tcgen05 核心存在局部潜力 |
| V16 scalar-rematerialization L1 | 加入当前布局接入成本 | 0.778523× | 证明 carrier/layout 可吞掉计算收益 |
| HMMA ValueSlice / dispatch | 并行度与 profile 分派 | 局部 1.076×–1.30×；跨 profile 可反转 | 证明 HMMA 仍需公平优化 |
| H3 P/W 重排 | 算法与数据流 | H12 0.9055×；H96 0.7581× | 完整负结果，按 gate 停止 |
| CAKE 完整 SM100 | tcgen05/TMEM + 数据流/角色/流水/dispatch | H12 2.4823×；H96 2.2532× | 给出完整迁移值得的存在性证据 |

最有价值的不是某个最大数字，而是相互制约的三条证据：

- Direct tcgen05 失败，说明“新 ISA 峰值更高”不足以推出算子加速；
- HMMA 仍能通过并行度与 dispatch 优化，说明公平基线不能停在官方 H0；
- CAKE 在九个 public profiles 全部正收益，说明 SM100 的价值需要通过完整数据流工程释放。

## 3. 主线在探索中经历的五次修正

### 修正一：从“指令迁移”改为“后端迁移”

早期问题容易被写成 HMMA 与 tcgen05 的一对一替换。Direct V128 和 V16 L0/L1 证据表明，指令本身、协议接入和生产 carrier 是不同问题。最终论文和答辩统一使用“完整 SM100 路径”与“guarded backend”作为部署对象。

### 修正二：从“官方 HMMA”改为“优化后 HMMA”作为公平对手

官方 H0 是实际部署基线，但不是 HMMA 技术上限。多轮 HMMA 分支发现 V64、V32 和 lookahead 的 profile 边界，说明 HMMA 仍有真实优化空间。公平因果问题因此被拆为 H0、H1、T0、T1、T2，而不是只比较官方与 CAKE。

### 修正三：从“单点 winner”改为“profile 矩阵”

H3 在一个 fixed T8192 case 达到 1.0219×，但正式几何平均低于 1；V32 在 nseq=3 提升约 7.6%，到 nseq=4 又变慢。单点结果会掩盖尾部、packed、head 数与 wave 边界。最终只有覆盖预注册 profile 矩阵并通过 gate 的候选才能晋级。

### 修正四：从“计时器输出”改为“执行身份 + 测量隔离”

同进程符号互插说明一个合理的时间数字仍可能对应错误的被测对象。正式证据链增加源码、构建和二进制身份，并将相互比较的扩展放入隔离 worker。这一修正比继续增加采样次数更重要。

### 修正五：从“agent 记住结果”改为“agent 记住适用条件”

HMMA 与 tcgen05 的物理机制不同，直接迁移“V16 好”“某个 schedule 坏”会造成刻舟求剑。双层 IR 最终只允许跨路线迁移验证方法、物理变量和重开条件；性能数字、winner、stop 决定必须绑定原路线和 scope。

## 4. MARPE 在项目中具体做了什么

MARPE 的中文含义是 **多智能体运行时画像演化**（Multi-Agent Runtime Profile Evolution）。它被放在数据分析之前，因为数据不是先被动产生、再由 agent 总结；agent 框架决定了候选如何提出、如何分配 GPU、什么证据可以晋级、失败如何写回下一轮。

每一轮的闭环为：

1. 从 Program IR 读取真实执行路线、语义合同与物理结构；
2. 从 Experience IR 检索与当前 scope 匹配的观察、反例和重开条件；
3. HMMA 与 tcgen05 两个分支各提出一个结构假说；
4. 在相近新增机会预算下分别编译、校验身份、检查 output/state 并做性能 screen；
5. 达到门槛的候选进入更严格 qualification；
6. 正负结果都以可追溯 writeback 更新经验，而不是只保存 winner。

框架来源与改造边界已经在论文中明确：

- 复用 PIKE 的并行分支搜索与候选分配思想；
- 复用 Atrex Kernel Agent 的隔离 GPU 执行、profiling、Git episode 和 ABBA 验证能力；
- 本项目新增 HMMA/tcgen05 双分支 Program IR、带作用域的 Experience IR、公平预算协议和证明式候选晋级。

当前实验是窄宽度双分支闭环。它证明了框架可以组织证据、暴露反例并约束结论，但没有匹配的单 agent 消融，因此不能声称 multi-agent 本身带来了性能优势。

## 5. 双层 IR 应该怎样组织，才能积累经验而不误导

### 5.1 Program IR：记录“实际跑了什么”

至少包含：

- semantic contract：输入、dtype、output、final state、packed 语义和舍入要求；
- execution identity：source、build、module path、binary hash、实际 route；
- physical structure：logical/physical MNK、tile、ISA、layout、storage、warp roles、barrier、pipeline；
- machine context：设备、物理 SM 数、编译目标、driver/CUDA；
- measurement scope：microprobe、kernel-only、public-full、single-call 或 two-call。

Program IR 的作用不是让所有新方案符合旧模板，而是确保候选执行后能回答“它到底改变了什么”。新 carrier 无法表达时，应扩展 IR，而不是把未知字段当通配符。

### 5.2 Experience IR：记录“这个结果允许怎样使用”

每条经验建议固定为六部分：

1. **Observation**：在什么条件下观察到什么；
2. **Mechanism status**：只是相关、阶段定位，还是有受控干预；
3. **Scope key**：路线、profile、硬件、语义、测量层级；
4. **Allowed use**：用于排序、复测、停止当前候选或部署；
5. **Forbidden extrapolation**：不能推到哪些路线或 workload；
6. **Reopen key**：出现什么新结构、新证据或新环境时重新验证。

例如 direct V128 tcgen05 应写成：“在 B300、Phase-6、V128、grid12、inner64 的 L0 条件下为 0.919742×；相同候选无需重复，但 thin-N、跨阶段 TMEM 驻留或新布局会重新开启。”不能压缩成“tcgen05 更慢”。

ValueSlice 应写成：“在 H12 underfill 条件下增加独立 CTA 有效；迁移时复用寻找并行工作和核算复制成本的假说。”不能写成“V16 永远更快”。

## 6. 八轮双分支探索真正积累了哪些知识

### HMMA 分支

- H12 nseq=6 时 V64 相对 V128 为 1.17–1.22×；到 nseq=7/8，收益降到实用门槛以下。
- nseq=3 时 V32 相对 V64 约 1.076×，并在偏斜输入复现；nseq=4 时降到 0.9618×。
- Phase-1 lookahead L3 相对 L2 为 0.999978×，在相同 public extension scope 下性能中性。
- 因此有效知识是按 CTA wave 与 profile 选择分段 route，继续堆 lookahead 深度缺少价值。

### tcgen05 分支

- H96 s173 对 CAKE 在 exact profile 的四块资格验证达到 1.0330×。
- virtual-152、TAIL4、MINIMAX172 和理论 load-minimax MAX169 均失败；MAX169 为 0.9383×。
- 将 scalar rematerialization 移出重复生命周期后，preferred-layout 相对 scalar control 提升 1.227×，但相对 HMMA 仍只有约 0.955×。
- 因此调度的数学目标不能代替端到端延迟；下一条高价值路线应改变生产 carrier、跨 phase TMEM 生命周期或舍入边界。

### 共同知识

- route identity 比 backend 名称更可靠；
- public-full 比孤立 core 更接近部署价值；
- output 与 final state 是同一正确性合同；
- 负结果用于缩小条件空间，不用于永久封禁一条 ISA；
- 当参数组合已被多个独立反例覆盖时，继续搜索需要引入新信息，而不是增加相同旋钮的排列。

## 7. 公平预算应怎样理解

第八轮冻结的公平协议给两侧各一个结构假说、有限的 repair/compile/correctness/screen/qualification 机会，并统一设备、语义合同、晋级门槛和身份要求。这是 **matched opportunity**，不是历史成熟度、代码量、累计人工或 wall time 完全相等。

这种预算适合回答“在相同新增探索机会下，两边当前还能发现什么”。它不能单独证明两种 ISA 的全局上限，也不能把一个已有成熟 carrier 与一个尚未接入 production 的 probe 当成严格因果对照。

## 8. 当前可以提交的结论

### 工程结论

在已测 B300 public profiles 上，完整 SM100 路径相对官方 HMMA 的 H12/H96 几何平均分别为 2.4823× 和 2.2532×，九个 profile 全部正收益。建议采用：

1. 保留 HMMA 作为跨代兼容和未覆盖 workload 的 fallback；
2. 为通过正确性、身份和 public-full 资格验证的 B300 profiles 启用 SM100 route；
3. dispatcher 按设备、head、序列形态和已验证 guard 选择路径；
4. 新 profile 进入前重新执行 output/state 和性能 gate。

### 研究结论

本项目证明了“完整 SM100 工程迁移存在显著价值”，同时证明“直接 tcgen05 替换”和“某个数学调度目标”都可能失败。它还建立了一个能把这些冲突结果保存在同一知识系统中的双层 IR 与双分支闭环。

### 尚需下一阶段闭合的问题

- 构建生产级 P3/P4 preferred-layout carrier；
- 在完全共同的 profile 与 public-call 边界上完成 H1/T1/T2 四块资格比较；
- 将 driver、CUDA、clock、power 和 thermal 状态直接写入每个正式 worker 回执；
- 若要回答产品级“全面迁移是否值得”，再加入真实流量占比、SM100 部署覆盖与维护成本。

这些是从“存在性与工程建议”走向“纯 ISA 因果与产品 ROI”的下一层问题，不影响当前 guarded backend 的结论。

## 9. 后续 agent 探索协议

1. **先选 scope，再检索经验。** 未知字段不得当作匹配。
2. **两条路线保留独立先验。** 跨路线只传递验证方法、物理变量和反例形式。
3. **每个提案必须说明新增信息。** 新 carrier、新 oracle、新 profile 反例或新理论下界至少具备一项。
4. **先做最便宜的证伪实验。** 编译与身份 → output/state → 两块 screen → 四块 qualification。
5. **晋级携带证据。** 候选必须带 source/binary identity、正确性、原始 samples、置信区间和适用范围。
6. **负结果保留 reopen key。** 禁止用一个失败 microprobe 阻断整条 tcgen05 路线。
7. **饱和按冻结边界命名。** 当前只能称“冻结 B300 与现有 carrier 下的默认高价值提案饱和”。
8. **下一阶段优先改变表示。** 当前最高价值工作是 production carrier 和共同 profile qualification，而不是增加相同 schedule 参数组合。

## 10. 答辩时如何讲清楚

十分钟答辩应保持原始三步主线：

- **复现与测量**：我们先证明官方实际执行 HMMA，并建立完整正确性和隔离测量合同；
- **分析**：低利用率来自独立工作不足，HMMA 仍有 profile 优化空间，tcgen05 的 core 潜力会被 carrier 成本吞掉；
- **挑战**：多条真实路线同时给出正负证据，最终完整 SM100 路径胜出，因此建议 guarded 发布。

MARPE 应在数据页之前出现，用一句话解释其作用：它不是答辩的独立主题，而是让两条路线在相近机会、相同 gate 和可追溯记忆下持续产生这些证据的研究工具。

现场最稳妥的一句结论是：

> 我们没有把新指令当作答案，而是把迁移拆成可验证的多条路径。实验表明局部替换会失败、优化后的 HMMA 仍有竞争力，但完整 SM100 数据流在已测 B300 profiles 上获得稳定收益，因此适合以带 guard、可回退的专用后端发布。
