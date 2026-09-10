# FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得

> **唯一主线问题：FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得**
>
> 原始任务文件标题“C1: FlashKDA——官方 kernel 停在 SM80 MMA”仅是背景标题，
> 不是另一条主线，也不能替代上面的主线语句。

## 一、原始问题

Kimi K3 的线性注意力部分使用 KDA（Kimi Delta Attention）。MoonshotAI 开源的
FlashKDA 已经是高性能 forward kernel，但其矩阵乘计算路径在 GB200/B300 上仍使用
SM80 世代的 `mma.sync`，而不是 SM90 `wgmma` 或 SM100 `tcgen05`。原始任务要求把
“为什么停在 SM80 指令”的量化论证做出来，或者用实现和实验推翻它。

原始交付要求为三层：

1. **复现**：在 B300 安装并运行官方 FlashKDA，用 benchmark、NCU 和 SASS 确认
   主计算路径及硬件行为；
2. **分析**：对 CHUNK、tcgen05 tile、递推并行度、性能瓶颈、BF16 state 精度和
   SM100a 专版决策逐项给出“结论 + 证据”；
3. **挑战**：实际实现并验证至少一条 SM100/算法/并行度路线，正确性对参考实现，
   性能对官方 FlashKDA；负结果同样作为结论。

本轮从始至终只回答下面这一条主线，不把某个讨论点或中间方案偷换成总问题：

> **FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得。**

## 二、最终回答

**值得发布 shape-guarded 的 SM100 专用路径，但不值得将“迁移”理解成机械替换
MMA 指令。**

在一张 NVIDIA B300 SXM6 AC、SM103a、BF16 public KDA forward 的已测条件下：

- 官方 recurrence SASS 有 3,640 条静态 HMMA，TCGEN/UTCMMA 为 0；
- 直接 V128 `mma.sync -> tcgen05` probe 为 0.919742x，裸替换没有收益；
- V16 preferred-layout core 为 1.615021x，但加入当前 scalar rematerialization 后为
  0.778523x，布局接入成本吞掉计算收益；
- H3 P/W 算法/数据流重排通过数值、SM103a 编译和 B300 correctness，但正式
  isolated-process CUPTI 中 H12/H96 geomean 仅 0.9055x/0.7581x，停止；
- CAKE 的完整 SM100 路径在同一 public contract 的正式 isolated-process CUPTI 中，
  H12 六形状 geomean 为 2.4823x，H96 三形状为 2.2532x，九形状全部正收益。

所以证据同时支持两件事：

1. 官方保留 HMMA 不是简单的“没有适配新 ISA”；对现有局部形状直接替换可能更慢；
2. 如果重做数据驻留、warp roles、流水、融合和 dispatch，完整 SM100 路径可以大幅
   胜过当前官方实现。

CAKE 的收益是 **full-stack treatment effect**，不是 `tcgen05` 单因素收益。当前可以
回答“从当前官方实现迁移到完整 SM100 工程方案是否值得”，尚不能精确回答“相对充分
优化的 HMMA，tcgen05 指令及其专属优化分别贡献多少”。

## 三、原始六个讨论点的回答

### 1. CHUNK=16 为什么成立，32/64 谁先破？

CHUNK16 同时平衡当前指数数值范围、16x16 Neumann 求逆代价和 HMMA 的 K16 自然
形状。机械扩大常量时，首先失效的是当前无 rescale 指数恢复路径：在既定最坏门控
模型下 C32/C64 都在第 18 个 token 首次出现 FTZ/overflow。其次是朴素扩大 Neumann
密集幂级数，按每序列计算模型分别增至 5.33x/26.67x。

这没有否定“大 CHUNK + rescale/block solve”。FLA safe/block 小形状实验中 C32/C64
均 finite，说明稳定大块属于算法重设路线，而不是简单修改 `CHUNK=16` 常量。

### 2. tcgen05 最小 tile 与 CHUNK16 是否匹配，直接替换是否有收益？

SM100 BF16、CTA-group 1 的合法 tcgen05 形状允许 K16，也允许 N 从 8 开始按 8 递增；
因此 CHUNK16 在形状上并非不合法。问题是 TMEM allocation、descriptor、commit/wait、
readback、布局转换和小 grid 的协议成本。

在最自然的 Phase-6 `m128n128k16` V128 条件下，即使把固定协议成本摊销 64 次，
tcgen05 仍只有 HMMA 的 0.919742x。这只否定该 instruction-only candidate，不否定
其他 thin-N、跨阶段驻留或完整 SM100 dataflow。

### 3. chunk 间有递推依赖，并行度还能从哪里来？

可以从独立 head、value dimension slice、不同序列和 prepare/consumer 角色流水获得，
但不能破坏同一序列的 token/chunk state 顺序。已验证的 HMMA ValueSlice 把 TP8/H12
K2 grid 从 12 CTA 扩为 96 CTA，在 fixed T8192 上把 0.7807 ms 降为 0.5698 ms，
latency 降低 27.0%。这证明主要瓶颈之一是独立工作不足，而不是 HMMA 算术吞吐本身。

persistent kernel、multi-head CTA 和 2-CTA 协同可以继续研究，但必须同时核算 state
依赖、barrier、资源占用和 tail；不能只凭 CTA 数量判断。

### 4. 负载是 compute-bound 还是 memory-bound？

对代表性的 TP8/H12 official recurrence，12 CTA 面对 148 SM，历史 NCU 的 SM/DRAM
throughput 仅为 2.64%/1.24%。它既没有把计算单元打满，也没有把 DRAM 打满；更准确
的描述是 parallelism/occupancy/critical-path limited。H96 和其他 route 必须单独看，
不能把 H12 的 limiter 无条件外推。

判断使用的指标包括 SM throughput、DRAM throughput、active warps/occupancy、CTA/wave
数量、Tensor Core 指令、L2/DRAM bytes 和 stall breakdown；单看 FLOP 或带宽峰值不足。

### 5. BF16 state 精度如何验证？

验证同时覆盖 output 和 final state，并区分两种契约：

- 官方 bitwise-compatible 路径要求适用位置保持 BF16 舍入 DAG 和 bitwise identity；
- 改变舍入树的算法等价路径，必须在看结果前冻结参考、指标和阈值，比较官方与候选
  对同一 FP64/FLA-style reference 的误差。

历史 campaign 的 200 条比较全部 finite，98 条 ValueSlice 对照 bitwise equal，独立
参考关系最坏 relative RMSE 为 0.9131%。H3 又补充 80-case single-chunk、32-scenario
recurrence、18-case input oracle 和 B300 8/8 output/final-state gate；H3 明确只承诺
algorithm-equivalent，不冒充 official-bitwise。

### 6. 如果是作者，v2 是否发布 SM100a 专版？

**发布，但采用显式、可回退的 shape-guarded backend，不立即无条件替换 portable
HMMA 默认路径。**

理由是：CAKE 已证明完整 SM100 路径在已测 H12/H96 上有大幅收益；另一方面 direct
swap、V16 integration 和 H3 又证明局部迁移会失败。正确工程策略是保留官方/portable
fallback，只对通过 output、state、public-scope performance 和 confidence gate 的 shape
激活 SM100 route。

既有 Stage3 guarded evolution 已在 H96/H64 六形状中激活五个、一个回退 CAKE，
部署 geomean 相对 CAKE 再提高 1.1420x。该结果支持 guarded release policy，但不等于
已经覆盖本轮全部 H12 profiles。

## 四、多路径结果

| 路径 | 改变内容 | 证据范围 | 结果 | 结论 |
|---|---|---|---|---|
| Official HMMA | 当前 K1/K2 与 SM80 MMA | official full forward、SASS、NCU | 基线 | 可移植 fallback |
| Direct V128 tcgen05 | Phase-6 instruction-only | B300 microprobe，V128/grid12 | 0.919742x | 该路径停止 |
| V16 tcgen05 | thin-N、grid96 mechanism | B300 L0/L1 probe | 1.6150x / 0.7785x | core 有潜力，转换失败 |
| P3/P4 TMEM lifecycle | 跨阶段驻留设计 | Typed design，未完成 CUDA carrier | 未测 | 后续机制路线 |
| H3 P/W | 改变 K1/K2 分界和 workspace | 完整实现、8-case GPU correctness、9-shape CUPTI | H12 0.9055x，H96 0.7581x | 按 gate 停止 |
| CAKE SM100 | tcgen05/TMEM + 全栈重组 | 9-shape isolated public-full CUPTI | H12 2.4823x，H96 2.2532x | 完整迁移值得 |
| Guarded evolution | CAKE 上继续做 route/schedule/dispatch 优化 | H96/H64 六形状 | 对 CAKE geomean 1.1420x | shape-guarded 部署 |

H3 的一个 fixed T8192 case 为 1.0219x，但不能推翻停止结论：H12 geomean 低于 1，
最差 case 0.7587x，H96 geomean 0.7581x。只选择局部 winner 会造成 cherry-pick。

## 五、比较公平性

### 公平的工程问题

官方/CAKE 正式比较保持相同 KDA、shape、seed、初态、public output/state contract 和
每调用成本；采用 cold-L2 CUPTI、20 warmup、每 block 100 samples、4 个 balanced
blocks，并让每个 worker 只加载一个实现。因此它公平回答：

> 当前官方实现换成完整 SM100 实现，在这些 deployment profiles 上是否值得？

### 尚未闭合的因果问题

官方与 CAKE 不是单变量关系。CAKE 是面向同一 public semantics 的生成式/重组织
SM100 full-stack route，不是只在 MoonshotAI 官方 kernel 上替换若干指令；它同时改变
fusion/split、work partition、warp roles、TMEM residency、barriers、pipeline、layout 和
shape dispatch。之后的 evolution 才是在 CAKE route 上继续优化。

因此现有 2.4823x/2.2532x 不能回答：

> tcgen05 相对最优 HMMA 的纯边际价值是多少？

### 下一层精确问题：tcgen05 选择权价值

真正要分离的是：

```text
H0：官方 HMMA
H1：通用优化 + HMMA-specific 优化后的最强 HMMA
T0：与 H1 尽量结构匹配、未使用跨阶段特权的 tcgen05/TMEM
T1：T0 + tcgen-specific residency/async/warp-role/pipeline 优化
T2：CAKE/evolution 最强 guarded deployment
```

应报告：

```text
HMMA 侧可用优化收益       = latency(H0) / latency(H1)
直接 ISA/必要协议收益      = latency(H1) / latency(T0)
tcgen 特殊优化附加收益     = latency(T0) / latency(T1)
最优 tcgen 相对最优 HMMA   = latency(H1) / latency(T1)
官方到最终部署的总收益     = latency(H0) / latency(T2)
```

H1 与 T1 必须获得相同候选数、工程工作量或自动搜索预算，并允许各自使用架构优势。
HMMA 可利用 register-fragment reuse、轻量 warp-local issue 和免除 TMEM 协议；tcgen05
可利用 TMEM accumulator residency、专门 MMA warp 异步 issue、跨 phase carrier 和更深
producer/MMA/consumer pipeline。

如果无法构造完全结构匹配的 T0，应将结果称为“tcgen05 + 必要协议包联合效应”，而
不是纯 ISA 因果。完整实验定义见
[`experiments/sm100_open_round1/TCGEN_OPTION_VALUE.md`](experiments/sm100_open_round1/TCGEN_OPTION_VALUE.md)。

## 六、内部有效性修正

早期官方/H3 同进程实验存在扩展符号互插：两个 `.so` 导出同名 weak/global CUDA
symbols，反转 load order 会显著改变绝对时间。即使 Python 使用 `RTLD_LOCAL`，仍不足
以保证该组合的测量隔离。

因此：

- 旧 CUDA-event 和同进程 CUPTI artifact 仅保留为 development diagnostic；
- H3 与 CAKE 的正式 magnitude 均改为 one-implementation-per-worker-process；
- 不删除旧结果，而是在 IR、index 和报告中明确标为 superseded；
- 正式 artifacts 保存原始 samples、block medians、bootstrap intervals、实现 identity
  和哈希。

## 七、Typed IR 交付

多路径 Typed IR 不是把所有候选塞进同一僵硬 schema，而是明确区分：

- semantic contract：同一 KDA、shape、dtype、舍入和 state 语义；
- physical contract：logical/physical MNK、tile、ISA、layout、storage、roles、barriers、
  pipeline 和生命周期；
- evidence contract：源码/二进制 identity、硬件、correctness、measurement scope、结果
  和适用边界；
- failure scope：失败只能成为带 applicability key 的 prior，不能升级成 verifier ban。

当前路径索引为
[`experiments/sm100_open_round1/paths/index.json`](experiments/sm100_open_round1/paths/index.json)。
H3 有完整 prototype-scope IR；CAKE 为 source-grounded partial reconstruction，已记录真实
routes、tcgen/TMEM family、fused-m128 resources/roles/pipelines，同时明确缺少完整 per-site
MNK、所有 descriptor layout、prepare-chain barrier graph 和 per-route SASS counts。

## 八、正式证据

### 官方、CHUNK 和历史主线

- 原始任务：[`../TASK.md`](../TASK.md)
- Stage11 迁移审查：[`STAGE11_MMA_MIGRATION_REVIEW.md`](STAGE11_MMA_MIGRATION_REVIEW.md)
- 历史机器可读结论：[`evidence/stage11_mma_migration_memory.json`](evidence/stage11_mma_migration_memory.json)
- 原 C1 报告：[`../../../../assignment02-github/team-projects/kimi-kda/docs/c1-final/FINAL_REPORT.md`](../../../../assignment02-github/team-projects/kimi-kda/docs/c1-final/FINAL_REPORT.md)

### 本轮正式结果

- CAKE/official isolated CUPTI：
  [`experiments/sm100_open_round1/evidence/b300_stage12_cake_official_isolated_pair.json`](experiments/sm100_open_round1/evidence/b300_stage12_cake_official_isolated_pair.json)
- H3/official isolated CUPTI：
  [`experiments/sm100_open_round1/evidence/h3_b300_isolated_process_cupti.json`](experiments/sm100_open_round1/evidence/h3_b300_isolated_process_cupti.json)
- H3 GPU correctness：
  [`experiments/sm100_open_round1/evidence/h3_gpu_correctness.json`](experiments/sm100_open_round1/evidence/h3_gpu_correctness.json)
- H3 source/binary manifest：
  [`experiments/sm100_open_round1/evidence/h3_source_manifest.json`](experiments/sm100_open_round1/evidence/h3_source_manifest.json)
- CAKE source/binary manifest：
  [`experiments/sm100_open_round1/evidence/cake_source_binary_manifest.json`](experiments/sm100_open_round1/evidence/cake_source_binary_manifest.json)
- CAKE partial IR：
  [`experiments/sm100_open_round1/ir/cake_sm100_full_stack.json`](experiments/sm100_open_round1/ir/cake_sm100_full_stack.json)
- H3 IR：
  [`experiments/sm100_open_round1/ir/h3_precomputed_pw_c16.json`](experiments/sm100_open_round1/ir/h3_precomputed_pw_c16.json)

### 答辩材料

- 64 问自审：
  [`experiments/sm100_open_round1/DEFENSE_SELF_AUDIT.md`](experiments/sm100_open_round1/DEFENSE_SELF_AUDIT.md)
- 本轮状态：
  [`experiments/sm100_open_round1/CURRENT_MAINLINE_STATUS.md`](experiments/sm100_open_round1/CURRENT_MAINLINE_STATUS.md)
- 多路径复盘：
  [`experiments/sm100_open_round1/ROUND1_REVIEW.md`](experiments/sm100_open_round1/ROUND1_REVIEW.md)

## 九、可说与不可说

可以说：

- 官方计算路径在 B300 上确实使用 HMMA 而非 TCGEN；
- 在已测 H12/H96 profiles 上，完整 SM100 CAKE 路径值得替换当前官方实现；
- 裸指令替换、布局接入和局部代数重排可能失败；
- tcgen05 的重要价值可能是扩展 TMEM residency、异步 issue、warp-role 和跨阶段流水
  的设计空间，而不是 opcode 本身；
- 应发布 guarded SM100 backend，并保留 portable fallback。

不可说：

- `tcgen05` 单指令带来 CAKE 的 2.4823x/2.2532x；
- direct V128 失败证明所有 tcgen05 路径都失败；
- H3 algorithm-equivalent 等于 official-bitwise；
- 单张 B300 kernel 结果等于完整模型或服务吞吐收益；
- H12 结果自动适用于所有 head counts；
- CAKE 已经与同预算最优 HMMA 完成纯因果比较。

## 十、交付判定与后续边界

原始主线已经完成阶段性、可答辩的回答：官方复现、六个讨论点、多条实质不同路径、
正负 CUDA 结果、output/final-state correctness、隔离性能证据、Typed IR、失败复盘和
guarded v2 决策都已形成闭环。

仍需继续研究的最重要问题是 tcgen05 选择权价值，即 H1/T0/T1 的结构匹配 best-vs-best
对照。这会提高论文的因果解释强度，但不推翻当前工程结论。跨 B200/B300 replica、
完整模型、长期数值、多 stream/多请求并发和服务部署属于后续挑战，应与本主线证据
分开报告。

## 十一、2026-09-10 双层 IR agent 两轮补充

按 HMMA proposer、tcgen05 proposer、evidence critic、deterministic executor 和经验回写
的闭环结构，又执行了两轮等 proposal/GPU challenger 数的开放探索。完整复盘见
[`experiments/sm100_open_round1/dual_ir_agent_rounds/TWO_ROUND_REVIEW.md`](experiments/sm100_open_round1/dual_ir_agent_rounds/TWO_ROUND_REVIEW.md)。

新增结论是：当前强化 HMMA 的 H12 packed dispatch 仍有明显空洞；V64 在 mixed6 与
balanced6 上分别比 V128 快 1.2172x 和 1.1723x，且 output/state 对官方 bitwise。
用这个 stronger HMMA 作分母后，同 profile CAKE 仍快 2.0246x 和 1.8334x。因此完整
SM100 路径的工程结论保持成立，但相对官方的数字确实高估了相对强 HMMA 的差距。

tcgen05 侧把 persistent grid 从物理 148 CTA 提到 virtual 152 CTA，虽把 H64/H96 mixed
的 schedule stride 从 126/173 降至 114/166，却分别只达到 incumbent 的 0.5908x/
0.5830x。该结果形成“schedule 经验必须绑定 physical SM count 与 wave 数”的 scoped
prior；它不否定较小 stride、persistent 调度或 tcgen05。本轮还发现 H96 s173 在一个
search-only 两 block epoch 中比 Cake 快 1.0366x，应进入正式 qualification，不能直接
激活部署。

这一补充回答的是“同等小规模新增搜索预算下，现有两条载体还能否找到改进”。两条
路线历史累计工程预算仍不相等，因此 H1/T0/T1 的结构匹配 best-vs-best 对照依然是
论文下一阶段要补的因果实验。

后续 agent 的上位范围、H0/H1/T0/T1/T2 挑战阶梯、开放 proposal plane、双层 IR 和
Python/native/header 执行身份要求已经统一到
[`FRAMEWORK_SCOPE_CORRECTION_20260910.md`](FRAMEWORK_SCOPE_CORRECTION_20260910.md)。
旧 Stage9 cpc grammar 只保留为历史 case-study backend，不再作为通用探索限制。
