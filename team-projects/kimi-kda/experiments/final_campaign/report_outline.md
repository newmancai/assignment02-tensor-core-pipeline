# C1 最终报告与 10 分钟答辩论证大纲

## 建议标题

**《FlashKDA 官方 Kernel 从 SM80 MMA 迁移到 SM100 是否值得？——面向 Kimi K3 的 B300/SM103 复现、量化分析与并行度重构挑战》**

## 0. 一句话结论与论证规则

### 一句话结论

**不值得把 FlashKDA 整体机械改写为 `tcgen05`；值得为 B300/SM103 发布一条受保护的专用执行路径，但其优先级应是 K2 recurrence 并行度重构与数据复用，而不是全面替换 MMA 指令。**

这个结论同时包含两个层次：

1. **对“只换指令”的回答是否定的。** K3 典型 `V=128` 的 Phase-6 上，`tcgen05` 在最乐观的 L0 预排布口径下，将 TMEM 开销摊薄 64 次后仍只有 `0.920×`，即比 `mma.sync` 慢约 `8.0%`；纳入 state/gate 与保守的 U 重排后只有 `0.256×`。
2. **对“是否存在 SM100 专用优化价值”的回答是有条件的肯定。** TP8 形状下 K2 只有 12 CTA 对 148 SM；ValueSlice 把 grid 扩为 96 CTA，在 `H=12,T=8192` fixed prefill 上将 `0.7807 ms` 降为 `0.5697 ms`（`27.0%` 降时），且与 V128 bitwise equal。

### 全文只使用三类陈述

- **已实测：** 给出 B300 job、形状、口径和原始文件。
- **纸面模型：** 明确写“估算”或“理想上界”，不与 NCU 实测流量混用。
- **系统推断：** 显式列出转化成 TTFT/并发/SLO goodput 所需的未知参数，不把 kernel 降时直接当成端到端收益。

---

## 1. 最终报告的论证主线

报告不按“做了哪些工具”编排，而按以下因果链展开：

```text
官方 FlashKDA 在 B300 上已经很快
        ↓ 但 SASS 证明仍用 SM80 mma.sync
新一代 Tensor Core 指令是否真的能解决当前瓶颈？
        ↓ NCU/Nsys：12 CTA 对 148 SM，SM/HBM 均远未饱和
首要问题不是峰值算力，而是 recurrence 串行依赖与整卡并行度
        ↓ 同时检验两条 SM100 路线
只换指令：Phase-6 tcgen05 microbench 在 K3 V128 上为负
并行度重构：ValueSlice 在低并发长 prefill 上为正，高并发短序列上可为负
        ↓
发布受保护的 sm100a/sm103a 路径，保留 V128 mma.sync fallback；
未来优先 cluster/TMA multicast，而非全量 tcgen05 重写
```

报告中始终区分：

- **SM80 MMA atom** 不等于“整个 kernel 是 SM80 kernel”。当前 FlashKDA 已使用 TMA/新架构能力，只是 Tensor Core 计算 atom 仍为 `mma.sync.m16n8k16`。
- **SM100 家族**与 **B300 的 SM103**。题目说“迁移到 SM100”，实验对象应准确写为 B300 / compute capability 10.3 / `sm_103a`。
- **KDA 算子性能**与 **Kimi serving 性能**。前者已实测，后者只能做有边界的推演。

---

## 2. 最终报告章节结构

### 摘要（0.5 页）

只写四件事：问题、方法、两个实验结果、发布决策。推荐摘要中出现的四个数字是：

- 官方 B300 benchmark 对 FLA chunk KDA 为 `1.79–3.42×`；
- 官方 K2 SASS 为 `3,640` 条静态 `HMMA.16816.F32.BF16`，`TCGEN/UTCMMA=0`；
- K3 TP8 形状的 ValueSlice 在 fixed `T=8192,H=12,D=128` 上降时 `27.0%`；
- K3 `V=128` Phase-6 的 `tcgen05` 乐观摊销口径仍慢约 `8.0%`。

### 1. 题目、对象与决策标准（1 页）

1. 写明 C1 的唯一中心问题：官方 FlashKDA 当前使用 SM80 MMA，迁移 SM100 是否值得。
2. 给出可操作的“值得”定义：
   - 正确性不退化；
   - 在 K3 代表形状上超过 run-to-run noise/guard band；
   - 收益可在实际 dispatcher 中被安全捕获；
   - 收益能覆盖架构专用代码、编译链和维护成本。
3. 形状分两层：官方对照用 `H=96,D=128,T=8192`；实际 TP8 每卡用 `H=12,D=128`。
4. 题面给出 K3 93 层中 69 层 KDA、24 层非 KDA/全注意力。若正文要把后者称为“Gated MLA”，必须另补官方架构来源；否则使用题面术语，避免无来源的架构命名。

### 2. 阶段一：复现与测量（2 页）

#### 2.1 环境与可复现性

- B300 SXM6，CC 10.3，148 SM，实测 shared memory/SM `233,472 B`，L2 `132,644,864 B`。
- FlashKDA commit `1ce47ea`，CUTLASS pin `5c149f5`，PyTorch `2.10.0+cu130`，FLA `0.5.2`。
- benchmark 口径：`warmup=30, iters=200, repeats=5`。
- 官方干净 worktree 和补丁 V128 分开加载；10 组 output/final-state 全部 bitwise equal，表明后续 V128 可作为官方兼容基线。

#### 2.2 官方 benchmark 复现

只在正文放 6 个主数据，完整 mean/min/max 放附录：

| 形状 | case | FlashKDA BF16 state | FLA chunk KDA | FlashKDA speedup |
|---|---|---:|---:|---:|
| H96 | fixed | 1.0304 ms | 2.4155 ms | 2.34× |
| H96 | ragged6 | 0.8612 ms | 2.4255 ms | 2.82× |
| H96 | 8×1024 | 0.6963 ms | 2.3814 ms | 3.42× |
| H64 | fixed | 0.9410 ms | 1.6856 ms | 1.79× |
| H64 | ragged6 | 0.6532 ms | 1.6990 ms | 2.60× |
| H64 | 8×1024 | 0.4740 ms | 1.5964 ms | 3.37× |

这一节的结论不是“官方 kernel 慢”，而是：**它已是强基线，因此新架构重写必须用真实净收益证明自己。**

#### 2.3 SASS 与 profile 确认

- 源码/SASS：K2 使用 `SM80_16x8x16_F32BF16BF16F32_TN`；静态 SASS 统计为 3,640 条 `HMMA.16816.F32.BF16`，没有 `TCGEN/UTCMMA`。
- TP8 `H=12` 官方 K2：12 CTA 对 148 SM，NCU duration `632.32 µs`，SM throughput `2.66%`，DRAM throughput `0.36%`，achieved occupancy `9.38%`，No Eligible `66.63%`。
- V16：96 CTA，duration `454.69 µs`，虽然加速，SM/DRAM throughput 仍只有 `7.18%/1.58%`。
- Nsys 同进程五轮 A/B：GPU projected span `3.416 → 2.505 ms`（`-26.7%`），时间变化集中在 K2 recurrence，prepare 几乎不变。

这一节回答讨论点 4：**它不是传统 Tensor Core compute-bound，也不是 HBM bandwidth-bound，而是 grid underfill + recurrence/TMA/issue-latency-bound。**

建议列出实际 NCU metric 类别：

- kernel duration / grid size；
- SM throughput 和 tensor pipe active；
- DRAM throughput / bytes；
- L2 hit rate；
- achieved occupancy / active warps；
- scheduler `No Eligible`。

不用单个 occupancy 或单个 No Eligible 得出因果；结论来自上述证据的交叉。

### 3. 阶段二：六个讨论点的“结论 + 证据”（4–5 页）

#### 3.1 讨论点 1：为什么 CHUNK=16，32/64 谁先破？

**结论：** 机械放大 CHUNK 时，最先破的是当前无 rescale 的指数数值路径，其次是 Neumann 密集扩展的算力代价；workspace 只温和上升，不是第一个破点。大 CHUNK 只有在加入 block/rescale/分解后才是新算法，不能当作常量修改。

| CHUNK | 最坏 `lower_bound=-5` 指数结果 | 首次 FTZ/overflow | 朴素 Neumann 每序列代价 | workspace/head |
|---:|---|---:|---:|---:|
| 16 | 可表示 | 无 | 1.00× | 6.750 MiB |
| 32 | 15 zero + 15 inf /通道 | token 18 | 5.33× | 7.125 MiB |
| 64 | 47 zero + 47 inf /通道 | token 18 | 26.67× | 8.063 MiB |

补充证据：FLA 的安全/block 实现在小形状 `T=128,H=1,D=128` 上 C32/C64 均 finite，中位延迟分别 `0.2500/0.2521 ms`；这只证明“通过改算法可以做对”，不证明完整 FlashKDA 改大 CHUNK 会变快。

#### 3.2 讨论点 2：`tcgen05` tile 与 CHUNK=16 匹配吗？

**结论：部分匹配，不支持整体机械替换。**

- SM100 BF16 1-CTA `tcgen05` 的 `M∈{64,128}`，`N=8..256` 且以 8 递增，`K=16`。不能误说“最小 N=64”。
- K2 Phase-6 是 `[128,16]@[16,V]`，自然对应 `m128nVk16`；对 `V=16/32/64/128` 全部合法，CHUNK=16 在这里正好是 K=16。
- K2 其他大量以 CHUNK 为 M 的 `[16,V]` phase 则不能保持当前朝向直接映射，需要转置/重排/重设数据流。

Phase-6 microbench 必须用两层口径表达：

- **L0（乐观下界）：** 操作数已在各自理想片上布局，收取 TMEM alloc/commit/wait/load/dealloc。
- **L1（保守上界）：** 再纳入 BF16 state 加载/门控/回写和每轮 U 重排；其 scalar copy 比精心集成更悲观。

关键数据（Job 17937，与 Job 17936 重复结论一致）：

| V / grid / inner | L0 `tcgen05` speedup | L1 `tcgen05` speedup | 解读 |
|---|---:|---:|---|
| 128 / 12 / 1 | 0.376× | 0.533× | 启动/TMEM 固定成本很大 |
| 128 / 12 / 64 | 0.920× | 0.256× | 即使摊薄，乐观口径仍负 |
| 16 / 12 / 64 | 1.501× | 0.778× | core 有潜力，但当前表示转换吃掉收益 |

因此不进入全 K2 `tcgen05` 集成是一个数据驱动的 stop decision，不是实现失败。此 probe 不是完整 K2，不能说它证明任何 `tcgen05` 重设都永远不可能。

#### 3.3 讨论点 3：recurrence 之外的并行度从哪来？

| 候选 | 可能收益 | 反例/代价 | 决策 |
|---|---|---|---|
| 更多 sequence/batch/head | 自然增加 CTA | 低并发、TP8 每卡 H=12 时不存在；并发高时反而不需额外切分 | 交给 serving 负载，不是 kernel 通用解 |
| 多 head 合并到一 CTA | 共享控制/数据 | CTA 数更少，state/smem 压力更大 | 不适合当前 underfill |
| persistent kernel | 摊薄 launch/设置，可重用片上数据 | 不消除单个 head 的 chunk 依赖，12 个 persistent CTA 仍然喂不满 148 SM | 只作第二阶段辅助 |
| 2-CTA/CTA Cluster | 允许 DSM 和 TMA multicast | 需要真正可分的维度；cluster residency/同步可降低 occupancy | 和 ValueSlice 结合最有价值 |
| ValueSlice | Value 行独立，无 reduction/atomic；CTA `12→96` | slice-independent input 被重复读取，高并发短序列可变慢 | **已实现的挑战路线** |

在这里把 SM103 的通信能力纳入主线：ValueSlice 会重复搬运 common inputs，未来可用 **CTA Cluster + TMA multicast** 让多个 slice CTA 共享一次输入搬运。对 `T=4096`、单 sequence-head 的纸面 source-request 模型，V16 为 `29.188 MiB`，理想 multicast 可节省 `23.734 MiB`（`81.3%`），回到接近 V128 的 `5.453 MiB`。必须标注这不是实测 HBM bytes。

#### 3.4 讨论点 4：compute-bound 还是 memory-bound？

本点已在 2.3 用 profile 回答，此处只归纳：

- 既非峰值 compute-bound，也非 HBM bandwidth-bound；
- 当前是小 grid 覆盖不足 + recurrence 依赖 + TMA/issue latency 的混合边界；
- ValueSlice 后仍有高 No Eligible，说明它解决了整卡并行度，没有消除 CTA 内依赖。

#### 3.5 讨论点 5：BF16 state 精度是否足够？

**结论：** 在本次长序列、ragged、state carry 和 long-memory gate 测试上，FlashKDA 相对 FP32 naive/FLA 参考的 output/state relative RMSE 均低于 1%；ValueSlice 不增加任何误差。但这是 kernel 数值对拍，不是模型级 perplexity/任务精度证明。

数据摘要：

- 200 条比较全部 finite 且通过阈值；
- 98 条 ValueSlice vs V128 比较全部 bitwise equal，最坏 relative RMSE `0`；
- `T=8192` random，Flash vs naive：output `0.5698%`，final state `0.4728%`；
- `T=8192` long-memory，Flash vs naive：output `0.8240%`，final state `0.7405%`；
- 最坏 Flash vs FLA chunk：output `0.9131%`，state `0.8151%`；
- 公开 BF16/FP32 state buffer 路径的本次误差数据相同，说明只把公开 state buffer 改为 FP32 并没有改变 kernel 内部 BF16 舍入点。不应写成“FP32 state 没用”，而应写成“现有 API 路径未形成全 FP32 recurrence 对照”。

#### 3.6 讨论点 6：假如我们是作者，v2 出不出 sm100a 专版？

**建议：出，但是受保护的 hybrid 专用路径，不出全量 `tcgen05` 分叉。**

发布内容：

1. 保留官方 V128 `mma.sync` 作 bitwise-compatible fallback；
2. 对 B300/SM103、低并发长 prefill 启用 ValueSlice；
3. varlen 必须使用 sequence-count/distribution-aware guard，不能看 total tokens 就选 V16；
4. 下一个原型是 cluster/TMA multicast 减少 common-input 重复读；
5. `tcgen05` 只保留为未来数据流重设的选择性实验，当前 Phase-6 数据不支持集成。

不利因素：架构专用二进制、CUDA 13/编译链、TMEM/cluster 调试、CI 覆盖、更大二进制、维护多个 layout。这些成本要求专版必须是 opt-in/guarded，而不能替换可移植基线。

### 4. 阶段三：挑战——ValueSlice 并行度重构（3 页）

#### 4.1 可分性和代码改动

- K2 state 的 Value 行彼此独立，沿 `V∈{16,32,64,128}` 切分不需要 reduction、atomic 或跨 CTA 通信。
- grid 由 `(N,H)` 变为 `(N,H,D/V)`；H12 下 V128/V64/V32/V16 对应 12/24/48/96 CTA。
- 总 Tensor FLOP 不变；收益来自更多独立 CTA，代价是 slice-independent input 重复搬运。
- 四个变体编译进同一扩展，不确定时回退 V128。

#### 4.2 正确性

建议按三层呈现：

1. 官方干净扩展 vs patched V128：10/10 tensor bitwise equal；
2. V16/V32/V64 vs V128：98/98 比较 bitwise equal；
3. FlashKDA vs 题目指定 `naive.py`/`chunk.py`：200/200 finite 且通过，最坏 relative RMSE 小于 1%。

必须附参考版本/hash：FLA 0.5.2，`naive.py` SHA-256 前缀 `60a32285`，`chunk.py` 前缀 `a15aa6ac`，`FLA_FLASH_KDA=0`，防止“参考意外又 dispatch 回 FlashKDA”。

#### 4.3 性能与反例

| packed/fixed 分布，均为 total T=8192,H=12 | V128 | 最佳强制切分 | 降时 | 对 dispatcher 的含义 |
|---|---:|---:|---:|---|
| fixed 1×8192 | 0.7807 ms | V16 0.5697 ms | 27.0% | 低并发长 prefill 应切细 |
| packed 1×8192 | 0.7850 ms | V16 0.5740 ms | 26.9% | 现 dispatcher 因 varlen 保守回退，是可修复的 missed opportunity |
| ragged6 | 0.3403 ms | V64 0.2810 ms | 17.4% | 不能对所有 varlen 固定一个 V |
| 8×1024 | 0.1561 ms | V64 0.1540 ms | 1.3% | 收益低于 3% guard，回退 V128 合理 |
| 32×256 | 0.1254 ms | V128 0.1254 ms | 0% | V16 反而慢 106%，高自然并发不应切分 |

这张表是挑战阶段的中心结果：**优化是 workload-dependent 的，负反例本身就是 dispatcher 设计的证据。**

### 5. 从 KDA kernel 到 Kimi K3 prefill、并发和 SLO goodput（2 页）

#### 5.1 能严格声称的影响

- FlashKDA 是 KDA **forward/prefill** 路径的直接优化对象。
- TP8 单请求每卡 H=12 时，ValueSlice 缓解整卡 underfill，因此最有可能改善低并发长 prompt 的 KDA prefill latency/TTFT 组成部分。
- 同一 total tokens 下，sequence 数从 1 增到 8/32 时，负载自身已提供并行度，ValueSlice 收益消失并可能伤害吞吐。

#### 5.2 不能直接声称的影响

- 不能把 `27.0%` KDA operator 降时写成 `27.0%` TTFT 降低。
- 不能用 69/93 层比例代替时间占比；KDA 层还包含投影、norm、gate 等工作，24 层非 KDA 的代价也不同。
- 不能声称 decode/TPOT 已加速。现有 synthetic state-carrying trace 中 `T=4096` prefill 启用 ValueSlice，后续 `T=1` 全部回退 V128；实际 Kimi/vLLM 纯 decode 还有独立 fused KDA decode 路径。
- 不能声称 SLO goodput 已提升；没有完整 serving scheduler、排队、batching、KV/state 管理与实际 SLO sweep。

#### 5.3 可以做的定量外推

设 `p` 为 FlashKDA forward 在整个 prefill 中的时间占比，实测算子降时 `r=0.27`，则：

```text
预计 prefill 降时 = p × r
理想容量加速 = 1 / (1 - p × r)
```

例如 `p=20%/40%/60%` 时，预计 prefill 降时仅为 `5.4%/10.8%/16.2%`。这是 Amdahl 敏感性分析，不是 Kimi 实测。SLO goodput 是否改善还取决于 TTFT 是否为绑定约束、排队非线性和高并发下 dispatcher 的选择。

#### 5.4 通信、环境和运营视角

- **跨 GPU 通信：** 单 B300 实验不测 NCCL；H=12 只是 TP8 的 per-GPU 计算形状。FlashKDA 局部降时不会自动减少 all-reduce/all-to-all 时间。
- **卡内通信：** Cluster/DSM/TMA multicast 恰好对应 ValueSlice 引入的跨 CTA common-input 重复读，比“通用地谈新通信特性”更贴题。
- **资源环境：** B300 更大显存/带宽/shared memory 可支持长上下文、更大 batch 和更深 pipeline，但在 12 CTA underfill 下这些峰值不会自动变成 latency 收益。
- **运营报告：** 市场/运营报告可帮助选 prompt length、并发度和 SLO 权重，不能代替 SASS/NCU/microbench 来回答 MMA 迁移是否值得。没有 Kimi K3+B300 官方线上 trace 时，不把第三方运营数据当主证据。

### 6. 威胁有效性、限制和最终结论（1 页）

限制必须主动说：

1. 只有一张 B300，不能实测 TP8/NCCL 或端到端 Kimi K3；
2. `tcgen05` 只做了真实 Phase-6 隔离 probe，不是完整 K2；
3. CHUNK32/64 的 FLA 只有小形状 safe-block 对照，没有重写整个 FlashKDA；
4. ValueSlice 的最优 V 依赖 sequence distribution，当前 varlen dispatcher 保守回退；
5. 没有模型级 accuracy、TTFT/TPOT 和 SLO goodput 结果；
6. NCU/Nsys/CUDA Event 的绝对时间不可混用，只在各自一致口径内 A/B。

最后用一张决策表收束：

| 路线 | 当前证据 | 决策 |
|---|---|---|
| 全面 `mma.sync → tcgen05` | K3 V128 Phase-6 L0 摊销后仍 0.920× | **停止，不集成** |
| CHUNK 32/64 机械放大 | 当前指数路径 token 18 FTZ/overflow，Neumann 5.33×/26.67× | **停止，除非改算法** |
| ValueSlice 并行度重构 | 低并发长 prefill 最高降时 27.0%，bitwise equal；短序列有反例 | **受保护发布** |
| Cluster + multicast | 纸面可消除大部分 common-input 重复请求 | **下一个 B300 挑战** |

---

## 3. 10 分钟答辩逐页设计（严格 600 秒）

### Slide 1　题目与结论先行（0:00–0:35，35 秒）

**标题：** 新 Tensor Core 更强，FlashKDA 就应重写吗？

- 题目原句 + 官方还在 B300 上使用 SM80 MMA 的悬念。
- 结论一行：不做全面 `tcgen05`；做受保护的 recurrence 并行度专版。
- 口头预告两个核心数据：`tcgen05 0.920×`，ValueSlice `-27.0%`。

**视觉：** 一张左右决策卡，不放背景大表。

### Slide 2　阶段一：复现可信基线（0:35–1:25，50 秒）

- B300/SM103、148 SM；commit/runtime/benchmark 口径。
- H96/H64 的 6 个 FlashKDA vs FLA 结果，突出 `1.79–3.42×`。
- 干净官方扩展 vs patched V128 正确性与性能同口径对齐。

**视觉：** 官方 benchmark 复现分组条形图，页脚写 Job 17926/17929。

### Slide 3　真正的计算路径与瓶颈（1:25–2:25，60 秒）

- K1 token-parallel，K2 head-parallel recurrence；TP8 下 K2 仅 12 CTA。
- SASS：3,640 HMMA，TCGEN=0。
- NCU：12 CTA/148 SM，SM 2.66%，DRAM 0.36%，occupancy 9.38%。
- 结论：先缺可并行工作，不是缺更高的理论峰值。

**视觉：** 现有 `kimi_kda_b300_bottleneck` 图的精简版：只留 CTA 覆盖、NCU throughput 和 K2 duration 三个面板。

### Slide 4　讨论点 1：CHUNK=16 不是偶然（2:25–3:20，55 秒）

- 一张表同时对照数值、Neumann、workspace/MMA 形状。
- C32/C64 首先在 token 18 发生 FTZ/overflow；朴素每序列 Neumann 代价为 5.33×/26.67×。
- FLA safe-block 能保证 finite，证明大 CHUNK 需要算法重设，不是改常量。

**视觉：** CHUNK 16/32/64 的三行红黄绿约束表。

### Slide 5　讨论点 2：`tcgen05` 纸面可行，实测不值（3:20–4:30，70 秒）

- 先纠正 tile 认识：Phase-6 `[128,16]@[16,V]` 可合法映射 `m128nVk16`；问题不是“指令发不出”。
- 再放 L0/L1、inner=1/64 的口径。
- 最重要的 V128 两个数：L0 摊销后 `0.920×`，L1 `0.256×`。
- 说出 stop decision：不花更多 B300 时间做全 K2 集成。

**视觉：** `tcgen05/mma.sync` speedup 横条图，`1.0×` 红色基准线；主图只放 grid=12, inner=64 的 V16/32/64/128，其他放附录。

### Slide 6　讨论点 3/4：反过来解决并行度（4:30–5:35，65 秒）

- Value 行独立，grid `12→24/48/96 CTA`，总 MMA FLOP 不变。
- Nsys/NCU 证明减时集中在 K2，但 V16 仍不是 compute/HBM 饱和。
- 一句交代其他候选的反例：多 head/CTA 会进一步减少 CTA，persistent 不消除 chunk 依赖，cluster 需先找到独立切分维度。

**视觉：** ValueSlice 前后 state 矩阵和 CTA-to-SM 映射示意图。

### Slide 7　挑战结果：正确，但不是所有负载都快（5:35–6:55，80 秒）

- 正确性：98/98 ValueSlice 比较 bitwise equal；200/200 参考对拍通过，最坏 relative RMSE <1%。
- 主性能图：fixed 1×8192 `-27.0%`，packed 1×8192 `-26.9%`，ragged6 最佳 V64 `-17.4%`。
- 反例：8×1024 最佳仅 `-1.3%`；32×256 选 V16 慢 `106%`。
- 结论：必须 distribution-aware dispatch，低置信度回退 V128。

**视觉：** 热力图（行=sequence distribution，列=V16/32/64/128，颜色=相对 V128 降时），旁边一个紧凑正确性卡。

### Slide 8　讨论点 5：BF16 state 边界（6:55–7:35，40 秒）

- 长序列 random/long-memory 与 naive/FLA 的 output/state relative RMSE 均 <1%。
- ValueSlice 不改变归约顺序，因此 bitwise equal。
- 当前 BF16/FP32 public state 路径精度相同，并未建立“全 FP32 内部 recurrence”对照；不做过度结论。

**视觉：** 4 根 relative-RMSE 柱 + 1% 阈值线，不放 200 行表。

### Slide 9　这对 Kimi K3 意味着什么（7:35–9:00，85 秒）

- 主箭头：KDA forward kernel → KDA prefill 组成部分 → TTFT 可能改善。
- 断开的箭头：decode/TPOT（独立路径）、NCCL（单卡未测）、SLO goodput（需完整 serving sweep）。
- 并发反转：低并发长 prompt 受益最大；高并发短 sequence 不应强制 ValueSlice。
- 用 `p×0.27` 给出 Amdahl 敏感性，而不声称端到端数字。
- 卡内通信下一步：Cluster + TMA multicast 共享 common inputs。

**视觉：** 系统边界流程图 + `p=20/40/60%` 的三点 Amdahl 小图。

### Slide 10　讨论点 6：发布决策（9:00–10:00，60 秒）

- 对三条路线给出 stop/go：
  - 全量 `tcgen05`：Stop；
  - 机械大 CHUNK：Stop；
  - guarded ValueSlice：Go。
- 下一步：cluster/multicast；只在新数据流能跨 phase 保留 TMEM 表示时重新打开 `tcgen05` gate。
- 最后一句回到题目：**SM100 值得利用，但“利用 SM100”不等于“把 SM80 MMA 全换掉”。**

**视觉：** 3 行决策矩阵（正确性/性能/移植性/决策）。

---

## 4. 需要制作的图表清单

### 答辩必需（优先级 P0）

1. **F1 官方 benchmark 复现条形图**
   - 数据：`data/raw/01_official_benchmark_17926.log`
   - 表现：H96/H64 下 fixed/ragged6/8×1024，FlashKDA vs FLA chunk；直接标注 1.79–3.42×。
2. **F2 K1/K2 + TP8 CTA underfill 结构图**
   - 数据/源码：官方 deep-dive、K2 launcher。
   - 表现：K1 `N×H×chunks`，K2 `N×H`，12 CTA 落到 148 SM。
3. **F3 SASS + NCU/Nsys 瓶颈组合图**
   - 已有：`experiments/figures/kimi_kda_b300_bottleneck.png/.svg`
   - 答辩版只保留 HMMA/TCGEN 计数、CTA 覆盖、SM/DRAM throughput、K2 duration。
4. **F4 CHUNK 16/32/64 三约束表**
   - 数据：`data/raw/04_chunk_analysis_17935.csv`
   - 表现：数值 FTZ/overflow、Neumann 每序列相对代价、workspace。
5. **F5 `tcgen05` 形状 + microbench speedup 图**
   - 数据：`data/raw/03_tcgen05_probe_17937.csv`
   - 表现：上半部 `m128nVk16` 映射，下半部 grid=12/inner=64 的 L0/L1 speedup，加 1.0× 线。
6. **F6 ValueSlice 数据流/网格图**
   - 数据：补丁与纸面 FLOP/source-request 模型。
   - 表现：V128 一个 CTA 与 V16 八个 CTA；common input 重复读用另一颜色。
7. **F7 sequence distribution × ValueSlice 性能热力图**
   - 数据：`data/raw/02_k3_shapes_17928.csv`
   - 表现：相对 V128 降时；同时显示正收益和负收益。
8. **F8 正确性证据卡/RMSE 图**
   - 数据：`data/raw/03_reference_correctness_17934.csv`、`baseline_parity.json`
   - 表现：10/10 官方兼容、98/98 bitwise、200/200 pass；长序列 RMSE <1%。
9. **F9 Kimi 系统边界 + Amdahl 图**
   - 数据：实测 `r=27%`，`p` 为敏感性参数。
   - 表现：实线箭头到 prefill component，虚线箭头到 TTFT/goodput，红色断开到 decode/NCCL。
10. **F10 最终 stop/go 决策矩阵**
    - 数据：上述所有证据的结论汇总。

### 报告附录需要（优先级 P1）

11. 完整官方 benchmark mean/min/max 表。
12. `tcgen05` 全部 V/grid/inner/L0/L1 实验表，附正确性 guardrail 和 SASS `UTCHMMA/HMMA` 样例。
13. NCU metric 原名、数值与解释表。
14. 正确性 case matrix（fixed/ragged、state mode/dtype、gate regime、seed）。
15. 环境、commit/hash、job 号、耗时与产物路径表。

---

## 5. 15 分钟 B300 限制如何写进方法学

这不是要道歉的“资源不足”，而是实验设计约束。报告建议单独放一个小节：

### 槽位策略

- 每个 sbatch 自包含：记录 GPU/driver/runtime/commit/hash，再运行有界 benchmark，最后写 CSV/log。
- 使用 `warmup=30, iters=200, repeats=5`；path 次序轮换，同进程 CUDA Event A/B，降低时钟/温度/动态状态偏差。
- 编译、正确性和性能分层；`tcgen05` 使用 guardrail build 先验证，只有 PASS 才进入 release timing。
- profiler 按单个关键 kernel/少量 metric set 分批，避免 NCU replay 在 15 分钟内失控。
- 每个挑战在运行前写 stop/go 规则；负结果立即停止扩大实现。

### 已完成 job 的实际槽位时间

| Job | 内容 | 日志墙钟时间 |
|---:|---|---:|
| 17926 | 官方 benchmark | 93 s |
| 17928 | K3 fixed/packed ValueSlice sweep | 12 s |
| 17929 | 官方 vs patched V128 parity | 6 s |
| 17934 | naive/chunk 参考正确性 | 33 s |
| 17935 | CHUNK 16/32/64 分析 + FLA microbench | 41 s |
| 17936/17937 | `tcgen05` 独立重复 probe | 各 16 s |

这张表证明实验流程确实适配 15 分钟槽位；它不能证明 profiler 的所有长任务都可在一个槽位完成。

---

## 6. 5 分钟提问的高频问题

### Q1：你们题目是 SM80 到 SM100，为什么挑战做 ValueSlice？

题目要求的是判断迁移是否值得，而不是预设新指令必然快。SASS/NCU 证明首要问题是 12 CTA 对 148 SM；Phase-6 microbench 进一步证明 K3 V128 的 `tcgen05` 乐观口径也未赢。ValueSlice 是题目允许的“并行度重构”挑战，直接针对实测瓶颈。

### Q2：`tcgen05` 的最小 tile 到底匹不匹配 CHUNK=16？

不能一句话答“匹配/不匹配”。Phase-6 的 K 正好是 16，M=128，因此 `m128nVk16` 完全合法；但其他大量 M=16 phase 不能保持当前朝向直接映射。我们选了最匹配的 Phase-6 先测，结果在 K3 V128 上仍为负。

### Q3：ValueSlice 为什么 bitwise equal？

切分的是互相独立的 Value 行，没有改变单个输出元素的运算和归约顺序，也没有 reduction/atomic。实测 98/98 对比 output/final state 全部 bitwise equal。

### Q4：为什么 V16 不是永远最快？

V16 把 CTA 数放大 8 倍，也会重复读取 slice-independent inputs。单个长 sequence 需要更多 CTA，因此受益；32 个短 sequence 已经有足够 CTA，额外切分只剩下重复流量和调度成本，V16 实测慢 106%。

### Q5：这能让 Kimi K3 的 TTFT 快多少？

目前不能给实测端到端数字。只能说低并发长 prefill 的 FlashKDA operator 降时 27%。若该 operator 占 prefill 时间比例为 `p`，则总 prefill 降时约为 `0.27p`。必须有完整 K3 serving trace 才能把 `p`、排队和 SLO goodput 测出来。

### Q6：为什么不能说 decode 也加速了？

现有 dispatcher 对 `T=1` 回退 V128，synthetic trace 中真正变快的是前面 `T=4096` prefill。此外，实际 serving 的纯 decode 走独立 fused KDA decode 路径。因此不能由 FlashKDA forward 数据推出 TPOT 改善。

### Q7：只有一张 B300、每次 15 分钟，数据可信吗？

限制了多 GPU/端到端结论，但不妨碍单 kernel 因果验证。所有关键对比使用同进程 CUDA Event A/B、固定 warmup/iters/repeats，记录环境和 commit；`tcgen05` 还在两个独立 job 中复现。我们把结论严格限定在 B300 单卡 operator/microbench，没有越界声称 TP8/NCCL/SLO。

---

## 7. 原始证据索引

- 题目原文：`assignment02-work/team/c1_flashkda/TASK.md`
- 官方 benchmark：`data/raw/01_official_benchmark_17926.log`
- K3 fixed/packed ValueSlice sweep：`data/raw/02_k3_shapes_17928.csv`
- 官方与 patched V128 parity：`data/raw/baseline_parity.json`
- naive/chunk 参考正确性：`data/raw/03_reference_correctness_17934.csv`
- CHUNK 数值/算力/workspace：`data/raw/04_chunk_analysis_17935.csv`
- `tcgen05` 重复 probe：`data/raw/03_tcgen05_probe_17937.csv`（独立复跑见 `tcgen05_probe/results/03_tcgen05_probe_17936.csv`）
- 旧批次 Nsys/NCU/SASS 瓶颈证据：`../BOTTLENECK_ANALYSIS.md`、`../data/`、`../artifacts/sass/`
- 核心补丁：`../../patches/0001-k2-value-slice-and-dispatch.patch`
