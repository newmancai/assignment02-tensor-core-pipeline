# FlashKDA 官方 Kernel 从 SM80 MMA 迁移到 SM100 是否值得？

## ——面向 Kimi K3 的 B300/SM103 复现、量化分析与并行度重构挑战

> C1 最终报告
> 实验日期：2026-09-03；主线增量复核至 2026-09-05；profile 补充验证至 2026-09-09
> 实验对象：NVIDIA B300 SXM6 AC（compute capability 10.3）

## 摘要

本报告回答一个单一问题：**MoonshotAI FlashKDA 的 Tensor Core 计算仍使用 SM80 世代的 `mma.sync`，是否值得为 B300/SM100 家族重写为 SM100 执行路径？**

我们的结论是：**不值得把 FlashKDA 整体机械改写为 `tcgen05`；值得交付一条受保护的 B300/SM103 专用路径，但当前最有价值的专用化是 K2 recurrence 的并行度与流水调度重构，而不是全面替换 MMA 指令。**

证据分三阶段建立。第一，在 B300 上复现官方 benchmark，FlashKDA 相对 FLA `chunk_kda` 为 **1.79–3.42×**；SASS 中有 **3,640 条静态 `HMMA.16816.F32.BF16`**，而 `TCGEN/UTCMMA=0`，确认题目所述计算路径。第二，针对六个讨论点完成纸面量化和实验：CHUNK 从 16 机械放大到 32/64 时，当前指数恢复路径都在第 18 个 token 首次出现 FTZ/overflow，朴素 Neumann 每序列代价分别增至 **5.33×/26.67×**；在最适合新指令的 Phase-6 `[128,16]@[16,128]` 上，`tcgen05+TMEM` 即使将固定成本摊薄 64 次，仍只有 `mma.sync` 的 **0.920×**。第三，我们挑战了真正的实测瓶颈：TP8 代表形状每卡只有 12 个 head，官方 K2 只有 **12 CTA 对 148 SM**。ValueSlice 将 grid 扩成 96 CTA，在 `T=8192,H=12,D=128` 上把 fixed prefill forward 从 **0.7807 ms 降至 0.5698 ms（−27.0%）**；dispatcher 也能在不读取 `cu_seqlens` 数值、不引入 GPU 到 CPU 同步的情况下，为 packed 单序列选择 V16，将 **0.7850 ms 降至 0.5740 ms（−26.9%）**。正确性 CSV 的 200 条比较全部 finite；其中 98 条 ValueSlice 对照全部 bitwise equal，所有独立参考关系的观测最坏 relative RMSE 为 0.9131%。

9 月 5 日的主线增量继续保留 `mma.sync`、D128/C16 和原状态数值契约，在 V16 CTA 内优化单 warp 流水。第一步将 Phase 6 的 `StatePrefetch` 从 1 增至 4：在独立正式 wrapper 中，相对旧 V16 的已测 eager 增量为**无初态约 7.4%–9.1%、有初态约 16.8%–19.6%**。第二步按初态合约选择 Phase 1 lookahead：无初态用 4、有初态用 2。Job 19934 的 34 个准入域性能形状在四种计时口径的中位数上全部获益，eager 相对 Phase-6-only P4 再降低 **5.15%–12.40%**；`T8192,H12` 同作业、同输入下相对强制 V128 的累计降时为**首段无初态 37.23%、有初态续段 45.52%**。这些百分比属于不同基线，不能相加，也不能与不同作业的绝对毫秒数拼接。

发布判断因此更精确：保留 V128 `mma.sync` fallback 和既有 guarded ValueSlice；Phase-6 P4 与 Phase-1 lookahead 作为**编译期默认关闭、单请求延迟导向**的 B300 候选。无初态双 stream 的两个请求 joined-pair 时间从 **1.147440 ms 增至 1.164832 ms（回归 1.52%）**，证明当前 shape guard 不等于运行时并发感知。该结论只覆盖单张 B300 上的 FlashKDA forward；没有把 27%、37% 或 46% 算子结果写成 Kimi K3 的 TTFT、TPOT、SLO goodput 或多卡收益。

贯穿这些结果的工程观点是：**Kernel is cheap，可信的 profile-to-policy 决策才昂贵。** 这里的 “cheap” 指候选 kernel 已容易生成，而不是 GPU 时间不重要。我们因此把最终贡献从若干孤立 patch 提炼为可自优化的 Runtime Profile Agent：上一轮的 verifier、screen 和 qualification 结论会进入 conclusions-only memory，驱动下一轮 proposal；工具完成 typed 去重、证据门控，并在预算耗尽或证据 plateau 时停止。论文阶段又在独立 H12 BT16 route 上，用禁止候选筛选的 W384/W768 前瞻实验验证 resident-grid-capacity 规则；四个 profile 为 **1.0133×–1.1354×**，最弱 Bonferroni 单侧 98.75% 下界为 **1.0129×**。该证据只支持 shadow recommendation，且不能与 ValueSlice 主线收益合并。

**关键词：** FlashKDA；Kimi Delta Attention；B300；SM103；Runtime Profile Agent；Agent 自优化；`mma.sync`；`tcgen05`；TMEM；ValueSlice；软件预取；prefill

---

## 1. 题目、研究对象与决策标准

### 1.1 唯一中心问题

题面 C1 要求：**“FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得。”** 本报告不把“SM103 有哪些新功能”当作答案，也不预设新指令一定更快，而是判断：在 Kimi K3 的真实矩阵规模、依赖结构和部署形状下，哪一种 SM100 专用化能产生可交付的净收益。

题面给定 Kimi K3 共 93 层，其中 69 层采用 KDA；官方对照形状为 `T=8192,H=96,D=128`。在 TP8 部署中，每卡 KDA head 数是 `96/8=12`。这两个形状承担不同任务：H96 用于复现官方表，H12 用于暴露真实 per-GPU 并行度问题。报告不以“69/93 层”代替时间占比。

还需澄清一个容易混淆的表述：这里的“SM80 kernel”只指 Tensor Core MMA atom。当前 FlashKDA 已使用 TMA 等新架构搬运能力；更准确地说，它是**在 B300 上运行、使用新搬运机制，但矩阵乘 atom 仍选择 `mma.sync.m16n8k16` 的 kernel**。

### 1.2 “值得迁移”的判据

我们把“值得”定义为同时满足四项：

1. 正确性不退化，并与题目指定的 `naive.py`、`chunk.py` 参考对拍；
2. 在 K3 代表形状上的收益稳定超过短期计时噪声和 dispatcher guard band；
3. 收益能被安全的运行时策略捕获，并有明确反例和 fallback；
4. 收益足以覆盖架构专用二进制、CUDA 13 编译链、TMEM/cluster 调试和持续 CI 的维护成本。

全文用三种标签区分证据边界：

- **[实测]** B300 上得到的 benchmark、SASS、NCU 或正确性数据，均给出 job 和原始文件；
- **[纸面模型]** 根据矩阵规模、数据类型或请求次数推算的代价，不冒充 profiler 实测；
- **[系统推断]** 从 operator 数据到 Kimi prefill、并发和 SLO 的条件性外推。

### 1.3 “Kernel is cheap”：为什么需要 Runtime Profile Agent

生成或改写一个 kernel candidate 已经相对便宜；难点是避免在错误 workload、错误物理 route 或错误统计口径上选中它。本项目因此不把贡献定义为“又写了几个 kernel”，而是实现一个 **Runtime Profile Agent**，把 C1 的迁移决策拆成五个可审计环节。

| 环节 | 结构化对象 | 要避免的误判 |
|---|---|---|
| Profile | local heads、seq lengths、total/max chunks、packed/fixed、SM count | 用 H96 官方表代替 TP8 每卡 H12，或忽略长序列关键路径 |
| Diagnose | SASS、NCU/Nsys/CUPTI、grid geometry、resource receipt | 看到旧指令就假定 compute-bound，或把低 occupancy 当唯一因果 |
| Propose | typed schedule delta、canonical candidate ID | 让自然语言 Agent 直接改任意 CUDA，或重复测同一语义候选 |
| Verify/measure | output/final state、scope parity、paired timing、置信下界 | winner's curse、跨 job 拼绝对时间、只报正样本 |
| Resolve | profile/resource/evidence-bound policy 与 fallback reason | 把单点 winner 写死，或把未知域默认为已支持 |

Agent 可以提出 ValueSlice、prefetch、cpc 或 `tcgen05` 候选，但 typed verifier、compiler receipt、correctness oracle、B300 executor 和 activation gate 保持确定性。这样，`tcgen05` 的负结果也是有效产出；ValueSlice 与 prefetch 的正结果也只有在反例和 fallback 明确后才进入候选策略。

#### Agent 如何自优化

自优化发生在候选策略层，而不是让模型在线修改生产 kernel。第 `r` 轮的 proposal source 会读取前 `r-1` 轮的 **conclusions-only memory**，其中只保存 candidate ID、通过或拒绝状态、证据引用和置信区间，不保存隐藏推理或未经压缩的测量数组。新候选先被规范化为 typed schedule 和 content-addressed ID；语义重复候选只测一次，verifier 拒绝原因会成为下一轮约束。通过 screen 的 top-k 候选进入独立 qualification，只有正确性、scope parity、正的置信下界和 fallback 全部满足时才更新 incumbent。若没有新候选、测量预算耗尽、连续若干轮没有更强证据，循环分别以 `no_new_candidates`、`screen_budget_exhausted` 或 `evidence_plateau` 停止。

因此它的“学习信号”不是 Agent 投票，而是可重放的 verifier diagnostic 和硬件证据。LLM 或多角色 Agent 可以扩大 proposal 的覆盖面，但不能自行越过 verifier、measurement 或 activation gate。当前原型证明了这套闭环程序与停止逻辑，尚未证明生产环境中的长期在线自演化收益。

论文阶段的 profile 工具源码随作业归档在 [`tools/runtime-profile-agent/`](../../tools/runtime-profile-agent/)。它不是完整生产 dispatcher，而是把“profile → 物理假设 → typed proposal → 测量 → shadow policy”闭环做成可复查的原型。

### 1.4 SM103 新能力分别是什么，与题目有什么关系

| 能力 | 含义 | 与 FlashKDA C1 的关系 |
|---|---|---|
| `tcgen05 + TMEM` | Blackwell 数据中心 Tensor Core 的异步 MMA 路径；累加器放在独立 Tensor Memory，而非普通寄存器，需要 alloc、descriptor、commit/wait 和读回协议 | 是“SM80 MMA 迁移 SM100”的直接候选，但必须把协议与布局成本一起计时；本项目在 Phase-6 实测 |
| TMA | Tensor Memory Accelerator，按张量描述符异步搬运 global/shared 数据，减少普通线程参与地址计算和搬运 | 官方 FlashKDA 已使用，因此不能把“改用 TMA”算作本项目的新迁移收益 |
| CTA Cluster / DSM / multicast | 多 CTA 组成 cluster；可访问 distributed shared memory，并让一次 TMA 搬运 multicast 给多个 CTA | 可针对 ValueSlice 的公共输入重复请求，但 cluster residency、同步和真实 bytes 尚需实测 |
| 更大 shared memory | SM80/A100 上限约 164 KiB/SM；本卡实测 233,472 B，即约 228 KiB/SM | 可容纳更深 staging 或更多协作数据，但不会自动修复只有 12 CTA 的整卡 underfill |
| FP8/FP6/FP4 | 更低精度 Tensor Core 与 block scaling 能提高适配 GEMM 的峰值和降低流量 | 更适合先评估 KDA 前后投影；recurrent state/局部逆若降精度，必须重新做数值与模型精度验证 |

这张表的作用是建立候选机制，不是用规格表代替实验。TMA 与 cluster 始于 SM90；`tcgen05`/TMEM 才是本题最直接的 Blackwell 计算路径。规格与指令约束参考 NVIDIA [CUDA Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) 和 [CUTLASS tcgen05 API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html)。

9 月 5 日保留下来的 Phase-6/Phase-1 预取不是一项新的 SM103 ISA 能力，而是对 B300 上既有 `mma.sync` 路径的调度专用化。它的意义在于进一步抬高迁移基线：若新的 MMA/TMEM 数据流不能击败“ValueSlice + 分阶段预取”的完整 forward，就不足以覆盖专版维护成本。

---

## 2. 阶段一：复现与测量

### 2.1 环境与可复现性

| 项目 | 版本或实测值 |
|---|---|
| GPU | NVIDIA B300 SXM6 AC，CC 10.3，148 SM |
| 单 SM shared memory | 233,472 B（约 228 KiB） |
| L2 | 132,644,864 B |
| Driver | 580.126.09 |
| PyTorch / CUDA runtime | PyTorch 2.10.0+cu130 / CUDA API 13.0 |
| FlashKDA | commit `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` |
| CUTLASS | pin `5c149f5` |
| FLA | 0.5.2 |
| 官方 benchmark 口径 | warmup 30，iters 200，repeats 5 |

官方干净 worktree 与补丁扩展分开加载，避免“baseline 也被补丁污染”。[实测，Job 17929] 5 个 case 的 output/final state 共 10 个 tensor 比较全部 bitwise equal；H12/T8192 的官方扩展和 patched V128 中位时间相差小于 1%，符合跨进程短测噪声范围。原始记录见 [`baseline_parity.json`](../../experiments/final_campaign/data/raw/baseline_parity.json)、[`official_timing.json`](../../experiments/final_campaign/data/raw/official_timing.json) 和 [`patched_v128_timing.json`](../../experiments/final_campaign/data/raw/patched_v128_timing.json)。

### 2.2 官方 benchmark 复现

[实测，Job 17926] 使用官方配置复现的主结果如下。表中 speedup 由同一行 `FLA chunk KDA / FlashKDA` 计算。

| 形状 | case | FlashKDA，BF16 state | FLA chunk KDA | FlashKDA speedup |
|---|---|---:|---:|---:|
| `T8192,H96,D128` | fixed | 1.0304 ms | 2.4155 ms | 2.34× |
| 同上 | ragged6 | 0.8612 ms | 2.4255 ms | 2.82× |
| 同上 | 8×1024 | 0.6963 ms | 2.3814 ms | 3.42× |
| `T8192,H64,D128` | fixed | 0.9410 ms | 1.6856 ms | 1.79× |
| 同上 | ragged6 | 0.6532 ms | 1.6990 ms | 2.60× |
| 同上 | 8×1024 | 0.4740 ms | 1.5964 ms | 3.37× |

完整 mean/min/max 及不同 public-state 模式见 [`01_official_benchmark_17926.log`](../../experiments/final_campaign/data/raw/01_official_benchmark_17926.log)。这些结果接近仓库的 GB200 表，说明官方 FlashKDA 已是强基线；后续重写不能仅用理论峰值证明价值。

### 2.3 SASS：题目所述 SM80 MMA 路径成立

[实测] K2 源码选择 `SM80_16x8x16_F32BF16BF16F32_TN`。对 recurrence cubin 的静态 SASS 汇总得到：

- `HMMA.16816.F32.BF16`：3,640 条；
- `TCGEN`/`UTCMMA`：0 条。

汇总和样例见 [`sass_opcode_summary.csv`](../../experiments/data/sass_opcode_summary.csv) 及 [`BOTTLENECK_ANALYSIS.md`](../../experiments/BOTTLENECK_ANALYSIS.md)。静态条数不等于每次调用的动态指令总数，但足以回答“编译后到底走哪类 Tensor Core 指令”。

### 2.4 NCU：真正的限制不是峰值 Tensor Core 或 HBM

[实测，Job 17965] 我们最终在题目代表的 per-GPU 形状 `T=8192,H=12,D=128` 上重采 NCU。两条路径采用同一 metric set，但 NCU replay 后的绝对 duration 不能与 CUDA Event benchmark 混用。

FlashKDA forward 的 K1 可以沿 token/chunk 展开；K2 则负责跨 chunk recurrent state update，同一 sequence-head 的下一 chunk 依赖上一 chunk state。官方 K2 grid 是 `(N,H)`，所以 TP8 单请求不是“总模型有 96 个 head”，而是每卡只有 12 个长生命周期 CTA。这正是 targeted NCU 选择 H12 的原因。

| 路径 | recurrence grid | CTA/148 SM 覆盖上限 | NCU duration | SM throughput | DRAM throughput | tensor pipe elapsed* |
|---|---:|---:|---:|---:|---:|---:|
| 官方 V128 | `1×12×1=12` | 8.1% | 1.27 ms | 2.64% | 1.24% | 2.48% |
| ValueSlice V16 | `1×12×8=96` | 64.9% | 901.22 µs | 7.22% | 1.83% | 3.50% |

原始 NCU 日志见 [`05_targeted_ncu_17965.log`](../../experiments/final_campaign/data/raw/05_targeted_ncu_17965.log)，导出结果见 [`05_targeted_ncu_summary_17965.csv`](../../experiments/final_campaign/data/raw/05_targeted_ncu_summary_17965.csv) 和 [`05_targeted_ncu_metrics_17965.csv`](../../experiments/final_campaign/data/raw/05_targeted_ncu_metrics_17965.csv)。

\* 表中采用相对整个 kernel elapsed cycles 的 `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`。同一 CSV 还导出了 active-cycle 口径：V128/V16 为 30.98%/5.43%。后者只在 SM 已活跃的周期上归一化；V16 激活更多 SM、把每个 CTA 工作切薄，因而 active-cycle 比例下降与整卡 duration 缩短并不矛盾。两种口径都不能脱离 grid 单独解释整卡计算利用。

结合 12/148 的 grid 覆盖、2.64% SM throughput、1.24% DRAM throughput 和 ValueSlice 扩 grid 后的降时，结论是：**官方 H12 K2 既不是传统 Tensor Core compute-bound，也不是 HBM bandwidth-bound；首要边界是 grid underfill、chunk 间 recurrence 依赖和 CTA 内 TMA/issue latency。** 同一 NCU 中，V128/V16 的 achieved occupancy 为 9.37%/4.69%，No Eligible 为 67.00%/85.06%；这说明更多 SM 被激活与单个活跃 SM 的 warp occupancy 下降可以同时发生，ValueSlice 并未消除 CTA 内等待。旧批次 Nsys 在 `T=4096,H=12` 同进程五轮 A/B 中也显示 GPU projected span `3.416→2.505 ms（−26.7%）`，且变化集中于 K2，prepare 几乎不变。该交叉证据见 [`BOTTLENECK_ANALYSIS.md`](../../experiments/BOTTLENECK_ANALYSIS.md)。

9 月 5 日的后续 profile 正是沿着“ValueSlice 已扩 grid、但 CTA 内仍有等待”继续推进。Phase-6 P4 和 Phase-1 lookahead 都保持 96 CTA、96 threads 和约 49.664 KiB shared memory；收益主要伴随 issue/eligible 提高和 short-scoreboard 降低，而不是 occupancy 上升或 HBM bytes 下降。详细数据放在 §4.5–4.6，不能与本节 Job 17965 的 V128/V16 profiler 绝对时间串成一条跨作业加速链。

### 2.5 Agent 如何在挑战之前得出主线判断

我们由确定性 parser 直接从上述 SASS、H12 targeted NCU 和
Phase-6 `tcgen05` 原始 CSV 生成同一个 `MmaMigrationProfile`，
再由 Runtime Profile Agent 的确定性
assessment 先生成结论，再生成候选。这一顺序避免了“先选一个想做的
kernel，再为它挑证据”。

| Agent finding | 物理证据 | 对 MMA 搜索的约束 |
|---|---|---|
| `MMA001` | recurrence SASS: HMMA 3,640，TCGEN/UTCMMA 0 | 题面指令路径成立 |
| `MMA002` | 12 CTA / 148 SM；SM/DRAM 2.64%/1.24% | 不得把“旧 MMA”等同于 compute-bound |
| `MMA003` | state 的 128 个 Value 行语义独立，切分不改变单元素归约顺序 | 先搜 Value 并行分解和 issue overlap |
| `MMA004` | V128 Phase-6 L0/L1 为 0.920×/0.256× | 停止 direct swap |
| `MMA005` | V16 core-only L0 1.501×，加入整合包络后 L1 0.778× | 若重开 `tcgen05`，必须搜跨 phase TMEM residency，而非孤立指令 |
| `MMA006` | 主导 phase M=16，当前 atom M=16，`tcgen05` 最小 M=64；只有 Phase-6 M=128 自然匹配 | 不对整条 recurrence 做指令级机械替换 |
| `MMA007` | V128 active blocks/SM: L0 `12→1`，L1 `5→1`；L1 SMEM `41472→45580 B`，registers `38→39` | 必须把与 TMEM/协议路径伴随的驻留下降纳入 gate，不只看单条 MMA 吞吐 |

因此 Agent 在进入挑战前已回答主问题：**不全面机械迁移
`tcgen05`；保留已验证的 `m16n8k16` 路径，先优化它的独立工作暴露和
TMA/MMA load-use 距离。** 输入及可重放输出分别见
[`c1_b300_h12_mma_profile.json`](../../experiments/runtime_profile_evolution/c1_b300_h12_mma_profile.json)
和
[`c1_b300_h12_mma_assessment.json`](../../experiments/runtime_profile_evolution/c1_b300_h12_mma_assessment.json)。
其中不包含挑战后的 ValueSlice 性能结果；`12→96 CTA`、挑战降时和并发反例只在阶段三用于验证候选并回写 memory。

---

## 3. 阶段二：六个讨论点——结论与证据

本节是对 Agent 主线判断的逐项展开，而不是在挑战结果之后倒推理由。挑战阶段只实现和验证本节保留的候选。

### 3.1 讨论点一：CHUNK=16 的三个理由，32/64 谁先破？

**结论：** CHUNK=16 同时照顾 BF16/FP32 指数范围、16×16 Neumann 求逆代价和 `m16n8k16` 自然形状。机械扩大时，**首先破的是当前无 rescale 的指数数值路径，其次是朴素 Neumann 密集扩展的计算代价；workspace 只是温和上升，不是第一约束。**

[纸面模型 + B300 FLA 小探针，Job 17935]

| CHUNK | `lower_bound=-5` 指数结果 | 首次 FTZ/overflow | 朴素 Neumann 每序列代价 | workspace/head |
|---:|---|---:|---:|---:|
| 16 | 可表示 | 无 | 1.00× | 6.750 MiB |
| 32 | 每通道 15 zero + 15 inf | token 18 | 5.33× | 7.125 MiB |
| 64 | 每通道 47 zero + 47 inf | token 18 | 26.67× | 8.063 MiB |

推算的依据是最坏累计门控 `C×(-5)` 及当前指数恢复方式；它不是随机输入出现 overflow 的概率。Neumann 模型是把 C16 当前密集幂级数直接延伸到更大 C，C32/C64 每 chunk 分别需要 8/10 个密集矩阵乘，虽然 chunk 数下降，总序列计算仍升至 5.33×/26.67×。完整公式和字段见 [`04_chunk_analysis_17935.csv`](../../experiments/final_campaign/data/raw/04_chunk_analysis_17935.csv)。

[实测，Job 17935] FLA 的 safe/block 设计在小形状 `T=128,H=1,D=128` 上，C32/C64 均 finite，中位延迟分别为 0.2500/0.2521 ms，输出 relative RMSE 为 0.3882%/0.3733%。这证明“通过 rescale/block solve 可以把大 CHUNK 做对”，但**不证明把 FlashKDA 的常量从 16 改为 32/64会加速**。大 CHUNK 已成为算法重设，而非指令级移植。

### 3.2 讨论点二：`tcgen05` 最小 tile 与 CHUNK=16 匹配吗？

**结论：部分匹配，但不支持整体机械替换。**

SM100 BF16、CTA-group 1 的 `tcgen05` 支持 `M∈{64,128}`、`N=8..256` 且以 8 递增、`K=16`。因此不能笼统说“CHUNK16 与 tcgen05 不匹配”，也不能误说最小 N 是 64：

- K2 Phase-6 为 `[128,16]@[16,V]`，自然映射到 `m128nVk16`，V16/32/64/128 全部合法；
- K2 其他大量以 CHUNK 为 M 的 `[16,V]` phase，保持当前朝向不能直接映射，需要转置、重排并重设跨 phase 数据流。

[实测，Jobs 17936/17937] 我们选择**最有利于新指令**的 Phase-6 做隔离 probe。两边均使用 BF16 输入、FP32 累加；SASS 确认基线为 `HMMA`、候选为 `UTCHMMA`。L0 让两边在进入计算时各自获得偏好的片上布局，是对 tcgen05 乐观的下界；L1 再加入 state/gate 和保守 scalar U 重排，用于暴露集成风险，但不是优化后实现的上界。

| V / grid / inner | L0 `mma/tcgen` | L1 `mma/tcgen` | 直接解释 |
|---|---:|---:|---|
| 128 / 12 / 1 | 0.376× | 0.533× | one-shot 的 TMEM 协议固定成本大 |
| 128 / 12 / 64 | 0.920× | 0.256× | 即使摊薄，K3 V128 的乐观口径仍慢 8.7% |
| 128 / 148 / 64 | 0.904× | 0.256× | 满机 grid 探针方向相同 |
| 16 / 12 / 64 | 1.501× | 0.778× | core 有形状相关潜力，但当前转换会吃掉收益 |

主结果、边界和完整表见 [`analysis_tcgen05.md`](../../experiments/final_campaign/analysis_tcgen05.md)、[`03_tcgen05_probe_17937.csv`](../../experiments/final_campaign/data/raw/03_tcgen05_probe_17937.csv) 与 [`03_tcgen05_probe_17937.sass`](../../experiments/final_campaign/data/raw/03_tcgen05_probe_17937.sass)。32 个 timing row 在两个独立 job 间，MMA/TCGEN 中位时间相对变化的中位数均约 0.004%。

这支持一个明确的 stop decision：**不集成“保持当前 K2 数据流、只替换 Phase-6 MMA”的版本。** 但本 probe 不是完整 K2，也没有跨 Phase 1/3/4/6 保持 TMEM-resident 数据；它不能证明所有 SM100 数据流重设永远无收益，更不能把表中微秒直接外推成 K3 prefill 变化。

### 3.3 讨论点三：chunk 间有状态依赖，并行度还能从哪里来？

**结论：** chunk 间真正的递推不能凭空并行；可利用的是 sequence、head 或 Value 行等正交维度。对 TP8 单请求 H12，最直接的是 Value 维；cluster 只有在先找到这种独立维度后才有用。

| 候选 | 可能收益 | 反例与成本 | 当前判断 |
|---|---|---|---|
| 更多 batch/sequence/head | 自然增加 CTA | 低并发单请求时不存在；高并发时本就不缺 CTA | 由 serving 负载提供，不是通用 kernel 解 |
| 多 head 合入一个 CTA | 可共享控制或某些输入 | 让 CTA 数更少，shared-memory/state 压力更高 | 与当前 underfill 方向相反 |
| persistent kernel | 摊薄 launch/设置，利于片上重用 | 12 个 persistent CTA 仍喂不满 148 SM，也不消除每 head 的 chunk 依赖 | 可作辅助，不是首要解 |
| 2-CTA/CTA Cluster | 提供 DSM、cluster sync、TMA multicast | 必须先存在可分任务；cluster residency 和同步会压 occupancy | 与 ValueSlice 结合最合理 |
| ValueSlice | Value 行独立，CTA `12→96`，无 reduction/atomic | slice-independent 输入被重复读取；短序列/高自然并行度会变慢 | 本项目第一级挑战路线 |
| CTA 内分阶段预取 | 在 V16 单 compute warp 内重叠独立 keyblock 的 load/use，保持 chunk 递推和算术次序 | 增加寄存器生命周期；无初态 Phase-1 L4 有控制 spill；双 stream 可能回归 | 9 月 5 日默认关闭的延迟候选 |

[纸面模型] `T=4096`、单 sequence-head 下，V16 的 source-request 为 29.188 MiB，其中多个 slice 重复请求 common inputs。若 8-CTA cluster 可以理想 multicast 一次这些输入，可节省 23.734 MiB（81.3%），把请求量降到接近 V128 的 5.453 MiB。这个数是源请求模型，**不是实测 HBM bytes，也没有计入 cluster occupancy/同步成本**。

### 3.4 讨论点四：这是 compute-bound 还是 memory-bound？

**结论：两者都不是传统意义上的主瓶颈。**

[实测] Job 17965 中官方 H12 的 SM/DRAM throughput 仅 2.64%/1.24%，却因 grid 只有 12 CTA 最多覆盖 8.1% 的 SM；V16 将 grid 扩到 96 CTA 后，NCU duration 从 1.27 ms 降至 901.22 µs，但 SM/DRAM throughput 仍只有 7.22%/1.83%。结合 Nsys 中减时集中在 K2，可将当前边界描述为：

```text
grid underfill + chunk recurrence critical path + CTA 内 TMA/issue latency
```

不能仅凭低 occupancy 或高 No Eligible 下因果结论，也不能用 active-SM 的 tensor pipe 指标代替整卡利用率。可靠判断来自 duration、grid、SM/DRAM throughput、L2、active warps、scheduler stall 和 timeline 的交叉。

这也解释了为什么“B300 BF16 峰值更高”没有自动转化为 K2 低延迟：当 136 个左右的 SM 无 CTA 可执行时，提高每条 Tensor Core 指令的峰值并未解决首要限制。

[实测，Jobs 19901/19935] ValueSlice 解决的是第一级 grid underfill，分阶段预取处理的是第二级 CTA 内 issue latency。Phase-6 `StatePrefetch=1→4` 的目标 K2 achieved occupancy 几乎不变（4.690%→4.688%），但 issue active 从 14.72% 升至 18.12%，short-scoreboard cycles/issued-instruction 从 0.9010 降至 0.4649。最终 Phase-1 候选相对 Phase-6-only 基线也呈现 issue/eligible 上升和 short-scoreboard 下降。该组计数器是一次 16-pass replay 的机制旁证，不是重复性能样本；stall 指标是每条已发射指令对应的周期，不能相加成 wall-time 分解。

### 3.5 讨论点五：BF16 state 精度怎么验证，结果如何？

**结论：在本次 kernel 级长序列、ragged、state carry 和 long-memory gate 测试中，BF16 路径相对 FP32 naive/FLA reference 的 output/state relative RMSE 均低于 1%，ValueSlice 没有增加误差；但这不是模型级 perplexity 或任务精度证明。**

[实测，Job 17934]

- 200 条 comparison row 全部 finite；本报告报告观测误差，不把脚本记录字段包装成统一的模型精度阈值；
- 98 条 V16/V32/V64 vs V128 比较全部 bitwise equal，最坏 relative RMSE 为 0；
- `T=8192` random，Flash vs naive：output 0.5698%，final state 0.4728%；
- `T=8192` long-memory，Flash vs naive：output 0.8240%，final state 0.7405%；
- Flash vs FLA chunk 最坏：output 0.9131%，state 0.8151%；
- K3 H12 fixed vs FLA：output 0.5737%，state 0.4777%。

测试明确关闭参考实现回落到 FlashKDA：`FLA_FLASH_KDA=0`。参考版本为 FLA 0.5.2；`naive.py` SHA-256 为 `60a32285…f016`，`chunk.py` 为 `a15aa6ac…e9b8`。完整 case、gate、state mode、dtype 和 seed 见 [`03_reference_correctness_17934.csv`](../../experiments/final_campaign/data/raw/03_reference_correctness_17934.csv) 与 [`03_reference_correctness_17934.log`](../../experiments/final_campaign/data/raw/03_reference_correctness_17934.log)。

本轮公开 BF16/FP32 state buffer 路径得到相同精度，是因为现有 FlashKDA 在内部仍按 BF16 舍入点计算。正确说法是“**只把 public state buffer 改成 FP32 没有构成全 FP32 recurrence 对照**”，而不是“FP32 state 没用”。模型级结论还需长文本 perplexity、下游任务和真实 checkpoint。

### 3.6 讨论点六：假如我们是作者，v2 出不出 sm100a 专版？

**结论：出 guarded hybrid 专版，不出全面 `tcgen05` 分叉。**

建议发布内容：

1. 保留官方 V128 `mma.sync`，作为 bitwise-compatible、跨形状 fallback；
2. 对 B300/SM103 的已标定 single-call shape 域保留 ValueSlice，并由部署侧保证其低并发使用条件；
3. packed 单序列按 fixed B1 进入已标定策略，多序列 varlen 在没有分布模型时仍回退 V128；
4. 将入口 hardening 作为公共安全修复；Phase-6 P4 与 Phase-1 lookahead 则只在显式 `sm_103a`、双编译开关和既有 ValueSlice guard 同时满足时构建；
5. 由于当前 guard 不能感知其它 stream/request，Phase-1 候选默认关闭，由部署侧明确选择 latency 模式；吞吐模式暂留 Phase-6-only 或旧路径；
6. 优先取得真实请求分布和并发矩阵，再评估 CTA Cluster + TMA multicast；`tcgen05` 只在跨多个 phase 的转置/TMEM-resident 数据流通过新 gate 后重新考虑。

专版的负担包括 CUDA 13/`sm_103a` 构建链、更大二进制、多个 layout 的 CI 以及与通用版本长期同步。`N=1` 只描述一次调用内部的 sequence 数，不代表 GPU 上没有另一个活跃请求；因此“低并发”目前是部署契约，不是 dispatcher 已实现的在线检测能力。专版必须 opt-in/guarded，不能替换已经很强的可移植基线。

下表是本项目的最终路线决策：

| 路线 | 当前证据 | 决策 |
|---|---|---|
| 全面 `mma.sync→tcgen05` | K3 V128 Phase-6 的 L0 摊销口径仅 0.920× | **停止直接替换，不集成** |
| CHUNK 32/64 机械放大 | token 18 FTZ/overflow；Neumann 5.33×/26.67× | **停止，除非改算法** |
| ValueSlice recurrence 重构 | 单请求长 prefill 降时约 27%，bitwise equal；短序列有明确反例 | **受保护发布** |
| Phase-6 `StatePrefetch=4` | 相对旧 V16：无初态约 7%–9%，有初态约 17%–20%；有初态双 stream 仅约 0.47% | **默认关闭的 opt-in RC** |
| Phase-1 L4/L2 | 相对 Phase-6-only eager 再降 5.15%–12.40%；无初态双 stream 回归 1.52% | **保留候选，不作吞吐默认** |
| Cluster + TMA multicast | 纸面上可去除大部分 common-input 重复请求，尚无 cluster 实测 | **并发/端到端之后再评估** |

---

## 4. 阶段三：挑战——ValueSlice 与分阶段流水重构

### 4.1 为什么这仍然是“迁移 SM100 是否值得”的挑战

题目允许挑战“只换指令、大 CHUNK+rescale、并行度重构”之一。已有实验否定了所测 Phase-6 direct swap 的性能收益，并发现机械扩大 CHUNK 会使当前指数路径数值失效；这些结果没有否定大 CHUNK + rescale/block solve，也没有穷尽其他 SM100 数据流。该阶段选择**并行度重构**，针对 B300/SM103 上观察到的 12 CTA/148 SM，完成题目允许的一条挑战路线。它保留 HMMA，因此属于相关优化证据，不能代替对 MMA 迁移价值的分析。

9 月 3 日的 ValueSlice 先利用 Value 行独立性增加 CTA；9 月 5 日的增量再在每个 V16 CTA 内利用同一 chunk 的独立 keyblock 拉开 load/use 距离。前者处理卡级 underfill，后者处理单 compute warp 的发射等待；两级都保持时间 chunk 之间的 recurrence 串行依赖。

### 4.2 分解方式与算术不变量

K2 state 的 Value 行相互独立。将 `D=128` 按 `V∈{16,32,64,128}` 切分，每个 CTA 只更新一个 `V×D` state slice，不需要跨 CTA reduction、atomic 或改变单个输出元素的归约顺序。grid 从 `(N,H)` 变成 `(N,H,D/V)`：

| slice | 每个 sequence-head 的 CTA | H12 总 CTA |
|---:|---:|---:|
| V128 | 1 | 12 |
| V64 | 2 | 24 |
| V32 | 4 | 48 |
| V16 | 8 | 96 |

[纸面模型] 设 `C=16,D=128`，每 CTA、每 recurrence tile 的 Tensor Core 工作为

```text
F_cta_tile(V) = 6·C·D·V + 4·C²·V
```

slice 数为 `D/V`，因此总 Tensor FLOP 与 V 无关。优化来自更多独立 CTA，代价是重复搬运 slice-independent 输入。核心实现见 [`0001-k2-value-slice-and-dispatch.patch`](../../patches/0001-k2-value-slice-and-dispatch.patch)。

### 4.3 正确性：三层对拍

1. [实测，Job 17929] 官方干净扩展 vs patched V128：10/10 tensor bitwise equal；
2. [实测，Job 17934] V16/V32/V64 vs V128：98/98 comparison row bitwise equal；
3. [实测，Job 17934] 其余 102 条独立参考关系全部 finite，其中 92 条是 FlashKDA vs 题目指定 `naive.py`/`chunk.py`，10 条是 FLA chunk vs naive；观测最坏 relative RMSE 为 0.9131%。

bitwise equal 的原因不是容差宽松，而是 ValueSlice 仅拆分互相独立的 Value 行，没有改变每一行内部的运算/归约顺序。

### 4.4 性能：正收益、反例和 workload-dependent 最优点

[实测，Job 17947] 所有 case 均为 total `T=8192,H=12,D=128`，表中使用 3 个 repeat median 的中位数。

| 输入分布 | V128 | 最佳强制 slice | 最佳延迟 | 相对 V128 | dispatcher 结论 |
|---|---:|---:|---:|---:|---|
| fixed 1×8192 | 0.7807 ms | V16 | 0.5691 ms | −27.1% | auto 选 V16；auto 为 0.5698 ms（−27.0%） |
| packed 1×8192 | 0.7850 ms | V16 | 0.5740 ms | −26.9% | **升级后 auto 选 V16** |
| packed ragged6 | 0.3403 ms | V64 | 0.2810 ms | −17.4% | 未标定，auto 保守 V128 |
| packed 8×1024 | 0.1561 ms | V64 | 0.1540 ms | −1.3% | 低于 3% guard，V128 合理 |
| packed 32×256 | 0.1253 ms | V128 | 0.1253 ms | 0% | V16 为 0.2585 ms，慢约 106% |

完整逐轮 CUDA Event 数据和 decision dump 见 [`05_dispatch_upgrade_17947.csv`](../../experiments/final_campaign/data/raw/05_dispatch_upgrade_17947.csv) 与 [`05_dispatch_upgrade_17947.log`](../../experiments/final_campaign/data/raw/05_dispatch_upgrade_17947.log)。最关键的反例是：相同 total tokens 并不代表相同最优 V；sequence 数增多、每条变短后，自然并行度已足够，额外切分只剩调度和重复流量成本。

原 dispatcher 将所有 `cu_seqlens!=None` 视为未标定 varlen，因而错过 packed 单请求。增量补丁 [`0002-dispatch-packed-single-sequence.patch`](../../patches/0002-dispatch-packed-single-sequence.patch) 只读取 `cu_seqlens.numel()` 元数据，识别 `nseq=1` 并复用 fixed B1 策略；它不读取设备上的长度值，不引入 host sync。`nseq>1` 仍保守回退。Job 17947 同时验证了 fixed 与 packed 单序列都选 V16，而 ragged6、8×1024、32×256 都选 V128，且所有强制 slice 仍与 V128 bitwise equal。

### 4.5 Phase 6：把 V16 的状态预取窗口从 1 扩到 4

[实测，Jobs 19901/19903] ValueSlice 把 H12 的 K2 扩成 96 CTA 后，每个 V16 CTA 仍只有一个 compute warp。Phase 6 同一 chunk 内有八个独立 keyblock；`StatePrefetch=4` 先装入 0–3，消费并补入 4–7，再排空后半段。矩阵指令、state FMA、BF16 舍入点、输出与 barrier 契约不变。

干净四补丁候选使用真实未改 Python wrapper，并与独立旧 V16 二进制同作业比较。`T8192,H12`、BF16 initial+final 的结果为：

| 计时口径 | 旧 V16 / `StatePrefetch=1` | V16 / `StatePrefetch=4` | 新增降时 |
|---|---:|---:|---:|
| eager CUDA Event | 0.569712 ms | 0.459184 ms | 19.40% |
| CUDA Graph replay | 0.566816 ms | 0.454304 ms | 19.85% |
| 256 MiB cache perturbation 后的 CUDA Event | 0.571376 ms | 0.459808 ms | 19.53% |
| 双 stream 两请求 joined pair | 0.672208 ms | 0.669040 ms | 0.47% |

状态接口补测不能省略：带初态（both/in）的已测 eager 增量约 **16.8%–19.6%**，无初态（out/none）约 **7.4%–9.1%**。前者对应续段状态 carry，不能代替首段 prefill headline。120 条干净 release out/final 跨路径比较逐位通过；定向 memcheck、synccheck 均退出 0 且 `ERROR SUMMARY: 0 errors`。multi-GPU 与单独 alias-extension 检查仍为 SKIP。原始 wrapper/state 数据见 [`release_19901.log`](../../experiments/mainline_20260905/data/release_19901.log) 和 [`state_matrix_19903.log`](../../experiments/mainline_20260905/data/state_matrix_19903.log)。

该候选默认编译关闭。它要求显式 `FLASH_KDA_CUDA_ARCHS=103a` 与 `FLASH_KDA_ENABLE_V16_PREFETCH4=1`，且运行时已有策略选中 V16、D128/C16、非 FP32 public state、N1/H12、`2048≤T≤8192`。编译后仅在运行时取消 flag 无效；若要恢复旧 V16 / `StatePrefetch=1`，需要使用未启用该 flag 的独立二进制。

### 4.6 Phase 1：按初态合约选择 L4/L2，并保留并发负例

[实测，Jobs 19934/19935] 在 Phase-6 P4 内，Phase 1 的 `k @ state` / `q @ state` 使用 k/q/state 三组 fragment ring：无 initial state 选择 lookahead 4，有 initial state 选择 2。两个消费者完成后才覆盖 slot，仍按原 k=0..7 顺序累加；HasStateOut 不增加另一维调参，fixed 与合法 packed 单序列采用同一 HasStateIn 规则。

以下 headline 只来自 Job 19934 的同输入、同作业配对；P4 基线、Phase-1 候选和强制 V128 是三个独立加载的真实 wrapper 路径。表中为三轮 round median 的中位数：

| `T8192,H12` fixed，BF16 | Phase-6-only P4 | Phase-1 候选 | 相对 P4 | 同作业强制 V128 | 相对 V128 累计降时 |
|---|---:|---:|---:|---:|---:|
| 无 initial、有 final：首段 | 0.950848 ms | 0.862208 ms | 9.32% | 1.373680 ms | 37.23% |
| 有 initial、有 final：续段 | 0.796144 ms | 0.743968 ms | 6.55% | 1.365632 ms | 45.52% |

40 个性能形状中，34 个位于准入域、6 个是回退对照。34 个准入域形状的 eager/Graph/cache-perturbed/host-wall 四种中位口径全部获益，eager 增量为 **5.15%–12.40%**；408 个准入域 paired-round gain 全为正，最小 3.72%。这不等于穷尽 `2048..8192` 的每个整数或真实请求分布；6 个回退点的四口径中位漂移约在 ±0.9% 内，单轮最坏回归 1.32%。完整数据见 [`clean_19934.log`](../../experiments/mainline_20260905/data/clean_19934.log)。

必须同时展示的反例是两个 stream 各运行一个 T8192 请求的 joined-pair：有初态基本持平；无初态由 **1.147440 ms 增至 1.164832 ms，回归 1.52%**，三轮方向一致。这个时间不能除以二称作单请求 latency，也不能据此声称 serving throughput。当前 `N=1` guard 只看单次调用内部形状，无法知道设备上是否还有另一个 stream；因此候选只能由调用方明确选择低并发 latency 模式，不能无条件替换 Phase-6-only 构建。

验证口径保持分层：120 条主比较和 14 条尾块/状态补比较逐位通过；状态链覆盖真正从 None 开始再 carry final state；80 条 Graph 跨路径比较、80 条计时后跨路径比较和 4 条双 stream 正确性比较通过。每项 sanitizer 各覆盖 20 条定向比较并报告 0 errors，但不是全矩阵 racecheck。9 月 3 日的 naive/FLA 参考精度仍由 Job 17934 提供；这些新增 bitwise 检查是相对既有实现的回归验证，不是新的高精度 oracle。

SASS/NCU 也保留不利事实：无初态 L4 使用 56 registers、8 B stack，并产生与 SASS 精确吻合的 245,760 个 local spill 请求；动态 `inst_executed` 反而增加 0.69%。有初态 L2 使用 72 registers、无 local spill。两者保持 Phase-1 32 条 HMMA 和全 tile 52 条 HMMA，issue/eligible 提高、short-scoreboard 降低，说明即使存在控制开销也可净胜；但编译器同时重排了 Phase 6，不能把全部节省归因到某一个 PC。四份 NCU 各是一次 16-pass replay、未固定 clocks/caches，只作机制互证。原始 CSV 见 [`mainline_20260905/data/`](../../experiments/mainline_20260905/data/)。

Phase-1 候选同样默认关闭，且依赖 Phase-6 P4：构建需要同时设置 `FLASH_KDA_ENABLE_V16_PREFETCH4=1` 与 `FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH=1`。完整补丁链、hash、回滚和已知 SKIP 见 [`mainline_20260905/`](../../experiments/mainline_20260905/)。

### 4.7 为什么没有继续把 `tcgen05` 塞进完整 K2

挑战阶段不要求正向加速，关键是数据驱动的 stop/go。Phase-6 是整个 K2 中最自然的 `m128nVk16` 候选；如果在 L0、inner=64 的乐观口径下，正式 V128 仍慢 8.7%，完整集成还要承担 U 布局转换、TMEM 生命周期、barrier、跨 phase 兼容和回归测试，就没有足够的正向信号占用剩余 B300 槽位。我们因此把工程时间投入能直接击中 NCU 瓶颈、并已通过正确性和性能 gate 的 ValueSlice。

9 月 5 日的新结果进一步提高了比较门槛：未来 `tcgen05` 候选必须在完整 forward 中击败 guarded ValueSlice + 分阶段预取，而不能只战胜 9 月 3 日的 V128 microbench 基线。它仍不证明跨 Phase 1/3/4/6 的 TMEM-resident 重写永远无收益。

### 4.8 Runtime Profile Agent：从固定 cpc 到物理容量规则

论文阶段又在独立的 H12 BT16 CAKE-generated prepare/chain route 上验证 profile 工具。它不修改 MMA、数值算法或 public ABI，只让工具根据 workload 与编译后物理资源推荐 prepare 的 chunks-per-CTA。

容量规则为 `cpc_cap(W,H,S,R) = ceil(W / floor(S*R/H))`，其中 `W/H/S/R` 分别是 total chunks、local heads、SM 数和该 prepare kernel 的实测 resident CTA/SM。本次 `H=12,S=148,R=5`，resident grid capacity 为 740 CTA；`R=5` 绑定该 kernel 的 45,056 B shared memory、128 threads、寄存器和 occupancy receipt，不是 B300 通用常数。

首次 GPU 查询前封存 W384/W768、cpc7/cpc13、八个独立进程 epoch、ABBA/BAAB 顺序与 Bonferroni 单侧 98.75% 判据，并禁止 candidate screen 或邻域搜索。

| Profile | W | cpc9→预测值 | grid 变化 | Speedup | 单侧 98.75% lower |
|---|---:|---:|---:|---:|---:|
| Fixed 6144 | 384 | 9→7 | 516→660 | 1.0133× | 1.0129× |
| Balanced 4-way | 384 | 9→7 | 516→660 | 1.0331× | 1.0322× |
| Fixed 12288 | 768 | 9→13 | 1032→720 | 1.0494× | 1.0488× |
| Balanced 4-way | 768 | 9→13 | 1032→720 | 1.1354× | 1.1352× |

四个 profile 的 output/final state 最大绝对差均为 0，八个 process epoch 全部同方向。W384 需要增加 CTA、W768 需要减少 CTA，两边都加速，因此结果否定“CTA 越多越好”和“CTA 越少越好”的单调解释。

资格通过后才运行 CUPTI。prepare 节省 3.89–31.57 µs，解释 97.98%–101.16% 的 full-span 节省；chain 的 95% 区间均在事前冻结的 ±1% 等价带内。该证据把 profile 工具从事后调参推进到前瞻预测和阶段机制互证，但当前只产生 shadow recommendation。完整说明和证书见 [`runtime_profile_evolution/`](../../experiments/runtime_profile_evolution/)，不能与官方 FlashKDA ValueSlice/P4/Phase-1 百分比合并。

这一容量规则是闭环可迁移的物理结论之一：Agent 不记住“cpc7 永远更快”，而是记住候选成立所需的 workload 和 resource receipt，再由下一轮 proposal 针对新的 `W/H/S/R` 重新计算。W384 选择更小 cpc、W768 选择更大 cpc，正好说明自优化应学习物理条件与适用域，而不是记忆单点赢家。

---

## 5. 从算子到 Kimi K3：prefill、并发、SLO、通信与环境边界

### 5.1 能严格声称什么

[实测 + 系统映射] FlashKDA forward 是 KDA prefill 的直接组成部分。TP8 单请求时每卡 H12，官方 K2 只有 12 CTA；ValueSlice 因而最可能改善**低并发、长 prompt 的 KDA prefill latency**，进而改善 TTFT 的一个组成部分。Job 17947 的 packed 1×8192 是比 fixed tensor 更贴近 serving 调用约定的证据：升级后的 dispatcher 确实捕获了这个单请求机会。

[实测] 当 total tokens 同为 8192、sequence 数从 1 增到 8/32 时，负载自身提供更多并行度，ValueSlice 收益消失甚至反转。因此“并发更高”不是自动获得更高 ValueSlice speedup；实际是低并发单请求最受益，高并发应回退或选更粗 slice。

[实测] 9 月 5 日候选在已测单请求域内进一步改善完整 FlashKDA forward；但双 stream 无初态 joined-pair 回归 1.52%。因此可以严格声称“存在经过验证的单请求 latency 候选”，不能声称 dispatcher 已能识别设备级低并发。部署方必须用真实调度器/请求分布决定是否启用独立 latency 构建。

### 5.2 不能直接声称什么

- **不能说 TTFT 降低 27%、37% 或 46%。** 这些分别是旧 ValueSlice 或 9 月 5 日候选在指定 state 合约和单请求形状下的 FlashKDA forward operator 降时；KDA 层还包含投影、norm、gate 等计算，模型还有非 KDA 层。
- **不能说 TPOT/decode 已加速。** 当前策略对 `T=1` 回退 V128；实际 serving 的纯 decode 还有独立 fused KDA decode 路径。
- **不能说 SLO goodput 或并发吞吐已提升。** 本项目只有一个双 stream、双请求 joined-pair microbench；没有运行完整 Kimi checkpoint、scheduler、continuous batching、排队和 SLO sweep。
- **不能说 TP8 通信已改善。** 单 B300 只能模拟 per-GPU H12 计算形状，没有实测 NCCL all-reduce/all-to-all；局部 K2 降时不会自动减少跨卡通信。

### 5.3 Amdahl 敏感性，而非端到端结果

[系统推断] 令 `p` 为 FlashKDA forward 在完整 prefill wall time 中的占比。先保留 9 月 3 日 guarded ValueSlice 的 `r=0.27` 作为既有策略的敏感性示例；忽略重叠变化时：

```text
prefill 降时约为 p·r
理想容量加速约为 1 / (1 - p·r)
```

| 假设 p | 预计 prefill 降时 | 理想容量加速 |
|---:|---:|---:|
| 20% | 5.4% | 1.057× |
| 40% | 10.8% | 1.121× |
| 60% | 16.2% | 1.193× |

这只是敏感性分析。9 月 5 日最终候选的 `r` 不能统一替换成一个更大的数字：同 Job 19934 中首段无初态为 0.3723，有初态续段为 0.4552，且双 stream 已有负例。SLO goodput 是否提升还取决于 TTFT/TPOT 哪个约束绑定、请求长度分布、batching、排队非线性以及新路径对并发资源驻留的影响。

### 5.4 通信视角：卡内与跨卡必须分开

**卡内通信。** ValueSlice 的收益来自跨 CTA 并行，成本恰好是 common inputs 被多次 TMA 请求。SM100 家族的 CTA Cluster、DSM 和 TMA multicast 对应一个具体问题：多个 slice CTA 能否共享一次 common-input 搬运。它们仍是合理的后续数据复用手段，但 9 月 5 日的双 stream 负例把真实并发矩阵、调用分布和交付集成排到了更高优先级。之后仍需用 2/4/8-CTA cluster microbench 实测 cluster residency、multicast bytes、同步和 duration，才可把 81.3% 纸面 request reduction 写成性能收益。

**跨卡通信。** TP8 会带来模型并行通信，但 FlashKDA 内部的 ValueSlice 没有跨 GPU 通信，也不改变 collective 的消息体。因此需要分别测 compute-only 与 NCCL-overlapped timeline，才能判断 0.21 ms 级单算子节省会被通信隐藏、暴露还是放大。

### 5.5 环境与运营视角

B300 更大的 shared memory、L2、显存、带宽和新低精度能力，能支持更深 pipeline、更大 batch、长上下文和 KDA 前后投影 GEMM；但 Job 17965 证明，12 CTA 的 K2 不会仅因硬件峰值更高而自动变快。FP8/FP6/FP4 更适合首先评估投影 GEMM；在没有模型精度证据时，不应直接用于 recurrent state 和 16×16 inverse。

运营报告的合理作用是提供 prompt length、并发分布、TTFT/TPOT SLO 权重和流量波峰，从而决定 dispatcher 标定域与收益加权；它不能代替 SASS/NCU/microbench 回答 MMA 迁移本身。当前没有 Kimi K3+B300 官方线上 trace，因此本报告没有用第三方“运营提升”数字填补端到端证据空缺。

当前两个预取特性均是 build-time opt-in，运行时取消环境变量不能改变已编译二进制；同时现有诊断只解释 ValueSlice，不完整报告 Phase-6/Phase-1 子变体。生产化之前应补充构建身份与实际子变体可观察性，并完成多 GPU guard、单独 alias build、包级安装/回滚和完整模型/服务测试。

### 5.6 15 分钟 B300 权限如何影响方法，而非降低结论标准

每个 Slurm job 自包含环境、commit/hash、GPU 信息、warmup、计时和 CSV 落盘；编译、正确性、性能和 profiler 分层，关键路径采用小 metric set，避免 NCU replay 超时。负结果触发预先设定的 stop gate，不继续扩大实现。9 月 3 日 campaign 的关键 job 均远低于 15 分钟：官方 benchmark 93 s，ValueSlice sweep 12 s，参考正确性 33 s，CHUNK 分析 41 s，tcgen05 probe 16 s，最终 NCU 15 s。9 月 5 日的 P4 release、state matrix、Phase-1 clean acceptance 和 matched NCU 分别归档为 Jobs 19901/19903/19934/19935，不与前述 214 s 汇总或跨作业绝对时间合并。资源限制影响的是“无法跑多卡/完整 K3 serving”，不影响单卡 instruction、kernel 和 dispatcher 因果链的可复核性。

---

## 6. 威胁有效性与限制

1. 只有一张 B300，未实测 TP8/NCCL 或端到端 Kimi K3；H12 只是 per-GPU 计算形状，multi-GPU guard 和单独 alias-extension GPU 检查仍为 SKIP。
2. `tcgen05` 是真实 Phase-6 隔离 probe，不是完整 K2；L0 偏乐观，L1 的 scalar 重排偏保守。
3. CHUNK32/64 的安全路径只在 FLA 小形状上验证，没有完成 FlashKDA 大 CHUNK 重写。
4. ValueSlice 的最优 V 依赖 sequence distribution；当前生产可辩护的 dispatcher 只新增 packed 单序列，multi-sequence varlen 仍回退。
5. BF16 state 结论是 kernel 数值对拍，不是模型级 accuracy/perplexity。
6. 只有一次双 stream、双请求 joined-pair microbench，没有真实 serving 并发吞吐、TTFT、TPOT 或 SLO goodput；Amdahl 表只用于说明端到端收益边界。
7. NCU、Nsys、CUDA Event 和 host wall 的 instrumentation 不同，绝对时间只在各自一致口径内 A/B；Job 19934 只在开头采样到一次 1095 MHz，Jobs 19935 与更早 profile 的 GPC 频率也不同，报告不跨工具/跨作业拼接毫秒数或按频率“校正”。
8. Phase-6/Phase-1 guard 接受 `2048≤T≤8192`，但性能只覆盖有限整数、state/layout 组合；34 个准入域样本不等于穷尽整个区间。
9. Phase-1 无初态 L4 存在 8 B stack 和真实 local spill；其 SASS 还伴随 Phase 6 重排。NCU 支持调度机制，但不能把全部收益归因到某一个 PC 或声称消除了动态指令。
10. 两级预取默认编译关闭；现有 guard 不感知其它 stream/request，尚未完成实际包安装/回滚、完整模型或多产品资格验证。
11. Runtime Profile Agent 的 capacity rule 只在同一 B300/H12/BT16 CAKE-generated route 与同一 prepare resource receipt 上前瞻验证；它不是官方 FlashKDA 补丁的叠加加速，也没有跨 kernel、跨 GPU 验证。

---

## 7. 最终结论

对 C1 的直接回答是：

> **SM100 值得利用，但“利用 SM100”不等于“把 SM80 MMA 全换掉”。**

官方 FlashKDA 在 B300 上仍使用 `mma.sync` 不是一个仅靠“代际更老”就能判错的决定。CHUNK16 与现有算法/数值/MMA 形状高度耦合；正式 K3 V128 的 Phase-6 `tcgen05+TMEM` 在最乐观摊销 probe 中仍慢 8.7%，当前没有全面 instruction rewrite 的投资依据。

相反，B300 上最明确的瓶颈是 TP8 单请求 H12 导致的 12 CTA/148 SM underfill。ValueSlice 保持总 Tensor FLOP 和每个 Value 行的运算顺序不变，把 grid 扩到 96 CTA，fixed 与 packed 单请求均获得约 27% operator 降时。9 月 5 日又证明，在 96 CTA 内优化 Phase-6/Phase-1 load-use 调度，可让 `T8192,H12` 同作业相对 V128 的完整 forward 累计降低 37.23%（首段无初态）或 45.52%（有初态续段）。这两个数字不是同一 state 合约，也不是端到端模型收益。

多短序列反例和无初态双 stream 1.52% 回归证明新路径不能无条件启用。因此正确的产品形式是：**V128 `mma.sync` fallback + guarded ValueSlice + 默认关闭、由部署明确选择的 Phase-6/Phase-1 latency 候选**。下一步先补真实并发/调用分布、端到端 K3 和多卡集成，再决定是否默认启用以及是否投入 Cluster/TMA multicast。

Runtime Profile Agent 将这个产品判断抽象为更一般的原则：**不要把 kernel winner 写死，要把策略绑定到 workload profile 与编译后物理 receipt。** W384/W768 的方向反转实验表明，目标不是单调增加或减少 CTA，而是让 prepare grid 落入合适的 resident-capacity 区间。该证据来自另一条 route，所以当前只作为 shadow policy 和方法论，不改写正式 dispatcher。

如果未来要重新打开 `tcgen05` 路线，下一道 gate 不是再做一次孤立指令替换，而是让 Phases 1/3/4/6 共享转置布局和 TMEM 生命周期，并在完整四阶段边界内同时通过最终 guarded 基线的正确性、净性能、并发和 profiler 判据。在此之前，全面 SM100 MMA 重写不值得；受保护的 SM100 并行度与流水专用化值得继续推进。

---

## 附录 A：证据索引与复现入口

| 内容 | Job | 原始证据 |
|---|---:|---|
| 官方 benchmark | 17926 | [`01_official_benchmark_17926.log`](../../experiments/final_campaign/data/raw/01_official_benchmark_17926.log) |
| K3 fixed/packed slice sweep | 17928 | [`02_k3_shapes_17928.csv`](../../experiments/final_campaign/data/raw/02_k3_shapes_17928.csv) |
| 官方 vs patched V128 parity | 17929 | [`baseline_parity.json`](../../experiments/final_campaign/data/raw/baseline_parity.json) |
| naive/chunk reference correctness | 17934 | [`03_reference_correctness_17934.csv`](../../experiments/final_campaign/data/raw/03_reference_correctness_17934.csv) |
| CHUNK 16/32/64 | 17935 | [`04_chunk_analysis_17935.csv`](../../experiments/final_campaign/data/raw/04_chunk_analysis_17935.csv) |
| tcgen05 Phase-6 probe | 17936/17937 | [`analysis_tcgen05.md`](../../experiments/final_campaign/analysis_tcgen05.md)、[`03_tcgen05_probe_17937.csv`](../../experiments/final_campaign/data/raw/03_tcgen05_probe_17937.csv) |
| packed 单请求 dispatcher 升级 | 17947 | [`05_dispatch_upgrade_17947.csv`](../../experiments/final_campaign/data/raw/05_dispatch_upgrade_17947.csv) |
| H12/T8192 targeted NCU | 17965 | [`05_targeted_ncu_17965.log`](../../experiments/final_campaign/data/raw/05_targeted_ncu_17965.log) |
| 旧批次 Nsys/NCU/SASS 诊断 | 14991 等 | [`BOTTLENECK_ANALYSIS.md`](../../experiments/BOTTLENECK_ANALYSIS.md) |
| Phase-6 P4 干净候选 | 19901 | [`release_19901.log`](../../experiments/mainline_20260905/data/release_19901.log)、[匹配 NCU CSV](../../experiments/mainline_20260905/data/) |
| Phase-6 四种 state 合约 | 19903 | [`state_matrix_19903.log`](../../experiments/mainline_20260905/data/state_matrix_19903.log) |
| Phase-1 干净验收与双流负例 | 19934 | [`clean_19934.log`](../../experiments/mainline_20260905/data/clean_19934.log)、[`memcheck`](../../experiments/mainline_20260905/data/clean_19934_memcheck.log)、[`synccheck`](../../experiments/mainline_20260905/data/clean_19934_synccheck.log) |
| Phase-1 SASS/NCU 机制互证 | 19935 | [`clean_profile_19935.log`](../../experiments/mainline_20260905/data/clean_profile_19935.log)、[四份 NCU CSV](../../experiments/mainline_20260905/data/) |
| Runtime Profile Agent 前瞻容量规则与 CUPTI 归因 | 2026-09-09 | [`实验说明`](../../experiments/runtime_profile_evolution/README.md)、[`资格证书`](../../experiments/runtime_profile_evolution/b300_prospective_capacity_certificate.json)、[`机制证书`](../../experiments/runtime_profile_evolution/b300_prepare_mechanism_certificate.json) |

代码交付：

- [`0001-k2-value-slice-and-dispatch.patch`](../../patches/0001-k2-value-slice-and-dispatch.patch)：ValueSlice 四变体与资源感知 dispatcher；
- [`0002-dispatch-packed-single-sequence.patch`](../../patches/0002-dispatch-packed-single-sequence.patch)：packed 单序列无同步 dispatch；
- [`0003-release-entry-hardening.patch`](../../patches/0003-release-entry-hardening.patch)：设备 guard、实际 beta 基址对齐与构建宏一致性修复；
- [`0004-guarded-v16-prefetch4.patch`](../../patches/0004-guarded-v16-prefetch4.patch)：默认关闭的 Phase-6 `StatePrefetch=4`；
- [`0005-guarded-phase1-lookahead.patch`](../../patches/0005-guarded-phase1-lookahead.patch)：默认关闭、依赖 0004 的 Phase-1 L4/L2；
- [`tcgen05_probe/phase6_probe.cu`](../../experiments/final_campaign/tcgen05_probe/phase6_probe.cu)：SM100 Phase-6 独立 microbench；
- [`validate_fla_references.py`](../../experiments/final_campaign/validate_fla_references.py)：naive/chunk 正确性矩阵；
- [`chunk_analysis.py`](../../experiments/final_campaign/chunk_analysis.py)：CHUNK 数值/Neumann/workspace 模型。

baseline 版本与 ValueSlice 基础工作树 hash 见 [`SOURCE_MANIFEST.md`](../../SOURCE_MANIFEST.md)；9 月 5 日五补丁链、最终源码/二进制 hash、复算脚本和已知限制见 [`mainline_20260905/README.md`](../../experiments/mainline_20260905/README.md)、[`BUILD_MANIFEST.json`](../../experiments/mainline_20260905/data/BUILD_MANIFEST.json) 与 [`SHA256SUMS`](../../experiments/mainline_20260905/SHA256SUMS)。题目原文见 `assignment02-work/team/c1_flashkda/TASK.md`；官方 FlashKDA baseline 为 commit `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`，CUTLASS pin `5c149f5`。

## 附录 B：数字口径自查

- 官方 speedup 均由 Job 17926 同一行 `FLA/Flash` 计算，范围 1.79–3.42×；
- ValueSlice 百分比均按 `(V128−candidate)/V128`，Job 17947 三个 repeat median 再取中位数；
- “慢 106%”指 `(V16/V128−1)`，不是 speedup；
- tcgen05 的 `0.920×` 是 `mma_time/tcgen_time`，小于 1 表示 tcgen05 更慢；
- NCU Job 17965 的 1.27 ms/901.22 µs 只在相同 profiler 口径内比较，不与 0.7807/0.5698 ms 的 CUDA Event 绝对值混用；
- 表中 tensor pipe 使用 elapsed-cycle 口径 2.48%/3.50%；active-cycle 口径 30.98%/5.43% 仅作分母差异说明，未被表述为整卡 Tensor Core 利用率；
- “98/98 bitwise”指 ValueSlice comparison rows；“200/200 finite”指正确性 CSV 全部 rows（98 条 ValueSlice、92 条 Flash-vs-reference、10 条 reference-vs-reference），不表示 200 个独立模型样本；
- Phase-6 P4 的百分比以旧 V16 / `StatePrefetch=1` 为基线；Phase-1 的 5.15%–12.40% 以 Phase-6-only P4 为基线；37.23%/45.52% 以 Job 19934 同作业强制 V128 为基线，三组数字不能相加；
- Job 19934 的 40 个性能形状包含 34 个准入域和 6 个回退对照；“四种口径全部获益”只指 34 个准入域的 median，不表示测试了 guard 区间的每个整数；
- 37.23% 对应无 initial、有 final 的首段；45.52% 对应有 initial、有 final 的续段，不能互相替代；
- 双 stream 的 1.147440→1.164832 ms 是两个完整请求的 joined-pair 时间，回归 1.52%，不除以二、不包装为单请求 latency；
- Job 19935 的 245,760 是 local spilling requests，不是 bytes、DRAM misses 或可直接分摊的耗时；四份 NCU 各为一次 16-pass replay，不是 16 个独立样本；
- Job 19934 只在开头采样到一次 1095 MHz；不同 job 的绝对毫秒和不同工具的 duration 不横向拼接，也不按频率比例校正；
- 0004/0005 均为编译期默认关闭；`N=1` 不代表 GPU 上只有一个活跃请求，运行时取消构建 flag 不能关闭已编译路径；
- 27%、37% 和 46% 都是指定形状/状态契约下的 FlashKDA forward operator 降时，不是 TTFT、TPOT、SLO goodput 或多卡收益实测。
- Runtime Profile Agent 的 1.0133×–1.1354× 来自另一条 H12 BT16 CAKE-generated route；四项比较以八个独立进程 epoch 为统计单位并使用 Bonferroni 单侧 98.75% 下界，不能与 ValueSlice/P4/Phase-1 百分比合并。
