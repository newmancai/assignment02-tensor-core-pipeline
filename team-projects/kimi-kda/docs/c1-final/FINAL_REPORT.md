# FlashKDA 官方 Kernel 从 SM80 MMA 迁移到 SM100 是否值得？

## ——面向 Kimi K3 的 B300/SM103 复现、量化分析与并行度重构挑战

> C1 最终报告
> 实验日期：2026-09-03
> 实验对象：NVIDIA B300 SXM6 AC（compute capability 10.3）

## 摘要

本报告回答一个单一问题：**MoonshotAI FlashKDA 的 Tensor Core 计算仍使用 SM80 世代的 `mma.sync`，是否值得为 B300/SM100 家族重写为 SM100 执行路径？**

我们的结论是：**不值得把 FlashKDA 整体机械改写为 `tcgen05`；值得发布一条受保护的 B300/SM103 专用路径，但当前最有价值的专用化是 K2 recurrence 的并行度重构，而不是全面替换 MMA 指令。**

证据分三阶段建立。第一，在 B300 上复现官方 benchmark，FlashKDA 相对 FLA `chunk_kda` 为 **1.79–3.42×**；SASS 中有 **3,640 条静态 `HMMA.16816.F32.BF16`**，而 `TCGEN/UTCMMA=0`，确认题目所述计算路径。第二，针对六个讨论点完成纸面量化和实验：CHUNK 从 16 机械放大到 32/64 时，当前指数恢复路径都在第 18 个 token 首次出现 FTZ/overflow，朴素 Neumann 每序列代价分别增至 **5.33×/26.67×**；在最适合新指令的 Phase-6 `[128,16]@[16,128]` 上，`tcgen05+TMEM` 即使将固定成本摊薄 64 次，仍只有 `mma.sync` 的 **0.920×**。第三，我们挑战了真正的实测瓶颈：TP8 代表形状每卡只有 12 个 head，官方 K2 只有 **12 CTA 对 148 SM**。ValueSlice 将 grid 扩成 96 CTA，在 `T=8192,H=12,D=128` 上把 fixed prefill forward 从 **0.7807 ms 降至 0.5698 ms（−27.0%）**；最新 dispatcher 也能在不读取 `cu_seqlens` 数值、不引入 GPU 到 CPU 同步的情况下，为 packed 单序列选择 V16，将 **0.7850 ms 降至 0.5740 ms（−26.9%）**。正确性 CSV 的 200 条比较全部 finite；其中 98 条 ValueSlice 对照全部 bitwise equal，所有独立参考关系的观测最坏 relative RMSE 为 0.9131%。

这些数据支持一个 hybrid 发布决策：保留 V128 `mma.sync` 作为兼容 fallback，只在低并发长 prefill 的已标定域启用 ValueSlice；下一步用 CTA Cluster/TMA multicast 消除 slice 间公共输入的重复搬运。该结论只覆盖 B300 单卡 FlashKDA forward。我们没有将 27% 算子降时夸大为 Kimi K3 的 27% TTFT、TPOT 或 SLO goodput 改善。

**关键词：** FlashKDA；Kimi Delta Attention；B300；SM103；`mma.sync`；`tcgen05`；TMEM；ValueSlice；prefill

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

### 1.3 SM103 新能力分别是什么，与题目有什么关系

| 能力 | 含义 | 与 FlashKDA C1 的关系 |
|---|---|---|
| `tcgen05 + TMEM` | Blackwell 数据中心 Tensor Core 的异步 MMA 路径；累加器放在独立 Tensor Memory，而非普通寄存器，需要 alloc、descriptor、commit/wait 和读回协议 | 是“SM80 MMA 迁移 SM100”的直接候选，但必须把协议与布局成本一起计时；本项目在 Phase-6 实测 |
| TMA | Tensor Memory Accelerator，按张量描述符异步搬运 global/shared 数据，减少普通线程参与地址计算和搬运 | 官方 FlashKDA 已使用，因此不能把“改用 TMA”算作本项目的新迁移收益 |
| CTA Cluster / DSM / multicast | 多 CTA 组成 cluster；可访问 distributed shared memory，并让一次 TMA 搬运 multicast 给多个 CTA | 可针对 ValueSlice 的公共输入重复请求，但 cluster residency、同步和真实 bytes 尚需实测 |
| 更大 shared memory | SM80/A100 上限约 164 KiB/SM；本卡实测 233,472 B，即约 228 KiB/SM | 可容纳更深 staging 或更多协作数据，但不会自动修复只有 12 CTA 的整卡 underfill |
| FP8/FP6/FP4 | 更低精度 Tensor Core 与 block scaling 能提高适配 GEMM 的峰值和降低流量 | 更适合先评估 KDA 前后投影；recurrent state/局部逆若降精度，必须重新做数值与模型精度验证 |

这张表的作用是建立候选机制，不是用规格表代替实验。TMA 与 cluster 始于 SM90；`tcgen05`/TMEM 才是本题最直接的 Blackwell 计算路径。规格与指令约束参考 NVIDIA [CUDA Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) 和 [CUTLASS tcgen05 API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html)。

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

---

## 3. 阶段二：六个讨论点——结论与证据

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
| ValueSlice | Value 行独立，CTA `12→96`，无 reduction/atomic | slice-independent 输入被重复读取；短序列/高并发会变慢 | 本项目挑战路线 |

[纸面模型] `T=4096`、单 sequence-head 下，V16 的 source-request 为 29.188 MiB，其中多个 slice 重复请求 common inputs。若 8-CTA cluster 可以理想 multicast 一次这些输入，可节省 23.734 MiB（81.3%），把请求量降到接近 V128 的 5.453 MiB。这个数是源请求模型，**不是实测 HBM bytes，也没有计入 cluster occupancy/同步成本**。

### 3.4 讨论点四：这是 compute-bound 还是 memory-bound？

**结论：两者都不是传统意义上的主瓶颈。**

[实测] Job 17965 中官方 H12 的 SM/DRAM throughput 仅 2.64%/1.24%，却因 grid 只有 12 CTA 最多覆盖 8.1% 的 SM；V16 将 grid 扩到 96 CTA 后，NCU duration 从 1.27 ms 降至 901.22 µs，但 SM/DRAM throughput 仍只有 7.22%/1.83%。结合 Nsys 中减时集中在 K2，可将当前边界描述为：

```text
grid underfill + chunk recurrence critical path + CTA 内 TMA/issue latency
```

不能仅凭低 occupancy 或高 No Eligible 下因果结论，也不能用 active-SM 的 tensor pipe 指标代替整卡利用率。可靠判断来自 duration、grid、SM/DRAM throughput、L2、active warps、scheduler stall 和 timeline 的交叉。

这也解释了为什么“B300 BF16 峰值更高”没有自动转化为 K2 低延迟：当 136 个左右的 SM 无 CTA 可执行时，提高每条 Tensor Core 指令的峰值并未解决首要限制。

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
2. 对 B300/SM103、低并发长 prefill 的已标定域启用 ValueSlice；
3. packed 单序列按 fixed B1 进入已标定策略，多序列 varlen 在没有分布模型时仍回退 V128；
4. 下一原型用 CTA Cluster + TMA multicast 共享 ValueSlice 的 common inputs；
5. `tcgen05` 暂不并入 K2，只在跨多个 phase 的转置/TMEM-resident 数据流通过新 gate 后重新考虑。

专版的负担包括 CUDA 13/`sm_103a` 构建链、TMEM 和 cluster debug、更大二进制、多个 layout 的 CI 以及与通用版本长期同步。因此专版必须 opt-in/guarded，不能替换已经很强的可移植基线。

下表是本项目的最终路线决策：

| 路线 | 当前证据 | 决策 |
|---|---|---|
| 全面 `mma.sync→tcgen05` | K3 V128 Phase-6 的 L0 摊销口径仅 0.920× | **停止直接替换，不集成** |
| CHUNK 32/64 机械放大 | token 18 FTZ/overflow；Neumann 5.33×/26.67× | **停止，除非改算法** |
| ValueSlice recurrence 重构 | 单请求长 prefill 降时约 27%，bitwise equal；短序列有明确反例 | **受保护发布** |
| Cluster + TMA multicast | 纸面上可去除大部分 common-input 重复请求 | **下一项 B300 挑战** |

---

## 4. 阶段三：挑战——ValueSlice 并行度重构

### 4.1 为什么这仍然是“迁移 SM100 是否值得”的挑战

题目允许挑战“只换指令、大 CHUNK+rescale、并行度重构”之一。前两条在进入完整实现前已被定量 gate 否决：Phase-6 direct swap 的乐观 microbench 为负，机械大 CHUNK 首先数值失效。本项目因此选择**并行度重构**，直接针对 B300/SM103 上观察到的 12 CTA/148 SM，而不是为使用新指令而使用新指令。

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

### 4.5 为什么没有继续把 `tcgen05` 塞进完整 K2

挑战阶段不要求正向加速，关键是数据驱动的 stop/go。Phase-6 是整个 K2 中最自然的 `m128nVk16` 候选；如果在 L0、inner=64 的乐观口径下，正式 V128 仍慢 8.7%，完整集成还要承担 U 布局转换、TMEM 生命周期、barrier、跨 phase 兼容和回归测试，就没有足够的正向信号占用剩余 B300 槽位。我们因此把工程时间投入能直接击中 NCU 瓶颈、并已通过正确性和性能 gate 的 ValueSlice。

---

## 5. 从算子到 Kimi K3：prefill、并发、SLO、通信与环境边界

### 5.1 能严格声称什么

[实测 + 系统映射] FlashKDA forward 是 KDA prefill 的直接组成部分。TP8 单请求时每卡 H12，官方 K2 只有 12 CTA；ValueSlice 因而最可能改善**低并发、长 prompt 的 KDA prefill latency**，进而改善 TTFT 的一个组成部分。Job 17947 的 packed 1×8192 是比 fixed tensor 更贴近 serving 调用约定的证据：升级后的 dispatcher 确实捕获了这个单请求机会。

[实测] 当 total tokens 同为 8192、sequence 数从 1 增到 8/32 时，负载自身提供更多并行度，ValueSlice 收益消失甚至反转。因此“并发更高”不是自动获得更高 ValueSlice speedup；实际是低并发单请求最受益，高并发应回退或选更粗 slice。

### 5.2 不能直接声称什么

- **不能说 TTFT 降低 27%。** 27% 是 FlashKDA forward operator 在 H12/T8192 单请求上的降时；KDA 层还包含投影、norm、gate 等计算，模型还有非 KDA 层。
- **不能说 TPOT/decode 已加速。** 当前策略对 `T=1` 回退 V128；实际 serving 的纯 decode 还有独立 fused KDA decode 路径。
- **不能说 SLO goodput 已提升。** 本项目没有运行完整 Kimi checkpoint、scheduler、continuous batching、排队和 SLO sweep。
- **不能说 TP8 通信已改善。** 单 B300 只能模拟 per-GPU H12 计算形状，没有实测 NCCL all-reduce/all-to-all；局部 K2 降时不会自动减少跨卡通信。

### 5.3 Amdahl 敏感性，而非端到端结果

[系统推断] 令 `p` 为 FlashKDA forward 在完整 prefill wall time 中的占比，实测 operator 降时 `r=0.27`，忽略重叠变化时：

```text
prefill 降时约为 p·r
理想容量加速约为 1 / (1 - p·r)
```

| 假设 p | 预计 prefill 降时 | 理想容量加速 |
|---:|---:|---:|
| 20% | 5.4% | 1.057× |
| 40% | 10.8% | 1.121× |
| 60% | 16.2% | 1.193× |

这只是敏感性分析。SLO goodput 是否提升还取决于 TTFT/TPOT 哪个约束绑定、请求长度分布、batching、排队非线性以及新路径对并发资源驻留的影响。

### 5.4 通信视角：卡内与跨卡必须分开

**卡内通信。** ValueSlice 的收益来自跨 CTA 并行，成本恰好是 common inputs 被多次 TMA 请求。SM100 家族的 CTA Cluster、DSM 和 TMA multicast 对应一个具体问题：多个 slice CTA 能否共享一次 common-input 搬运。它们不是背景功能列表，而是本挑战的下一步数据复用手段。需用 2/4/8-CTA cluster microbench 实测 cluster residency、multicast bytes、同步和 duration，才可把 81.3% 纸面 request reduction 写成性能收益。

**跨卡通信。** TP8 会带来模型并行通信，但 FlashKDA 内部的 ValueSlice 没有跨 GPU 通信，也不改变 collective 的消息体。因此需要分别测 compute-only 与 NCCL-overlapped timeline，才能判断 0.21 ms 级单算子节省会被通信隐藏、暴露还是放大。

### 5.5 环境与运营视角

B300 更大的 shared memory、L2、显存、带宽和新低精度能力，能支持更深 pipeline、更大 batch、长上下文和 KDA 前后投影 GEMM；但 Job 17965 证明，12 CTA 的 K2 不会仅因硬件峰值更高而自动变快。FP8/FP6/FP4 更适合首先评估投影 GEMM；在没有模型精度证据时，不应直接用于 recurrent state 和 16×16 inverse。

运营报告的合理作用是提供 prompt length、并发分布、TTFT/TPOT SLO 权重和流量波峰，从而决定 dispatcher 标定域与收益加权；它不能代替 SASS/NCU/microbench 回答 MMA 迁移本身。当前没有 Kimi K3+B300 官方线上 trace，因此本报告没有用第三方“运营提升”数字填补端到端证据空缺。

### 5.6 15 分钟 B300 权限如何影响方法，而非降低结论标准

每个 Slurm job 自包含环境、commit/hash、GPU 信息、warmup、计时和 CSV 落盘；编译、正确性、性能和 profiler 分层，关键路径采用小 metric set，避免 NCU replay 超时。负结果触发预先设定的 stop gate，不继续扩大实现。已完成的关键 job 均远低于 15 分钟：官方 benchmark 93 s，ValueSlice sweep 12 s，参考正确性 33 s，CHUNK 分析 41 s，tcgen05 probe 16 s，最终 NCU 15 s。资源限制影响的是“无法跑多卡/完整 K3 serving”，不影响单卡 instruction、kernel 和 dispatcher 因果链的可复核性。

---

## 6. 威胁有效性与限制

1. 只有一张 B300，未实测 TP8/NCCL 或端到端 Kimi K3；H12 只是 per-GPU 计算形状。
2. `tcgen05` 是真实 Phase-6 隔离 probe，不是完整 K2；L0 偏乐观，L1 的 scalar 重排偏保守。
3. CHUNK32/64 的安全路径只在 FLA 小形状上验证，没有完成 FlashKDA 大 CHUNK 重写。
4. ValueSlice 的最优 V 依赖 sequence distribution；当前生产可辩护的 dispatcher 只新增 packed 单序列，multi-sequence varlen 仍回退。
5. BF16 state 结论是 kernel 数值对拍，不是模型级 accuracy/perplexity。
6. 没有 TTFT、TPOT、并发吞吐或 SLO goodput 实测；Amdahl 表只用于说明端到端收益边界。
7. NCU、Nsys、CUDA Event 的 instrumentation 不同，绝对时间只在各自一致口径内 A/B；报告没有跨工具混算 speedup。
8. 单卡、单 GPU 样本不能给出跨机器置信区间；tcgen05 在同卡背靠背 job 中复现，ValueSlice 则有旧批次三次独立复跑支持方向一致。

---

## 7. 最终结论

对 C1 的直接回答是：

> **SM100 值得利用，但“利用 SM100”不等于“把 SM80 MMA 全换掉”。**

官方 FlashKDA 在 B300 上仍使用 `mma.sync` 不是一个仅靠“代际更老”就能判错的决定。CHUNK16 与现有算法/数值/MMA 形状高度耦合；正式 K3 V128 的 Phase-6 `tcgen05+TMEM` 在最乐观摊销 probe 中仍慢 8.7%，当前没有全面 instruction rewrite 的投资依据。

相反，B300 上最明确的瓶颈是 TP8 单请求 H12 导致的 12 CTA/148 SM underfill。ValueSlice 保持总 Tensor FLOP 和每个 Value 行的运算顺序不变，把 grid 扩到 96 CTA，fixed 与 packed 单请求均获得约 27% operator 降时。多短序列反例又证明它不能无条件启用，因此正确的产品形式是：**V128 `mma.sync` fallback + 已标定低并发长 prefill 的 guarded ValueSlice + 后续 Cluster/TMA multicast 数据复用**。

如果未来要重新打开 `tcgen05` 路线，下一道 gate 不是再做一次孤立指令替换，而是让 Phases 1/3/4/6 共享转置布局和 TMEM 生命周期，并在完整四阶段边界内同时通过 V128 正确性、净性能和 occupancy/profiler 三项判据。在此之前，全面 SM100 MMA 重写不值得；受保护的 SM100 并行度专用化值得。

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

代码交付：

- [`0001-k2-value-slice-and-dispatch.patch`](../../patches/0001-k2-value-slice-and-dispatch.patch)：ValueSlice 四变体与资源感知 dispatcher；
- [`0002-dispatch-packed-single-sequence.patch`](../../patches/0002-dispatch-packed-single-sequence.patch)：packed 单序列无同步 dispatch；
- [`tcgen05_probe/phase6_probe.cu`](../../experiments/final_campaign/tcgen05_probe/phase6_probe.cu)：SM100 Phase-6 独立 microbench；
- [`validate_fla_references.py`](../../experiments/final_campaign/validate_fla_references.py)：naive/chunk 正确性矩阵；
- [`chunk_analysis.py`](../../experiments/final_campaign/chunk_analysis.py)：CHUNK 数值/Neumann/workspace 模型。

baseline 版本与 ValueSlice 基础工作树 hash 见 [`SOURCE_MANIFEST.md`](../../SOURCE_MANIFEST.md)；packed 单序列升级以 0002 的自包含 diff 和 Job 17947 日志复核。题目原文见 `assignment02-work/team/c1_flashkda/TASK.md`；官方 FlashKDA baseline 为 commit `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`，CUTLASS pin `5c149f5`。

## 附录 B：数字口径自查

- 官方 speedup 均由 Job 17926 同一行 `FLA/Flash` 计算，范围 1.79–3.42×；
- ValueSlice 百分比均按 `(V128−candidate)/V128`，Job 17947 三个 repeat median 再取中位数；
- “慢 106%”指 `(V16/V128−1)`，不是 speedup；
- tcgen05 的 `0.920×` 是 `mma_time/tcgen_time`，小于 1 表示 tcgen05 更慢；
- NCU Job 17965 的 1.27 ms/901.22 µs 只在相同 profiler 口径内比较，不与 0.7807/0.5698 ms 的 CUDA Event 绝对值混用；
- 表中 tensor pipe 使用 elapsed-cycle 口径 2.48%/3.50%；active-cycle 口径 30.98%/5.43% 仅作分母差异说明，未被表述为整卡 Tensor Core 利用率；
- “98/98 bitwise”指 ValueSlice comparison rows；“200/200 finite”指正确性 CSV 全部 rows（98 条 ValueSlice、92 条 Flash-vs-reference、10 条 reference-vs-reference），不表示 200 个独立模型样本；
- 27% 是 FlashKDA forward operator 降时，不是 TTFT、TPOT 或 SLO goodput 实测。
