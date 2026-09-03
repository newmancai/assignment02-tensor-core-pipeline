# C1 最终提交入口

> **本文件是本题唯一的最终提交入口。** 仓库中较早的 `README`、`report-draft.md`、`defense-outline.md` 和 `report_outline.md` 只保留为过程记录；结论、补丁顺序和数据口径一律以本目录的最终文件为准。

## 题目

**FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得。**

最终标题：

**《FlashKDA 官方 Kernel 从 SM80 MMA 迁移到 SM100 是否值得？——面向 Kimi K3 的 B300/SM103 复现、量化分析与并行度重构挑战》**

任务按题目要求分为三阶段：

1. 复现与测量：在 B300 上复现官方 benchmark，并以 SASS/NCU 确认计算路径和瓶颈；
2. 分析：逐项回答 CHUNK、`tcgen05`、recurrence 并行度、compute/memory 边界、BF16 state 和专版发布决策；
3. 挑战：实现 ValueSlice 并行度重构，以题目指定参考实现验证正确性，并与官方 FlashKDA 比较性能。

## 一句话结论

**不值得把 FlashKDA 整体机械改写为 `tcgen05`；值得发布一条受保护的 B300/SM103 专用路径，但当前最有价值的专用化是 K2 recurrence 的 ValueSlice 并行度重构，而不是全面替换 MMA 指令。**

四条决定性证据：

- 官方 FlashKDA 在 B300 上已经是强基线，相对 FLA `chunk_kda` 为 **1.79–3.42×**；
- K2 静态 SASS 中有 **3,640 条 `HMMA.16816.F32.BF16`**，`TCGEN/UTCMMA=0`；
- TP8 代表形状 `T=8192,H=12,D=128` 下，官方 K2 只有 **12 CTA 对 148 SM**；ValueSlice V16 扩到 96 CTA，fixed 与 packed 单序列的 CUDA Event 延迟分别降低约 **27.0%/26.9%**；
- 对最自然的 Phase-6 `[128,16]@[16,128]`，`tcgen05+TMEM` 在 L0、inner=64 的乐观摊销口径下仍只有 `mma.sync` 的 **0.920×**，即候选慢约 **8.7%**。

因此最终产品决策是：保留 V128 `mma.sync` fallback；只在已标定的低并发长 prefill 域启用 ValueSlice；下一步再评估 CTA Cluster + TMA multicast 对公共输入重复搬运的消除效果。

## 最终交付物

| 交付物 | 入口 | 用途 |
|---|---|---|
| 最终报告 | [`FINAL_REPORT.md`](FINAL_REPORT.md) | 完整三阶段论证、六个讨论点、挑战结果和系统边界 |
| 10 页答辩 | [`FlashKDA_SM100_decision_defense.pptx`](FlashKDA_SM100_decision_defense.pptx) | 10 分钟主讲材料 |
| 逐页讲稿 | [`DEFENSE_SCRIPT.md`](DEFENSE_SCRIPT.md) | 每页时间预算、口播重点和转场 |
| 追问准备 | [`Q_AND_A.md`](Q_AND_A.md) | 15 分钟提问环节的高概率问题与边界回答 |
| 核心补丁 1 | [`0001-k2-value-slice-and-dispatch.patch`](../../patches/0001-k2-value-slice-and-dispatch.patch) | V16/V32/V64/V128 K2 ValueSlice 与资源感知 dispatcher |
| 核心补丁 2 | [`0002-dispatch-packed-single-sequence.patch`](../../patches/0002-dispatch-packed-single-sequence.patch) | packed 单序列无 host sync 地复用 fixed B1 标定策略 |
| 最终源码快照 | [`implementation/current/`](../../experiments/final_campaign/implementation/current/) | 两个补丁依次应用后的 Python dispatch 层核对快照 |
| `tcgen05` 挑战探针 | [`tcgen05_probe/`](../../experiments/final_campaign/tcgen05_probe/) | Phase-6 真实 `UTCHMMA`/TMEM microbench、SASS 和结果 |
| 实验脚本 | [`experiments/final_campaign/`](../../experiments/final_campaign/) 中的 `run_*.sbatch`、`*.py` | B300 上的自包含复现入口 |
| 原始证据 | [`data/raw/`](../../experiments/final_campaign/data/raw/) | 带 Slurm job ID 的日志、逐行 CSV 和 JSON |
| NCU 证据清单 | [`artifacts/ncu/`](../../experiments/final_campaign/artifacts/ncu/) | Job 17965 的公开说明与 SHA-256；`.ncu-rep` 仅本地归档 |
| 汇总数据与图 | [`data/summary_metrics.csv`](../../experiments/final_campaign/data/summary_metrics.csv)、[`figures/`](../../experiments/final_campaign/figures/) | 从原始数据确定性生成的答辩图表 |
| 交付审计 | [`SUBMISSION_AUDIT.md`](SUBMISSION_AUDIT.md) | 补丁可应用性、脚本语法、数字和措辞审计 |

## 基线、环境与计时口径

| 项目 | 最终实验环境 |
|---|---|
| GPU | NVIDIA B300 SXM6 AC，compute capability 10.3，148 SM |
| 每 SM shared memory | 233,472 B，约 228 KiB |
| L2 | 132,644,864 B |
| Driver | 580.126.09 |
| CUDA / PyTorch | CUDA API 13.0；PyTorch 2.10.0+cu130 |
| FlashKDA | `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` |
| CUTLASS | pin `5c149f5` |
| FLA | 0.5.2；参考运行设置 `FLA_FLASH_KDA=0` |
| 官方 benchmark | warmup 30，iters 200，repeats 5 |
| 挑战 benchmark | warmup 20，iters 200，repeats 3 或 5；同一扩展内 A/B |

普通性能数字来自 CUDA Event；NCU duration 只在相同 profiler 配置的 V128/V16 之间比较，不能与 CUDA Event 的绝对延迟混算。所有 Slurm 时间均不含排队；下表的实际 wall time 来自日志中的 UTC 起止时间，也不含第一次准备环境的人工等待。

## 两个补丁必须依次应用

从 MoonshotAI/FlashKDA 的固定基线开始：

```bash
git clone --recurse-submodules https://github.com/MoonshotAI/FlashKDA.git
cd FlashKDA
git checkout 1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b
git submodule update --init --recursive

git apply --check /path/to/0001-k2-value-slice-and-dispatch.patch
git apply /path/to/0001-k2-value-slice-and-dispatch.patch
git apply --check /path/to/0002-dispatch-packed-single-sequence.patch
git apply /path/to/0002-dispatch-packed-single-sequence.patch
git diff --check
```

不能只应用 `0001`：这样虽然有 ValueSlice 和第一版 dispatcher，但 packed 单序列仍会被当成未建模 varlen 回退到 V128，无法复现 Job 17947 的最终 auto policy。两个补丁在审计中已按上述顺序通过 `git apply --check` 和 `git diff --check`。

补丁 SHA-256：

```text
d80377cf156b52e2b8fb64f72e2129bbad05d30c466cd641b10ed8971f798667  0001-k2-value-slice-and-dispatch.patch
f2165d0cc1b4e99a241e10c624d81dfd5d682fb6a7e7fdc92e3aff0989c18172  0002-dispatch-packed-single-sequence.patch
```

在 B300 上构建最终扩展：

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export FLASH_KDA_CUDA_ARCHS=103a
export NVCC_THREADS=8
python setup.py build_ext \
  --build-lib build/integrated/lib \
  --build-temp build/integrated/temp \
  --force

PYTHONPATH=build/integrated/lib:. python - <<'PY'
import flash_kda_C
print(flash_kda_C.__file__)
print(flash_kda_C.get_device_characteristics())
PY
```

运行任何 patched 实验前，应让新扩展排在 `PYTHONPATH` 首位，并打印 `flash_kda_C.__file__`；这避免 Python 意外加载旧 `.so`。官方 baseline 则在未打补丁的独立 worktree 中运行。

## 推荐复现实验顺序

每个脚本都请求一张 GPU、上限 15 分钟。脚本中的服务器路径是本次集群环境的留档；迁移环境时只需替换 `repo`、`campaign`、`integrated`、`python_bin` 和输出目录。最小最终复现顺序如下：

| 顺序 | 阶段与脚本 | 已完成 Job | 实际 GPU wall time | 最终证据 |
|---:|---|---:|---:|---|
| 1 | 官方干净基线：[`run_01_official_benchmark.sbatch`](../../experiments/final_campaign/run_01_official_benchmark.sbatch) | 17926 | 93 s | [`01_official_benchmark_17926.log`](../../experiments/final_campaign/data/raw/01_official_benchmark_17926.log) |
| 2 | 官方扩展与 patched V128 等价性：[`run_02b_baseline_parity.sbatch`](../../experiments/final_campaign/run_02b_baseline_parity.sbatch) | 17929 | 6 s | [`baseline_parity.json`](../../experiments/final_campaign/data/raw/baseline_parity.json) |
| 3 | 题目指定 naive/chunk 参考对拍：[`run_03_reference_correctness.sbatch`](../../experiments/final_campaign/run_03_reference_correctness.sbatch) | 17934 | 33 s | [`03_reference_correctness_17934.csv`](../../experiments/final_campaign/data/raw/03_reference_correctness_17934.csv) |
| 4 | CHUNK 16/32/64 数值、计算和 workspace：[`run_04_chunk_analysis.sbatch`](../../experiments/final_campaign/run_04_chunk_analysis.sbatch) | 17935 | 41 s | [`04_chunk_analysis_17935.csv`](../../experiments/final_campaign/data/raw/04_chunk_analysis_17935.csv) |
| 5 | Phase-6 `tcgen05`：[`tcgen05_probe/run_03_tcgen05_probe.sbatch`](../../experiments/final_campaign/tcgen05_probe/run_03_tcgen05_probe.sbatch) | 17937 | 16 s | [`03_tcgen05_probe_17937.csv`](../../experiments/final_campaign/data/raw/03_tcgen05_probe_17937.csv) |
| 6 | 最终 ValueSlice/dispatcher sweep：[`run_05_dispatch_upgrade.sbatch`](../../experiments/final_campaign/run_05_dispatch_upgrade.sbatch) | 17947 | 10 s | [`05_dispatch_upgrade_17947.csv`](../../experiments/final_campaign/data/raw/05_dispatch_upgrade_17947.csv) |
| 7 | H12/T8192 targeted NCU：[`run_05_targeted_ncu.sbatch`](../../experiments/final_campaign/run_05_targeted_ncu.sbatch) | 17965 | 15 s | [`05_targeted_ncu_summary_17965.csv`](../../experiments/final_campaign/data/raw/05_targeted_ncu_summary_17965.csv) |

以上七个最终 job 的实测 GPU wall time 合计约 **214 s（3 分 34 秒）**，因此都能独立放进 15 分钟权限窗口。一次全新 CUDA 扩展编译和集群排队不计入这个总数，应在 GPU 申请前完成或单独留窗口。

历史 Job 17928（12 s）记录了只应用 `0001` 时的初始 fixed/packed sweep，用于展示旧 dispatcher 错过 packed 单序列；最终政策已经由 Job 17947/`0002` 取代。若要逐字复现 17928，应在应用 `0002` 之前运行 [`run_02_k3_shapes.sbatch`](../../experiments/final_campaign/run_02_k3_shapes.sbatch)；正常验证最终提交时直接运行 Job 17947 对应脚本即可。

## 证据索引

| 要回答的问题 | 结论 | 权威证据 |
|---|---|---|
| 官方实现是否仍是 SM80 MMA？ | 是；静态 SASS 为 3,640 条 HMMA，TCGEN/UTCMMA 为 0 | [`FINAL_REPORT.md §2.3`](FINAL_REPORT.md#23-sass题目所述-sm80-mma-路径成立)、[`sass_opcode_summary.csv`](../../experiments/data/sass_opcode_summary.csv) |
| 官方 baseline 是否复现？ | 是；六个 H96/H64 case 相对 FLA 为 1.79–3.42× | [`01_official_benchmark_17926.log`](../../experiments/final_campaign/data/raw/01_official_benchmark_17926.log) |
| CHUNK 32/64 能否机械放大？ | 不能；当前指数路径均在 token 18 首次 FTZ/overflow，朴素 Neumann 每序列代价为 5.33×/26.67× | [`04_chunk_analysis_17935.csv`](../../experiments/final_campaign/data/raw/04_chunk_analysis_17935.csv) |
| `tcgen05` 是否值得直接替换？ | 当前不值得；K3 V128 Phase-6 L0/inner64 仅 0.920× | [`analysis_tcgen05.md`](../../experiments/final_campaign/analysis_tcgen05.md)、[`03_tcgen05_probe_17937.csv`](../../experiments/final_campaign/data/raw/03_tcgen05_probe_17937.csv)、[`03_tcgen05_probe_17937.sass`](../../experiments/final_campaign/data/raw/03_tcgen05_probe_17937.sass) |
| 当前瓶颈是什么？ | 不是峰值计算或 HBM 饱和；是 12 CTA underfill + recurrence/issue latency | [`05_targeted_ncu_summary_17965.csv`](../../experiments/final_campaign/data/raw/05_targeted_ncu_summary_17965.csv)、[`05_targeted_ncu_metrics_17965.csv`](../../experiments/final_campaign/data/raw/05_targeted_ncu_metrics_17965.csv) |
| ValueSlice 是否有效？ | fixed/packed 单序列约 −27.0%/−26.9%；高自然并行度有反例 | [`05_dispatch_upgrade_17947.csv`](../../experiments/final_campaign/data/raw/05_dispatch_upgrade_17947.csv) |
| 正确性是否保持？ | 200/200 comparison row 全部 finite；其中 98/98 ValueSlice 对 V128 bitwise equal | [`03_reference_correctness_17934.csv`](../../experiments/final_campaign/data/raw/03_reference_correctness_17934.csv)、[`03_reference_correctness_17934.log`](../../experiments/final_campaign/data/raw/03_reference_correctness_17934.log) |
| 是否发布 SM100 专版？ | 发布 guarded hybrid，不发布全面 `tcgen05` 分叉 | [`FINAL_REPORT.md §3.6`](FINAL_REPORT.md#36-讨论点六假如我们是作者v2-出不出-sm100a-专版) |

正确性口径必须原样保留：**200/200 是 finite，不是 200/200 统一硬阈值通过；98/98 才是 ValueSlice 相对 V128 的 bitwise equal。** 长序列与 K3 参考关系的观测最坏 relative RMSE 为 0.9131%，但这些行没有统一预注册 hard threshold。

Job 17965 两份 NCU 原始报告已在本地归档；二进制 `.ncu-rep` 会嵌入主机、账户和 GPU 元数据，因此不提交到公开仓库。公开仓库保留完整 log、long/summary CSV、公式级 metric 名称以及 [`artifacts/ncu/SHA256SUMS`](../../experiments/final_campaign/artifacts/ncu/SHA256SUMS)：

```text
82731991d300d7419f6f8f69d7efecbbe299a111c364ae75938ff2af1b2ada50  05_official_v128_h12_t8192_17965.ncu-rep
9a2555a0b9aa159c4e7026b773cc99c90d7c06a3927f73c0dcb46239e9d05c0e  05_valueslice_v16_h12_t8192_17965.ncu-rep
```

## 数据解释的硬边界

1. **27% 是 FlashKDA forward operator 降时，不是 Kimi K3 的 27% TTFT、TPOT 或 SLO goodput。** 若该算子占完整 prefill 的比例为 `p`，一阶 prefill 降时才约为 `0.27p`。
2. **H12 是 TP8 的 per-GPU 计算形状，不是单卡完整 K3。** 单 B300 没有实测 TP8 NCCL、scheduler、continuous batching 或完整 checkpoint。
3. **本挑战只覆盖 forward/prefill。** `T=1` 回退 V128，纯 decode 还有独立 fused KDA decode 路径，不能声称 TPOT 已改善。
4. **`tcgen05` 结果是 Phase-6 隔离 probe，不是完整 K2。** 它否决“保持现有数据流只换指令”，但不能否决未来跨 Phase 1/3/4/6 的 TMEM-resident 数据流重写。
5. **CHUNK32/64 的安全实现只做了 FLA 小形状探针。** 当前数据否决机械改常量，不否决加入 rescale/block solve 的新算法。
6. **ValueSlice 的最佳 V 依赖请求分布。** packed 单序列已由 `0002` 捕获；`nseq>1` 的未建模 varlen 继续回退 V128。
7. **NCU 百分比必须写明分母。** 主表使用 elapsed-cycle tensor pipe 2.48%/3.50%；active-cycle 30.98%/5.43% 只描述活跃 SM 的活跃周期，不能当成整卡 Tensor Core 利用率。
8. **BF16 state 结论是 kernel 数值对拍。** public state buffer 改为 FP32 并未改变内部 BF16 舍入点，不能据此断言“完整 FP32 recurrence 没有价值”，也没有模型级 perplexity/任务精度证据。

## 最终 go/no-go

| 路线 | 决策 | 重新开启或发布的门槛 |
|---|---|---|
| 全面 `mma.sync → tcgen05` | **NO-GO** | 跨多个 K2 phase 共用布局/TMEM 生命周期，并在完整 K2 上同时通过正确性、净性能和 profiler gate |
| CHUNK 32/64 机械放大 | **NO-GO** | 必须先加入 rescale/block solve，并证明总序列计算和 workspace 的净收益 |
| guarded ValueSlice | **GO** | 仅启用已标定的 B300/SM103、低并发长 prefill 域；保留 V128 fallback |
| Cluster + TMA multicast | **NEXT** | 以 2/4/8-CTA cluster 实测 residency、同步、multicast bytes 和端到端 K2 duration |
