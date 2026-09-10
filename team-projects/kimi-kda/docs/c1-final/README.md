# C1 最终提交入口

> 本目录保留 2026-09-03 至 2026-09-09 的阶段性交付。2026-09-10 冻结的最终 PPT、论文、报告、Agent 代码和证据统一见 [`../../c1-sm100-delivery/README.md`](../../c1-sm100-delivery/README.md)。本页以下内容作为过程记录保留。

## 题目

**FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得。**

最终标题：

**《FlashKDA 官方 Kernel 从 SM80 MMA 迁移到 SM100 是否值得？——面向 Kimi K3 的 B300/SM103 复现、量化分析与并行度重构挑战》**

任务按题目要求分为三阶段：

1. 复现与测量：在 B300 上复现官方 benchmark，由 Agent 把 SASS、grid、NCU 和 `tcgen05` probe 组成同 workload 的 measurement receipt；
2. 分析：Agent 先从 receipt 回答“整体迁移是否值得”并列出证据，再把 MMA 工作排序为 `KEEP / MEASURE / STOP`；然后逐项回答 CHUNK、recurrence、compute/memory、BF16 state 和发布决策；
3. 挑战：验证 Agent 根据瓶颈选出的两条优先路线——ValueSlice CTA 间并行度与 CTA 内 Phase-6/Phase-1 流水；以题目指定参考实现和逐位回归验证正确性，并与分层基线比较性能。

## 一句话结论

**Kernel is cheap，可信的 profile-to-policy 决策才昂贵：不值得把 FlashKDA 整体机械改写为 `tcgen05`；应由 Runtime Profile Agent 根据 workload、物理资源、正确性与统计证据选择候选，并用上一轮实测结论自优化下一轮 proposal。**

七条决定性证据：

- 官方 FlashKDA 在 B300 上已经是强基线，相对 FLA `chunk_kda` 为 **1.79–3.42×**；
- K2 静态 SASS 中有 **3,640 条 `HMMA.16816.F32.BF16`**，`TCGEN/UTCMMA=0`；
- TP8 代表形状 `T=8192,H=12,D=128` 下，官方 K2 只有 **12 CTA 对 148 SM**；ValueSlice V16 扩到 96 CTA，fixed 与 packed 单序列的 CUDA Event 延迟分别降低约 **27.0%/26.9%**；
- 对最自然的 Phase-6 `[128,16]@[16,128]`，`tcgen05+TMEM` 在 L0、inner=64 的乐观摊销口径下仍只有 `mma.sync` 的 **0.920×**，即候选慢约 **8.7%**；
- Phase-6 `StatePrefetch=4` 相对旧 V16 / `StatePrefetch=1`，在已测 eager 场景中无初态新增约 **7.4%–9.1%**、有初态新增约 **16.8%–19.6%**；
- Phase-1 按初态选择 L4/L2 后，Job 19934 的 34 个准入域形状在四种中位计时口径上全部获益；`T8192,H12` 同作业相对 V128 累计降低 **37.23%（首段无初态）/45.52%（有初态续段）**，但无初态双 stream joined-pair **回归 1.52%**。
- Runtime Profile Agent 在独立 H12 BT16 route 上，禁止候选筛选地前瞻预测 W384/W768 的 cpc7/cpc13；四个 profile 为 **1.0133×–1.1354×**，最弱单侧 98.75% 下界 **1.0129×**，CUPTI 将 **97.98%–101.16%** 的 full-span 节省归到 prepare。

因此最终产品决策是：保留 V128 `mma.sync` fallback；在已标定域使用 guarded ValueSlice；0004/0005 两级预取保持 build-time 默认关闭，由调用方明确选择低并发 latency 模式。当前 guard 不感知设备上的其它 stream/request，真实并发矩阵、完整 Kimi K3 和多卡集成优先于继续扩大启用域；CTA Cluster + TMA multicast 保留为其后的候选。

Profile 工具当前只生成 resource/evidence-bound shadow recommendation，不把 cpc、V16 或任何单点 winner 写死。补充 capacity 结果来自另一条物理 route，不能与 ValueSlice/P4/Phase-1 收益合并。

Agent 的自优化边界也必须说清：`conclusions-only memory → 下一轮 typed proposal → verifier/screen/qualification → incumbent 或停止` 已实现并由 CPU 测试覆盖；它不是生产 dispatcher 的在线自改，也没有用 Agent 投票替代硬件证据。

Agent 在这个项目中的起点不是挑战：可重放的
[`c1_b300_h12_mma_assessment.json`](../../experiments/runtime_profile_evolution/c1_b300_h12_mma_assessment.json)
已将主线测量压成 7 条物理 finding，新增 tile 几何与 compiled residency，并在看到挑战结果前输出“不全面换 `tcgen05`，先优化当前 `mma.sync` 的并行分解与 issue overlap”。挑战是对这个判断的验证，而不是 Agent 首次出现的地方。

## 最终交付物

| 交付物 | 入口 | 用途 |
|---|---|---|
| 最终报告 | [`FINAL_REPORT.md`](FINAL_REPORT.md) | 完整三阶段论证、六个讨论点、挑战结果和系统边界 |
| 10 页答辩 | [`FlashKDA_SM100_academic_defense_20260909_v2.pptx`](FlashKDA_SM100_academic_defense_20260909_v2.pptx) | 学术风格；复现与测量→Agent 先给结论与证据→挑战验证→回写下一轮 |
| 逐页讲稿 | [`DEFENSE_SCRIPT.md`](DEFENSE_SCRIPT.md) | 每页时间预算、口播重点和转场 |
| 追问准备 | [`Q_AND_A.md`](Q_AND_A.md) | 15 分钟提问环节的高概率问题与边界回答 |
| 核心补丁 1 | [`0001-k2-value-slice-and-dispatch.patch`](../../patches/0001-k2-value-slice-and-dispatch.patch) | V16/V32/V64/V128 K2 ValueSlice 与资源感知 dispatcher |
| 核心补丁 2 | [`0002-dispatch-packed-single-sequence.patch`](../../patches/0002-dispatch-packed-single-sequence.patch) | packed 单序列无 host sync 地复用 fixed B1 标定策略 |
| 入口硬化补丁 3 | [`0003-release-entry-hardening.patch`](../../patches/0003-release-entry-hardening.patch) | 同设备检查、CUDAGuard、实际 beta 基址对齐与构建宏一致性 |
| Phase-6 候选补丁 4 | [`0004-guarded-v16-prefetch4.patch`](../../patches/0004-guarded-v16-prefetch4.patch) | 默认关闭的 V16 `StatePrefetch=4` |
| Phase-1 候选补丁 5 | [`0005-guarded-phase1-lookahead.patch`](../../patches/0005-guarded-phase1-lookahead.patch) | 默认关闭、依赖 0004 的初态感知 L4/L2 |
| ValueSlice 源码快照 | [`implementation/current/`](../../experiments/final_campaign/implementation/current/) | 0001/0002 后的 Python dispatch 层核对快照；9 月 5 日最终身份以 manifest 为准 |
| `tcgen05` 挑战探针 | [`tcgen05_probe/`](../../experiments/final_campaign/tcgen05_probe/) | Phase-6 真实 `UTCHMMA`/TMEM microbench、SASS 和结果 |
| 实验脚本 | [`experiments/final_campaign/`](../../experiments/final_campaign/) 中的 `run_*.sbatch`、`*.py` | B300 上的自包含复现入口 |
| 原始证据 | [`data/raw/`](../../experiments/final_campaign/data/raw/) | 带 Slurm job ID 的日志、逐行 CSV 和 JSON |
| 9 月 5 日增量归档 | [`mainline_20260905/`](../../experiments/mainline_20260905/) | 五补丁顺序、Jobs 19901/19903/19934/19935、复算脚本、hash 和已知限制 |
| NCU 证据清单 | [`artifacts/ncu/`](../../experiments/final_campaign/artifacts/ncu/) | Job 17965 的公开说明与 SHA-256；`.ncu-rep` 仅本地归档 |
| 汇总数据与图 | [`data/summary_metrics.csv`](../../experiments/final_campaign/data/summary_metrics.csv)、[`figures/`](../../experiments/final_campaign/figures/) | 从原始数据确定性生成的答辩图表 |
| 交付审计 | [`SUBMISSION_AUDIT.md`](SUBMISSION_AUDIT.md) | 补丁可应用性、脚本语法、数字和措辞审计 |
| Runtime Profile Agent | [`../../tools/runtime-profile-agent/`](../../tools/runtime-profile-agent/) | workload/物理 receipt、typed proposal、测量、置信门槛与 fallback 的闭环工具 |
| C1 MMA 主线判断 | [`../../experiments/runtime_profile_evolution/c1_b300_h12_mma_assessment.json`](../../experiments/runtime_profile_evolution/c1_b300_h12_mma_assessment.json) | SASS/NCU/tcgen05 receipt 到迁移结论和候选优先级的确定性重放 |
| Profile 前瞻证据 | [`../../experiments/runtime_profile_evolution/`](../../experiments/runtime_profile_evolution/) | W384/W768 容量规则资格证书和 CUPTI prepare/chain 机制证书 |

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
| 9 月 5 日 clean benchmark | 三轮随机变体顺序；每轮分别记录 eager/Graph/cache perturbation/host wall；只做同 job 配对 |

普通性能数字来自 CUDA Event；Graph replay、cache-perturbed CUDA Event 和同步 host wall 分列，不混成一个样本池。NCU duration 只在同 job、相同 profiler 配置的配对路径之间比较，不能与 CUDA Event 的绝对延迟混算。Job 19934 只在开头记录一次 1095 MHz 时钟采样，早期作业和 Job 19935 频率不同；因此不同 job 的绝对毫秒不横向拼接，也不按频率比例校正。所有 Slurm 时间均不含排队。

## 五补丁链与默认关闭的构建开关

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
git apply --check /path/to/0003-release-entry-hardening.patch
git apply /path/to/0003-release-entry-hardening.patch
git apply --check /path/to/0004-guarded-v16-prefetch4.patch
git apply /path/to/0004-guarded-v16-prefetch4.patch
git apply --check /path/to/0005-guarded-phase1-lookahead.patch
git apply /path/to/0005-guarded-phase1-lookahead.patch
git diff --check
```

不能只应用 `0001`：这样虽然有 ValueSlice 和第一版 dispatcher，但 packed 单序列仍会被当成未建模 varlen 回退到 V128，无法复现 Job 17947 的 auto policy。`0003` 是公共入口 hardening；`0004`/`0005` 是 build-time 默认关闭的性能候选，且 `0005` 依赖 `0004`。五补丁顺序、最终源码/二进制身份和已知 SKIP 见 [`mainline_20260905/README.md`](../../experiments/mainline_20260905/README.md) 与 [`BUILD_MANIFEST.json`](../../experiments/mainline_20260905/data/BUILD_MANIFEST.json)。

补丁 SHA-256：

```text
d80377cf156b52e2b8fb64f72e2129bbad05d30c466cd641b10ed8971f798667  0001-k2-value-slice-and-dispatch.patch
f2165d0cc1b4e99a241e10c624d81dfd5d682fb6a7e7fdc92e3aff0989c18172  0002-dispatch-packed-single-sequence.patch
0f888bf678f0111b5ad394bd4beda4641d472dbca9fc5e02c2e7988fc5bbaefe  0003-release-entry-hardening.patch
246a6aa347a1779215d5cdb72d84f2be18357ea0b77be8df987c8c46118b9a96  0004-guarded-v16-prefetch4.patch
fae72eccda8eea94d5609fd30df75f9855dc9c9c00231300b2af00f89da910d1  0005-guarded-phase1-lookahead.patch
```

在 B300 上构建最终扩展：

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export FLASH_KDA_CUDA_ARCHS=103a
export FLASH_KDA_ENABLE_V16_PREFETCH4=1
export FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH=1
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

两项预取 flag 都不设置时保持旧预取路径；只设置 Phase-6 flag 可构建 P4-only；Phase-1 flag 未同时启用 Phase-6 时，构建应拒绝。它们是编译期开关，运行时 unset 不能改变已生成二进制。运行时 `FLASH_KDA_K2_VALUE_SLICE=128` 可回退 V128 / 两级 prefetch=1，但若要恢复旧 V16 或 Phase-6-only V16，需要使用对应独立二进制或重新构建。

运行任何 patched 实验前，应让新扩展排在 `PYTHONPATH` 首位，并打印 `flash_kda_C.__file__` 与构建身份；这避免 Python 意外加载旧 `.so`。官方 baseline、Phase-6-only 和 Phase-1 候选应使用独立源/构建目录，不覆盖现有安装。

## 推荐复现实验顺序

下表是 9 月 3 日首轮 campaign。每个脚本都请求一张 GPU、上限 15 分钟。脚本中的服务器路径是本次集群环境的留档；迁移环境时需替换 `repo`、`campaign`、`integrated`、`python_bin` 和输出目录。

| 顺序 | 阶段与脚本 | 已完成 Job | 实际 GPU wall time | 最终证据 |
|---:|---|---:|---:|---|
| 1 | 官方干净基线：[`run_01_official_benchmark.sbatch`](../../experiments/final_campaign/run_01_official_benchmark.sbatch) | 17926 | 93 s | [`01_official_benchmark_17926.log`](../../experiments/final_campaign/data/raw/01_official_benchmark_17926.log) |
| 2 | 官方扩展与 patched V128 等价性：[`run_02b_baseline_parity.sbatch`](../../experiments/final_campaign/run_02b_baseline_parity.sbatch) | 17929 | 6 s | [`baseline_parity.json`](../../experiments/final_campaign/data/raw/baseline_parity.json) |
| 3 | 题目指定 naive/chunk 参考对拍：[`run_03_reference_correctness.sbatch`](../../experiments/final_campaign/run_03_reference_correctness.sbatch) | 17934 | 33 s | [`03_reference_correctness_17934.csv`](../../experiments/final_campaign/data/raw/03_reference_correctness_17934.csv) |
| 4 | CHUNK 16/32/64 数值、计算和 workspace：[`run_04_chunk_analysis.sbatch`](../../experiments/final_campaign/run_04_chunk_analysis.sbatch) | 17935 | 41 s | [`04_chunk_analysis_17935.csv`](../../experiments/final_campaign/data/raw/04_chunk_analysis_17935.csv) |
| 5 | Phase-6 `tcgen05`：[`tcgen05_probe/run_03_tcgen05_probe.sbatch`](../../experiments/final_campaign/tcgen05_probe/run_03_tcgen05_probe.sbatch) | 17937 | 16 s | [`03_tcgen05_probe_17937.csv`](../../experiments/final_campaign/data/raw/03_tcgen05_probe_17937.csv) |
| 6 | 最终 ValueSlice/dispatcher sweep：[`run_05_dispatch_upgrade.sbatch`](../../experiments/final_campaign/run_05_dispatch_upgrade.sbatch) | 17947 | 10 s | [`05_dispatch_upgrade_17947.csv`](../../experiments/final_campaign/data/raw/05_dispatch_upgrade_17947.csv) |
| 7 | H12/T8192 targeted NCU：[`run_05_targeted_ncu.sbatch`](../../experiments/final_campaign/run_05_targeted_ncu.sbatch) | 17965 | 15 s | [`05_targeted_ncu_summary_17965.csv`](../../experiments/final_campaign/data/raw/05_targeted_ncu_summary_17965.csv) |

以上七个首轮 job 的实测 GPU wall time 合计约 **214 s（3 分 34 秒）**，因此都能独立放进 15 分钟权限窗口。一次全新 CUDA 扩展编译和集群排队不计入这个总数，应在 GPU 申请前完成或单独留窗口。214 s **不包含** 9 月 5 日增量，也不应与其跨作业绝对时间合并。

历史 Job 17928（12 s）记录了只应用 `0001` 时的初始 fixed/packed sweep，用于展示旧 dispatcher 错过 packed 单序列；最终政策已经由 Job 17947/`0002` 取代。若要逐字复现 17928，应在应用 `0002` 之前运行 [`run_02_k3_shapes.sbatch`](../../experiments/final_campaign/run_02_k3_shapes.sbatch)；正常验证最终提交时直接运行 Job 17947 对应脚本即可。

9 月 5 日增量按以下顺序复核；这些作业的 headline 只与各自同 job 基线比较：

| 顺序 | 阶段 | Job | 权威证据 |
|---:|---|---:|---|
| 1 | Phase-6 P4 干净 wrapper/NCU | 19901 | [`release_19901.log`](../../experiments/mainline_20260905/data/release_19901.log)、[NCU CSV](../../experiments/mainline_20260905/data/) |
| 2 | Phase-6 四 state 合约与尾长 | 19903 | [`state_matrix_19903.log`](../../experiments/mainline_20260905/data/state_matrix_19903.log) |
| 3 | Phase-1 干净验收、四口径和双流负例 | 19934 | [`clean_19934.log`](../../experiments/mainline_20260905/data/clean_19934.log)、[`memcheck`](../../experiments/mainline_20260905/data/clean_19934_memcheck.log)、[`synccheck`](../../experiments/mainline_20260905/data/clean_19934_synccheck.log) |
| 4 | Phase-1 matched SASS/NCU | 19935 | [`clean_profile_19935.log`](../../experiments/mainline_20260905/data/clean_profile_19935.log)、[四份 CSV](../../experiments/mainline_20260905/data/) |

Job 19934 的归档复算命令（从 `docs/c1-final/` 执行）：

```bash
python3 ../../experiments/mainline_20260905/scripts/summarize_clean.py \
  ../../experiments/mainline_20260905/data/clean_19934.log --format markdown
python3 -m unittest discover \
  -s ../../experiments/mainline_20260905/scripts -p 'test_summarize_clean.py'
```

## 证据索引

| 要回答的问题 | 结论 | 权威证据 |
|---|---|---|
| 官方实现是否仍是 SM80 MMA？ | 是；静态 SASS 为 3,640 条 HMMA，TCGEN/UTCMMA 为 0 | [`FINAL_REPORT.md §2.3`](FINAL_REPORT.md#23-sass题目所述-sm80-mma-路径成立)、[`sass_opcode_summary.csv`](../../experiments/data/sass_opcode_summary.csv) |
| 官方 baseline 是否复现？ | 是；六个 H96/H64 case 相对 FLA 为 1.79–3.42× | [`01_official_benchmark_17926.log`](../../experiments/final_campaign/data/raw/01_official_benchmark_17926.log) |
| CHUNK 32/64 能否机械放大？ | 不能；当前指数路径均在 token 18 首次 FTZ/overflow，朴素 Neumann 每序列代价为 5.33×/26.67× | [`04_chunk_analysis_17935.csv`](../../experiments/final_campaign/data/raw/04_chunk_analysis_17935.csv) |
| `tcgen05` 是否值得直接替换？ | 当前不值得；K3 V128 Phase-6 L0/inner64 仅 0.920× | [`analysis_tcgen05.md`](../../experiments/final_campaign/analysis_tcgen05.md)、[`03_tcgen05_probe_17937.csv`](../../experiments/final_campaign/data/raw/03_tcgen05_probe_17937.csv)、[`03_tcgen05_probe_17937.sass`](../../experiments/final_campaign/data/raw/03_tcgen05_probe_17937.sass) |
| 当前瓶颈是什么？ | 不是峰值计算或 HBM 饱和；是 12 CTA underfill + recurrence/issue latency | [`05_targeted_ncu_summary_17965.csv`](../../experiments/final_campaign/data/raw/05_targeted_ncu_summary_17965.csv)、[`05_targeted_ncu_metrics_17965.csv`](../../experiments/final_campaign/data/raw/05_targeted_ncu_metrics_17965.csv) |
| ValueSlice 是否有效？ | fixed/packed 单序列约 −27.0%/−26.9%；高自然并行度有反例 | [`05_dispatch_upgrade_17947.csv`](../../experiments/final_campaign/data/raw/05_dispatch_upgrade_17947.csv) |
| Phase-6 P4 是否有效？ | 相对旧 V16 / `StatePrefetch=1`：无初态约 7%–9%，有初态约 17%–20%；默认关闭 | [`release_19901.log`](../../experiments/mainline_20260905/data/release_19901.log)、[`state_matrix_19903.log`](../../experiments/mainline_20260905/data/state_matrix_19903.log) |
| Phase-1 L4/L2 是否有效？ | 34 个准入域形状四种中位口径均获益；但无初态双流回归 1.52% | [`clean_19934.log`](../../experiments/mainline_20260905/data/clean_19934.log)、[`mainline_20260905/README.md`](../../experiments/mainline_20260905/README.md) |
| 新收益的机制是什么？ | grid 不变、occupancy 近似不变；issue/eligible 改善、short-scoreboard 降低；无初态 L4 有真实 spill | [`release_19901_baseline_ncu.csv`](../../experiments/mainline_20260905/data/release_19901_baseline_ncu.csv)、[`clean_profile_19935.log`](../../experiments/mainline_20260905/data/clean_profile_19935.log) |
| 正确性是否保持？ | 200/200 comparison row 全部 finite；其中 98/98 ValueSlice 对 V128 bitwise equal | [`03_reference_correctness_17934.csv`](../../experiments/final_campaign/data/raw/03_reference_correctness_17934.csv)、[`03_reference_correctness_17934.log`](../../experiments/final_campaign/data/raw/03_reference_correctness_17934.log) |
| 是否发布 SM100 专版？ | 保留 guarded hybrid；0004/0005 仅作默认关闭 latency 候选，不发布全面 `tcgen05` 分叉 | [`FINAL_REPORT.md §3.6`](FINAL_REPORT.md#36-讨论点六假如我们是作者v2-出不出-sm100a-专版) |

正确性口径必须原样保留：**200/200 是 finite，不是 200/200 统一硬阈值通过；98/98 才是 ValueSlice 相对 V128 的 bitwise equal。** 长序列与 K3 参考关系的观测最坏 relative RMSE 为 0.9131%，但这些行没有统一预注册 hard threshold。

9 月 5 日 clean 验收是另一层回归证据：120 条主比较 + 14 条尾块/状态补比较逐位通过；另有 80 条 Graph 跨路径、80 条计时后跨路径和 4 条双 stream 正确性比较。它们验证新候选相对既有 V128 路径不改结果，不是新增 naive/FP32 oracle。每项 sanitizer 各覆盖 20 条定向比较并报告 `ERROR SUMMARY: 0 errors`，不是全矩阵 racecheck。

Job 17965 两份 NCU 原始报告已在本地归档；二进制 `.ncu-rep` 会嵌入主机、账户和 GPU 元数据，因此不提交到公开仓库。公开仓库保留完整 log、long/summary CSV、公式级 metric 名称以及 [`artifacts/ncu/SHA256SUMS`](../../experiments/final_campaign/artifacts/ncu/SHA256SUMS)：

```text
82731991d300d7419f6f8f69d7efecbbe299a111c364ae75938ff2af1b2ada50  05_official_v128_h12_t8192_17965.ncu-rep
9a2555a0b9aa159c4e7026b773cc99c90d7c06a3927f73c0dcb46239e9d05c0e  05_valueslice_v16_h12_t8192_17965.ncu-rep
```

## 数据解释的硬边界

1. **27%、37.23% 和 45.52% 都是指定形状/state 合约下的 FlashKDA forward operator 降时，不是 Kimi K3 的 TTFT、TPOT 或 SLO goodput。** 37.23% 对应首段无初态，45.52% 对应有初态续段；不能互换或相加。
2. **H12 是 TP8 的 per-GPU 计算形状，不是单卡完整 K3。** 单 B300 没有实测 TP8 NCCL、scheduler、continuous batching 或完整 checkpoint。
3. **本挑战只覆盖 forward/prefill。** `T=1` 回退 V128，纯 decode 还有独立 fused KDA decode 路径，不能声称 TPOT 已改善。
4. **`tcgen05` 结果是 Phase-6 隔离 probe，不是完整 K2。** 它否决“保持现有数据流只换指令”，但不能否决未来跨 Phase 1/3/4/6 的 TMEM-resident 数据流重写。
5. **CHUNK32/64 的安全实现只做了 FLA 小形状探针。** 当前数据否决机械改常量，不否决加入 rescale/block solve 的新算法。
6. **ValueSlice 的最佳 V 依赖请求分布；`N=1` 不是设备级低并发检测。** packed 单序列已由 `0002` 捕获，`nseq>1` 回退 V128；但两个独立 `N=1` 调用可在不同 stream 并发，当前 guard 看不到这一点。
7. **NCU 百分比必须写明分母。** 主表使用 elapsed-cycle tensor pipe 2.48%/3.50%；active-cycle 30.98%/5.43% 只描述活跃 SM 的活跃周期，不能当成整卡 Tensor Core 利用率。
8. **BF16 state 结论是 kernel 数值对拍。** public state buffer 改为 FP32 并未改变内部 BF16 舍入点，不能据此断言“完整 FP32 recurrence 没有价值”，也没有模型级 perplexity/任务精度证据。
9. **9 月 5 日百分比使用分层基线。** Phase-6 P4 相对旧 V16，Phase-1 相对 Phase-6-only P4，37.23%/45.52% 相对 Job 19934 同作业强制 V128；不同 job 的绝对毫秒、CUDA Event、host wall 和 NCU duration 不混算。
10. **40 个性能形状不是 40 个独立生产测试或 guard 全覆盖。** 其中 34 个在准入域、6 个是回退对照；有限采样没有遍历 `2048..8192` 的每个整数或真实请求分布。
11. **无初态双 stream 是必须保留的负例。** joined-pair `1.147440→1.164832 ms`，回归 1.52%；这阻止 Phase-1 候选成为默认吞吐路径。
12. **新预取路径默认关闭且资格验证不完整。** 0004/0005 是 build-time opt-in；multi-GPU、单独 alias build、实际安装/回滚和完整模型/服务仍未完成。无初态 L4 还有 8 B stack 和真实 local spill，不能声称“全局无 spill”。

## 最终 go/no-go

| 路线 | 决策 | 重新开启或发布的门槛 |
|---|---|---|
| 全面 `mma.sync → tcgen05` | **NO-GO** | 跨多个 K2 phase 共用布局/TMEM 生命周期，并在完整 K2 上击败最终 guarded 基线，同时通过正确性、并发和 profiler gate |
| CHUNK 32/64 机械放大 | **NO-GO** | 必须先加入 rescale/block solve，并证明总序列计算和 workspace 的净收益 |
| guarded ValueSlice | **GO** | 只在已标定 B300/SM103 shape 域使用；部署侧仍需控制并发，保留 V128 fallback |
| Phase-6 P4 | **OPT-IN RC** | 显式 `sm_103a` 构建、默认关闭；按初态分开解释收益，继续保留旧 V16 二进制 |
| Phase-1 L4/L2 | **LATENCY RC / THROUGHPUT NO-GO** | 默认关闭；真实并发矩阵消除 1.52% 双流负例后，才考虑吞吐默认 |
| Cluster + TMA multicast | **DEFERRED NEXT** | 先完成真实调用分布、端到端/多卡与交付集成，再测 2/4/8-CTA residency、同步、bytes 和完整 K2 duration |
