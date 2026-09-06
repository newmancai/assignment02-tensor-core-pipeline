# C1 交付审计（正式交付 2026-09-06）

## 2026-09-05 主线增量封包复核

9 月 3 日的中心判断没有被推翻：全面 `mma.sync→tcgen05` 仍是 NO-GO，guarded ValueSlice 仍是正确方向。9 月 5 日新增的是 ValueSlice 之后的两级 CTA 内调度：Phase-6 `StatePrefetch=4`，以及按 HasStateIn 选择 L4/L2 的 Phase-1 lookahead。它们提高了最终基线，也暴露了无初态双 stream 回归，因此交付状态必须写成**默认关闭的单请求 latency 候选**，不能写成普遍 serving 吞吐优化。

本轮已把增量结果纳入 [`FINAL_REPORT.md`](FINAL_REPORT.md)、本目录 [`README.md`](README.md) 和本审计；9 月 5 日证据归档在 [`experiments/mainline_20260905/`](../../experiments/mainline_20260905/)，代码以 0001→0002→0003→0004→0005 的顺序交付。新版 [`FlashKDA_SM100_decision_defense_20260906.pptx`](FlashKDA_SM100_decision_defense_20260906.pptx)、[`DEFENSE_SCRIPT.md`](DEFENSE_SCRIPT.md) 与 [`Q_AND_A.md`](Q_AND_A.md) 已同步三层优化、并发负例和默认关闭边界。

### 增量结论与数据口径

| 结论 | 权威口径 | 审计判断 |
|---|---|---|
| Phase-6 P4 | Job 19901 相对旧 V16 / `StatePrefetch=1`；T8192 initial+final eager `0.569712→0.459184 ms`，降低 19.40%；Job 19903 无初态已测 eager 约 7.4%–9.1% | 通过；续段与首段必须分开，不把 19.40%当首次 prefill headline |
| Phase-1 L4/L2 | Job 19934 相对 Phase-6-only P4；34 个准入域形状四种中位计时均获益，eager 5.15%–12.40% | 通过；不能与 Phase-6 百分比相加 |
| 最终候选 vs V128 | 同 Job 19934、同输入强制 V128；T8192 首段无初态 37.23%，有初态续段 45.52% | 通过；是两种 state 合约，不是 TTFT/模型收益 |
| 并发负例 | 两 stream、两个完整 T8192 请求 joined-pair；无初态 `1.147440→1.164832 ms`，回归 1.52%，三轮同向 | 必须展示；阻止 Phase-1 成为默认吞吐路径 |
| Phase-1 正确性 | 120 主比较 + 14 补比较逐位通过；80 Graph 跨路径、80 计时后跨路径、4 双流比较通过 | 通过；是相对既有实现的回归，不是新高精度 oracle |
| Sanitizer | memcheck/synccheck 各 20 条定向比较，退出 0、`ERROR SUMMARY: 0 errors` | 通过但有边界；不是全矩阵 racecheck |
| SASS/NCU | 数学 HMMA 骨架保持；issue/eligible 改善、short-scoreboard 降低；无初态 L4 有 8 B stack 与 245,760 local spill requests | 通过；不能写“全局无 spill”“减少动态指令”或把 requests 当 bytes |

### 2026-09-05 封包检查

- 五补丁链必须按顺序应用；0003 是入口 hardening，0004/0005 是 build-time 默认关闭的性能候选，0005 依赖 0004。
- 0004/0005 要求显式 `FLASH_KDA_CUDA_ARCHS=103a`；启用最终候选还需同时设置 `FLASH_KDA_ENABLE_V16_PREFETCH4=1` 与 `FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH=1`。
- `N=1` 只表示一次调用内一个 sequence，不表示 GPU 上只有一个请求；当前 guard 没有跨 stream 并发感知。
- Job 19934 只在开头记录一次 1095 MHz 时钟采样，Job 19935 与更早 profile 的频率不同。只使用同 job 配对比例，不拼跨作业绝对毫秒，也不按频率比例校正。
- 40 个性能形状由 34 个准入域与 6 个回退点构成，不等于遍历 `2048..8192` 的每个整数。回退点四口径中位漂移约 ±0.9%，但单轮最坏回归 1.32%。
- multi-GPU、单独 alias-extension GPU 检查、实际包安装/回滚、完整 Kimi K3、TP8/NCCL 和真实 serving 并发矩阵仍未完成；这些是明确披露的资格边界。
- Cluster/TMA multicast 仍是候选，不是已实现结果；新增优先级应先处理真实请求分布、并发取舍、可观察性和端到端集成。

本地封包自检已执行：五个补丁的 SHA-256 与 manifest 一致；`summarize_clean.py` 对归档 Job 19934 返回 `PASS`；30 个汇总器反例单元测试全部通过；新版 PPTX 已完成 10 页全量渲染以及包完整性、字体和布局检查，未发现告警。补丁 hash 为：

```text
d80377cf156b52e2b8fb64f72e2129bbad05d30c466cd641b10ed8971f798667  0001-k2-value-slice-and-dispatch.patch
f2165d0cc1b4e99a241e10c624d81dfd5d682fb6a7e7fdc92e3aff0989c18172  0002-dispatch-packed-single-sequence.patch
0f888bf678f0111b5ad394bd4beda4641d472dbca9fc5e02c2e7988fc5bbaefe  0003-release-entry-hardening.patch
246a6aa347a1779215d5cdb72d84f2be18357ea0b77be8df987c8c46118b9a96  0004-guarded-v16-prefetch4.patch
fae72eccda8eea94d5609fd30df75f9855dc9c9c00231300b2af00f89da910d1  0005-guarded-phase1-lookahead.patch
```

下文保留 9 月 3 日“发现问题—关闭问题”的历史审计轨迹；其中旧的 P0/P1 状态不代表 9 月 5 日当前结论。

## 2026-09-03 最终封包复核（历史）

下文保留的是生成最终交付物**之前**的审计快照；其中列出的提交阻塞项现已全部关闭：

- 已生成 [`FINAL_REPORT.md`](FINAL_REPORT.md)、[`FlashKDA_SM100_decision_defense.pptx`](FlashKDA_SM100_decision_defense.pptx) 和本目录 [`README.md`](README.md)，旧文件只作为过程记录；
- 最终复现入口已明确依次应用 `0001`、`0002`，并纳入 Job 17947 的 packed 单序列策略；
- 正确性统一写为“200/200 finite，其中 98/98 ValueSlice bitwise equal”，不再声称统一阈值通过；
- Job 17965 的两份 `.ncu-rep` 已本地归档，本地 SHA-256 与 B300 远端逐字节一致；公开仓库在 [`artifacts/ncu/`](../../experiments/final_campaign/artifacts/ncu/) 保留说明与校验和，不提交会嵌入环境元数据的二进制报告；
- NCU 主结论采用 elapsed-cycle tensor 口径 2.48%/3.50%，active-cycle 口径只用于解释分母差异；
- 答辩 PPT 已逐页渲染检查，10/10 页含 `[Sources]` speaker-notes block，越界检测通过。

因此，当前最终提交入口是 [`README.md`](README.md)；下文的 P0/P1 项仅用于保留“发现问题—关闭问题”的审计轨迹，不再表示当前阻塞。

## 2026-09-03 审计结论（历史）

**核心技术证据已经形成闭环，代码补丁本身可交付；但当前目录还不能原样作为最终提交。** 六个讨论点都有对应的分析或实验，其中 `tcgen05` 负结果、ValueSlice 正/负反例、题目指定参考实现对拍都很有说服力。提交前的主要风险不在 kernel，而在入口文档仍指向旧结论、`0002` 没有进入公开复现步骤，以及报告中有两处会让评审误读数据口径的表述。

本次只做只读检查并新增本审计文件，没有修改旧报告、补丁或他人正在生成的交付物。

## 2026-09-03 已执行的检查（历史）

- 在题目提供的 FlashKDA 快照上初始化临时 Git 仓库；`0001` 的 `git apply --check` 通过，实际应用 `0001` 后 `0002` 的 `git apply --check` 也通过。
- 两个补丁依次应用后 `git diff --check` 通过；所得 `flash_kda/__init__.py` 与 `flash_kda/dispatch.py` 和 `implementation/current/` 中的快照逐字节一致。
- `final_campaign/*.py` 全部通过 Python 语法编译检查；所有 Slurm 脚本通过 `bash -n`。
- 重算 `02_k3_shapes_17928.csv` 和 `05_dispatch_upgrade_17947.csv` 的 repeat-median 汇总；报告中的 `27.0%`、`26.9%`、`17.4%`、`1.3%` 和 `V16` 在 `32x256` 上慢约 `106%` 均与原始 CSV 一致。
- 检查正确性 CSV：共 200 行，全部 finite；其中 98 行 ValueSlice 对 V128 全部 bitwise equal，最大 relative RMSE 为 0；对 FLA chunk 的观测最坏 relative RMSE 为 `0.9131%`，对 naive 为 `0.8240%`。
- 检查 CHUNK CSV、`tcgen05` 两次结果、旧 NCU/SASS/Nsys 证据和新 `H12,T8192` targeted NCU 汇总；关键数字均能回到原始文件。

## 2026-09-03 六个讨论点的证据覆盖（历史）

| 讨论点 | 状态 | 主要证据 | 审计判断 |
|---|---|---|---|
| 1. CHUNK=16；32/64 谁先破 | 通过 | `chunk_analysis.py`、Job 17935 CSV/log | 数值范围、朴素 Neumann 代价、workspace 和 FLA safe-block 小探针都有量化。必须保留“朴素扩展/小形状探针”的限定。 |
| 2. `tcgen05` tile 与 CHUNK=16；只换指令是否有收益 | 通过 | `tcgen05_probe/`、Jobs 17936/17937、SASS | Phase-6 `m128nVk16` 合法，K3 `V=128` 的 L0/L1 均给出负结果；边界写得清楚，没有冒充完整 K2。 |
| 3. recurrence 还能从哪里找并行度 | 通过 | ValueSlice 补丁、候选/反例表、Job 17928 | 已实现 ValueSlice；多 head/CTA、persistent、2-CTA/cluster 的反例齐全。Cluster/multicast 仍是纸面下一步，不能写成已实现。 |
| 4. compute-bound 还是 memory-bound | 通过 | 旧 T4096 NCU/Nsys/SASS；Job 17965 T8192 targeted NCU | 低 SM/HBM 利用率、12→96 CTA 和 scheduler 指标支持“underfill + recurrence/issue-latency”，不是传统 compute/HBM 饱和。 |
| 5. BF16 state 精度 | 有边界地通过 | Job 17934，题目指定 `naive.py`/`chunk.py` hash | 长序列、long-memory、ragged 和 state carry 均有数据，最坏观测误差 <1%；但不是模型级精度，也不是完整 FP32-internal recurrence 对照。 |
| 6. 是否发布 sm100a 专版 | 通过 | `report_outline.md` 的 stop/go 表及上述全部证据 | “不做全面 tcgen05；发布 guarded ValueSlice；下一步 cluster/multicast”的决策证据充分。最终提交仍需把它落入正式报告和答辩稿。 |

## 2026-09-03 阻塞项 / P0（已关闭）

### P0-1：正式报告和正式 10 分钟答辩稿尚未成为唯一入口

`final_campaign/report_outline.md` 是完整大纲，但还不是最终报告或演示文稿。仓库根入口仍把读者导向旧的 `docs/report-draft.md` 和 `docs/defense-outline.md`。旧答辩稿仍写“`tcgen05` 最小 M=64、M16 利用率至多 25%”作为主要论证，并把 M128 microbench 写成下一步；新证据已经证明 Phase-6 可合法映射 `m128nVk16`，并且 microbench 已完成。若原样提交，评审很容易先读到旧结论。

**提交门槛：** 产出最终报告和严格 10 分钟的最终答辩稿；根 README 只链接最终版本，旧文档明确标成历史草稿或移出提交入口。最终版本必须以 Job 17937/17947/17965 为准。

### P0-2：公开复现步骤没有应用 `0002`

根 README 和 `experiments/README.md` 都只执行：

```bash
git apply 0001-k2-value-slice-and-dispatch.patch
```

因此按当前文档复现出来的代码仍会把 packed single sequence 当作未标定 varlen 并回退 V128，无法复现 Job 17947 的 auto policy。`SOURCE_MANIFEST.md` 中的 Python 文件 hash 也对应 `0001` 后、`0002` 前的版本。

**提交门槛：** 明确要求依次应用 `0001`、`0002`，更新最终源码 hash/manifest，并把 Job 17947 纳入结果索引。若不想提交两段补丁，则应合并为一段最终补丁，但不能让文档和实际 policy 分叉。

## 2026-09-03 P1（已关闭）

### P1-1：`report_outline.md` 对 packed single sequence 的描述已经过期

性能表仍写“现 dispatcher 因 varlen 保守回退，是可修复的 missed opportunity”，限制章节也写“当前 varlen dispatcher 保守回退”。这描述的是 Job 17928/`0001`，不是 Job 17947/`0002`。Job 17947 已证明 packed `1x8192` 的 auto 选择 V16，`0.7850 ms → 0.5740 ms`，降低 `26.87%`；ragged6、8x1024、32x256 仍安全回退 V128。

建议改成：**“`0002` 已关闭单 packed sequence 的漏选；N>1 的未建模 varlen 继续回退。”** 性能结论可继续使用 17928 的强制 sweep，但 dispatcher 结论要引用 17947。

### P1-2：“200/200 通过阈值”不是准确的预注册口径

Job 17934 的 200 行确实全部 finite，且 `within_limit=True`；但脚本只对 `<=512` token smoke case 设置 `2%` hard relative-RMSE threshold，对长序列/K3 行的 `hard_rel_rmse_limit` 是空值，此时 `within_limit` 只表示 finite。ValueSlice 的 98 行另有严格 0 阈值。

因此正式报告不要写“200/200 均通过误差阈值”。准确说法是：

- `200/200` 比较全部 finite；
- `98/98` ValueSlice 对 V128 bitwise equal；
- 所有设置 hard limit 的 smoke/bitwise 行均通过；
- 长序列与 K3 形状的**观测**最坏 relative RMSE 为 `0.9131%`，但没有为这些行预注册 hard threshold。

### P1-3：缺少 final campaign 的总 README 和一次性证据索引

目前只有 `data/raw/README.md` 和单项 `README_TARGETED_NCU.md`，没有说明执行顺序、基线/补丁树、产物到 job 的映射。编号还存在两组 `03`（reference、tcgen05）和两组 `05`（dispatch、targeted NCU），单看文件名无法判断先后。

建议新增总 README，至少包含：环境、commit/submodule、依次应用两个补丁、脚本→job→产物表、每个脚本的预计墙钟、哪些结果是 final、哪些是历史证据，以及硬编码远端路径如何替换。编号不必为了美观重命名已有原始日志，但要解释。

### P1-4：新 targeted NCU 的可归档性还不完整

Job 17965 的 log、long CSV 和 summary CSV 已在 `data/raw/`，足以核对数值；但当前目录没有对应两份 `.ncu-rep`、metric query 文件或 final-campaign checksum manifest，log 也没有记录 FlashKDA/CUTLASS 源码 commit/hash。旧 T4096 NCU 报告有 `artifacts/SHA256SUMS`，新 T8192 证据没有同等级的完整性记录。

若二进制报告因体积不提交，至少归档 query 文本、为 log/CSV/SASS/补丁生成 SHA-256 清单，并在 README 中明确 `.ncu-rep` 的保存位置与不提交原因。正式报告引用新 NCU 时必须保留 metric 全名和单位。

### P1-5：不要把 tensor metric 的 active-cycle 分母当成整卡利用率

Job 17965 的 tensor 指标是 `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`：官方 V128 为 `30.98%`，V16 为 `5.43%`。这里的分母是 active cycles，不是 elapsed/device peak；整卡 SM throughput 仍只有 `2.64%/7.22%`。答辩若放 tensor 指标，必须连同 metric 全名或明确写“活跃周期内”，不能与整卡 `Compute (SM) Throughput` 并列成同口径百分比。

## 2026-09-03 P2（历史建议）

- `report_outline.md` 开头把 `0.920x` 解释为“慢约 8.0%”；严格按时间比是 `1/0.920-1≈8.7%`，后文已经写对。统一为 8.7% 或只写 0.920x。
- `data/raw/` 与 `tcgen05_probe/results/` 各保存一份 Job 17937 CSV/log/SASS，内容 hash 相同；建议在总 README 指定唯一权威位置，避免未来只更新一边。
- `data/raw/README.md` 说 CSV 都由父目录脚本派生，但 correctness、chunk、tcgen05、shape CSV 实际由实验直接写出。可改为“verbatim logs and machine-readable outputs”。
- Job 17947 的 policy test 验证了 decision，benchmark 也验证了 auto timing；ValueSlice 强制路径与 V128 已逐 bit 对拍。若时间允许，可再加一条直接调用 Python wrapper 的 packed-one auto output/state 对 V128 测试，使 policy→kernel 的端到端链更直观。这不是当前正确性的阻塞项。
- 官方 benchmark 目前只有日志，没有结构化 CSV；最终制图若从日志解析，应把解析脚本一并归档，避免手工抄表。

## 2026-09-03 `0002` policy 安全性结论（历史）

对 FlashKDA 的合法 packed 输入约定，**未发现语义或同步风险**：

- `cu_seqlens.numel()` 只读取张量元数据，不发生 GPU→CPU copy 或 device synchronize；
- packed `N=1` 时，K2 的 sequence/head grid 和 recurrence length 与 fixed `B=1,T=T_total` 相同，Job 17947 也实测 auto 选择与 fixed 相同；
- `N!=1` 仍标记为未建模 varlen并回退 V128，没有把 ragged6 的偶然最佳 V64 冒险泛化到未知分布；
- 架构、SM 数、L2、长度、head 数、state dtype 和 guard band 仍由 `dispatch.py` 原有门槛控制。

剩余边界是有意的保守范围，不是 bug：只捕获 packed single sequence；FP32 public state 在 8192 token 上没有 calibration，会安全回退；非法/不规范 `cu_seqlens` 仍由现有 API 契约和底层校验负责。

## 2026-09-03 原始数字一致性摘要（历史）

| 结论 | 重算/核对结果 |
|---|---:|
| fixed `1x8192,H12` V16 相对 V128 | `0.780688 → 0.569728 ms`，降低 `27.02%` |
| packed `1x8192,H12` 强制 V16 | `0.784960 → 0.573984 ms`，降低 `26.88%` |
| packed single auto（`0002`） | `0.784960 → 0.574016 ms`，降低 `26.87%` |
| ragged6 最佳 V64 | `0.340304 → 0.280992 ms`，降低 `17.43%` |
| 8x1024 最佳 V64 | `0.156128 → 0.154048 ms`，降低 `1.33%`，低于 3% guard |
| 32x256 强制 V16 | `0.125376 → 0.258576 ms`，慢 `106.24%` |
| `tcgen05` V128/grid12/L0/inner64 | `mma/tcgen=0.919742x`，候选慢约 `8.73%` |
| CHUNK 32/64 朴素 Neumann 每序列代价 | `5.333x / 26.667x` |
| Job 17965 targeted NCU | grid `12→96`；duration `1.27 ms→901.22 us`；SM `2.64%→7.22%`；DRAM `1.24%→1.83%` |

这些数字之间未发现实质矛盾。需要修正的是文档版本和口径，而不是原始实验结果。
