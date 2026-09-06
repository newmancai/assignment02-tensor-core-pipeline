# Kimi KDA 大作业：FlashKDA on B300

## 一句话结论

FlashKDA 在 B300 上仍使用 SM80 `mma.sync`，但主要瓶颈并不是“Tensor Core 指令太旧”，而是 K2 recurrence 在 TP8 形状下只有很少的长生命周期 CTA。直接替换为 `tcgen05` 未通过性能 gate；当前更值得保留的是 guarded hybrid 路线：V128 兼容回退，加上默认关闭的 V16/Phase-6 P4/Phase-1 lookahead 低并发候选。

正式交付统一从 [`docs/c1-final/README.md`](docs/c1-final/README.md) 进入。

## 当前完成度

| 交付项 | 状态 | 证据 |
|---|---|---|
| 官方 FlashKDA B300 复现 | 已完成 | 基线 commit `1ce47ea`；NCU/SASS 已确认 K2 主路径与低并行度问题 |
| SM100 路线定量分析 | 已完成 | `tcgen05` direct swap 未过 gate；保留 V128 fallback + guarded SM103 候选 |
| ValueSlice 与后续优化 | 已完成候选验证 | `0001`–`0005`：ValueSlice、dispatcher hardening、Phase-6 P4、Phase-1 lookahead |
| 资源感知 dispatcher | 已完成固定形状 B300 候选版 | 越界退回 V128；P4/Phase-1 为 `sm_103a` 编译期显式 opt-in |
| 正确性 | 已完成已测路径验收 | 200/200 finite、98/98 ValueSlice bitwise；干净候选 120+14 cross-path bitwise；memcheck/synccheck 0 error |
| 性能 | 已完成至 2026-09-05 的 B300 主线验证 | 同 Job19934 相对 V128：首段降低 37.23%，有初态续段降低 45.52%；无初态双流回归 1.52% |
| Trace/profile 瓶颈图 | 已完成 | 真实 Nsys/NCU 数据、多面板 PNG/SVG 与原始报告均已归档 |
| 报告 | 已完成正式版 | `docs/c1-final/FINAL_REPORT.md` |
| 答辩 | 已完成新版 10 页材料 | `docs/c1-final/FlashKDA_SM100_decision_defense_20260906.pptx`、讲稿与 Q&A |

## 为什么不直接把 `mma.sync` 换成 `tcgen05`

K2 的核心小矩阵以 `M=16` 为主，正好贴合 `mma.sync.m16n8k16`。SM100 BF16 `tcgen05` 的有效 tile 更大，机械替换会引入低利用率、TMEM 分配和异步同步开销。NCU 中官方 TP8/H12 K2 的 SM throughput 约 2.5%、achieved occupancy 约 9.4%，说明先增加独立工作比更换 Tensor Core 指令更有价值。

后续工作的优先级是：

1. 先补并发 1/2/4/8、真实调用分布、完整 K3/TP8/NCCL 与端到端 TTFT/SLO gate；
2. 再评估 CTA Cluster + TMA multicast，共享被 ValueSlice 重复读取的 slice-independent inputs；
3. 仅在 `M=128` 的 state-update 阶段继续评估 `tcgen05`，而不是整条 kernel 机械替换。

## 目录

- `docs/c1-final/`：正式报告、新版 10 页答辩、逐页讲稿、Q&A 与提交审计。
- `patches/0001-*.patch`–`0005-*.patch`：从 ValueSlice 到 guarded Phase-1 候选的有序补丁链。
- `experiments/mainline_20260905/`：最新主线证据、构建身份、NCU、sanitizer、双流负例和 SHA-256 清单。
- `docs/report-draft.md`、`docs/defense-outline.md`：历史过程材料，不作为最终口径。
- `experiments/README.md`：复现实验流程与口径。
- `experiments/BOTTLENECK_ANALYSIS.md`：Nsys + NCU 联合瓶颈分析、指标解释与限制。
- `experiments/figures/kimi_kda_b300_bottleneck.png`：可直接放入报告/群文档的浓缩实测图；同目录含 SVG。
- `experiments/artifacts/`：公开仓库保存紧凑 SASS 样例；含服务器元数据的 Nsight 原始报告只在本机同目录归档，不公开上传。
- `experiments/integrated_validation_20260901.log`：最新 B300 独立复跑日志。
- `SOURCE_MANIFEST.md`：本机/服务器一致性与关键文件校验和。

## 应用补丁

```bash
git clone --recurse-submodules https://github.com/MoonshotAI/FlashKDA.git
cd FlashKDA
git checkout 1ce47ea
git apply /path/to/0001-k2-value-slice-and-dispatch.patch
git apply /path/to/0002-dispatch-packed-single-sequence.patch
git apply /path/to/0003-release-entry-hardening.patch
git apply /path/to/0004-guarded-v16-prefetch4.patch
git apply /path/to/0005-guarded-phase1-lookahead.patch
```

构建与验证依赖 B300、CUDA 13.0、Python 3.12 和仓库的 CUTLASS 子模块。补丁依赖、编译期开关与复现命令见 [`experiments/mainline_20260905/README.md`](experiments/mainline_20260905/README.md)。

## 结果口径

这里的百分比是 FlashKDA forward operator 或 state-carrying KDA trace 的延迟降低，不是完整 Kimi serving 的 tokens/s、TTFT 或端到端吞吐提升。报告和答辩必须保持这个边界。
