# Kimi KDA 大作业：FlashKDA on B300

## 一句话结论

FlashKDA 在 B300 上仍使用 SM80 `mma.sync`，但主要瓶颈并不是“Tensor Core 指令太旧”，而是 K2 recurrence 在 TP8 形状下只有很少的长生命周期 CTA，导致 SM 利用率极低。我们的主线优化沿 Value 维切分 K2 state，每个 CTA 负责一个 `ValueSlice × D` 状态片段，在不改变总 MMA FLOP 的前提下增加并行 CTA 数。

## 当前完成度

| 交付项 | 状态 | 证据 |
|---|---|---|
| 官方 FlashKDA B300 复现 | 已完成 | 基线 commit `1ce47ea`；NCU/SASS 已确认 K2 主路径与低并行度问题 |
| SM100 路线定量分析 | 已完成第一版 | `docs/report-draft.md` 与资源模型 |
| ValueSlice 内核 | 已完成 | V16/V32/V64/V128 四个变体；补丁见 `patches/` |
| 资源感知 dispatcher | 已完成固定形状 B300 候选版 | 3% + 5 µs guard band；越界退回 V128 |
| 正确性 | 已完成 | fixed BF16/FP32、ragged、state in/out、CUDA Graph；与 V128 bitwise equal |
| 性能 | 已完成并于 2026-09-01 复跑 | forward 高价值区间降低 9.37%–26.10%；stateful trace 降低 5.68% |
| 报告 | 已有可编辑初稿 | `docs/report-draft.md` |
| 答辩 | 已有讲述骨架 | `docs/defense-outline.md` |

## 为什么不直接把 `mma.sync` 换成 `tcgen05`

K2 的核心小矩阵以 `M=16` 为主，正好贴合 `mma.sync.m16n8k16`。SM100 BF16 `tcgen05` 的有效 tile 更大，机械替换会引入低利用率、TMEM 分配和异步同步开销。NCU 中官方 TP8/H12 K2 的 SM throughput 约 2.5%、achieved occupancy 约 9.4%，说明先增加独立工作比更换 Tensor Core 指令更有价值。

后续真正值得做的 SM100 专版是：

1. Value-parallel CTA decomposition；
2. CTA Cluster + TMA multicast，共享被 ValueSlice 重复读取的 K-only workspace；
3. 仅在 `M=128` 的 state-update 阶段评估 `tcgen05`，而不是整条 kernel 机械替换。

## 目录

- `patches/0001-k2-value-slice-and-dispatch.patch`：基于 MoonshotAI/FlashKDA `1ce47ea` 的完整核心补丁，含新 dispatcher 和验证程序。
- `docs/report-draft.md`：报告正文初稿。
- `docs/defense-outline.md`：答辩页序与讲述重点。
- `experiments/README.md`：复现实验流程与口径。
- `experiments/integrated_validation_20260901.log`：最新 B300 独立复跑日志。
- `SOURCE_MANIFEST.md`：本机/服务器一致性与关键文件校验和。

## 应用补丁

```bash
git clone --recurse-submodules https://github.com/MoonshotAI/FlashKDA.git
cd FlashKDA
git checkout 1ce47ea
git apply /path/to/0001-k2-value-slice-and-dispatch.patch
```

构建与验证依赖 B300、CUDA 13.0、Python 3.12 和仓库的 CUTLASS 子模块。具体命令见 `experiments/README.md`。

## 结果口径

这里的百分比是 FlashKDA forward operator 或 state-carrying KDA trace 的延迟降低，不是完整 Kimi serving 的 tokens/s、TTFT 或端到端吞吐提升。报告和答辩必须保持这个边界。
