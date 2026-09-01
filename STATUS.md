# 完成状态

## 已完成并实测

- M0：0.1、0.2、0.3
- M1：1.1、1.2、1.3、1.4、1.5
- M2：2.1、2.2、2.3
- M6：6.1

M0/M1/M6 的 GPU 结果来自 NVIDIA B300；M2 为 host 判测。

## 尚待完成

- M3：3.1–3.4
- M4：4.1–4.3、4.5；4.4 为选做
- M5：5.1–5.5；5.3(d) 为选做
- 团队 C2 尚待推进

## 团队 C1：Kimi KDA / FlashKDA

- 已在 B300 上复现并定位 K2 低并行度瓶颈。
- 已实现 V16/V32/V64/V128 ValueSlice 与受保护的资源感知 dispatcher。
- fixed BF16/FP32、ragged、stateful 和 CUDA Graph 正确性通过。
- 2026-09-01 Slurm Job 14592 独立复跑：forward 高价值区间降低
  9.37%–26.10%，state-carrying trace 降低 5.68%。
- 报告初稿、答辩骨架、可应用到官方 `1ce47ea` 的补丁已归档在
  [`team-projects/kimi-kda/`](team-projects/kimi-kda/)。
- 仍需团队补成员信息，并决定是否继续做 CTA Cluster + TMA multicast、
  选择性 tcgen05 和完整模型级实验。

未完成目录保留课程骨架，仅用于后续协作，不应被描述为已经通过判测。

## 提交前收口

- 填写成员姓名、日期和环境中仍为空的字段。
- 合并 B/C 的代码、实测表格和原理说明。
- 在目标 GPU 上重新运行最终判测。
- 公开前确认课程允许发布题目衍生代码。
