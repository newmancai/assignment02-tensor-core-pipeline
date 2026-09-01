# Assignment 02 · Tensor Core & Pipeline

本仓库按“模块 → 小题”归档 Assignment 02 的代码、实验记录和原理说明。
它是便于学习、交接和复现实验的整理版，不替代课程原始题面。

## 当前进度

| 模块 | 内容 | 状态 |
|---|---|---|
| M0 | 环境、架构与 Roofline | 已完成 |
| M1 | fragment、`mma.sync`、`ldmatrix` | 已完成 |
| M2 | descriptor、swizzle | 已完成 |
| M3 | `tcgen05` | 待 B 完成，当前为题目骨架 |
| M4 | 完整 GEMM、TMA、pipeline | 待 B/C 完成，当前为题目骨架 |
| M5 | 低精度与 block scaling | 待 C 完成，当前为题目骨架 |
| M6 | TileLang lowering 对照 | 已完成 |
| Team C1 | Kimi KDA / FlashKDA on B300 | 已完成 ValueSlice 主线、报告初稿与独立复跑 |

## 目录

- [`M0-environment-and-roofline/`](M0-environment-and-roofline/)：0.1–0.3
- [`M1-fragment-and-mma/`](M1-fragment-and-mma/)：1.1–1.5
- [`M2-descriptor-and-swizzle/`](M2-descriptor-and-swizzle/)：2.1–2.3
- [`M3-tcgen05/`](M3-tcgen05/)：3.1–3.4
- [`M4-gemm/`](M4-gemm/)：4.1–4.5
- [`M5-low-precision/`](M5-low-precision/)：5.1–5.5
- [`M6-tilelang/`](M6-tilelang/)：6.1
- [`docs/`](docs/)：完整报告与 A 负责人交接记录
- [`team-optional/`](team-optional/)：团队题原始说明
- [`team-projects/kimi-kda/`](team-projects/kimi-kda/)：Kimi KDA 代码补丁、B300 实验、报告与答辩材料

## 快速开始

环境与编译方法见 [`BUILD.md`](BUILD.md)，完成状态见 [`STATUS.md`](STATUS.md)。
公开 GitHub 仓库前，请先看 [`OPEN_SOURCE_CHECKLIST.md`](OPEN_SOURCE_CHECKLIST.md)。

## 已验证环境

- NVIDIA B300 SXM6 AC
- CUDA 13.0 / NVCC 13.0.88
- 默认目标：`compute_100f` / `sm_100f`
- TileLang 0.1.13

## 学习顺序

建议按 `fragment 映射 → MMA 输入打包 → ldmatrix → descriptor/swizzle →
WGMMA/TMA pipeline → TileLang lowering` 阅读。每个小题目录内保留实现或说明文档。
