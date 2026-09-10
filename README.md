# Assignment 02 · Tensor Core & Pipeline

本仓库是 Assignment 02 的提交归档：M0–M6 保留逐题实现、判测输出、实验分析与
GPU 标注；团队大作业 C1 另含代码、正式报告、答辩材料和论文。课程原始题面不在
仓库内重复发布，要求与交付物的逐项映射见 [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md)。

## 提交入口

| 交付物 | 入口 | 当前状态 |
|---|---|---|
| M0–M6 总报告 | [`docs/full-report.md`](docs/full-report.md) | 个人作业；必做项完整，不设成员署名 |
| M0–M6 代码与逐题说明 | [`M0-environment-and-roofline/`](M0-environment-and-roofline/) 至 [`M6-tilelang/`](M6-tilelang/) | Host/B300 判测通过 |
| B300 文本证据 | [`docs/evidence/b300-results.md`](docs/evidence/b300-results.md) | 含 Job、GPU、PASS 与性能数据 |
| C1 冻结提交包 | [`team-projects/kimi-kda/c1-sm100-delivery/`](team-projects/kimi-kda/c1-sm100-delivery/) | 代码、报告、PPTX/PDF、论文、证据齐全 |
| 构建与复现 | [`BUILD.md`](BUILD.md) | 本机检查与 B300 运行入口 |
| 完成状态 | [`STATUS.md`](STATUS.md) | 必做/选做边界与尚需人工填写项 |

## 模块完成度

| 模块 | 内容 | 状态 |
|---|---|---|
| M0 | 环境、架构与 Roofline | 已完成 |
| M1 | fragment、`mma.sync`、`ldmatrix` | 已完成 |
| M2 | descriptor、swizzle | 已完成 |
| M3 | `tcgen05` | 3.1–3.4 已完成并在 B300 验证 |
| M4 | 完整 GEMM、TMA、pipeline | 4.1–4.3、4.5 已完成并在 B300 验证；4.4 为选做 |
| M5 | 低精度与 block scaling | 5.1–5.5 必做项已完成；Host/B300 回归通过 |
| M6 | TileLang lowering 对照 | 已完成 |
| Team C1 | FlashKDA 从 SM80 MMA 迁移到 SM100 是否值得 | 冻结内容完成，2026-09-11 署名复核完成 |

## C1 大作业速览

团队“奶龙必胜”：蔡雨洋、李奥、赵骋。

在已测 B300 工作负载上，机械地将 HMMA 替换为 `tcgen05` 没有形成稳定收益；
围绕数据驻留、布局、并行度、流水和 dispatch 协同重组的 SM100 全栈路径取得
明确收益。因此交付建议是：只对已认证 profile 启用受条件保护的 SM100 后端，
其余场景回退 HMMA。

- [正式技术报告](team-projects/kimi-kda/c1-sm100-delivery/03_reports/SM100_MAINLINE_DELIVERY.md)
- [论文 PDF](team-projects/kimi-kda/c1-sm100-delivery/02_paper/main.pdf)
- [课程提交答辩 PDF（署名版）](team-projects/kimi-kda/c1-sm100-delivery/01_slides/C1_FlashKDA_SM100_奶龙必胜_署名版_20260910.pdf)
- [课程提交答辩 PPTX（署名版）](team-projects/kimi-kda/c1-sm100-delivery/01_slides/C1_FlashKDA_SM100_奶龙必胜_署名版_20260910.pptx)
- [公开脱敏答辩版本](team-projects/kimi-kda/c1-sm100-delivery/01_slides/README.md)
- [代码、证据与复现总入口](team-projects/kimi-kda/c1-sm100-delivery/README.md)

`team-projects/kimi-kda/docs/c1-final/` 保留 2026-09-03 至 2026-09-09 的过程
版本，仅用于追溯，不作为最终提交入口。

## 目录导览

- [`M0-environment-and-roofline/`](M0-environment-and-roofline/)：0.1–0.3
- [`M1-fragment-and-mma/`](M1-fragment-and-mma/)：1.1–1.5
- [`M2-descriptor-and-swizzle/`](M2-descriptor-and-swizzle/)：2.1–2.3
- [`M3-tcgen05/`](M3-tcgen05/)：3.1–3.4
- [`M4-gemm/`](M4-gemm/)：4.1–4.5
- [`M5-low-precision/`](M5-low-precision/)：5.1–5.5
- [`M6-tilelang/`](M6-tilelang/)：6.1
- [`docs/`](docs/)：总报告、逐模块复现记录与 B300 证据
- [`team-optional/`](team-optional/)：团队题原始骨架
- [`team-projects/kimi-kda/`](team-projects/kimi-kda/)：C1 全部研究与交付材料

## 已验证环境

- NVIDIA B300 SXM6 AC
- CUDA 13.0 / NVCC 13.0.88
- 默认目标：`compute_100f` / `sm_100f`
- TileLang 0.1.13

建议按 `fragment 映射 → MMA 输入打包 → ldmatrix → descriptor/swizzle →
WGMMA/TMA pipeline → TileLang lowering` 阅读。公开仓库前的隐私、课程授权和
许可证边界见 [`OPEN_SOURCE_CHECKLIST.md`](OPEN_SOURCE_CHECKLIST.md)。
