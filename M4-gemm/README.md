# M4 · 完整 GEMM

负责人：B（4.1–4.3）与 C（4.5）。全部必做项已完成并通过 B300 回归。

| 小题 | 内容 | 状态 |
|---|---|---|
| 4.1 | tiled GEMM | [完成：exact PASS，49.9 TFLOPS](4.1-tiled/README.md) |
| 4.2 | TMA staging | [完成：exact PASS，279.5 TFLOPS](4.2-tma/README.md) |
| 4.3 | 多级 pipeline | [完成：stage sweep 全部 exact PASS](4.3-pipeline/README.md) |
| 4.4 | 扩展题 | 选做 |
| 4.5 | thin GEMM | [完成：63 点三次复跑与 Roofline 分析](4.5-thin-gemm/README.md) |

性能阶梯、两种形状的 stage sweep 和原始输出见各小题 README 及
[`docs/evidence/b300-results.md`](../docs/evidence/b300-results.md)。

