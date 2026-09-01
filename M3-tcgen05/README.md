# M3 · SM100 `tcgen05`

负责人：B。3.1–3.4 已完成；CUDA 实现已在 NVIDIA B300 / CUDA 13.0 上验证。

| 小题 | 内容 | 状态与材料 |
|---|---|---|
| 3.1 | `tcgen05`/TMEM 概念 | [判断、容量计算与理由](3.1-concepts/README.md) |
| 3.2 | 单 tile GEMM | [实现、fence 对照和五组 seed](3.2-single-tile/README.md) |
| 3.3 | mbarrier debug | [状态机、错误复现和修复结果](3.3-mbarrier-debug/README.md) |
| 3.4 | CTA pair | [group 1/2、NCU 数据和分析](3.4-cta-pair/README.md) |

B300 汇总数据与完整复现输出见
[`docs/evidence/b300-results.md`](../docs/evidence/b300-results.md)。

