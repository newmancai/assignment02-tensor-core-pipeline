# C 负责人交接：4.5 与 M5

## 范围

已完成必做题：

- 4.5 thin GEMM；
- 5.1 FP8 outlier；
- 5.2 block scaling；
- 5.3(a) E2M1、5.3(b) NVFP4 quant、5.3(c) ceiling probe；
- 5.4 fused RMSNorm + NVFP4；
- 5.5 W4A16 与 NVFP4 对照。

明确未纳入：4.4 Optional、5.3(d) Optional、5.4 题面末尾额外选做优化、团队
C1/C2。后两类团队材料与本次非团队必做交付保持分离。

## 主要交付

| 范围 | 代码/报告 | 验证 |
|---|---|---|
| 4.5 | [`M4-gemm/4.5-thin-gemm/`](../M4-gemm/4.5-thin-gemm/README.md) | B300 63 点 × 3 次 |
| 5.1 | [`quant_outlier.py`](../M5-low-precision/5.1-fp8-outlier/quant_outlier.py) | 4 pytest |
| 5.2 | [`block_scale_sim.py`](../M5-low-precision/5.2-block-scaling/block_scale_sim.py) | 3 pytest |
| 5.3 | [`M5-low-precision/`](../M5-low-precision/README.md) | 硬件逐位、逐 byte、cuBLASLt、ceiling/NCU |
| 5.4 | [`04_fused_rms_nvfp4.cu`](../M5-low-precision/5.4-fused-rmsnorm-nvfp4/04_fused_rms_nvfp4.cu) | 十形状 × 3 次、NCU |
| 5.5 | [`5.5-concepts/README.md`](../M5-low-precision/5.5-concepts/README.md) | 概念互审 |

总报告的 C 部分已写入 [`full-report.md`](full-report.md)，集中证据见
[`M5-low-precision/evidence/b300-final-regression.md`](../M5-low-precision/evidence/b300-final-regression.md)。

## 最终结果摘要

- Host：`7 passed`。
- E2M1：202864 个候选与 B300 硬件逐位一致。
- NVFP4 quant：三形状 packed data/SF `bad=0`；三组 cuBLASLt FP4 GEMM PASS。
- 同形 quant/probe：五次中位数约 99.5%，量化通路接近自身访存上限。
- Fused RMSNorm + NVFP4：十形状全部 PASS；三次中位数相对公平两步基线
  1.06–1.84×。
- 4.5：大 K 在 M≤16 严重塌落，M≈1024 转 compute-bound，M≥4096 进入
  约 1.1–1.3 PFLOPS 平台；K=128 始终 memory-bound。

## 复现命令

Host：

```bash
python -m pytest \
  M5-low-precision/5.1-fp8-outlier/test_quant_outlier.py \
  M5-low-precision/5.2-block-scaling/test_block_scale.py -q
```

B300：

```bash
sbatch M4-gemm/4.5-thin-gemm/run_b300.sbatch
sbatch M5-low-precision/run_b300.sbatch
```

所有 GPU 程序只能在 Slurm allocation 中运行；登录节点只编译和提交。
