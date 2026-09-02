# M5 · 低精度与 block scaling

负责人：C。必做题 5.1–5.5 已全部完成；Host 判测与 NVIDIA B300 Slurm
回归均通过。5.3(d) 按题面为选做，本次未纳入必做交付。

| 小题 | 内容 | 状态 |
|---|---|---|
| 5.1 | FP8 outlier | [完成：4 个 pytest + 定量报告](5.1-fp8-outlier/README.md) |
| 5.2 | block scaling | [完成：3 个 pytest + 代数对照](5.2-block-scaling/README.md) |
| 5.3(a) | E2M1 编码 | [完成：202864 点硬件逐位 PASS](5.3a-e2m1/README.md) |
| 5.3(b) | NVFP4 quant | [完成：逐 byte 与 cuBLASLt 消费端 PASS](5.3b-nvfp4-quant/README.md) |
| 5.3(c) | ceiling probe | [完成：十形状 + NCU 归因](5.3c-ceiling-probe/README.md) |
| 5.3(d) | 扩展实验 | Optional，未纳入必做交付 |
| 5.4 | RMSNorm + NVFP4 融合 | [完成：十形状 PASS，1.06–1.84×](5.4-fused-rmsnorm-nvfp4/README.md) |
| 5.5 | W4A16 与 NVFP4 概念 | [完成](5.5-concepts/README.md) |

## 快速回归

Host：

```bash
python -m pytest \
  M5-low-precision/5.1-fp8-outlier/test_quant_outlier.py \
  M5-low-precision/5.2-block-scaling/test_block_scale.py -q
```

B300（从仓库根目录提交）：

```bash
sbatch M5-low-precision/run_b300.sbatch
```

最终环境、判测、逐形状中位数与 NCU 摘要见
[`evidence/b300-final-regression.md`](evidence/b300-final-regression.md)。

