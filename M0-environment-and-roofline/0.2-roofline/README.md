# 0.2 · 峰值与机器平衡点

完整推导见 [`derivation.md`](derivation.md)。

采用官方 B300 dense BF16 峰值与 up-to HBM 带宽时：

```text
machine balance = 2250 TFLOP/s / 8000 GB/s
                = 281.25 FLOP/byte
```

单条 MMA 的局部算术强度不能直接代表完整 GEMM；完整 kernel 可以通过 tile
复用、shared memory、寄存器/TMEM 和流水重叠提高整体算术强度。

