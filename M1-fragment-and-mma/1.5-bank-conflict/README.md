# 1.5 · Stride 与 bank conflict

B300 实测：

| stride | cycles/iter | wavefront | bank conflict |
|---:|---:|---:|---:|
| 32 B | 9.72 | 16384 | 8192 |
| 64 B | 10.75 | 32768 | 24576 |
| 128 B | 16.06 | 65536 | 57344 |
| 144 B | 9.23 | 8192 | 0 |

归一化到 32 B 的预测 wavefront 比例为 `1, 2, 4, 0.5`。128 B 的 cycles
只增长到约 1.65 倍，是因为多个 warp 可以隐藏/重叠部分串行化，循环中也有
固定流水开销；wavefront 比例不会机械等于总周期比例。

