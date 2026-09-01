# 4.2 实验记录

| 字段 | 记录 |
|---|---|
| 环境 | NVIDIA B300 SXM6 AC；CUDA/NVCC 13.0.88；`compute_100f/sm_100f`；4096³；Slurm Job 14793 |
| Commit | 本次 B 部分提交（父提交 `6574c37afe144e99d50418dd38c83a8cdef8d2a7`；最终 hash 见仓库历史） |
| 运行命令 | [README 中的编译与运行命令](README.md) |
| 正确性 | 与 cuBLAS BF16 输出逐元素位模式比较，`exact PASS` |
| 性能数据 | 0.492 ms，279.5 TFLOPS；cuBLAS 978.8 TFLOPS；达成率 28.6% |
| 现象 | 相对 4.1 提升约 5.6 倍；单个 barrier 同时承担 TMA/MMA completion 时大形状会挂死，拆为 full/empty 后稳定 |
| 结论 | TMA 消除大量普通地址生成/load/store 指令；单缓冲仍串行等待，尚需多级流水隐藏延迟 |
| 证据链接 | [CUDA 源码](02_tma.cu)、[TensorMap 与同步分析](README.md)、[B300 输出](../../docs/evidence/b300-results.md#42-tma) |
