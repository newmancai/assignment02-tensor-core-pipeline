# 4.1 实验记录

| 字段 | 记录 |
|---|---|
| 环境 | NVIDIA B300 SXM6 AC；CUDA/NVCC 13.0.88；Nsight Compute 2025.3.1；`compute_100f/sm_100f`；4096³；benchmark Job 14793，NCU Job 14904 |
| Commit | 本次 B 部分提交（父提交 `6574c37afe144e99d50418dd38c83a8cdef8d2a7`；最终 hash 见仓库历史） |
| 运行命令 | [README 中的 Slurm/NCU 命令](README.md)；[profile_ncu.sh](profile_ncu.sh) |
| 正确性 | 与 cuBLAS BF16 输出逐元素位模式比较，`exact PASS` |
| 性能数据 | 2.753 ms，49.9 TFLOPS；cuBLAS 976.2 TFLOPS；达成率 5.1% |
| 现象 | NCU：SM 46.62%、Memory 30.97%、DRAM 0.38%、Issue Slots Busy 41.40%，约 13% excessive global sectors |
| 结论 | 主要瓶颈是普通 staging 的地址/指令开销、访问合并和单缓冲延迟，不是 HBM 带宽或 Tensor Core 峰值 |
| 证据链接 | [CUDA 源码](01_tiled.cu)、[NCU 文本](evidence/m41-details.txt)、[NCU 报告](evidence/m41-detailed.ncu-rep)、[B300 输出](../../docs/evidence/b300-results.md#41-tiled-gemm) |
