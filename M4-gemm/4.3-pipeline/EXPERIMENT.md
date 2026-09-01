# 4.3 实验记录

| 字段 | 记录 |
|---|---|
| 环境 | NVIDIA B300 SXM6 AC；CUDA/NVCC 13.0.88；Nsight Compute 2025.3.1；`compute_100f/sm_100f`；benchmark Job 14793/14931，occupancy Job 14903，NCU Job 14933 |
| Commit | 本次 B 部分提交（父提交 `6574c37afe144e99d50418dd38c83a8cdef8d2a7`；最终 hash 见仓库历史） |
| 运行命令 | [README 中的 stage sweep/Slurm 命令](README.md)；[occupancy_stages.sh](occupancy_stages.sh) |
| 正确性 | S=2/3/4/6 在 4096³ 与 256×4096×16384 两种形状全部 `exact PASS` |
| 性能数据 | 4096³：301.8/288.5/253.3/183.3 TFLOPS；thin-M：168.5/207.4/189.4/210.5 TFLOPS |
| 现象 | 四种 stage 实际均为 1 block/SM；4096³ 有约 13.8 waves，thin-M 的 128 CTA 不足一个完整 wave |
| 结论 | 深 stage 对低 grid 并发形状更有价值；性能回落不能归因于 blocks/SM 逐级下降，但 shared memory 仍以每 stage 24 KiB 消耗容量余量 |
| 证据链接 | [CUDA 源码](03_pipeline.cu)、[S=2 NCU 文本](evidence/m43-s2-details.txt)、[S=2 NCU 报告](evidence/m43-s2-basic.ncu-rep)、[B300 sweep 输出](../../docs/evidence/b300-results.md#43-stage-sweep) |
