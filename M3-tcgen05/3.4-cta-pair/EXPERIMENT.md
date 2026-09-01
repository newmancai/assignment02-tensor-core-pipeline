# 3.4 实验记录

| 字段 | 记录 |
|---|---|
| 环境 | NVIDIA B300 SXM6 AC；CUDA/NVCC 13.0.88；`compute_100f/sm_100f`；Nsight Compute；Slurm |
| Commit | 本次 B 部分提交（父提交 `6574c37afe144e99d50418dd38c83a8cdef8d2a7`；最终 hash 见仓库历史） |
| 运行命令 | [README 中的编译与 NCU 命令](README.md) |
| 正确性 | `cta_group::1` 与 `cta_group::2` 均 PASS |
| 性能数据 | shared memory/block 24588/20492 B；shared-store wavefront 778/650；单 tile 时间在噪声范围 |
| 现象 | group 2 每 CTA 只 staging 一半 B，A staging 不变，因此总 shared 流量下降但不会减半 |
| 结论 | group-2 协作以 cluster/TMEM 机制换取较低的每 CTA shared-memory 占用，可为 M4 深流水腾出容量 |
| 证据链接 | [CUDA 源码](04_cta_pair.cu)、[原理、数据与结论](README.md)、[B300/NCU 输出](../../docs/evidence/b300-results.md#34-cta-pair) |
