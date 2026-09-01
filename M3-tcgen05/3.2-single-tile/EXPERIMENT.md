# 3.2 实验记录

| 字段 | 记录 |
|---|---|
| 环境 | NVIDIA B300 SXM6 AC；CUDA/NVCC 13.0.88；`compute_100f/sm_100f`；Slurm Job 14793 |
| Commit | 本次 B 部分提交（父提交 `6574c37afe144e99d50418dd38c83a8cdef8d2a7`；最终 hash 见仓库历史） |
| 运行命令 | [README 中的编译与判测命令](README.md) |
| 正确性 | seeds 1、7、42、1234、99999 全部 PASS |
| 性能数据 | 单 tile 正确性题，不作为性能结论 |
| 现象 | 正确版全部通过；定义 `OMIT_PROXY_FENCE` 后本次五组 seed 仍通过 |
| 结论 | 阴性对照不证明 fence 可删除；generic shared 写到 async proxy 读的可见性仍需 proxy fence 保证 |
| 证据链接 | [CUDA 源码](02_single_tile.cu)、[完整原理与结果](README.md)、[B300 原始输出](../../docs/evidence/b300-results.md#32-正确版本) |
