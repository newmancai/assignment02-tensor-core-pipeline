# 3.3 实验记录

| 字段 | 记录 |
|---|---|
| 环境 | NVIDIA B300 SXM6 AC；CUDA/NVCC 13.0.88；`compute_100f/sm_100f`；Slurm Job 14793 |
| Commit | 本次 B 部分提交（父提交 `6574c37afe144e99d50418dd38c83a8cdef8d2a7`；最终 hash 见仓库历史） |
| 运行命令 | [README 中的复现命令](README.md) |
| 正确性 | 修复版 rounds=1/2/4、seeds=42/7 全部 PASS |
| 性能数据 | barrier debug 题，不要求吞吐性能；错误版采用 20 s timeout |
| 现象 | `BUGGY_PHASE` 下 rounds=1 PASS，rounds=2/4 失败或超时 |
| 结论 | barrier generation 每轮翻转，等待相位必须是 `round & 1`；固定 phase 0 会等待旧代际 |
| 证据链接 | [CUDA 源码](03_bug_mbarrier.cu)、[状态机与分析](README.md)、[B300 原始输出](../../docs/evidence/b300-results.md#33-buggy_phase-复现版本) |
