# 4.5 B300 原始证据

`run_b300.sbatch` 会在本目录生成：

- `slurm-<job-id>.out`：作业、GPU、CUDA 编译器元数据以及三次完整输出；
- `run-1.txt`、`run-2.txt`、`run-3.txt`：三个独立进程的原始 benchmark 表。

报告表格使用三次运行逐形状、逐 M 的中位数；原始输出不做手工改写。
