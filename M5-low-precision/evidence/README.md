# M5 B300 evidence

- [`b300-final-regression.md`](b300-final-regression.md)：最终统一回归、三次
  5.4 复跑中位数、逐形状 ceiling 与 Nsight Compute 摘要；
- [`../run_b300.sbatch`](../run_b300.sbatch)：可复现的编译、判测和三次计时脚本。

所有 GPU 程序均在 Slurm 分配的 NVIDIA B300 上运行；登录节点只用于编译和
提交作业。
