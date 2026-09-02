# 4.5 实验记录

| 字段 | 记录 |
|---|---|
| 环境 | NVIDIA B300 SXM6 AC；CUDA/NVCC 13.0；`compute_100f/sm_100f`；Slurm Job 15340 |
| 代码 | [`05_thin_gemm.cu`](05_thin_gemm.cu) |
| 运行方法 | [`run_b300.sbatch`](run_b300.sbatch)；3 个独立进程，每点先 warmup |
| 形状 | 7 个 Kimi K3 投影 `(N,K)` × 9 个 M，共 63 点 |
| 理论参数 | 2250 TFLOPS dense BF16；8000 GB/s；平衡点 281.25 FLOP/byte |
| 核心结果 | M≤16 吞吐塌落；大 K 在 M≈1024 转 compute-bound，M≥4096 进入约 1.1–1.3 PFLOPS 平台；K=128 始终 memory-bound |
| 证据 | [`evidence/`](evidence/) 三次原始输出；[`summarize_results.py`](summarize_results.py) 逐点中位数生成器 |
