# 4.5 · Thin GEMM

状态：已完成。在 NVIDIA B300 上覆盖 7 个 Kimi K3 投影形状和
`M=1..65536` 的 63 个点，保存三次独立进程原始输出并以逐点中位数报告。

需要覆盖 decode 小 M、过渡区和较大 chunked-prefill M，记录理论 roof、实测
吞吐、相对 cuBLAS 表现，并解释小 M 时并行度和 launch/访存成本的影响。

## 方法与 Roofline

程序使用 cuBLAS BF16 输入/输出、FP32 累加，因而这里测到的就是生产库的
Tensor Core 路径；“达成率塌掉”不是手写 kernel 质量问题。每个点先 warmup，
随后按 M 运行 20–200 次 CUDA event 计时。理论口径沿用 0.2：B300 dense BF16
峰值 2250 TFLOPS、带宽 8000 GB/s，机器平衡点 281.25 FLOP/byte。

```text
AI       = 2MNK / [2(MK + NK + MN)]
roof     = min(2250 TFLOPS, AI × 8000 GB/s / 1000)
TC 达成率 = measured TFLOPS / 2250
BW 达成率 = effective GB/s / 8000
```

当 `AI<281.25` 时表中 roof 判为 memory-bound，否则判为 compute-bound。
该模型把 A/W/D 各算一次必要流量，适合解释趋势；缓存复用和固定 launch 成本
会让极小 M 偏离连续 roofline 假设。

## 三次复跑中位数

下表给出完整 M 轴中的五个关键位置，单位 TFLOPS。63 行的 AI、roof、时间、
TFLOPS、TC/BW 达成率可由随仓库提交的原始输出一键生成。

| 投影 `(N,K)` | M=1 | M=16 | M=256 | M=1024 | M=65536 |
|---|---:|---:|---:|---:|---:|
| `f_b_proj` (1536,128) | 0.1 | 1.5 | 23.9 | 86.7 | 413.2 |
| `q_b_proj` (2304,1536) | 0.6 | 20.1 | 296.1 | 723.3 | 1265.1 |
| `o_proj` (7168,1536) | 3.5 | 60.9 | 561.7 | 792.6 | 1311.3 |
| `fused_qkv_a_proj` (2112,7168) | 3.0 | 46.1 | 449.9 | 876.5 | 1302.6 |
| `in_proj_qkvgfab` (6288,7168) | 4.6 | 72.8 | 860.0 | 1162.8 | 1292.1 |
| `dense_down_proj` (7168,8448) | 4.4 | 73.0 | 909.6 | 934.1 | 1317.6 |
| `dense_gate_up_proj` (16896,7168) | 5.7 | 95.1 | 1005.8 | 1140.1 | 1322.6 |

大 K 形状的边界与平台：

| M | Roofline 状态 | 代表现象 |
|---:|---|---|
| 1–16 | memory-bound，但固定 launch/并行度不可忽略 | 0.6–95.1 TFLOPS；M=16 的 BW roof 达成率约 16.0%–74.5% |
| 64–256 | memory-bound 过渡区 | M=256 为 296.1–1005.8 TFLOPS，AI 约 200–244 |
| 1024 | 大 K 形状跨过机器平衡点 | AI 约 485–851，开始 compute-bound |
| 4096–65536 | compute 平台 | 约 1.02–1.32 PFLOPS；M=65536 为官方 BF16 峰值的 56.2%–58.8% |

`f_b_proj(K=128)` 是例外：AI 随 M 增大只渐近到约 128，实测 M=65536 时
仍只有 117.9 FLOP/byte，始终低于 281.25；因此始终 memory-bound。它在小 M
时连权重矩阵都很小，固定 launch、调度、Tensor Core tile 填充和低 K setup
主导，M=16 仅达到 memory roof 的 1.3%。即使 M=65536，也只有 413.2 TFLOPS、
43.8% memory roof。

## 结论：Tensor Core 什么时候帮不上忙

1. 对大 K 投影，`M<256` 后吞吐开始明显塌落，`M≤16` 是最严重的 decode 区；
   此时输出 tile/CTA 太少，Tensor Core、TMA/内部 staging 与 cuBLAS dispatch 的
   固定成本无法摊薄。
2. `M≈4096` 后多数形状进入 1.1–1.3 PFLOPS 平台，M=65536 达到官方 dense
   BF16 峰值的约 56%–59%。
3. 小 M 的低“TC 峰值达成率”主要说明计算单元没有足够工作；大权重形状同时
   可达到较高“BW roof 达成率”，说明时间主要花在一次性流过权重，而非做满
   Tensor Core FMA。两个分母回答的是不同问题，不能只看 TFLOPS 百分比。
4. vLLM 在 M≤16 改用 skinny CUDA Core FMA kernel，是因为直接按 decode
   形状流过权重可以绕过通用 GEMM 的 Tensor Core/TMA setup、tile padding 和
   launch/dispatch 成本；算力峰值更低，但端到端延迟反而可下降。

## 复现与证据

从仓库根目录提交：

```bash
sbatch M4-gemm/4.5-thin-gemm/run_b300.sbatch
python M4-gemm/4.5-thin-gemm/summarize_results.py \
  M4-gemm/4.5-thin-gemm/evidence/run-1.txt \
  M4-gemm/4.5-thin-gemm/evidence/run-2.txt \
  M4-gemm/4.5-thin-gemm/evidence/run-3.txt
```

环境：NVIDIA B300 SXM6 AC，CUDA 13.0；Slurm Job `15340`。原始输出见
[`evidence/`](evidence/)，脚本不会手工改写数据。

