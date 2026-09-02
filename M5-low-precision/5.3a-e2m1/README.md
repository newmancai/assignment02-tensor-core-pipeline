# 5.3(a) · E2M1 编码器

状态：已完成，并在 NVIDIA B300（CUDA 13.0）上与硬件逐位对拍通过。

`03a_encode_check.cu` 用 CUDA 原生 FP4 转换结果检查自定义编码器。需要明确
符号位、幅值格点、舍入和饱和规则。

## 实现

正数 magnitude code `0..7` 对应
`0, 0.5, 1, 1.5, 2, 3, 4, 6`。相邻格点的中点与 RN-even 结果为：

| 中点 | 结果 | magnitude code |
|---:|---:|---:|
| 0.25 | 0 | 0 |
| 0.75 | 1 | 2 |
| 1.25 | 1 | 2 |
| 1.75 | 2 | 4 |
| 2.5 | 2 | 4 |
| 3.5 | 4 | 6 |
| 5.0 | 4 | 6 |

因此实现可以在这些精确可表示的中点交替使用 `<=` 与 `<`，让 tie 总是落到
尾数位为 0 的偶数 code。绝对值超过 5 后都编码为最大幅值 6，因而自然实现
`satfinite`；`signbit` 负责符号位并保留 `-0`。

## 复现与结果

从仓库根目录执行：

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p build/m5
nvcc -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_100f,code=sm_100f \
  M5-low-precision/5.3a-e2m1/03a_encode_check.cu \
  -o build/m5/03a_encode_check
srun -G 1 --time 00:05:00 ./build/m5/03a_encode_check
```

B300 Slurm Job `15331`：

```text
PASS: 202864 values match hardware
```

