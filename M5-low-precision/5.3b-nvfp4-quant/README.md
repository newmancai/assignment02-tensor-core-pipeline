# 5.3(b) · NVFP4 quant kernel

状态：已完成；逐 byte 判测和 cuBLASLt FP4 GEMM 消费端测试均在 B300 通过。

目录包含量化入口、E2M1 编码、scale-factor swizzle 工具以及 FP4 GEMM 消费端
测试。需要同时验证 packed data 与 E4M3 scale-factor 布局。

## Kernel 组织

- 一个线程负责 K 方向连续 16 个 BF16，依次执行 `amax → E4M3 SF →
  反量化 SF → 8 次 FP4x2`，从而避免组内同步并与 host 参考的 `fmaxf`
  运算顺序完全一致。
- 8 个 `__nv_fp4x2_e2m1` 结果拼成一个 `uint64_t` 后一次写回；偶数元素在
  低 nibble，奇数元素在高 nibble。
- SF 使用 `sf_swizzled_offset(row, kGroup, numKTiles)` 写入 Tensor Core
  要求的 `[numMTiles, numKTiles, 32, 4, 4]` 布局；调用方预先将 padding
  清零。
- 采用 256 线程的 grid-stride kernel。B300 上 kernel 使用 39 registers/thread，
  每个 SM 最多驻留 6 个 CTA，因此将大矩阵 grid 限为 `6 × SM`，避免
  `8 × SM` 带来的第二个 partial wave。该调整把 `4096×7168` 从约
  61.8 us 降至约 55.7 us。

`cuobjdump --dump-sass` 可看到每组对应的 8 条
`F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`，确认没有退化为软件比较链。

## 构建与判测

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p build/m5
nvcc -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_100f,code=sm_100f \
  M5-low-precision/5.3b-nvfp4-quant/03b_nvfp4_quant.cu \
  -o build/m5/03b_nvfp4_quant
nvcc -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_100f,code=sm_100f \
  M5-low-precision/5.3b-nvfp4-quant/test_fp4_gemm.cu \
  -lcublasLt -o build/m5/test_fp4_gemm
srun -G 1 --time 00:10:00 bash -lc \
  './build/m5/03b_nvfp4_quant && ./build/m5/test_fp4_gemm'
```

逐 byte 判测（Job `15336`；大形状性能采用 5 次复跑中位数）：

| M | K | 判测 | 时间 | 有效带宽 |
|---:|---:|---|---:|---:|
| 128 | 1024 | PASS, bad=0 | 5.13 us | 66 GB/s |
| 200 | 4096 | PASS, bad=0 | 6.16 us | 341 GB/s |
| 4096 | 7168 | PASS, bad=0 | 55.73 us | 1350 GB/s |

cuBLASLt 消费端结果（Job `15331`）：

| M | N | K | maxrel | 结果 |
|---:|---:|---:|---:|---|
| 128 | 128 | 1024 | 3.880e-3 | PASS |
| 256 | 512 | 4096 | 3.891e-3 | PASS |
| 200 | 128 | 1024 | 3.880e-3 | PASS（覆盖 M 非 128 倍数的 SF padding） |

三组 maxrel 都约为 `3.9e-3`，符合 BF16 输出舍入误差预期，并同时验证 packed
FP4 数据、E4M3 SF 字节与 swizzled SF 布局可被真实 Tensor Core 路径消费。

