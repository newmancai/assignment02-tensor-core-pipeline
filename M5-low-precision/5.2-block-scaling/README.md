# 5.2 · Block scaling

状态：已完成。`block_scale_sim.py` 用 FP64 模拟 row/column scale 和沿 K
分块 scale 的恢复位置；它只验证代数与求和分组，不模拟窄精度量化误差。

## 运行与结果

从仓库根目录执行：

```bash
python -m pytest M5-low-precision/5.2-block-scaling/test_block_scale.py -q
python M5-low-precision/5.2-block-scaling/block_scale_sim.py
```

本地环境为 Python 3.12.7、PyTorch 2.7.1+cu118（CPU 路径）。判测结果：

```text
...                                                                      [100%]
3 passed in 1.78s
```

固定 seed=3，`M=7, N=5, K=512, SEG=128` 的数值对照为：

| 实现 | max abs error vs FP64 GEMM | mean abs error |
|---|---:|---:|
| row/col，点积后恢复一次 | 4.263256e-14 | 1.338612e-14 |
| K-block，每段恢复 | 3.197442e-14 | 9.566950e-15 |
| K-block，错误地仅恢复第一段 | 2.216998e+1 | 6.697280e+0 |

前两行的 `1e-14` 量级差异来自 FP64 除法、乘法和分段加法的舍入/重分组，
在测试的 `rtol=atol=2e-13` 内。第三行不是舍入噪声，而是错误地把随 K
block 变化的 scale 当成了常量。

## 代数等价关系

代码将 `B` 存为 `[N,K]`，所以参考 GEMM 是 `C=A @ B.T`。

### row/column scale：可以只恢复一次

令每个 A 行和 B 行（即输出列）分别只有一个 scale：

```text
A[m,k] = sA[m] * qA[m,k]
B[n,k] = sB[n] * qB[n,k]

C[m,n]
  = sum_k A[m,k] * B[n,k]
  = sA[m] * sB[n] * sum_k qA[m,k] * qB[n,k]
```

`sA[m]*sB[n]` 对整个 K 归约不变，因此可先完成归一化矩阵的完整点积，
再在 `[M,N]` 输出上乘一次 scale 外积。

### K-block scale：必须逐段恢复

令 `g(k)=floor(k/SEG)`，scale 随 K block 改变：

```text
C[m,n]
  = sum_g sA[m,g] * sB[n,g]
          * sum_{k in block g} qA[m,k] * qB[n,k]
```

这里每个 `g` 的 scale 乘积通常不同，不能从 `sum_g` 外提出。正确顺序是
“段内归一化点积 → 乘回该段 `sA*sB` → 累加到 C”。只有当所有 block 的
`sA[m,g]*sB[n,g]` 恰好相等时，才可能退化成一次恢复；故意错误的对照拿
第 0 段 scale 恢复全部 partial sum，最大绝对误差达到 22.17。

实现还检查 A/B 的 K 维、device、scale shape、scale 有限且为正，以及
`K % SEG == 0`，避免未覆盖的 K 尾段被静默丢弃。

## 为什么 scale 分组常沿 K

GEMM 的每个输出是沿 K 的点积。沿 K 连续分组后，一个 block 内的输入
共享 scale，硬件/软件可先形成对应 partial sum，再在它进入最终累加前
恢复一次。这样 scale 的生命周期与 reduction 分段一致，也能把某个 K
局部 outlier 的影响限制在该段。若 scale 在 K 内变化却等到整个点积结束
才恢复，就丢失了每段应有的不同权重，正是第三个测试覆盖的错误。

## scale 粒度权衡

| 分组示例 | 精度与 outlier 隔离 | scale 数量/流量 | 典型取舍 |
|---|---|---|---|
| 128×128 | 最粗，一个极值可影响整 tile | 每 16,384 元素一个 | metadata 最少，适合 tile 动态范围较均匀时 |
| 1×128 | 每行的 128 个 K 元素独立 | 每 128 元素一个 | 与 K partial sum 对齐，精度与开销折中 |
| 1×16 | 最细，局部动态范围最贴合 | 每 16 元素一个 | 小值保留最好，但 scale load、存储和处理最多 |

从 1×128 缩小到 1×16，scale 数量严格增加 **8 倍**。若每个 scale 占
1 byte，则 metadata 从 `1/128=0.0078125 byte/元素` 增至
`1/16=0.0625 byte/元素`；相对于 4-bit 数据的 `0.5 byte/元素`，理想
存储开销分别约为 **1.5625%** 和 **12.5%**（未计对齐与布局 padding）。
精度收益来自每个 scale 只需覆盖更窄的局部 amax 范围：outlier 污染范围
从 128 个元素降到 16 个，零阈值和量化步长通常随局部 scale 一起下降；
代价则是 8 倍 scale metadata、更频繁的加载/转换，以及更复杂的布局。

