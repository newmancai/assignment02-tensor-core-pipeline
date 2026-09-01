# 4.2 · TMA 单缓冲 GEMM

标准化实验元数据见 [`EXPERIMENT.md`](EXPERIMENT.md)。

状态：实现、cuBLAS 严格对拍和 B300 性能实验已完成。

实现文件：[`02_tma.cu`](02_tma.cu)

## TensorMap

host 使用 `cuTensorMapEncodeTiled` 分别创建 A、B 的 2D tensor map：

- 数据类型：BF16；
- global dimensions：`{K, M}` 与 `{K, N}`；
- box dimensions：`{BK, BM}` 与 `{BK, BN}`；
- interleave：none；
- swizzle：128B；
- L2 promotion：128B；
- OOB fill：none。

当前题面形状严格按 tile 对齐，因此不需要 OOB zero fill。若扩展到任意形状，
需要同时调整 grid、TensorMap OOB 策略和 epilogue 边界判断。

## kernel 同步

每个 K tile 的顺序是：

1. elected lane 对 `full` barrier 执行
   `mbarrier.arrive.expect_tx`，声明 A+B 共 24576 B transaction；
2. 发射 A、B 两条 2D TMA load；
3. 等待 `full` 当前 phase，确认 TMA 搬运完成；
4. 发射四条 k16 tcgen05 MMA；
5. commit 到 `empty` barrier；
6. 等待 `empty` 当前 phase，确认 MMA 不再读取 shared tile；
7. 进入下一 K tile。

TMA completion 与 MMA completion 使用两个独立 barrier。早期版本尝试复用
同一个 barrier，小形状可以通过，但 4096³ 大 grid 会挂死；拆成
`full/empty` 后稳定通过。

## 4.1 staging 开销对比

4.1 的 shared-memory staging 包含：

- 每线程 global 地址计算；
- global load 指令；
- swizzle 后 shared 地址计算；
- shared store 指令；
- LSU/issue 带宽占用；
- 全 CTA 同步和 proxy 可见性处理。

改用 TMA 后，global tensor 寻址、批量 global load、shared-memory 写入和
swizzle 搬运由 TMA 引擎完成，不再由大量普通 CUDA 指令执行。CUDA 线程仍
负责 TensorMap 参数准备、TMA 发射以及 barrier 同步。

## 正确性与性能

4096³：

```text
time = 0.492 ms   279.5 TFLOPS
cuBLAS 0.140 ms 978.8 TFLOPS; exact PASS (bad=0); attainment 28.6%
```

相对同轮 4.1 的 49.9 TFLOPS，TMA 使性能提高约 5.6 倍，说明 4.1 的主要时间
确实消耗在普通 staging 指令和数据供给上。4.1 的 NCU detailed profile 进一步
给出 SM 46.62%、Memory 30.97%、DRAM 0.38%、Issue Slots Busy 41.40%，并检测
到约 13% excessive global sectors；这些证据说明 4.1 没有达到 HBM 带宽上限，
而是被普通地址计算、load/store 发射、访问合并和串行等待限制。详细证据见
[4.1 NCU 分析](../4.1-tiled/README.md#nsight-compute-分析)。

## 新瓶颈

本题仍是单缓冲：

`wait TMA -> MMA -> wait MMA -> next TMA`

TMA 降低了搬运的指令开销，但没有让下一 tile 的搬运和当前 tile 的计算
重叠。剩余主要问题是 TMA/MMA 串行延迟，4.3 用多级环形缓冲解决这一点。

完整输出见 [B300 实验归档](../../docs/evidence/b300-results.md)。
