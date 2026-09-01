# 3.4 · cta_group::1 与 cta_group::2

标准化实验元数据见 [`EXPERIMENT.md`](EXPERIMENT.md)。

状态：实现、正确性与 Nsight Compute 实验已完成。

实现文件：[`04_cta_pair.cu`](04_cta_pair.cu)

## 实验配置

两种实现都计算 m256n64k64 BF16 GEMM：

- `cta_group::1`：两个独立 CTA，每个 CTA 计算 m128n64k64，并各自
  staging 完整的 64×64 B tile。
- `cta_group::2`：一个包含两个 CTA 的 cluster 发射 m256 MMA；两个 CTA
  各 staging 自己的 128×64 A tile，并各保存一半 B。

group 2 路径使用协作式 TMEM alloc/dealloc、cluster 同步和
`tcgen05.commit...multicast::cluster`，使两个 CTA 收到同一 MMA batch
的完成通知。

## (a) shared memory 与 TMEM 预测

每 CTA 的主要 tile 容量：

| 实现 | A | B | A+B |
|---|---:|---:|---:|
| group 1 | 16 KiB | 8 KiB | 24 KiB |
| group 2 | 16 KiB | 4 KiB | 20 KiB |

因此 group 2 的 B shared memory 是 group 1 的一半；但 A 不变，所以
A+B 总容量只减少约 1/6。两种方案每个 CTA 仍对应 m128n64 的 64-column
TMEM 半区，TMEM 占用不会随 B staging 减半。

B300 实测（包含 barrier/TMEM 地址等少量静态 shared 对象）：

```text
smem/block: cta_group::1 = 24588 B, cta_group::2 = 20492 B
::1  PASS(bad=0)
::2  PASS(bad=0)
```

## (b) Nsight Compute shared-store 流量

指标：

`l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum`

| 实现 | shared-store wavefront | 相对 group 1 |
|---|---:|---:|
| group 1 | 778 | 100% |
| group 2 | 650 | 83.5% |

流量下降约 16.5%，与“B 减半但 A 不变”的容量分析一致。它不会下降 50%，
因为 A staging 和控制开销没有减少。

## (c) 对 M4 pipeline 的价值

group 2 每 stage、每 CTA 可节省约 4 KiB B shared memory。这部分容量可用于：

- 增加 pipeline stage 数，吸收更长的 TMA 延迟；
- 在相同 stage 数下提高 blocks/SM；
- 扩大 tile，提高数据复用。

是否转化为性能收益取决于原 kernel 是否正好受 shared-memory occupancy
边界限制。若瓶颈已经是 Tensor Core、TMEM 或 cluster 同步，收益会变小。

## (d) 硬件机制

`cta_group::2` 依赖 thread-block cluster、分布式 shared memory 和跨 CTA
硬件协作。数据中心 GEMM 的 tile 大、运行时间长、复用高，容易摊薄 cluster
协调成本；数据中心 GPU 也更愿意投入面积支持更强的互联、同步和片上容量。
因此这类机制更常见于 B300 等数据中心 GPU，而不一定出现在消费级 GPU。

## 时间数据的解释

单 tile 普通计时曾得到约 20–22 us，但题面明确说明差异处于噪声范围。
本题结论只依据正确性、shared-memory 容量和 NCU 流量，不用单 tile 耗时
宣称 group 2 更快或更慢。

完整原始输出见 [B300 实验归档](../../docs/evidence/b300-results.md)。
