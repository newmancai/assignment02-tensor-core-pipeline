# 3.2 · tcgen05 单 tile GEMM

标准化实验元数据见 [`EXPERIMENT.md`](EXPERIMENT.md)。

状态：代码与 B300 正确性实验已完成。

实现文件：[`02_single_tile.cu`](02_single_tile.cu)

## 目标

由一个 128-thread CTA 计算 m128n64k64 BF16 GEMM，使用 FP32 累加和
`cta_group::1`。数据路径为：

`global -> K-major/128B-swizzled smem -> tcgen05.mma -> TMEM -> tcgen05.ld -> global`

## 七步实现

1. lane 0 初始化 arrival count 为 1 的 mbarrier；warp 0 协作分配 64 个
   TMEM column。
2. 128 个线程把 A(128×64) 和 B(64×64，按 N×K 存储)搬入 shared memory，
   使用 128B swizzle。
3. 执行 `fence.proxy.async.shared::cta`，再同步全 CTA。
4. warp 0 的 elected lane 发射四条 m128n64k16 MMA；第一条
   `scale_c=0`，后三条累加。
5. commit 到 mbarrier，并等待 phase 0 完成。
6. 四个 warp 各读取 32 条 TMEM lane，每次读取 8 个 FP32 column，执行
   `tcgen05.wait::ld` 后写回 global。
7. 全 CTA 同步，确认所有 warp 都完成读取后释放 TMEM。

## descriptor 与 swizzle

SM100 descriptor 使用 version=1、layout=2（128B swizzle），SBO=1024 B。
物理地址以 8 行 × 128 B 为 atom：

`atom*1024 + row_in_atom*128 + ((chunk16B xor row_in_atom)*16) + byte_in_chunk`

这与 2.3 的 `swizzle_128B` 一致，也是本题在真实 tcgen05 指令上的硬件
验证。

## B300 正确性

```text
PASS seed=1
PASS seed=7
PASS seed=42
PASS seed=1234
PASS seed=99999
JUDGE: PASS
```

## 移除 proxy fence 的实验

源码提供 `OMIT_PROXY_FENCE` 宏。使用该宏编译后，本次 B300 上五个 seed
仍全部 PASS：

```text
PASS seed=1
PASS seed=7
PASS seed=42
PASS seed=1234
PASS seed=99999
```

结论不是“fence 可以删除”。普通 shared store 属于 generic proxy，
tcgen05 通过 async proxy 消费 shared memory；缺少 proxy fence 时，内存
模型没有建立两者之间的可见性与顺序保证。本次没有复现错误只说明该具体
硬件时序没有暴露问题，属于未定义排序下的偶然正确现象。

## 编译与运行

```bash
nvcc -O3 -std=c++17 \
  -gencode arch=compute_100f,code=sm_100f \
  02_single_tile.cu -o /tmp/m32
/tmp/m32 42
```

反事实实验：

```bash
nvcc -O3 -std=c++17 -DOMIT_PROXY_FENCE \
  -gencode arch=compute_100f,code=sm_100f \
  02_single_tile.cu -o /tmp/m32-no-fence
```

完整原始输出见 [B300 实验归档](../../docs/evidence/b300-results.md)。
