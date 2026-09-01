# 3.1 · tcgen05 与 TMEM 概念题

状态：已完成。

## (a) tcgen05.ld 的 lane 可见范围

结论：正确。

TMEM 逻辑地址的高 16 bit 表示 lane 偏移，低 16 bit 表示 column 偏移。
一次 `tcgen05.ld` 由一个 warp 执行，32 个线程分别接收对应的 32 条
TMEM lane。m128 输出需要四个 warp 分别从
`taddr + ((warp * 32) << 16)` 开始读取自己的 32 行。读取指令之后还必须
执行 `tcgen05.wait::ld`，之后寄存器结果才可使用。

## (b) tcgen05.mma 的发射方式

结论：正确。

`mma.sync` 由一个 warp 协作完成，WGMMA 由一个 warpgroup 协作完成；
`tcgen05.mma` 则由收敛 warp 中的 elected lane 发射，硬件异步执行。
输入通过 shared-memory descriptor 描述，FP32 accumulator 保存在 TMEM。
发射线程返回不表示计算已经完成。

## (c) TMEM 结果能否直接通过 TMA 写回

结论：错误。

TMA 支持 global/shared memory 间的异步 tensor 搬运，但不能直接读取
TMEM。典型写回路径是：

`TMEM -> tcgen05.ld -> registers -> global memory`

因此 epilogue 仍需要先把 accumulator 读到各 warp 的寄存器。

## (d) TMEM 容量计算

结论：正确。

一个 SM 的 TMEM 总容量为：

`128 lane × 512 column × 4 B = 262144 B = 256 KiB`

m128n256 FP32 accumulator 占用：

`128 × 256 × 4 B = 131072 B = 128 KiB`

因此它使用 256 个 TMEM column，恰好是全部 512 column 的一半。

## (e) tcgen05.commit 是否阻塞

结论：错误。

`tcgen05.commit...mbarrier::arrive` 将此前发射的 MMA 组成完成批次，并在
该批次真正完成时到达 mbarrier；commit 指令本身不会阻塞到全部 MMA 完成。
只有等待对应 barrier generation 成功后，才可以安全执行 `tcgen05.ld`
或覆盖 MMA 仍可能读取的 shared-memory tile。

## 小结

tcgen05 的核心变化是把 accumulator 从线程寄存器移到 TMEM，并把
`mma -> commit -> mbarrier wait -> ld -> wait::ld` 组成显式的异步完成链。
后续 3.2–4.3 的同步均建立在这条链上。
