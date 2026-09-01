# 3.2 · `tcgen05` 单 tile GEMM

状态：待 B 完成。`02_single_tile.cu` 仍包含“按七步实现”的 TODO。

预期产物：完整 kernel、`judge_tile.sh` 的 PASS 结果、七步执行顺序，以及移除
`fence.proxy.async` 后的可复现实验现象。

