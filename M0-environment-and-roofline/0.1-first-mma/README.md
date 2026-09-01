# 0.1 · 最小 Tensor Core 程序

## 文件

- `01_first_mma.cu`：最小 MMA kernel 与 host 正确性检查。
- `query_device.cu`：查询设备 compute capability、SM 数、时钟和总线参数。

## B300 结果

```text
D[0][0]=2 D[0][7]=2 D[15][0]=2 D[15][7]=2
PASS
```

用 `compute_120a/sm_120a` 编译后在 B300 上运行会得到
`cudaErrorNoKernelImageForDevice`。原因是 fatbin 中没有 B300 可执行的兼容
SASS，也没有可供它 JIT 的兼容 PTX。

