# Prob 6.1：TileLang lowering 对照

## 实验配置

- TileLang：`0.1.13`（题目 `pyproject.toml` 固定版本）
- GEMM：`1024×1024×1024`，FP16 输入、FP32 累加
- tile：`BM=128, BN=128, BK=64`
- `threads=128`，`num_stages=3`
- 只编译与保存 lowering，不要求在 H100 上运行

完整编译验证：

```text
COMPILE_SM90A_PASS 4333
COMPILE_SM100A_PASS 5207
```

## 对照表

| 项目 | sm_90a | sm_100a |
|---|---|---|
| 选中的 Tensor Core 指令 | `wgmma.m64n128k16`（生成 C++ 为 `tl::wgmma_ss`） | 当前固定版本实际回退到 `mma.sync.m16n8k16`，并用 `ldmatrix.x4/.x4.trans` 供数；没有选中 tcgen05 |
| Tensor Core descriptor | lowering 在 kernel 内创建 `GmmaDescriptor desc_a/desc_b`，调用 `initialize_wgmma_descriptor` | 因使用 `mma.sync`，操作数进入寄存器 fragment，不生成 tcgen05 smem descriptor |
| TMA descriptor | A/B 作为 `CUtensorMap A_desc/B_desc` kernel 参数，由 TileLang host/runtime 构造并传入 | 同左 |
| smem swizzle 布局 | layout inference / lowering 根据目标、tile 和 TMA/WGMMA 约束确定，并编码进 TensorMap 与地址映射 | layout inference / lowering 为 TMA staging 与 `ldmatrix` 访问确定布局 |
| 数据搬入 smem | 编译器生成 `tl::tma_load`、三 stage mbarrier 与 producer/consumer warp specialization | 同左 |

### 关于 sm_100a 的版本结论

题面要求根据固定版本核实 sm100 codegen 支持范围。本实验同时尝试了：

```text
{"kind":"cuda", "arch":"sm_100a"}
{"kind":"cuda", "arch":"sm_100f", "code":["sm_100a"]}
```

两种写法都能完成 CUDA cubin 编译，但 TileLang 0.1.13 对这个普通 FP16
`T.gemm` 都选择 `mma.sync + ldmatrix`，没有自动 lower 到 tcgen05/TMEM。
因此不能在报告中把“理论上 sm100 支持 tcgen05”写成“本次 lowering 已经生成
tcgen05”；生成源码才是本题应记录的真实证据。

## 与 M2–M4 手写实现对照

### DSL 自动完成

- 根据 target 与 dtype 选择可用的 Tensor Core 指令族；
- 推导每个 lane 的 fragment 布局和 `ldmatrix` 地址；
- 为 WGMMA 构造 smem matrix descriptor；
- 决定 shared-memory staging 布局和对齐；
- 生成 TMA TensorMap 参数、`tma_load`、mbarrier phase；
- 把三阶段 pipeline 展开成 producer/consumer warp specialization；
- 插入等待、到达、commit 和 fence。

### 程序员仍然决定

- `BM/BN/BK` tile 大小；
- `threads` 与 `num_stages`；
- dtype、累加精度、矩阵转置关系；
- grid 映射、问题形状以及必要时的 GEMM policy；
- 是否需要针对目标架构改配置并进行 benchmark/autotune。

## “谁负责”表新增行

| 工作 | CUDA 手写 | TileLang |
|---|---|---|
| Tensor Core 指令选择与供数布局 | 程序员选择 `mma/wgmma/tcgen05`，手推 fragment/descriptor/swizzle，并写同步与搬运 | 编译器 lowering 根据 target、dtype、tile 自动选择并生成；程序员通过 target、tile、threads、stages 间接约束 |

## 生成文件

```text
m6_lowering_output/sm_90a_generated.cu
m6_lowering_output/sm_90a_lowered_tir.txt
m6_lowering_output/sm_90a_key_lines.txt
m6_lowering_output/sm_100a_generated.cu
m6_lowering_output/sm_100a_lowered_tir.txt
m6_lowering_output/sm_100a_key_lines.txt
```

脚本：`kernels/m6_tilelang_lowering.py`

参考：[TileLang instructions / T.gemm](https://github.com/tile-ai/tilelang/blob/main/docs/programming_guides/instructions.md)、
[TileLang 0.1.13 target 文档](https://www.tilelang.com/get_started/targets.html)。
