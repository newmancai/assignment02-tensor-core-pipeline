# 6.1 · TileLang lowering

## 文件

- `m6_tilelang_lowering.py`：构造 kernel 并导出 lowered TIR/CUDA。
- `analysis.md`：WGMMA/TMA/pipeline 对照与职责划分。
- `generated/`：`sm_90a`、`sm_100a` 的 TIR、CUDA 和关键行。

## 实测结论

- `sm_90a`：生成 WGMMA、TMA、GmmaDescriptor、mbarrier 和三级流水。
- TileLang 0.1.13 下，本实验的普通 FP16 `T.gemm` 即使目标为 `sm_100a`
  仍选择 `mma.sync.m16n8k16 + ldmatrix`，未出现 `tcgen05`。
- 两个 target 的完整编译均成功：`COMPILE_SM90A_PASS`、
  `COMPILE_SM100A_PASS`。

这项结果是固定版本的实测行为，不应泛化为所有 TileLang 版本或所有算子。

