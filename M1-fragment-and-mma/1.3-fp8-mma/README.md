# 1.3 · 单 tile FP8 MMA

`03_mma_fp8.cu` 手工按 1.1 的映射装载 E4M3 byte，打包为 MMA 需要的 b32，
执行 `m16n8k32.row.col.f32.e4m3.e4m3.f32`。

判测：

```text
PASS seed=1
PASS seed=7
PASS seed=42
PASS seed=1234
PASS seed=99999
JUDGE: PASS
```

`judge_mma_fp8.sh` 用于运行多 seed 判测。

