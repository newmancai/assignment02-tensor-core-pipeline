# M1 · Fragment 与 `mma.sync`

| 小题 | 内容 | 状态 |
|---|---|---|
| 1.1 | FP8 fragment 映射 | PASS |
| 1.2 | 修复下半 fragment 取数错误 | PASS |
| 1.3 | 手写单 tile FP8 MMA | 5 seeds PASS |
| 1.4 | 手工装载与 `ldmatrix` 对照 | PASS |
| 1.5 | stride、wavefront 与 bank conflict | B300 实测完成 |

各 CUDA 文件中的 `../common.h` 指向本目录的公共头文件。

