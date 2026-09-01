# 1.4 · `ldmatrix`

本题同时保留手工 byte 装载与 `ldmatrix` 路径：

- A：`ldmatrix.sync.aligned.m8n8.x4.shared.b16`
- B（n-major）：`.x2`，不使用 `.trans`

三个 seed 的 manual 与 ldsm 路径均为 `PASS(0)`。PTX 中手工路径需要多次
`ld.shared.u8` 与 pack/shift/or；`ldmatrix` 路径只需一条 `.x4` 和一条 `.x2`。

