# 2.2 · SM100 descriptor

编码规则：

```text
(saddr >> 4)
| ((LBO >> 4) << 16)
| ((SBO >> 4) << 32)
| (1 << 46)
| (layout << 61)
```

三个判测场景的 `LBO/SBO/layout`：

```text
1: 128 / 1024 / 0
2:   0 / 1024 / 2
3:   0 / 1024 / 2
```

`02_descriptor.cu` 的 host 判测为 3/3 `PASS`。K-major 与 MN-major 的区别
体现在 staging 的逻辑坐标解释；在题面场景 2/3 中不改变最终 descriptor 数值。

