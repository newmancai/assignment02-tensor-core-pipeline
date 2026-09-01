# 2.3 · Swizzle

128B 模式的核心映射：

```text
chunk' = (colByte >> 4) XOR (row & 7)
offset = row*128 + chunk'*16 + (colByte & 15)
```

XOR 对固定 row 是自逆置换，因此不会让两个原始 chunk 映到同一地址；它同时
把连续行的 chunk 分散到不同 bank group，降低规则步长导致的冲突。

`03_swizzle.cu` 的 128B、64B、32B 三种模式均判测 `PASS`。

