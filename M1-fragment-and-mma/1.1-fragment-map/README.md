# 1.1 · FP8 fragment 映射

令 `group=lane>>2`、`tig=lane&3`，寄存器内 byte 序号为 `i`：

```text
A.row = group + 8*((i>>2)&1)
A.col = 4*tig + (i&3) + 16*(i>>3)
B.k   = 4*tig + (i&3) + 16*(i>>2)
B.n   = group
```

`01_fragment_map.cu` 在 host 上穷举并检查所有映射，结果为 `PASS`。

