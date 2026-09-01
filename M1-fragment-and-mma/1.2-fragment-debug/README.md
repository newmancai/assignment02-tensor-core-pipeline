# 1.2 · Fragment bug

错误版本的 `a2/a3/a6/a7` 重复读取 A 的上半 0–7 行，因此 D 的下半部分
基本复制上半输出。修复方式是把对应 row 从 `group` 改为 `group + 8`，K 坐标
不变。

修复后的 `02_bug_fragment.cu` 在 B300 上判测 `PASS`。

