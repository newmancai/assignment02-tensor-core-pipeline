# 3.3 · mbarrier phase 调试

标准化实验元数据见 [`EXPERIMENT.md`](EXPERIMENT.md)。

状态：修复、错误版复现和 B300 判测均已完成。

实现文件：[`03_bug_mbarrier.cu`](03_bug_mbarrier.cu)

## 问题根因

程序每轮执行：

`mma -> commit(mbarrier arrive) -> wait -> tcgen05.ld`

mbarrier 是可复用的 generation barrier。一次 generation 完成后 phase
翻转，并把 pending arrival count 重置为初始化值。因此正确等待序列不是
固定 phase 0，而是：

`0, 1, 0, 1, ...`

修复代码使用：

```cpp
mbar_wait(mbar_addr, static_cast<uint32_t>(round & 1));
```

源码还提供 `BUGGY_PHASE` 宏，定义后恢复“每轮固定等待 phase 0”的错误，
用于复现实验。

## barrier 状态变化

正确版本：

```text
init                 phase=0  pending=1
round 0 commit done  phase=0  pending:1->0  => next phase=1, pending=1
round 1 commit done  phase=1  pending:1->0  => next phase=0, pending=1
round 2 commit done  phase=0  pending:1->0  => next phase=1, pending=1
round 3 commit done  phase=1  pending:1->0  => next phase=0, pending=1
```

错误版本在 round 1 仍等待 phase 0。此时 phase 0 已属于上一 generation，
等待可能针对错误代际提前返回，也可能无法等到期望状态。若提前返回，
`tcgen05.ld` 会与尚未完成的 MMA 竞争；若不能返回，程序挂死。

## B300 现象

错误版本（`-DBUGGY_PHASE`，seed=42，单进程 20 秒超时）：

| rounds | 现象 |
|---:|---|
| 1 | PASS |
| 2 | 失败或超时 |
| 4 | 失败或超时 |

```text
PASS seed=42
BUGGY_R2_FAILED_OR_TIMEOUT
BUGGY_R4_FAILED_OR_TIMEOUT
```

修复版本：

```text
rounds=1 PASS seed=42
rounds=1 PASS seed=7
rounds=2 PASS seed=42
rounds=2 PASS seed=7
rounds=4 PASS seed=42
rounds=4 PASS seed=7
JUDGE: PASS
```

## 结论

barrier parity 描述的是某个 generation，而不是一个可以永久等待的固定
完成位。每次复用都必须推进 phase。并且只有等待当前 MMA batch 的 commit
arrival 后，才能执行依赖 TMEM 结果的 `tcgen05.ld`。

## 复现

```bash
nvcc -O3 -std=c++17 \
  -gencode arch=compute_100f,code=sm_100f \
  03_bug_mbarrier.cu -o /tmp/m33

nvcc -O3 -std=c++17 -DBUGGY_PHASE \
  -gencode arch=compute_100f,code=sm_100f \
  03_bug_mbarrier.cu -o /tmp/m33-bug
```

完整原始输出见 [B300 实验归档](../../docs/evidence/b300-results.md)。
