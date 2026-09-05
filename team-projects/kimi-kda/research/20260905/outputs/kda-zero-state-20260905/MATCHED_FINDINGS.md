# Job 19918：同输入、零初态对照的独立审计

## 结论

本轮把此前“是否给了非零初态”的数学混杂因素剥离了：**同一组 q/k/v/g/beta 输入，`initial_state=None` 和显式 BF16 全零初态得到 bitwise 相同的输出，但显式零初态仍明显更快；普通非零初态的时延则接近显式零初态。** 这强烈支持继续检查 state-presence specialization/初始化路径与全函数编译的交互，而不是把主要差距归因于初态数值不同。

T8192 默认 gate 下，现有 release 的 GPU Event eager 中位数为：`None` **0.548320 ms** → 复用零初态 **0.459136 ms** → 每调用创建零初态 **0.461696 ms**；普通非零初态 **0.459232 ms**。后两种零语义对照分别较真实 `None` 路径减少 **16.265% / 15.798%**。这是真实完整 wrapper 的 GPU-timeline 测量，不是 CPU 分配 wall time，也不是批准引入缓存 zero tensor 的生产结论。

唯一输入日志：[matched_19918.log](matched_19918.log#L4)。没有与 Job 19901/19903 合并采样。release SHA256 为 `34e2c68cf80de0bd24278afb035fdcd44e4a6205241dfcb9aab4ee95891bc486`，B300 SM10.3、148 SM、132644864 B L2；默认 seed 为 20260906，不能把它当作上一轮相同随机输入的直接复测。

## 1. 完成与正确性：PASS，但自比较须扣除

纯标准库汇总器针对该脚本的固定 schema 独立核验，报告 `PASS`、无缺项：

| 记录 | 实际数量 | 严格含义 |
|---|---:|---|
| `performance` | 180 | 10 shapes × 6 arms × 3 rounds，各臂每轮恰好一次 |
| 每行 timing scope | 3 | eager / graph / cache_perturbed，每轮 count 分别 60 / 60 / 30 |
| `correctness` | 50 PASS | 40 个非自比较 + 10 个 `legacy_none` reference 自比较 |
| `post_timing_correctness` | 50 PASS | 40 个非自比较 + 同样 10 个 `legacy_none` 自比较；日志未显式打 self 标签，按名字和代码识别 |
| `nonzero_correctness` | 10 PASS | 各 case 的非零初态结果，对同输入同非零初态的旧 V128 reference |
| `shape_complete` | 10 | case/shape 均准确且唯一 |
| `matched_complete` | 1 | `shapes=10`，位于主实验所有检查及计时之后 |

全部应有的 `out` / `final_state` 检查均为 bitwise=true、finite=true；无 final-state case 只检查 `out`。总共 110 行检查里，20 行是 reference 自比较；其余 90 行也不是 90 个互相独立的输入点，而包含同一批输入的计时前后复验。终止标记在 [matched_19918.log:305](matched_19918.log#L305)。

代码每个 case 只调用一次 `make_case`，各 arm 的 buffers 对公共 inputs 浅拷贝，q/k/v/g/beta 等对象保持共享；每个 arm 单独分配 out/final_state。`release_zero_each` 确实在每次调用闭包内执行 `torch.zeros_like(prototype)` 并写回实际传给 wrapper 的 `b['initial_state']`，不是误将复用 arm 标成动态创建。见 [matched_probe.py:32](matched_probe.py#L32)、[动态零状态闭包](matched_probe.py#L45)。

零语义路径的 reference 是 **旧 auto 的 `legacy_none`**，不是强制旧 V128。非零路径才显式使用旧 V128 reference：[matched_probe.py:67](matched_probe.py#L67)。这足以验证当前跨路径 parity，不能写成“本轮所有路径均对旧 V128 验证”。非零路径没有单独的 post-timing 检查；零语义各路径有。

## 2. `state_mode` 不是每个 arm 的实际模板状态

脚本保留 `make_case` 的原始 meta，然后有意替换每个 arm 的 initial state。必须按下面的实际调用理解，不能按 metadata 把六臂都称作相同 `HasStateIn`：

| Arm | 实际 initial state | meta=`both` 时的实际合约 | meta=`none` 时的实际合约 |
|---|---|---|---|
| release_none / legacy_none | None | 无初态，有终态输出（out-only state contract） | 无初态，无终态输出 |
| release_zero / legacy_zero | 预分配 BF16 +0 | 零初态 + 终态输出 | 零初态、无终态输出（initial-only） |
| release_zero_each | 每调用创建 BF16 +0 | 零初态 + 终态输出 | 零初态、无终态输出（initial-only） |
| release_nonzero | 预分配 BF16 非零 state | 非零初态 + 终态输出 | 非零初态、无终态输出（initial-only） |

meta=`none` 的最后一个 case 是刻意的数学等价对照，不是日志或 API 正确性错误。零语义臂要求一致；非零 arm 只与自身非零 reference 比较。

## 3. T8192 默认 gate：精确命名基线

下表是每臂三轮中位数再取中位数，单位 ms：

| Arm | Eager GPU Event | Graph replay | Pre-call cache perturbation |
|---|---:|---:|---:|
| legacy_none | 0.601760 | 0.598736 | 0.603152 |
| legacy_zero | 0.569984 | 0.566848 | 0.571392 |
| release_none | 0.548320 | 0.544384 | 0.549888 |
| release_zero | 0.459136 | 0.456320 | 0.459856 |
| release_zero_each | 0.461696 | 0.457280 | 0.464304 |
| release_nonzero | 0.459232 | 0.456288 | 0.459776 |

| 对照 | Eager 减少 | Graph 减少 | Perturbed 减少 |
|---|---:|---:|---:|
| release_zero vs release_none | 16.265% | 16.177% | 16.373% |
| release_zero_each vs release_none | 15.798% | 16.000% | 15.564% |
| release_none vs legacy_none | 8.881% | 9.078% | 8.831% |
| legacy_zero vs legacy_none | 5.281% | 5.326% | 5.266% |

在同一 T8192 case，nonzero−zero 的 eager / graph / perturbed 差值仅 **+0.096 / −0.032 / −0.080 µs**。有无随机非零初态不是这里 ~89 µs 差距的必要条件。即使在旧 binary，显式零初态也已快约 5%；release 的差距扩大到约 16%，说明 state-presence 的编译/初始化差异并非 P4 之前完全不存在。

## 4. 全部 shapes 与反例范围

每格为相对 **release_none** 的百分比减少；顺序均为 **eager / graph / perturbation**。`gate=random` 表示代码默认随机 g；gate=-8 是独立 case，case 内六臂共享输入，但不是与 random case 只改 gate 的完全配对试验。

| Case | Shape | 复用零初态 | 每调用创建零初态 |
|---:|---|---:|---:|
| 0 | fixed T2048, random gate, final-state on | 13.846 / 15.534 / 14.244 | 12.388 / 14.100 / 11.237 |
| 1 | fixed T2048, gate=-8, final-state on | 13.894 / 15.512 / 14.701 | 12.491 / 14.100 / 11.385 |
| 2 | fixed T4096, random gate, final-state on | 15.854 / 15.277 / 15.835 | 14.425 / 15.254 / 14.075 |
| 3 | fixed T4096, gate=-8, final-state on | 15.432 / 15.361 / 15.675 | 14.426 / 15.349 / 14.062 |
| 4 | fixed T8192, random gate, final-state on | 16.265 / 16.177 / 16.373 | 15.798 / 16.000 / 15.564 |
| 5 | fixed T8192, gate=-8, final-state on | 16.498 / 16.552 / 16.398 | 15.684 / 16.176 / 15.490 |
| 6 | packed single T8192, final-state on | 16.481 / 16.802 / 16.254 | 15.799 / 16.245 / 15.514 |
| 7 | fixed tail T2049, final-state on | 15.022 / 14.119 / 14.764 | 13.616 / 12.795 / 12.132 |
| 8 | fixed tail T8191, final-state on | 16.369 / 16.479 / 16.286 | 15.629 / 16.104 / 15.383 |
| 9 | fixed T8192, final-state off | 16.591 / 16.860 / 16.535 | 15.859 / 16.485 / 15.807 |

复用零初态在所有 shape/channel 中改善 13.846–16.860%；每调用创建零初态改善 11.237–16.485%。这只是观测 extrema，不是置信区间。后者最差的 **单个 paired round** 仍改善 **10.909%**（case 0 perturbation）。脚本保留所有 3 轮以及最坏退化，不把 p10/p90 当 CI。

但“zero 与 nonzero 完全一样快”仍不能绝对化。例如 case 0 graph，nonzero 比 zero 慢 1.952 µs；case 5 graph 慢 2.016 µs；case 7 perturbation 慢 0.880 µs。它们远小于主要 gap，却说明存在剩余波动/值或地址缓存因素，尚未执行协议中的同地址内容交叉对照。

固定默认 gate 的 `release_none − release_zero` eager gap 在 T2048/4096/8192 为 **20.512 / 44.912 / 89.184 µs**。同输入下仍近似随长度增长，比仅比较不同数学初态更有力地支持稳态 specialization 机制。具体因果份额仍未被单独隔离。

## 5. 不可误报的成本与归因

`zero_each − zero_reuse` 是 **组合调用差值**。T8192 默认 gate 为 eager **2.560 µs**、graph **0.960 µs**、perturbed **4.448 µs**。它含分配请求/填零/缓存与调度路径变化，不能被命名为 CPU allocation wall cost，也不能被命名为单个 fill kernel 时长。一个明显反例是 T4096 graph 差值仅 **0.064 µs**，gate=-8 为 **0.032 µs**；不要据此宣称填零 kernel 只需几十纳秒。

Graph 捕获的是完整调用闭包，因此动态零状态 arm 的 fill 在 capture 中，但 replay 不执行每次 eager 的 Python 分配代码。缓存扰动发生在 start event 前；后续 zero fill 和 K1 会改变缓存状态。它不是 K2 冷缓存证明。计时实现：[matched_probe.py:81](matched_probe.py#L81)。

本轮无 synchronized CPU wall-time、无 zero_refill_each_call、无 state-buffer 逐调用地址/immutability snapshot、无逐 timing 行 dispatch 解释。auto 环境变量在 run script 中被清除，调用处也使用 `dispatch()` 清理 override，但每 shape 实际 K2 模板仍不能仅凭 timing 行断言。两份代表性 NCU profile 的 main-log exit 均 0，sidecar 均有匹配的 `profile_complete`；它们各为 **15-pass** 仪表化采样，且报告 uncontrolled-cache warning。其测量不得与非仪表化时延混合，也不自动覆盖所有臂/shape。

因此当前最强且准确的结论是：**相同零初态数学语义能够通过 state-present 路径达到明显更低的 wrapper GPU-timeline latency，且计入每调用零张量构造的 GPU 范围后仍有较大收益；零值本身不是慢路径的主要必要条件。** 下一步应验证真正保留 `initial_state=None` 的内核候选能否恢复这部分收益，连同编译资源/循环 SASS/同步正确性复验。不要将复用零张量改写成已经完成的生产优化，不要声称 CPU 端、冷 K2、并发 serving 或其它 H/N/FP32/packed 多序列已获同等收益。

## 6. 审计复算

```sh
python3 outputs/kda-zero-state-20260905/summarize_matched.py \
  outputs/kda-zero-state-20260905/matched_19918.log --format markdown
python3 -m unittest discover -s outputs/kda-zero-state-20260905 \
  -p 'test_summarize_matched.py' -v
```

省略 `--format markdown` 输出 JSON：每轮 count/median/p10/p90、完整八组基线对照、paired-round 差值与最坏退化、实际 state-presence 语义映射。10 个纯内存 fixture 全部通过。汇总器不读取远端、不载入 GPU 库、不写结果文件；程序正常退出不等于实验通过，应读取报告 `status`。`PASS` 仅指该 matched schema 完整且记录的 parity 通过，不表示上述所有缺失控制已经补齐，也不是生产准入结论。
