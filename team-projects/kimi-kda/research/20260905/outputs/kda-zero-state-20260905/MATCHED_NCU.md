# Matched zero-state：NCU 与 SASS 的独立交叉复核

2026-09-05。仅读取本地 Job 19918 CSV/log、实验脚本及初始化消融编译日志；未运行 GPU、未改代码。下文 `none → zero` 始终指 `initial_state=None → 预先创建的 BF16 全零 initial_state`，不是零状态 → 非零状态。

## 结论

本次相同零初态数学语义下，有 state buffer 的 K2 profile 仍更快：**499.104 → 409.408 µs，下降 17.9714%**。主要同向变化是 issue/eligible 提升、short-scoreboard 与 sleeping 的每指令等待下降；shared memory、SMEM 驻留上限和 achieved occupancy 基本不变。与此前 [SASS 复核](SASS_REVIEW.md) 中的循环不变量保留、Phase 1 load/use 间隔差异方向一致。

这使“只是先前比较了非零与零数学初态”的解释不足以独立解释现象。仍不能区分全部收益究竟来自 steady-state 指令调度、初始化 TMA vs shared 清零、流水线起步或 cache/address 行为；没有 PC/warp 角色定位证据，不能声称已经证明 Phase 1 导致全部收益。

## 1. 对照是否真正匹配

- 两份 wide CSV 各 3 行、607 列，按 header / unit / 唯一 data 解析。两份 NCU log 均记录同一 release SHA-256：`34e2c68cf80de0bd24278afb035fdcd44e4a6205241dfcb9aab4ee95891bc486`，与先前 SASS 二进制一致。
- [matched_probe.py](matched_probe.py) 的 profile 分支固定 T8192，默认 B1/H12/D128、BF16 state-out、非 packed。每个独立 profile 进程使用 seed `20260906`，以相同顺序生成一次 q/k/v/g/beta/原始 state，然后都创建 `zeros_like(state)`；只有传入的 state presence 不同。这里是脚本约束下跨进程重建的匹配数据，并非两个 profile 共享对象地址；日志没有逐输入 tensor hash。
- 同进程常规实验确实让各 arm 共用输入对象。`matched_19918.log` 有 10 个 shape complete、`matched_complete(shapes=10)`、50 条初次 correctness、50 条 post-timing correctness、10 条 nonzero correctness，状态均 PASS。每组 50 条包含 10 条 legacy-none reference 自检，不能算作 50 个独立候选测试。profile 分支本身提前返回，不做逐位 compare；常规实验的通过不是这两个 profile 数据的现场逐位证据。
- `--launch-count 1` 筛选 recurrence，只计 K2，不包括建零 tensor、wrapper、K1 或整个 forward。两份 profile 退出码均为 0。

裁剪掉类型树后的 kernel 尾参数确认精确目标：

```text
release_none: ..., 16,128,16,3,2,96,0,1,0,0,4>(...)
release_zero: ..., 16,128,16,3,2,96,1,1,0,0,4>(...)
                                  SI SO FP VL P
```

所以两者都是 V16/P4；不同参数确实是 `HasStateIn`，而非 ValueSlice、FP32、packed 或 Prefetch 路径切换。

## 2. 资源与发射计数器

| 指标 | none | zero | 解释 |
| --- | ---: | ---: | --- |
| Grid / block | (1,12,8) / (96,1,1) | 相同 | 都是 96 CTA、每 CTA 3 warps |
| Registers / thread | 63 | 70 | 与两实例 SASS metadata 一致 |
| Allocated registers / thread | 64 | 72 | 分配开销增加 8 |
| Register-limited CTA / SM | 10 | 9 | 降低，但仍不决定总驻留上限 |
| Shared-memory-limited CTA / SM | 4 | 4 | 相同 |
| Dynamic / driver shared | 48.640 / 1.024 Kbyte | 相同 | 总计 49.664 Kbyte，即 48.5 KiB |
| Waves / SM | 0.16 | 0.16 | 相同 |
| Achieved occupancy | 4.685466% | 4.687828% | 差 0.002362 个百分点，基本不变 |
| Active warps / SM active cycle | 2.998698 | 3.000210 | 基本不变 |
| Eligible warps / SMSP active cycle | 0.156311 | 0.181120 | +15.8716% |
| Issue active / peak sustained active | 15.632103% | 18.110458% | +2.478355 个百分点；相对 +15.8543% |
| Warp cycles / instruction | 6.426305 | 5.552461 | −13.5979% |
| K2 duration | 499.104 µs | 409.408 µs | −89.696 µs；−17.9714% |

寄存器更多却不降低当前 shared-memory 限制的理论 CTA 上限，且 achieved occupancy 并未提升；因此这次不能把收益称为 occupancy 优化。96 CTA 的小 grid 也不支持对高并发驻留行为作外推。

## 3. Stall：short-scoreboard 与 wait 是不同指标

以下全部取自 `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`，是 **cycles / issued instruction**，不是 kernel wall-time 百分比；CSV unit 行虽为 `inst`，不能将 2.071388 写成 207.1% 耗时。[NVIDIA WarpStateStats 定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)

| Reason | none | zero | 相对变化 |
| --- | ---: | ---: | ---: |
| Short scoreboard | 0.713769 | 0.465810 | −34.7394% |
| Sleeping | 2.071388 | 1.288930 | −37.7746% |
| Wait | 1.182506 | 1.161428 | −1.7825% |
| Long scoreboard | 0.884271 | 1.049505 | +18.6859% |
| MIO throttle | 0.139649 | 0.112643 | −19.3385% |
| Barrier | 0.013432 | 0.015969 | +18.8877%，但绝对差仅 0.002537 |

与 SASS 的交叉解读：

- SASS 已定位 none 每 tile 重取 TID/CTA、重算部分不变量，并存在 state LDSM 后立即消费 HMMA 的 Phase 1 片段。当前 issue/eligible 增加及 short-scoreboard 下降，与较好地隐藏依赖延迟相容；**汇总值不能把等待精确分给那个 PC**，更不能从此推出具体节省多少 cycles。
- Sleeping 大幅下降，与流水线中等待者减少的解释相容，但没有 compute/load/store warp 的分项或 PC 证据，不把它称为某一个 barrier 的测量。
- 独立 `wait` 仅小幅下降，而 long-scoreboard 明显上升。因而不能写“所有依赖等待都改善”；有 state buffer 额外读取初态也是改变了的条件，不能据此断言 long-scoreboard 增量只来自那次读取。
- 分母与发射指令数量有关。这些 ratio 的百分比变化不能当耗时份额相加，也不能乘 duration 推出各阶段时间。

## 4. 本轮缺失的字段不可从旧 profile 补齐

本轮没有 `inst_executed` 原始总数、`derived__local_spilling_requests` 或其百分比列。仅有 `sm__inst_executed.*.pct_of_peak_sustained_elapsed` 等归一化指标，不能反称动态指令数。因此既不能在此报告“指令减少了 X 条”，也不能把 Job 19901 的 41,818,128 / 40,144,608 挪进这个 matched contrast。

无 spill 的已有支持来自同 SHA 二进制的 SASS resource `STACK/LOCAL=0`、无 LDL/STL，以及相应 ptxas 0 spill stores/loads；不是这两份 CSV 新测出的 local-spill counter。

两份本轮 CSV 都是 **15 replay passes、0 warmup passes**，不是上一轮的 16 passes。它们来自一次目标 launch，为多指标采集而重放；不是 15 次独立统计试验。脚本为 `--cache-control none --clock-control none`，日志也有两项不一致性 warning。NCU 单点 duration 不能代替交错重复 eager/graph/cache-perturbed 计时，也不能混合 Job 19901 与 19918 的数值计算新的样本分布。

## 5. Job 19920 初始化消融：目前只能记下编译提示

已读 [experiment.patch](experiment.patch) 与 [build_experiment.log](build_experiment.log)。该独立实验扩展为 `flash_kda_zero_C`，加入 `InitStrategy` 和 runtime `load_initial_state`；不是上面被 profile 的原 release 二进制。以下固定 D128/V16/P4、state-out=true、BF16、非 packed：

| InitStrategy / 对照 | 编译 `HasStateIn` | Registers | Stack / spill stores / spill loads |
| --- | ---: | ---: | --- |
| 0：原 scalar zero 路径 | false | 63 | 0 / 0 / 0 |
| 1：all-thread uint4 vector zero | false | 63 | 0 / 0 / 0 |
| 2：one-warp uint4 vector zero | false | 63 | 0 / 0 / 0 |
| 3：unified runtime init | true | 63 | 0 / 0 / 0 |
| 0：有初态静态对照 | true | 70 | 0 / 0 / 0 |

证据位置分别为 build log 157、167、162、257、252 行的 function properties；末模板顺序为 `...SI,SO,FP,VL,P4,InitStrategy`。策略 3 的模板 SI=true 不等于这次调用真的有初态：launch 固定编译 SI=true，实际 None 调用传 runtime `load_initial_state=false`，在同一 kernel 内清零。

重要静态提示：**把模板 SI 固定为 true 没有自动继承原有初态实例的 70-register 编译结果**；unified 实例为 63。但寄存器数一样也不保证 SASS 排序一样，寄存器数少更不证明性能会差。vector/one-warp 的初始化、分支、同步或 steady schedule 是否改变，必须等实际 binary/SASS 和匹配 runtime 结果。此表不是策略排名或性能预测，未据编译日志宣告 Job 19920 通过。

## 原始证据

| 文件 | SHA-256 |
| --- | --- |
| [none NCU CSV](matched_19918_release_none_ncu.csv) | `9f9293ad3d09bdfc07bc231e48e210df0c6f5407df1c54fbf7f1c5f31c777d73` |
| [zero NCU CSV](matched_19918_release_zero_ncu.csv) | `682471510f05c5d58261c200e540dee253cfcfdbacff635bb5ba6ec3e670eac1` |
| [消融 build log](build_experiment.log) | `7fe89b6e5cd4363532e1a9498c2dcda1ab93dcd5174adb0adad4f18e0404e3dc` |

补充：[none profile log](matched_19918_release_none_ncu.log)、[zero profile log](matched_19918_release_zero_ncu.log)、[执行脚本](run_matched.sbatch)、[同二进制 SASS 证据](sass_targets.json)。
