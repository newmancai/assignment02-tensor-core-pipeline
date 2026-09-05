# Guarded V16 StatePrefetch4：独立 NCU 复核

2026-09-05。仅离线读取已有 CSV、构建日志和 profiling 脚本；未重新运行 GPU、未修改 release。

## 结论

两份 profile 确实比较了同一 `B=1,T=8192,H=12,D=128`、BF16 state-in + state-out、非 packed 的 V16 recurrence kernel；release 符号确认启用 **StatePrefetch=4**。P4 多用了 16 个寄存器/线程，但 CTA 布局、shared memory、shared-memory 限制的驻留 CTA 上限及 achieved occupancy 基本不变。与此同时，eligible warps 和 issue efficiency 上升、short-scoreboard 与 sleeping 的每指令等待比值下降。

这与“增加寄存器中的预取距离，改善依赖隐藏及流水线进度”的解释一致；**不是 occupancy 提升，也不是这份汇总 profile 对某段代码因果贡献的证明**。本次单个 K2 的 NCU duration 下降 21.3215%，不能替代重复 CUDA-event 测量的性能分布，亦不能当作整个 forward 的加速比例。

## 对象与数据格式核验

- CSV 均为 3 行、644 列：第 1 行 header，第 2 行 unit，第 3 行唯一数据；用 `csv.reader` 解析，未对巨大 kernel name 作逗号切分。
- [release_probe.py:261](../release_probe.py#L261) 的 profile 分支仅调用一次 `make_case(8192)`；默认参数为 H12、B1、BF16、双向 state buffer、无 `cu_seqlens`。`dispatch()` 清空 force/off override，调用实际 Python wrapper。
- [run_release.sbatch:29](../run_release.sbatch#L29) 筛选 `_flash_kda_fwd_recurrence`，`--launch-count 1`；因此 CSV duration 只覆盖选中的 K2，不包括 Python、workspace allocation 或其它 kernel。
- 两份日志均记录 CC 10.3、148 SM、L2=132,644,864 B；candidate `.so` SHA-256 均为 `34e2c68cf80de0bd24278afb035fdcd44e4a6205241dfcb9aab4ee95891bc486`。

裁剪掉 CuTe 类型树之后，kernel name 的末端模板实参为：

```text
baseline: ..., 16, 128, 16, 3, 2, 96, 1, 1, 0, 0>(...)
release:  ..., 16, 128, 16, 3, 2, 96, 1, 1, 0, 0, 4>(...)
               C    D   V  IS OS  NT SI SO FP VL  P
```

其中 `C=ChunkSize, IS/OS=Input/OutputStages, NT=NumThreads, SI/SO=HasStateIn/Out, FP=StateFP32, VL=IsVarlen, P=StatePrefetch`。baseline 原模板没有 P 参数，原 schedule 为 P1；release 的最后一个 `4` 是模板实参，绝非从文件名或 Python dispatch 推断。

## 资源、发射与工作量

数值直接来自 CSV；百分数与比例不混用。变化率均按 `(release/baseline-1)` 计算。

| 指标 | baseline | release P4 | 解读 |
| --- | ---: | ---: | --- |
| Grid / CTA | (1,12,8) / 96 | (1,12,8) / 96 | 相同 |
| Block / threads | (96,1,1) / 96 | (96,1,1) / 96 | 相同；每 CTA 3 warps |
| Registers / thread | 54 | 70 | +16 |
| Allocated registers / thread | 56 | 72 | 分配粒度后的开销也 +16 |
| Dynamic shared / CTA | 48.640 Kbyte | 48.640 Kbyte | 相同 |
| Driver shared / CTA | 1.024 Kbyte | 1.024 Kbyte | 相同 |
| Static shared / CTA | 0 B | 0 B | 相同 |
| Total / allocated shared / CTA | 49.664 Kbyte | 49.664 Kbyte | 相同 |
| Shared-memory config | 200.704 Kbyte | 200.704 Kbyte | 相同 |
| Resident CTA limit：shared memory | 4 | 4 | 仍为最小资源上限 |
| Resident CTA limit：registers | 12 | 9 | 降低，但均大于 4 |
| Resident CTA limit：barrier / warp / block | 7 / 21 / 32 | 7 / 21 / 32 | 相同 |
| Waves / SM | 0.16 | 0.16 | 同一个小 grid |
| Achieved occupancy | 4.689962% | 4.687892% | 近乎相同，非改进来源 |
| Active warps / SM active cycle | 3.001575 | 3.000251 | 近乎相同 |
| Eligible warps / SMSP active cycle | 0.147231 | 0.181195 | +23.0685% |
| Issue active / peak sustained active | 14.722274% | 18.124651% | +3.402377 个百分点；相对 +23.1104% |
| Warp cycles / instruction | 6.828234 | 5.549118 | −18.7327% |
| `inst_executed` | 41,818,128 | 40,144,608 | −4.0019%；不是 FLOPs |
| `gpu__time_duration.sum` | 521.088 µs | 409.984 µs | −111.104 µs；−21.3215% |

CSV 的 `Kbyte` 为十进制单位：48.640 Kbyte = 48,640 B = 47.5 KiB；加 1,024 B driver 后总计 49,664 B = 48.5 KiB。不能写成 48.640 KiB。96 个 CTA 少于 148 个 SM；本结果不能说明高并发/大 grid 下的驻留行为或吞吐。

## Stall ratio：单位不是百分比

下表来自 `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`。CSV unit 行标记为 `inst`，但 WarpStateStats 对该图的语义明确为每条已发射指令在相应状态经历的平均 cycles。因此保留为 **cycles / issued instruction**；例如 sleeping=2.259631 不能写成 225.9631% 的耗时。[NVIDIA WarpStateStats 定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)

| Stall reason | baseline | release P4 | 相对变化 |
| --- | ---: | ---: | ---: |
| Short scoreboard | 0.901041 | 0.464868 | −48.4077% |
| Sleeping | 2.259631 | 1.291526 | −42.8435% |
| Long scoreboard | 0.933360 | 1.047803 | +12.2614% |
| Wait | 1.131349 | 1.160641 | +2.5891% |
| MIO throttle | 0.147919 | 0.112517 | 下降 |
| Barrier | 0.015224 | 0.015960 | 小幅上升 |

Short scoreboard 对应非 L1TEX 的 MIO 依赖，常见但不唯一来源是 shared memory；long scoreboard 对应 L1TEX 依赖；sleeping 表示 warp 的线程处于 blocked/yielded/sleep 状态。[NVIDIA Warp Stall Reasons](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons)

据此，本次 profile 支持“发射端等待结构改善”的关联证据，但不能仅凭 short-scoreboard 下降将全部收益归因于 Phase 6 的某条 shared load。sleeping 下降可能与 producer/consumer 等待减少一致，却未按 PC 或 warp 角色定位；`inst_executed` 下降也可能包含等待循环/编译调度差异，而非减少算法算术工作。long-scoreboard 和 wait 均上升，不能宣称全部延迟瓶颈都已改善。各 ratio 的分母随指令发射变化，不应直接相加当 wall-time 百分比，也不能乘 duration 分摊各阶段耗时。

## Spill：目标 P4 无 spill，但整个构建不是无 spill

实际 profile 的 `derived__local_spilling_requests` 两侧均为 0；`derived__local_spilling_requests_pct` 两侧均为 **`no data`**，没有证据把该列补成 0%。

[build_release.log:220](../build_release.log#L220) 的目标实例尾部为 `...ELb1ELb1ELb0ELb0ELi4EEv...`，ptxas 明确记录 `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`，70 registers、9 barriers，与 NCU 寄存器数一致。全部 8 个编译 P4 实例（state-in/out 四组合 × fixed/varlen）均为 0 stack、0 spill stores、0 spill loads；无 state-in 时为 63 registers，有 state-in 时为 70。

不过，[build_release.log:60](../build_release.log#L60) 确有 3 个 **V64 / FP32 / P1** 的 spill warning，均不在 P4 guard 内：

| State-in / out / varlen | Stack B | Spill stores B | Spill loads B |
| --- | ---: | ---: | ---: |
| 1 / 0 / 0 | 8 | 4 | 4 |
| 1 / 1 / 0 | 32 | 48 | 68 |
| 1 / 1 / 1 | 32 | 44 | 84 |

所以可陈述“P4 的额外寄存器没有引入已编译 P4 变体的 spill，profile 目标也未观察到 local spilling request”；不可陈述“整个 release spill-free”。这些 P1 warning 是否相对原构建新增，不在本次仅有 release 构建日志的证据范围内。

## 实验解释边界与后续用途

结合脚本、CSV 与日志可确认 **1 个目标 launch、16 replay passes、0 warmup passes**；两份日志均明确警告 cache 与 clock 未由 NCU 控制。16 passes 是采集不同指标的重放，不是 16 个统计样本；`--cache-control none` 不清缓存，`--clock-control none` 不锁 GPU 时钟，不能据此认定外部已锁频或各 pass 的 cache 一致。[NVIDIA Replay / Cache / Clock Control](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay)

因此建议将这份 NCU 只作为重复计时结果的机制旁证：提高 issue/eligible、降低部分等待、决定理论驻留数量的 shared-memory 上限未变。不要从单个 H12T8192 fixed 双 state profile 推广到所有 T、packed、无 state-in、FP32、大 batch 或其它 GPU。若后续需要建立更强因果链，应在保持输入/时钟/缓存协议一致的前提下定位 Phase 6 的 SASS/PC stall 与指令混合；本复核未运行这些追加实验。

## 原始证据指纹

| 文件 | SHA-256 |
| --- | --- |
| [baseline CSV](../release_19901_baseline_ncu.csv) | `3ca5f1a193a187d283f1c96bc6e83fe8fd2ffb562627859ca88fefae6e1464ab` |
| [release CSV](../release_19901_release_ncu.csv) | `08cc58e97812f55da515eaabb4eb19b0b3dade517c7b23b20831dd4bb8b152fb` |
| [build log](../build_release.log) | `1cd14cbc65cf4464fd503c6757c6297a8f3205b6da432c08d9ff79a5caad46e6` |

补充 provenance：[baseline NCU log](../release_19901_baseline_ncu.log)、[release NCU log](../release_19901_release_ncu.log)。
