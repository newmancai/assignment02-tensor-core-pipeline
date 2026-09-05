# FlashKDA 性能证据独立审查（2026-09-05）

审查对象：`assignment02-github/team-projects/kimi-kda/experiments/final_campaign`、实现快照和 `docs/c1-final`。本审查未修改 kernel，未自行访问远端；结合主任务在 B300 上执行的 Job19844 两轮扩展实验更新结论。源码路径相对于 `assignment02-github/team-projects/kimi-kda`，新实验 CSV/log 与本报告位于同一目录。

## 结论

**新 B300 实测已经推翻“V128 L0 摊销后始终不如 MMA”的外推。** Job19844 两轮均在 grid12 的 inner96、grid148 的 inner128 首次测得 TCGEN 更快；inner512 时分别约为 **1.2116–1.2119×**、**1.1377×**。实际已采样的交叉区间分别是 `(64,96]`、`(96,128]`，不能从稀疏扫描声称精确交叉点。其原因与审查指出的固定成本混入 `T(64)/64` 一致，但新实验仍只是相同 operands 的 L0 probe，**不证明完整 K2 加速**；带 scalar U 重排和 state 更新的 L1 在 inner512 仍只有约 0.261×。

ValueSlice 在已测 H12 长 prefill 上的约 27% forward 降时有可信的对照证据，不能因局部 profiler 解释的不足而否定。当前合理决策是保留已验证路径，同时重新打开 TCGEN 的生命周期/数据流研究 gate，而非直接发布 TCGEN kernel。

对 dispatcher 的准确描述是“特定 B300 拓扑上的离线延迟拟合，加固定阈值和域外回退”，还不是根据实时 L2 压力、服务并发或置信区间作决策的鲁棒调度器。Cluster/multicast 是有理由探索的候选，但当前数据没有证明它比缩短每个 chunk 的等待链更值得优先投入。

### Job19845：补充的 paired graph 与调用前缓存扰动

主任务随后完成 [Job19845](followup_19845.log)，同一卡 fixed B1/T8192/H12/D128、BF16 state，三轮中位数汇总如下：

| 场景 | V128 / ms | V16 / ms | 降时 |
|---|---:|---:|---:|
| eager，预热 | 0.782784 | 0.569008 | 27.31% |
| eager，调用前写256MiB缓冲 | 0.783408 | 0.570272 | 27.21% |
| graph，预热 | 0.779904 | 0.566784 | 27.33% |
| graph，replay前写256MiB缓冲 | 0.780272 | 0.567168 | 27.31% |

这补齐了本文后文指出的“最终 campaign 缺少同口径 paired graph A/B”覆盖。auto wrapper 的 output/state 相对 V128 均 bitwise equal。缓冲写操作在计时外；它扰动的是完整调用之前的 cache，K1仍会重建/预热供K2读取的workspace，不能称为K2冷缓存测试，也没有覆盖持续并发竞争。复算脚本 [summarize_review.py](summarize_review.py) 同时读取该日志，提取三轮汇总及边界对照。

## 1. [P1，论证缺陷已证实] `inner=64` 并未隔离固定成本

定位：`tcgen05_probe/phase6_probe.cu:775–791` 用 CUDA Event 计量多个完整 launch，再除以 launch 数；`phase6_probe.cu:883–884` 将完整 kernel 时间除以 inner。报告 `docs/c1-final/FINAL_REPORT.md:150–161,282–284` 把这个归一化量作为“即使摊薄仍没有核心收益”的停止依据。数据出自 `tcgen05_probe/results/03_tcgen05_probe_17937.csv`。

对模型 `T(n)=a+b*n`，`T(64)/64=a/64+b`；两条路径不同的固定成本不会被除法消除。审查首先利用归档两个点算出以下**解释性斜率信号**，据此提出扩大扫描：

| V128 的 L0 | MMA 斜率 `(T64−T1)/63` | TCGEN 斜率 | MMA/TCGEN |
|---|---:|---:|---:|
| grid=12 | 0.499157 µs | 0.388952 µs | 1.2833× |
| grid=148 | 0.501681 µs | 0.390260 µs | 1.2855× |

grid12 的完整原数值为 MMA `6.165280→37.612159 µs`、TCGEN `16.390240→40.894241 µs`，inner 从 1 增加到 64。两点拟合的固定项分别约为 `5.6661/16.0013 µs`；TCGEN 的残余固定项显然可能掩盖更好的每轮斜率。grid148 方向一致。

原来这两个点本身不能证明线性或稳态加速。现在主任务用现成二进制完成 Job19844：V128、grid12/148、inner=1/4/16/64/96/128/256/512；每点 warmup20、iters100、repeats5、交替路径，整个 sweep 再运行两轮。原始文件为 [第一轮 CSV](tcgen_sweep_19844_1.csv)、[第二轮 CSV](tcgen_sweep_19844_2.csv)、[完整日志](review_19844.log) 和 [执行脚本](run_review.sbatch)。两轮在同一个 job、同一 GPU，不能称为跨 GPU 或跨时段置信区间。

| grid | inner | L0 MMA/TCGEN，第一轮 | 第二轮 |
|---:|---:|---:|---:|
| 12 | 64 | 0.982446× | 0.983072× |
| 12 | 96 | 1.023575× | 1.027044× |
| 12 | 128 | 1.078886× | 1.079218× |
| 12 | 256 | 1.159756× | 1.159225× |
| 12 | 512 | 1.211895× | 1.211583× |
| 148 | 64 | 0.903874× | 0.903890× |
| 148 | 96 | 0.963235× | 0.963190× |
| 148 | 128 | 1.012949× | 1.012921× |
| 148 | 256 | 1.101074× | 1.101406× |
| 148 | 512 | 1.137683× | 1.137703× |

grid12 的 inner512 完整 kernel latency 两轮分别为 MMA `258.275528/258.232632 µs`、TCGEN `213.117123/213.136635 µs`；grid148 则为 MMA `259.337921/259.299526 µs`、TCGEN `227.952633/227.914886 µs`。加速比来自同一行的完整时间比，而不是从拟合推导。

用每轮 inner128/256/512 三点作最小二乘，仅用于分解形状的描述性结果如下：

| grid | MMA slope，两轮范围 | TCGEN slope，两轮范围 | slope 比，两轮范围 |
|---:|---:|---:|---:|
| 12 | 0.490908–0.490984 µs/inner | 0.386572–0.386680 µs/inner | 1.2695–1.2701× |
| 148 | 0.493709–0.493825 µs/inner | 0.414963–0.415061 µs/inner | 1.1898× |

这些不是数学上的无穷 inner 极限或 Tensor Core atom latency。TCGEN/grid148 的相邻区间斜率从 inner128→256 的约 `0.407` 升到 256→512 的约 `0.418 µs/inner`，三点拟合最大残差约 `0.615 µs`，说明不能假定整个扫描严格线性。相较旧 Job17937，grid12/inner64 的比值也从 0.920× 变为约 0.983×；因此应以内轮 A/B 判定，不能拼接不同时段的点构造更精确曲线。

L1 的 inner512 两轮比值在 grid12 为 `0.260886/0.260899×`，grid148 为 `0.260731/0.260736×`。这清楚地区分了“长驻留 L0 已出现正收益”和“当前保守物化实现仍很慢”。复算脚本 [summarize_review.py](summarize_review.py) 只用标准库，输出逐点时间、实际采样交叉区间、三点拟合及相邻区间斜率 JSON。

安全的原结论仍然成立：当前 one-shot 方案慢，完整 K2 尚未实现，不能发布 tcgen05 路径。应撤回的是将 `inner=64` 作为“已充分摊薄”的充分证据，以及据此关闭更长 TMEM 生命周期的研究机会。

### 循环是否真的执行了计算

没有发现把重复 MMA 全部优化掉的证据。`phase6_probe.cu:103` 和 `:360` 使用 `asm volatile`。归档 release SASS 的 V128/L0 MMA 函数从第 7544 行开始，回边 PC `0x1930→0xca0` 覆盖 HMMA；TCGEN 函数从第 5731 行开始，回边 PC `0x3310→0x1490` 覆盖 UTCHMMA、16 组 `LDTM.x8` 及 CTA barrier。

但 L0 并不是 recurrence：

- MMA 的 B fragment 在循环前加载（`:156–171`），A/B 在所有 inner 中不变；每次 MMA 从零累加（`:99–109`）。
- TCGEN 每轮也从零累加（`:358–368`），每轮 commit/wait、读回 TMEM，并在覆盖 accumulator 前同步（`:369–412`）。
- 两边仅最后一个 inner 将 L0 结果写到 global（`:216–221`、`:403–404`）；真实上游 U 变化、state 更新和下一 chunk 的输入搬运没有发生。
- 归档 L0 正确性仅检验 inner=1；L1 才显式检验 inner=1/2/4（`:719`、`:724–761`）。新 Job19844 又先运行 validate-only，在 grid12 对 L1 扩展到 inner128，全部 PASS；L0 长 inner、grid148 和 inner512 仍不是逐元素验证覆盖。

因此，新实测只支持“相同 operands、该协议、驻留生命周期下存在 crossover”。不能把它乘以 512 个 chunk，直接宣称真实 T8192 K2 加速。下一级应保留每 chunk BF16 rounding、变化的 U 和实际 state 消费顺序。

### [P1，静态证据] 大固定项并不全是 TMEM 协议

L0 包含一次 global staging 和最终 global writeback，而且两条路径的合并访存模式不同。TCGEN 的 `row=warp*32+lane`（`:395`），写出为 `global_delta[row*V+col+j]`（`:403–404`）；V128 时一个 scalar store 中相邻 lane 地址相隔 `128*4=512 B`。MMA 的 `row=...+lane/4`、列由 `lane%4` 决定（`:134–135,196–221`），多个 lane 覆盖同一行相邻列。因此两边虽写相同字节数，warp 内请求合并形态并不相同。

TCGEN 的 B staging 也按转置后物理布局顺序遍历，再读取原始 `B[k*V+n]`（`:311–316`）；其相邻 lane 的 k 变化会使 V128 读取地址跳过 128 个 BF16 元素。MMA 是连续读取逻辑 B（`:143–144`）。这些额外地址/事务成本属于当前 probe 的具体物化方案，不是 Tensor Core opcode 或 TMEM 分配的必然成本。真实 Phase-6 只需更新片上 state，并不必然执行同样的 global delta 写出。

因而 `inner=1` 的 2.5–2.7× 差异也不能全归因于 alloc/commit/wait。当前实验可以回答“两段具体 standalone microkernel 谁快”，但不足以把 TCGEN L0 称为所有合理直接集成方案的严格“乐观下界”。应分别测 staging/写出固定项，或从相同片上/片外边界进入和退出。

## 2. [已证实] ValueSlice 的延迟收益强于其瓶颈归因

Job17947 逐 repeat median 再取中位数得到：

| 输入 | V128 | V16 | V32 | V64 |
|---|---:|---:|---:|---:|
| fixed 1×8192 | 0.780672 | 0.569072 | 0.584400 | 0.634896 |
| packed 1×8192 | 0.784960 | 0.573952 | 0.584128 | 0.633280 |
| packed ragged6 | 0.340320 | 0.321888 | 0.287232 | 0.281024 |
| packed 8×1024 | 0.156112 | 0.313792 | 0.168544 | 0.154032 |
| packed 32×256 | 0.125280 | 0.258464 | 0.172384 | 0.133504 |

单位 ms。原始来源：`data/raw/05_dispatch_upgrade_17947.csv`。

最强论据是同一总 token 数下存在明确的正收益和反例，而且对照不是随意选的慢 baseline：独立 official 与 patched V128 有 10/10 tensor bitwise 对照和小于 1% 的时间差；强制 slices 有独立数值验证。最终测量在每个 repeat 随机化 variant 顺序（`benchmark_k3_shapes.py:159–164`）。旧 Nsys 在 T4096/H12 中也把变化定位到 recurrence，prepare 几乎不变（`experiments/BOTTLENECK_ANALYSIS.md:25–30`）。这些证据共同支持“优化确实生效”。

解释上的边界：CTA 数 `12→96` 是强机制证据，但这个改动同时改变线程数 `192→96`、寄存器数 `73→54`、shared memory 及每 CTA 的算术/搬运比例。不是一个只改变 grid、其余因素固定的消融实验。因此不能把全部 27% 归因于“更多 SM”这一个量，也不能从 8 倍 CTA 推出接近 8 倍性能。

一个有用但仅供立项的模型：假设只有 value-dependent 工作按 1/8 缩短，其余时间不变，固定 case 的两点可拆成约 `0.539 ms` 不随 ValueSlice 缩短的部分，以及 V128 下约 `0.242 ms` 可缩短部分。这个模型未区分 K1、同步、common input 和计算，不是 profiler 归因；它提示 V16 已接近“只继续切 value”这条路线的收益上限，下一步应解释剩余共同开销，而非默认再加 CTA。

## 3. [证据不足] 总 source-request 的减少不是 latency 改善的下界

报告 `FINAL_REPORT.md:175` 的 V16 `29.188 MiB`、理想 multicast 可省 `23.734 MiB / 81.3%`，已经正确标明是 source-request 模型，不是 HBM bytes。这项限定应继续保留。

但 `:217,229,322` 把 cluster/multicast 设为下一主要路线，优先级仍需实验支撑。最终 targeted NCU 的指标只包括 SpeedOfLight、LaunchStats、Occupancy、SchedulerStats 和少量 tensor 指标（`run_05_targeted_ncu.sbatch:56–66`）；没有输出 L2/TMA 请求 bytes 或 shared transaction/stall 的细分。旧 T4096/H12 NCU 已有 `L2 hit=81.15%→86.51%`（`BOTTLENECK_ANALYSIS.md:38–39`），说明 source 重复请求并不等价于等比例 HBM 重读。

在最终 NCU 的同一归一化口径下，DRAM throughput×duration 为 V128 `1.24%×1.27 ms`、V16 `1.83%×0.90122 ms`，两者比值约 1.047。这个粗略量不能代替 byte counter，却足以提醒：不能把 5 倍级 source-request 模型直接当成测到的 DRAM 增幅。旧 T4096 批次得到的比例又不同，进一步说明 cache 状态和采样口径需要控制。

比完整 cluster 重写更廉价的 gate：固定 shape，分别做 warm/cold/多 buffer 轮转；读取实际 L2 sectors、DRAM bytes、TMA/shared 阻塞指标；加入 2/4/8 CTA 的小型 multicast 原型，计入 cluster placement/residency 和同步。只有 latency 而不只是 source bytes 改善，才支持扩大实现。

## 4. [P2，表述与覆盖缺口] guard band 不等于风险/置信度模型

`implementation/current/flash_kda/dispatch.py` 的实际策略是：

- 仅 B300 CC10.3、148 SM、L2 容量±5%（`:144–149`）；sequence-head 数在 1..96（`:150–151`）。
- BF16 的 token anchors 为 2048/4096/8192；中间 token 数线性插值；FP32 只有 T4096（`:53–81,96–112`）。
- 用截距、sequence-head 线性项、额外 CTA layer 项预测延迟（`:175–180`）；候选最多两个 CTA layer（`:170–174`）。
- 候选比 V128 预测快至少 5 µs 且 3% 才启用（`:185–195`）。

这里的 `_COMMON_TILE_BYTES`、reuse footprint 和 `reuse_over_l2` 仅用于诊断输出；它们没有进入 scores。`resident_blocks_per_sm` 除了等于 0 时排除候选，也没有进入 latency score。L2 是设备容量 guard，不能观测其他 kernel/请求占掉多少 cache；`batch*heads` 是当前调用规模，不能观测其他 stream 上的并发。函数/报告不应让读者误以为已实现实时流量/并发感知。

我用纯 CPU 对有效设备参数枚举 BF16 所有整数 T2048..8192、sequence-head1..96，再加 FP32 T4096，共 590,016 个输入：454,804 次选择通过 guard，135,190 次是 `gain_below_guard_band`，22 次 `official_predicted_fastest`。所以 guard 并非死代码；问题是它不是根据每个点的拟合误差或方差动态决定的置信界。

最终报告 `:275` 用 packed8×1024 的 1.3% 收益解释 fallback，但该 case 实际先在 `dispatch.py:142–143` 被 `varlen_not_calibrated` 拒绝，从未评估 gain guard。它是“保守回退与观测小收益一致”的例子，不能算阈值分支的实验覆盖。已有旧日志 `experiments/integrated_validation_20260902.log:8–19` 的 H18/19、37/38、74/75 则是真正的边界验证，且提供了 graph_auto 时间。

最值钱的新验证不是重复 H12，而是 held-out T3072/T6144、H18/19、37/38、74/75，在 warm/cold/并发 stream 下直接比较全部强制 slices 和 auto。报告每点 auto 对最佳候选的 regret，以及相对 V128 的最坏回退/退化，而不仅是平均拟合误差。

## 5. [测量边界] CUDA Event、cache、graph 和 tail

`benchmark_k3_shapes.py:93–114` 在同一批 tensors 上预热后反复调用，用每次调用两端 CUDA Event 计时；每个 case 结束才 `empty_cache()`（`:181–182`）。`torch.cuda.empty_cache()` 是 allocator 行为，不是 GPU L2 flush。wrapper 每次申请 workspace（`implementation/current/flash_kda/__init__.py:105`），通常可由 caching allocator 复用地址。

这证明的是预热、重复输入缓冲下的完整 forward 调用延迟，且有 Python/CUDA 提交路径；不能直接把其中 p90 解释为线上请求的 tail latency。若 CPU 提交跟不上 GPU，Event 区间也可能包括提交之间的 GPU 空闲；对于约 0.57–0.78 ms 主 case，这比 tcgen05 的几微秒 one-shot probe 更不容易主导结果，但仍应明确测试模式。

最终 campaign 没有 paired graph baseline/variant sweep。不能说项目完全没测 graph：旧 integrated validation 日志确实记录 graph_auto 比 eager_auto 更快，例如 BF16/H18/T4096 `0.313968→0.300592 ms`；只是该表没有 graph_official，不能从中构造同口径 graph speedup。生产 graph 与常规调用需要各自 paired A/B；首次 graph capture 是否提前缓存 device metadata 也应检查，因为 `_device_characteristics`/dispatcher 有首次调用初始化和 lru cache。

两条 tcgen05 路径采用交替顺序和独立 job 复跑，这是良好的短期稳定性控制。然而 32 行跨两个紧邻 job 的约 0.004% 中位差异并不是跨设备置信区间，更不能修复 fixed-cost 混入 slope 的系统性问题。精密重复性与有效归因是两个要求。

## 最有价值的后续 gate

1. 原 probe 的 inner 扩展已在 Job19844 完成，L0 crossover 两轮重现。下一步控制 staging/消费布局与实际 state 依赖，不能直接映射到完整 KDA。
2. 对真正当前最快的 V16 路径测每 chunk 的共同等待成本，并在 controlled-cache 条件下验证 dispatcher regret。
3. TCGEN 已出现长 inner 正收益；下一原型应以变化的 U/state、真实 BF16 rounding 和跨 phase 生命周期为边界，而不是单独把一种 MMA opcode 换掉。
4. Cluster/multicast 以实际 bytes 与 latency 的小原型通过为 gate；不能凭 81.3% source-request reduction 预定它是下一最佳优化。

现有产物最可取之处，是保留强基线、bitwise 对照、反例和 fallback。接下来最需要提高的不是实验数量，而是把“可复现的曲线”进一步拆成可证伪的固定成本、稳态成本、数值语义和真实 workload 边界。
