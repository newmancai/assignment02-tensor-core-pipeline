# B300 `tcgen05 + TMEM` Phase-6 微基准：结论与证据边界

## 结论先行

对于 Kimi K3 的官方形状 `D=128, V=128, chunk=16`，**不建议把 FlashKDA K2 Phase 6 中的 `mma.sync` 直接替换为 `tcgen05 + TMEM` 并集成进正式 kernel**。

理由不是“Blackwell Tensor Core 没有价值”，而是本次最有利于 `tcgen05` 的 L0 实验仍没有出现正收益：在把一次性成本摊薄 64 次后，`tcgen05` 在 `grid=12` 和 `grid=148` 下分别比 `mma.sync` 慢 **8.7%** 和 **10.6%**。若每个 Phase 6 独立完成 TMEM 分配、提交、等待、读回与释放（`inner=1`），则分别慢 **2.66 倍** 和 **2.53 倍**。而当前 K2 的 `U` 在 Phase 4 后已经位于适合 `mma.sync` 的寄存器 fragment 中；直接换指令还会额外引入布局转换，L0 并未向 `tcgen05` 收取这笔真实集成成本。

这是一项明确的 **Phase-6 instruction-swap no-go**，不是完整 SM100/SM103 执行引擎的 no-go。把 Phases 1/3/4/6 一起改成转置数据流、让中间量长期驻留 TMEM、并与 TMA/producer warp 重叠，是另一个尚未实验的问题；本微基准不能否定或证明它。

## 实际比较了什么

Phase 6 的核心矩阵乘是：

```text
delta_state[128,V] = k_restored_t[128,16] @ U[16,V]
state[128,V] = BF16(FP32(state) * g_total[row] + delta_state)
```

两条路径都使用 BF16 输入、FP32 累加：

- 基线：四个 compute warp 使用 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`。当 `V=128` 时，每 CTA、每次 Phase 6 动态执行 128 个 `m16n8k16` atom。
- 候选：一个 CTA-group 1 `tcgen05.mma`，形状为 `m128nVk16`，FP32 accumulator 位于 TMEM；`V=16/32/64/128` 都符合实际指令约束。
- 生成的 SASS 中，基线确实是 `HMMA.16816.F32.BF16`，候选确实是 `UTCHMMA`，因此结果不是编译器回退到同一条旧路径。

测试覆盖：

- `V = 16, 32, 64, 128`；`V=128` 才是当前 Kimi K3 的主要决策形状。
- `grid=12`：对应 `H=96, TP=8` 时单请求约 12 个本地 head/CTA。
- `grid=148`：对应 B300 的 148 个 SM，近似一 SM 一 CTA 的满机并行探针。
- `inner=1`：一次完整 kernel launch，保留 staging、TMEM alloc/commit/wait/load/dealloc 等一次性成本。
- `inner=64`：在一次 launch 内重复 64 次并以 `kernel_time/64` 归一化，用来摊薄 launch、staging 和 TMEM 分配成本；它不是 64 个真实 chunk 的完整 recurrence。
- 每个 timing row：30 次 warmup，200 次 launch/重复，5 个重复取中位数；每轮交替两条路径的测量顺序。

### L0 与 L1 的边界

| 层级 | 测量内容 | 可以怎样解读 |
| --- | --- | --- |
| L0 | 两边都包含 global-to-shared staging、Phase-6 GEMM、结果消费/写出和完整 launch；进入计算时，各自得到偏好的 on-chip 布局 | 对 `tcgen05` 较有利的核心执行探针。它没有收取“把 Phase-4 寄存器中的 `U` 改造成 tcgen descriptor/TMEM 数据流”的真实成本。 |
| L1 | 加入真实的 BF16 state load/update/store 和逐行 FP32 gate；`tcgen05` 每个 inner 还用标量 shared-to-shared copy 将 `U[16,V]` 重排成 descriptor 布局 | 一个故意保守的物化实现，用于暴露布局转换风险；不是优化后的 `stmatrix`、`tcgen05.st` 或 TMEM-resident 实现，不能当作未来最佳实现的性能上界。 |

## B300 实测结果

主结果来自 Slurm job `17937`：NVIDIA B300 SXM6 AC，compute capability 10.3，148 SM，CUDA driver/runtime API 13.0，driver 580.126.09；测量开始和结束时记录的 SM clock 均为 1905 MHz。

下表中的 `MMA/TCGEN` 大于 1 才表示 `tcgen05` 更快。

### 官方 `V=128` 形状

| 层级 | grid | inner | `mma.sync` kernel / phase (µs) | `tcgen05` kernel / phase (µs) | MMA/TCGEN | 直接观察 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| L0 | 12 | 1 | 6.165 / 6.165 | 16.390 / 16.390 | 0.376 | `tcgen05` 慢 2.66× |
| L0 | 148 | 1 | 7.297 / 7.297 | 18.453 / 18.453 | 0.395 | `tcgen05` 慢 2.53× |
| L0 | 12 | 64 | 37.612 / 0.588 | 40.894 / 0.639 | 0.920 | 摊薄后 `tcgen05` 仍慢 8.7% |
| L0 | 148 | 64 | 38.903 / 0.608 | 43.039 / 0.672 | 0.904 | 摊薄后 `tcgen05` 仍慢 10.6% |
| L1 | 12 | 1 | 16.400 / 16.400 | 30.746 / 30.746 | 0.533 | 保守物化路径慢 1.87× |
| L1 | 148 | 1 | 18.433 / 18.433 | 31.745 / 31.745 | 0.581 | 保守物化路径慢 1.72× |
| L1 | 12 | 64 | 305.025 / 4.766 | 1190.211 / 18.597 | 0.256 | 每轮标量重排使候选慢约 3.90× |
| L1 | 148 | 64 | 305.311 / 4.770 | 1190.796 / 18.606 | 0.256 | 每轮标量重排使候选慢约 3.90× |

### 形状扫描：只看更有利的 L0、`inner=64`

| V | grid=12：MMA / TCGEN phase (µs) | MMA/TCGEN | grid=148：MMA / TCGEN phase (µs) | MMA/TCGEN |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.384 / 0.256 | 1.501 | 0.384 / 0.256 | 1.501 |
| 32 | 0.256 / 0.288 | 0.889 | 0.256 / 0.288 | 0.889 |
| 64 | 0.328 / 0.357 | 0.918 | 0.352 / 0.384 | 0.916 |
| 128 | 0.588 / 0.639 | 0.920 | 0.608 / 0.672 | 0.904 |

`V=16` 的 1.50× 正收益说明这个探针能观察到形状相关的 crossover，也说明不能笼统地说“`tcgen05` 一定更慢”。但 K3 的目标是 `V=128`；不能用 `V=16` 的结果替代正式形状的结论。

### 正确性、资源与复现性

- 使用 PTXAS TMEM bounds guardrail 的独立构建先做 fail-fast 检查，随后重新 release 构建才计时。
- 在 `grid=12` 上，L0 的两条路径对所有四个 V 都与 CPU reference 做逐元素 FP32 精确比较并 `bad=0`；L1 在 `inner=1/2/4` 下对 BF16 state 做逐 bit 比较并全部 `bad=0`。`inner=2/4` 同时覆盖了 mbarrier parity 的交替。该检查证明当前确定性测试向量与这些 barrier 轮次正确，不等价于对任意生产输入或完整 KDA 的数值证明。
- 正确性只显式跑了 `grid=12`；`grid=148` 是性能覆盖，不应写成已逐元素验证。
- release 版本 `V=128` 的资源报告为：L0 `mma.sync/tcgen05 = 34/32` registers、`8192/8204 B` static shared memory；L1 为 `38/39` registers、`41472/45580 B`。CUDA occupancy API 对 `tcgen05` 的所有形状均报告 1 active block/SM，而 `V=128` 基线为 L0 12、L1 5。这个 API 信号提示 TMEM 可能限制并发，但没有 profiler 证据前，不能直接换算成真实 SLO 或 goodput 损失。
- jobs `17936` 与 `17937` 在同一块 GPU 上背靠背复跑。32 个匹配 timing rows 中，MMA 和 TCGEN kernel 中位数的相对变化中位数均约 0.004%；最大变化分别为 0.465% 和 0.679%，speedup 最大变化 0.591%。这支持“差异大于本次同环境短期抖动”，但不是跨 GPU、跨时段的置信区间。

## 为什么当前不值得直接集成

1. **在最关键、也最偏向候选的判据上已经没有收益。** `V=128, inner=64` 摊薄了 TMEM 分配和 launch 等固定成本，L0 仍在两个 grid 上落后；继续做 direct swap 缺少正向性能信号。
2. **它不是一条可原位替换的指令。** 当前 K2 在 Phase 4 后把 `U` 留在 `mma.sync` 的寄存器 B fragment 中。`tcgen05` 要求 descriptor 布局、TMEM accumulator、异步 commit/barrier，以及结果从 TMEM 读回。简单替换会破坏已有的寄存器数据流。
3. **K=16 太浅，固定协议成本显著。** `tcgen05` 把 `m128n128k16` 合成一次大 issue，但 alloc、proxy fence、commit、mbarrier wait、TMEM load/dealloc 并未消失。`inner=1` 的 2.5–2.7× 劣势正是 direct-per-phase 用法的风险。
4. **布局转换可以淹没计算收益。** L1 的标量重排不是最终设计，却实证说明：若不能把 `U^T` 自上游直接保持在合适布局中，数据搬运会远大于少发 MMA 指令的收益。
5. **潜在并发方向也没有加分证据。** API 报告的 1 CTA/SM 需要 profiler 复核，但至少当前没有证据表明 TMEM 路径能改善多请求驻留或 SLO goodput。

因此，当前合理动作是停止“仅替换 Phase-6 MMA atom”的正式集成，而不是在完整 FlashKDA 中投入更多 B300 时隙验证一个已经未通过 L0 gate 的方案。

## 这份结果能说什么、不能说什么

可以说：

- B300/SM103 能原生编译并正确执行这里使用的 BF16 `tcgen05 + TMEM` 路径。
- 对真实 Phase-6 矩阵规模，naive one-shot 路径明显落后；即使把一次性成本摊薄，`V=128` 仍没有超过现有 `mma.sync` 执行原语。
- 因此，“保持现有 K2 数据流，只把 Phase 6 的 MMA 换成 `tcgen05`”目前不值得集成。

不能说：

- 不能把表中微秒或 8.7–10.6% 直接当成完整 K2、完整 KDA、69 层 KDA 或 Kimi K3 prefill 的变化。
- 不能据此宣布所有 SM100/SM103 rewrite 都不值得。跨 Phases 1/3/4/6 的 TMEM-resident 转置数据流可能摊薄协议成本并消除中间搬运，但尚未实现。
- 不能据此推导并发、端到端吞吐、TTFT、TPOT 或 SLO goodput；这些需要整 kernel 和服务级 workload。
- 不能据此判断 TMA、CTA Cluster/DSM/multicast、低精度 FP8/FP6/FP4 的收益；本实验没有使用或隔离它们。
- 不能把 L1 的约 3.9× 劣势当作优化后 `tcgen05` 设计的必然结果；它有意使用保守的标量 shared-to-shared 重排。

## 若继续探索，下一道 gate 应该是什么

只有在研究目标是“SM100 数据流重构”而非“指令替换”时，才值得再做一个小型 L2 原型：让 Phases 1/3/4/6 共享转置布局和 TMEM 生命周期，避免 Phase 4 `U` 先生成旧式寄存器 fragment 再重排，并保留 FlashKDA 现有的 BF16 rounding point。L2 必须同时满足：

1. `V=128`、`grid=12/148` 正确性通过；
2. 在完整四阶段边界内稳定超过基线，而不只是 instruction issue 数更少；
3. profiler 证明 TMEM、barrier、shared-memory 和 occupancy 没有把收益抵消。

只有 L2 通过后，才进入完整 K2 集成、官方 forward shape 回归以及 K3 prefill/并发/SLO goodput 测试。

## 可复核证据

- [微基准源码](./tcgen05_probe/phase6_probe.cu)
- [实验定义、限制与复现命令](./tcgen05_probe/README.md)
- [15 分钟 Slurm 脚本](./tcgen05_probe/run_03_tcgen05_probe.sbatch)
- [job 17937 CSV](./tcgen05_probe/results/03_tcgen05_probe_17937.csv)
- [job 17937 完整日志](./tcgen05_probe/results/03_tcgen05_probe_17937.log)
- [job 17936 重复 CSV](./tcgen05_probe/results/03_tcgen05_probe_17936.csv)
- [release SASS](./tcgen05_probe/results/phase6_probe.sass)
