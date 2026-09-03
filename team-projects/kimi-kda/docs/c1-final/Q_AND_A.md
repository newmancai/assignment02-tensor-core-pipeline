# C1 答辩追问准备

以下回答以最终报告和 Jobs 17926/17934/17935/17937/17947/17965 为准。回答原则是先给结论，再给一条决定性证据，最后主动说明边界；不要把 operator、模型和 serving 三种证据混在一起。

## 1. `tcgen05` 是更新的 Tensor Core 指令，为什么反而不快？

因为 K2 不是足够大、足够规则、能长期驻留新数据流的大 GEMM。`tcgen05` 不是把一条 `mma.sync` 换成另一条指令：它还需要 TMEM 分配、descriptor、异步 issue、commit/wait、累加器读回和释放。K2 Phase-6 的 K 只有 16，one-shot 工作很薄，固定协议成本占比高；K2 其他 phase 还存在 `[16,V]` 朝向和布局转换问题。

我们选择最有利的 `[128,16]@[16,128]` Phase-6，且在 L0 中预先给双方各自偏好的片上布局。即便把固定成本摊薄 64 次，Job 17937 的 `mma_time/tcgen_time` 仍只有 **0.919742×**，即 `tcgen05` 慢约 **8.73%**。所以否决的是“维持当前数据流、只换指令”，不是宣称所有 Blackwell Tensor Core 设计都更慢。

## 2. 你们的 `tcgen05` microbench 公平吗？为什么既有 L0 又有 L1？

L0 是对 `tcgen05` 偏乐观的下界：两边在计时前拿到自己偏好的片上布局，`tcgen05` 只额外承担真实 TMEM 协议。L1 再加入 BF16 state/gate 和保守 scalar U 重排，用来暴露集成风险；它比精心设计的跨 phase 实现更悲观。因此两者构成边界，而不是把单点当完整 K2。

探针做了三项防误判：BF16 输入/FP32 累加一致；L0 的 12 个 correctness case 为 FP32 exact，L1 的 inner 1/2/4 为 BF16 bitwise；SASS 确认基线走 `HMMA`、候选走 `UTCHMMA`。Job 17937 的 32 个 timing row 与独立 Job 17936 方向一致。边界仍然是：它没有让 Phase 1/3/4/6 共用一个 TMEM-resident 数据流。

## 3. `tcgen05` 的 tile 到底和 CHUNK=16 匹不匹配？

答案是“部分匹配”。SM100 BF16、CTA-group 1 支持 `M∈{64,128}`、`N=8..256` 且步长 8、`K=16`。Phase-6 `[128,16]@[16,V]` 对 V16/32/64/128 都能自然映射为 `m128nVk16`，这里 CHUNK16 正好是 K16。

但 K2 其他大量 phase 把 CHUNK 放在 M 维，形状是 `[16,V]`；M16 不能保持当前朝向直接映射，需要转置、重排和跨 phase 数据流重设。所以不能说“CHUNK16 完全不支持 tcgen05”，也不能从 Phase-6 可映射反推“整个 K2 可以机械替换”。

## 4. H12 从哪里来？为什么不用官方 H96 做瓶颈分析？

题目给出的 K3 KDA 配置是 96 heads、D128；TP8 时每张卡负责 `96/8=12` 个 head。官方 H96 适合复现其 benchmark，H12 则是 TP8 下的 per-GPU 计算形状，用来研究实际每卡 grid 并行度。K2 recurrence 的 grid 主要按 sequence 和 head 展开，所以单请求 H12 就只有 12 CTA。

必须同时说明边界：我们只有一张 B300，H12 是对 per-GPU compute shape 的模拟，不是完整 TP8 运行；没有据此声称 NCCL 或端到端 K3 已实测。

## 5. 单卡 H12 会不会人为制造 underfill？真实 TP8 还有别的请求或并发。

它代表的是低并发、单个长 prompt 的延迟敏感场景，不代表所有线上负载。真实 continuous batching 会增加 sequence 数，可能自然填满 SM；这正是我们没有无条件启用 V16 的原因。

Job 17947 给出了反例：total tokens 同为 8192，单序列 V16 约快 27%，8×1024 的最佳收益只有 1.3%，32×256 时 V16 反而慢约 106%。所以结论不是“线上总会 underfill”，而是“存在清晰的低并发长 prefill 域，且 dispatcher 必须识别请求分布”。

## 6. 为什么 ValueSlice 仍然算“SM80 MMA 迁移 SM100”的挑战？它没有使用 `tcgen05`。

题目把挑战路线明确列为“只换指令、大 CHUNK + rescale、并行度重构”三选一。迁移的工程问题是目标架构上什么执行引擎值得发布，不是必须让二进制出现某条新 opcode。

我们先在 SM103 上实测到 12 CTA 对 148 SM 的 underfill，再用 Phase-6 probe 否决直接换指令，最后选择 Value 行这一正交维度把 grid 扩到 96 CTA。dispatcher 还根据 compute capability、SM/L2、形状和 state dtype 做 guard。它是由 B300 资源结构驱动、在 B300 上标定的 SM100-family 专用化；同时保留 V128 `mma.sync` fallback，正好回答“专版该怎么出”。

## 7. ValueSlice 为什么能 bitwise equal？跨 CTA 拆分通常会改变归约顺序。

这里切的是相互独立的 Value 行，不是把同一个 dot-product 的归约维拆给多个 CTA。每个输出元素仍在一个 CTA 内按原来的顺序完成，slice 之间不需要 reduction 或 atomic，因此没有跨 CTA 浮点求和重排。

证据是 Job 17934：V16/V32/V64 对 V128 的 **98/98 comparison row 全部 bitwise equal**，最坏 relative RMSE 为零；Job 17929 的 patched V128 与官方扩展也有 10/10 output/final-state tensor bitwise equal。

## 8. “200/200 正确”具体是什么意思？有没有统一阈值？

准确说法是：Job 17934 的正确性 CSV 共 200 条 comparison row，**200/200 全部 finite**。其中 98 条 ValueSlice 对 V128 使用严格 bitwise 判据并全部相等；其余包括 FlashKDA 对题目指定 `naive.py`/`chunk.py` 和 FLA chunk 对 naive 的关系，观测最坏 relative RMSE 为 **0.9131%**。

不能说“200/200 统一 hard threshold 通过”：脚本只给短 smoke case 预设了 2% hard limit，长序列和 K3 行没有统一预注册阈值。答辩应把“finite”“bitwise”“观测 RMSE”三种陈述分开。

## 9. BF16 state 的精度结论是什么？为什么 FP32 public state 没更准？

在本轮 kernel 数值对拍中，长序列、ragged、state carry 和 long-memory gate 都保持 finite，参考关系的观测误差低于 1%。例如 T8192 long-memory 对 naive 的 output/final-state relative RMSE 为 **0.8240%/0.7405%**；Flash 对 FLA chunk 的观测最坏 output/state 为 **0.9131%/0.8151%**。

public state buffer 改成 FP32 没有形成“全 FP32 recurrence”，因为现有 FlashKDA 内部仍保留 BF16 舍入点，所以两种 public buffer 路径出现相同误差并不等于“FP32 state 没价值”。我们只证明 ValueSlice 没增加 kernel 误差；模型级 perplexity、长文本任务和真实 checkpoint 仍未验证。

## 10. 为什么说它既不是 compute-bound，也不是 memory-bound？

Job 17965 的官方 H12 K2 只有 12 CTA，最多同时覆盖 148 个 SM 的 8.1%；同时整卡 SM throughput 只有 2.64%，DRAM throughput 只有 1.24%。若是传统 compute-bound 或 HBM-bound，至少相应资源应接近饱和。

ValueSlice 将 CTA 提到 96 后，NCU duration 从 1.27 ms 降为 901.22 µs，SM/DRAM 也只升到 7.22%/1.83%。结合 recurrence 依赖、No Eligible 和 timeline，最谨慎的归纳是 **grid underfill + recurrence critical path + CTA 内 TMA/issue latency**。我们没有用单个 occupancy 或单个 stall metric 独立下因果结论。

## 11. NCU 里 tensor pipe 为什么一个口径从 30.98% 降到 5.43%，另一个却从 2.48% 升到 3.50%？

两个指标分母不同。`pct_of_peak_sustained_active` 只在 SM 已活跃的周期上归一化；V16 激活更多 SM、每个 CTA 工作更薄，因此单个活跃窗口里 Tensor pipe 占比可能下降，得到 30.98%→5.43%。`pct_of_peak_sustained_elapsed` 相对整个 kernel elapsed cycles，得到 2.48%→3.50%，更接近整次调用视角。

两者都不能单独当整卡 Tensor Core 利用率。答辩主判断依赖 grid、duration、SM throughput、DRAM throughput、occupancy 和 scheduler 指标的交叉；且 NCU 1.27 ms/901.22 µs 只做 profiler 内 A/B，不与 CUDA Event 0.7807/0.5698 ms 混算。

## 12. 为什么 CHUNK=32/64 不值得？B300 shared memory 更大，难道装不下吗？

shared memory 不是第一个破点。按 H12/T8192 纸面模型，workspace/head 从 C16 的 6.750 MiB 只升到 C32 的 7.125 MiB、C64 的 8.063 MiB；真正先失效的是当前没有 rescale 的指数恢复路径，C32/C64 都在 token 18 首次 FTZ/overflow，每通道分别有 15/47 组 zero 和 inf。

第二个代价是朴素扩展 Neumann 密集幂级数：总序列计算变为 C16 的 5.33×/26.67×。FLA safe/block 小探针证明加入 rescale/block solve 可以做对，但那已经是算法重构，不能作为“机械放大 CHUNK”的性能证据。

## 13. ValueSlice 增加重复搬运，会不会把收益吃掉，或者伤害并发吞吐？

会，所以它必须受保护。V16 把一个 head 切成 8 个 CTA，q/k/gate 等 common inputs 会被重复请求；在高自然并行度的 32×256 case，V16 确实慢约 106%。这不是实验异常，而是 dispatcher guard 的关键反例。

当前策略只捕获已标定的 fixed/packed 单长序列，多序列 varlen 回退 V128。下一步才是 CTA Cluster + TMA multicast：让同一 head 的 slice CTA 共享 common-input 搬运。纸面 source-request 模型在 T4096 单 head 上给出最多 81.3% common-input 请求可消除，但这不是 HBM 实测，必须连同 cluster residency 和同步成本一起 profile。

## 14. packed 单序列为什么可以当 fixed B1？读取 `cu_seqlens` 会不会触发同步？

对合法 packed 输入，`cu_seqlens.numel()-1` 就是 sequence 数。nseq=1 时，K2 的 sequence/head grid 和 recurrence length 与 fixed B1、相同 total T 一致，所以可以复用固定序列的已标定策略。

补丁 `0002` 只读取 `numel()`，这是 tensor metadata；它不读取设备上 `cu_seqlens` 的数值，不发生 GPU 到 CPU copy，也不引入 device synchronize。Job 17947 的 policy test 和计时都证明 packed-one auto 选择 V16；nseq>1 仍被标为未建模 varlen 并回退 V128。

## 15. 27% operator 加速能让 Kimi K3 的 TTFT 或 SLO goodput 提升多少？

目前不能给一个实测端到端百分比。令 FlashKDA forward 占完整 prefill wall time 的比例为 `p`，实测 operator 降时为 `r=0.27`，在忽略重叠变化的一阶 Amdahl 模型中，prefill 降时约为 `p·r`，理想容量加速为 `1/(1-p·r)`。

例如 p=20%/40%/60% 时，预计 prefill 降时是 5.4%/10.8%/16.2%，理想容量加速约 1.057×/1.121×/1.193×。这仍不是 SLO goodput：后者取决于 TTFT/TPOT 哪个约束绑定、prompt 分布、continuous batching、排队非线性、计算通信重叠和高并发时 dispatcher 是否回退。需要完整 checkpoint + serving scheduler 的 sweep 才能回答。

## 16. K3 有 69/93 层 KDA，为什么不能直接把 27% 乘以 69/93？

因为 69/93 是层数比例，不是 wall-time 比例。不同层的 FLOP、访存、通信和融合程度不同；每个 KDA 层还包括输入投影、norm、gate 和输出投影，FlashKDA forward 只是其中一段。24 个非 KDA 层的代价也不能按层数等权。

正确外推需要 profile 得到 operator 在完整 prefill 中的时间占比 `p`，再用 `0.27p` 做第一阶估计，并检查与 NCCL/GEMM 的重叠是否改变。当前报告只给 p 的敏感性表，不把架构层数当性能数据。

## 17. 这项工作会改善 decode/TPOT 吗？

本轮不能声称会。ValueSlice dispatcher 对 `T=1` 回退 V128；真实 serving 的纯 decode 还有独立 fused KDA decode 路径。本实验优化的是 FlashKDA forward/prefill，最可能影响长 prompt 的 KDA prefill latency 和 TTFT 组成部分。

若要研究 TPOT，应单独 profile fused decode、state update、batching 和通信，不能把 prefill kernel 的 27% 外推到单 token decode。

## 18. 只有一张 B300、每次 15 分钟，结论可靠吗？

对单卡 instruction/kernel/dispatcher 因果链，证据是可复核的：官方基线、parity、参考正确性、CHUNK、`tcgen05`、dispatcher 和 targeted NCU 被拆成自包含 job；实际 wall time 分别为 93、6、33、41、16、10、15 秒，均远低于限制。`tcgen05` 还有独立 Job 17936/17937 的方向复现，ValueSlice 也有早期独立批次的同方向证据。

限制同样明确：只有单 GPU 样本，没有跨机器置信区间；没有 TP8 NCCL、完整 K3 checkpoint、线上 trace 或 SLO sweep。因此我们把结论限定在 B300 单卡 FlashKDA forward 和有条件的系统推断，不把资源限制掩盖掉。

## 19. 如果未来重写完整 K2，什么结果会让你们重新开启 `tcgen05` 路线？

下一版不能再是孤立 Phase-6 指令替换，而应让 Phase 1/3/4/6 共享转置后的操作数布局和 TMEM 生命周期，避免每个小 phase 重复 alloc/搬运/读回。重新开启需要同时过三道 gate：

1. V128 和代表 V16/V32 形状对 V128/reference 的正确性；
2. 完整 K2 在 fixed、packed 单序列和多序列反例上的净延迟，而不是 microbench core 吞吐；
3. NCU 中 grid、TMEM/TMA、scheduler、occupancy 和整卡 elapsed 指标能解释收益来源。

在最乐观 Phase-6 V128 已慢 8.7% 的当前证据下，没有必要先支付完整集成和长期维护成本。

## 20. 如果你们是作者，sm100a/sm103a 专版到底怎么发布？

发布 guarded hybrid，而不是替换通用实现：

- V128 `mma.sync` 保持默认兼容 fallback；
- 在 B300/SM103、BF16 state、低并发长 prefill 的已标定形状启用 ValueSlice；
- packed nseq=1 进入 fixed B1 策略，未建模的 nseq>1 varlen 回退；
- 保留环境变量强制 V，便于回归、A/B 和现场禁用；
- CUDA 13/`sm_103a` 单独构建与 CI，持续覆盖 bitwise、reference、CUDA Graph 和反例；
- 下一原型是 cluster/TMA multicast，不在当前版本合入全面 `tcgen05`。

这样把 27% 的已证实收益与架构专用二进制、维护多个 layout、TMEM/cluster 调试的成本隔离开；一旦未命中已标定域就回到官方路径。

## 21. 为什么官方 FlashKDA 已经比 FLA 快 1.79–3.42×，还值得做 ValueSlice？

“相对另一个实现快”不等于“已经贴近目标硬件上限”。官方 H96 benchmark 证明 baseline 强；H12 targeted NCU 则揭示 TP8 单请求部署形状出现新的 12 CTA/148 SM underfill。两者研究对象不同，没有矛盾。

ValueSlice 也没有推翻官方设计：它保留相同 `mma.sync`、相同总 Tensor FLOP 和 bitwise 结果，只给低并发长 prefill 增加正交并行度。最终方案因此是对官方路径的 guarded 补充，而不是宣称官方 kernel 整体设计错误。

## 22. 通信角度看，这个优化可能被 NCCL 完全隐藏吗？

可能，所以目前不能声称 TP8 端到端收益。ValueSlice 本身没有跨 GPU 通信，也不改变 collective 的消息体；局部 K2 节省约 0.21 ms 是否暴露在关键路径上，取决于 KDA 与投影、collective 的执行顺序和重叠。

需要在真实 TP8 上同时保留 compute-only 和 NCCL-overlapped timeline，比较 K2 降时前后的关键路径、通信气泡和 TTFT。若 K2 完全被 collective 覆盖，端到端收益会小；若 K2 位于未重叠的串行区，收益才会显现。单 B300 只能把这个未知量明确列出，不能替代多卡实验。

## 23. B300 的 FP8/FP6/FP4 为什么没有用于 KDA recurrence？

新低精度更适合先放在大而规则的输入/输出投影 GEMM，那里更容易利用 Tensor Core 峰值，也容易通过 scale 设计控制误差。KDA recurrence 包含长时状态传播、指数 gate 和 16×16 inverse，误差可能随序列累积；本轮连 BF16 state 的结论都只到 kernel 数值层，尚无模型级精度证据。

因此低精度 recurrence 不属于“免费迁移”。若要挑战，应分别设计 state、gate、inverse 的 scaling，做长上下文误差增长、perplexity 和任务级验证，再谈性能；不能仅依据 B300 支持 FP4/FP6 就宣布可用。

## 一页速记

| 若只来得及说一句 | 回答 |
|---|---|
| 总结论 | 不做全面 `tcgen05`；做 guarded ValueSlice；下一步 cluster/multicast |
| `tcgen05` | V128 Phase-6 乐观摊销仍 0.920×，慢 8.7% |
| 瓶颈 | 12 CTA 对 148 SM，SM/DRAM 2.64%/1.24%，首要是 underfill |
| ValueSlice | grid 12→96；fixed/packed-one 约 −27.0%/−26.9% |
| 反例 | 32×256 的 V16 慢 106%，所以必须 guarded |
| 正确性 | 200/200 finite；其中 98/98 ValueSlice 对 V128 bitwise |
| BF16 state | 最坏观测 reference RMSE <1%，但不是模型级证明，也非全 FP32 内部对照 |
| Kimi 系统 | 只改善 prefill 算子；TTFT 约 `0.27p`；无 TPOT/goodput/NCCL 实测 |
| 发布 | V128 fallback + 已标定 ValueSlice + 架构/形状 guard |
