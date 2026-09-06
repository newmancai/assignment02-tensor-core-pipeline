# C1 答辩追问准备

以下回答以原迁移证据 Jobs 17926/17934/17935/17937/17947/17965，以及后续主线 Jobs 19896/19901/19903/19934/19935 为准。回答原则是先给结论，再给一条决定性证据，最后主动说明边界；不要把 operator、模型和 serving 三种证据混在一起，也不要把不同作业、不同 GPU 频率下的绝对毫秒拼接。最新性能 headline 只采用同一 Job 内的配对比例。

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

Job 17947 给出了静态形状反例：total tokens 同为 8192，单序列 V16 约快 27%，8×1024 的最佳收益只有 1.3%，32×256 时 V16 反而慢约 106%。Job19934 又给出动态并发反例：两个 stream 的无初态 joined pair 从 1.147440 增到 1.164832 ms，回归 1.52%，三轮方向一致；有初态 pair 基本持平。`N=1` 只表示一次调用有一个 sequence，不表示设备上没有别的请求，当前 guard 也不检测运行时并发。

## 6. 为什么 ValueSlice 仍然算“SM80 MMA 迁移 SM100”的挑战？它没有使用 `tcgen05`。

题目把挑战路线明确列为“只换指令、大 CHUNK + rescale、并行度重构”三选一。迁移的工程问题是目标架构上什么执行引擎值得发布，不是必须让二进制出现某条新 opcode。

我们先在 SM103 上实测到 12 CTA 对 148 SM 的 underfill，再用 Phase-6 probe 否决直接换指令，随后用 Value 行把 grid 扩到 96 CTA，并继续用 Phase-6 state prefetch 和 Phase-1 fragment lookahead 改善 CTA 内 load/use 调度。三层都保留 `mma.sync` 和原数值顺序，但由 B300 的 grid、资源和生成代码证据驱动。Python 策略只提供 device/shape guard，后两层还是 build-time opt-in，且没有实时并发感知；因此应称为 SM103 专用候选，而不是已经完成的通用生产 dispatcher。

## 7. ValueSlice 为什么能 bitwise equal？跨 CTA 拆分通常会改变归约顺序。

这里切的是相互独立的 Value 行，不是把同一个 dot-product 的归约维拆给多个 CTA。每个输出元素仍在一个 CTA 内按原来的顺序完成，slice 之间不需要 reduction 或 atomic，因此没有跨 CTA 浮点求和重排。

证据是 Job 17934：V16/V32/V64 对 V128 的 **98/98 comparison row 全部 bitwise equal**，最坏 relative RMSE 为零；Job 17929 的 patched V128 与官方扩展也有 10/10 output/final-state tensor bitwise equal。最新 Job19934 的 Phase-1 候选另有 120 条主比较和 14 条尾块/状态补比较，存在的 output/final state 均相对 V128 finite 且 bitwise；这证明调度重排没有改变已测结果，但不能替代独立高精度算法 oracle。

## 8. “200/200 正确”具体是什么意思？有没有统一阈值？

准确说法是：Job 17934 的正确性 CSV 共 200 条 comparison row，**200/200 全部 finite**。其中 98 条 ValueSlice 对 V128 使用严格 bitwise 判据并全部相等；其余包括 FlashKDA 对题目指定 `naive.py`/`chunk.py` 和 FLA chunk 对 naive 的关系，观测最坏 relative RMSE 为 **0.9131%**。

不能说“200/200 统一 hard threshold 通过”：脚本只给短 smoke case 预设了 2% hard limit，长序列和 K3 行没有统一预注册阈值。答辩应把“finite”“bitwise”“观测 RMSE”三种陈述分开。Job19934 的 120+14 是最新候选的跨路径 bitwise 回归，Graph 自比较、计时后复查和 sanitizer 也不能累加成更多独立算法参考形状。

## 9. BF16 state 的精度结论是什么？为什么 FP32 public state 没更准？

在本轮 kernel 数值对拍中，长序列、ragged、state carry 和 long-memory gate 都保持 finite，参考关系的观测误差低于 1%。例如 T8192 long-memory 对 naive 的 output/final-state relative RMSE 为 **0.8240%/0.7405%**；Flash 对 FLA chunk 的观测最坏 output/state 为 **0.9131%/0.8151%**。

public state buffer 改成 FP32 没有形成“全 FP32 recurrence”，因为现有 FlashKDA 内部仍保留 BF16 舍入点，所以两种 public buffer 路径出现相同误差并不等于“FP32 state 没价值”。我们只证明 ValueSlice 没增加 kernel 误差；模型级 perplexity、长文本任务和真实 checkpoint 仍未验证。

## 10. 为什么说它既不是 compute-bound，也不是 memory-bound？

Job 17965 的官方 H12 K2 只有 12 CTA，最多同时覆盖 148 个 SM 的 8.1%；同时整卡 SM throughput 只有 2.64%，DRAM throughput 只有 1.24%。若是传统 compute-bound 或 HBM-bound，至少相应资源应接近饱和。

ValueSlice 将 CTA 提到 96 后，NCU duration 从 1.27 ms 降为 901.22 µs，SM/DRAM 也只升到 7.22%/1.83%。结合 recurrence 依赖、No Eligible 和 timeline，最谨慎的归纳是 **grid underfill + recurrence critical path + CTA 内 TMA/issue latency**。

后续 Job19901 又说明瓶颈是分层的：V16/P1 到 V16/P4 的 achieved occupancy 都约 4.69%，但 issue active 从 14.72% 升到 18.12%，eligible warps 从 0.1472 升到 0.1812，short-scoreboard cycles/issued instruction 从 0.9010 降到 0.4649，profile K2 duration 从 521.088 降到 409.984 µs。它与软件预取改善依赖隐藏一致，但只是一份 replay profile，不构成全部节省的独立因果证明。

## 11. NCU 里 tensor pipe 为什么一个口径从 30.98% 降到 5.43%，另一个却从 2.48% 升到 3.50%？

两个指标分母不同。`pct_of_peak_sustained_active` 只在 SM 已活跃的周期上归一化；V16 激活更多 SM、每个 CTA 工作更薄，因此单个活跃窗口里 Tensor pipe 占比可能下降，得到 30.98%→5.43%。`pct_of_peak_sustained_elapsed` 相对整个 kernel elapsed cycles，得到 2.48%→3.50%，更接近整次调用视角。

两者都不能单独当整卡 Tensor Core 利用率。答辩主判断依赖 grid、duration、SM throughput、DRAM throughput、occupancy 和 scheduler 指标的交叉；且 NCU 1.27 ms/901.22 µs 只做 profiler 内 A/B，不与 CUDA Event 0.7807/0.5698 ms 混算。

## 12. 为什么 CHUNK=32/64 不值得？B300 shared memory 更大，难道装不下吗？

shared memory 不是第一个破点。按 H12/T8192 纸面模型，workspace/head 从 C16 的 6.750 MiB 只升到 C32 的 7.125 MiB、C64 的 8.063 MiB；真正先失效的是当前没有 rescale 的指数恢复路径，C32/C64 都在 token 18 首次 FTZ/overflow，每通道分别有 15/47 组 zero 和 inf。

第二个代价是朴素扩展 Neumann 密集幂级数：总序列计算变为 C16 的 5.33×/26.67×。FLA safe/block 小探针证明加入 rescale/block solve 可以做对，但那已经是算法重构，不能作为“机械放大 CHUNK”的性能证据。

## 13. ValueSlice 增加重复搬运，会不会把收益吃掉，或者伤害并发吞吐？

会，所以它必须受保护。V16 把一个 head 切成 8 个 CTA，q/k/gate 等 common inputs 会被重复请求；在高自然并行度的 32×256 case，V16 确实慢约 106%。这不是实验异常，而是 dispatcher guard 的关键反例。

当前策略只捕获已标定的 fixed/packed 单长序列，多序列 varlen 回退 V128；后续 prefetch guard 仍不知道设备实时并发，Job19934 的双流负例说明 shape guard 并不足以保证吞吐无回归。因此下一步优先做真实并发矩阵和调用方 latency/throughput 模式，再做完整 K3/TP8 critical path。CTA Cluster + TMA multicast 仍可用于共享 common input，但纸面最多 81.3% 只是 source-request 模型，不是 HBM 字节或 duration 实测，优先级低于已暴露的并发问题。

## 14. packed 单序列为什么可以当 fixed B1？读取 `cu_seqlens` 会不会触发同步？

对合法 packed 输入，`cu_seqlens.numel()-1` 就是 sequence 数。nseq=1 时，K2 的 sequence/head grid 和 recurrence length 与 fixed B1、相同 total T 一致，所以可以复用固定序列的已标定策略。

补丁 `0002` 只读取 `numel()`，这是 tensor metadata；它不读取设备上 `cu_seqlens` 的数值，不发生 GPU 到 CPU copy，也不引入 device synchronize。Job 17947 的 policy test 和计时都证明 packed-one auto 选择 V16；nseq>1 仍被标为未建模 varlen 并回退 V128。

## 15. 当前分层 operator 加速能让 Kimi K3 的 TTFT 或 SLO goodput 提升多少？

目前不能给一个实测端到端百分比。令 FlashKDA forward 占完整 prefill wall time 的比例为 `p`，在忽略重叠变化的一阶 Amdahl 模型中，prefill 降时约为 `p·r`。ValueSlice 层可用 `r=0.27`；最新 Job19934 的匹配 T8192 单请求候选，首段无初态和续段有初态分别是 `r=0.3723`、`r=0.4552`。这两个数必须分开，不能平均、相加，也不能外推到高并发。

以已验证范围更宽的 ValueSlice 层为例，p=20%/40%/60% 时，一阶 prefill 降时仍只是 5.4%/10.8%/16.2%。最新 37.23%/45.52% 是默认关闭候选在两个状态合约上的同作业累计 operator 结果，不是生产 SLO 预测。SLO goodput 还取决于 TTFT/TPOT 哪个约束绑定、prompt 分布、continuous batching、排队非线性和计算通信重叠，需要完整 checkpoint 与 scheduler sweep 才能回答。

## 16. K3 有 69/93 层 KDA，为什么不能直接把 27% 乘以 69/93？

因为 69/93 是层数比例，不是 wall-time 比例。不同层的 FLOP、访存、通信和融合程度不同；每个 KDA 层还包括输入投影、norm、gate 和输出投影，FlashKDA forward 只是其中一段。24 个非 KDA 层的代价也不能按层数等权。

正确外推需要 profile 得到 operator 在完整 prefill 中的时间占比 `p`，再选与实际状态合约、并发和启用策略匹配的 `r` 做 `p·r` 第一阶估计，并检查与 NCCL/GEMM 的重叠是否改变。当前报告不把架构层数或默认关闭候选的最佳点当生产性能数据。

## 17. 这项工作会改善 decode/TPOT 吗？

本轮不能声称会。ValueSlice dispatcher 对 `T=1` 回退 V128；真实 serving 的纯 decode 还有独立 fused KDA decode 路径。本实验优化的是 FlashKDA forward/prefill，最可能影响长 prompt 的 KDA prefill latency 和 TTFT 组成部分。

若要研究 TPOT，应单独 profile fused decode、state update、batching 和通信，不能把 ValueSlice 的 27% 或最新候选的 37.23%/45.52% 外推到单 token decode。

## 18. 只有一张 B300、每次 15 分钟，结论可靠吗？

对单卡 instruction/kernel/selector 因果链，证据是可复核的：官方基线、parity、参考正确性、CHUNK、`tcgen05`、ValueSlice、Phase-6 P4、Phase-1 和 targeted NCU 被拆成自包含 job。`tcgen05` 有 Job17936/17937 的方向复现，Phase-1 有 Job19934 的干净构建与完整矩阵。

限制同样明确：只有单 GPU 样本，没有跨机器置信区间；Job19934 仅在开头采样到 1095 MHz，Job19935 约 1.08 GHz，早先 Job19918 约 1.91 GHz，因此只采用同作业配对比例，不跨作业比较绝对毫秒或按频率校正。还没有 TP8 NCCL、完整 K3 checkpoint、线上 trace 或 SLO sweep。

## 19. 如果未来重写完整 K2，什么结果会让你们重新开启 `tcgen05` 路线？

下一版不能再是孤立 Phase-6 指令替换，而应让 Phase 1/3/4/6 共享转置后的操作数布局和 TMEM 生命周期，避免每个小 phase 重复 alloc/搬运/读回。重新开启需要同时过三道 gate：

1. 正确性 gate：V128 和代表 V16/V32 形状通过 V128/reference 对拍；
2. 性能 gate：完整 K2 在 fixed、packed 单序列和多序列反例上取得净延迟收益，而不是只有 microbench core 吞吐；
3. 机制 gate：NCU 中的 grid、TMEM/TMA、scheduler、occupancy 和整卡 elapsed 指标能够解释收益来源。

在最乐观 Phase-6 V128 已慢 8.7%，而现有 `mma.sync` 调度主线又进一步变快的证据下，没有必要优先支付完整 TCGEN 集成和长期维护成本。当前优先级是并发取舍、真实 K3/TP8 和交付验收；只有跨 phase TMEM 方案击败这个更强基线，才重新开启该路线。

## 20. 如果你们是作者，sm100a/sm103a 专版到底怎么发布？

当前只能交付隔离的 guarded hybrid 候选，而不是替换通用实现：

- V128 `mma.sync` 保持默认兼容 fallback；
- 在 B300/SM103、BF16 state、低并发长 prefill 的已标定形状启用 ValueSlice；
- packed nseq=1 进入 fixed B1 策略，未建模的 nseq>1 varlen 回退；
- Phase-6 需要构建时 `FLASH_KDA_CUDA_ARCHS=103a` 和 `FLASH_KDA_ENABLE_V16_PREFETCH4=1`；Phase-1 还需要 `FLASH_KDA_ENABLE_V16_PHASE1_PREFETCH=1`；默认均关闭；
- 当前无运行时并发检测，Phase-1 只适合明确接受 latency/throughput 取舍的调用方；
- 运行时 `FLASH_KDA_K2_VALUE_SLICE=128` 可回到 V128，但不能恢复旧 V16；恢复 Phase6-only 或原 V16 需要单独二进制或重新构建；
- CUDA 13/`sm_103a` 单独构建与 CI，持续覆盖 bitwise、reference、CUDA Graph 和反例；
- 在真实并发、K3/TP8、多 GPU guard、alias、安装和回滚验收前，不安装覆盖或发布为默认路径。

还要说明 `103a` 是编译目标检查，不是对所有兼容设备的运行时 B300 产品识别；raw/强制 V16 会绕过 Python 硬件策略，`explain_k2_dispatch()`目前也只解释 ValueSlice，不报告实际 prefetch 子变体。

## 21. 为什么官方 FlashKDA 已经比 FLA 快 1.79–3.42×，还值得做 ValueSlice？

“相对另一个实现快”不等于“已经贴近目标硬件上限”。官方 H96 benchmark 证明 baseline 强；H12 targeted NCU 则揭示 TP8 单请求部署形状出现新的 12 CTA/148 SM underfill。两者研究对象不同，没有矛盾。

ValueSlice 也没有推翻官方设计：它保留相同 `mma.sync`、相同总 Tensor FLOP 和 bitwise 结果，只给低并发长 prefill 增加正交并行度。Phase-6/Phase-1 又在同一数值合约内改善 CTA 内调度。最终方案因此是对官方路径的分层、guarded 补充，而不是宣称官方 kernel 整体设计错误或更新指令天然更快。

## 22. 通信角度看，这个优化可能被 NCCL 完全隐藏吗？

可能，所以目前不能声称 TP8 端到端收益。ValueSlice 本身没有跨 GPU 通信，也不改变 collective 的消息体；局部 K2 节省约 0.21 ms 是否暴露在关键路径上，取决于 KDA 与投影、collective 的执行顺序和重叠。

需要在真实 TP8 上同时保留 compute-only 和 NCCL-overlapped timeline，比较 K2 降时前后的关键路径、通信气泡和 TTFT。若 K2 完全被 collective 覆盖，端到端收益会小；若 K2 位于未重叠的串行区，收益才会显现。单 B300 只能把这个未知量明确列出，不能替代多卡实验。

## 23. B300 的 FP8/FP6/FP4 为什么没有用于 KDA recurrence？

新低精度更适合先放在大而规则的输入/输出投影 GEMM，那里更容易利用 Tensor Core 峰值，也容易通过 scale 设计控制误差。KDA recurrence 包含长时状态传播、指数 gate 和 16×16 inverse，误差可能随序列累积；本轮连 BF16 state 的结论都只到 kernel 数值层，尚无模型级精度证据。

因此低精度 recurrence 不属于“免费迁移”。若要挑战，应分别设计 state、gate、inverse 的 scaling，做长上下文误差增长、perplexity 和任务级验证，再谈性能；不能仅依据 B300 支持 FP4/FP6 就宣布可用。

## 24. ValueSlice、Phase-6 P4 和 Phase-1 lookahead 分别解决什么？

它们对应两个层级的瓶颈。ValueSlice 沿相互独立的 Value 行切分，把 H12 的 grid 从 12 CTA 扩成 96 CTA，解决跨 CTA 的整卡 underfill。切成 V16 后，每个 CTA 内仍要沿 chunk recurrence 串行推进；Phase-6 P4 用四槽环预取八个独立 keyblock 的 state，Phase-1 则为 k、q、state 建 fragment ring，改善 CTA 内 load/use 调度。

三层都保留 D128/C16、`mma.sync`、FP32 更新和原 BF16 舍入点。P4 和 Phase-1 不是减少数学工作量或更换注意力算法，而是在 ValueSlice 后的更强基线上改善发射连续性。

## 25. 为什么 Phase-1 无初态用 lookahead4，有初态只用 lookahead2？

这是消融选择，不是从“零状态计算更少”推导出来的。None 与显式 BF16 零初态在同一 q/k/v/g/beta 下数学等价，但 state-present 模板仍更快，说明差异涉及不同 `HasStateIn` 模板实例的寄存器生命周期和生成代码。实测 ring4 对无初态稳定获益，却会让有初态路径退化，所以候选采用 `HasStateIn=false→4`、`true→2`。

当前证据支持“按状态合约选择小 lookahead”，不支持“已经完全定位所有 PC 级因果”。无初态实例仍有 8 B stack 和真实 LDL/STL，也不能宣称收益来自消除 spill。

## 26. 27%、19.40%、9.32%、37.23% 和 45.52% 应怎样理解？

27% 是早期 ValueSlice 对 V128 的第一层结果。19.40% 是 Job19901 中有初态 Phase-6 P4 相对旧 V16/P1 的新增降时；同一实验的无初态新增收益是 9.02%。Job19934 的 Phase-1 相对旧 P4，新增长度 T8192 的无初态/有初态收益是 9.32%/6.55%；37.23%/45.52% 则已经是新候选相对同作业 V128 的累计收益。

这些百分比不能相加。Job19934 的绝对时间高于早期作业与 GPU 频率差异一致，但只采样了作业开头时钟，不能跨作业拼毫秒、按频率校正或挑选各作业最好值。答辩只报同一 Job、同输入、同计时口径的配对比例。

## 27. 既然 34 个准入域点都变快，为什么最新候选仍默认关闭？

“34 个点都变快”指四种单调用计时口径的三轮中位数均为正；eager 新增收益为 5.15%–12.40%。但双流无初态 joined pair 三轮一致回归，median 增加 1.52%。这已经足以否决“所有生产负载无回归”的说法。

当前 guard 看的是一次调用的 N/H/T、dtype、slice 和编译目标，不知道 GPU 上同时有多少请求。`N=1` 不是低并发检测。因此最新 Phase-1 只能是调用方明确选择的 latency opt-in；吞吐导向部署应保留旧 P4 或 V128，并用真实并发 1/2/4/8 重新标定。

## 28. 为什么无 initial state 的绝对延迟反而可能高于有 initial state？

不能解释成无初态做了更多数学工作。二者编译为不同 `HasStateIn` 模板实例，寄存器分配和生成代码不同；同输入的 None 与显式零初态对照也保留了速度差异。现有 SASS/NCU 指向 load/use 调度与额外控制开销，但尚未完成全程固定时钟、PC 级 stall 的受控因果分析。

因此可说的是“状态接口选择影响代码生成和最佳 lookahead”，不能说“零状态数值导致变慢”或“已经找到唯一根因”。

## 29. 最新干净候选具体通过了什么，还缺什么？

Job19934 使用独立源和 build 目录、原 `setup.py` 构建的新 `.so`。它通过 120 条主跨路径比较、14 条尾块/状态补比较、从 None 开始的三步状态链、80 条跨路径 Graph 检查、80 次计时后跨路径检查和 4 次并发正确性比较；存在的 output/final state 均 finite 且 bitwise。memcheck 与 synccheck 各 20 次定向对照均退出 0、零错误。

这些检查不是全模型精度或全面 racecheck。multi-GPU guard 与单独 alias 扩展仍为 SKIP，候选也没有实际安装覆盖、完整 K3 checkpoint、TP8/NCCL 或 serving SLO 验证。因此应称“干净、可复现、单卡通过的 opt-in candidate”，不能称“生产发布完成”。

## 一页速记

| 若只来得及说一句 | 回答 |
|---|---|
| 总结论 | 不做全面 `tcgen05`；ValueSlice GO；P4/Phase-1 为默认关闭的分层 opt-in |
| `tcgen05` | V128 Phase-6 乐观摊销仍 0.920×，慢 8.7% |
| 瓶颈 | 12 CTA 对 148 SM，SM/DRAM 2.64%/1.24%，首要是 underfill |
| ValueSlice | grid 12→96；fixed/packed-one 约 −27.0%/−26.9% |
| Phase-6 P4 | 相对旧 V16：有初态 −19.40%，无初态 −9.02% |
| Phase-1 | 相对 P4：首段 −9.32%，续段 −6.55%；相对同 Job V128 累计 −37.23%/−45.52% |
| 反例 | 32×256 的 V16 慢 106%；无初态双流 joined pair 回归 1.52% |
| 正确性 | 旧证据 200/200 finite、98/98 ValueSlice bitwise；最新候选另有 120+14 跨路径 bitwise |
| BF16 state | 最坏观测 reference RMSE <1%，但不是模型级证明，也非全 FP32 内部对照 |
| Kimi 系统 | 只改善 prefill operator；按实际合约使用 `p·r`；无 TPOT/goodput/NCCL 实测 |
| 发布 | 未安装/未发布；build-time opt-in；无实时并发感知；V128 runtime fallback |
| 下一步 | 先做并发矩阵、K3/TP8 和交付验收，再评估 cluster/跨 phase TCGEN |
