# C1 10 分钟答辩逐页讲稿

对应演示文稿：[`FlashKDA_SM100_decision_defense.pptx`](FlashKDA_SM100_decision_defense.pptx)

总时间严格按 **10 页 / 600 秒** 设计。正常语速约每分钟 220–260 个汉字；现场应优先说每页的“必须说”，若被打断则跳过“可省略”。所有性能数字只在各自一致的计时口径内比较。

## 时间总表

| 页码 | 主题 | 时间 | 累计 |
|---:|---|---:|---:|
| 1 | 问题与最终判断 | 45 s | 0:45 |
| 2 | 官方 benchmark 复现 | 55 s | 1:40 |
| 3 | SASS 与 NCU：真正瓶颈 | 70 s | 2:50 |
| 4 | CHUNK=16，为什么不能机械放大 | 55 s | 3:45 |
| 5 | `tcgen05+TMEM` 直接替换实验 | 70 s | 4:55 |
| 6 | 挑战设计：ValueSlice | 65 s | 6:00 |
| 7 | 性能、反例与最终 dispatcher | 70 s | 7:10 |
| 8 | 正确性与 BF16 state 边界 | 55 s | 8:05 |
| 9 | 从 operator 到 Kimi K3/通信/SLO | 65 s | 9:10 |
| 10 | go/no-go 与最终回答 | 50 s | 10:00 |

---

## 第 1 页：问题与最终判断（45 秒）

### 必须说

“我们只回答 C1 一个问题：FlashKDA 官方 kernel 的 Tensor Core atom 仍是 SM80 `mma.sync`，迁到 SM100 值不值得？我们的答案分两层：**不值得机械地全面换成 `tcgen05`；值得为 B300/SM103 做受保护的专用路径，但优先级是 recurrence 并行度，而不是指令代际。**

我们按题目规定走三阶段：先复现和测量，再逐项分析，最后动手挑战。挑战允许负结果，所以我们的目标不是证明新硬件一定快，而是给出可发布的 stop/go 决策。”

### 可省略

“SM80”在这里仅指 MMA atom，不代表整个 kernel 都停留在 SM80；官方实现已经使用 TMA 等较新的搬运能力。

### 转场

“先确认基线够不够可信，以及题目说的 SM80 MMA 是否真的出现在 B300 二进制里。”

---

## 第 2 页：官方 benchmark 复现（55 秒）

### 必须说

“我们固定 FlashKDA commit `1ce47ea`、CUTLASS `5c149f5`，在 B300、CUDA 13、PyTorch 2.10 上按官方口径 warmup 30、iters 200、repeats 5 复现。

H96 是 K3 官方对照形状：fixed、ragged6、8×1024 的 FlashKDA 分别是 1.0304、0.8612、0.6963 毫秒，相对 FLA chunk 是 2.34、2.82、3.42 倍。H64 三组是 1.79 到 3.37 倍。

这页的重点不是说官方慢，而是说明它已经是强基线；SM100 重写必须证明真实净收益，不能只引用峰值算力。我们还用独立官方 worktree 对 patched V128 做 parity，10 个 output/final-state tensor 全部 bitwise equal，排除了 baseline 被补丁污染。”

### 指图

先指 H96 三组，再扫一眼 H64；不要逐个念完整表格。

### 转场

“强基线之后，关键问题变成：它在 B300 上到底受什么限制？”

---

## 第 3 页：SASS 与 NCU——真正瓶颈（70 秒）

### 必须说

“SASS 先确认题目事实：K2 有 3,640 条静态 `HMMA.16816.F32.BF16`，`TCGEN/UTCMMA` 为零，所以矩阵乘确实仍走 SM80 世代 atom。

但 profiler 给出的瓶颈不是 Tensor Core 峰值。在 K3 TP8 代表形状 `T8192,H12,D128` 上，每卡 96 除以 8，只剩 12 个 head。官方 recurrence grid 就是 12 CTA，对 B300 的 148 个 SM，单波最多只覆盖 8.1%。Job 17965 中官方 V128 的 NCU duration 是 1.27 毫秒，SM throughput 2.64%，DRAM throughput 1.24%，tensor pipe 的 elapsed-cycle 口径只有 2.48%。

ValueSlice 把 grid 扩成 96 CTA 后，NCU duration 降到 901.22 微秒，SM/DRAM 升到 7.22%/1.83%。所以它既不是传统 compute-bound，也不是 HBM bandwidth-bound；首要边界是 **grid underfill、chunk recurrence critical path 和 CTA 内 TMA/issue latency**。”

### 口径提醒

“同一 CSV 里的 tensor active-cycle 数字是 30.98% 和 5.43%，分母只含 SM 活跃周期，不能当整卡利用率。这里主图只使用 elapsed-cycle 2.48%/3.50%。”

### 转场

“既然不是算力峰值不够，下面先检验两个最直观的 SM100 迁移想法：放大 CHUNK 和更换 MMA。”

---

## 第 4 页：CHUNK=16，为什么不能机械放大（55 秒）

### 必须说

“CHUNK=16 同时绑定三件事：BF16/FP32 指数范围、16×16 Neumann 求逆代价和 `m16n8k16` 的自然形状。我们把 32/64 分别量化。

在当前 `lower_bound=-5`、没有 rescale 的指数恢复路径里，C32 和 C64 都在第 18 个 token 首次出现 FTZ/overflow；每通道分别产生 15 组和 47 组 zero/inf。若把当前密集 Neumann 级数朴素扩展，总序列代价不是下降，而是 C16 的 5.33 倍和 26.67 倍。workspace 每 head 只从 6.750 MiB 升到 7.125 和 8.063 MiB，所以最先破的是数值，其次是求逆计算，不是显存。

FLA 的 safe/block 小探针证明加 rescale 可以把 C32/C64 做对，但这已经是算法重设，不能把它描述成改一个常量就会加速。”

### 转场

“那么保持 CHUNK16，只把最适配的一段换成 `tcgen05` 呢？”

---

## 第 5 页：`tcgen05+TMEM` 直接替换实验（70 秒）

### 必须说

“SM100 BF16、CTA-group 1 的 `tcgen05` 支持 M 为 64 或 128、N 从 8 到 256 且步长 8、K 等于 16。K2 Phase-6 是 `[128,16]@[16,V]`，所以对 V16、32、64、128 都可以自然映射为 `m128nVk16`。也就是说，不能用‘tile 不匹配’草率否决它。

我们因此专门选择对新指令最有利的 Phase-6 做真实 B300 probe。L0 让两边都先拿到偏好的片上布局，但 `tcgen05` 仍支付 TMEM alloc、descriptor、commit/wait、读回和 dealloc；L1 再加入 state/gate 和保守 U 重排。

K3 正式 V128、grid12、inner64 时，L0 的 `mma_time/tcgen_time` 只有 0.920，也就是 `tcgen05` 慢 8.7%；L1 更只有 0.256。V16 的 L0 可以到 1.501 倍，说明 core 并非完全没有潜力，但 L1 降到 0.778，转换成本会吃掉收益。

所以我们的 stop decision 是：**不把‘保持现有 K2 数据流、只换 Phase-6 指令’集成进正式 kernel。** 这个 probe 不否决未来跨多个 phase 保持 TMEM-resident 的重写。”

### 转场

“直接换指令没有正信号，我们把挑战转向 NCU 已经指出的 12 CTA underfill。”

---

## 第 6 页：挑战设计——ValueSlice（65 秒）

### 必须说

“K2 的 chunk 之间有 state 依赖，不能凭空并行；但 state 的 128 个 Value 行彼此独立。ValueSlice 沿 Value 行切成 V128、64、32、16，每个 CTA 只更新自己的 `V×D` state slice，不需要 reduction、atomic，也不改变单个输出元素的归约顺序。

grid 从 `(N,H)` 变成 `(N,H,D/V)`。H12 下，V128、V64、V32、V16 分别是 12、24、48、96 CTA。总 Tensor FLOP 不变，收益来自更高整卡并行度；代价是 q、k、gate 等 slice-independent 输入会被多个 CTA 重复请求。

这仍然是 SM100 迁移挑战，因为题目明确允许‘并行度重构’，而我们的改动由 B300/SM103 的 148 SM underfill 触发，并以该架构的资源和 workload 做 dispatcher gate。迁移的目标是用好目标架构，不是强制出现某条新指令。”

### 指图

从 12 CTA 指到 96 CTA，再指出“FLOP 不变 / common input 重复”这组交换关系。

### 转场

“关键是这种切分不能只展示最佳点，必须同时展示反例，并让 dispatcher 安全地捕获它。”

---

## 第 7 页：性能、反例与最终 dispatcher（70 秒）

### 必须说

“所有 case 的 total tokens 都是 8192、H12、D128。fixed 单序列 V128 是 0.7807 毫秒，V16 是约 0.569 毫秒，auto 为 0.5698，降低 27.0%。packed 单序列更贴近 serving 调用：V128 0.7850，最终 auto 0.5740，降低 26.9%。

但同样 8192 tokens，ragged6 最优是 V64，降低 17.4%；8×1024 的最佳收益只有 1.3%，低于 3% guard；32×256 时 V16 反而慢 106%。这说明总 token 数相同也不能用同一个 slice，高并发短序列本身已经提供 CTA。

补丁 `0002` 修复了 packed 单序列漏选：只读 `cu_seqlens.numel()` 元数据，nseq=1 复用 fixed B1 标定，不读取 GPU 上的长度值，也不触发 host sync；nseq 大于 1 继续保守回退 V128。最终形式不是无条件 V16，而是 **guarded ValueSlice + V128 fallback**。”

### 转场

“然后我们验证，性能收益没有靠放宽误差换来。”

---

## 第 8 页：正确性与 BF16 state 边界（55 秒）

### 必须说

“正确性分三层。第一，官方扩展和 patched V128 的 10 个 tensor 全部 bitwise equal。第二，V16/V32/V64 对 V128 的 98 条 comparison row 全部 bitwise equal，最坏 relative RMSE 为零。第三，对题目指定的 FLA `naive.py` 和 `chunk.py` 做独立参考对拍，完整 CSV 的 200 条 comparison row 全部 finite；所有独立参考关系的观测最坏 relative RMSE 是 0.9131%。

这里我们刻意不说‘200/200 统一阈值通过’，因为长序列和 K3 行没有统一预注册 hard threshold。我们只报告 finite 和观测误差。

BF16 state 方面，T8192 long-memory 对 naive 的 output/state 是 0.8240%/0.7405%。但 public buffer 切成 FP32 后内部仍按 BF16 舍入，所以这不是完整 FP32 recurrence 对照，也不能替代模型级 perplexity 或任务精度。”

### 转场

“最后把 27% 算子收益放回 Kimi 系统，说明哪些能说、哪些不能说。”

---

## 第 9 页：从 operator 到 Kimi K3、通信和 SLO（65 秒）

### 必须说

“FlashKDA forward 是 KDA prefill 的直接组成部分，所以最可能受益的是 TP8、低并发、长 prompt 的 TTFT 组成部分。但 **27% 只是算子降时，不是 TTFT 降 27%**。

设 FlashKDA forward 占完整 prefill 的比例为 `p`，一阶 prefill 降时约为 `0.27p`。当 p 是 20%、40%、60% 时，只能推断 5.4%、10.8%、16.2% 的 prefill 降时；SLO goodput 还取决于 TTFT/TPOT 哪个约束绑定、continuous batching、排队和资源重叠。69/93 是层数比例，不是时间比例。

本挑战不声称 decode/TPOT 加速：T=1 会回退 V128，纯 decode 还有独立 fused KDA decode。也不声称 TP8 通信改善：单 B300 没有 NCCL 实测，局部 K2 不改变 collective 消息体。

卡内下一步则很明确：ValueSlice 会重复搬公共输入，CTA Cluster、DSM 和 TMA multicast 可以让 slice CTA 共享一次搬运。纸面 source-request 模型给出最多 81.3% common-input 请求可消除，但必须再测 cluster residency、同步和真实 duration。”

### 转场

“因此最终不是简单的‘迁’或‘不迁’，而是一张按证据分层的发布决策表。”

---

## 第 10 页：go/no-go 与最终回答（50 秒）

### 必须说

“最终四项决策：

第一，全面 `mma.sync` 到 `tcgen05` 是 NO-GO，因为正式 V128 的乐观 Phase-6 probe 仍慢 8.7%。

第二，CHUNK32/64 机械放大是 NO-GO，因为当前数值路径在 token18 失效，朴素 Neumann 代价增长 5.33 倍和 26.67 倍。

第三，guarded ValueSlice 是 GO：它针对 12 CTA/148 SM 的真实瓶颈，fixed 和 packed 单序列约降时 27%，98/98 ValueSlice 对照 bitwise equal，同时用高并发反例界定启用域。

第四，Cluster + TMA multicast 是 NEXT，用来减少 ValueSlice 的重复搬运。

所以对题目的直接回答是：**SM100 值得利用，但利用 SM100 不等于把 SM80 MMA 全换掉。当前值得交付的是受保护的并行度专版，不值得交付的是全面指令替换。**”

### 结束句

“我们的结果既给出了正向加速，也用负实验说明了哪些路线不值得继续，这就是本题要求的迁移决策。”

---

## 现场节奏与应急删减

- 8:05 前必须讲完第 8 页；否则第 9 页只说 `0.27p`、无 decode/NCCL 实测和 cluster 下一步。
- 第 2 页不逐项念 H64；第 4 页不展开 Neumann 公式；第 5 页不解释所有 V/grid 组合。
- 若老师提前追问，在当前页用一句话回答后说“这个边界在第 8/9/10 页会完整回答”，不要打乱主线。
- 最后必须保留 20 秒完整说出“NO-GO / NO-GO / GO / NEXT”和一句话结论。
- 所有数字若一时记不清，优先报方向、形状和证据 job，不现场猜小数。
