# C1 10 分钟答辩逐页讲稿

对应演示文稿：[`FlashKDA_SM100_decision_defense_20260906.pptx`](FlashKDA_SM100_decision_defense_20260906.pptx)

总时间严格按 **10 页 / 600 秒** 设计。正常语速约每分钟 220–260 个汉字；现场应优先说每页的“必须说”，若被打断则跳过“可省略”。所有性能数字只在各自一致的计时口径内比较。

## 时间总表

| 页码 | 主题 | 时间 | 累计 |
|---:|---|---:|---:|
| 1 | 问题与分级判断 | 45 s | 0:45 |
| 2 | 官方 benchmark 复现 | 55 s | 1:40 |
| 3 | SASS 与 NCU：真正瓶颈 | 70 s | 2:50 |
| 4 | CHUNK=16，为什么不能机械放大 | 55 s | 3:45 |
| 5 | `tcgen05+TMEM` 直接替换实验 | 70 s | 4:55 |
| 6 | 第一层优化：ValueSlice | 65 s | 6:00 |
| 7 | 三层优化阶梯 | 70 s | 7:10 |
| 8 | 单调用收益与并发负例 | 55 s | 8:05 |
| 9 | 验证闭环与系统边界 | 65 s | 9:10 |
| 10 | STOP / KEEP / NEXT | 50 s | 10:00 |

---

## 第 1 页：问题与分级判断（45 秒）

### 必须说

“我们只回答 C1 一个问题：FlashKDA 官方 kernel 的 Tensor Core atom 仍是 SM80 `mma.sync`，迁到 SM100 值不值得？我们的答案是：**不值得机械地全面换成 `tcgen05`；值得为 B300/SM103 做受保护的专用路径，但优先级是 recurrence 并行度和流水调度，而不是指令代际。**

最终形成了三层优化：ValueSlice 增加跨 CTA 并行度，Phase-6 Prefetch4 隐藏 state 读取延迟，Phase-1 再按有无初态选择 lookahead。第一层结论最稳；后两层是默认关闭、尚未发布的低并发延迟候选，因为最新双流实验已经暴露吞吐取舍。”

### 可省略

“SM80”在这里仅指 MMA atom，不代表整个 kernel 都停留在 SM80；官方实现已经使用 TMA 等较新的搬运能力。所有实测结论限定在单张 B300/SM103，不外推为所有 SM100-family 产品结论。

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

ValueSlice 把 grid 扩成 96 CTA 后，NCU duration 降到 901.22 微秒，SM/DRAM 升到 7.22%/1.83%。所以它既不是传统 compute-bound，也不是 HBM bandwidth-bound；第一层边界是 **grid underfill**，扩展 grid 后仍要处理 chunk recurrence critical path 和 CTA 内 TMA/issue latency。这也解释了为什么后续优化继续落在流水调度，而不是换 MMA。”

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

## 第 6 页：第一层优化——ValueSlice（65 秒）

### 必须说

“K2 的 chunk 之间有 state 依赖，不能凭空并行；但 state 的 128 个 Value 行彼此独立。ValueSlice 沿 Value 行切成 V128、64、32、16，每个 CTA 只更新自己的 `V×D` state slice，不需要 reduction、atomic，也不改变单个输出元素的归约顺序。

grid 从 `(N,H)` 变成 `(N,H,D/V)`。H12 下，V128、V64、V32、V16 分别是 12、24、48、96 CTA。总 Tensor FLOP 不变，收益来自更高整卡并行度；代价是 q、k、gate 等 slice-independent 输入会被多个 CTA 重复请求。

这仍然是 SM100 迁移挑战，因为题目明确允许‘并行度重构’，而我们的改动由 B300/SM103 的 148 SM underfill 触发，并以该架构的资源和 workload 做 shape/device guard。迁移的目标是用好目标架构，不是强制出现某条新指令。ValueSlice 解决的是跨 CTA 的第一层 underfill，但每个 V16 CTA 内部仍有串行 state 流水，这成为后两层优化的入口。”

### 指图

从 12 CTA 指到 96 CTA，再指出“FLOP 不变 / common input 重复”这组交换关系。

### 转场

“下面把 ValueSlice、Phase-6 和 Phase-1 放在同一条性能阶梯上，先说明三层各自解决什么。”

---

## 第 7 页：三层优化阶梯（70 秒）

### 必须说

“这一页把主线压成三层。官方 V128/P1 在 TP8 的 H12 形状只有 12 CTA。第一层 ValueSlice 把它扩成 96 CTA，Job17947 的 fixed 单序列降时 27.0%，解决整卡 grid underfill。

第二层不再增加 CTA，而把 Phase-6 的 state 预取窗口从 1 拉到 4。Job19901 相对旧 V16，有初态新增降时 19.40%，无初态新增 9.02%。

第三层为 Phase-1 的 k、q、state 建 fragment ring，无初态用 lookahead4，有初态用 lookahead2。Job19934 中，完整候选相对同作业 V128，首段降低 37.23%，续段降低 45.52%。

三层作用于不同边界：ValueSlice 解决跨 CTA underfill，后两层拉长 V16 CTA 内的 load-use 距离。最后一层的 37.23% 和 45.52% 已是累计结果，不能再与 27%、19.40% 或 9.02% 相加；不同 Job 的绝对毫秒也不横向拼接。”

### 转场

“单调用的阶梯成立后，下一页看它在准入域和两个 stream 下是否仍然成立。”

---

## 第 8 页：单调用收益与并发负例（55 秒）

### 必须说

“Job19934 只用同输入、同作业配对。34 个准入域单调用点在四种计时口径的中位数上都获益，Phase-1 相对已有 P4 的 eager 增量为 5.15% 到 12.40%。T8192 上，无初态从 0.950848 降到 0.862208 毫秒，降低 9.32%；有初态从 0.796144 降到 0.743968 毫秒，降低 6.55%。

但两个 stream 给出必须公开的反例：无初态 joined pair 从 1.147440 增到 1.164832 毫秒，回归 1.52%，三轮一致；有初态只回归约 0.03%，基本持平。joined pair 不能除以二冒充单请求 latency。

当前 guard 只看 `N1/H12/D128/C16/BF16/T2K–8K`，不感知 GPU 的实时并发。因此 0004 和 0005 均默认关闭；运行时可以强制 V128，但恢复 Phase6-only V16 需要单独二进制。”

### 转场

“并发负例决定了启用边界，下一页再看数值验证是否闭环，以及 operator 收益能外推到哪里。”

---

## 第 9 页：验证闭环与系统边界（65 秒）

### 必须说

“Kernel 侧的验证闭环成立。原参考矩阵 200 条 comparison row 全部 finite，其中 98 条 ValueSlice 对 V128 逐位一致，独立参考关系的观测最坏 relative RMSE 是 0.913%。最新干净候选又完成 120 加 14 条跨路径 bitwise 回归，state chain、Graph、计时后检查和双流正确性通过；memcheck 与 synccheck 都是零错误。

边界同样明确：BF16 state 仍只是 kernel 数值对拍，public FP32 buffer 不等于全 FP32 recurrence，也没有完整模型 accuracy 证据。

系统外推只展示首段。若 FlashKDA forward 占完整 prefill 的比例为 `p`，Job19934 的 37.23% operator 降时给出一阶敏感性 `0.3723p`；p 为 20%、40%、60% 时，对应约 7.4%、14.9%、22.3%。这不是 37% TTFT，更不是 TPOT、SLO goodput、TP8/NCCL 或多请求 throughput 实测。69/93 是层数比例，不是 wall-time 占比。”

### 转场

“单卡 kernel 证据闭环，但系统证据仍有缺口，所以最后压缩成 STOP、KEEP 和 NEXT。”

---

## 第 10 页：STOP / KEEP / NEXT（50 秒）

### 必须说

“STOP 是全面 `mma.sync` 到 `tcgen05`：V128 Phase-6 的乐观 probe 只有 0.920 倍，当前不合入；只有跨 phase、TMEM-resident 的完整数据流过 gate，才重新开启。

KEEP 是 V128 fallback 加 guarded V16/P4/Phase1。ValueSlice 保留 shape/device guard；P4 和 Phase-1 默认关闭，完整候选相对同作业 V128 降低 37.23% 和 45.52%，但随时能回退 V128。

NEXT 是并发矩阵与真实 K3/TP8 关键路径。先解决无初态双流 1.52% 回归和实时并发不可见，再评估 Cluster/multicast 与跨 Phase TCGEN。

一句话回答：**SM100 的价值在重塑并行与数据流，不在替换一条看起来更新的指令。**”

### 结束句

“我们的结果既给出了正向加速，也用负实验说明了哪些路线不值得继续，这就是本题要求的迁移决策。”

---

## 现场节奏与应急删减

- 8:05 前必须讲完第 8 页；否则第 9 页只说最新 120+14 bitwise、`0.3723p` 只是敏感性分析，以及无模型/TP8 实测。
- 第 2 页不逐项念 H64；第 4 页不展开 Neumann 公式；第 5 页不解释所有 V/grid 组合。
- 若老师提前追问，在当前页用一句话回答后说“这个边界在第 8/9/10 页会完整回答”，不要打乱主线。
- 最后必须保留 20 秒完整说出“STOP / KEEP / NEXT”和一句话结论。
- 所有数字若一时记不清，优先报方向、形状和证据 job，不现场猜小数。
