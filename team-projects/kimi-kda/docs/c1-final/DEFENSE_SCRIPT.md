# C1 答辩讲稿（10 分钟）

对应演示文稿：`FlashKDA_SM100_academic_defense_20260909_v2.pptx`

叙事顺序固定为：**复现与测量 → 分析（先给结论和证据）→ 挑战。** Runtime Profile Agent 从第一阶段开始组织 measurement receipt，不是挑战阶段的事后包装。

## 时间总表

| 页 | 主题 | 时间 | 累计 |
|---:|---|---:|---:|
| 1 | 问题与范围 | 35 s | 0:35 |
| 2 | 评价路径 | 40 s | 1:15 |
| 3 | 官方复现 | 55 s | 2:10 |
| 4 | SASS/NCU measurement receipt | 65 s | 3:15 |
| 5 | Agent 主线结论与证据 | 70 s | 4:25 |
| 6 | Agent 选中的 ValueSlice 挑战 | 55 s | 5:20 |
| 7 | 三层挑战结果 | 75 s | 6:35 |
| 8 | 验收与并发反例 | 65 s | 7:40 |
| 9 | Agent 自优化闭环 | 75 s | 8:55 |
| 10 | 结论与边界 | 65 s | 10:00 |

## 第 1 页：问题与范围

“我们只回答 C1：FlashKDA 的 Tensor Core atom 仍是 SM80 世代 `mma.sync`，在 B300/SM103 上迁移到 SM100 是否值得。我们限定在单张 B300、Kimi K3 代表的 H12 prefill recurrence，不把 operator 结果写成完整模型或 serving 收益。”

转场：“判断不从新指令的峰值出发，而从可复现证据出发。”

## 第 2 页：评价路径

“我们严格按三个阶段。第一，Agent 将官方复现、SASS、grid、NCU 和隔离 probe 组成同 workload receipt。第二，Agent 必须在挑战前回答‘值不值得’，并把 MMA 候选排为 KEEP、MEASURE 或 STOP。第三，只实现前面证据保留的路线，再把正负结果回写下一轮。”

“新 opcode 本身不是收益证据；完整实现成本、正确性、统计稳定性和 fallback 共同决定迁移价值。”

## 第 3 页：官方复现

“在固定 commit `1ce47ea`、warmup 30、iters 200、repeats 5 下，官方 FlashKDA 在 H96/H64 共六个 case 上相对 FLA chunk KDA 为 1.79 到 3.42 倍。这证明官方已是强基线，任何 SM100 重写都必须证明真实净收益，不能只引用理论峰值。”

指图：“蓝色是 FlashKDA，灰色是 FLA，纵轴是延迟，越低越好。”

## 第 4 页：SASS/NCU measurement receipt

“SASS 中 recurrence 有 3,640 条 `HMMA.16816.F32.BF16`，`TCGEN/UTCMMA=0`，因此题面所说的旧 MMA 路径成立。但 TP8 时每卡只有 `96/8=12` 个 head，K2 recurrence 也只有 12 CTA，对应 148 个 SM。”

“Job17965 中 SM throughput 为 2.64%，DRAM throughput 为 1.24%，tensor elapsed 为 2.48%。这不支持‘Tensor Core 算力不够’或‘HBM 带宽饱和’。第一层物理边界是 grid underfill，其后才是 chunk recurrence 和 CTA 内 TMA/MMA issue latency。”

## 第 5 页：Agent 主线结论与证据

“在挑战之前，Agent 的 assessment 已生成结论。第一，全面 direct swap 暂停：最有利的 Phase-6 V128 中，`tcgen05` 的 L0 也只有 `mma.sync` 的 0.920 倍，L1 是 0.256 倍。第二，机械放大 CHUNK 暂停：C32/C64 先碰到数值范围与 Neumann 代价。第三，保留当前 `m16n8k16` 路径，优先搜索 CTA 并行分解和 CTA 内 issue overlap。”

“主线回答是：不值得把 FlashKDA 整体机械迁移到 `tcgen05`；值得围绕 B300 workload 重构并行度与流水，并保留 V128 fallback。”

转场：“下面的挑战不用来事后构造这个结论，而用来验证它选中的路线。”

## 第 6 页：Agent 选中的 ValueSlice 挑战

“chunk 之间的 state 依赖不能凭空并行，但 state 的 128 个 Value 行相互独立。ValueSlice V16 让每个 CTA 只处理 16 行，把 H12 的 grid 从 12 CTA 扩到 96 CTA。它不需要 reduction 或 atomic，也不改变每个输出元素的归约顺序。”

“成本是重复请求 slice-independent 公共输入，因此它必须有 workload/device guard。”

## 第 7 页：三层挑战结果

“第一层 ValueSlice 解决 CTA 数，同作业 V16 相对 V128 降时约 27.0%。第二层 Phase-6 `StatePrefetch=4` 拉长 state load-use 距离，对旧 V16/P1 的新增收益是无初态 9.02%、有初态 19.40%。第三层 Phase-1 按状态合约选 L4/L2，Job19934 中最终候选相对同作业 V128 在首段降低 37.23%，续段降低 45.52%。”

“这些数使用不同基线，不能相加，不能跨 job 拼接绝对毫秒。”

## 第 8 页：验收与并发反例

“挑战路径的 34 个准入域配置均获益，120+14 条跨路径 output/final-state 比较逐位一致，memcheck 和 synccheck 均为零报告错误。但无初态双 stream joined-pair latency 回归 1.52%，三轮同向。”

“因此这是默认关闭的单请求 latency candidate，不是 production default。现有 guard 不感知设备上其他 stream/request，必须保留 V128 fallback。”

## 第 9 页：Agent 自优化闭环

“现在可以看完整 Agent：Measure 读取 SASS、NCU、grid 和 probe；Assess 先输出主问题答案和物理 finding；Propose 只生成与瓶颈一致的 typed MMA candidate；Verify 检查语义、正确性和配对统计；Update 才更新 incumbent 和 conclusions-only memory。”

“对当前数据，Agent 学到的不是‘永远不用 `tcgen05`’。V16 core-only L0 曾达 1.501 倍，但加入整合成本后 L1 降到 0.778 倍；所以下次若重开，搜索目标应是跨 phase TMEM residency，而不是重复 direct swap。”

“当前 44 项 CPU 测试覆盖原始证据取数、主线 assessment 与闭环逻辑。W384/W768 前瞻试验 4/4 通过，但容量策略仍仅是 shadow recommendation。”

## 第 10 页：结论与边界

“第一，不值得全面把 `mma.sync` 机械替换为 `tcgen05`；B300 专用化应先解决并行度和流水。第二，ValueSlice 和分阶段预取在单请求域获得收益，但并发反例阻止默认启用。第三，Runtime Profile Agent 已经从复现与测量开始，用物理证据回答主问题、选择 MMA 候选，并用正负结果优化下一轮。”

“Kernel is cheap 的含义是候选生成不再稀缺；真正昂贵的是把 profile 变成可信、可回退的 policy 决策。”

最后主动说明：“结论限定于单张 B300 与已测 FlashKDA forward；完整 Kimi K3、TP8/NCCL、真实并发和 SLO 尚未验证。”

## 现场节奏

- 第 5 页必须完整说出主线结论，不能跳过后再讲。
- 超时时可缩短第 7 页的分层数字，但保留“基线不同，不能相加”。
- 第 8 页的 `+1.52%` 并发反例和默认关闭边界不可省略。
- 第 9 页先说 Agent 在前两阶段做了什么，再说 conclusions-only memory 和自优化。
