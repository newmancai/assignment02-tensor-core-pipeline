# C1 任务定义

团队：奶龙必胜

## 目标

FlashKDA 官方 kernel 当前在 B300 上的递推矩阵乘仍使用 SM80 世代的 HMMA。项目需要通过复现、测量、分析和实际挑战，判断迁移到 SM100 原生路径是否值得。

## 三步交付

1. **复现与测量**：复现官方 FlashKDA，确认真实执行指令，建立正确性、性能和 profiler 证据。
2. **分析（结论 + 证据）**：解释 CHUNK、并行度、瓶颈、BF16 state、HMMA 与 tcgen05 的收益条件。
3. **挑战**：实际实现并测量 HMMA 与 SM100/tcgen05 候选。正结果和负结果都必须绑定测量范围。

## 本项目对问题的操作化

在同一张 NVIDIA B300 SXM6 AC、同一 KDA public contract 下，比较：

- 当前官方 HMMA；
- 经过新增探索后的 HMMA 候选；
- tcgen05/TMEM 机制候选；
- 完整 SM100 工程路径。

“迁移值得”被定义为：某个经过正确性、执行身份和 public-full 性能验证的 SM100 路径，在明确的设备与 workload guard 下，能稳定胜过对应基线，并且可以保留可靠回退路径。

