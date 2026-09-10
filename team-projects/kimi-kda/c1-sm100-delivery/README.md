# C1：FlashKDA 迁移到 SM100 是否值得？

团队：**奶龙必胜**  
冻结日期：2026-09-10  
研究设备：NVIDIA B300 SXM6 AC（SM103a）

## 结论

在已测 B300 工作负载上，单独把 HMMA 替换为 tcgen05 没有形成稳定收益；围绕数据驻留、布局、并行度、流水和 dispatch 协同重组的 SM100 全栈路径获得明确收益。适合的交付形态是：**对已认证 profile 启用受条件保护的 SM100 后端，其余场景回退 HMMA。**

这个结论来自三步：复现与测量锁定执行身份和计时边界；HMMA/tcgen05 双分支 Agent 在公平新增机会下提出、验证并回写候选；挑战阶段通过局部机制实验和 public-full 隔离测量确定哪些路线晋级。

## 快速入口

1. [`01_slides/C1_FlashKDA_SM100_奶龙必胜_公开脱敏版_20260910.pdf`](01_slides/C1_FlashKDA_SM100_奶龙必胜_公开脱敏版_20260910.pdf)：15 页答辩固定版。
2. [`01_slides/C1_FlashKDA_SM100_奶龙必胜_公开脱敏版_20260910.pptx`](01_slides/C1_FlashKDA_SM100_奶龙必胜_公开脱敏版_20260910.pptx)：可编辑答辩稿。
3. [`DEFENSE_BRIEF.md`](DEFENSE_BRIEF.md)：答辩主线、必记数字和追问边界。
4. [`02_paper/main.pdf`](02_paper/main.pdf)：论文版核心叙述；源文件在同目录。
5. [`RETROSPECTIVE.md`](RETROSPECTIVE.md)：完整研究复盘。
6. [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md)：结论到证据的索引。
7. [`REPRODUCTION.md`](REPRODUCTION.md)：B300 复验顺序与计时要求。

## 归档结构

| 目录 | 内容 |
|---|---|
| `00_task/` | C1 目标与研究问题 |
| `01_slides/` | 公开脱敏 PPTX、嵌入字体 PDF及字体审计 |
| `02_paper/` | 论文 PDF、LaTeX、参考文献和模板来源 |
| `03_reports/` | 最终报告、方法复盘、IR 审查和阶段性交接 |
| `04_evidence/` | 答辩核心数字对应的精选证据 |
| `05_reproduction/` | correctness、public-full、双分支和幻灯片复现脚本 |
| `06_agent_framework/` | Program IR、Experience IR、multi-agent 闭环、测试和运行记忆 |
| `07_experiment_history/` | 双分支实验树及候选结果，排除可重建的大张量 |
| `08_patches/` | 关键实现补丁 |
| `third_party/` | PIKE/Atrex 的锁定版本、许可证与 NOTICE |

## 框架怎样工作

MARPE（迁移感知运行时配置演化框架）让 HMMA 与 tcgen05 两个分支共享同一套正确性、执行身份、public-call 计时和新增机会记账。Program IR 记录“改了什么”，Experience IR 记录“在哪个 profile 下、凭什么成立、何时可以重新打开”。候选只有携带完整证明材料，才能从想法进入 screen、qualification，再晋级为 guarded route。

框架复用 PIKE 的并行分支搜索和候选分配，复用 Atrex Kernel Agent 的隔离 GPU 执行、profiling、Git episode 与 ABBA 验证。原创边界和未证明事项见 [`OPEN_SOURCE_AUDIT.md`](OPEN_SOURCE_AUDIT.md)。

## 证据口径

- microprobe 只证明局部机制，不能替代完整公开接口结果；
- 所有 headline 性能采用隔离进程、冷 L2、完整状态回写和 balanced/ABBA 测量；
- 负结果只停止相同候选与作用域，不升级为整条路线禁令；
- SM100 路径的收益按 full-stack treatment effect 表述，不归因于单条 tcgen05 指令；
- 当前双分支闭环没有证明 multi-agent 相对单 agent 的性能优势。

`MANIFEST.sha256` 校验本目录全部冻结文件，`SOURCE_MAP.tsv` 说明各部分来源角色。
