# C1：FlashKDA 迁移到 SM100 是否值得？

团队：**奶龙必胜**；成员：**蔡雨洋、李奥、赵骋**

基线冻结日期：2026-09-10；Stage 12 搜索闭包复核：2026-09-11

研究设备：NVIDIA B300 SXM6 AC（SM103a）

## 结论

在已测 B300 工作负载上，单独把 HMMA 替换为 tcgen05 没有形成稳定收益；围绕数据驻留、布局、并行度、流水和 dispatch 协同重组的 SM100 全栈路径获得明确收益。Stage 12 进一步在 P3→P4 双 MMA 探针中确认：单次 TMEM 生命周期在长驻留时可达 HMMA 的 1.43–1.45x，但短驻留仍回退。适合的交付形态是：**对已认证 profile 启用受条件保护的 SM100 后端，其余场景回退 HMMA。**

这个结论来自三步：复现与测量锁定执行身份和计时边界；HMMA/tcgen05 双分支 Agent 在公平新增机会下提出、验证并回写候选；挑战阶段通过局部机制实验和 public-full 隔离测量确定哪些路线晋级。

## 快速入口

1. [`03_reports/SM100_MAINLINE_DELIVERY.md`](03_reports/SM100_MAINLINE_DELIVERY.md)：C1 正式技术报告，完整回答题目主线与六个讨论点。
2. [`02_paper/main.pdf`](02_paper/main.pdf)：8 页论文；LaTeX 源码与一键构建入口在同目录。
3. [`01_slides/C1_FlashKDA_SM100_奶龙必胜_署名版_20260911.pdf`](01_slides/C1_FlashKDA_SM100_奶龙必胜_署名版_20260911.pdf)：15 页课程提交答辩固定版。
4. [`01_slides/C1_FlashKDA_SM100_奶龙必胜_署名版_20260911.pptx`](01_slides/C1_FlashKDA_SM100_奶龙必胜_署名版_20260911.pptx)：可编辑课程提交答辩稿，已嵌入 Agent 探索闭环、Typed IR 流水和局部知识闭包三张方法图。
5. [`DEFENSE_BRIEF.md`](DEFENSE_BRIEF.md)：答辩主线、必记数字和追问边界。
6. [`RETROSPECTIVE.md`](RETROSPECTIVE.md)：完整研究复盘与后续指导。
7. [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md)：结论到证据的索引。
8. [`REPRODUCTION.md`](REPRODUCTION.md)：B300 复验顺序与计时要求。
9. [`03_reports/STAGE12_TYPED_IR_SEARCH_CLOSURE_20260911.md`](03_reports/STAGE12_TYPED_IR_SEARCH_CLOSURE_20260911.md)：双分支搜索上界、冗余重排清理、B300 新结果与分层闭包。

## 归档结构

| 目录 | 内容 |
|---|---|
| `00_task/` | C1 目标与研究问题 |
| `01_slides/` | 课程署名版与公开脱敏版 PPTX/PDF、字体与版式审计 |
| `02_paper/` | 论文 PDF、LaTeX、参考文献和模板来源 |
| `03_reports/` | 最终报告、方法复盘、IR 审查和阶段性交接 |
| `04_evidence/` | 答辩核心数字对应的精选证据 |
| `05_reproduction/` | correctness、public-full、双分支和幻灯片复现脚本 |
| `06_agent_framework/` | Program IR、Experience IR、multi-agent 闭环、测试和运行记忆 |
| `07_experiment_history/` | 双分支实验树及候选结果，排除可重建的大张量 |
| `08_patches/` | 关键实现补丁 |
| `third_party/` | PIKE/Atrex 的锁定版本、许可证与 NOTICE |

## 框架怎样工作

MARPE（迁移感知运行时配置演化框架）让 HMMA 与 tcgen05 两个分支共享同一套正确性、执行身份、public-call 计时和新增机会记账。Program IR 记录“改了什么”，Experience IR 记录“在哪个 profile 下、凭什么成立、何时可以重新打开”。Stage 12 又加入跨路线 `KernelDesign`、带 shape/dtype/layout/carrier 的 Dataflow IR、五角色 patch 组合、机制 niche archive 与有限域闭包证书。候选只有携带完整证明材料，才能从想法进入 screen、qualification，再晋级为 guarded route。

框架复用 PIKE 的并行分支搜索和候选分配，复用 Atrex Kernel Agent 的隔离 GPU 执行、profiling、Git episode 与 ABBA 验证。原创边界和未证明事项见 [`OPEN_SOURCE_AUDIT.md`](OPEN_SOURCE_AUDIT.md)。

## 证据口径

- microprobe 只证明局部机制，不能替代完整公开接口结果；
- 所有 headline 性能采用隔离进程、冷 L2、完整状态回写和 balanced/ABBA 测量；
- 负结果只停止相同候选与作用域，不升级为整条路线禁令；
- SM100 路径的收益按 full-stack treatment effect 表述，不归因于单条 tcgen05 指令；
- 当前双分支闭环没有证明 multi-agent 相对单 agent 的性能优势。

Stage 12 已关闭三个新增局部搜索域：HMMA V32/V64 的 one/two-block-per-warp 映射、tcgen05 V16 Phase-6 的 logical/producer-ready global-B 布局，以及 P3/P4 双 MMA 的 HMMA/shared-carrier lifecycle 对照。完整 P1–P6 public-call 移植、显式 D-fragment→A-TMEM 变换和算法族 dispatch 仍开放；这里的“闭包”不等于全局 GPU 最优。

`MANIFEST.sha256` 校验本目录全部冻结文件，`SOURCE_MAP.tsv` 说明各部分来源角色。
